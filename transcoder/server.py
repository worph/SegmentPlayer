#!/usr/bin/env python3
"""
HLS Live Transcoder Server
Transcodes video segments on-demand using FFmpeg.

Design: Single-threaded transcoding with one-ahead prefetch.
- One FFmpeg process gets 100% CPU for fastest segment time
- After serving segment N, prefetch N+1 in background thread
- No parallel transcoding (would split CPU and increase latency)
"""
from __future__ import annotations

import os
import subprocess
import hashlib
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs
import json
import re

# Import Stremio addon handler
from stremio import StremioHandler

# Configuration
MEDIA_DIR = os.environ.get('MEDIA_DIR', '/data/media')
CACHE_DIR = os.environ.get('CACHE_DIR', '/data/cache')
SEGMENT_DURATION = int(os.environ.get('SEGMENT_DURATION', '4'))
PORT = int(os.environ.get('PORT', '8080'))
PREFETCH_SEGMENTS = int(os.environ.get('PREFETCH_SEGMENTS', '4'))  # How many segments to prefetch ahead

os.makedirs(CACHE_DIR, exist_ok=True)

# Resolution presets: (width, height, crf)
RESOLUTIONS = {
    'original': (None, None, 23),
    '1080p': (1920, 1080, 23),
    '720p': (1280, 720, 24),
    '480p': (854, 480, 25),
    '360p': (640, 360, 26),
}


class SegmentManager:
    """
    Manages segment transcoding with single-threaded execution.

    Design rationale:
    - Single FFmpeg gets 100% CPU → fastest per-segment time
    - Multi-ahead prefetch uses idle time while user watches current segment
    - Multiple requests for same segment wait on shared Event
    - Prefetch queue allows scheduling multiple segments ahead
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._in_progress: dict[str, threading.Event] = {}  # key -> completion event
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: list[tuple[str, callable]] = []  # [(key, transcode_fn), ...]

        # Metrics
        self.total_segments = 0
        self.cache_hits = 0
        self.transcoding = False
        self.total_transcode_time = 0.0
        self.last_segment_time = 0.0
        self.min_segment_time = float('inf')
        self.max_segment_time = 0.0
        self._transcode_times: list[float] = []

    def get_segment(self, key: str, transcode_fn) -> str | None:
        """
        Get a segment, transcoding if necessary.
        If another thread is transcoding this segment, wait for it.
        """
        with self._lock:
            self.total_segments += 1

            # Check if already being transcoded
            if key in self._in_progress:
                event = self._in_progress[key]
                wait_for_other = True
            else:
                # We'll do the transcoding
                event = threading.Event()
                self._in_progress[key] = event
                wait_for_other = False

        if wait_for_other:
            # Wait for the other thread to finish
            completed = event.wait(timeout=120)
            if completed:
                # Other thread finished, file should exist - call transcode_fn which returns cached path
                return transcode_fn()
            else:
                # Timeout
                return None

        # We're the one transcoding
        try:
            result = transcode_fn()
            return result
        finally:
            with self._lock:
                self._in_progress.pop(key, None)
            event.set()  # Wake up any waiters

    def schedule_prefetch(self, key: str, transcode_fn):
        """Schedule prefetch for a segment (queued, processed sequentially)."""
        with self._lock:
            # Skip if already cached or in progress
            if key in self._in_progress:
                return

            # Skip if already in queue
            if any(k == key for k, _ in self._prefetch_queue):
                return

            # Add to queue
            self._prefetch_queue.append((key, transcode_fn))

            # Start prefetch thread if not running
            if not self._prefetch_thread or not self._prefetch_thread.is_alive():
                self._prefetch_thread = threading.Thread(
                    target=self._process_prefetch_queue,
                    daemon=True
                )
                self._prefetch_thread.start()

    def _process_prefetch_queue(self):
        """Process prefetch queue sequentially (single-threaded for max CPU per segment)."""
        while True:
            with self._lock:
                if not self._prefetch_queue:
                    return  # Queue empty, thread exits

                key, transcode_fn = self._prefetch_queue.pop(0)

                # Skip if already in progress
                if key in self._in_progress:
                    continue

                event = threading.Event()
                self._in_progress[key] = event

            try:
                transcode_fn()
            finally:
                with self._lock:
                    self._in_progress.pop(key, None)
                event.set()

    def is_in_progress(self, key: str) -> bool:
        with self._lock:
            return key in self._in_progress

    def record_cache_hit(self):
        with self._lock:
            self.total_segments += 1
            self.cache_hits += 1

    def record_transcode_time(self, elapsed: float):
        with self._lock:
            self.total_transcode_time += elapsed
            self.last_segment_time = elapsed
            if elapsed < self.min_segment_time:
                self.min_segment_time = elapsed
            if elapsed > self.max_segment_time:
                self.max_segment_time = elapsed
            self._transcode_times.append(elapsed)
            if len(self._transcode_times) > 100:
                self._transcode_times.pop(0)

    def get_metrics(self) -> dict:
        with self._lock:
            avg_time = sum(self._transcode_times) / len(self._transcode_times) if self._transcode_times else 0
            # Transcode ratio: time to generate / segment duration
            # e.g., 25% means 4s segment takes 1s to generate (good)
            # 100% means realtime (borderline), >100% means buffering
            last_ratio = (self.last_segment_time / SEGMENT_DURATION * 100) if self.last_segment_time > 0 else 0
            avg_ratio = (avg_time / SEGMENT_DURATION * 100) if avg_time > 0 else 0
            min_ratio = (self.min_segment_time / SEGMENT_DURATION * 100) if self.min_segment_time != float('inf') else 0
            max_ratio = (self.max_segment_time / SEGMENT_DURATION * 100) if self.max_segment_time > 0 else 0
            return {
                'total_segments': self.total_segments,
                'cache_hits': self.cache_hits,
                'cache_misses': self.total_segments - self.cache_hits,
                'cache_hit_rate': (self.cache_hits / self.total_segments * 100) if self.total_segments > 0 else 0,
                'prefetch_active': self._prefetch_thread.is_alive() if self._prefetch_thread else False,
                'prefetch_queue_size': len(self._prefetch_queue),
                'total_transcode_time': round(self.total_transcode_time, 2),
                'last_segment_time': round(self.last_segment_time, 2),
                'avg_segment_time': round(avg_time, 2),
                'min_segment_time': round(self.min_segment_time, 2) if self.min_segment_time != float('inf') else 0,
                'max_segment_time': round(self.max_segment_time, 2),
                'segment_duration': SEGMENT_DURATION,
                # Transcode ratio (lower is better, <100% required for smooth playback)
                'transcode_ratio_last': round(last_ratio, 1),
                'transcode_ratio_avg': round(avg_ratio, 1),
                'transcode_ratio_min': round(min_ratio, 1),
                'transcode_ratio_max': round(max_ratio, 1),
            }

    def reset_metrics(self):
        with self._lock:
            self.total_segments = 0
            self.cache_hits = 0
            self.total_transcode_time = 0.0
            self.last_segment_time = 0.0
            self.min_segment_time = float('inf')
            self.max_segment_time = 0.0
            self._transcode_times.clear()


# Global segment manager
segment_manager = SegmentManager()

# Global Stremio handler (BASE_URL set from environment)
stremio_handler = StremioHandler()


def get_file_hash(filepath: str) -> str:
    """Generate cache key from filepath."""
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


def get_video_info(filepath: str) -> dict | None:
    """Get video metadata using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', filepath],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except Exception:
        return None


def get_segment_path(file_hash: str, audio: int, resolution: str, segment: int) -> str:
    """Get cache path for a muxed segment (legacy, kept for compatibility)."""
    return os.path.join(CACHE_DIR, file_hash, f"seg_a{audio}_{resolution}_{segment:05d}.ts")


def get_video_segment_path(file_hash: str, resolution: str, segment: int) -> str:
    """Get cache path for a video-only segment."""
    return os.path.join(CACHE_DIR, file_hash, f"seg_v_{resolution}_{segment:05d}.ts")


def transcode_segment(filepath: str, file_hash: str, audio: int, resolution: str, segment: int) -> str | None:
    """
    Transcode a single segment using FFmpeg.

    Optimized for maximum single-segment speed:
    - libdav1d: fast multi-threaded AV1 decoder (if applicable)
    - ultrafast preset: prioritize speed over compression
    - threads 0: use all CPU cores for both decode and encode
    - tune zerolatency: reduce encoding latency
    """
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    output = get_segment_path(file_hash, audio, resolution, segment)

    # Already cached?
    if os.path.exists(output):
        return output

    start_offset = segment * SEGMENT_DURATION
    preset = RESOLUTIONS.get(resolution, RESOLUTIONS['original'])
    width, height, crf = preset

    # Get CPU count for threading
    cpu_count = os.cpu_count() or 8

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        # Decoder threading (libdav1d for AV1 uses these)
        '-threads', str(cpu_count),
        '-ss', str(start_offset),
        '-i', filepath,
        '-t', str(SEGMENT_DURATION),
        '-map', '0:v:0',
        '-map', f'0:a:{audio}',
        # Video output: optimized for speed with explicit threading
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-x264-params', f'threads={cpu_count}:lookahead_threads={min(cpu_count, 8)}',
        '-force_key_frames', 'expr:gte(t,0)',
        # Audio: try copy (faster), FFmpeg will fail if incompatible with mpegts
        '-c:a', 'aac', '-b:a', '128k', '-ac', '2',
        # Output format
        '-f', 'mpegts',
        '-mpegts_copyts', '1',
        '-output_ts_offset', str(start_offset),
    ]

    if width and height:
        cmd.extend(['-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'])

    cmd.append(output)

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.time() - start_time
        if result.returncode == 0 and os.path.exists(output):
            segment_manager.record_transcode_time(elapsed)
            return output
        print(f"FFmpeg error: {result.stderr.decode()[-200:]}")
    except Exception as e:
        print(f"Transcode error: {e}")

    return None


def transcode_video_segment(filepath: str, file_hash: str, resolution: str, segment: int) -> str | None:
    """
    Transcode a video segment with all audio tracks muxed in.

    Audio is included in video segments for maximum compatibility.
    Separate audio playlists are also available for players that support HLS alternate audio.
    """
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    output = get_video_segment_path(file_hash, resolution, segment)

    # Already cached?
    if os.path.exists(output):
        return output

    start_offset = segment * SEGMENT_DURATION
    preset = RESOLUTIONS.get(resolution, RESOLUTIONS['original'])
    width, height, crf = preset

    # Get CPU count for threading
    cpu_count = os.cpu_count() or 8

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-threads', str(cpu_count),
        '-ss', str(start_offset),
        '-i', filepath,
        '-t', str(SEGMENT_DURATION),
        '-map', '0:v:0',
        '-map', '0:a?',  # All audio streams (optional, ? means don't fail if no audio)
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-x264-params', f'threads={cpu_count}:lookahead_threads={min(cpu_count, 8)}',
        '-force_key_frames', 'expr:gte(t,0)',
        '-c:a', 'aac', '-b:a', '128k', '-ac', '2',
        '-f', 'mpegts',
        '-mpegts_copyts', '1',
        '-output_ts_offset', str(start_offset),
    ]

    if width and height:
        cmd.extend(['-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'])

    cmd.append(output)

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.time() - start_time
        if result.returncode == 0 and os.path.exists(output):
            segment_manager.record_transcode_time(elapsed)
            return output
        print(f"FFmpeg video error: {result.stderr.decode()[-200:]}")
    except Exception as e:
        print(f"Video transcode error: {e}")

    return None


def get_or_transcode_video_segment(filepath: str, file_hash: str, resolution: str, segment: int, info: dict) -> bytes | None:
    """Get video-only segment data, transcoding if necessary."""
    output = get_video_segment_path(file_hash, resolution, segment)
    key = f"{file_hash}:v:{resolution}:{segment}"

    # Check if a transcode is in progress for this segment
    if segment_manager.is_in_progress(key):
        result = segment_manager.get_segment(
            key,
            lambda: transcode_video_segment(filepath, file_hash, resolution, segment)
        )
        if result and os.path.exists(result):
            trigger_video_prefetch(filepath, file_hash, resolution, segment, info)
            with open(result, 'rb') as f:
                return f.read()
        return None

    # Fast path: cached
    if os.path.exists(output):
        segment_manager.record_cache_hit()
        trigger_video_prefetch(filepath, file_hash, resolution, segment, info)
        with open(output, 'rb') as f:
            return f.read()

    # Need to transcode
    result = segment_manager.get_segment(
        key,
        lambda: transcode_video_segment(filepath, file_hash, resolution, segment)
    )

    if result and os.path.exists(result):
        trigger_video_prefetch(filepath, file_hash, resolution, segment, info)
        with open(result, 'rb') as f:
            return f.read()

    return None


def trigger_video_prefetch(filepath: str, file_hash: str, resolution: str, current: int, info: dict):
    """Prefetch upcoming video-only segments."""
    duration = float(info.get('format', {}).get('duration', 0))
    total = int(duration / SEGMENT_DURATION) + 1

    for offset in range(1, PREFETCH_SEGMENTS + 1):
        next_seg = current + offset
        if next_seg >= total:
            break

        output = get_video_segment_path(file_hash, resolution, next_seg)
        if os.path.exists(output):
            continue

        key = f"{file_hash}:v:{resolution}:{next_seg}"
        seg = next_seg
        segment_manager.schedule_prefetch(
            key,
            lambda s=seg: transcode_video_segment(filepath, file_hash, resolution, s)
        )


def get_or_transcode_segment(filepath: str, file_hash: str, audio: int, resolution: str, segment: int, info: dict) -> bytes | None:
    """Get segment data, transcoding if necessary."""
    output = get_segment_path(file_hash, audio, resolution, segment)
    key = f"{file_hash}:{audio}:{resolution}:{segment}"

    # Check if a transcode is in progress for this segment
    # If so, we must wait for it rather than trying to read a partial file
    if segment_manager.is_in_progress(key):
        # Wait for the in-progress transcode
        result = segment_manager.get_segment(
            key,
            lambda: transcode_segment(filepath, file_hash, audio, resolution, segment)
        )
        if result and os.path.exists(result):
            trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
            with open(result, 'rb') as f:
                return f.read()
        return None

    # Fast path: cached (only safe if no transcode in progress)
    if os.path.exists(output):
        segment_manager.record_cache_hit()
        trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
        with open(output, 'rb') as f:
            return f.read()

    # Need to transcode
    result = segment_manager.get_segment(
        key,
        lambda: transcode_segment(filepath, file_hash, audio, resolution, segment)
    )

    if result and os.path.exists(result):
        trigger_prefetch(filepath, file_hash, audio, resolution, segment, info)
        with open(result, 'rb') as f:
            return f.read()

    return None


def trigger_prefetch(filepath: str, file_hash: str, audio: int, resolution: str, current: int, info: dict):
    """
    Prefetch upcoming segments (configurable via PREFETCH_SEGMENTS).

    Schedules transcoding for next N segments sequentially.
    SegmentManager ensures only one transcode runs at a time.
    """
    duration = float(info.get('format', {}).get('duration', 0))
    total = int(duration / SEGMENT_DURATION) + 1

    for offset in range(1, PREFETCH_SEGMENTS + 1):
        next_seg = current + offset
        if next_seg >= total:
            break

        output = get_segment_path(file_hash, audio, resolution, next_seg)
        if os.path.exists(output):
            continue  # Already cached

        key = f"{file_hash}:{audio}:{resolution}:{next_seg}"
        # Use closure to capture segment number correctly
        seg = next_seg
        segment_manager.schedule_prefetch(
            key,
            lambda s=seg: transcode_segment(filepath, file_hash, audio, resolution, s)
        )


def generate_master_playlist(info: dict, resolution_filter: str = None) -> str:
    """
    Generate HLS master playlist with multi-audio support.

    Uses separate audio playlists (audio_N.m3u8) for HLS alternate audio.
    Video segments are video-only, audio comes from separate audio tracks.
    """
    streams = info.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']

    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]

    # Audio tracks - each with separate playlist URI for HLS alternate audio
    for i, a in enumerate(audio_streams):
        lang = a.get('tags', {}).get('language', 'und')
        title = a.get('tags', {}).get('title', '')
        channels = a.get('channels', 2)

        # Build descriptive name like MetaMesh: "Japanese - AAC 2.0"
        codec = a.get('codec_name', 'AAC').upper()
        ch_str = f"{channels}.0" if channels else "2.0"
        if title:
            name = title
        else:
            lang_name = lang.upper() if lang != 'und' else f"Audio {i+1}"
            name = f"{lang_name} - {codec} {ch_str}"

        # URI points to separate audio playlist for proper audio track switching
        lines.append(f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{name}",LANGUAGE="{lang}",CHANNELS="{channels}",DEFAULT=YES,AUTOSELECT=YES,URI="audio_{i}.m3u8"')

    # Subtitle tracks
    for i, s in enumerate(subtitle_streams):
        lang = s.get('tags', {}).get('language', 'und')
        title = s.get('tags', {}).get('title', '')
        name = f"{title}" if title else (lang.upper() if lang != 'und' else f"Subtitle {i+1}")
        lines.append(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{name}",LANGUAGE="{lang}",URI="subtitle_{i}.m3u8"')

    # Video variants - include all quality levels like MetaMesh
    if video:
        src_w = video.get('width', 1920)
        src_h = video.get('height', 1080)

        subs_ref = ',SUBTITLES="subs"' if subtitle_streams else ''
        audio_ref = ',AUDIO="audio"' if audio_streams else ''

        if resolution_filter:
            # Single quality requested (e.g., master_720p.m3u8)
            res_name = resolution_filter
            w, h, _ = RESOLUTIONS.get(res_name, RESOLUTIONS['original'])
            width = w or src_w
            height = h or src_h
            bw = {None: 5000000, 1080: 4000000, 720: 2500000, 480: 1200000, 360: 800000}.get(h, 5000000)
            lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={width}x{height}{audio_ref}{subs_ref}')
            lines.append(f"playlist_{res_name}_video.m3u8")
        else:
            # Full ABR ladder - all qualities from source down
            # (res_key, target_height, bandwidth)
            quality_order = [
                ('original', None, 5000000, 'source'),
                ('1080p', 1080, 4000000, '1080p'),
                ('720p', 720, 2500000, '720p'),
                ('480p', 480, 1200000, '480p'),
                ('360p', 360, 800000, '360p'),
            ]
            for res_key, target_h, bw, playlist_name in quality_order:
                # Skip resolutions higher than source
                if target_h and target_h > src_h:
                    continue
                w, h, _ = RESOLUTIONS.get(res_key, RESOLUTIONS['original'])
                width = w or src_w
                height = h or src_h
                label = f"{height}p (Original)" if res_key == 'original' else f"{height}p"
                lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={width}x{height}{audio_ref}{subs_ref},NAME="{label}"')
                lines.append(f"playlist_{playlist_name}_video.m3u8")

    return "\n".join(lines) + "\n"


def generate_audio_playlist(info: dict, audio_index: int) -> str:
    """Generate audio-only HLS playlist pointing to audio-only segments."""
    duration = float(info.get('format', {}).get('duration', 0))
    num_segments = int(duration / SEGMENT_DURATION) + 1

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        ""
    ]

    for i in range(num_segments):
        seg_dur = min(SEGMENT_DURATION, duration - (i * SEGMENT_DURATION))
        if seg_dur > 0.1:
            lines.append(f"#EXTINF:{seg_dur:.3f},")
            lines.append(f"audio_{audio_index}_{i:05d}.ts")

    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines)


def transcode_audio_segment(filepath: str, file_hash: str, audio_index: int, segment: int) -> str | None:
    """Transcode audio-only segment."""
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    output_file = os.path.join(cache_dir, f"audio_{audio_index}_{segment:05d}.ts")

    if os.path.exists(output_file):
        return output_file

    start_time = segment * SEGMENT_DURATION

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', filepath,
        '-t', str(SEGMENT_DURATION),
        '-map', f'0:a:{audio_index}',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ac', '2',
        '-f', 'mpegts',
        '-mpegts_copyts', '1',
        '-output_ts_offset', str(start_time),
        output_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_file):
            return output_file
        else:
            print(f"FFmpeg audio error: {result.stderr.decode()}")
            return None
    except Exception as e:
        print(f"Audio transcode error: {e}")
        return None


def generate_stream_playlist(info: dict, audio: int, resolution: str) -> str:
    """Generate HLS video-only stream playlist.

    Video segments are video-only (no audio) to support HLS alternate audio.
    Audio comes from separate audio track playlists (audio_N.m3u8).
    """
    duration = float(info.get('format', {}).get('duration', 0))
    num_segments = int(duration / SEGMENT_DURATION) + 1

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        ""
    ]

    for i in range(num_segments):
        seg_dur = min(SEGMENT_DURATION, duration - (i * SEGMENT_DURATION))
        if seg_dur > 0.1:
            lines.append(f"#EXTINF:{seg_dur:.3f},")
            # Video-only segments (no audio muxed)
            lines.append(f"seg_v_{resolution}_{i:05d}.ts")

    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines)


def generate_subtitle_playlist(info: dict, sub_index: int) -> str:
    """Generate subtitle playlist."""
    duration = float(info.get('format', {}).get('duration', 0))
    return "\n".join([
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{int(duration) + 1}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "",
        f"#EXTINF:{duration:.3f},",
        f"subtitle_{sub_index}.vtt",
        "#EXT-X-ENDLIST"
    ])


def extract_subtitle(filepath: str, file_hash: str, sub_index: int) -> str | None:
    """Extract subtitle as WebVTT."""
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    vtt_file = os.path.join(cache_dir, f"subtitle_{sub_index}.vtt")

    if os.path.exists(vtt_file):
        with open(vtt_file, 'r', encoding='utf-8') as f:
            return f.read()

    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', filepath, '-map', f'0:s:{sub_index}', '-c:s', 'webvtt', vtt_file],
            capture_output=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(vtt_file):
            with open(vtt_file, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Subtitle error: {e}")

    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Quiet logging

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_data(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

    def do_HEAD(self):
        """Handle HEAD requests - return 200 OK for valid paths."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # For stremio and transcode endpoints, just return 200 OK
        if path.startswith('/stremio/') or path.startswith('/transcode/'):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            if path.endswith('.m3u8'):
                self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            elif path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            elif path.endswith('.ts'):
                self.send_header('Content-Type', 'video/mp2t')
            elif path.endswith('.vtt'):
                self.send_header('Content-Type', 'text/vtt')
            self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        # Stremio addon endpoints
        if path == '/stremio/manifest.json':
            return self.handle_stremio_manifest()

        m = re.match(r'^/stremio/catalog/(\w+)/([^/]+)(?:/([^.]+))?\.json$', path)
        if m:
            return self.handle_stremio_catalog(m.group(1), m.group(2), m.group(3))

        m = re.match(r'^/stremio/meta/(\w+)/([^/]+)\.json$', path)
        if m:
            return self.handle_stremio_meta(m.group(1), m.group(2))

        m = re.match(r'^/stremio/stream/(\w+)/([^/]+)\.json$', path)
        if m:
            return self.handle_stremio_stream(m.group(1), m.group(2))

        # Metrics
        if path in ('/metrics', '/transcode/metrics'):
            return self.send_json(segment_manager.get_metrics())

        if path == '/transcode/reset-metrics':
            segment_manager.reset_metrics()
            return self.send_json({'status': 'ok'})

        # Direct file serving
        m = re.match(r'^/direct/(.+)$', path)
        if m:
            return self.handle_direct_file(m.group(1))

        # Master playlist (all resolutions)
        m = re.match(r'^/transcode/(.+?)/master\.m3u8$', path)
        if m:
            return self.handle_master(m.group(1))

        # Quality-specific master playlist (e.g., master_720p.m3u8)
        m = re.match(r'^/transcode/(.+?)/master_(\w+)\.m3u8$', path)
        if m:
            return self.handle_master(m.group(1), m.group(2))

        # Stream playlist (old naming)
        m = re.match(r'^/transcode/(.+?)/stream_a(\d+)_(\w+)\.m3u8$', path)
        if m:
            return self.handle_stream(m.group(1), int(m.group(2)), m.group(3))

        # Video playlist (MetaMesh-style naming: playlist_source_video.m3u8, playlist_720p_video.m3u8)
        m = re.match(r'^/transcode/(.+?)/playlist_(\w+)_video\.m3u8$', path)
        if m:
            resolution = m.group(2)
            # Map 'source' to 'original', keep '720p' etc as-is (RESOLUTIONS keys use 'p' suffix)
            if resolution == 'source':
                resolution = 'original'
            return self.handle_stream(m.group(1), 0, resolution)

        # Audio playlist (for HLS alternate audio)
        m = re.match(r'^/transcode/(.+?)/audio_(\d+)\.m3u8$', path)
        if m:
            return self.handle_audio_playlist(m.group(1), int(m.group(2)))

        # Audio segment
        m = re.match(r'^/transcode/(.+?)/audio_(\d+)_(\d+)\.ts$', path)
        if m:
            return self.handle_audio_segment(m.group(1), int(m.group(2)), int(m.group(3)))

        # Video-only segment (for HLS alternate audio support)
        m = re.match(r'^/transcode/(.+?)/seg_v_(\w+)_(\d+)\.ts$', path)
        if m:
            return self.handle_video_segment(m.group(1), m.group(2), int(m.group(3)))

        # Muxed video+audio segment (legacy, kept for compatibility)
        m = re.match(r'^/transcode/(.+?)/seg_a(\d+)_(\w+)_(\d+)\.ts$', path)
        if m:
            return self.handle_segment(m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))

        # Subtitle playlist (both subtitle_N and subs_N naming)
        m = re.match(r'^/transcode/(.+?)/(?:subtitle|subs)_(\d+)\.m3u8$', path)
        if m:
            return self.handle_subtitle_playlist(m.group(1), int(m.group(2)))

        # Subtitle VTT (both subtitle_N and subs_N naming for compatibility)
        m = re.match(r'^/transcode/(.+?)/(?:subtitle|subs)_(\d+)\.vtt$', path)
        if m:
            return self.handle_subtitle_vtt(m.group(1), int(m.group(2)))

        self.send_error(404)

    # Stremio addon handlers
    def handle_stremio_manifest(self):
        data, content_type = stremio_handler.handle_manifest()
        self.send_data(data, content_type)

    def handle_stremio_catalog(self, catalog_type: str, catalog_id: str, extra: str = None):
        extra_dict = {}
        if extra:
            # Parse extra params like "search=query"
            for param in extra.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    extra_dict[key] = unquote(value)

        data, content_type = stremio_handler.handle_catalog(catalog_type, catalog_id, extra_dict)
        self.send_data(data, content_type)

    def handle_stremio_meta(self, meta_type: str, meta_id: str):
        data, content_type = stremio_handler.handle_meta(meta_type, meta_id)
        if data:
            self.send_data(data, content_type)
        else:
            self.send_error(404, "Video not found")

    def handle_stremio_stream(self, stream_type: str, stream_id: str):
        # Determine base URL from request headers (must be done on every request)
        # Priority: X-Forwarded-Host > Host header
        forwarded_host = self.headers.get('X-Forwarded-Host', '')
        forwarded_proto = self.headers.get('X-Forwarded-Proto', 'https')  # Default to https for safety
        host = self.headers.get('Host', 'localhost:8080')

        if forwarded_host:
            base_url = f"{forwarded_proto}://{forwarded_host}"
        else:
            # No proxy - use Host header, assume http for local
            base_url = f"http://{host}"

        data, content_type = stremio_handler.handle_stream(stream_type, stream_id, base_url)
        if data:
            self.send_data(data, content_type)
        else:
            self.send_error(404, "Video not found")

    def get_file_info(self, filepath: str):
        """Get file path and info, or send error."""
        full_path = os.path.join(MEDIA_DIR, filepath)
        if not os.path.exists(full_path):
            self.send_error(404, f"File not found: {filepath}")
            return None, None, None

        info = get_video_info(full_path)
        if not info:
            self.send_error(500, "Could not probe file")
            return None, None, None

        return full_path, get_file_hash(filepath), info

    def handle_master(self, filepath: str, resolution: str = None):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return
        self.send_data(generate_master_playlist(info, resolution).encode(), 'application/vnd.apple.mpegurl')

    def handle_stream(self, filepath: str, audio: int, resolution: str):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return
        self.send_data(generate_stream_playlist(info, audio, resolution).encode(), 'application/vnd.apple.mpegurl')

    def handle_segment(self, filepath: str, audio: int, resolution: str, segment: int):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        data = get_or_transcode_segment(full_path, file_hash, audio, resolution, segment, info)
        if data:
            self.send_data(data, 'video/mp2t')
        else:
            self.send_error(500, "Transcode failed")

    def handle_video_segment(self, filepath: str, resolution: str, segment: int):
        """Handle video-only segment request (no audio muxed)."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        data = get_or_transcode_video_segment(full_path, file_hash, resolution, segment, info)
        if data:
            self.send_data(data, 'video/mp2t')
        else:
            self.send_error(500, "Video transcode failed")

    def handle_audio_playlist(self, filepath: str, audio_index: int):
        """Handle audio playlist request."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return
        self.send_data(generate_audio_playlist(info, audio_index).encode(), 'application/vnd.apple.mpegurl')

    def handle_audio_segment(self, filepath: str, audio_index: int, segment: int):
        """Handle audio-only segment request."""
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        output_file = transcode_audio_segment(full_path, file_hash, audio_index, segment)
        if output_file and os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                self.send_data(f.read(), 'video/mp2t')
        else:
            self.send_error(500, "Audio transcode failed")

    def handle_subtitle_playlist(self, filepath: str, sub_index: int):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return
        self.send_data(generate_subtitle_playlist(info, sub_index).encode(), 'application/vnd.apple.mpegurl')

    def handle_subtitle_vtt(self, filepath: str, sub_index: int):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return

        content = extract_subtitle(full_path, file_hash, sub_index)
        if content:
            self.send_data(content.encode('utf-8'), 'text/vtt')
        else:
            self.send_error(500, "Subtitle extraction failed")

    def handle_direct_file(self, filepath: str):
        """Serve raw video file with range request support for seeking."""
        full_path = os.path.join(MEDIA_DIR, filepath)
        if not os.path.exists(full_path):
            self.send_error(404, f"File not found: {filepath}")
            return

        # Determine content type from extension
        ext = os.path.splitext(filepath)[1].lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.m4v': 'video/x-m4v',
            '.ts': 'video/mp2t',
            '.m2ts': 'video/mp2t',
        }
        content_type = content_types.get(ext, 'application/octet-stream')

        file_size = os.path.getsize(full_path)
        range_header = self.headers.get('Range')

        if range_header:
            # Parse range request (e.g., "bytes=0-1023")
            range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
            if range_match:
                start = int(range_match.group(1)) if range_match.group(1) else 0
                end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

                # Clamp to file size
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)  # Partial Content
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', length)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with open(full_path, 'rb') as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 64 * 1024  # 64KB chunks
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return

        # No range request - serve entire file
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', file_size)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        with open(full_path, 'rb') as f:
            chunk_size = 64 * 1024  # 64KB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.wfile.write(chunk)


class ThreadedServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address), daemon=True)
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    print(f"HLS Transcoder starting on port {PORT}")
    print(f"Media: {MEDIA_DIR} | Cache: {CACHE_DIR} | Segment: {SEGMENT_DURATION}s")
    print(f"Mode: Single-threaded + one-ahead prefetch (ultrafast preset)")
    ThreadedServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
