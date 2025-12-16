#!/usr/bin/env python3
"""
HLS Live Transcoder Server
Transcodes video segments on-demand using FFmpeg.
"""
from __future__ import annotations

import os
import subprocess
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
import json
import re
from typing import Optional, Dict, Callable

# Configuration
MEDIA_DIR = os.environ.get('MEDIA_DIR', '/data/media')
CACHE_DIR = os.environ.get('CACHE_DIR', '/data/cache')
SEGMENT_DURATION = int(os.environ.get('SEGMENT_DURATION', '4'))
PORT = int(os.environ.get('PORT', '8080'))
MAX_WORKERS = int(os.environ.get('MAX_CONCURRENT_TRANSCODES', '4'))
PREFETCH_COUNT = int(os.environ.get('PREFETCH_SEGMENTS', '3'))

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
    """Manages segment transcoding with proper concurrency handling."""

    def __init__(self, max_workers: int):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

        # Metrics
        self.total_segments = 0
        self.cache_hits = 0
        self.active_jobs = 0
        self.total_transcode_time = 0.0
        self.last_segment_time = 0.0
        self.min_segment_time = float('inf')
        self.max_segment_time = 0.0
        self._transcode_times: list[float] = []  # Last N times for avg

    def get_segment(self, key: str, transcode_fn) -> str | None:
        """
        Get a segment, transcoding if necessary.
        Handles concurrent requests for the same segment.
        Returns the path to the segment file or None on failure.
        """
        with self._lock:
            self.total_segments += 1

            # Check if already being transcoded
            if key in self._futures:
                future = self._futures[key]
            else:
                future = None

        if future:
            # Wait for existing transcode to complete
            try:
                return future.result(timeout=120)
            except Exception:
                return None

        # We need to transcode - submit the job
        with self._lock:
            # Double-check another thread didn't start it
            if key in self._futures:
                future = self._futures[key]
            else:
                self.active_jobs += 1
                future = self._executor.submit(self._transcode_wrapper, key, transcode_fn)
                self._futures[key] = future

        try:
            return future.result(timeout=120)
        except Exception:
            return None

    def _transcode_wrapper(self, key: str, transcode_fn) -> str | None:
        """Wrapper that cleans up after transcoding."""
        try:
            return transcode_fn()
        finally:
            with self._lock:
                self.active_jobs -= 1
                self._futures.pop(key, None)

    def prefetch(self, key: str, transcode_fn):
        """Submit a prefetch job (fire and forget)."""
        with self._lock:
            if key in self._futures:
                return
            self.active_jobs += 1
            future = self._executor.submit(self._transcode_wrapper, key, transcode_fn)
            self._futures[key] = future

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
            return {
                'total_segments': self.total_segments,
                'cache_hits': self.cache_hits,
                'cache_misses': self.total_segments - self.cache_hits,
                'cache_hit_rate': (self.cache_hits / self.total_segments * 100) if self.total_segments > 0 else 0,
                'active_jobs': self.active_jobs,
                'total_transcode_time': round(self.total_transcode_time, 2),
                'last_segment_time': round(self.last_segment_time, 2),
                'avg_segment_time': round(avg_time, 2),
                'min_segment_time': round(self.min_segment_time, 2) if self.min_segment_time != float('inf') else 0,
                'max_segment_time': round(self.max_segment_time, 2),
                'segment_duration': SEGMENT_DURATION,
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

    def is_in_progress(self, key: str) -> bool:
        """Check if a segment is currently being transcoded."""
        with self._lock:
            return key in self._futures


# Global segment manager
segment_manager = SegmentManager(MAX_WORKERS)


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
    """Get cache path for a segment."""
    return os.path.join(CACHE_DIR, file_hash, f"seg_a{audio}_{resolution}_{segment:05d}.ts")


def transcode_segment(filepath: str, file_hash: str, audio: int, resolution: str, segment: int) -> str | None:
    """Transcode a single segment using FFmpeg."""
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    output = get_segment_path(file_hash, audio, resolution, segment)

    # Already cached?
    if os.path.exists(output):
        return output

    start_offset = segment * SEGMENT_DURATION
    preset = RESOLUTIONS.get(resolution, RESOLUTIONS['original'])
    width, height, crf = preset

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', str(start_offset),
        '-i', filepath,
        '-t', str(SEGMENT_DURATION),
        '-map', '0:v:0',
        '-map', f'0:a:{audio}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
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
        print(f"FFmpeg error: {result.stderr.decode()[-200:]}")
    except Exception as e:
        print(f"Transcode error: {e}")

    return None


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
    """Prefetch upcoming segments."""
    duration = float(info.get('format', {}).get('duration', 0))
    total = int(duration / SEGMENT_DURATION) + 1

    for i in range(1, PREFETCH_COUNT + 1):
        next_seg = current + i
        if next_seg >= total:
            break

        output = get_segment_path(file_hash, audio, resolution, next_seg)
        if os.path.exists(output):
            continue

        key = f"{file_hash}:{audio}:{resolution}:{next_seg}"
        segment_manager.prefetch(
            key,
            lambda s=next_seg: transcode_segment(filepath, file_hash, audio, resolution, s)
        )


def generate_master_playlist(info: dict) -> str:
    """Generate HLS master playlist."""
    streams = info.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']

    lines = ["#EXTM3U", "#EXT-X-VERSION:4", ""]

    # Audio tracks
    for i, a in enumerate(audio_streams):
        lang = a.get('tags', {}).get('language', f'aud{i}')
        title = a.get('tags', {}).get('title', f'Audio {i+1}')
        default = "YES" if i == 0 else "NO"
        lines.append(f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{title} ({lang})",LANGUAGE="{lang}",DEFAULT={default},AUTOSELECT=YES,URI="stream_a{i}_original.m3u8"')

    # Subtitle tracks
    for i, s in enumerate(subtitle_streams):
        lang = s.get('tags', {}).get('language', f'sub{i}')
        title = s.get('tags', {}).get('title', f'Subtitle {i+1}')
        lines.append(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{title} ({lang})",LANGUAGE="{lang}",DEFAULT=NO,AUTOSELECT=YES,URI="subs_{i}.m3u8"')

    lines.append("")

    # Video variants
    if video:
        src_w = video.get('width', 1920)
        src_h = video.get('height', 1080)
        subs = ',SUBTITLES="subs"' if subtitle_streams else ''

        for name, (w, h, _) in RESOLUTIONS.items():
            if name != 'original' and h and h > src_h:
                continue

            width = w or src_w
            height = h or src_h
            bw = {None: 5000000, 1080: 4000000, 720: 2500000, 480: 1200000, 360: 800000}.get(h, 5000000)

            lines.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={width}x{height},AUDIO="audio"{subs}')
            lines.append(f"stream_a0_{name}.m3u8")

    return "\n".join(lines)


def generate_stream_playlist(info: dict, audio: int, resolution: str) -> str:
    """Generate HLS stream playlist."""
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
            lines.append(f"seg_a{audio}_{resolution}_{i:05d}.ts")

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
        f"subs_{sub_index}.vtt",
        "#EXT-X-ENDLIST"
    ])


def extract_subtitle(filepath: str, file_hash: str, sub_index: int) -> str | None:
    """Extract subtitle as WebVTT."""
    cache_dir = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_dir, exist_ok=True)

    vtt_file = os.path.join(cache_dir, f"subs_{sub_index}.vtt")

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
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        # Metrics
        if path in ('/metrics', '/transcode/metrics'):
            return self.send_json(segment_manager.get_metrics())

        if path == '/transcode/reset-metrics':
            segment_manager.reset_metrics()
            return self.send_json({'status': 'ok'})

        # Master playlist
        m = re.match(r'^/transcode/(.+?)/master\.m3u8$', path)
        if m:
            return self.handle_master(m.group(1))

        # Stream playlist
        m = re.match(r'^/transcode/(.+?)/stream_a(\d+)_(\w+)\.m3u8$', path)
        if m:
            return self.handle_stream(m.group(1), int(m.group(2)), m.group(3))

        # Segment
        m = re.match(r'^/transcode/(.+?)/seg_a(\d+)_(\w+)_(\d+)\.ts$', path)
        if m:
            return self.handle_segment(m.group(1), int(m.group(2)), m.group(3), int(m.group(4)))

        # Subtitle playlist
        m = re.match(r'^/transcode/(.+?)/subs_(\d+)\.m3u8$', path)
        if m:
            return self.handle_subtitle_playlist(m.group(1), int(m.group(2)))

        # Subtitle VTT
        m = re.match(r'^/transcode/(.+?)/subs_(\d+)\.vtt$', path)
        if m:
            return self.handle_subtitle_vtt(m.group(1), int(m.group(2)))

        self.send_error(404)

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

    def handle_master(self, filepath: str):
        full_path, file_hash, info = self.get_file_info(filepath)
        if not info:
            return
        self.send_data(generate_master_playlist(info).encode(), 'application/vnd.apple.mpegurl')

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
    print(f"Media: {MEDIA_DIR} | Cache: {CACHE_DIR} | Segment: {SEGMENT_DURATION}s | Workers: {MAX_WORKERS}")
    ThreadedServer(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
