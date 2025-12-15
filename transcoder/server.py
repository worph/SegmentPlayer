#!/usr/bin/env python3
"""
Live HLS Transcoder Server
Transcodes video segments on-demand using FFmpeg
"""

import os
import subprocess
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
import json
import re

MEDIA_DIR = os.environ.get('MEDIA_DIR', '/data/media')
CACHE_DIR = os.environ.get('CACHE_DIR', '/data/cache')
SEGMENT_DURATION = int(os.environ.get('SEGMENT_DURATION', '4'))
PORT = int(os.environ.get('PORT', '8080'))

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def get_file_hash(filepath):
    """Generate short hash for cache key"""
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


def get_video_info(filepath):
    """Get video duration and stream info using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except:
        return None


def generate_master_playlist(filepath, file_hash, info):
    """Generate HLS master playlist - use EXT-X-MEDIA for audio tracks"""
    streams = info.get('streams', [])

    video_streams = [s for s in streams if s.get('codec_type') == 'video']
    audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
    subtitle_streams = [s for s in streams if s.get('codec_type') == 'subtitle']

    playlist = "#EXTM3U\n#EXT-X-VERSION:4\n\n"

    # Add audio tracks as EXT-X-MEDIA
    for i, audio in enumerate(audio_streams):
        lang = audio.get('tags', {}).get('language', f'aud{i}')
        title = audio.get('tags', {}).get('title', f'Audio {i+1}')
        default = "YES" if i == 0 else "NO"
        playlist += f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{title} ({lang})",LANGUAGE="{lang}",DEFAULT={default},AUTOSELECT=YES,URI="stream_{i}.m3u8"\n'

    # Add subtitle tracks as EXT-X-MEDIA
    for i, sub in enumerate(subtitle_streams):
        lang = sub.get('tags', {}).get('language', f'sub{i}')
        title = sub.get('tags', {}).get('title', f'Subtitle {i+1}')
        default = "NO"
        playlist += f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{title} ({lang})",LANGUAGE="{lang}",DEFAULT={default},AUTOSELECT=YES,URI="subs_{i}.m3u8"\n'

    playlist += "\n"

    # Add single video variant that references audio and subtitle groups
    if video_streams and audio_streams:
        v = video_streams[0]
        width = v.get('width', 1920)
        height = v.get('height', 1080)
        subs_attr = ',SUBTITLES="subs"' if subtitle_streams else ''
        playlist += f'#EXT-X-STREAM-INF:BANDWIDTH=4000000,RESOLUTION={width}x{height},AUDIO="audio"{subs_attr}\n'
        playlist += "stream_0.m3u8\n"

    return playlist


def generate_subtitle_playlist(filepath, file_hash, sub_index, info):
    """Generate HLS subtitle playlist"""
    duration = float(info.get('format', {}).get('duration', 0))

    playlist = "#EXTM3U\n#EXT-X-VERSION:3\n"
    playlist += f"#EXT-X-TARGETDURATION:{int(duration) + 1}\n"
    playlist += "#EXT-X-MEDIA-SEQUENCE:0\n"
    playlist += "#EXT-X-PLAYLIST-TYPE:VOD\n\n"
    playlist += f"#EXTINF:{duration:.3f},\n"
    playlist += f"subs_{sub_index}.vtt\n"
    playlist += "#EXT-X-ENDLIST\n"

    return playlist


def extract_subtitle(filepath, file_hash, sub_index):
    """Extract subtitle track as WebVTT"""
    cache_path = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_path, exist_ok=True)

    vtt_file = os.path.join(cache_path, f"subs_{sub_index}.vtt")

    # Check cache
    if os.path.exists(vtt_file):
        with open(vtt_file, 'r', encoding='utf-8') as f:
            return f.read()

    # Extract subtitle to VTT
    cmd = [
        'ffmpeg', '-y',
        '-i', filepath,
        '-map', f'0:s:{sub_index}',
        '-c:s', 'webvtt',
        vtt_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if os.path.exists(vtt_file):
            with open(vtt_file, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Subtitle extraction error: {e}")

    return None


def generate_stream_playlist(filepath, file_hash, audio_index, info):
    """Generate HLS stream playlist for video+audio"""
    duration = float(info.get('format', {}).get('duration', 0))
    num_segments = int(duration / SEGMENT_DURATION) + 1

    playlist = "#EXTM3U\n#EXT-X-VERSION:3\n"
    playlist += f"#EXT-X-TARGETDURATION:{SEGMENT_DURATION}\n"
    playlist += "#EXT-X-MEDIA-SEQUENCE:0\n"
    playlist += "#EXT-X-PLAYLIST-TYPE:VOD\n\n"

    for i in range(num_segments):
        seg_duration = min(SEGMENT_DURATION, duration - (i * SEGMENT_DURATION))
        if seg_duration > 0.1:
            playlist += f"#EXTINF:{seg_duration:.3f},\n"
            playlist += f"seg_{audio_index}_{i:05d}.ts\n"

    playlist += "#EXT-X-ENDLIST\n"
    return playlist


def transcode_segment(filepath, file_hash, audio_index, segment_num):
    """Transcode a single segment with video + specific audio track"""
    cache_path = os.path.join(CACHE_DIR, file_hash)
    os.makedirs(cache_path, exist_ok=True)

    segment_file = os.path.join(cache_path, f"seg_{audio_index}_{segment_num:05d}.ts")

    # Check cache first
    if os.path.exists(segment_file):
        with open(segment_file, 'rb') as f:
            return f.read()

    start_time = segment_num * SEGMENT_DURATION

    # FFmpeg command with proper timestamp handling
    # -ss before -i for fast seeking, then use -copyts and output_ts_offset to fix timestamps
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', filepath,
        '-t', str(SEGMENT_DURATION),
        '-map', '0:v:0',
        '-map', f'0:a:{audio_index}',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-force_key_frames', 'expr:gte(t,0)',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ac', '2',
        '-f', 'mpegts',
        '-mpegts_copyts', '1',
        '-output_ts_offset', str(start_time),
        segment_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()[-500:]}")
        if os.path.exists(segment_file):
            with open(segment_file, 'rb') as f:
                return f.read()
    except Exception as e:
        print(f"Transcode error: {e}")

    return None


class TranscoderHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Range')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Paths:
        # /transcode/{filepath}/master.m3u8
        # /transcode/{filepath}/stream_{audio_idx}.m3u8
        # /transcode/{filepath}/seg_{audio_idx}_{segment_num}.ts
        # /transcode/{filepath}/subs_{sub_idx}.m3u8
        # /transcode/{filepath}/subs_{sub_idx}.vtt

        match = re.match(r'^/transcode/(.+?)/(master\.m3u8|stream_(\d+)\.m3u8|seg_(\d+)_(\d+)\.ts|subs_(\d+)\.(m3u8|vtt))$', path)

        if not match:
            self.send_error(404, "Invalid path")
            return

        filepath = match.group(1)
        resource = match.group(2)
        full_path = os.path.join(MEDIA_DIR, filepath)

        if not os.path.exists(full_path):
            self.send_error(404, f"File not found: {filepath}")
            return

        file_hash = get_file_hash(filepath)
        info = get_video_info(full_path)

        if not info:
            self.send_error(500, "Could not probe file")
            return

        if resource == 'master.m3u8':
            content = generate_master_playlist(full_path, file_hash, info)
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode())

        elif resource.startswith('stream_'):
            audio_index = int(match.group(3))
            content = generate_stream_playlist(full_path, file_hash, audio_index, info)
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode())

        elif resource.startswith('seg_'):
            audio_index = int(match.group(4))
            segment_num = int(match.group(5))

            data = transcode_segment(full_path, file_hash, audio_index, segment_num)

            if data:
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp2t')
                self.send_header('Content-Length', len(data))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(500, "Transcode failed")

        elif resource.startswith('subs_'):
            sub_index = int(match.group(6))
            ext = match.group(7)

            if ext == 'm3u8':
                content = generate_subtitle_playlist(full_path, file_hash, sub_index, info)
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(content.encode())
            else:  # vtt
                content = extract_subtitle(full_path, file_hash, sub_index)
                if content:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/vtt')
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                else:
                    self.send_error(500, "Subtitle extraction failed")


def main():
    print(f"Starting HLS Transcoder on port {PORT}")
    print(f"Media directory: {MEDIA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Segment duration: {SEGMENT_DURATION}s")

    server = HTTPServer(('0.0.0.0', PORT), TranscoderHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
