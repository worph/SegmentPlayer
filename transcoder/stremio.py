"""
Stremio Addon for SegmentPlayer

Provides a Stremio-compatible addon that lists media files and serves HLS streams.
This is a simple addon that:
1. Lists all video files as a catalog
2. Provides HLS stream URLs for playback in Stremio

Endpoints (Stremio protocol):
- GET /stremio/manifest.json - Addon manifest
- GET /stremio/catalog/:type/:id.json - List of videos
- GET /stremio/meta/:type/:id.json - Video metadata
- GET /stremio/stream/:type/:id.json - Stream URLs
"""
from __future__ import annotations

import os
import json
import hashlib
import subprocess
from urllib.parse import quote, unquote
from typing import Optional

# Use same config as main server
MEDIA_DIR = os.environ.get('MEDIA_DIR', '/data/media')
BASE_URL = os.environ.get('BASE_URL', '')  # e.g., http://192.168.1.100:8080

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.ts', '.m2ts'}

# Manifest definition
MANIFEST = {
    "id": "com.segmentplayer.addon",
    "version": "1.0.0",
    "name": "SegmentPlayer",
    "description": "Stream your local media library with on-the-fly HLS transcoding",
    "logo": "https://raw.githubusercontent.com/user/segmentplayer/main/logo.png",
    "resources": [
        "catalog",
        {
            "name": "meta",
            "types": ["movie", "series"],
            "idPrefixes": ["sp_"]
        },
        {
            "name": "stream",
            "types": ["movie", "series"],
            "idPrefixes": ["sp_"]
        }
    ],
    "types": ["movie", "series"],
    "catalogs": [
        {
            "type": "movie",
            "id": "segmentplayer_all",
            "name": "SegmentPlayer",
            "extra": [
                {"name": "search", "isRequired": False},
                {"name": "genre", "isRequired": False}
            ]
        }
    ],
    "idPrefixes": ["sp_"],
    "behaviorHints": {
        "configurable": False,
        "configurationRequired": False
    }
}


def get_file_id(filepath: str) -> str:
    """Generate a unique ID for a file path."""
    return "sp_" + hashlib.md5(filepath.encode()).hexdigest()[:16]


def get_filepath_from_id(file_id: str) -> Optional[str]:
    """Reverse lookup: find filepath from ID by scanning media directory."""
    # This is inefficient but works for small collections
    # For production, consider using a database or cache
    prefix = file_id.replace("sp_", "")

    for root, dirs, files in os.walk(MEDIA_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, MEDIA_DIR)
                if hashlib.md5(rel_path.encode()).hexdigest()[:16] == prefix:
                    return rel_path

    return None


def get_video_info(filepath: str) -> Optional[dict]:
    """Get video metadata using ffprobe."""
    full_path = os.path.join(MEDIA_DIR, filepath)
    if not os.path.exists(full_path):
        return None

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', full_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def scan_media_files() -> list[dict]:
    """Scan media directory and return list of video files."""
    videos = []

    for root, dirs, files in os.walk(MEDIA_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, MEDIA_DIR)

                # Get file size
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0

                # Extract info from filename
                name = os.path.splitext(filename)[0]

                # Get folder path (for genre/category filtering)
                folder = os.path.dirname(rel_path)
                if not folder:
                    folder = "Root"

                # Try to detect series (e.g., "Show Name S01E01")
                import re
                series_match = re.search(r'[Ss](\d{1,2})[Ee](\d{1,2})', filename)

                videos.append({
                    'id': get_file_id(rel_path),
                    'path': rel_path,
                    'name': name,
                    'filename': filename,
                    'size': size,
                    'folder': folder,
                    'is_series': bool(series_match),
                    'season': int(series_match.group(1)) if series_match else None,
                    'episode': int(series_match.group(2)) if series_match else None
                })

    # Sort by folder first, then by name
    videos.sort(key=lambda x: (x['folder'].lower(), x['name'].lower()))
    return videos


def get_all_folders() -> list[str]:
    """Get all unique folders containing video files."""
    folders = set()
    for root, dirs, files in os.walk(MEDIA_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, MEDIA_DIR)
                folder = os.path.dirname(rel_path)
                folders.add(folder if folder else "Root")
    return sorted(folders)


def format_size(size: int) -> str:
    """Format file size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def create_catalog_response(search: Optional[str] = None, genre: Optional[str] = None) -> dict:
    """Create catalog response with all videos."""
    videos = scan_media_files()

    # Apply genre (folder) filter if provided
    if genre:
        videos = [v for v in videos if v['folder'] == genre]

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        videos = [v for v in videos if search_lower in v['name'].lower()]

    metas = []
    for video in videos:
        meta = {
            "id": video['id'],
            "type": "movie",
            "name": video['name'],
            "poster": "",  # We don't have posters
            "description": f"File: {video['filename']}\nSize: {format_size(video['size'])}",
            "genres": [video['folder']],  # Use folder as genre for filtering
        }

        # Add folder info to description
        if video['folder'] != "Root":
            meta["description"] = f"Folder: {video['folder']}\n" + meta["description"]

        # Add series info if detected
        if video['is_series']:
            meta["type"] = "series"
            meta["description"] += f"\nSeason {video['season']}, Episode {video['episode']}"

        metas.append(meta)

    return {"metas": metas}


def create_meta_response(file_id: str) -> Optional[dict]:
    """Create detailed metadata response for a video."""
    filepath = get_filepath_from_id(file_id)
    if not filepath:
        return None

    info = get_video_info(filepath)
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]

    meta = {
        "id": file_id,
        "type": "movie",
        "name": name,
        "description": f"File: {filename}",
    }

    if info:
        # Add duration
        duration = float(info.get('format', {}).get('duration', 0))
        if duration > 0:
            meta["runtime"] = format_duration(duration)
            meta["description"] += f"\nDuration: {format_duration(duration)}"

        # Add video info
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                codec = stream.get('codec_name', 'unknown')
                meta["description"] += f"\nVideo: {width}x{height} ({codec})"
                break

        # Add audio info
        audio_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'audio']
        if audio_streams:
            audio_info = []
            for i, a in enumerate(audio_streams):
                lang = a.get('tags', {}).get('language', f'Track {i+1}')
                codec = a.get('codec_name', 'unknown')
                audio_info.append(f"{lang} ({codec})")
            meta["description"] += f"\nAudio: {', '.join(audio_info)}"

    return {"meta": meta}


def create_stream_response(file_id: str, base_url: str) -> Optional[dict]:
    """Create stream response with HLS URLs."""
    filepath = get_filepath_from_id(file_id)
    if not filepath:
        return None

    # Encode filepath for URL
    encoded_path = '/'.join(quote(part, safe='') for part in filepath.split('/'))

    # Get video info for quality options
    info = get_video_info(filepath)

    streams = []
    subtitles = []
    video_height = 0
    audio_info = ""

    if info:
        # Get subtitle tracks
        subtitle_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'subtitle']
        for i, sub in enumerate(subtitle_streams):
            lang = sub.get('tags', {}).get('language', 'und')
            title = sub.get('tags', {}).get('title', '')

            # Build subtitle entry - id must be globally unique
            subtitles.append({
                "id": f"{file_id}-sub-{i}",
                "url": f"{base_url}/transcode/{encoded_path}/subs_{i}.vtt",
                "lang": lang,
            })

        # Get audio tracks info for title
        audio_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'audio']
        if len(audio_streams) > 1:
            # Get unique languages
            audio_langs = []
            for aud in audio_streams:
                lang = aud.get('tags', {}).get('language', 'und')
                if lang not in audio_langs:
                    audio_langs.append(lang)
            audio_info = f" | Audio: {'/'.join(audio_langs)}"
        elif len(audio_streams) == 1:
            lang = audio_streams[0].get('tags', {}).get('language', 'und')
            audio_info = f" | Audio: {lang}"

        # Get video height
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_height = stream.get('height', 0)
                break

    # Determine quality options based on source resolution
    qualities = []
    if video_height >= 1080:
        qualities = [('original', f'Original ({video_height}p)'), ('1080p', '1080p'), ('720p', '720p'), ('480p', '480p'), ('360p', '360p')]
    elif video_height >= 720:
        qualities = [('original', f'Original ({video_height}p)'), ('720p', '720p'), ('480p', '480p'), ('360p', '360p')]
    elif video_height >= 480:
        qualities = [('original', f'Original ({video_height}p)'), ('480p', '480p'), ('360p', '360p')]
    else:
        qualities = [('original', f'Original ({video_height}p)'), ('360p', '360p')]

    # Create quality-based streams using quality-specific master playlists
    # This ensures audio/subtitle selection works for all quality levels
    for quality_id, quality_title in qualities:
        # Use master.m3u8 for 'original', master_{quality}.m3u8 for specific qualities
        if quality_id == 'original':
            url = f"{base_url}/transcode/{encoded_path}/master.m3u8"
        else:
            url = f"{base_url}/transcode/{encoded_path}/master_{quality_id}.m3u8"

        stream_entry = {
            "url": url,
            "title": f"{quality_title}{audio_info}",
            "name": "SegmentPlayer",
            "behaviorHints": {
                "notWebReady": False
            }
        }
        if subtitles:
            stream_entry["subtitles"] = subtitles
        streams.append(stream_entry)

    return {"streams": streams}


class StremioHandler:
    """Handler for Stremio addon requests."""

    def __init__(self, base_url: str = ""):
        self.base_url = base_url or BASE_URL

    def handle_manifest(self) -> tuple[bytes, str]:
        """Return addon manifest."""
        return json.dumps(MANIFEST).encode(), 'application/json'

    def handle_catalog(self, catalog_type: str, catalog_id: str, extra: dict = None) -> tuple[bytes, str]:
        """Return catalog of videos."""
        search = extra.get('search') if extra else None
        genre = extra.get('genre') if extra else None
        response = create_catalog_response(search, genre)
        return json.dumps(response).encode(), 'application/json'

    def handle_meta(self, meta_type: str, meta_id: str) -> tuple[Optional[bytes], str]:
        """Return video metadata."""
        response = create_meta_response(meta_id)
        if response:
            return json.dumps(response).encode(), 'application/json'
        return None, 'application/json'

    def handle_stream(self, stream_type: str, stream_id: str, base_url: str = None) -> tuple[Optional[bytes], str]:
        """Return stream URLs."""
        # Use provided base_url or fall back to configured one
        url = base_url or self.base_url
        response = create_stream_response(stream_id, url)
        if response:
            return json.dumps(response).encode(), 'application/json'
        return None, 'application/json'
