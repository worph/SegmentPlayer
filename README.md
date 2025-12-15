# SegmentPlayer

On-the-fly HLS streaming server with live segment-level transcoding.

## Features

- **Direct HLS streaming** - Uses nginx-vod-module for on-the-fly HLS packaging (no pre-transcoding)
- **Live segment transcoding** - FFmpeg transcodes incompatible codecs (HEVC, VP9, AV1, etc.) at the segment level
- **Multi-audio track support** - Switch between audio tracks (e.g., English/Japanese dubs)
- **Embedded subtitle extraction** - Extracts embedded subtitles to WebVTT format
- **Modern web player** - Responsive interface with hls.js
- **Instant seeking** - No waiting for full transcoding

## Supported Formats

| Format | Direct Streaming | Live Transcoding |
|--------|-----------------|------------------|
| H.264 + AAC | ✅ | ✅ |
| H.265/HEVC | ❌ | ✅ |
| VP9/AV1 | ❌ | ✅ |
| AC3/DTS Audio | ❌ | ✅ |
| 10-bit Video | ❌ | ✅ |

## Quick Start

### Using Docker Compose

```yaml
services:
  segment-player:
    image: ghcr.io/worph/segment-player:latest
    ports:
      - "8080:80"
    volumes:
      - /path/to/your/videos:/data/media:ro
      - segment-cache:/data/cache
    restart: unless-stopped

volumes:
  segment-cache:
```

### Using Docker Run

```bash
docker run -d \
  --name segment-player \
  -p 8080:80 \
  -v /path/to/your/videos:/data/media:ro \
  -v segment-cache:/data/cache \
  ghcr.io/worph/segment-player:latest
```

Then open http://localhost:8080 in your browser.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEDIA_DIR` | Path to media files inside container | `/data/media` |
| `CACHE_DIR` | Path for transcoded segment cache | `/data/cache` |
| `SEGMENT_DURATION` | HLS segment duration in seconds | `4` |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Web player interface |
| `/api/files/` | JSON listing of media files |
| `/hls/{file}/master.m3u8` | Direct HLS streaming (nginx-vod-module) |
| `/transcode/{file}/master.m3u8` | Live transcoded HLS stream |
| `/media/` | Direct file access |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SegmentPlayer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    nginx (port 80)                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Web Player  │  │  VOD Module │  │   Proxy     │  │   │
│  │  │   (/)       │  │   (/hls/)   │  │ (/transcode)│  │   │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘  │   │
│  └───────────────────────────────────────────┼─────────┘   │
│                                              │              │
│  ┌───────────────────────────────────────────▼─────────┐   │
│  │              Python Transcoder (port 8080)          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Playlist   │  │   Segment   │  │  Subtitle   │  │   │
│  │  │  Generator  │  │  Transcoder │  │  Extractor  │  │   │
│  │  └─────────────┘  └──────┬──────┘  └─────────────┘  │   │
│  └──────────────────────────┼──────────────────────────┘   │
│                             │                               │
│                      ┌──────▼──────┐                       │
│                      │   FFmpeg    │                       │
│                      └─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
        │                                      │
        ▼                                      ▼
   /data/media                            /data/cache
   (video files)                      (transcoded segments)
```

## How It Works

1. **Direct Streaming**: For compatible videos (H.264+AAC), nginx-vod-module packages the file into HLS segments on-the-fly without any transcoding.

2. **Live Transcoding**: For incompatible codecs, the Python transcoder:
   - Generates HLS playlists dynamically from file metadata (ffprobe)
   - Transcodes each 4-second segment on-demand when requested
   - Caches transcoded segments for subsequent playback
   - Extracts embedded subtitles to WebVTT format

3. **Audio Track Selection**: Multiple audio tracks are exposed as separate HLS variant streams, allowing seamless switching in the player.

## Building from Source

```bash
# Clone the repository
git clone https://github.com/Worph/SegmentPlayer.git
cd SegmentPlayer

# Build the Docker image
docker build -t segment-player .

# Run locally
docker run -p 8080:80 -v /path/to/videos:/data/media:ro segment-player
```

## License

MIT License
