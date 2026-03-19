# SegmentPlayer

On-the-fly HLS streaming server with live segment-level transcoding.

## Features

- **Direct HLS streaming** - Uses nginx-vod-module for on-the-fly HLS packaging (no pre-transcoding)
- **Live segment transcoding** - FFmpeg transcodes incompatible codecs (HEVC, VP9, AV1, etc.) at the segment level
- **Adaptive quality** - Automatically adjusts encoding preset/CRF based on CPU performance
- **Multi-audio track support** - Switch between audio tracks (e.g., English/Japanese dubs)
- **Embedded subtitle extraction** - Extracts embedded subtitles to WebVTT format
- **Modern web player** - Responsive interface with hls.js, quality selection, real-time metrics
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
| `PREFETCH_SEGMENTS` | Number of segments to prefetch ahead | `4` |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Web player interface |
| `/api/files/` | JSON listing of media files |
| `/hls/{file}/master.m3u8` | Direct HLS streaming (nginx-vod-module) |
| `/transcode/{file}/master.m3u8` | Live transcoded HLS stream |
| `/direct/{file}` | Direct file download with range request support |
| `/transcode/metrics` | Transcoding performance statistics |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SegmentPlayer                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      nginx (port 80)                        │  │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────────────────┐  │  │
│  │  │Web Player │  │VOD Module │  │   Proxy to Transcoder  │  │  │
│  │  │   (/)     │  │  (/hls/)  │  │ (/transcode, /direct)  │  │  │
│  │  └───────────┘  └───────────┘  └───────────┬────────────┘  │  │
│  └────────────────────────────────────────────┼───────────────┘  │
│                                               │                   │
│  ┌────────────────────────────────────────────▼───────────────┐  │
│  │               Python Transcoder (port 8080)                 │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  │  │
│  │  │ Playlist  │  │  Segment  │  │ Subtitle  │  │ Direct  │  │  │
│  │  │ Generator │  │ Transcoder│  │ Extractor │  │  File   │  │  │
│  │  └───────────┘  └─────┬─────┘  └───────────┘  └─────────┘  │  │
│  └───────────────────────┼────────────────────────────────────┘  │
│                          │                                        │
│                   ┌──────▼──────┐                                │
│                   │   FFmpeg    │                                │
│                   └─────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
         │                                       │
         ▼                                       ▼
    /data/media                             /data/cache
    (video files)                       (transcoded segments)
```

## How It Works

1. **Direct Streaming**: For compatible videos (H.264+AAC), nginx-vod-module packages the file into HLS segments on-the-fly without any transcoding.

2. **Live Transcoding**: For incompatible codecs, the Python transcoder:
   - Generates HLS playlists dynamically from file metadata (ffprobe)
   - Transcodes each 4-second segment on-demand when requested
   - Caches transcoded segments for subsequent playback
   - Extracts embedded subtitles to WebVTT format

3. **Audio Track Selection**: Multiple audio tracks are exposed as separate HLS variant streams, allowing seamless switching in the player.

## Transcoding Performance Design

The transcoder is designed for **maximum single-segment performance** rather than parallel transcoding. Here's the reasoning:

### Why Not Parallel Transcoding?

Parallel segment transcoding (e.g., 4 concurrent FFmpeg processes) splits CPU resources:
- Each segment gets only 25% of CPU → **slower per-segment time**
- User waits longer for the first segment to be ready
- For live playback, latency matters more than throughput

### Sequential + Smart Prefetch Approach

Instead, we use:
1. **Single-threaded transcoding**: One FFmpeg process gets 100% CPU → fastest possible segment time
2. **Multi-ahead prefetch**: After serving segment N, queue segments N+1 through N+4 for transcoding
3. **No idle time**: While user watches current segment, upcoming segments are being prepared

This gives:
- Minimum latency for initial playback
- No buffering during normal playback (prefetch fills the gap)
- Better seek performance (more segments pre-cached)
- Simpler architecture, fewer race conditions

### FFmpeg Optimizations

- **Adaptive preset**: Automatically adjusts x264 preset based on transcode ratio (target 60-80%)
  - Starts at `fast`, can range from `ultrafast` to `medium`
  - If ratio > 80%: uses faster preset for speed
  - If ratio < 60%: uses slower preset for better quality
- **Adaptive CRF**: Quality offset adjusts alongside preset (0 to +7)
- `-threads 0`: Use all available CPU cores for single segment
- `-tune zerolatency`: Reduce encoding latency

## Benchmarking Transcode Performance

The transcoder exposes a **transcode ratio** metric that measures encoding efficiency:

```
Transcode Ratio = (time to generate segment / segment duration) × 100%
```

- **< 60%**: Excellent - CPU has headroom, quality can increase
- **60-80%**: Target - optimal balance of speed and quality
- **80-100%**: Warning - quality may decrease to maintain playback
- **> 100%**: Critical - cannot keep up with playback, will buffer

### Running Benchmarks

1. Start the container with test media
2. Clear cache and reset metrics:
   ```bash
   curl -s http://localhost:8080/transcode/reset-metrics
   docker exec segment-player sh -c 'rm -rf /data/cache/*'
   ```
3. Request 5 segments sequentially:
   ```bash
   for i in 0 1 2 3 4; do
     curl -s "http://localhost:8080/transcode/YOUR_FILE.mkv/seg_a0_original_0000${i}.ts" -o /dev/null
   done
   ```
4. Check metrics:
   ```bash
   curl -s http://localhost:8080/transcode/metrics | python3 -c "
   import sys,json
   d=json.load(sys.stdin)
   print(f'Avg: {d[\"transcode_ratio_avg\"]}% | Min: {d[\"transcode_ratio_min\"]}% | Max: {d[\"transcode_ratio_max\"]}%')
   "
   ```

### Reference Benchmarks

Tested on 16-core CPU with 4-second segments using FFmpeg 7.0.2 (libdav1d 1.4.x):

| Source Format | Avg Ratio | Notes |
|---------------|-----------|-------|
| AV1 1080p | ~20-25% | Excellent - modern libdav1d |
| HEVC/x265 10-bit 1080p | ~25% | Excellent - fast decode |

### Optimization History

| Optimization | Result |
|--------------|--------|
| Higher CRF (28 vs 23) | No improvement - decode is bottleneck |
| Remove `-tune zerolatency` | Regression - keep it |
| 8s segments (vs 4s) | No improvement on ratio |
| Copy audio (`-c:a copy`) | Marginal - kept AAC for reliability |
| **Upgrade FFmpeg 4.3→7.0** | **4x faster AV1 decode!** |

**Key Finding**: The biggest improvement came from upgrading FFmpeg/libdav1d. The old libdav1d 0.8.x was the bottleneck for AV1 content (85-100% ratio). FFmpeg 7.0.2 with libdav1d 1.4.x achieves ~20-25% ratio.

## Development

A `docker-compose.yml` is included for local development. It mounts `./www` as a read-only volume so you can edit frontend files (HTML, CSS, JS) and refresh the browser without rebuilding the image.

```bash
# Clone and start
git clone https://github.com/Worph/SegmentPlayer.git
cd SegmentPlayer

# Download test media (see media/README.md for details)
curl -L -o media/BigBuckBunny.mp4 "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
curl -L -o media/Sintel.mkv "https://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.720p.mkv"

# Build and start (defaults to ./media for video files)
docker compose up --build

# Or point to your own media library
MEDIA_DIR=/path/to/your/videos docker compose up --build

# Open http://localhost:8080 in your browser
```

Frontend changes (files in `www/`) take effect on browser refresh. Backend changes (`transcoder/server.py`, `nginx/nginx.conf`) require a rebuild:

```bash
docker compose up --build
```

### Container Management

```bash
# Stop
docker compose down

# View logs
docker compose logs -f

# Rebuild and restart
docker compose up --build -d

# Clear transcoded segment cache
docker compose exec segmentplayer rm -rf /data/cache/*
```

## Building from Source

```bash
# Build the Docker image
docker build -t segment-player .

# Run directly with docker run
docker run -d \
  --name segment-player \
  -p 8080:80 \
  -v /path/to/videos:/data/media:ro \
  -v segment-cache:/data/cache \
  segment-player
```

## License

MIT License
