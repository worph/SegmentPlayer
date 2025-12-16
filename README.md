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
2. **One-ahead prefetch**: After serving segment N, immediately start transcoding N+1
3. **No idle time**: While user watches segment N, segment N+1 is being prepared

This gives:
- Minimum latency for initial playback
- No buffering during normal playback (prefetch fills the gap)
- Simpler architecture, fewer race conditions

### FFmpeg Optimizations

- `-preset ultrafast`: Prioritize speed over compression efficiency
- `-threads 0`: Use all available CPU cores for single segment
- `-tune zerolatency`: Reduce encoding latency
- Hardware acceleration: NVENC/VAAPI/QSV can be enabled if GPU available

## Benchmarking Transcode Performance

The transcoder exposes a **transcode ratio** metric that measures encoding efficiency:

```
Transcode Ratio = (time to generate segment / segment duration) × 100%
```

- **< 25%**: Excellent - segment generates 4x faster than realtime
- **25-75%**: Good - comfortable margin for smooth playback
- **75-100%**: Warning - barely keeping up, may buffer on seek
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
