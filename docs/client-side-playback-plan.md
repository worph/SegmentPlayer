# Client-Side Playback — Implementation Plan

## Problem

Server-side transcoding (FFmpeg on NAS) is unusable on low-power hardware: either the quality/resolution is terrible or the loading time is unacceptable. The current architecture has only two modes:

1. **nginx-vod-module HLS** — fast but only works with H.264 + AAC in MP4/TS containers
2. **Server-side FFmpeg transcoding** — handles everything else but requires significant CPU

Most media collections contain MKV files with codecs like HEVC, VP9, AV1, AC3, DTS, FLAC, Opus — all of which modern browsers can decode natively or via WebCodecs, but can't play because the **container** (MKV) isn't supported by `<video>`.

## Solution

Add a **client-side playback tier** that demuxes containers in the browser using WebAssembly (libav.js) and decodes using the browser's hardware-accelerated WebCodecs API. The NAS only serves raw files via `/direct/` — zero transcoding.

## Library Stack

| Library | Role |
|---------|------|
| **libav.js** | WASM port of FFmpeg libraries. Demux MKV/AVI/MOV containers, extract packets |
| **libavjs-webcodecs-bridge** | Convert between libav.js packets and WebCodecs `EncodedVideoChunk`/`EncodedAudioChunk` |
| **libavjs-webcodecs-polyfill** | Polyfill WebCodecs for browsers without native support (Firefox). Falls back to WASM decode |
| **TransAVormer** | High-level orchestrator. Declarative "input → output" transforms using libav.js + WebCodecs |
| **AVGuesser** | Dynamic loader for modular libav.js WASM variants (avoids shipping a giant universal build) |

## Architecture: Three-Tier Playback

```
playFile(path)
    │
    ├─ Tier 1: nginx-vod-module HLS
    │   URL: /hls/{path}/master.m3u8
    │   When: H.264 + AAC in MP4/TS
    │   How: HLS.js → <video> (current behavior, unchanged)
    │
    ├─ Tier 2: Client-side demux + decode  ← NEW
    │   URL: /direct/{path} (range requests)
    │   When: Browser can decode the codecs but can't open the container
    │         (HEVC/VP9/AV1 in MKV, AC3/DTS/FLAC audio, etc.)
    │   How: libav.js demux → WebCodecs decode → MediaSource re-mux → <video>
    │
    └─ Tier 3: Server-side FFmpeg transcoding (last resort)
        URL: /transcode/{path}/master.m3u8
        When: Client can't decode the codecs either
        How: Current behavior, unchanged
```

### Tier Selection Logic

The client needs to know the file's codecs **before** choosing a tier. Currently this info is only available after a failed HLS attempt or after the server probes. We need a **lightweight probe endpoint**.

---

## Phase 1: Server-Side Probe Endpoint

### 1.1 Add `/probe/{filepath}` endpoint to `server.py`

Returns ffprobe metadata as JSON. This is cheap — ffprobe reads only the file header, no transcoding.

**Request:**
```
GET /probe/Movies/example.mkv
```

**Response:**
```json
{
  "container": "matroska",
  "duration": 7200.5,
  "file_size": 4500000000,
  "video": {
    "codec": "hevc",
    "profile": "Main 10",
    "width": 3840,
    "height": 2160,
    "bit_depth": 10,
    "fps": 23.976,
    "bitrate": 4500000
  },
  "audio": [
    {
      "index": 0,
      "codec": "ac3",
      "channels": 6,
      "sample_rate": 48000,
      "language": "eng",
      "title": "Surround 5.1"
    },
    {
      "index": 1,
      "codec": "aac",
      "channels": 2,
      "sample_rate": 48000,
      "language": "eng",
      "title": "Stereo"
    }
  ],
  "subtitles": [
    {
      "index": 0,
      "codec": "subrip",
      "language": "eng",
      "title": "English"
    },
    {
      "index": 1,
      "codec": "ass",
      "language": "fre",
      "title": "French"
    }
  ]
}
```

**Implementation:**

In `server.py`, add a new handler in `do_GET`:
```python
m = re.match(r'^/probe/(.+)$', path)
if m:
    return self.handle_probe(m.group(1))
```

The `handle_probe()` method calls the existing `get_video_info()` and reshapes the ffprobe JSON into the clean format above. This reuses the existing ffprobe infrastructure — no new dependencies.

**Add caching:** Store probe results in a dict keyed by file hash. Probe data is small and doesn't change for a given file.

### 1.2 Add nginx proxy rule

```nginx
location /probe/ {
    proxy_pass http://127.0.0.1:8080;
}
```

---

## Phase 2: Client-Side Codec Capability Detection

### 2.1 Create `www/js/codec-detect.js`

Detect what the browser can decode natively via WebCodecs and MediaSource.

```javascript
var ClientCodecs = {
    // Cache detection results
    _cache: {},

    // Check if WebCodecs can decode a video codec
    async canDecodeVideo(codec, width, height) {
        var config = buildVideoDecoderConfig(codec, width, height);
        if (!config) return false;
        try {
            var support = await VideoDecoder.isConfigSupported(config);
            return support.supported === true;
        } catch (e) {
            return false;
        }
    },

    // Check if WebCodecs can decode an audio codec
    async canDecodeAudio(codec, sampleRate, channels) {
        var config = buildAudioDecoderConfig(codec, sampleRate, channels);
        if (!config) return false;
        try {
            var support = await AudioDecoder.isConfigSupported(config);
            return support.supported === true;
        } catch (e) {
            return false;
        }
    },

    // Determine best playback tier for a probed file
    async chooseTier(probeData) {
        // Tier 1: native HLS via vod-module
        if (isTier1Compatible(probeData)) return "hls";

        // Tier 2: client-side demux+decode
        if (await isTier2Capable(probeData)) return "client";

        // Tier 3: server transcode
        return "transcode";
    }
};
```

**Tier 1 check** (matches what nginx-vod-module can handle):
```javascript
function isTier1Compatible(probe) {
    var validVideoCodecs = ["h264"];
    var validAudioCodecs = ["aac"];
    var validContainers = ["mov,mp4,m4a,3gp,3g2,mj2", "mpegts"];
    return validContainers.includes(probe.container)
        && validVideoCodecs.includes(probe.video.codec)
        && probe.audio.every(a => validAudioCodecs.includes(a.codec));
}
```

**Tier 2 check** (can the browser decode these codecs client-side):
```javascript
async function isTier2Capable(probe) {
    var videoOk = await ClientCodecs.canDecodeVideo(
        probe.video.codec, probe.video.width, probe.video.height
    );
    // At least one audio track must be decodable
    var audioOk = false;
    for (var a of probe.audio) {
        if (await ClientCodecs.canDecodeAudio(a.codec, a.sample_rate, a.channels)) {
            audioOk = true;
            break;
        }
    }
    return videoOk && audioOk;
}
```

**Codec string mapping** — WebCodecs needs specific codec strings:
```javascript
function mapVideoCodec(probeCodec, profile, bitDepth) {
    switch (probeCodec) {
        case "hevc": case "h265":
            return bitDepth === 10 ? "hev1.2.4.L153.B0" : "hev1.1.6.L153.B0";
        case "h264": case "avc":
            return "avc1.640034";  // High profile L5.2
        case "vp9":
            return bitDepth === 10 ? "vp09.02.10.10" : "vp09.00.10.08";
        case "av1":
            return bitDepth === 10 ? "av01.0.12M.10" : "av01.0.12M.08";
        default:
            return null;
    }
}
```

### 2.2 WebCodecs polyfill loading

For browsers without WebCodecs (Firefox):
```javascript
async function ensureWebCodecs() {
    if (typeof VideoDecoder !== "undefined") return true;
    // Dynamically load polyfill
    await loadScript("vendor/libavjs-webcodecs-polyfill.min.js");
    return typeof VideoDecoder !== "undefined";
}
```

---

## Phase 3: Client-Side Demux Engine

### 3.1 Create `www/js/client-demuxer.js`

Uses libav.js to demux container formats and extract raw packets.

```javascript
class ClientDemuxer {
    constructor(fileUrl) {
        this.fileUrl = fileUrl;   // /direct/{path}
        this.libav = null;
        this.formatCtx = null;
        this.streams = [];
    }

    async init() {
        this.libav = await LibAV.LibAV();
        // Open remote file via range requests (libav.js supports this)
        await this.libav.mkreadaheadfile("input", this.fileUrl);
        [this.formatCtx] = await this.libav.ff_init_demuxer_file("input");
        this.streams = await this.libav.ff_read_frame_all_streams(this.formatCtx);
    }

    // Read packets in a loop, yielding to the decoder
    async *readPackets(streamIndex) {
        // Use ff_read_frame to get individual packets
        // Yield { data, pts, dts, duration, keyframe }
    }

    async seek(timeSeconds) {
        var timeBase = this.streams[streamIndex].time_base;
        var timestamp = timeSeconds / timeBase;
        await this.libav.avformat_seek_file(this.formatCtx, streamIndex, timestamp);
    }

    async destroy() {
        await this.libav.avformat_close_input(this.formatCtx);
        await this.libav.unlink("input");
    }
}
```

**Key design decisions:**

- **Range request access**: libav.js's `mkreadaheadfile` fetches data on-demand via HTTP range requests to `/direct/{path}`. The existing `handle_direct_file` already supports range requests — no server changes needed.
- **Stream selection**: Only demux the selected video + audio streams (not all streams at once) to minimize bandwidth and memory.
- **Packet buffering**: Maintain a sliding window of ~10 seconds of demuxed packets ahead of playback position.

### 3.2 Seeking Strategy

MKV containers have a Cues element (index of keyframes). libav.js reads this during `ff_init_demuxer_file`. Seeking works via:

1. `avformat_seek_file()` → jumps to nearest keyframe before target time
2. Flush decoders
3. Decode from keyframe, discard frames until target PTS
4. Resume playback

**No server-side keyframe index needed** — the container already has this information, and libav.js reads it via range requests (only fetches the relevant bytes of the file header/cues).

---

## Phase 4: Client-Side Decode + Render Pipeline

### 4.1 Architecture Decision: TransAVormer vs Manual Pipeline

**Option A — TransAVormer (recommended for v1):**
TransAVormer provides a declarative API for media transforms. For our use case:

```javascript
var transformer = await TransAVormer.build({
    input: { url: "/direct/" + filepath },
    output: {
        video: { codec: "copy" },   // No re-encode, just decode for display
        audio: { codec: "copy" },
        format: "mp4"               // Re-mux to fragmented MP4
    }
});
```

TransAVormer handles the libav.js ↔ WebCodecs bridge internally. The output is a stream of fMP4 fragments that can be fed into MediaSource Extensions for `<video>` playback.

**Option B — Manual pipeline (if TransAVormer doesn't fit):**
Wire libav.js demuxer → webcodecs-bridge → WebCodecs decoder → renderer manually. More control, more code.

**Recommendation:** Start with TransAVormer. If its abstractions don't support our seeking/audio-switching needs, fall back to manual wiring in specific areas.

### 4.2 Create `www/js/client-player.js` — Rendering Strategy

#### Strategy: Transmux to Fragmented MP4 → MediaSource Extensions → `<video>`

This is the **preferred approach** because it reuses the native `<video>` element with all its built-in controls (play/pause, seek bar, fullscreen, PiP, volume).

```
/direct/{path}  ──range requests──→  libav.js (demux MKV)
                                        │
                                   raw packets
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                         Video packets        Audio packets
                              │                    │
                    WebCodecs decode?     WebCodecs decode?
                    (only if re-encode    (only if re-encode
                     needed)               needed)
                              │                    │
                              └─────────┬──────────┘
                                        │
                              libav.js (mux to fMP4)
                                        │
                                  fMP4 fragments
                                        │
                              MediaSource.appendBuffer()
                                        │
                                    <video> element
```

**Key insight: For most MKV files, we don't need to decode at all.** If the browser supports the codecs natively (HEVC, VP9, AV1 via MediaSource), we just **transmux** — strip the MKV container, re-wrap packets in fragmented MP4, feed to MediaSource. This is extremely fast and lightweight:

- No decode/re-encode cycle
- No canvas rendering needed
- Native `<video>` controls work
- Hardware-accelerated decode happens naturally in the `<video>` element
- A/V sync is handled by the browser's media pipeline

**When is decode needed?** Only for codecs that MediaSource doesn't accept (rare — most browsers accept HEVC, VP9, AV1, AAC, Opus, FLAC via MSE). In that case: decode via WebCodecs, re-encode to a compatible codec, then mux to fMP4.

#### Capability check for transmux-only path

```javascript
async function canTransmux(probeData) {
    // Check if MediaSource accepts this codec as-is
    var videoMime = buildMSEMimeType(probeData.video);
    var audioMime = buildMSEMimeType(probeData.audio[0]);
    return MediaSource.isTypeSupported(videoMime)
        && MediaSource.isTypeSupported(audioMime);
}

function buildMSEMimeType(track) {
    // e.g., 'video/mp4; codecs="hev1.1.6.L153.B0"'
    var codecStr = mapVideoCodec(track.codec, track.profile, track.bit_depth);
    var container = track.type === "video" ? "video/mp4" : "audio/mp4";
    return container + '; codecs="' + codecStr + '"';
}
```

### 4.3 ClientPlayer Class

```javascript
class ClientPlayer {
    constructor(videoElement) {
        this.video = videoElement;
        this.mediaSource = null;
        this.videoBuffer = null;
        this.audioBuffer = null;
        this.demuxer = null;
        this.abortController = null;
    }

    async load(filepath, probeData) {
        this.cleanup();
        this.abortController = new AbortController();

        var fileUrl = "/direct/" + encodeFilePath(filepath);

        // Initialize MediaSource
        this.mediaSource = new MediaSource();
        this.video.src = URL.createObjectURL(this.mediaSource);

        await new Promise(resolve => {
            this.mediaSource.addEventListener("sourceopen", resolve, { once: true });
        });

        // Add source buffers for video and audio
        var videoMime = buildMSEMimeType(probeData.video);
        var audioMime = buildMSEMimeType(probeData.audio[selectedAudioTrack]);
        this.videoBuffer = this.mediaSource.addSourceBuffer(videoMime);
        this.audioBuffer = this.mediaSource.addSourceBuffer(audioMime);

        // Start transmux pipeline
        this.demuxer = new ClientDemuxer(fileUrl);
        await this.demuxer.init();
        this.startTransmuxLoop();
    }

    async startTransmuxLoop() {
        // Continuously:
        // 1. Demux packets from libav.js
        // 2. Mux to fMP4 fragments (via libav.js muxer)
        // 3. Append to MediaSource buffers
        // 4. Manage buffer window (remove old data, stay ahead by ~30s)
    }

    async seek(time) {
        // 1. Abort current demux loop
        // 2. Flush MediaSource buffers
        // 3. Seek demuxer to keyframe before target
        // 4. Resume transmux loop from new position
    }

    switchAudioTrack(trackIndex) {
        // 1. Remove current audio SourceBuffer
        // 2. Add new SourceBuffer with new codec mime
        // 3. Seek demuxer audio stream to current video position
        // 4. Resume
    }

    cleanup() {
        if (this.abortController) this.abortController.abort();
        if (this.demuxer) this.demuxer.destroy();
        if (this.mediaSource && this.mediaSource.readyState === "open") {
            this.mediaSource.endOfStream();
        }
    }
}
```

### 4.4 Buffer Management

```
Time: 0=====[buffered]=====30=====60=====90...
              ↑ current        ↑ fetch ahead
              playback         limit (30s)

- Keep 30 seconds ahead of playback position
- Evict data more than 60 seconds behind playback
- Pause fetching when buffer is full
- Resume when buffer drops below 15 seconds ahead
- On seek: flush buffers, re-fill from new position
```

---

## Phase 5: Integration with Existing Player

### 5.1 Modify `playFile()` flow in `player.js`

```javascript
async function playFile(filePath, fileName) {
    // ... existing setup code (update UI, destroy previous player) ...

    // NEW: Probe the file first
    var probeData = await probeFile(filePath);

    if (!probeData) {
        // Probe failed, fall back to existing behavior
        playViaTier1(filePath, fileName);
        return;
    }

    // NEW: Choose playback tier
    var tier = await ClientCodecs.chooseTier(probeData);

    switch (tier) {
        case "hls":
            playViaTier1(filePath, fileName);
            break;
        case "client":
            playViaClientSide(filePath, fileName, probeData);
            break;
        case "transcode":
            playTranscoded(
                "/transcode/" + encodeFilePath(filePath) + "/master.m3u8",
                fileName, true
            );
            break;
    }
}
```

`playViaTier1()` = current HLS.js code path (extracted from current `playFile`).

### 5.2 `playViaClientSide()` implementation

```javascript
async function playViaClientSide(filePath, fileName, probeData) {
    setStatus("Preparing...", "#4dabf7", true);

    SP.state.isTranscoding = false;
    SP.state.isClientSide = true;  // NEW state flag
    updateQualityDisplay();

    // Initialize client player
    SP.state.clientPlayer = new ClientPlayer(SP.elements.video);
    await SP.state.clientPlayer.load(filePath, probeData);

    // Populate audio track dropdown from probe data
    if (probeData.audio.length > 0) {
        SP.elements.audioSelect.innerHTML = probeData.audio.map(function(track, i) {
            var label = track.title || track.language || "Track " + (i + 1);
            return '<option value="' + i + '">' + label + '</option>';
        }).join("");
        SP.elements.audioSelect.disabled = probeData.audio.length <= 1;
    }

    // Populate subtitle dropdown from probe data
    if (probeData.subtitles.length > 0) {
        SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>' +
            probeData.subtitles.map(function(sub, i) {
                var label = sub.title || sub.language || "Track " + (i + 1);
                return '<option value="' + i + '">' + label + '</option>';
            }).join("");
        SP.elements.subtitleSelect.disabled = false;
    }

    // Quality dropdown: only "Original" in client-side mode (no transcoding = no quality options)
    SP.elements.resolutionSelect.innerHTML = '<option value="original">Original</option>';
    SP.elements.resolutionSelect.disabled = true;

    setStatus("Playing (client-side)", "#51cf66");

    SP.elements.video.play().catch(function() {});
}
```

### 5.3 Update controls for client-side mode

**Audio switching:**
```javascript
// In initAudioControl() change handler:
if (SP.state.isClientSide && SP.state.clientPlayer) {
    SP.state.clientPlayer.switchAudioTrack(parseInt(newVal));
    return;
}
// ... existing HLS.js audio switching code ...
```

**Subtitle handling in client-side mode:**

Two approaches for extracting subtitles:

1. **Text-based subtitles (SRT, ASS, WebVTT):** Extract client-side using libav.js demuxer. Read subtitle packets, convert to VTT, create a Blob URL, attach as `<track>`.

2. **Image-based subtitles (PGS, VobSub):** These require OCR or bitmap rendering. Skip for v1 — show as unavailable.

```javascript
// In subtitle select handler:
if (SP.state.isClientSide && SP.state.clientPlayer) {
    var subIndex = parseInt(SP.elements.subtitleSelect.value);
    await SP.state.clientPlayer.loadSubtitleTrack(subIndex);
    return;
}
```

### 5.4 Update `SP.state` in `config.js`

```javascript
state: {
    // ... existing fields ...
    isClientSide: false,       // True when using client-side demux+decode
    clientPlayer: null,        // ClientPlayer instance
    probeCache: {}             // filepath → probeData cache
}
```

### 5.5 Cleanup on file switch

In the existing cleanup path at the top of `playFile()`:
```javascript
if (SP.state.clientPlayer) {
    SP.state.clientPlayer.cleanup();
    SP.state.clientPlayer = null;
}
SP.state.isClientSide = false;
```

---

## Phase 6: Subtitle Extraction (Client-Side)

### 6.1 Create `www/js/client-subtitles.js`

```javascript
class ClientSubtitleExtractor {
    // Extract text subtitle track from already-open demuxer
    async extract(demuxer, subtitleStreamIndex) {
        var packets = [];
        // Read all subtitle packets for this stream
        // (subtitles are small — reading all at once is fine)

        // Convert to WebVTT format
        var vtt = "WEBVTT\n\n";
        for (var pkt of packets) {
            vtt += formatTimestamp(pkt.pts) + " --> " + formatTimestamp(pkt.pts + pkt.duration) + "\n";
            vtt += decodeSubtitlePacket(pkt.data, codecName) + "\n\n";
        }

        return vtt;
    }

    // Decode subtitle packet data based on codec
    decodeSubtitlePacket(data, codec) {
        switch (codec) {
            case "subrip":  // SRT
                return stripSRTFormatting(textDecoder.decode(data));
            case "ass": case "ssa":
                return parseASSDialogue(textDecoder.decode(data));
            case "webvtt":
                return textDecoder.decode(data);
            default:
                return "";  // Image-based subs not supported
        }
    }
}
```

### 6.2 Attach to `<video>` element

```javascript
async loadSubtitleTrack(subIndex) {
    var vtt = await this.subtitleExtractor.extract(this.demuxer, subIndex);
    var blob = new Blob([vtt], { type: "text/vtt" });
    var url = URL.createObjectURL(blob);

    // Remove existing tracks
    this.video.querySelectorAll("track").forEach(t => t.remove());

    var track = document.createElement("track");
    track.kind = "subtitles";
    track.src = url;
    track.default = true;
    this.video.appendChild(track);
    track.track.mode = "showing";
}
```

---

## Phase 7: Vendor Dependencies & Build

### 7.1 Directory structure

```
www/
├── vendor/
│   ├── libav/
│   │   ├── libav-4.8.6.1-webcodecs.js     # Main libav.js (WebCodecs variant)
│   │   ├── libav-4.8.6.1-webcodecs.wasm   # WASM binary
│   │   └── libav-4.8.6.1-webcodecs.worker.js
│   ├── libavjs-webcodecs-bridge.min.js
│   ├── libavjs-webcodecs-polyfill.min.js
│   ├── transavormer.min.js
│   └── avguesser.min.js
├── js/
│   ├── config.js
│   ├── utils.js
│   ├── ui.js
│   ├── router.js
│   ├── browser.js
│   ├── player.js          # Modified: probe → tier selection
│   ├── controls.js        # Modified: client-side audio/sub switching
│   ├── metrics.js
│   ├── codec-detect.js    # NEW: WebCodecs capability detection
│   ├── client-demuxer.js  # NEW: libav.js demuxer wrapper
│   ├── client-player.js   # NEW: MediaSource + transmux pipeline
│   ├── client-subtitles.js # NEW: Client-side subtitle extraction
│   └── app.js             # Modified: init new modules
└── index.html             # Modified: load new scripts + vendor libs
```

### 7.2 Script loading order in `index.html`

```html
<!-- Vendor: loaded before app scripts -->
<script src="vendor/avguesser.min.js"></script>
<script src="vendor/transavormer.min.js"></script>
<script src="vendor/libavjs-webcodecs-polyfill.min.js"></script>

<!-- App scripts (order matters) -->
<script src="js/config.js"></script>
<script src="js/utils.js"></script>
<script src="js/ui.js"></script>
<script src="js/codec-detect.js"></script>      <!-- NEW: after utils, before player -->
<script src="js/client-demuxer.js"></script>     <!-- NEW -->
<script src="js/client-subtitles.js"></script>   <!-- NEW -->
<script src="js/client-player.js"></script>      <!-- NEW: after demuxer + subtitles -->
<script src="js/router.js"></script>
<script src="js/browser.js"></script>
<script src="js/player.js"></script>
<script src="js/controls.js"></script>
<script src="js/metrics.js"></script>
<script src="js/app.js"></script>
```

### 7.3 WASM variant selection via AVGuesser

libav.js ships many WASM variants (codec combinations). AVGuesser picks the right one at runtime based on what codecs are needed. This avoids loading a 20MB+ universal WASM when only a 2MB variant is needed.

```javascript
// On page load or first client-side play
var variant = await AVGuesser.findBestVariant({
    demux: ["matroska", "avi", "mov"],
    // No decode needed for transmux — but load decode support for fallback
    decode: { video: ["hevc", "vp9", "av1"], audio: ["ac3", "dts", "flac"] }
});
await LibAV.load(variant);
```

### 7.4 Dockerfile changes

Add vendor files to the Docker image:

```dockerfile
# Download vendor libraries during build
RUN mkdir -p /www/vendor/libav && \
    wget -O /www/vendor/libav/libav.js https://unpkg.com/libav.js@4.8.6.1/... && \
    wget -O /www/vendor/transavormer.min.js https://unpkg.com/transavormer/... && \
    wget -O /www/vendor/avguesser.min.js https://unpkg.com/avguesser/...
```

Alternatively, use npm at build time and copy outputs to `www/vendor/`.

---

## Phase 8: Metrics & Status Display

### 8.1 Client-side metrics

When playing in client-side mode, show relevant stats in the metrics panel:

```javascript
// New metrics for client-side mode:
{
    "mode": "client-side",
    "demux_format": "matroska",
    "video_codec": "hevc (passthrough)",
    "audio_codec": "ac3 → decoded",
    "buffer_ahead": 22.5,        // seconds of buffer ahead of playback
    "bytes_downloaded": 450000000,
    "download_speed": 12500000,  // bytes/sec
    "demux_time_avg": 2.3,       // ms per packet
    "frames_decoded": 14400,
    "frames_dropped": 3
}
```

### 8.2 Status display

Update the status dot and text for client-side mode:

| State | Status Text | Color |
|-------|-------------|-------|
| Probing | "Analyzing..." | Blue (pulsing) |
| Loading WASM | "Loading decoder..." | Blue (pulsing) |
| Buffering | "Buffering..." | Yellow (pulsing) |
| Playing | "Client-side" | Green |
| Seeking | "Seeking..." | Blue (pulsing) |
| Error | "Decode error" | Red |
| Fallback | "Falling back to server..." | Yellow |

---

## Phase 9: Error Handling & Fallbacks

### 9.1 Graceful degradation chain

```
Client-side play attempt
    │
    ├─ WebCodecs not available? → load polyfill → retry
    │
    ├─ WASM load fails? → fall back to Tier 3 (server transcode)
    │
    ├─ Demux error? → fall back to Tier 3
    │
    ├─ Decode error (unsupported profile/level)?
    │   ├─ Try polyfill software decode
    │   └─ If still fails → fall back to Tier 3
    │
    ├─ MediaSource rejected codec?
    │   ├─ Try decode + re-encode path
    │   └─ If still fails → fall back to Tier 3
    │
    └─ Playback stall (buffer underrun > 10s)?
        └─ Show warning, offer "Switch to server transcoding" button
```

### 9.2 User override

Add a toggle in the UI to force a specific tier:

```html
<div class="control-group">
    <label>Mode:</label>
    <select id="playbackMode">
        <option value="auto">Auto</option>
        <option value="hls">Direct HLS</option>
        <option value="client">Client decode</option>
        <option value="transcode">Server transcode</option>
    </select>
</div>
```

This lets users manually override the auto-detection if it chooses wrong.

---

## Phase 10: MKV-Specific Considerations

MKV is the primary target format. Key things to handle:

### 10.1 Container features

| MKV Feature | Handling |
|-------------|----------|
| Multiple video tracks | Show first video track only (rare to have multiple) |
| Multiple audio tracks | Full support via audio dropdown + `switchAudioTrack()` |
| Embedded subtitles (SRT/ASS) | Extract client-side via libav.js |
| Embedded subtitles (PGS/VobSub) | Show as "unavailable" in v1 (image-based, needs OCR) |
| Chapters | Parse from container, show in seek bar (future enhancement) |
| Attachments (fonts) | Needed for ASS styling. Extract and load as `@font-face` (stretch goal) |
| Ordered chapters / segment linking | Not supported (extremely rare, complex) |
| Variable frame rate | WebCodecs handles this via per-frame timestamps |
| HDR metadata (HEVC/AV1) | Pass through — browser handles HDR if display supports it |

### 10.2 Common MKV codec combinations

| Combination | Tier | Client-side approach |
|-------------|------|---------------------|
| HEVC + AAC | 2 | Transmux only (no decode needed) |
| HEVC + AC3/EAC3 | 2 | Transmux video, decode+re-encode audio to AAC if MSE rejects AC3 |
| HEVC + DTS | 2 | Transmux video, decode+re-encode audio to AAC (DTS not in MSE) |
| HEVC + FLAC | 2 | Transmux video, decode+re-encode audio to AAC or Opus |
| HEVC + Opus | 2 | Transmux both (Opus supported in MSE) |
| VP9 + Opus | 2 | Transmux both (WebM codecs in MP4 — or output WebM) |
| AV1 + Opus | 2 | Transmux both |
| H.264 + AAC (in MKV) | 2 | Transmux to MP4 (could also be Tier 1 if remuxed) |
| H.264 + AC3 | 2 | Transmux video, decode+re-encode audio |
| MPEG-2 + MP2 | 3 | Server transcode (old codecs, no WebCodecs support) |

### 10.3 Audio re-encoding when needed

Some audio codecs (DTS, TrueHD) aren't supported by MediaSource. In these cases:

```
AC3/DTS/FLAC packets → WebCodecs AudioDecoder → PCM frames
    → WebCodecs AudioEncoder (Opus or AAC) → encoded packets
    → libav.js muxer → fMP4 audio fragments
    → MediaSource audio SourceBuffer
```

This is lightweight — audio encoding is trivial compared to video. Opus encoding at 128kbps is real-time even on weak devices.

**Important:** libavjs-webcodecs-polyfill does **not** cover H.264 or AAC encoding (patent-encumbered). For audio re-encode, target **Opus** (royalty-free, excellent quality, wide MSE support) as the default output codec.

---

## Implementation Order

### Sprint 1 — Foundation (Probe + Detection)
1. Add `/probe/` endpoint to `server.py` + nginx route
2. Create `codec-detect.js` with `chooseTier()` logic
3. Modify `playFile()` to call probe and select tier
4. Test: verify probe returns correct data, tier selection matches expectations

### Sprint 2 — Basic MKV Transmux Playback
5. Set up vendor dependencies (libav.js, TransAVormer, AVGuesser)
6. Create `client-demuxer.js` — open MKV via range requests, read packets
7. Create `client-player.js` — transmux to fMP4, feed MediaSource
8. Test: play HEVC+AAC MKV file client-side (transmux only, no decode)

### Sprint 3 — Audio Track Switching + Subtitles
9. Implement `switchAudioTrack()` in ClientPlayer
10. Create `client-subtitles.js` — extract SRT/ASS from MKV client-side
11. Wire up audio and subtitle dropdowns for client-side mode
12. Test: switch between audio tracks, enable/disable subtitles

### Sprint 4 — Audio Re-encoding
13. Add audio decode+re-encode path for AC3/DTS/FLAC
14. Target Opus output codec
15. Test: play HEVC+DTS MKV, verify audio decodes and plays

### Sprint 5 — Seeking + Buffer Management
16. Implement `seek()` in ClientPlayer (flush + re-fill from keyframe)
17. Implement buffer window management (evict old, cap ahead)
18. Test: seek forward/backward in MKV, verify smooth playback

### Sprint 6 — Polish + Fallbacks
19. Add error handling and automatic fallback to Tier 3
20. Add playback mode override dropdown
21. Update metrics panel for client-side stats
22. Update status display for all client-side states
23. WebCodecs polyfill loading for Firefox

### Sprint 7 — Testing + Edge Cases
24. Test matrix: HEVC, VP9, AV1 × AAC, AC3, DTS, Opus, FLAC
25. Test on low-power devices (verify no server CPU usage)
26. Test on multiple browsers (Chrome, Firefox, Safari, Edge)
27. Test large files (>10GB) — verify range requests work
28. Test with 10-bit HDR content
29. Memory profiling — verify no leaks during long playback

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TransAVormer doesn't support seeking well | High | Fall back to manual libav.js + WebCodecs wiring for seek |
| HEVC not in MediaSource on Firefox | Medium | Polyfill handles decode, but no hardware accel. Show warning |
| Large WASM download on first play | Medium | AVGuesser loads minimal variant; cache via Service Worker |
| Memory pressure on 4K content | Medium | Strict buffer limits (30s ahead, evict aggressively) |
| libav.js range request performance | Medium | Use readahead buffer; tune fetch chunk size |
| Audio re-encoding introduces latency | Low | Encode in small chunks; Opus encodes fast even on mobile |
| ASS subtitle rendering (styled) | Low | v1: strip formatting, show plain text. v2: use libass.js |

---

## Out of Scope (Future)

- **Image-based subtitle rendering** (PGS, VobSub) — requires canvas overlay or libass.js
- **Chapter markers** on the seek bar
- **ASS subtitle styling** (fonts, positioning, effects) — needs libass.js WASM
- **Client-side video transcoding** (e.g., HEVC→H.264) — too CPU-heavy for browser
- **Offline playback** / download + local decode
- **Chromecast / AirPlay** from client-side mode (would need server remux)
- **Multi-video-track** support (commentary tracks, etc.)
