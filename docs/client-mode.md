# Client-tier playback (`client` mode)

This document describes the **client** playback tier in depth — the most
non-obvious part of SegmentPlayer. For the big-picture "four tiers" overview and
where this fits, see `CLAUDE.md`. This file is the reference when you need to
touch the in-browser demux / re-mux / re-encode pipeline.

## Why this tier exists

The server can only do two cheap things well: serve raw bytes (`/direct/`) and
package H.264+AAC into HLS (`/hls/`, nginx-vod-module). Everything else —
HEVC/AV1 video, AC3/DTS/EAC3/TrueHD audio, odd containers like MKV — would
otherwise force the expensive **server transcode** tier.

The `client` tier avoids that: the browser fetches the raw file over HTTP range
requests and does **all demux + re-mux + audio re-encode itself**, then hands a
fragmented-MP4 stream to a native `<video>` element via MediaSource Extensions
(MSE). Video is almost always **passthrough** (the browser hardware-decodes
HEVC/AV1/VP9 directly); only the audio is re-encoded. The server stays at ~0%
CPU — it just streams bytes.

## When it is chosen

`chooseTier()` in `www/js/codec-detect.js` returns `"client"` when **all** hold:

- The browser is **not** able to play the file natively (`direct` tier lost).
- MSE can handle MP4 output (`canMSEHandleMP4()` — tests avc1/hev1/av01).
- MSE can play **Opus-in-MP4** (`clientTierSupportsOpusOutput()` tests
  `audio/mp4; codecs="opus"`). **This is why Safari never gets the client tier**
  — it has no Opus-in-MP4 MSE support and falls through to `hls`/`transcode`.
- MSE supports the file's **video** codec (`buildMSEMimeType("video", …)` →
  `MediaSource.isTypeSupported`).
- Every audio track is decodable, or there is no audio
  (`canWebCodecsDecodeAudio()` over `WEBCODECS_AUDIO_CODECS`).

The user can also force it via the **Mode** dropdown (`localStorage.sp_playback_mode`).

> **Key invariant:** the client tier **always** re-encodes audio to Opus,
> regardless of the source codec (yes, even Opus→Opus). That is why the muxer's
> audio half is unconditionally `opus` and why Opus-in-MP4 support is a hard
> gate above. Do not "optimize" this into a conditional passthrough without also
> teaching the muxer and `chooseTier()` about a second audio codec.

## Modules

| File | Class / global | Role |
|------|----------------|------|
| `www/js/codec-detect.js` | (functions) | Tier selection + MSE/WebCodecs capability probes + ffprobe→MSE codec-string mapping |
| `www/js/client-player.js` | `ClientPlayer(videoEl)` | Orchestrator: MSE SourceBuffer, the pump loop, seek/restart, self-healing watchdog, `onFatal` → transcode fallback |
| `www/js/client-demuxer.js` | `ClientDemuxer(url, size)` | libav.js wrapper: opens `/direct/` via range reads, yields packets per stream |
| `www/js/client-muxer.js` | `ClientMuxer(libav)` | libav.js wrapper: re-muxes packets into fragmented MP4; synthesizes DTS for B-frame streams |
| `www/js/audio-reencode.js` | `AudioReencoder()` | Decodes any audio codec → re-encodes to Opus via WebCodecs (native or libav polyfill) |
| `www/js/client-subtitles.js` | `ClientSubtitleCollector` | Extracts embedded SRT/ASS → WebVTT cues in-browser |

Vendor dependencies:

| File | Role |
|------|------|
| `www/vendor/libav-6.8.8.0-sp-audio.wasm.{js,wasm}` + `libav-sp-audio.js` | **Custom** libav.js build with AC3/EAC3/DTS/TrueHD/ALAC decoders. See `docs/libav-custom-build.md`. Do **not** replace with npm libav.js. |
| `www/vendor/libavjs-webcodecs-polyfill.js` | WebCodecs polyfill backed by libav, used when native `AudioDecoder` rejects a codec. Carries **three** local patches (below). |

## Data flow

```
/direct/{path}                     ← HTTP range reads (libav onblockread, 32KB; demuxer caches 5MB read-ahead)
   │
   ▼  ClientDemuxer.readPackets()   (libav ff_read_frame_multi)
   ├── video packets ───────────────────────────────────────────┐
   ├── audio packets → AudioReencoder.submitPackets() (decode→Opus encode)
   │                   AudioReencoder.drainPackets() → Opus pkts ─┤
   └── subtitle packets → ClientSubtitleCollector.collect()       │
                                                                  ▼
                                       ClientMuxer.mux({video, opus-audio})
                                          (interleave by DTS → fragmented MP4)
                                                                  │
                                                                  ▼
                                       SourceBuffer.appendBuffer(fMP4)
                                                                  │
                                                                  ▼
                                       native <video> (hardware decode, browser A/V sync)
```

A single `MediaSource` holds one video SourceBuffer and one audio SourceBuffer
on the same timeline, so the **browser** guarantees A/V sync — there is no hidden
`<audio>` sibling in this tier (that mechanism belongs to a different, older code
path). `appendBuffer` is async; the pump waits on `updateend` between appends.

## The pump loop, backpressure, and staleness

`ClientPlayer` runs an async **pump**: read a batch → re-encode audio → mux →
append → repeat. Backpressure and lifetime are managed by:

- **Buffer-ahead throttle** — if buffered-ahead > `CLIENT_BUFFER_AHEAD` (30s),
  the pump sleeps ~1s instead of reading more.
- **Buffer-behind eviction** — data older than `CLIENT_BUFFER_BEHIND` (30s)
  behind `currentTime` is removed from the SourceBuffer to bound memory.
- **Two staleness guards, do not confuse them:**
  - `SP.state.loadToken` (global, in `config.js`) — bumped on every
    `playFileSetup()`. `playFileClient()` in `player.js` captures the token and
    bails if the user has since loaded another file. Any new async step in the
    client load path must respect this.
  - `ClientPlayer._seekGeneration` (per-player, internal) — bumped on every
    seek/restart. Each async stage of a restart (demuxer abort, SourceBuffer
    clear, first-packet peek, muxer re-init, pump restart) captures the
    generation and bails if a newer seek superseded it. This is what keeps rapid
    scrubbing from interleaving two pipelines.

## Seeking

On a `<video>` `seeking` event (or a watchdog-driven recovery):

1. Stop the pump, abort in-flight range reads, clear the SourceBuffer.
2. `ClientDemuxer.seek()` to the keyframe at or before the target
   (`AVSEEK_FLAG_BACKWARD`).
3. **Peek** the first batch to find the minimum PTS across video+audio and set
   the SourceBuffer `timestampOffset` accordingly; cache that batch as the
   pump's primer so it is not re-read.
4. Re-init the muxer with fresh codecpars and **reset the audio re-encoder**
   (Opus is a stateful codec — failing to reset corrupts audio after a seek).

## Self-healing watchdog and transcode fallback

A 1s watchdog detects two failure modes and re-seeks to the current position to
recover:

- **Pump dead + buffer low** — the pump exited and buffer-ahead dropped below
  `CLIENT_PUMP_DEAD_REFILL_AHEAD` (2s).
- **Playback wedged** — `currentTime` not advancing while buffer-ahead < ~0.5s
  for `CLIENT_STALL_RECOVER_MS` (12s).

Recoveries are rate-limited: more than `CLIENT_RECOVER_MAX` (4) within
`CLIENT_RECOVER_WINDOW_MS` (30s) is treated as unrecoverable and fires
`ClientPlayer.onFatal(reason)`. In `player.js`, `onFatal` is wired to a
**fallback that switches the file to the server `transcode` tier** — so a file
the client pipeline can't keep up with degrades gracefully to FFmpeg.

EOF is guarded too: an EOF more than `CLIENT_EOF_GUARD_SEC` (10s) from the
probed duration is treated as a spurious short read (common on flaky networks),
and the watchdog re-seeks rather than calling `endOfStream()`.

## Audio re-encode: native WebCodecs vs. libav polyfill

`AudioReencoder.init()` picks a decoder per the source codec:

1. **Native first** — try `AudioDecoder.isConfigSupported()` for the mapped
   WebCodecs string (`ac-3`, `ec-3`, `mp4a.40.2`, `mp3`, `opus`, `flac`, …).
2. **Polyfill fallback** — if native rejects (common: AC3/EAC3/DTS on
   Chromium/Linux), lazily load `libavjs-webcodecs-polyfill.js` pointed at the
   **already-loaded** sp-audio libav instance (no second WASM download) and use
   its `AudioDecoder`.

The **encoder** is always Opus (48 kHz). Decoded frames are resampled/downmixed
as needed, then encoded; output timestamps are rescaled to 1/48000 ticks before
being handed to the muxer.

`POLYFILL_VERSION` in `audio-reencode.js` (currently `"sp2"`) is appended as a
`?v=` cache-buster on the polyfill URL — bump it whenever you re-apply the
patches below so browsers don't serve a stale polyfill.

### The three polyfill patches (`grep "SegmentPlayer patch"`)

The polyfill is **not** stock — if it is ever refreshed from upstream, all three
must be re-applied and `POLYFILL_VERSION` bumped:

1. **Expose extra audio decoders** (~line 627) — adds AC3/EAC3/DTS(dca)/TrueHD/
   AAC/MP3/MP2/ALAC to the polyfill's supported-codecs list so they advertise as
   decodable when native WebCodecs lacks them.
2. **Codec-string → avcodec-name mapping** (~line 692) — maps WebCodecs strings
   to libav decoder names (`ec-3`→`eac3`, `ac-3`→`ac3`, `dts`→`dca`,
   `mp4a`→`aac`, …) so the polyfill opens the right FFmpeg codec.
3. **Planar sample-format labelling** (~line 1321) — upstream labels planar
   libav formats (e.g. `AV_SAMPLE_FMT_FLTP`) with the *interleaved* name (`f32`)
   while the data is stored as concatenated planes, so a consumer doing
   `copyTo({format:"f32-planar"})` reads every other sample and garbles audio.
   The patch labels planar formats correctly (`f32-planar`, `s16-planar`, …).
   Without it, polyfill-decoded audio is corrupted.

## Muxer: DTS synthesis for B-frame streams (`client-muxer.js`)

MKV stores packets in **presentation order** with no DTS, but fragmented MP4
requires monotonically-increasing **decode** timestamps. `ClientMuxer` therefore
synthesizes DTS:

- A **warmup phase** measures the B-frame reorder depth (how far PTS values are
  displaced from sorted order), freezing once it sees enough keyframes / packets
  (`WARMUP_MIN/MAX_PACKETS`, `REORDER_DEPTH_CAP`).
- In steady state it keeps a small lookahead queue and emits each packet with
  `DTS = the i-th smallest PTS`, guaranteeing monotonic DTS.
- If the synthesized DTS would violate sanity (too far ahead of PTS, or not
  strictly increasing), it grows the reorder depth; after enough clamp failures
  it falls back to `DTS = PTS` passthrough.
- **AV1** in MKV is already display-order-coded → synthesis is disabled
  (passthrough). **HEVC** sample entries are written as `hvc1` (parameter sets in
  the sample description), which Chrome's D3D11 path decodes much faster than
  `hev1` — and `codec-detect.js` maps HEVC to `hvc1.…` so `isTypeSupported`
  matches what the muxer actually emits.

## Tuning knobs (`config.js`)

| Key | Default | Effect |
|-----|---------|--------|
| `CLIENT_BUFFER_AHEAD` | 30 | Pump sleeps when buffered-ahead exceeds this (s) |
| `CLIENT_BUFFER_BEHIND` | 30 | Evict buffered data older than this behind `currentTime` (s) |
| `CLIENT_STALL_RECOVER_MS` | 12000 | Pump alive but wedged this long → force re-seek |
| `CLIENT_PUMP_DEAD_REFILL_AHEAD` | 2 | Pump exited → recover once buffer-ahead < this (s) |
| `CLIENT_EOF_GUARD_SEC` | 10 | EOF farther than this from duration is treated as spurious |
| `CLIENT_RECOVER_WINDOW_MS` | 30000 | Sliding window for recovery-rate escalation |
| `CLIENT_RECOVER_MAX` | 4 | Recoveries allowed in window before `onFatal` → transcode |
| `CLIENT_MAX_416_RETRIES` | 6 | Consecutive in-range HTTP 416s tolerated before accepting real EOF |

## Gotchas for future edits

- **Audio is always Opus.** See the invariant above.
- **Reset the audio re-encoder on every seek**, or post-seek audio corrupts.
- **Keep the muxer's HEVC `hvc1` tag and `codec-detect.js`'s `hvc1.…` string in
  sync** — a mismatch makes `isTypeSupported` reject a file the muxer can
  actually produce.
- **All three polyfill patches are load-bearing**, including the easy-to-miss
  planar-format one. Re-apply and bump `POLYFILL_VERSION` after any refresh.
- Respect **both** `loadToken` (file switch) and `_seekGeneration` (seek) in any
  new async step.
