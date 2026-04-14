# Building the Custom `sp-audio` libav.js Variant

The client-side pipeline uses a custom libav.js variant — **`sp-audio`** — vendored at
`www/vendor/libav-6.8.8.0-sp-audio.wasm.{js,wasm}` and `www/vendor/libav-sp-audio.js`.
This document explains what the variant contains, why it exists, and how to rebuild
it from source.

## Why a custom variant

The stock `webcodecs` variant that ships on npm is deliberately minimal — it
assumes the browser's native WebCodecs handles all audio decode. That
assumption breaks on some Chromium builds (notably Linux desktop and some
headless contexts) where `AudioDecoder.isConfigSupported()` rejects `ec-3`
(E-AC3), `ac-3`, or `dts`.

When native WebCodecs rejects a codec, `www/js/audio-reencode.js` falls back
to the libav-backed `LibAVWebCodecs.AudioDecoder` from
`vendor/libavjs-webcodecs-polyfill.js`. The polyfill uses whichever LibAV
wrapper we pass it — so if our vendored libav.js WASM has the needed audio
decoders compiled in, they become available via the polyfill. That's exactly
what `sp-audio` provides.

No published libav.js variant (as of 6.8.8) bundles AC3/EAC3/DTS/TrueHD
decoders together with the mp4/matroska demuxers and libopus encoder we
already depend on. Hence: one Docker build against upstream source.

## What `sp-audio` contains

Starting from the stock `webcodecs-avf` variant (mp4/mkv/ogg/webm/flac/wav
demuxers + libopus encoder + common video parsers + avfilter/swresample/
swscale), `sp-audio` adds:

- `decoder-ac3`, `parser-ac3`
- `decoder-eac3`
- `decoder-dca` (DTS Coherent Acoustics)
- `decoder-truehd`
- `decoder-aac`
- `decoder-mp3`, `decoder-mp2`, `parser-mpegaudio`
- `decoder-alac`

The resulting WASM is **~3.75 MB** (versus ~2.25 MB for stock `webcodecs`).
The libopus encoder and all existing demuxers are preserved — the muxer,
Opus codecpar factory, and `fixAudioParams` path all continue to work
unchanged.

## Prerequisites

- Docker (tested with 29.3.0 on WSL2). The build container needs bind-mount
  access to the working directory — on WSL2 the host path must be under
  `/mnt/<drive>/…` (i.e. the project directory itself works; `/tmp` on the
  Linux side does *not*).
- `xz-utils` on the host to extract the upstream libav.js source tarball.
- `curl` and `tar`.
- Roughly **4 GB** of free disk space during the build.
- Roughly **25-30 minutes** of wall-clock time for a full cold build.

## Build steps

### 1. Stage the libav.js 6.8.8 sources

```bash
mkdir -p /tmp/libavjs && cd /tmp/libavjs
curl -sL https://registry.npmjs.org/libav.js/-/libav.js-6.8.8.tgz -o libav.tgz
tar -xzf libav.tgz
apt-get install -y xz-utils     # one-time; only if missing
mkdir -p /tmp/libavjs/package/sources/libavjs-src
tar -xJf /tmp/libavjs/package/sources/libav.js.tar.xz \
    -C /tmp/libavjs/package/sources/libavjs-src
```

The npm tarball ships both the source of the libav.js wrapper itself
(`sources/libav.js.tar.xz`) and every third-party dependency archive
(ffmpeg, opus, libaom, libvorbis, libvpx, lame, libogg, zlib,
emfiberthreads). Copy all of these into the working tree.

### 2. Stage under a Docker-visible path

On WSL2, Docker only sees bind-mounts that resolve to the Windows-backed
filesystem. Work under the project directory:

```bash
BUILD=/d/workspace/sandbox/SegmentPlayer/.build-tmp
mkdir -p "$BUILD"
cp -r /tmp/libavjs/package/sources/libavjs-src/. "$BUILD/"
mkdir -p "$BUILD/sources"
cp /tmp/libavjs/package/sources/*.tar.{gz,xz} "$BUILD/sources/"
```

### 3. Generate the `sp-audio` variant config

```bash
cd "$BUILD/configs"
node mkconfig.js sp-audio '[
  "avformat","avcodec",
  "format-ogg","format-webm","format-mp4",
  "parser-opus","codec-libopus",
  "format-flac","parser-flac","codec-flac",
  "format-wav","codec-pcm_f32le",
  "parser-aac","parser-vp8","parser-vp9","parser-av1","parser-h264","parser-hevc",
  "bsf-extract_extradata","bsf-vp9_metadata","bsf-av1_metadata",
  "bsf-h264_metadata","bsf-hevc_metadata",
  "avfilter","swresample","swscale","audio-filters","video-filters",
  "parser-ac3","decoder-ac3","decoder-eac3",
  "decoder-aac","decoder-mp3","decoder-mp2","parser-mpegaudio",
  "decoder-dca","decoder-alac","decoder-truehd"
]'
```

(Pass the JSON array as a single argument on one line; the multi-line
form above is for readability.) Verify the generated flags:

```bash
grep -E '^--enable-(decoder|parser)' \
  "$BUILD/configs/configs/sp-audio/ffmpeg-config.txt" | sort -u
```

### 4. Build the emsdk+pkg-config image

The upstream project ships `Dockerfile.development`, which is just
`emscripten/emsdk` plus `pkg-config`:

```bash
cd "$BUILD"
docker build -t libavbuild:emsdk -f Dockerfile.development .
```

### 5. Run the variant build

```bash
cd "$BUILD"
docker run --rm \
  -v "$(pwd)":/build -w /build \
  libavbuild:emsdk \
  bash -c 'npm install --silent && make build-sp-audio'
```

The build emits:

- `dist/libav-6.8.8.0-sp-audio.wasm.js` (loader, ~280 KB)
- `dist/libav-6.8.8.0-sp-audio.wasm.wasm` (WASM binary, ~3.75 MB)
- `dist/libav-sp-audio.js` (unversioned entry wrapper, ~23 KB)
- …plus asm.js, threaded, and debug variants we don't ship.

The build takes ~25 minutes on a modern laptop. Most of it is FFmpeg
configure (~3 min) and `libavcodec` compilation (~15 min).

### 6. Install into `www/vendor/`

```bash
cd /d/workspace/sandbox/SegmentPlayer
cp .build-tmp/dist/libav-6.8.8.0-sp-audio.wasm.js   www/vendor/
cp .build-tmp/dist/libav-6.8.8.0-sp-audio.wasm.wasm www/vendor/
cp .build-tmp/dist/libav-sp-audio.js                www/vendor/
```

`www/js/client-demuxer.js` already points at
`vendor/libav-sp-audio.js` via its `loadVendor()` call.

### 7. Verify at runtime

Open the player, hard-reload the page (bump `?v=N` in `www/index.html` if
needed to defeat browser caching), and in DevTools run:

```js
const libav = await LibAV.LibAV();
const names = ["ac3","eac3","dca","aac","opus","flac","mp3","mp2","truehd","alac","libopus"];
const dec = {};
for (const n of names) dec[n] = (await libav.avcodec_find_decoder_by_name(n)) > 0;
const encLibopus = (await libav.avcodec_find_encoder_by_name("libopus")) > 0;
libav.terminate();
({ variant: LibAV.CONFIG, version: LibAV.VER, decoders: dec, encoder_libopus: encLibopus })
```

Expected:

```
variant: "sp-audio"
version: "6.8.8.0"
decoders: { ac3: true, eac3: true, dca: true, aac: true, ... }
encoder_libopus: true
```

## The polyfill patch

Upstream `libavjs-webcodecs-polyfill.js` only recognizes the codec strings
it was shipped with (FLAC/Opus/Vorbis for audio). To route `ec-3`, `ac-3`,
`dts`, `mp4a`, `mp3`, `truehd`, `alac`, `mp2` to the libav decoders we
added, `www/vendor/libavjs-webcodecs-polyfill.js` carries an in-file patch
around two places, both marked `SegmentPlayer patch:` in comments:

1. `codecs()` — adds the WebCodecs→avcodec name pairs to the scan list so
   `isConfigSupported()` can discover these decoders at polyfill load time.
2. `decoder()` — adds `case` arms that map `"ec-3" → "eac3"`,
   `"ac-3" → "ac3"`, `"dts" → "dca"`, `"mp4a" → "aac"`, etc.

If the vendor polyfill is ever refreshed from upstream, re-apply those two
patches. `audio-reencode.js` also defines `POLYFILL_VERSION` — bump it to
force browsers to re-fetch the patched polyfill (browser caches the
vendor URL on first load).

## Regenerating after a libav.js upgrade

When a new libav.js release is worth adopting:

1. Update the `curl` URL in step 1 to the new version.
2. Update `LIBAVJS_VERSION` references in filenames
   (`libav-<version>-sp-audio.wasm.*`) if the version bumped the
   LIBAVJS_VERSION_BASE.
3. Repeat steps 2–6.
4. Update `www/js/client-demuxer.js` if the loader filename changed.
5. Smoke-test with the decoder probe in step 7, then the full verification
   pass (play AAC, switch to EAC3, seek, confirm drift <5 ms).

The `sp-audio` variant config does not need to change unless upstream
adds new fragments we'd want to include.

## Troubleshooting

- **`No rule to make target 'build-sp-audio'`** — the config at
  `configs/configs/sp-audio/` is missing. Re-run `mkconfig.js` (step 3).
- **Docker container sees empty directory / missing configs** — your
  working tree is on a path Docker Desktop can't bind-mount. On WSL2
  this means `/tmp` or other pure-Linux paths; move under
  `/mnt/<drive>/…` (equivalently, the project directory).
- **`Unable to locate package pkg-config`** — run `apt-get update` before
  `apt-get install -y pkg-config` inside the container. The
  `libavbuild:emsdk` image built in step 4 already includes it.
- **Configure completes but compile fails for a specific codec** — some
  FFmpeg decoders require parsers or bsfs we didn't enable. Add the
  relevant `parser-<name>` or `bsf-<name>` fragments to the config and
  re-run step 3.
