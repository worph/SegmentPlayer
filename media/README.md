# Test Media Files

This folder holds local video files for development testing. Its contents (except this README) are gitignored.

## Recommended test files

Download these free, open-licensed clips to cover the main codec/container combinations:

| File | Codecs | Container | Tests |
|------|--------|-----------|-------|
| **Big Buck Bunny** (180p MP4) | H.264 + AAC | MP4 | Direct playback, remux |
| **Sintel** (720p MKV) | H.264 + AAC (or AC3) | MKV | MKV native support, remux fallback |
| **Matroska test5** (576p MKV) | H.264 + 2x AAC | MKV | Multi-audio track switching, 8 subtitle languages |
| **Tears of Steel** (4K VP9/WebM) | VP9 + Opus | WebM | Direct on Chrome/Firefox, transcode on Safari |

### Download commands

```bash
# Big Buck Bunny - MP4 H.264 (direct playback test)
curl -L -o media/BigBuckBunny.mp4 \
  "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"

# Sintel - MKV H.264 (remux/native MKV test)
curl -L -o media/Sintel.mkv \
  "https://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.720p.mkv"

# Matroska test5 - MKV with 2 audio tracks + 8 subtitle tracks (multi-audio test)
curl -L -o media/test5.mkv \
  "https://github.com/ietf-wg-cellar/matroska-test-files/raw/master/test_files/test5.mkv"
```

### What each file tests

- **MP4 + H.264/AAC** — Should play via Direct mode on all browsers (zero server cost)
- **MKV + H.264/AAC** — Direct on Chrome/Firefox (native MKV), Remux on Safari, verifies fallback chain
- **MKV + multi-audio** — Tests audio track switching in both direct and transcode modes (Elephants Dream, CC-BY, from [Matroska test suite](https://github.com/ietf-wg-cellar/matroska-test-files))
- **WebM + VP9** — Direct on Chrome/Firefox, Transcode on Safari/older browsers
- **HEVC / AV1** — Always Transcode mode, tests adaptive quality and segment prefetch
