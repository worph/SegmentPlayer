/* Codec Detection - probe files, detect browser capabilities, choose playback tier */

// Vendor script loader with dedup
var _vendorLoaded = {};
function loadVendor(name, url) {
    if (_vendorLoaded[name]) return _vendorLoaded[name];
    _vendorLoaded[name] = new Promise(function(resolve, reject) {
        var s = document.createElement("script");
        s.src = url;
        s.onload = resolve;
        s.onerror = function() {
            delete _vendorLoaded[name];
            reject(new Error("Failed to load vendor: " + name));
        };
        document.head.appendChild(s);
    });
    return _vendorLoaded[name];
}

// Fetch probe data from server (uses getProbeData from utils.js)
// This is a wrapper that also stores in SP.state.probeCache
async function probeFile(filePath) {
    if (SP.state.probeCache[filePath]) {
        return SP.state.probeCache[filePath];
    }
    var data = await getProbeData(filePath);
    if (data) {
        SP.state.probeCache[filePath] = data;
    }
    return data;
}

// Map ffprobe video codec to MSE codec string
function mapVideoCodecToMSE(codec, profile, bitDepth, width, height) {
    if (!codec) return null;
    switch (codec.toLowerCase()) {
        case "h264": case "avc":
            // Build proper avc1 string based on profile and resolution
            var profileHex = "42E0"; // Baseline
            if (profile === "High" || profile === "High 10") profileHex = "6400";
            else if (profile === "Main") profileHex = "4D40";
            else if (profile === "Baseline" || profile === "Constrained Baseline") profileHex = "42E0";
            // Level based on height (standard video resolutions)
            var level = "1F"; // 3.1 default
            var h = height || 0;
            if (h <= 240) level = "15";       // 2.1
            else if (h <= 480) level = "1E";  // 3.0
            else if (h <= 720) level = "1F";  // 3.1
            else if (h <= 1080) level = "28"; // 4.0
            else if (h <= 1440) level = "32"; // 5.0
            else level = "33";                // 5.1 (4K)
            return "avc1." + profileHex + level;
        case "hevc": case "h265":
            // 'hvc1' (parameter sets in sample description box) instead of 'hev1'
            // (inline parameter sets per fragment). Chrome's D3D11 HEVC path is
            // substantially faster with hvc1 — our muxer also writes hvc1 sample
            // entries, so this must match for MediaSource.isTypeSupported.
            return bitDepth >= 10 ? "hvc1.2.4.L153.B0" : "hvc1.1.6.L153.B0";
        case "vp9":
            return bitDepth >= 10 ? "vp09.02.10.10" : "vp09.00.10.08";
        case "av1":
            return bitDepth >= 10 ? "av01.0.12M.10" : "av01.0.12M.08";
        case "vp8":
            return "vp08";
        default:
            return null;
    }
}

// Map ffprobe audio codec to MSE codec string
function mapAudioCodecToMSE(codec) {
    if (!codec) return null;
    switch (codec.toLowerCase()) {
        case "aac":
            return "mp4a.40.2";
        case "opus":
            return "opus";
        case "flac":
            return "flac";
        case "ac3":
            return "ac-3";
        case "eac3": case "e-ac-3":
            return "ec-3";
        case "mp3":
            return "mp4a.69";
        case "vorbis":
            return "vorbis";
        default:
            return null;
    }
}

// Build a full MSE MIME type string for a track
// type: "video" or "audio"
// trackInfo: probe data for the track (video object or audio[] entry)
function buildMSEMimeType(type, trackInfo) {
    var codecStr;
    if (type === "video") {
        codecStr = mapVideoCodecToMSE(trackInfo.codec, trackInfo.profile, trackInfo.bit_depth,
            trackInfo.width, trackInfo.height);
    } else {
        codecStr = mapAudioCodecToMSE(trackInfo.codec);
    }
    if (!codecStr) return null;
    var container = type === "video" ? "video/mp4" : "audio/mp4";
    return container + '; codecs="' + codecStr + '"';
}

// Check if file can play via Tier 1 (nginx-vod-module HLS: H.264+AAC in MP4/TS)
function isTier1Compatible(probeData) {
    if (!probeData || !probeData.video) return false;

    var validContainers = ["mov,mp4,m4a,3gp,3g2,mj2", "mpegts", "mp4"];
    var containerMatch = validContainers.some(function(c) {
        return probeData.container.indexOf(c) !== -1;
    });
    if (!containerMatch) return false;

    if (probeData.video.codec !== "h264") return false;

    // All audio tracks must be AAC
    if (probeData.audio.length === 0) return false;
    return probeData.audio.every(function(a) {
        return a.codec === "aac";
    });
}

// Check if MediaSource supports a given MIME type
function canMSEPlay(mimeType) {
    if (!mimeType) return false;
    if (typeof MediaSource === "undefined") return false;
    return MediaSource.isTypeSupported(mimeType);
}

// Audio codecs that WebCodecs can typically decode (for re-encode to Opus)
var WEBCODECS_AUDIO_CODECS = ["ac3", "eac3", "e-ac-3", "dts", "flac", "opus", "aac", "mp3", "vorbis"];

// Check if WebCodecs can decode an audio codec (for re-encoding path)
function canWebCodecsDecodeAudio(codec) {
    if (typeof AudioDecoder === "undefined") return false;
    return WEBCODECS_AUDIO_CODECS.indexOf(codec.toLowerCase()) !== -1;
}

// Check if MSE supports any MP4 output at all (needed for client-side transmux)
function canMSEHandleMP4() {
    if (typeof MediaSource === "undefined") return false;
    // Test basic MP4 support with common codecs
    return MediaSource.isTypeSupported('video/mp4; codecs="avc1.42E01E"') ||
           MediaSource.isTypeSupported('video/mp4; codecs="avc1.640028"') ||
           MediaSource.isTypeSupported('video/mp4; codecs="hev1.1.6.L93.B0"') ||
           MediaSource.isTypeSupported('video/mp4; codecs="av01.0.08M.08"');
}

// Check if the browser can play the file natively via <video> element (no demuxing needed)
function canPlayDirect(probeData) {
    var mimeMap = { mp4: "video/mp4", webm: "video/webm", m4v: "video/x-m4v", ogg: "video/ogg" };
    var mime = mimeMap[probeData.container];
    if (!mime) return false;
    var videoCodecStr = mapVideoCodecToMSE(probeData.video.codec, probeData.video.profile,
        probeData.video.bit_depth, probeData.video.width, probeData.video.height);
    if (!videoCodecStr) return false;
    var testMime = mime + '; codecs="' + videoCodecStr + '"';
    var v = document.createElement("video");
    return v.canPlayType(testMime) === "probably";
}

// Choose the best playback tier for a probed file
// Returns: "direct" | "client" | "hls" | "transcode"
function chooseTier(probeData) {
    if (!probeData || !probeData.video) return "transcode";

    // Direct play: browser can handle the container+codec natively (e.g. MP4 H.264)
    // This is the fastest path — no demuxing, no MSE, no transcoding.
    if (canPlayDirect(probeData)) {
        console.log("[Tier] Direct (native playback:", probeData.container, probeData.video.codec + ")");
        return "direct";
    }

    // Tier 2 (client-side transmux) is preferred when available — it handles
    // any container/codec combo the browser can decode via MSE, without relying
    // on nginx-vod-module or HLS.js transmuxing which can be unreliable.
    if (canMSEHandleMP4()) {
        var videoMime = buildMSEMimeType("video", probeData.video);
        if (canMSEPlay(videoMime)) {
            var hasPlayableAudio = probeData.audio.length === 0 || probeData.audio.some(function(a) {
                var audioMime = buildMSEMimeType("audio", a);
                if (canMSEPlay(audioMime)) return true;
                if (canWebCodecsDecodeAudio(a.codec)) return true;
                return false;
            });

            if (hasPlayableAudio) {
                console.log("[Tier] Client-side (video:", probeData.video.codec + ", audio:",
                    probeData.audio.map(function(a) { return a.codec; }).join("/") + ")");
                return "client";
            }
        }
    }

    // Tier 1: native HLS via nginx-vod-module (fallback for browsers without MSE MP4)
    if (isTier1Compatible(probeData)) {
        console.log("[Tier] HLS (H.264+AAC, compatible container)");
        return "hls";
    }

    console.log("[Tier] Transcode (video:", probeData.video.codec + ")");
    return "transcode";
}
