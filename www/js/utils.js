/* Utility functions */

function formatSize(bytes) {
    if (bytes === 0) return "0 B";
    var k = 1024;
    var sizes = ["B", "KB", "MB", "GB"];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function isVideo(name) {
    var lower = name.toLowerCase();
    return SP.config.videoExtensions.some(function(ext) {
        return lower.endsWith(ext);
    });
}

function isSubtitle(name) {
    var lower = name.toLowerCase();
    return SP.config.subtitleExtensions.some(function(ext) {
        return lower.endsWith(ext);
    });
}

function getIcon(item) {
    if (item.type === "directory") return "📁";
    if (isVideo(item.name)) return "🎬";
    if (isSubtitle(item.name)) return "📝";
    return "📄";
}

function encodeFilePath(path) {
    return path.split('/').map(function(part) {
        return encodeURIComponent(part);
    }).join('/');
}

function extractLang(subName, baseName) {
    var withoutBase = subName.substring(baseName.length);
    var match = withoutBase.match(/[._]([a-z]{2,3})/i);
    return match ? match[1].toUpperCase() : "Unknown";
}

async function getProbeData(filePath) {
    try {
        var response = await fetch("/api/probe/" + encodeFilePath(filePath));
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        return null;
    }
}

function detectPlaybackMode(probeData) {
    if (!probeData) return "transcode";

    var container = probeData.container;
    var vcodec = probeData.video_codec;
    var acodec = probeData.audio_codec;
    var testVideo = document.createElement('video');

    // Build MIME strings to test native playback
    var mimeTests = [];

    if (vcodec === 'h264' || vcodec === 'avc1') {
        var audioCodecStr = (acodec === 'aac') ? ',mp4a.40.2' : '';
        if (container === 'mp4' || container === 'm4v' || container === 'mov') {
            mimeTests.push('video/mp4; codecs="avc1.64001f' + audioCodecStr + '"');
        }
        if (container === 'mkv') {
            mimeTests.push('video/x-matroska; codecs="avc1.64001f' + audioCodecStr + '"');
        }
        if (container === 'ts' || container === 'm2ts') {
            mimeTests.push('video/mp2t; codecs="avc1.64001f' + audioCodecStr + '"');
        }
    } else if ((vcodec === 'vp9' || vcodec === 'vp8') && container === 'webm') {
        var webmAudio = (acodec === 'opus') ? ',opus' : (acodec === 'vorbis') ? ',vorbis' : '';
        mimeTests.push('video/webm; codecs="' + vcodec + webmAudio + '"');
    } else if (vcodec === 'hevc' || vcodec === 'h265') {
        if (container === 'mp4' || container === 'm4v' || container === 'mov') {
            mimeTests.push('video/mp4; codecs="hev1.1.6.L93.B0"');
        }
    } else if (vcodec === 'av1') {
        if (container === 'mp4') {
            mimeTests.push('video/mp4; codecs="av01.0.08M.08"');
        }
        if (container === 'webm') {
            mimeTests.push('video/webm; codecs="av01.0.08M.08"');
        }
    }

    // Test if browser can play any of the MIME types natively
    for (var i = 0; i < mimeTests.length; i++) {
        var result = testVideo.canPlayType(mimeTests[i]);
        if (result === 'probably' || result === 'maybe') {
            return "direct";
        }
    }

    // Fallback: if H.264+AAC, nginx-vod can remux to HLS
    if ((vcodec === 'h264' || vcodec === 'avc1') && (acodec === 'aac' || acodec === 'mp3')) {
        return "remux";
    }

    return "transcode";
}
