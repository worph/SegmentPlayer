/* Client Subtitles - extract text subtitles from container via libav.js */

function ClientSubtitleExtractor(demuxer) {
    this.demuxer = demuxer;
}

// Extract a subtitle track and return as WebVTT string
// subStreamIndex: the absolute stream index in the container
// codec: the subtitle codec name (subrip, ass, webvtt, etc.)
ClientSubtitleExtractor.prototype.extract = async function(subStreamIndex, codec) {
    if (!this.demuxer || !this.demuxer.libav) {
        throw new Error("Demuxer not initialized");
    }

    // Image-based subtitles can't be converted to VTT
    var imageSubs = ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"];
    if (imageSubs.indexOf(codec) >= 0) {
        return null;
    }

    SP.log.debug("Subtitles", "Extracting stream", subStreamIndex, "codec:", codec);

    // Save current demuxer position (we'll need to seek back)
    var currentTime = 0;

    // Seek to beginning for subtitle extraction
    await this.demuxer.seek(0);

    // Read all subtitle packets for this stream
    var subtitlePackets = [];
    var eof = false;
    var maxIterations = 50000; // safety limit

    while (!eof && maxIterations-- > 0) {
        var result = await this.demuxer.readPackets(2 * 1024 * 1024); // 2MB batches
        eof = result.eof;

        // Collect subtitle packets for our stream
        if (result.packets[subStreamIndex]) {
            for (var i = 0; i < result.packets[subStreamIndex].length; i++) {
                subtitlePackets.push(result.packets[subStreamIndex][i]);
            }
        }
    }

    SP.log.debug("Subtitles", "Extracted", subtitlePackets.length, "subtitle packets");

    if (subtitlePackets.length === 0) {
        return "WEBVTT\n\n";
    }

    var stream = this.demuxer.streams[subStreamIndex];
    var timeBase = 1;
    if (stream && stream.time_base_num && stream.time_base_den) {
        timeBase = stream.time_base_num / stream.time_base_den;
    }

    return buildVTTFromPackets(subtitlePackets, timeBase, codec);
};

// Matroska ContentCompression (zlib) leaves subtitle packets deflated —
// libav.js strips a few things but doesn't inflate. Detect the zlib header
// (0x78 0x01 / 0x9C / 0xDA) and inflate via the platform DecompressionStream.
function looksZlib(data) {
    return data && data.length >= 2 && data[0] === 0x78 &&
        (data[1] === 0x01 || data[1] === 0x9C || data[1] === 0xDA);
}

async function inflateZlib(data) {
    var ds = new DecompressionStream("deflate");
    var w = ds.writable.getWriter();
    w.write(data);
    w.close();
    var buf = await new Response(ds.readable).arrayBuffer();
    return new Uint8Array(buf);
}

async function decompressPacketData(data) {
    if (looksZlib(data)) {
        try {
            return await inflateZlib(data);
        } catch (e) {
            SP.log.warn("Subtitles", "zlib inflate failed:", e);
        }
    }
    return data;
}

// Build a WebVTT string from an array of subtitle packets.
// Reusable by both the full-file extractor and piggyback collection.
async function buildVTTFromPackets(packets, timeBase, codec) {
    if (!packets || packets.length === 0) return "WEBVTT\n\n";

    // Sort by PTS so packets collected out-of-order (e.g. after seeks) are correct
    var sorted = packets.slice().sort(function(a, b) {
        return (a.pts || 0) - (b.pts || 0);
    });

    // Deduplicate identical packets collected from overlapping reads — but
    // NOT distinct cues that legitimately share the same PTS (ASS screener
    // signs and stacked dialogue lines routinely declare multiple co-starting
    // events). Key by pts + payload length + a cheap byte-rolling hash so
    // genuine duplicates collapse while distinct cues survive.
    function fingerprint(pkt) {
        var d = pkt.data;
        var len = d ? d.byteLength : 0;
        var h = 0;
        if (d) {
            var bytes = (d instanceof Uint8Array) ? d : new Uint8Array(d);
            // Sample up to 32 bytes evenly across the payload — bounded cost,
            // separates animated sign lines whose first bytes are similar.
            var step = Math.max(1, Math.floor(len / 32));
            for (var k = 0; k < len; k += step) {
                h = ((h << 5) - h + bytes[k]) | 0;
            }
        }
        return pkt.pts + ":" + len + ":" + h;
    }
    var seen = Object.create(null);
    var deduped = [];
    for (var i = 0; i < sorted.length; i++) {
        var key = fingerprint(sorted[i]);
        if (!seen[key]) {
            seen[key] = true;
            deduped.push(sorted[i]);
        }
    }

    var vtt = "WEBVTT\n\n";
    var decoder = new TextDecoder("utf-8");

    for (var i = 0; i < deduped.length; i++) {
        var pkt = deduped[i];
        var startTime = (pkt.pts || 0) * timeBase;
        var duration = (pkt.duration || 0) * timeBase;
        var endTime = startTime + duration;

        if (duration <= 0 || startTime < 0) continue;

        var data = await decompressPacketData(pkt.data);
        var text = decoder.decode(data);
        var cleanText = decodeSubtitleText(text, codec);

        if (cleanText.trim() === "") continue;

        vtt += formatVTTTimestamp(startTime) + " --> " + formatVTTTimestamp(endTime) + "\n";
        vtt += cleanText + "\n\n";
    }

    return vtt;
}

// Decode subtitle text based on codec format
function decodeSubtitleText(raw, codec) {
    switch (codec) {
        case "subrip":
        case "srt":
            return stripSRTFormatting(raw);
        case "ass":
        case "ssa":
            return parseASSDialogue(raw);
        case "webvtt":
            return raw;
        default:
            // Try as plain text
            return raw;
    }
}

// Strip SRT formatting tags (bold, italic, font, etc.)
function stripSRTFormatting(text) {
    return text
        .replace(/<[^>]+>/g, "")     // HTML tags
        .replace(/\{[^}]+\}/g, "")   // SSA-style override tags
        .trim();
}

// Parse ASS/SSA Dialogue line to plain text
function parseASSDialogue(text) {
    // Full Dialogue line: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    var match = text.match(/^Dialogue:\s*\d+,[^,]*,[^,]*,[^,]*,[^,]*,\d+,\d+,\d+,[^,]*,(.*)$/);
    if (match) {
        text = match[1];
    } else {
        // libav.js packet data: ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        // (no Dialogue: prefix, no timestamps — those are in packet metadata)
        // Skip 8 comma-separated fields to get the actual text
        var commaCount = 0;
        for (var i = 0; i < text.length; i++) {
            if (text[i] === ",") {
                commaCount++;
                if (commaCount === 8) {
                    text = text.substring(i + 1);
                    break;
                }
            }
        }
    }

    return text
        .replace(/\\N/g, "\n")       // ASS newline
        .replace(/\\n/g, "\n")       // ASS soft newline
        .replace(/\{[^}]*\}/g, "")   // Override tags like {\b1}, {\an8}, etc.
        .trim();
}

// Format seconds as VTT timestamp (HH:MM:SS.mmm)
function formatVTTTimestamp(seconds) {
    if (seconds < 0) seconds = 0;
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = Math.floor(seconds % 60);
    var ms = Math.round((seconds % 1) * 1000);

    return pad2(h) + ":" + pad2(m) + ":" + pad2(s) + "." + pad3(ms);
}

function pad2(n) {
    return n < 10 ? "0" + n : "" + n;
}

function pad3(n) {
    if (n < 10) return "00" + n;
    if (n < 100) return "0" + n;
    return "" + n;
}

// Create a Blob URL for a VTT string and attach it to the video element.
//
// Removing a <track> element does NOT remove its TextTrack from
// videoElement.textTracks in Chromium — entries accumulate for the video's
// lifetime. The client-mode subtitle collector calls this every 5 packet
// batches, so we must (a) disable every existing TextTrack before adding the
// new one, and (b) enable the *new* track by reference rather than assuming
// it lands at textTracks[0]. Otherwise stale tracks stay in "showing" mode
// and the user sees overlapping/duplicated cues.
function attachVTTToVideo(videoElement, vttString, label) {
    videoElement.querySelectorAll("track").forEach(function(t) { t.remove(); });
    for (var i = 0; i < videoElement.textTracks.length; i++) {
        videoElement.textTracks[i].mode = "disabled";
    }

    var blob = new Blob([vttString], { type: "text/vtt" });
    var url = URL.createObjectURL(blob);

    var track = document.createElement("track");
    track.kind = "subtitles";
    track.src = url;
    track.label = label || "Subtitles";
    track.default = true;
    videoElement.appendChild(track);

    track.addEventListener("load", function() {
        if (track.track) track.track.mode = "showing";
    });

    return track;
}
