/* Audio Re-encoder - decode unsupported audio codecs and re-encode to Opus via WebCodecs */

function AudioReencoder() {
    this.decoder = null;
    this.encoder = null;
    this.outputChunks = [];
    this.ready = false;
    this.inputCodec = null;
    this.outputSampleRate = 48000;
    this.outputChannels = 2;
    this._flushResolve = null;
}

// Map ffprobe codec names to WebCodecs codec strings for AudioDecoder
function mapAudioCodecToWebCodecs(codec) {
    switch (codec.toLowerCase()) {
        case "ac3": return "ac-3";
        case "eac3": case "e-ac-3": return "ec-3";
        case "aac": return "mp4a.40.2";
        case "mp3": return "mp3";
        case "opus": return "opus";
        case "flac": return "flac";
        case "vorbis": return "vorbis";
        case "dts":
            // DTS is not universally supported by WebCodecs
            // Will need polyfill fallback
            return "dts";
        default: return null;
    }
}

// Check if an audio codec needs re-encoding for MSE playback
function audioNeedsReencode(codec) {
    var mseCodec = mapAudioCodecToMSE(codec);
    if (!mseCodec) return true;
    var mimeType = 'audio/mp4; codecs="' + mseCodec + '"';
    return !MediaSource.isTypeSupported(mimeType);
}

AudioReencoder.prototype.init = async function(inputCodec, sampleRate, channels) {
    this.inputCodec = inputCodec;

    // Ensure WebCodecs is available (load polyfill for Firefox if needed)
    if (typeof AudioDecoder === "undefined") {
        await loadVendor("webcodecs-polyfill", "vendor/libavjs-webcodecs-polyfill.js");
        if (typeof AudioDecoder === "undefined") {
            throw new Error("WebCodecs AudioDecoder not available");
        }
    }

    var self = this;

    // Initialize decoder for source codec
    var decoderCodec = mapAudioCodecToWebCodecs(inputCodec);
    if (!decoderCodec) {
        throw new Error("Unsupported audio codec for WebCodecs: " + inputCodec);
    }

    var decoderConfig = {
        codec: decoderCodec,
        sampleRate: sampleRate || 48000,
        numberOfChannels: channels || 2
    };

    var support = await AudioDecoder.isConfigSupported(decoderConfig);
    if (!support.supported) {
        throw new Error("WebCodecs cannot decode " + inputCodec + " (codec: " + decoderCodec + ")");
    }

    this.decoder = new AudioDecoder({
        output: function(frame) {
            self._onDecodedFrame(frame);
        },
        error: function(e) {
            console.error("[AudioReencoder] Decode error:", e);
        }
    });

    this.decoder.configure(decoderConfig);

    // Initialize Opus encoder
    // Downmix to stereo for compatibility
    this.outputChannels = Math.min(channels || 2, 2);
    this.outputSampleRate = 48000; // Opus works best at 48kHz

    this.encoder = new AudioEncoder({
        output: function(chunk, metadata) {
            self._onEncodedChunk(chunk, metadata);
        },
        error: function(e) {
            console.error("[AudioReencoder] Encode error:", e);
        }
    });

    this.encoder.configure({
        codec: "opus",
        sampleRate: this.outputSampleRate,
        numberOfChannels: this.outputChannels,
        bitrate: 128000
    });

    this.ready = true;
    console.log("[AudioReencoder] Initialized:", inputCodec, "→ Opus",
        this.outputChannels + "ch", this.outputSampleRate + "Hz");
};

AudioReencoder.prototype._onDecodedFrame = function(frame) {
    // Channel downmix (if needed) is handled internally by WebCodecs AudioEncoder
    this.encoder.encode(frame);
    frame.close();
};

AudioReencoder.prototype._onEncodedChunk = function(chunk, metadata) {
    // Convert EncodedAudioChunk to a packet-like object for the muxer
    var data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);

    this.outputChunks.push({
        data: data,
        pts: chunk.timestamp,
        dts: chunk.timestamp,
        duration: chunk.duration || 0,
        flags: (chunk.type === "key") ? 1 : 0
    });
};

// Process a batch of raw audio packets from the demuxer
// packets: array of Packet objects from libav.js ff_read_frame_multi
// Returns: array of re-encoded packet-like objects ready for muxing
AudioReencoder.prototype.processPackets = async function(packets) {
    if (!this.ready || !packets || packets.length === 0) return [];

    this.outputChunks = [];

    // Feed each packet to the decoder
    for (var i = 0; i < packets.length; i++) {
        var pkt = packets[i];
        try {
            var chunk = new EncodedAudioChunk({
                type: (pkt.flags & 1) ? "key" : "delta",
                timestamp: pkt.pts || 0,
                duration: pkt.duration || 0,
                data: pkt.data
            });
            this.decoder.decode(chunk);
        } catch (e) {
            console.warn("[AudioReencoder] Failed to decode packet:", e);
        }
    }

    // Flush decoder to get all decoded frames pushed to encoder
    try {
        await this.decoder.flush();
        await this.encoder.flush();
    } catch (e) {
        // Decoder/encoder may have been closed by an error callback
    }

    var result = this.outputChunks;
    this.outputChunks = [];
    return result;
};

AudioReencoder.prototype.destroy = function() {
    if (this.decoder) {
        try { this.decoder.close(); } catch (e) {}
        this.decoder = null;
    }
    if (this.encoder) {
        try { this.encoder.close(); } catch (e) {}
        this.encoder = null;
    }
    this.ready = false;
    this.outputChunks = [];
};
