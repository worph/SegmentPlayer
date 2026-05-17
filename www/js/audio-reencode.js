/* Audio Re-encoder - decode arbitrary audio codecs and re-encode to Opus via WebCodecs.
 *
 * Streaming model: submitPackets() feeds the decoder fire-and-forget;
 * drainPackets() returns whatever the encoder has emitted since the last drain.
 * This decouples audio encode from the video pump — the decoder and encoder
 * run asynchronously via their own WebCodecs queues, and we pick up output
 * whenever it's ready rather than blocking on per-batch flush.
 *
 * Decoder fallback: when the browser's native AudioDecoder rejects a codec
 * (AudioDecoder.isConfigSupported returns false — common for AC3/EAC3/DTS on
 * some Chromium Linux builds), we fall back to the libav.js-backed
 * LibAVWebCodecs.AudioDecoder from vendor/libavjs-webcodecs-polyfill.js. The
 * polyfill is configured to share our already-loaded LibAV wrapper so no
 * second WASM is downloaded — its decoder pool just creates additional
 * instances against the same WASM. Native stays the fast-path (hardware-
 * accelerated where available); polyfill is strictly a fallback.
 *
 * Output timestamps are rescaled from WebCodecs microseconds to the Opus
 * native 1/48000 timebase that the muxer is initialized with.
 */

var OPUS_TB_DEN = 48000; // output timebase denominator (Hz)

// Lazy one-time bootstrap for the polyfill. Loads the vendor JS, then calls
// LibAVWebCodecs.load({LibAV}) pointing it at our existing LibAV wrapper so
// its decoder instances reuse our WASM build. Resolves with the polyfill's
// exported namespace; subsequent calls return the cached promise.
// Bump this when the vendored polyfill is patched so previously-cached copies
// aren't reused. Matches the ?v= suffix convention used in index.html.
var POLYFILL_VERSION = "sp2";

var _polyfillReady = null;
function ensurePolyfillReady() {
    if (_polyfillReady) return _polyfillReady;
    _polyfillReady = (async function() {
        await loadVendor("webcodecs-polyfill",
            "vendor/libavjs-webcodecs-polyfill.js?v=" + POLYFILL_VERSION);
        if (typeof LibAVWebCodecs === "undefined") {
            throw new Error("LibAVWebCodecs polyfill failed to load");
        }
        await LibAVWebCodecs.load({
            LibAV: (typeof LibAV !== "undefined") ? LibAV : undefined
        });
        return LibAVWebCodecs;
    })();
    return _polyfillReady;
}

function AudioReencoder() {
    this.decoder = null;
    this.encoder = null;
    this._outputQueue = [];
    this.ready = false;
    this.inputCodec = null;
    this.inputSampleRate = 48000;
    this.inputChannels = 2;
    this.inputExtradata = null;
    // Source stream timebase (ticks-to-seconds = num/den). Used to convert
    // demuxer packet timestamps (stream-native ticks) into the microseconds
    // that WebCodecs requires on EncodedAudioChunk. Default 1/1000000 (µs).
    this.inputTimeBaseNum = 1;
    this.inputTimeBaseDen = 1000000;
    this.outputSampleRate = 48000;
    this.outputChannels = 2;
    // Decoder/chunk class pair selected at configure-time. Defaults to globals
    // (native WebCodecs); swapped to LibAVWebCodecs.* when native rejects.
    this._AudioDecoderClass = (typeof AudioDecoder !== "undefined") ? AudioDecoder : null;
    this._EncodedAudioChunkClass = (typeof EncodedAudioChunk !== "undefined") ? EncodedAudioChunk : null;
    this._usingPolyfill = false;
    // Bresenham accumulator for frame-count resampling — bounded by srcRate, so
    // long-run output count converges to the exact srcFrames × out/in ratio.
    this._sampleAccum = 0;
    // Timestamp anchor: the first input frame's timestamp seeds this, then all
    // subsequent output AudioData timestamps are derived from a running output-
    // sample count so the encoder sees a self-consistent (sample-count, ts) pair.
    this._anchorTsUs = null;
    this._totalOutSamples = 0;
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
            // DTS is not universally supported by WebCodecs; the libav.js
            // polyfill fallback handles it.
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

AudioReencoder.prototype._buildDecoderConfig = function() {
    var decoderCodec = mapAudioCodecToWebCodecs(this.inputCodec);
    if (!decoderCodec) {
        throw new Error("Unsupported audio codec for WebCodecs: " + this.inputCodec);
    }
    var cfg = {
        codec: decoderCodec,
        sampleRate: this.inputSampleRate || 48000,
        numberOfChannels: this.inputChannels || 2
    };
    // Pass AudioSpecificConfig / codec private data when we have it. WebCodecs
    // trusts `description` over our channel-count guesses, which matters for
    // AAC streams where the container channel field is zero and the real
    // layout lives in the ASC bytes (patched by demuxer.fixAudioParams).
    if (this.inputExtradata && this.inputExtradata.length > 0) {
        cfg.description = this.inputExtradata;
    }
    return cfg;
};

// Pick a decoder implementation that actually supports the current config.
// Tries native first (hardware-accelerated where possible), falls back to the
// libav.js polyfill (CPU decode in WASM) when native rejects the codec.
// Throws if neither supports it.
AudioReencoder.prototype._selectDecoderImpl = async function() {
    var cfg = this._buildDecoderConfig();

    // Native path.
    if (typeof AudioDecoder !== "undefined") {
        try {
            var nativeSupport = await AudioDecoder.isConfigSupported(cfg);
            if (nativeSupport && nativeSupport.supported) {
                this._AudioDecoderClass = AudioDecoder;
                this._EncodedAudioChunkClass = EncodedAudioChunk;
                this._usingPolyfill = false;
                return;
            }
        } catch (e) { /* fall through to polyfill */ }
    }

    // Polyfill path.
    var pf;
    try {
        pf = await ensurePolyfillReady();
    } catch (e) {
        throw new Error("No decoder available for " + this.inputCodec
            + " (native rejected, polyfill load failed: " + e.message + ")");
    }
    var pfSupport = await pf.AudioDecoder.isConfigSupported(cfg);
    if (!pfSupport || !pfSupport.supported) {
        throw new Error("No decoder available for " + this.inputCodec
            + " (native rejected, polyfill also rejected)");
    }
    this._AudioDecoderClass = pf.AudioDecoder;
    this._EncodedAudioChunkClass = pf.EncodedAudioChunk;
    this._usingPolyfill = true;
    SP.log.debug("AudioReencoder", "Using libav polyfill for codec:", this.inputCodec);
};

AudioReencoder.prototype._createDecoder = function() {
    var self = this;
    this.decoder = new this._AudioDecoderClass({
        output: function(frame) { self._onDecodedFrame(frame); },
        error: function(e) { SP.log.error("AudioReencoder", "Decode error:", e); }
    });
    this.decoder.configure(this._buildDecoderConfig());
};

AudioReencoder.prototype._createEncoder = function() {
    var self = this;
    // Encoder stays native — AudioEncoder for Opus is reliably available.
    this.encoder = new AudioEncoder({
        output: function(chunk, metadata) { self._onEncodedChunk(chunk, metadata); },
        error: function(e) { SP.log.error("AudioReencoder", "Encode error:", e); }
    });
    this.encoder.configure({
        codec: "opus",
        sampleRate: this.outputSampleRate,
        numberOfChannels: this.outputChannels,
        bitrate: 128000
    });
};

AudioReencoder.prototype.init = async function(inputCodec, sampleRate, channels, extradata, timeBase) {
    this.inputCodec = inputCodec;
    this.inputSampleRate = sampleRate || 48000;
    this.inputChannels = channels || 2;
    this.inputExtradata = extradata || null;
    if (timeBase && timeBase.den > 0) {
        this.inputTimeBaseNum = timeBase.num;
        this.inputTimeBaseDen = timeBase.den;
    }

    // If native AudioDecoder/AudioEncoder are missing entirely (Firefox until
    // ~2024), load the polyfill globally to provide both. On modern browsers
    // this is a no-op and native WebCodecs is used.
    if (typeof AudioDecoder === "undefined" || typeof AudioEncoder === "undefined") {
        var pf = await ensurePolyfillReady();
        if (typeof AudioDecoder === "undefined") AudioDecoder = pf.AudioDecoder;
        if (typeof EncodedAudioChunk === "undefined") EncodedAudioChunk = pf.EncodedAudioChunk;
        if (typeof AudioEncoder === "undefined") AudioEncoder = pf.AudioEncoder;
        if (typeof AudioData === "undefined") AudioData = pf.AudioData;
    }

    // Select decoder implementation (native fast-path or polyfill fallback).
    await this._selectDecoderImpl();

    // Downmix to stereo for compatibility
    this.outputChannels = Math.min(this.inputChannels, 2);
    this.outputSampleRate = OPUS_TB_DEN; // Opus works best at 48 kHz

    this._createDecoder();
    this._createEncoder();

    this.ready = true;
    SP.log.debug("AudioReencoder", "Initialized:", inputCodec, "->", "Opus",
        this.outputChannels + "ch", this.outputSampleRate + "Hz",
        this._usingPolyfill ? "(via polyfill)" : "(native)");
};

// Resample and/or downmix a decoded AudioData to the encoder's expected
// format using an OfflineAudioContext (which uses the browser's built-in
// high-quality resampler and standard ITU-R BS.775-1 downmix). WebCodecs
// AudioEncoder requires frames to match encoder.configure() exactly — it
// does NOT auto-convert. Common mismatch: source is 44.1 kHz or multichannel.
//
// Polyfill note: when decode runs through libav.js, the resulting AudioData
// is a polyfill class. Native AudioEncoder may reject non-native AudioData
// via internal type checks, so we always route polyfill frames through this
// repackage path (which constructs a fresh native AudioData from the samples).
AudioReencoder.prototype._convertFrame = async function(frame) {
    var srcRate = frame.sampleRate;
    var srcCh = frame.numberOfChannels;
    var srcFrames = frame.numberOfFrames;
    var sameFormat = srcRate === this.outputSampleRate && srcCh === this.outputChannels;
    if (sameFormat && !this._usingPolyfill) {
        return frame;
    }

    // Extract planar float samples from the decoded frame.
    var planes = new Array(srcCh);
    for (var c = 0; c < srcCh; c++) {
        var buf = new Float32Array(srcFrames);
        frame.copyTo(buf, { planeIndex: c, format: "f32-planar" });
        planes[c] = buf;
    }

    // Seed the timestamp anchor from the first input frame so output stays
    // tied to the source media timeline (including post-seek positions).
    if (this._anchorTsUs === null) this._anchorTsUs = frame.timestamp;

    frame.close();

    // Bresenham frame-count selection: maintain an integer accumulator so
    // outFrames alternates between floor and ceil of the true ratio, keeping
    // the long-run average exact. Bounded by srcRate, never grows unbounded.
    this._sampleAccum += srcFrames * this.outputSampleRate;
    var outFrames = Math.floor(this._sampleAccum / srcRate);
    this._sampleAccum -= outFrames * srcRate;
    if (outFrames < 1) outFrames = 1;

    // Render through an OfflineAudioContext at the target rate + channel count.
    // The OAC handles resampling (src buffer's rate -> context rate) and
    // channel-count reduction (connection downmix to destination.channelCount).
    var oac = new OfflineAudioContext(this.outputChannels, outFrames, this.outputSampleRate);
    var srcBuf = oac.createBuffer(srcCh, srcFrames, srcRate);
    for (var ch = 0; ch < srcCh; ch++) srcBuf.copyToChannel(planes[ch], ch);
    var bs = oac.createBufferSource();
    bs.buffer = srcBuf;
    bs.connect(oac.destination);
    bs.start();
    var rendered = await oac.startRendering();

    // Pack rendered buffer into interleaved f32 for AudioData construction.
    var renderedFrames = rendered.length;
    var interleaved = new Float32Array(renderedFrames * this.outputChannels);
    for (var co = 0; co < this.outputChannels; co++) {
        var chData = rendered.getChannelData(co);
        for (var i = 0; i < renderedFrames; i++) {
            interleaved[i * this.outputChannels + co] = chData[i];
        }
    }

    // Derive timestamp from cumulative output samples so the (numberOfFrames,
    // timestamp) pair the encoder sees is internally consistent across the run.
    var newTsUs = this._anchorTsUs + Math.round(
        this._totalOutSamples * 1e6 / this.outputSampleRate);
    this._totalOutSamples += renderedFrames;

    return new AudioData({
        format: "f32",
        sampleRate: this.outputSampleRate,
        numberOfFrames: renderedFrames,
        numberOfChannels: this.outputChannels,
        timestamp: newTsUs,
        data: interleaved
    });
};

AudioReencoder.prototype._onDecodedFrame = async function(frame) {
    try {
        var converted = await this._convertFrame(frame);
        try {
            this.encoder.encode(converted);
        } finally {
            converted.close();
        }
    } catch (e) {
        // Encoder may have been closed during teardown — silently drop.
        try { frame.close(); } catch (_) {}
    }
};

AudioReencoder.prototype._onEncodedChunk = function(chunk, metadata) {
    // Rescale timestamp from WebCodecs microseconds to the muxer's 1/48000
    // timebase (the audio stream's time_base set at ff_init_muxer time).
    var ptsTicks = Math.round((chunk.timestamp || 0) * OPUS_TB_DEN / 1e6);
    var durTicks = Math.round((chunk.duration || 0) * OPUS_TB_DEN / 1e6);

    // Optional drift-tap — populated only when window.__driftTap is enabled.
    // Captures per-chunk timestamp directly from the encoder, bypassing the
    // muxer-lastDts staircase noise that polluted earlier measurements.
    if (typeof window !== "undefined" && window.__driftTap && window.__driftTap.enabled) {
        window.__driftTap.outputs.push({
            ts_us: chunk.timestamp,
            dur_us: chunk.duration,
            bytes: chunk.byteLength,
            wall: performance.now()
        });
    }

    var data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);

    this._outputQueue.push({
        data: data,
        pts: ptsTicks,
        dts: ptsTicks,
        duration: durTicks,
        flags: (chunk.type === "key") ? 1 : 0
    });
};

// Feed a batch of raw audio packets to the decoder. Fire-and-forget — the
// WebCodecs decoder queues internally and emits decoded frames asynchronously
// via _onDecodedFrame; encoded chunks arrive at _onEncodedChunk and accumulate
// in _outputQueue. Use drainPackets() to collect.
//
// Packet timestamps arrive in the source stream's native timebase ticks; we
// convert to microseconds here before handing to WebCodecs.
AudioReencoder.prototype.submitPackets = function(packets) {
    if (!this.ready || !packets || packets.length === 0) return;
    // ticks * tickToUs = microseconds
    var tickToUs = 1e6 * this.inputTimeBaseNum / this.inputTimeBaseDen;
    for (var i = 0; i < packets.length; i++) {
        var pkt = packets[i];
        try {
            var tsUs = Math.round((pkt.pts || 0) * tickToUs);
            var durUs = Math.round((pkt.duration || 0) * tickToUs);
            if (typeof window !== "undefined" && window.__driftTap && window.__driftTap.enabled) {
                window.__driftTap.inputs.push({ ts_us: tsUs, dur_us: durUs, dataLen: (pkt.data && pkt.data.byteLength) || 0 });
            }
            var chunk = new this._EncodedAudioChunkClass({
                type: (pkt.flags & 1) ? "key" : "delta",
                timestamp: tsUs,
                duration: durUs,
                data: pkt.data
            });
            this.decoder.decode(chunk);
        } catch (e) {
            SP.log.warn("AudioReencoder", "Failed to submit packet:", e);
        }
    }
};

// Return and clear the encoded-chunk queue. Non-blocking — returns whatever
// the encoder has emitted so far. Call this once per pump iteration.
AudioReencoder.prototype.drainPackets = function() {
    if (this._outputQueue.length === 0) return [];
    var out = this._outputQueue;
    this._outputQueue = [];
    return out;
};

// Await full decode+encode drain and return remaining packets. Use on EOF to
// recover any still-queued tail frames before writing the muxer trailer.
AudioReencoder.prototype.flush = async function() {
    if (!this.ready) return [];
    try {
        await this.decoder.flush();
        await this.encoder.flush();
    } catch (e) {
        // Decoder/encoder may have been closed by an error callback.
    }
    return this.drainPackets();
};

// Reset decoder and encoder state without changing input/output configuration.
// Used on seek: Opus is stateful and any half-filled 20 ms frame would corrupt
// output after the seek discontinuity.
AudioReencoder.prototype.reset = async function() {
    if (!this.ready) return;
    // Close and recreate — WebCodecs .reset() semantics vary across polyfill
    // implementations, and recreating is straightforward and cheap.
    try { this.decoder.close(); } catch (e) {}
    try { this.encoder.close(); } catch (e) {}
    this._outputQueue = [];
    // Re-anchor timestamps to the first post-seek input frame and clear the
    // Bresenham accumulator so frame-count rounding stays unbiased after seek.
    this._sampleAccum = 0;
    this._anchorTsUs = null;
    this._totalOutSamples = 0;
    this._createDecoder();
    this._createEncoder();
};

// Swap the input codec for track-switching. Output config (Opus 48 kHz stereo)
// is unchanged, so the muxer's audio codecpar stays valid and the MSE init
// segment does not change.
//
// Verifies decoder support BEFORE tearing down the existing decoder so that
// a failed switch (e.g. neither native nor polyfill handles the codec) leaves
// the previous audio pipeline intact. Re-selects between native/polyfill based
// on the NEW codec's config — a switch from AAC (native) to EAC3 (polyfill)
// swaps the decoder class as well.
AudioReencoder.prototype.reconfigureInput = async function(inputCodec, sampleRate, channels, extradata, timeBase) {
    var prev = {
        inputCodec: this.inputCodec,
        inputSampleRate: this.inputSampleRate,
        inputChannels: this.inputChannels,
        inputExtradata: this.inputExtradata,
        inputTimeBaseNum: this.inputTimeBaseNum,
        inputTimeBaseDen: this.inputTimeBaseDen,
        _AudioDecoderClass: this._AudioDecoderClass,
        _EncodedAudioChunkClass: this._EncodedAudioChunkClass,
        _usingPolyfill: this._usingPolyfill
    };
    this.inputCodec = inputCodec;
    this.inputSampleRate = sampleRate || 48000;
    this.inputChannels = channels || 2;
    this.inputExtradata = extradata || null;
    if (timeBase && timeBase.den > 0) {
        this.inputTimeBaseNum = timeBase.num;
        this.inputTimeBaseDen = timeBase.den;
    }

    try {
        await this._selectDecoderImpl();
    } catch (e) {
        // Roll back — neither native nor polyfill supports the new codec.
        this.inputCodec = prev.inputCodec;
        this.inputSampleRate = prev.inputSampleRate;
        this.inputChannels = prev.inputChannels;
        this.inputExtradata = prev.inputExtradata;
        this.inputTimeBaseNum = prev.inputTimeBaseNum;
        this.inputTimeBaseDen = prev.inputTimeBaseDen;
        this._AudioDecoderClass = prev._AudioDecoderClass;
        this._EncodedAudioChunkClass = prev._EncodedAudioChunkClass;
        this._usingPolyfill = prev._usingPolyfill;
        throw e;
    }

    // Keep the same outputChannels — changing it mid-stream would invalidate
    // the already-written OpusHead. If the new track is mono and the previous
    // was stereo, Opus will re-expand to stereo silently.
    await this.reset();
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
    this._outputQueue = [];
};
