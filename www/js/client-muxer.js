/* Client Muxer - libav.js wrapper for muxing packets into fragmented MP4 */

var _muxerInstanceId = 0;

// Build an AVCodecParameters* describing Opus @ sampleRate/channels for use as
// an output stream in ff_init_muxer. WebCodecs emits raw Opus packets — libav
// never sees the encode path, so we need a codecpar built from scratch with
// correct OpusHead extradata (19 bytes: magic, version, channels, pre_skip,
// input sample rate, output gain, mapping family).
//
// Rather than hand-build the extradata (easy to get pre_skip, mapping family,
// or channel mapping table wrong), we use libav's libopus encoder purely as
// a codecpar factory: alloc a context, open it (libopus generates OpusHead),
// copy the resulting codecpar out, free the context. The encoder itself is
// never fed frames.
async function buildOpusCodecpar(libav, channels, sampleRate) {
    channels = channels || 2;
    sampleRate = sampleRate || 48000;
    var enc = await libav.avcodec_find_encoder_by_name("libopus");
    if (!enc) throw new Error("libopus encoder not available in libav build");
    var ctx = await libav.avcodec_alloc_context3(enc);
    if (!ctx) throw new Error("avcodec_alloc_context3(libopus) returned null");
    try {
        await libav.AVCodecContext_sample_rate_s(ctx, sampleRate);
        // This libav.js build's libopus accepts flt (packed float) or s16 only,
        // not the fltp (planar float) that mainline FFmpeg uses. The sample
        // format here is purely for avcodec_open2 to succeed — actual encoding
        // runs through WebCodecs, which supplies its own samples to the muxer
        // as pre-encoded Opus packets.
        await libav.AVCodecContext_sample_fmt_s(ctx, libav.AV_SAMPLE_FMT_FLT);
        if (typeof libav.AVCodecContext_ch_layout_nb_channels_s === "function") {
            await libav.AVCodecContext_ch_layout_nb_channels_s(ctx, channels);
        }
        if (typeof libav.AVCodecContext_channels_s === "function") {
            await libav.AVCodecContext_channels_s(ctx, channels);
        }
        if (typeof libav.AVCodecContext_channel_layoutmask_s === "function") {
            // FL+FR = 0x3, FC = 0x4
            var mask = (channels === 1) ? 0x4 : 0x3;
            await libav.AVCodecContext_channel_layoutmask_s(ctx, mask);
        }
        await libav.AVCodecContext_bit_rate_s(ctx, 128000);
        if (typeof libav.AVCodecContext_time_base_s === "function") {
            await libav.AVCodecContext_time_base_s(ctx, 1, sampleRate);
        }
        var ret = await libav.avcodec_open2(ctx, enc, 0);
        if (ret < 0) throw new Error("avcodec_open2(libopus) failed: " + ret);
        var codecpar = await libav.avcodec_parameters_alloc();
        if (!codecpar) throw new Error("avcodec_parameters_alloc returned null");
        await libav.avcodec_parameters_from_context(codecpar, ctx);
        return codecpar;
    } finally {
        try { await libav.avcodec_free_context_js(ctx); } catch (e) {}
    }
}

function ClientMuxer(libav) {
    this.libav = libav;
    this._deviceName = "mux_" + (++_muxerInstanceId) + ".mp4";
    this.oc = 0;      // output format context
    this.fmt = 0;     // output format
    this.pb = 0;      // IO context
    this.pkt = 0;     // reusable packet
    this.outputChunks = [];
    this.initialized = false;
    this.streamMap = {};        // input stream index -> output stream index
    this.lastStreamDts = {};    // last DTS written per output stream (cross-batch monotonicity)

    // Per-stream DTS-synthesis state (keyed by outputIdx).
    //
    // MKV demuxers set DTS = PTS, which breaks B-frame content: fMP4 needs
    // monotonic DTS in decode order with PTS reordered via signed CTS offsets.
    // For video streams we measure reorder depth adaptively during a warmup
    // phase, then use a bounded lookahead queue of that depth to synthesize
    // correct DTS values. For audio/other, we pass DTS = PTS through unchanged.
    this._streamKind = {};                // "video" | "audio" | "other"
    this._timeBase = {};                  // { num, den } per output stream
    this._reorderDepth = {};              // integer reorder depth (null until measured for video)
    this._depthMeasured = {};             // true once reorderDepth is finalized
    this._warmupBuf = {};                 // packets held during reorder-depth measurement
    this._warmupPtsHistory = {};          // PTS values observed during warmup (decode order)
    this._runningMaxDisplacement = {};    // current max displacement during warmup
    this._stableCount = {};               // packets since last displacement change
    this._warmupKeyframes = {};           // count of keyframes seen during warmup
    this._lookahead = {};                 // bounded queue of packets awaiting DTS emission
    this._ptsSorted = {};                 // ascending PTS values parallel to _lookahead
    this._packetDurationSeen = {};        // demuxer populated pkt.duration > 0 at least once
    this._packetDurationReliable = {};    // Tier A trustworthy (cross-checked against modal)
    this._durationFallback = {};          // cached modal/fallback duration (ticks)
    this._lastEmittedPts = {};            // previous emitted packet's PTS
    this._dtsShift = {};                  // per-stream DTS shift (= reorderDepth × modalDuration)
    this._clampCount = {};                // monotonicity clamp trigger count
    this._synthesisFallback = {};         // revert to DTS = PTS if clamp count exceeds threshold
    this._isHevc = {};                    // true for HEVC video streams (enables CRA flag filtering)
}

// Tunables
ClientMuxer.WARMUP_MIN_PACKETS = 16;
ClientMuxer.WARMUP_MIN_PACKETS_NO_B = 32;
ClientMuxer.WARMUP_MAX_PACKETS = 64;
ClientMuxer.REORDER_DEPTH_CAP = 16;
ClientMuxer.CLAMP_FALLBACK_THRESHOLD = 5;

// Initialize muxer with stream configurations from demuxer
// streamInfos: array of { inputIndex, codecpar, time_base_num, time_base_den }
ClientMuxer.prototype.init = async function(streamInfos) {
    // Create streaming writer device for fMP4 output
    var deviceName = this._deviceName;
    await this.libav.mkstreamwriterdev(deviceName);

    // Capture output bytes via onwrite callback
    var self = this;
    this.libav.onwrite = function(name, position, buffer) {
        if (name === deviceName) {
            // Must copy the buffer — it may be a subarray of WASM memory
            self.outputChunks.push(new Uint8Array(buffer));
        }
    };

    // Build stream contexts array for ff_init_muxer
    // Each entry: [codecpar, time_base_num, time_base_den]
    var streamCtxs = [];
    for (var i = 0; i < streamInfos.length; i++) {
        var info = streamInfos[i];
        streamCtxs.push([info.codecpar, info.time_base_num, info.time_base_den]);
        this.streamMap[info.inputIndex] = i;
        this._timeBase[i] = { num: info.time_base_num, den: info.time_base_den };
        this._resetStreamState(i);
        // HEVC detection comes from probe-data codec name (libav.js doesn't
        // export AV_CODEC_ID_HEVC directly). We use this to gate CRA-flag
        // filtering in the mux loop — see the "HEVC open-GOP filter" block.
        var cname = (info.codecName || "").toLowerCase();
        this._isHevc[i] = (cname === "hevc" || cname === "h265");
    }

    // Initialize muxer
    var ret = await this.libav.ff_init_muxer({
        format_name: "mp4",
        filename: deviceName,
        open: true,
        codecpars: true
    }, streamCtxs);

    this.oc = ret[0];
    this.fmt = ret[1];
    this.pb = ret[2];

    // Set fragmented MP4 flags (must be before writing header)
    // frag_keyframe: new fragment at each keyframe (IDR + CRA for HEVC)
    // empty_moov: no data in initial moov atom (streaming compatible)
    // default_base_moof: required for MSE compatibility
    // negative_cts_offsets: allows B-frame CTS (PTS-DTS) to be negative.
    //   Still useful as a safety net even with correct DTS synthesis — some
    //   B-pyramid patterns produce legitimately negative CTS offsets.
    // For HEVC open-GOP content, we don't change fragmentation here — instead
    // RASL leading pictures (the samples with PTS earlier than their CRA
    // keyframe) are stripped in the mux loop below. See the "HEVC RASL
    // stripping" block.
    await this.libav.av_opt_set(this.oc, "movflags",
        "frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets",
        this.libav.AV_OPT_SEARCH_CHILDREN);

    // Note on open-GOP handling: HEVC content uses CRA random-access points
    // whose leading B-pictures have PTS earlier than the CRA itself. Chrome's
    // MSE flags "sync sample with later PTS than dependent non-sync sample"
    // and halves playback pacing. We handle this at the packet level by
    // clearing AV_PKT_FLAG_KEY on CRA / BLA packets in the mux loop below —
    // see the "HEVC open-GOP filter" block. That also has the side effect
    // that only real IDR frames trigger frag_keyframe fragment boundaries,
    // which is exactly what we want. No min_frag_duration is set here
    // because buffering 10s of samples before the first fragment emit would
    // exceed the pump-loop watchdog timeout.

    // Inspect each output stream's codec_type to drive the DTS-synthesis
    // branch (video needs full synthesis; audio/other pass through).
    try {
        var nbStreams = await this.libav.AVFormatContext_nb_streams(this.oc);
        for (var si = 0; si < nbStreams; si++) {
            var streamPtr = await this.libav.AVFormatContext_streams_a(this.oc, si);
            var codecparPtr = await this.libav.AVStream_codecpar(streamPtr);
            var codecType = await this.libav.AVCodecParameters_codec_type(codecparPtr);

            if (codecType === this.libav.AVMEDIA_TYPE_VIDEO) {
                this._streamKind[si] = "video";
                // For HEVC, force the sample-entry codec_tag to 'hvc1' (instead
                // of the default 'hev1'). With 'hev1', VPS/SPS/PPS parameter
                // sets are inline per fragment, and Chrome's D3D11 hardware
                // HEVC path re-parses them on every fragment — which caps
                // decode throughput below real-time. 'hvc1' puts parameter
                // sets only in the sample description box, letting the hw
                // decoder init once and stay in its fast path.
                // FOURCC = MKTAG('h','v','c','1') = 0x31637668 (little-endian).
                if (this._isHevc[si] &&
                    typeof this.libav.AVCodecParameters_codec_tag_s === "function") {
                    try {
                        await this.libav.AVCodecParameters_codec_tag_s(codecparPtr, 0x31637668);
                        SP.log.debug("Muxer", "Set codec_tag='hvc1' on HEVC output stream", si);
                    } catch (e) {
                        SP.log.warn("Muxer", "Failed to set hvc1 tag:", e.message || e);
                    }
                }
            } else if (codecType === this.libav.AVMEDIA_TYPE_AUDIO) {
                this._streamKind[si] = "audio";
                this._reorderDepth[si] = 0;
                this._depthMeasured[si] = true;
            } else {
                this._streamKind[si] = "other";
                this._reorderDepth[si] = 0;
                this._depthMeasured[si] = true;
            }
        }
    } catch (e) {
        SP.log.warn("Muxer", "Could not inspect stream kinds:", e.message || e);
    }

    // Write header
    await this.libav.avformat_write_header(this.oc, 0);

    // Flush to ensure ftyp data reaches onwrite callback
    await this.libav.avio_flush(this.pb);

    // Collect init segment (ftyp) written by avformat_write_header.
    // The moov box is written lazily by the mp4 muxer on the first fragment.
    this._initSegment = this._collectOutput();

    // Allocate packet for writing
    this.pkt = await this.libav.av_packet_alloc();

    this.initialized = true;
};

// Reset all per-stream DTS-synthesis state for a given output index.
// Called from init() once per stream, and implicitly by constructing a new
// ClientMuxer on seek (which destroys the old one and makes a fresh instance).
ClientMuxer.prototype._resetStreamState = function(outputIdx) {
    this._streamKind[outputIdx] = "video"; // default; set precisely after init_muxer
    this._reorderDepth[outputIdx] = null;
    this._depthMeasured[outputIdx] = false;
    this._warmupBuf[outputIdx] = [];
    this._warmupPtsHistory[outputIdx] = [];
    this._runningMaxDisplacement[outputIdx] = 0;
    this._stableCount[outputIdx] = 0;
    this._warmupKeyframes[outputIdx] = 0;
    this._lookahead[outputIdx] = [];
    this._ptsSorted[outputIdx] = [];
    this._packetDurationSeen[outputIdx] = false;
    this._packetDurationReliable[outputIdx] = true;
    this._durationFallback[outputIdx] = 0;
    this._lastEmittedPts[outputIdx] = null;
    this._clampCount[outputIdx] = 0;
    this._synthesisFallback[outputIdx] = false;
    this._isHevc[outputIdx] = false;
    this._dtsShift[outputIdx] = 0;
};

// True if any video stream is still measuring reorder depth (i.e. may return
// null from mux() without that indicating a pipeline stall).
ClientMuxer.prototype.isWarmingUp = function() {
    for (var k in this._depthMeasured) {
        if (this._streamKind[k] === "video" && !this._depthMeasured[k]) return true;
    }
    return false;
};

// Read-only snapshot of per-stream DTS-synthesis state for the first video
// stream, for the metrics UI. Returns null if no video stream is registered.
ClientMuxer.prototype.getVideoStats = function() {
    for (var k in this._streamKind) {
        if (this._streamKind[k] !== "video") continue;
        return {
            reorderDepth: this._reorderDepth[k],
            depthMeasured: !!this._depthMeasured[k],
            modalDuration: this._durationFallback[k] || 0,
            warmingUp: !this._depthMeasured[k],
            clampCount: this._clampCount[k] || 0,
            pktDurationReliable: !!this._packetDurationReliable[k],
            pktDurationSeen: !!this._packetDurationSeen[k],
            synthesisFallback: !!this._synthesisFallback[k],
            timeBase: this._timeBase[k] || null
        };
    }
    return null;
};

// Inspect the first VCL NAL unit of a length-prefixed HEVC packet and return
// its nal_unit_type (0-31 for VCL/picture NALs). Returns -1 if no VCL NAL is
// found in the packet (e.g. the packet contains only parameter sets).
//
// HEVC packets in mp4 format are a concatenation of [4-byte length][NAL]
// chunks. Non-VCL NALs (VPS=32, SPS=33, PPS=34, SEI=39/40, ...) can precede
// the picture NAL inside a single packet, so we iterate.
//
// HEVC NAL header is 2 bytes: [0_bit][nal_unit_type:6][nuh_layer_id:6][nuh_temporal_id_plus1:3]
// which spans a 16-bit big-endian word. The type lives in bits 1-6 of byte 0.
function _hevcFirstVclNalType(data) {
    if (!data || data.length < 5) return -1;
    var offset = 0;
    while (offset + 5 <= data.length) {
        var len = (data[offset] << 24 >>> 0) |
                  (data[offset + 1] << 16) |
                  (data[offset + 2] << 8) |
                  data[offset + 3];
        if (len <= 0 || offset + 4 + len > data.length) return -1;
        var nalType = (data[offset + 4] >> 1) & 0x3F;
        if (nalType <= 31) return nalType;   // VCL / picture NAL
        offset += 4 + len;
    }
    return -1;
}

// Binary insert `val` into ascending `arr`.
function _sortedInsert(arr, val) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    arr.splice(lo, 0, val);
}

// Remove the first occurrence of `val` from ascending `arr`.
function _sortedRemove(arr, val) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    if (lo < arr.length && arr[lo] === val) arr.splice(lo, 1);
}

// Mux a batch of packets and return the fMP4 fragment data
// packets: Record<inputStreamIndex, Packet[]> (as returned by ff_read_frame_multi)
// Returns: Uint8Array of fMP4 data, or null if no data (warmup in progress)
ClientMuxer.prototype.mux = async function(packets) {
    if (!this.initialized) return null;

    // Clear output buffer
    this.outputChunks = [];

    // Accumulate packets ready for ff_write_multi. Cross-stream interleaving
    // by DTS happens at the end after all streams have been processed.
    var flatPackets = [];

    for (var inputIdx in packets) {
        var outputIdx = this.streamMap[inputIdx];
        if (outputIdx === undefined) continue;
        var pkts = packets[inputIdx];
        if (!pkts || pkts.length === 0) continue;

        for (var i = 0; i < pkts.length; i++) {
            var pkt = pkts[i];

            // Clamp negative PTS (rare container quirk)
            if (pkt.pts < 0) { pkt.pts = 0; pkt.ptshi = 0; }

            // Track whether the demuxer ever populates pkt.duration — drives
            // Tier A duration selection after warmup.
            if (pkt.duration > 0) this._packetDurationSeen[outputIdx] = true;

            pkt.stream_index = outputIdx;

            // Note: we previously stripped HEVC leading pictures (RADL/RASL,
            // NAL types 6-9) to avoid Chrome's "RAP with later PTS than
            // dependent non-key" warning on open-GOP content. That worked
            // for the warning, but created PTS gaps where the stripped
            // frames used to be. FFmpeg's mp4 muxer then extended the
            // preceding sample's duration to span the gap — producing trun
            // sample_duration entries of 5000+ ticks (≈0.3 s at 16000 Hz
            // timebase) that Chrome used as the authoritative frame-time,
            // collapsing playback pacing to ~9 fps. Keeping all packets
            // in the stream preserves uniform sample durations even though
            // Chrome may warn about the RAP/PTS mismatch.

            if (this._streamKind[outputIdx] !== "video" || this._synthesisFallback[outputIdx]) {
                this._emitPassthrough(outputIdx, pkt, flatPackets);
            } else if (!this._depthMeasured[outputIdx]) {
                this._warmupPush(outputIdx, pkt, flatPackets);
            } else {
                this._steadyPush(outputIdx, pkt, flatPackets);
            }
        }
    }

    if (flatPackets.length === 0) {
        // No output yet — warmup still filling, or empty input. Keep
        // _initSegment held so it can be prepended to the first real fragment.
        return null;
    }

    // Sort across streams by DTS for proper interleaving
    flatPackets.sort(function(a, b) {
        return a.dts - b.dts || a.stream_index - b.stream_index;
    });

    // Write packets non-interleaved (DTS is already correct and monotonic)
    await this.libav.ff_write_multi(this.oc, this.pkt, flatPackets, false);

    // Flush the IO context to ensure all data reaches the onwrite callback.
    await this.libav.avio_flush(this.pb);

    var output = this._collectOutput();

    // Prepend init segment (ftyp) to first real output. We deliberately hold
    // _initSegment through any warmup null-returns so MSE only sees "init +
    // media" together, never init alone.
    if (this._initSegment && output && output.length > 0) {
        var combined = new Uint8Array(this._initSegment.length + output.length);
        combined.set(this._initSegment, 0);
        combined.set(output, this._initSegment.length);
        output = combined;
        this._initSegment = null;
    }

    return output;
};

// Passthrough emission for audio/other/synthesis-fallback: DTS = PTS, keep
// existing duration, enforce per-stream monotonic DTS.
ClientMuxer.prototype._emitPassthrough = function(outputIdx, pkt, flatPackets) {
    var dts = pkt.pts;
    var lastDts = this.lastStreamDts[outputIdx];
    if (lastDts !== undefined && dts <= lastDts) dts = lastDts + 1;
    pkt.dts = dts;
    pkt.dtshi = 0;
    this.lastStreamDts[outputIdx] = dts;
    flatPackets.push(pkt);
};

// Warmup: accumulate a video packet, update running max displacement, and
// freeze + drain if a trigger fires.
//
// Displacement of packet i = count of prior packets j < i with PTS_j > PTS_i.
// The max displacement observed over the warmup window equals the stream's
// B-frame reorder depth.
ClientMuxer.prototype._warmupPush = function(outputIdx, pkt, flatPackets) {
    var buf = this._warmupBuf[outputIdx];
    var history = this._warmupPtsHistory[outputIdx];
    var pts = pkt.pts;

    var displacement = 0;
    for (var i = 0; i < history.length; i++) {
        if (history[i] > pts) displacement++;
    }

    var prevMax = this._runningMaxDisplacement[outputIdx];
    if (displacement > prevMax) {
        this._runningMaxDisplacement[outputIdx] = displacement;
        this._stableCount[outputIdx] = 0;
    } else {
        this._stableCount[outputIdx] += 1;
    }

    history.push(pts);
    buf.push(pkt);
    if (pkt.flags & 1 /* AV_PKT_FLAG_KEY */) {
        this._warmupKeyframes[outputIdx] += 1;
    }

    // Freeze triggers (first condition that fires wins)
    var depth = this._runningMaxDisplacement[outputIdx];
    var n = buf.length;
    var stable = this._stableCount[outputIdx];
    var kfs = this._warmupKeyframes[outputIdx];
    var minPackets = (depth === 0) ? ClientMuxer.WARMUP_MIN_PACKETS_NO_B : ClientMuxer.WARMUP_MIN_PACKETS;
    var requiredStable = Math.max(depth + 4, 8);

    var shouldFreeze =
        n >= ClientMuxer.WARMUP_MAX_PACKETS ||
        (kfs >= 2 && n >= minPackets && stable >= requiredStable);

    if (shouldFreeze) {
        this._freezeDepthAndDrain(outputIdx, flatPackets);
    }
};

// Freeze the measured reorder depth, derive modal frame duration, and drain
// the warmup buffer through the steady-state pipeline.
ClientMuxer.prototype._freezeDepthAndDrain = function(outputIdx, flatPackets) {
    var depth = this._runningMaxDisplacement[outputIdx];
    // HEVC floor: real-world HEVC encodes routinely use B-pyramids with
    // depth 3+, but warmup can freeze at 2 if the first observed GOPs
    // happen not to expose the deepest pyramid level. Seeding a floor of
    // 3 covers the common case cheaply; the CTS-sanity peek in
    // _steadyPush is the correctness backstop for deeper pyramids.
    if (this._isHevc[outputIdx] && depth < 3) depth = 3;

    // Max-packets-freeze safety floor. _warmupPush freezes on the first of
    // two triggers: the confident stable branch (kfs >= 2 && stable >=
    // requiredStable) or the n >= WARMUP_MAX_PACKETS fallback. If we
    // exhausted the window without the confident branch firing and still
    // measured depth=0, we don't actually have evidence the stream lacks
    // B-frames — we just haven't seen one yet (long-GOP H.264 content can
    // easily go >64 packets between keyframes, e.g. 24fps movie encodes).
    // Seeding depth=2 here keeps the steady-state lookahead (and the
    // dtsShift computed below) wide enough to absorb the first B-frames
    // without tripping the monotonicity clamp, which in turn prevents the
    // fallback cascade into DTS=PTS passthrough that breaks B-frame streams.
    var hitMaxPackets = this._warmupBuf[outputIdx].length >= ClientMuxer.WARMUP_MAX_PACKETS;
    if (depth === 0 && hitMaxPackets) depth = 2;

    this._reorderDepth[outputIdx] = depth;
    this._depthMeasured[outputIdx] = true;

    // Modal frame duration = most common delta between adjacent *sorted* PTS
    // values in the warmup window. Sorted ordering gives us presentation-order
    // gaps, which equal frame duration for CFR content.
    var history = this._warmupPtsHistory[outputIdx].slice().sort(function(a, b) { return a - b; });
    var deltaCounts = new Map();
    var topDelta = 0;
    var topCount = 0;
    for (var i = 1; i < history.length; i++) {
        var d = history[i] - history[i - 1];
        if (d <= 0) continue;
        var c = (deltaCounts.get(d) || 0) + 1;
        deltaCounts.set(d, c);
        if (c > topCount) { topCount = c; topDelta = d; }
    }
    if (topDelta > 0) {
        this._durationFallback[outputIdx] = topDelta;
    } else {
        // Tier C — 24fps in stream timebase
        var tb = this._timeBase[outputIdx];
        this._durationFallback[outputIdx] = (tb && tb.num > 0)
            ? Math.round(tb.den / tb.num / 24)
            : 1;
        SP.log.warn("Muxer", "Stream", outputIdx,
            "warmup yielded no usable PTS deltas; using 24fps fallback duration",
            this._durationFallback[outputIdx]);
    }

    // Cross-check Tier A: if any warmup packet.duration disagrees with modal
    // by >10%, distrust demuxer-provided durations for this stream.
    var warmupBuf = this._warmupBuf[outputIdx];
    if (this._packetDurationSeen[outputIdx]) {
        var modal = this._durationFallback[outputIdx];
        var tolerance = modal * 0.1;
        for (var j = 0; j < warmupBuf.length; j++) {
            var d = warmupBuf[j].duration;
            if (d > 0 && Math.abs(d - modal) > tolerance) {
                this._packetDurationReliable[outputIdx] = false;
                break;
            }
        }
    }

    // DTS shift: the sort-assign algorithm produces DTS = (i-th smallest PTS)
    // in decode order. For B-pyramid streams this can emit DTS > PTS on
    // frames where decode position exceeds presentation rank. FFmpeg's mp4
    // muxer rewrites pts=dts on any such packet, creating duplicate PTS
    // values that Chrome MSE dedups — halving effective frame rate.
    // Shifting all DTS down by reorderDepth × modalDuration guarantees
    // CTS ≥ 0 in steady state: PTS[i] differs from its sorted rank by at
    // most reorderDepth positions, so PTS[i] ≥ sortedPTS[rank(i)] - shift.
    // Initial packets where shifted DTS would go negative are clamped to
    // monotonic non-negative values by _emitSynthesized.
    this._dtsShift[outputIdx] = depth * this._durationFallback[outputIdx];

    SP.log.debug("Muxer", "Stream", outputIdx,
        "frozen: reorderDepth=" + depth,
        "modalDuration=" + this._durationFallback[outputIdx],
        "dtsShift=" + this._dtsShift[outputIdx],
        "pktDurationReliable=" + this._packetDurationReliable[outputIdx]);

    // Drain warmup buffer through the steady-state pipeline
    this._warmupBuf[outputIdx] = [];
    this._warmupPtsHistory[outputIdx] = [];
    for (var k = 0; k < warmupBuf.length; k++) {
        this._steadyPush(outputIdx, warmupBuf[k], flatPackets);
    }
};

// Steady-state: push packet into the lookahead queue, emit head packets while
// the queue exceeds reorder depth.
//
// Invariant: when the queue holds reorderDepth+1 packets, the algorithm
// assigns the oldest decode-order packet a DTS equal to the i-th smallest
// PTS seen so far (where i advances each pop). That produces a monotonic
// DTS sequence equal to the sorted PTS values — correct for both flat B and
// B-pyramid streams when the lookahead is ≥ actual reorder depth.
//
// When lookahead is too shallow, the monotonicity clamp in _emitSynthesized
// catches the breakage reactively and grows reorderDepth. A proactive check
// is not viable here: legitimate pyramid streams emit negative CTS (DTS >
// PTS) at pyramid-level depth, so "DTS > PTS" does not discriminate between
// correct pyramid output and under-measurement.
ClientMuxer.prototype._steadyPush = function(outputIdx, pkt, flatPackets) {
    var la = this._lookahead[outputIdx];
    var sorted = this._ptsSorted[outputIdx];
    var depth = this._reorderDepth[outputIdx];

    la.push(pkt);
    _sortedInsert(sorted, pkt.pts);

    while (la.length > depth) {
        var head = la.shift();
        var dts = sorted[0];
        _sortedRemove(sorted, dts);
        this._emitSynthesized(outputIdx, head, dts, flatPackets);
    }
};

// Shared handler for both insufficient-depth signals:
//   - monotonicity clamp (DTS <= lastDts in _emitSynthesized), and
//   - CTS-sanity peek (candidate DTS > head.pts + 1.5×modal in _steadyPush).
// Grows reorder depth stickily and trips the synthesis-off fallback once the
// clamp count crosses CLAMP_FALLBACK_THRESHOLD. `reason` is a short tag for
// the log line so we can tell which path fired.
ClientMuxer.prototype._onDepthInsufficient = function(outputIdx, reason) {
    this._clampCount[outputIdx] += 1;
    if (this._clampCount[outputIdx] === 1) {
        SP.log.warn("Muxer", "Stream", outputIdx,
            reason + " — reorder depth likely insufficient. Growing.");
    }
    if (this._reorderDepth[outputIdx] < ClientMuxer.REORDER_DEPTH_CAP) {
        this._reorderDepth[outputIdx] += 1;
    }
    if (this._clampCount[outputIdx] >= ClientMuxer.CLAMP_FALLBACK_THRESHOLD) {
        SP.log.error("Muxer", "Stream", outputIdx,
            "exceeded clamp threshold; disabling DTS synthesis (DTS = PTS fallback).");
        this._synthesisFallback[outputIdx] = true;
    }
};

// Assign DTS + duration to a packet and append it to the write list.
// Preserves the original packet object reference so pkt.flags (including
// AV_PKT_FLAG_KEY) survives into ff_write_multi — crucial for frag_keyframe
// fragment boundaries to land on real I-frames.
ClientMuxer.prototype._emitSynthesized = function(outputIdx, pkt, dts, flatPackets) {
    // Apply DTS shift (see _freezeDepthAndDrain). Clamp negatives to 0; the
    // monotonicity guard below ensures subsequent DTS values advance even if
    // several initial packets bottom out at 0.
    var shift = this._dtsShift[outputIdx] || 0;
    if (shift > 0) {
        dts -= shift;
        if (dts < 0) dts = 0;
    }

    var lastDts = this.lastStreamDts[outputIdx];
    if (lastDts !== undefined && dts <= lastDts) {
        // Safety net — reached only when the CTS-sanity peek in _steadyPush
        // didn't catch an under-measured depth (e.g. first packet after
        // freeze, before the window is primed). Grow depth and clamp DTS.
        this._onDepthInsufficient(outputIdx, "DTS monotonicity clamp fired");
        dts = lastDts + 1;
    }

    pkt.dts = dts;
    pkt.dtshi = 0;
    this.lastStreamDts[outputIdx] = dts;

    // Duration: Tier A (trusted demuxer-provided) → Tier B/C (cached modal)
    var duration = this._computeDuration(outputIdx, pkt);
    if (duration > 0) {
        pkt.duration = duration;
        pkt.durationhi = 0;
    }

    this._lastEmittedPts[outputIdx] = pkt.pts;
    flatPackets.push(pkt);
};

// Pick a duration for this packet. Tier A uses the demuxer's own duration if
// cross-check found it reliable; otherwise falls back to the modal duration
// derived at warmup freeze (or a 24fps fallback if warmup couldn't measure).
//
// (FFmpeg's fragmented mp4 muxer actually computes per-sample trun durations
// from DTS deltas of consecutive written packets, overriding pkt.duration
// for all but the final sample of a fragment. So this value mostly matters
// for the tail sample. Kept faithful to the demuxer when possible.)
ClientMuxer.prototype._computeDuration = function(outputIdx, pkt) {
    if (this._packetDurationSeen[outputIdx]
            && this._packetDurationReliable[outputIdx]
            && pkt.duration > 0) {
        return pkt.duration;
    }
    return this._durationFallback[outputIdx] || 0;
};

// Drain all packets still buffered in warmup / lookahead into flatPackets.
// Called from flush() to recover up to reorderDepth tail packets that would
// otherwise be lost at EOF.
ClientMuxer.prototype._drainLookahead = function(flatPackets) {
    // Force-freeze any video stream still in warmup.
    for (var ka in this._streamKind) {
        if (this._streamKind[ka] !== "video") continue;
        if (!this._depthMeasured[ka]) {
            if (this._warmupBuf[ka].length > 0) {
                this._freezeDepthAndDrain(ka, flatPackets);
            } else {
                this._reorderDepth[ka] = 0;
                this._depthMeasured[ka] = true;
            }
        }
    }
    // Drain each stream's lookahead queue.
    for (var kb in this._streamKind) {
        var la = this._lookahead[kb];
        var sorted = this._ptsSorted[kb];
        if (!la) continue;
        while (la.length > 0) {
            var head = la.shift();
            var dts = sorted.length > 0 ? sorted[0] : head.pts;
            _sortedRemove(sorted, dts);
            this._emitSynthesized(kb, head, dts, flatPackets);
        }
    }
};

// Flush remaining buffered packets and write the trailer. Call from the
// pump loop on EOF, append the returned bytes to the SourceBuffer, then
// endOfStream().
ClientMuxer.prototype.flush = async function() {
    if (!this.initialized) return null;
    this.outputChunks = [];

    // Drain warmup + lookahead tails so the last reorderDepth frames don't
    // disappear from the end of playback.
    var tailPackets = [];
    this._drainLookahead(tailPackets);

    if (tailPackets.length > 0) {
        tailPackets.sort(function(a, b) {
            return a.dts - b.dts || a.stream_index - b.stream_index;
        });
        await this.libav.ff_write_multi(this.oc, this.pkt, tailPackets, false);
    }

    await this.libav.av_write_trailer(this.oc);
    await this.libav.avio_flush(this.pb);
    var output = this._collectOutput();

    // Edge case: file was shorter than one mux() batch — init segment never
    // got prepended to a media fragment. Ship it with whatever we have.
    if (this._initSegment && output && output.length > 0) {
        var combined = new Uint8Array(this._initSegment.length + output.length);
        combined.set(this._initSegment, 0);
        combined.set(output, this._initSegment.length);
        output = combined;
        this._initSegment = null;
    } else if (this._initSegment && (!output || output.length === 0)) {
        output = this._initSegment;
        this._initSegment = null;
    }

    return output;
};

// Collect accumulated output chunks into a single Uint8Array
ClientMuxer.prototype._collectOutput = function() {
    if (this.outputChunks.length === 0) return null;

    var totalLen = 0;
    for (var i = 0; i < this.outputChunks.length; i++) {
        totalLen += this.outputChunks[i].length;
    }

    var result = new Uint8Array(totalLen);
    var offset = 0;
    for (var i = 0; i < this.outputChunks.length; i++) {
        result.set(this.outputChunks[i], offset);
        offset += this.outputChunks[i].length;
    }

    this.outputChunks = [];
    return result;
};

ClientMuxer.prototype.destroy = async function() {
    if (this.libav && this.initialized) {
        try {
            if (this.pkt) await this.libav.av_packet_free(this.pkt);
            if (this.oc) await this.libav.ff_free_muxer(this.oc, this.pb);
        } catch (e) {
            SP.log.warn("Muxer", "Cleanup error:", e);
        }
        // Remove the virtual stream-writer device to free Emscripten FS memory
        try {
            await this.libav.unlink(this._deviceName);
        } catch (e) {
            // May already be gone — ignore
        }
    }
    this.initialized = false;
};
