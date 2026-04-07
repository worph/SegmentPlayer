/* Client Muxer - libav.js wrapper for muxing packets into fragmented MP4 */

function ClientMuxer(libav) {
    this.libav = libav;
    this.oc = 0;      // output format context
    this.fmt = 0;     // output format
    this.pb = 0;      // IO context
    this.pkt = 0;     // reusable packet
    this.outputChunks = [];
    this.outputPos = 0;
    this.initialized = false;
    this.streamMap = {}; // input stream index -> output stream index
}

// Initialize muxer with stream configurations from demuxer
// streamInfos: array of { inputIndex, codecpar, time_base_num, time_base_den }
ClientMuxer.prototype.init = async function(streamInfos) {
    // Create streaming writer device (no seeking back — required for fMP4)
    await this.libav.mkstreamwriterdev("output.mp4");

    // Capture output bytes
    var self = this;
    this.libav.onwrite = function(name, position, buffer) {
        if (name === "output.mp4") {
            // Must copy the buffer — it may be a subarray of WASM memory
            self.outputChunks.push(new Uint8Array(buffer));
            self.outputPos += buffer.length;
        }
    };

    // Build stream contexts array for ff_init_muxer
    // Each entry: [codecpar, time_base_num, time_base_den]
    var streamCtxs = [];
    for (var i = 0; i < streamInfos.length; i++) {
        var info = streamInfos[i];
        streamCtxs.push([info.codecpar, info.time_base_num, info.time_base_den]);
        this.streamMap[info.inputIndex] = i;
    }

    // Initialize muxer
    var ret = await this.libav.ff_init_muxer({
        format_name: "mp4",
        filename: "output.mp4",
        open: true,
        codecpars: true
    }, streamCtxs);

    this.oc = ret[0];
    this.fmt = ret[1];
    this.pb = ret[2];

    // Set fragmented MP4 flags (must be before writing header)
    // frag_keyframe: new fragment at each keyframe
    // empty_moov: no data in initial moov atom (streaming compatible)
    // default_base_moof: required for MSE compatibility
    await this.libav.av_opt_set(this.oc, "movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        this.libav.AV_OPT_SEARCH_CHILDREN);

    // Write header
    await this.libav.avformat_write_header(this.oc, 0);

    // Allocate packet for writing
    this.pkt = await this.libav.av_packet_alloc();

    this.initialized = true;
};

// Mux a batch of packets and return the fMP4 fragment data
// packets: Record<inputStreamIndex, Packet[]> (as returned by ff_read_frame_multi)
// Returns: Uint8Array of fMP4 data, or null if no data
ClientMuxer.prototype.mux = async function(packets) {
    if (!this.initialized) return null;

    // Clear output buffer
    this.outputChunks = [];

    // Remap stream indices and flatten into a single packet array
    // ff_write_multi expects a flat Packet[] — each packet's stream_index tells
    // the muxer which output stream it belongs to
    var flatPackets = [];
    for (var inputIdx in packets) {
        var outputIdx = this.streamMap[inputIdx];
        if (outputIdx === undefined) continue;
        var pkts = packets[inputIdx];
        for (var i = 0; i < pkts.length; i++) {
            pkts[i].stream_index = outputIdx;
            flatPackets.push(pkts[i]);
        }
    }

    if (flatPackets.length === 0) return null;

    // Write packets (interleaved)
    await this.libav.ff_write_multi(this.oc, this.pkt, flatPackets, true);

    // Collect output
    return this._collectOutput();
};

// Flush remaining data and write trailer
ClientMuxer.prototype.flush = async function() {
    if (!this.initialized) return null;
    this.outputChunks = [];
    await this.libav.av_write_trailer(this.oc);
    return this._collectOutput();
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
            console.warn("[Muxer] Cleanup error:", e);
        }
    }
    this.initialized = false;
};
