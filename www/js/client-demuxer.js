/* Client Demuxer - libav.js wrapper for container demuxing via range requests */

function ClientDemuxer(fileUrl, fileSize) {
    this.fileUrl = fileUrl;
    this.fileSize = fileSize;
    this.libav = null;
    this.fmtCtx = 0;
    this.streams = [];
    this.pkt = 0;
    this.videoStreamIndex = -1;
    this.audioStreamIndex = -1;
    this.duration = 0;
}

ClientDemuxer.prototype.init = async function() {
    // Load libav.js if not already loaded
    await loadVendor("libav", "vendor/libav-webcodecs.js");

    // Create libav instance (runs in Web Worker for non-blocking I/O)
    this.libav = await LibAV.LibAV();

    // Set up block reader device for range-request access
    await this.libav.mkblockreaderdev("input", this.fileSize);

    var self = this;
    this.libav.onblockread = async function(name, position, length) {
        try {
            var response = await fetch(self.fileUrl, {
                headers: { "Range": "bytes=" + position + "-" + (position + length - 1) }
            });
            var buffer = new Uint8Array(await response.arrayBuffer());
            await self.libav.ff_block_reader_dev_send(name, position, buffer);
        } catch (err) {
            console.error("[Demuxer] Range request failed:", err);
            // Send empty buffer to unblock libav
            await self.libav.ff_block_reader_dev_send(name, position, new Uint8Array(0));
        }
    };

    // Initialize demuxer
    var ret = await this.libav.ff_init_demuxer_file("input");
    this.fmtCtx = ret[0];
    this.streams = ret[1];

    // Find video and audio stream indices
    for (var i = 0; i < this.streams.length; i++) {
        var s = this.streams[i];
        if (s.codec_type === this.libav.AVMEDIA_TYPE_VIDEO && this.videoStreamIndex === -1) {
            this.videoStreamIndex = s.index;
        } else if (s.codec_type === this.libav.AVMEDIA_TYPE_AUDIO && this.audioStreamIndex === -1) {
            this.audioStreamIndex = s.index;
        }
    }

    // Allocate packet
    this.pkt = await this.libav.av_packet_alloc();

    // Get duration
    if (this.streams.length > 0 && this.streams[0].duration > 0) {
        this.duration = this.streams[0].duration;
    }

    console.log("[Demuxer] Initialized:", this.streams.length, "streams",
        "video:", this.videoStreamIndex, "audio:", this.audioStreamIndex);
};

// Set which audio stream to demux (by absolute stream index in the container)
ClientDemuxer.prototype.setAudioStream = function(streamIndex) {
    this.audioStreamIndex = streamIndex;
};

// Get the absolute stream index for the Nth audio track
ClientDemuxer.prototype.getAudioStreamIndex = function(audioTrackNum) {
    var count = 0;
    for (var i = 0; i < this.streams.length; i++) {
        if (this.streams[i].codec_type === this.libav.AVMEDIA_TYPE_AUDIO) {
            if (count === audioTrackNum) return this.streams[i].index;
            count++;
        }
    }
    return -1;
};

// Read a batch of packets. Returns { packets: Record<streamIndex, Packet[]>, eof: boolean }
ClientDemuxer.prototype.readPackets = async function(limitBytes) {
    if (!limitBytes) limitBytes = 1024 * 1024; // 1MB default

    var result = await this.libav.ff_read_frame_multi(this.fmtCtx, this.pkt, {
        limit: limitBytes,
        unify: false
    });

    var readResult = result[0];
    var packets = result[1];
    var eof = (readResult === -this.libav.AVERROR_EOF || readResult === this.libav.AVERROR_EOF);

    return { packets: packets, eof: eof };
};

// Seek to a time in seconds. Seeks backward to nearest keyframe.
ClientDemuxer.prototype.seek = async function(timeSeconds) {
    var stream = this.streams[this.videoStreamIndex] || this.streams[0];
    if (!stream) return;

    // Convert seconds to stream timebase
    var tb = stream.time_base_num / stream.time_base_den;
    var timestamp = timeSeconds / tb;
    var parts = this.libav.f64toi64(timestamp);

    await this.libav.avformat_seek_file_min(
        this.fmtCtx,
        stream.index,
        parts[0], parts[1], // timestamp lo, hi
        this.libav.AVSEEK_FLAG_BACKWARD
    );
};

ClientDemuxer.prototype.destroy = async function() {
    if (this.libav) {
        try {
            if (this.pkt) await this.libav.av_packet_free(this.pkt);
            if (this.fmtCtx) await this.libav.avformat_close_input_js(this.fmtCtx);
        } catch (e) {
            console.warn("[Demuxer] Cleanup error:", e);
        }
        this.libav.terminate();
        this.libav = null;
    }
};
