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
    this._abortController = null; // for cancelling in-flight range requests
    this._bytesRead = 0;          // cumulative bytes fetched from origin (for metrics)
}

// Read-only stats snapshot for the metrics UI.
ClientDemuxer.prototype.getStats = function() {
    return {
        bytesRead: this._bytesRead,
        fileSize: this.fileSize
    };
};

ClientDemuxer.prototype.init = async function() {
    // Load libav.js if not already loaded. We use a custom "sp-audio" variant
    // (built via Docker from the libav.js sources with configs/configs/sp-audio).
    // Same 6.8.8.0 API as the base webcodecs build, but with additional audio
    // decoders compiled in: AC3, EAC3, DTS (dca), AAC, MP3, MP2, TRUEHD, ALAC,
    // plus the libopus encoder we already use for muxing. These decoders back
    // the WebCodecs polyfill fallback in audio-reencode.js when the browser's
    // native AudioDecoder rejects a codec (e.g. EAC3 on some Chromium builds).
    await loadVendor("libav", "vendor/libav-sp-audio.js");

    // Create libav instance (runs in Web Worker for non-blocking I/O)
    this.libav = await LibAV.LibAV();

    // Suppress FFmpeg's benign "pts < dts" warnings from the mp4 muxer.
    // The negative_cts_offsets movflag handles B-frame reorder delay via
    // signed CTS offsets (CTTS v1 box), making these warnings harmless.
    // NOTE: the current libav.js WASM build routes stderr directly through
    // Emscripten's printErr in the Worker, bypassing av_log level filtering.
    // The call below sets the correct intent for future libav.js versions.
    await this.libav.av_log_set_level(this.libav.AV_LOG_ERROR);

    // Set up block reader device for range-request access
    await this.libav.mkblockreaderdev("input", this.fileSize);

    // Read-ahead cache: libav.js reads in tiny 32KB blocks, causing thousands
    // of sequential HTTP requests. We fetch a large chunk and serve subsequent
    // block reads from cache — one HTTP request covers many libav reads.
    var READ_AHEAD = 5 * 1024 * 1024; // 5MB (covers a full 4MB readPackets batch)
    this._readCache = null;       // { start, end, data }

    var self = this;
    this.libav.onblockread = async function(name, position, length) {
        // Serve from cache if available
        var cache = self._readCache;
        if (cache && position >= cache.start && position + length <= cache.end) {
            var offset = position - cache.start;
            var slice = cache.data.subarray(offset, offset + length);
            await self.libav.ff_block_reader_dev_send(name, position, slice);
            return;
        }

        // Cache miss — fetch a larger chunk (read-ahead)
        var fetchEnd = Math.min(position + READ_AHEAD, self.fileSize) - 1;
        var controller = new AbortController();
        self._abortController = controller;
        try {
            var response = await fetch(self.fileUrl, {
                headers: { "Range": "bytes=" + position + "-" + fetchEnd },
                signal: controller.signal
            });
            var buffer = new Uint8Array(await response.arrayBuffer());
            // Track origin bytes for metrics (only the fresh fetch, not cache hits)
            self._bytesRead += buffer.length;
            // Cache the full response
            self._readCache = { start: position, end: position + buffer.length, data: buffer };
            // Send the requested portion to libav
            var slice = buffer.subarray(0, Math.min(length, buffer.length));
            await self.libav.ff_block_reader_dev_send(name, position, slice);
        } catch (err) {
            if (err.name !== "AbortError") {
                SP.log.error("Demuxer", "Range request failed:", err);
            }
            await self.libav.ff_block_reader_dev_send(name, position, new Uint8Array(0));
        } finally {
            if (self._abortController === controller) {
                self._abortController = null;
            }
        }
    };

    // Initialize demuxer with increased probesize for reliable codec detection.
    // The default probesize is too small for range-request I/O, causing
    // "unspecified number of channels" errors on audio tracks.
    var fmtCtx = await this.libav.avformat_open_input_js("input", null, null);
    this.fmtCtx = fmtCtx;
    // Set probesize large enough for the demuxer to read audio frames and
    // determine channel layout (AAC with channelConfiguration=0 needs this)
    var probeSize = Math.min(this.fileSize, 50000000); // up to 50MB
    await this.libav.av_opt_set(fmtCtx, "probesize", String(probeSize), 0);
    await this.libav.av_opt_set(fmtCtx, "analyzeduration", "10000000", 0);
    var streamsRaw = await this.libav.avformat_find_stream_info(fmtCtx);
    // Re-read streams after find_stream_info
    var nbStreams = await this.libav.AVFormatContext_nb_streams(fmtCtx);
    this.streams = [];
    for (var i = 0; i < nbStreams; i++) {
        var streamPtr = await this.libav.AVFormatContext_streams_a(fmtCtx, i);
        var codecparPtr = await this.libav.AVStream_codecpar(streamPtr);
        var par = await this.libav.ff_copyout_codecpar(codecparPtr);
        par.index = i;
        par.time_base_num = await this.libav.AVStream_time_base_num(streamPtr);
        par.time_base_den = await this.libav.AVStream_time_base_den(streamPtr);
        par.duration = await this.libav.AVStream_duration(streamPtr);
        par.codecpar = codecparPtr;
        this.streams.push(par);
    }

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

    SP.log.debug("Demuxer", "Initialized:", this.streams.length, "streams",
        "video:", this.videoStreamIndex, "audio:", this.audioStreamIndex);
};

// Fix missing audio codec parameters using probe data from the server.
// libav.js range-request I/O sometimes can't fully probe audio streams.
ClientDemuxer.prototype.fixAudioParams = async function(probeData) {
    if (!probeData || !probeData.audio || probeData.audio.length === 0) return;

    var audioCount = 0;
    for (var i = 0; i < this.streams.length; i++) {
        var s = this.streams[i];
        if (s.codec_type !== this.libav.AVMEDIA_TYPE_AUDIO) continue;

        var probeAudio = probeData.audio[audioCount];
        if (!probeAudio) { audioCount++; continue; }

        // Fix channel count if missing
        var streamPtr = await this.libav.AVFormatContext_streams_a(this.fmtCtx, s.index);
        var codecparPtr = await this.libav.AVStream_codecpar(streamPtr);

        var channels = 0;
        if (typeof this.libav.AVCodecParameters_ch_layout_nb_channels === "function") {
            channels = await this.libav.AVCodecParameters_ch_layout_nb_channels(codecparPtr);
        } else if (typeof this.libav.AVCodecParameters_channels === "function") {
            channels = await this.libav.AVCodecParameters_channels(codecparPtr);
        }

        if (!channels || channels <= 0) {
            var probeChannels = probeAudio.channels || 2;
            SP.log.debug("Demuxer", "Fixing audio stream", s.index, "channels:", 0, "→", probeChannels);
            if (typeof this.libav.AVCodecParameters_ch_layout_nb_channels_s === "function") {
                await this.libav.AVCodecParameters_ch_layout_nb_channels_s(codecparPtr, probeChannels);
            }
            if (typeof this.libav.AVCodecParameters_channels_s === "function") {
                await this.libav.AVCodecParameters_channels_s(codecparPtr, probeChannels);
            }
            s.channels = probeChannels;

            // Patch the AAC AudioSpecificConfig extradata to set channelConfiguration.
            // Without this, the mp4 muxer writes an init segment with 0 channels
            // in the esds box, and Chrome's MSE rejects it.
            var codec = await this.libav.AVCodecParameters_codec_id(codecparPtr);
            var extraPtr = await this.libav.AVCodecParameters_extradata(codecparPtr);
            var extraSize = await this.libav.AVCodecParameters_extradata_size(codecparPtr);
            if (extraPtr && extraSize >= 2) {
                var extradata = await this.libav.copyout_u8(extraPtr, extraSize);
                // AudioSpecificConfig: 5 bits audioObjectType, 4 bits frequencyIndex,
                // 4 bits channelConfiguration, then variable.
                // Byte layout for common case (AOT ≤ 31):
                //   byte0: [AOT:5][freq:3]   byte1: [freq:1][channels:4][...]
                var channelBits = (extradata[1] >> 3) & 0x0F;
                if (channelBits === 0 && probeChannels > 0) {
                    // Map channel count to AAC channelConfiguration
                    var channelConfig = probeChannels <= 7 ? probeChannels : 0;
                    if (channelConfig > 0) {
                        extradata[1] = (extradata[1] & 0x87) | (channelConfig << 3);
                        await this.libav.copyin_u8(extraPtr, extradata);
                        SP.log.debug("Demuxer", "Patched AAC AudioSpecificConfig channelConfiguration →", channelConfig);
                    }
                }
            }
        }

        audioCount++;
    }
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

// Read the codec-private extradata (e.g. AAC AudioSpecificConfig, Vorbis
// identification/setup headers) for an audio stream so it can be passed to
// AudioDecoder.configure({ description }). Returns null if the stream has no
// extradata or is not an audio stream.
ClientDemuxer.prototype.getAudioExtradata = async function(absStreamIndex) {
    if (absStreamIndex < 0 || absStreamIndex >= this.streams.length) return null;
    var streamPtr = await this.libav.AVFormatContext_streams_a(this.fmtCtx, absStreamIndex);
    var codecparPtr = await this.libav.AVStream_codecpar(streamPtr);
    var extraPtr = await this.libav.AVCodecParameters_extradata(codecparPtr);
    var extraSize = await this.libav.AVCodecParameters_extradata_size(codecparPtr);
    if (!extraPtr || extraSize <= 0) return null;
    return await this.libav.copyout_u8(extraPtr, extraSize);
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

// Abort any in-flight range request to unblock readPackets quickly during seek
ClientDemuxer.prototype.abortReads = function() {
    if (this._abortController) {
        this._abortController.abort();
        this._abortController = null;
    }
    // Invalidate read-ahead cache — seek will read from a new position
    this._readCache = null;
};

// Seek to a time in seconds. Seeks backward to nearest keyframe.
ClientDemuxer.prototype.seek = async function(timeSeconds) {
    var stream = this.streams[this.videoStreamIndex] || this.streams[0];
    if (!stream) return;

    // Convert seconds to stream timebase
    var tb = stream.time_base_num / stream.time_base_den;
    var timestamp = timeSeconds / tb;
    var parts = this.libav.f64toi64(timestamp);

    // av_seek_frame with AVSEEK_FLAG_BACKWARD seeks to the keyframe
    // at or before the target timestamp.
    await this.libav.av_seek_frame(
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
            SP.log.warn("Demuxer", "Cleanup error:", e);
        }
        this.libav.terminate();
        this.libav = null;
    }
};
