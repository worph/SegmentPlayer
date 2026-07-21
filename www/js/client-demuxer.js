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
    // Optional callbacks the player wires for buffering UI. Fire on the first
    // retry of a given range fetch and on its eventual success. Kept as plain
    // properties so layering stays clean — the demuxer never reaches into the
    // player's status code directly.
    this.onStallStart = null;
    this.onStallEnd = null;
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

    // Opening the container reads the file head through the block-reader device
    // — i.e. across the same no-bandwidth-guarantee Range byte source every
    // other read uses. A cold or slow first read (classically a Usenet stream
    // still materialising, but any transient short read qualifies) can make
    // libav's format probe see truncated data and fail avformat_open_input with
    // AVERROR_INVALIDDATA ("Invalid data found when processing input") even
    // though the bytes are a perfectly valid container. avformat_open_input_js
    // then returns a null context, we find zero streams, and the caller
    // (probeWithLibav) latches the player to "No decodable video stream" — with
    // no retry, unlike every other read in this pipeline which retries forever.
    // So retry the open itself on a fresh libav instance with capped backoff.
    var OPEN_ATTEMPTS = 6;
    var OPEN_DELAYS_MS = [200, 500, 1000, 2000, 4000];
    var lastErr = null;
    for (var attempt = 0; attempt < OPEN_ATTEMPTS; attempt++) {
        try {
            await this._openInput();
        } catch (e) {
            lastErr = e;
        }
        if (this.fmtCtx && this.streams.length > 0) { lastErr = null; break; }
        // No usable context (open failed or found no streams). Drop this
        // attempt's libav instance and, unless we're out of tries, wait a beat
        // for the byte source to warm before opening a fresh one.
        if (!lastErr) lastErr = new Error("avformat_open_input produced no streams");
        await this._teardownLibav();
        if (attempt < OPEN_ATTEMPTS - 1) {
            SP.log.warn("Demuxer", "Open failed (attempt " + (attempt + 1) + " of "
                + OPEN_ATTEMPTS + "), retrying:", (lastErr && (lastErr.message || lastErr)) || lastErr);
            await new Promise(function(resolve) {
                setTimeout(resolve, OPEN_DELAYS_MS[Math.min(attempt, OPEN_DELAYS_MS.length - 1)]);
            });
        }
    }
    if (!this.fmtCtx || this.streams.length === 0) {
        throw lastErr || new Error("Demuxer failed to open input");
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

// One open attempt: stand up a fresh libav instance + block-reader device and
// open the container, populating this.libav / this.fmtCtx / this.streams. On a
// transient probe failure avformat_open_input_js returns a null context, so
// this.fmtCtx stays 0 and this.streams stays empty; init() decides whether to
// accept the result or teardown and retry.
ClientDemuxer.prototype._openInput = async function() {
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

        // Real EOF — libav probes past the end of the file. Empty buffer is
        // the legitimate EOF signal here; this branch is the only place we
        // intentionally hand libav 0 bytes.
        if (position >= self.fileSize) {
            await self.libav.ff_block_reader_dev_send(name, position, new Uint8Array(0));
            return;
        }

        // Cache miss — fetch a larger chunk (read-ahead). Retries forever on
        // transient network failure: handing libav an empty buffer here would
        // be interpreted as EOF mid-file, sealing the MediaSource and freezing
        // playback. Honor the demuxer's `_abortController` for legitimate
        // cancellation (seek, cleanup).
        var fetchEnd = Math.min(position + READ_AHEAD, self.fileSize) - 1;
        var controller = new AbortController();
        self._abortController = controller;
        try {
            var result = await self._fetchRangeWithRetry(position, fetchEnd, controller.signal);

            if (result.aborted || result.eof) {
                // Aborted: caller is doing a seek; subsequent av_seek_frame
                // clears libav's EOF state, so 0 bytes here is safe.
                // EOF: server returned 416, real end-of-file.
                await self.libav.ff_block_reader_dev_send(name, position, new Uint8Array(0));
                return;
            }

            var buffer = result.buffer;
            self._bytesRead += buffer.length;
            self._readCache = { start: position, end: position + buffer.length, data: buffer };
            var sendSlice = buffer.subarray(0, Math.min(length, buffer.length));
            await self.libav.ff_block_reader_dev_send(name, position, sendSlice);
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
    // A null context means the format probe failed (see init()'s retry note).
    // Bail before touching options / find_stream_info — those would deref a
    // null ctx in the worker. init() tears down and retries.
    if (!fmtCtx) return;
    // Set probesize large enough for the demuxer to read audio frames and
    // determine channel layout (AAC with channelConfiguration=0 needs this)
    var probeSize = Math.min(this.fileSize, 50000000); // up to 50MB
    await this.libav.av_opt_set(fmtCtx, "probesize", String(probeSize), 0);
    await this.libav.av_opt_set(fmtCtx, "analyzeduration", "10000000", 0);
    // Don't seek to EOF just to nail down duration. When a container's declared
    // duration is missing, find_stream_info's estimate_timings_from_pts SEEKS TO
    // NEAR-EOF and reads packets to derive the end timestamp. On a still-
    // materialising NZB stream the file fills head→tail sequentially, so that
    // tail read blocks on bytes that land last — a needless multi-second stall
    // for a value we don't need to START linear playback (MKV clusters carry
    // their own timestamps; the position bar can fill in duration lazily). This
    // suppresses THAT particular EOF seek.
    //
    // NOTE — this is a PARTIAL of the intended "non-seekable start" fix. The
    // ideal is to present the byte source to libav as non-seekable (pb->seekable
    // = 0) so it never seeks to EOF at open at all. That is NOT achievable with
    // this vendored libav.js 6.8.8.0 build: it exposes no AVIOContext accessor
    // (so pb->seekable can't be flipped), and its only non-seekable device
    // (mkreaderdev) is a pure sequential pipe that would break random-access
    // seeking + the position-based read-ahead below. The DOMINANT open-time EOF
    // read for MKV is matroska read_header following its SeekHead to tail-located
    // Tags/Cues INSIDE avformat_open_input_js (already returned above, before we
    // can set any option) — not suppressible here. The real remedy for that tail
    // stall is server-side: prefetch the tail segments so the bytes are present
    // when libav asks. Keep this line regardless; it's cheap, safe insurance.
    await this.libav.av_opt_set(fmtCtx, "skip_estimate_duration_from_pts", "1", 0);
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
};

// Tear down the current libav instance between open attempts (or on give-up),
// resetting demuxer state for a fresh _openInput(). Safe to call repeatedly and
// when the open never completed (this.pkt / this.fmtCtx still zero).
ClientDemuxer.prototype._teardownLibav = async function() {
    if (this.libav) {
        try {
            if (this.pkt) { await this.libav.av_packet_free(this.pkt); this.pkt = 0; }
            if (this.fmtCtx) { await this.libav.avformat_close_input_js(this.fmtCtx); }
        } catch (e) { /* best-effort — the instance is about to be terminated */ }
        try { this.libav.terminate(); } catch (e) {}
        this.libav = null;
    }
    this.fmtCtx = 0;
    this.streams = [];
    this._readCache = null;
};

// Fetch one byte range with retry-forever + capped exponential backoff.
// Returns { buffer, aborted, eof }:
//   - On success: { buffer: Uint8Array, aborted: false, eof: false }
//   - On HTTP 416 (server says past EOF): { buffer: null, aborted: false, eof: true }
//   - On caller abort via signal: { buffer: null, aborted: true, eof: false }
// The retry loop is deliberately unbounded: a transient network failure must
// not propagate as fake-EOF to libav, and the player has no better recovery
// than waiting for the network to come back. Cancellation is the caller's
// job via `abortReads()`. Fires `onStallStart` the first time a retry is
// scheduled, and `onStallEnd` once a fetch eventually succeeds.
ClientDemuxer.prototype._fetchRangeWithRetry = async function(position, fetchEnd, signal) {
    var DELAYS_MS = [250, 500, 1000, 2000, 4000, 5000];
    var attempt = 0;
    var stalled = false;
    var range416 = 0;
    var max416 = (SP.config && SP.config.CLIENT_MAX_416_RETRIES) || 6;
    var self = this;

    while (true) {
        if (signal.aborted) return { buffer: null, aborted: true, eof: false };

        try {
            var response = await fetch(self.fileUrl, {
                headers: { "Range": "bytes=" + position + "-" + fetchEnd },
                signal: signal
            });

            if (response.status === 416) {
                // Genuine end-of-file never reaches here: onblockread
                // short-circuits to a 0-byte EOF whenever position >= fileSize,
                // *before* any fetch. So a 416 on an in-range request is an
                // anomaly — a flaky proxy, a momentary server view of a smaller
                // size, or a probed file_size a few bytes too large. Treat it as
                // a transient failure and retry (same backoff + buffering UI as
                // a network error). Only after several consecutive 416s do we
                // accept it as real EOF, which covers the too-large-size case
                // without sealing the stream on a one-off 416.
                range416++;
                if (range416 >= max416) {
                    if (stalled && typeof self.onStallEnd === "function") self.onStallEnd();
                    return { buffer: null, aborted: false, eof: true };
                }
                throw new Error("HTTP 416 (in-range, transient #" + range416 + ")");
            }
            if (!response.ok) {
                throw new Error("HTTP " + response.status);
            }

            var buffer = new Uint8Array(await response.arrayBuffer());
            range416 = 0;
            if (stalled && typeof self.onStallEnd === "function") self.onStallEnd();
            return { buffer: buffer, aborted: false, eof: false };
        } catch (err) {
            if (err.name === "AbortError") {
                return { buffer: null, aborted: true, eof: false };
            }

            if (!stalled) {
                stalled = true;
                if (typeof self.onStallStart === "function") self.onStallStart();
            }

            var delay = DELAYS_MS[Math.min(attempt, DELAYS_MS.length - 1)];
            attempt++;
            SP.log.warn("Demuxer", "Range request failed (attempt " + attempt
                + ", retry in " + delay + "ms):", err.message || err.name || err);

            await new Promise(function(resolve) {
                if (signal.aborted) return resolve();
                var t = setTimeout(function() {
                    signal.removeEventListener("abort", onAbort);
                    resolve();
                }, delay);
                function onAbort() { clearTimeout(t); resolve(); }
                signal.addEventListener("abort", onAbort, { once: true });
            });
        }
    }
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
