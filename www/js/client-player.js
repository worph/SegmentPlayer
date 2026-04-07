/* Client Player - MediaSource pipeline for client-side container transmux */

function ClientPlayer(videoElement) {
    this.video = videoElement;
    this.mediaSource = null;
    this.sourceBuffer = null; // single muxed video+audio SourceBuffer
    this.demuxer = null;
    this.muxer = null;
    this.audioReencoder = null; // non-null when audio needs re-encoding
    this.subtitleExtractor = null;
    this.running = false;
    this.probeData = null;
    this.currentAudioTrack = 0;
    this._seekHandler = null;
    this._pumpPromise = null;
    this._needsAudioReencode = false;
}

ClientPlayer.prototype._buildMimeType = function(audioTrackIdx) {
    var probeData = this.probeData;
    var videoCodecStr = mapVideoCodecToMSE(probeData.video.codec, probeData.video.profile,
        probeData.video.bit_depth, probeData.video.width, probeData.video.height);
    var audioTrack = probeData.audio[audioTrackIdx] || probeData.audio[0];
    var audioCodecStr = audioTrack ? mapAudioCodecToMSE(audioTrack.codec) : null;

    var needsReencode = false;
    if (audioTrack && typeof audioNeedsReencode === "function" && audioNeedsReencode(audioTrack.codec)) {
        needsReencode = true;
        audioCodecStr = "opus";
    }

    var mimeType;
    if (videoCodecStr && audioCodecStr) {
        mimeType = 'video/mp4; codecs="' + videoCodecStr + ', ' + audioCodecStr + '"';
    } else if (videoCodecStr) {
        mimeType = 'video/mp4; codecs="' + videoCodecStr + '"';
    } else {
        throw new Error("No supported video codec for MSE");
    }

    return { mimeType: mimeType, audioTrack: audioTrack, needsReencode: needsReencode };
};

// Helper: wait for a SourceBuffer to finish updating
function waitForSBUpdate(sb) {
    if (!sb.updating) return Promise.resolve();
    return new Promise(function(resolve) {
        sb.addEventListener("updateend", resolve, { once: true });
    });
}

// Helper: get how many seconds are buffered ahead of currentTime
function getBufferedAhead(video) {
    if (!video.buffered || video.buffered.length === 0) return 0;
    for (var i = 0; i < video.buffered.length; i++) {
        if (video.buffered.start(i) <= video.currentTime && video.buffered.end(i) > video.currentTime) {
            return video.buffered.end(i) - video.currentTime;
        }
    }
    return 0;
}

// Helper: sleep for ms
function sleep(ms) {
    return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

ClientPlayer.prototype.load = async function(filepath, probeData, audioTrackIdx) {
    this.cleanup();
    this.probeData = probeData;
    this.currentAudioTrack = audioTrackIdx || 0;
    this.stats = { bytesRead: 0, packetsRead: 0, startTime: Date.now(), framesDropped: 0 };

    var fileUrl = "/direct/" + encodeFilePath(filepath);

    // 1. Create MediaSource
    this.mediaSource = new MediaSource();
    this.video.src = URL.createObjectURL(this.mediaSource);

    await new Promise(function(resolve, reject) {
        var timeout = setTimeout(function() {
            reject(new Error("MediaSource sourceopen timeout after 10s"));
        }, 10000);
        this.mediaSource.addEventListener("sourceopen", function() {
            clearTimeout(timeout);
            resolve();
        }, { once: true });
    }.bind(this));

    // 2. Build MIME type for muxed video+audio fMP4
    var mimeInfo = this._buildMimeType(this.currentAudioTrack);
    var mimeType = mimeInfo.mimeType;
    var audioTrack = mimeInfo.audioTrack;
    this._needsAudioReencode = mimeInfo.needsReencode;

    if (this._needsAudioReencode) {
        console.log("[ClientPlayer] Audio codec", audioTrack.codec, "needs re-encoding to Opus");
        this.audioReencoder = new AudioReencoder();
        await this.audioReencoder.init(
            audioTrack.codec,
            audioTrack.sample_rate || 48000,
            audioTrack.channels || 2
        );
    }

    console.log("[ClientPlayer] MSE MIME:", mimeType,
        this._needsAudioReencode ? "(audio re-encode: " + audioTrack.codec + " → Opus)" : "");

    if (!MediaSource.isTypeSupported(mimeType)) {
        throw new Error("MediaSource does not support: " + mimeType);
    }

    this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
    this.sourceBuffer.mode = "segments";

    // 3. Initialize demuxer
    setStatus("Loading decoder...", "#4dabf7", true);
    this.demuxer = new ClientDemuxer(fileUrl, probeData.file_size);
    await this.demuxer.init();

    // Set audio track
    if (audioTrack) {
        var absIndex = this.demuxer.getAudioStreamIndex(this.currentAudioTrack);
        if (absIndex >= 0) this.demuxer.setAudioStream(absIndex);
    }

    // 4. Initialize muxer
    this.muxer = new ClientMuxer(this.demuxer.libav);

    await this.muxer.init(this._buildStreamInfos());

    // 5. Set duration
    if (probeData.duration > 0) {
        this.mediaSource.duration = probeData.duration;
    }

    // 6. Setup seek handler
    this._setupSeekHandler();

    // 7. Start pump loop
    this.running = true;
    this._pumpPromise = this._pumpLoop();

    console.log("[ClientPlayer] Started playback pipeline");
};

ClientPlayer.prototype._pumpLoop = async function() {
    var firstData = true;
    while (this.running) {
        try {
            // Check buffer level — pause if we have enough
            var ahead = getBufferedAhead(this.video);
            if (ahead > SP.config.CLIENT_BUFFER_AHEAD) {
                setStatus("Client-side", "#51cf66");
                await sleep(1000);
                continue;
            }

            if (firstData) {
                setStatus("Buffering...", "#ffd43b", true);
            }

            // Read a batch of packets
            var result = await this.demuxer.readPackets(512 * 1024); // 512KB batch

            if (!this.running) break; // check again after async read

            if (result.eof && Object.keys(result.packets).length === 0) {
                // End of file
                console.log("[ClientPlayer] End of stream");
                if (this.mediaSource && this.mediaSource.readyState === "open") {
                    // Wait for any pending appends
                    await waitForSBUpdate(this.sourceBuffer);
                    this.mediaSource.endOfStream();
                }
                break;
            }

            // Filter packets to only include our selected video + audio streams
            var filteredPackets = {};
            var videoIdx = this.demuxer.videoStreamIndex;
            var audioIdx = this.demuxer.audioStreamIndex;

            if (result.packets[videoIdx]) {
                filteredPackets[videoIdx] = result.packets[videoIdx];
            }
            if (audioIdx >= 0 && result.packets[audioIdx]) {
                // Re-encode audio if needed (AC3/DTS → Opus)
                if (this._needsAudioReencode && this.audioReencoder) {
                    var reencoded = await this.audioReencoder.processPackets(result.packets[audioIdx]);
                    if (reencoded.length > 0) {
                        filteredPackets[audioIdx] = reencoded;
                    }
                } else {
                    filteredPackets[audioIdx] = result.packets[audioIdx];
                }
            }

            if (Object.keys(filteredPackets).length === 0) {
                continue;
            }

            // Mux to fMP4
            var fmp4Data = await this.muxer.mux(filteredPackets);

            if (!this.running) break;

            if (fmp4Data && fmp4Data.length > 0) {
                // Append to SourceBuffer
                await waitForSBUpdate(this.sourceBuffer);
                this.sourceBuffer.appendBuffer(fmp4Data);
                await waitForSBUpdate(this.sourceBuffer);

                // Track stats
                this.stats.bytesRead += fmp4Data.length;
                this.stats.packetsRead++;

                if (firstData) {
                    firstData = false;
                    setStatus("Client-side", "#51cf66");
                }

                // If first few appends don't produce playable data, fail fast
                if (this.stats.packetsRead <= 3 && this.video.readyState === 0) {
                    this.stats._noDataAppends = (this.stats._noDataAppends || 0) + 1;
                    if (this.stats._noDataAppends >= 3) {
                        throw new Error("First 3 appends produced no playable data");
                    }
                }

                // Evict old data
                this._evictOldData();
            }

        } catch (err) {
            if (!this.running) break; // cleanup in progress
            this.stats.errors = (this.stats.errors || 0) + 1;
            console.error("[ClientPlayer] Pump error (" + this.stats.errors + "):", err);

            if (this.stats.errors >= 3) {
                console.error("[ClientPlayer] Too many errors, stopping pipeline");
                setStatus("Decode error", "#ff6b6b");
                break;
            }

            // Brief pause before retry
            await sleep(500);
        }
    }
};

ClientPlayer.prototype._evictOldData = function() {
    if (!this.sourceBuffer || this.sourceBuffer.updating) return;

    var behind = this.video.currentTime - SP.config.CLIENT_BUFFER_BEHIND;
    if (behind > 1) {
        try {
            this.sourceBuffer.remove(0, behind);
        } catch (e) {
            // Ignore — may be updating
        }
    }
};

ClientPlayer.prototype._setupSeekHandler = function() {
    var self = this;
    this._seekHandler = function() {
        if (!self.running) return;
        self._handleSeek(self.video.currentTime);
    };
    this.video.addEventListener("seeking", this._seekHandler);
};

ClientPlayer.prototype._handleSeek = async function(targetTime) {
    console.log("[ClientPlayer] Seeking to", targetTime.toFixed(2) + "s");
    setStatus("Seeking...", "#4dabf7", true);

    // 1. Stop current pump
    this.running = false;
    if (this._pumpPromise) {
        await this._pumpPromise.catch(function() {});
    }

    try {
        // 2. Abort and clear SourceBuffer safely
        try {
            if (this.sourceBuffer.updating) {
                this.sourceBuffer.abort();
            }
        } catch (e) { /* abort may throw if not in correct state */ }

        // Small delay to let abort settle
        await sleep(50);

        try {
            await waitForSBUpdate(this.sourceBuffer);
            this.sourceBuffer.remove(0, Infinity);
            await waitForSBUpdate(this.sourceBuffer);
        } catch (e) {
            console.warn("[ClientPlayer] SourceBuffer clear error (non-fatal):", e);
        }

        // 3. Reset timestamp offset
        try { this.sourceBuffer.timestampOffset = 0; } catch (e) {}

        // 4. Seek demuxer to nearest keyframe before target
        await this.demuxer.seek(targetTime);

        // 5. Re-init muxer (need fresh fMP4 init segment after seek)
        await this.muxer.destroy();
        this.muxer = new ClientMuxer(this.demuxer.libav);
        await this.muxer.init(this._buildStreamInfos());

        // 6. Restart pump
        this.running = true;
        this._pumpPromise = this._pumpLoop();

    } catch (err) {
        console.error("[ClientPlayer] Seek error:", err);
        setStatus("Seek error", "#ff6b6b");
        // Try to recover by restarting the pump from wherever the demuxer is
        try {
            this.running = true;
            this._pumpPromise = this._pumpLoop();
        } catch (e) {}
    }
};

ClientPlayer.prototype.switchAudioTrack = async function(audioTrackIdx, resumeTime) {
    console.log("[ClientPlayer] Switching to audio track", audioTrackIdx);

    // Stop pump
    this.running = false;
    if (this._pumpPromise) {
        await this._pumpPromise.catch(function() {});
    }

    try {
        // Update audio stream in demuxer
        var absIndex = this.demuxer.getAudioStreamIndex(audioTrackIdx);
        if (absIndex < 0) {
            console.warn("[ClientPlayer] Audio track not found:", audioTrackIdx);
            return;
        }
        this.demuxer.setAudioStream(absIndex);
        this.currentAudioTrack = audioTrackIdx;

        // Remove old SourceBuffer
        if (this.sourceBuffer) {
            if (this.sourceBuffer.updating) this.sourceBuffer.abort();
            await waitForSBUpdate(this.sourceBuffer);
            this.mediaSource.removeSourceBuffer(this.sourceBuffer);
        }

        // Create new SourceBuffer with updated audio codec
        var mimeInfo = this._buildMimeType(audioTrackIdx);

        this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeInfo.mimeType);
        this.sourceBuffer.mode = "segments";

        // Update audio re-encoder if needed
        if (this.audioReencoder) {
            this.audioReencoder.destroy();
            this.audioReencoder = null;
        }
        var newAudioTrackInfo = this.probeData.audio[audioTrackIdx];
        if (newAudioTrackInfo && typeof audioNeedsReencode === "function" && audioNeedsReencode(newAudioTrackInfo.codec)) {
            this._needsAudioReencode = true;
            this.audioReencoder = new AudioReencoder();
            await this.audioReencoder.init(
                newAudioTrackInfo.codec,
                newAudioTrackInfo.sample_rate || 48000,
                newAudioTrackInfo.channels || 2
            );
        } else {
            this._needsAudioReencode = false;
        }

        // Re-init muxer with new audio stream
        await this.muxer.destroy();
        this.muxer = new ClientMuxer(this.demuxer.libav);
        await this.muxer.init(this._buildStreamInfos());

        // Seek demuxer to current time
        await this.demuxer.seek(resumeTime);

        // Set duration again (gets lost when SourceBuffer changes)
        if (this.probeData.duration > 0 && this.mediaSource.readyState === "open") {
            this.mediaSource.duration = this.probeData.duration;
        }

        // Restart pump
        this.running = true;
        this.video.currentTime = resumeTime;
        this._pumpPromise = this._pumpLoop();

    } catch (err) {
        console.error("[ClientPlayer] Audio switch error:", err);
        setStatus("Audio switch error", "#ff6b6b");
    }
};

// Extract a subtitle track and attach to the video element
ClientPlayer.prototype.loadSubtitleTrack = async function(subTrackIndex) {
    if (!this.demuxer) return;

    // Find the subtitle stream info from probe data
    var subInfo = this.probeData.subtitles[subTrackIndex];
    if (!subInfo) {
        console.warn("[ClientPlayer] Subtitle track not found:", subTrackIndex);
        return;
    }

    // Find the absolute stream index for this subtitle track
    var absIndex = -1;
    var subCount = 0;
    for (var i = 0; i < this.demuxer.streams.length; i++) {
        var s = this.demuxer.streams[i];
        if (s.codec_type === this.demuxer.libav.AVMEDIA_TYPE_SUBTITLE) {
            if (subCount === subTrackIndex) {
                absIndex = s.index;
                break;
            }
            subCount++;
        }
    }

    if (absIndex < 0) {
        console.warn("[ClientPlayer] Subtitle stream not found in demuxer");
        return;
    }

    // We need a separate demuxer instance for subtitle extraction
    // to avoid disrupting the main playback demuxer position
    if (!this.subtitleExtractor) {
        this.subtitleExtractor = new ClientSubtitleExtractor(this.demuxer);
    }

    // Pause pump loop during extraction
    var wasRunning = this.running;
    this.running = false;
    if (this._pumpPromise) {
        await this._pumpPromise.catch(function() {});
    }

    try {
        var vtt = await this.subtitleExtractor.extract(absIndex, subInfo.codec);
        if (vtt) {
            var label = subInfo.title || subInfo.language || "Track " + (subTrackIndex + 1);
            attachVTTToVideo(this.video, vtt, label);
        }
    } catch (err) {
        console.error("[ClientPlayer] Subtitle extraction error:", err);
    }

    // Seek back to current position and resume
    if (wasRunning) {
        var currentTime = this.video.currentTime;
        await this.demuxer.seek(currentTime);

        // Re-init muxer after seek
        await this.muxer.destroy();
        this.muxer = new ClientMuxer(this.demuxer.libav);
        await this.muxer.init(this._buildStreamInfos());

        this.running = true;
        this._pumpPromise = this._pumpLoop();
    }
};

// Build stream infos array for muxer initialization
ClientPlayer.prototype._buildStreamInfos = function() {
    var streamInfos = [];
    var videoStream = this.demuxer.streams[this.demuxer.videoStreamIndex];
    if (videoStream) {
        streamInfos.push({
            inputIndex: videoStream.index,
            codecpar: videoStream.codecpar,
            time_base_num: videoStream.time_base_num,
            time_base_den: videoStream.time_base_den
        });
    }
    var audioStreamIdx = this.demuxer.audioStreamIndex;
    if (audioStreamIdx >= 0) {
        var audioStream = this.demuxer.streams[audioStreamIdx];
        if (audioStream) {
            streamInfos.push({
                inputIndex: audioStream.index,
                codecpar: audioStream.codecpar,
                time_base_num: audioStream.time_base_num,
                time_base_den: audioStream.time_base_den
            });
        }
    }
    return streamInfos;
};

ClientPlayer.prototype.cleanup = function() {
    this.running = false;

    if (this._seekHandler) {
        this.video.removeEventListener("seeking", this._seekHandler);
        this._seekHandler = null;
    }

    if (this.audioReencoder) {
        this.audioReencoder.destroy();
        this.audioReencoder = null;
    }

    if (this.muxer) {
        this.muxer.destroy().catch(function() {});
        this.muxer = null;
    }

    if (this.demuxer) {
        this.demuxer.destroy().catch(function() {});
        this.demuxer = null;
    }

    if (this.mediaSource && this.mediaSource.readyState === "open") {
        try {
            this.mediaSource.endOfStream();
        } catch (e) {}
    }
    this.mediaSource = null;
    this.sourceBuffer = null;
};
