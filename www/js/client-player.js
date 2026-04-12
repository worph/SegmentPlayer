/* Client Player - MediaSource pipeline for client-side container transmux */

function ClientPlayer(videoElement) {
    this.video = videoElement;
    this.mediaSource = null;
    this.sourceBuffer = null; // single muxed video+audio SourceBuffer
    this.demuxer = null;
    this.muxer = null;
    this.audioReencoder = null; // non-null when audio needs re-encoding
    this.running = false;
    this.probeData = null;
    this.currentAudioTrack = 0;
    this._seekHandler = null;
    this._seekGeneration = 0;
    this._pumpPromise = null;
    this._needsAudioReencode = false;
    // Piggyback subtitle collection: accumulate subtitle packets during playback
    this._subtitlePackets = {};    // streamIndex -> Packet[]
    this._subtitleStreamIndices = []; // absolute indices of subtitle streams
    this._activeSubtitleTrack = -1;   // subtitle track index currently displayed (-1 = none)
    this._activeSubtitleAbsIdx = -1;  // absolute stream index of active subtitle
    this._subtitleUpdateCounter = 0;  // throttle progressive VTT rebuilds
}

ClientPlayer.prototype._buildMimeType = function(audioTrackIdx) {
    var probeData = this.probeData;
    var videoCodecStr = mapVideoCodecToMSE(probeData.video.codec, probeData.video.profile,
        probeData.video.bit_depth, probeData.video.width, probeData.video.height);

    // Skip audio in MSE — libav.js range-request I/O often can't fully probe
    // audio codec parameters (channels, frame_size), producing invalid fMP4 audio.
    // Video plays via MSE, audio plays via a separate direct <audio> element.
    var mimeType = 'video/mp4; codecs="' + videoCodecStr + '"';
    return { mimeType: mimeType, audioTrack: null, needsReencode: false, videoOnly: true };
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

    // 2. Build MIME type for video-only fMP4
    var mimeInfo = this._buildMimeType(this.currentAudioTrack);
    var mimeType = mimeInfo.mimeType;
    this._needsAudioReencode = false;

    console.log("[ClientPlayer] MSE MIME:", mimeType, "(video-only)");

    if (!MediaSource.isTypeSupported(mimeType)) {
        throw new Error("MediaSource does not support: " + mimeType);
    }

    this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
    this.sourceBuffer.mode = "segments";

    // 3. Initialize demuxer
    setStatus("Loading decoder...", "#4dabf7", true);
    this.demuxer = new ClientDemuxer(fileUrl, probeData.file_size);
    await this.demuxer.init();

    // Video-only: disable audio stream in demuxer
    this.demuxer.audioStreamIndex = -1;

    // Discover subtitle stream indices for piggyback collection
    this._subtitleStreamIndices = [];
    this._subtitlePackets = {};
    for (var i = 0; i < this.demuxer.streams.length; i++) {
        var s = this.demuxer.streams[i];
        if (s.codec_type === this.demuxer.libav.AVMEDIA_TYPE_SUBTITLE) {
            this._subtitleStreamIndices.push(s.index);
        }
    }

    // 4. Initialize muxer (video-only)
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

    // 8. Start direct audio playback alongside MSE video
    this._startDirectAudio(filepath, probeData);

    console.log("[ClientPlayer] Started video-only pipeline with direct audio");
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
            var result = await this.demuxer.readPackets(4 * 1024 * 1024); // 4MB batch

            if (!this.running) break; // check again after async read

            if (result.eof && Object.keys(result.packets).length === 0) {
                // End of file
                console.log("[ClientPlayer] End of stream");
                if (this.mediaSource && this.mediaSource.readyState === "open") {
                    // Drain the muxer's lookahead tail (up to reorderDepth
                    // trailing frames) and write the trailer. Without this,
                    // the last fragment of playback would be truncated.
                    try {
                        var tail = await this.muxer.flush();
                        if (tail && tail.length > 0) {
                            await waitForSBUpdate(this.sourceBuffer);
                            this.sourceBuffer.appendBuffer(tail);
                            await waitForSBUpdate(this.sourceBuffer);
                        }
                    } catch (e) {
                        console.warn("[ClientPlayer] Flush-on-EOF error:", e);
                    }
                    await waitForSBUpdate(this.sourceBuffer);
                    this.mediaSource.endOfStream();
                }
                break;
            }

            // Collect subtitle packets from this batch (piggyback extraction)
            this._collectSubtitlePackets(result.packets);

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
                    // Trigger autoplay — the initial play() call in player.js
                    // may have been rejected because no data was available yet
                    if (this.video.paused) {
                        this.video.play().catch(function() {});
                    }
                }

                // If first few appends don't produce playable data, fail fast.
                // Skip this check while the muxer is still measuring reorder
                // depth — during warmup, the muxer legitimately returns null
                // for the first few batches, which is not a pipeline stall.
                var stillWarmingUp = this.muxer && typeof this.muxer.isWarmingUp === "function"
                    && this.muxer.isWarmingUp();
                if (!stillWarmingUp && this.stats.packetsRead <= 3 && this.video.readyState === 0) {
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
    this._seekGeneration = 0;
    this._seekHandler = function() {
        // Always accept seeks — use generation counter to handle re-entrancy
        // instead of dropping seeks when running=false (which loses the user's
        // final seekbar position during an in-progress seek)
        self._handleSeek(self.video.currentTime);
    };
    this.video.addEventListener("seeking", this._seekHandler);
};

ClientPlayer.prototype._handleSeek = async function(targetTime) {
    // Generation counter prevents stale seeks from completing when a newer
    // seek has been requested (e.g., user dragging the seekbar rapidly)
    var generation = ++this._seekGeneration;

    console.log("[ClientPlayer] Seeking to", targetTime.toFixed(2) + "s (gen " + generation + ")");
    setStatus("Seeking...", "#4dabf7", true);

    // 1. Stop current pump and abort in-flight range requests so the
    //    pump loop exits immediately instead of waiting for network I/O
    this.running = false;
    if (this.demuxer) {
        this.demuxer.abortReads();
    }
    if (this._pumpPromise) {
        await this._pumpPromise.catch(function() {});
    }

    // A newer seek came in while we were waiting — abandon this one
    if (generation !== this._seekGeneration) return;

    try {
        // 2. Abort and clear SourceBuffer safely
        try {
            if (this.sourceBuffer.updating) {
                this.sourceBuffer.abort();
            }
        } catch (e) { /* abort may throw if not in correct state */ }

        try {
            await waitForSBUpdate(this.sourceBuffer);
            this.sourceBuffer.remove(0, Infinity);
            await waitForSBUpdate(this.sourceBuffer);
        } catch (e) {
            console.warn("[ClientPlayer] SourceBuffer clear error (non-fatal):", e);
        }

        if (generation !== this._seekGeneration) return;

        // 3. Reset timestamp offset
        try { this.sourceBuffer.timestampOffset = 0; } catch (e) {}

        // 4. Seek demuxer to nearest keyframe before target
        await this.demuxer.seek(targetTime);

        if (generation !== this._seekGeneration) return;

        // 5. Re-init muxer (need fresh fMP4 init segment after seek)
        await this.muxer.destroy();
        this.muxer = new ClientMuxer(this.demuxer.libav);
        await this.muxer.init(this._buildStreamInfos());

        if (generation !== this._seekGeneration) return;

        // 6. Restart pump
        this.running = true;
        this._pumpPromise = this._pumpLoop();

    } catch (err) {
        if (generation !== this._seekGeneration) return;
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
    this.currentAudioTrack = audioTrackIdx;

    if (!this._audioEl) {
        console.warn("[ClientPlayer] No direct audio element — audio switch not supported");
        return;
    }

    // With the /api/audio/ endpoint, each track is a separate extracted mp4.
    // Rebuild the <audio> src with the new track index and resume playback.
    var audio = this._audioEl;
    var filepath = SP.state.currentFile || SP.state.currentPath;
    var currentTime = this.video.currentTime;
    var wasPlaying = !this.video.paused;

    audio.pause();
    audio.src = this._audioTrackUrl(filepath, audioTrackIdx);
    audio.currentTime = currentTime;
    if (wasPlaying) {
        audio.play().catch(function() {});
    }
    console.log("[ClientPlayer] Reloaded audio element for track", audioTrackIdx);
};

// Collect subtitle packets from a readPackets result into the accumulator.
// Called from _pumpLoop on every batch — zero-cost when no subtitle streams exist.
ClientPlayer.prototype._collectSubtitlePackets = function(packets) {
    var dominated = false;
    for (var i = 0; i < this._subtitleStreamIndices.length; i++) {
        var subIdx = this._subtitleStreamIndices[i];
        if (packets[subIdx] && packets[subIdx].length > 0) {
            if (!this._subtitlePackets[subIdx]) {
                this._subtitlePackets[subIdx] = [];
            }
            for (var j = 0; j < packets[subIdx].length; j++) {
                this._subtitlePackets[subIdx].push(packets[subIdx][j]);
            }
            // If this is the active subtitle track, schedule a progressive update
            if (subIdx === this._activeSubtitleAbsIdx) {
                dominated = true;
            }
        }
    }
    if (dominated) {
        this._subtitleUpdateCounter++;
        if (this._subtitleUpdateCounter >= 5) {
            this._subtitleUpdateCounter = 0;
            this._refreshActiveSubtitle();
        }
    }
};

// Rebuild and re-attach the VTT for the currently active subtitle track
ClientPlayer.prototype._refreshActiveSubtitle = function() {
    if (this._activeSubtitleTrack < 0 || this._activeSubtitleAbsIdx < 0) return;

    var subInfo = this.probeData.subtitles[this._activeSubtitleTrack];
    if (!subInfo) return;

    var pkts = this._subtitlePackets[this._activeSubtitleAbsIdx];
    if (!pkts || pkts.length === 0) return;

    var stream = this.demuxer.streams[this._activeSubtitleAbsIdx];
    var timeBase = 1;
    if (stream && stream.time_base_num && stream.time_base_den) {
        timeBase = stream.time_base_num / stream.time_base_den;
    }

    var vtt = buildVTTFromPackets(pkts, timeBase, subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (this._activeSubtitleTrack + 1);
    attachVTTToVideo(this.video, vtt, label);
};

// Select a subtitle track from piggybacked packets.
// Shows whatever has been collected so far; updates progressively as playback continues.
ClientPlayer.prototype.loadSubtitleTrack = async function(subTrackIndex) {
    if (!this.demuxer) return;

    var subInfo = this.probeData.subtitles[subTrackIndex];
    if (!subInfo) {
        console.warn("[ClientPlayer] Subtitle track not found:", subTrackIndex);
        return;
    }

    // Find the absolute stream index
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

    // Mark this track as active for progressive updates
    this._activeSubtitleTrack = subTrackIndex;
    this._activeSubtitleAbsIdx = absIndex;
    this._subtitleUpdateCounter = 0;

    var pkts = this._subtitlePackets[absIndex] || [];
    console.log("[ClientPlayer] Showing subtitle track", subTrackIndex, "with", pkts.length, "packets collected so far");

    var stream = this.demuxer.streams[absIndex];
    var timeBase = 1;
    if (stream && stream.time_base_num && stream.time_base_den) {
        timeBase = stream.time_base_num / stream.time_base_den;
    }

    var vtt = buildVTTFromPackets(pkts, timeBase, subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (subTrackIndex + 1);
    attachVTTToVideo(this.video, vtt, label);
};

// Build stream infos array for muxer initialization (video-only)
ClientPlayer.prototype._buildStreamInfos = function() {
    var streamInfos = [];
    var videoStream = this.demuxer.streams[this.demuxer.videoStreamIndex];
    // Codec name comes from the original probe data — used by the muxer to
    // apply HEVC-specific open-GOP handling (clearing AV_PKT_FLAG_KEY on CRA
    // packets so MSE doesn't flag "sync sample with later PTS than dependent
    // non-sync sample").
    var videoCodec = this.probeData && this.probeData.video && this.probeData.video.codec;
    if (videoStream) {
        streamInfos.push({
            inputIndex: videoStream.index,
            codecpar: videoStream.codecpar,
            time_base_num: videoStream.time_base_num,
            time_base_den: videoStream.time_base_den,
            codecName: videoCodec
        });
    }
    return streamInfos;
};

// Build the audio-only mp4 URL for a given file + audio track.
// Uses the /api/audio endpoint which extracts audio-only (no video track) so
// Chrome doesn't allocate a duplicate D3D11VideoDecoder on the <audio> element.
ClientPlayer.prototype._audioTrackUrl = function(filepath, audioTrackIdx) {
    return "/api/audio/" + encodeFilePath(filepath) + "/track/" + (audioTrackIdx || 0) + ".m4a";
};

// Play audio directly via a hidden <audio> element synced to the MSE video
ClientPlayer.prototype._startDirectAudio = function(filepath, probeData) {
    if (!probeData.audio || probeData.audio.length === 0) return;

    this._audioEl = document.createElement("audio");
    this._audioEl.src = this._audioTrackUrl(filepath, this.currentAudioTrack);
    this._audioEl.preload = "auto";
    this._audioEl.style.display = "none";
    document.body.appendChild(this._audioEl);

    var video = this.video;
    var audio = this._audioEl;

    // Sync audio to video on play/pause/seek
    video.addEventListener("play", function() { audio.play().catch(function(){}); });
    video.addEventListener("pause", function() { audio.pause(); });
    video.addEventListener("seeking", function() { audio.currentTime = video.currentTime; });
    video.addEventListener("volumechange", function() {
        audio.volume = video.volume;
        audio.muted = video.muted;
    });

    // Start audio when video starts
    if (!video.paused) {
        audio.currentTime = video.currentTime;
        audio.play().catch(function(){});
    }
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

    if (this._audioEl) {
        this._audioEl.pause();
        this._audioEl.removeAttribute("src");
        // Calling .load() after stripping src forces the element to abort any
        // in-flight network activity for the previous file's audio track.
        try { this._audioEl.load(); } catch (e) {}
        if (this._audioEl.parentNode) this._audioEl.parentNode.removeChild(this._audioEl);
        this._audioEl = null;
    }

    this.currentAudioTrack = 0;
};
