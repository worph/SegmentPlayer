/* Client Player - MediaSource pipeline for client-side container transmux.
 *
 * Single MediaSource, single SourceBuffer containing muxed video + Opus audio.
 * Both tracks share one MediaSource timeline — the browser guarantees A/V
 * sync. Audio is always decoded via WebCodecs AudioDecoder and re-encoded to
 * Opus via WebCodecs AudioEncoder, then fed as a second input stream to the
 * ClientMuxer alongside the video. The muxer produces interleaved fMP4 with
 * both tracks in one init segment + media fragments.
 */

function ClientPlayer(videoElement) {
    this.video = videoElement;
    this.mediaSource = null;
    this.sourceBuffer = null;
    this.demuxer = null;
    this.muxer = null;
    this.audioReencoder = null;
    this.running = false;
    this.probeData = null;
    this.currentAudioTrack = 0;
    this._audioAbsIdx = -1;    // absolute stream index for the active audio track
    this._seekHandler = null;
    this._seekGeneration = 0;
    this._pumpPromise = null;
    // Piggyback subtitle collection: accumulate subtitle packets during playback
    this._subtitlePackets = {};       // streamIndex -> Packet[]
    this._subtitleStreamIndices = []; // absolute indices of subtitle streams
    this._activeSubtitleTrack = -1;   // subtitle track index currently displayed (-1 = none)
    this._activeSubtitleAbsIdx = -1;  // absolute stream index of active subtitle
    this._subtitleUpdateCounter = 0;  // throttle progressive VTT rebuilds
    this._primerBatch = null;         // first post-seek packet batch (peeked to derive muxer base)
}

// Combined video+audio MIME for the single muxed SourceBuffer. Audio half is
// always "opus" because we always re-encode to Opus regardless of source.
ClientPlayer.prototype._buildMimeType = function() {
    var pd = this.probeData;
    var v = mapVideoCodecToMSE(pd.video.codec, pd.video.profile,
        pd.video.bit_depth, pd.video.width, pd.video.height);
    if (!v) throw new Error("Cannot map video codec: " + pd.video.codec);
    if (pd.audio && pd.audio.length > 0) {
        return 'video/mp4; codecs="' + v + ',opus"';
    }
    return 'video/mp4; codecs="' + v + '"';
};

function waitForSBUpdate(sb) {
    if (!sb.updating) return Promise.resolve();
    return new Promise(function(resolve) {
        sb.addEventListener("updateend", resolve, { once: true });
    });
}

function getBufferedAhead(video) {
    if (!video.buffered || video.buffered.length === 0) return 0;
    for (var i = 0; i < video.buffered.length; i++) {
        if (video.buffered.start(i) <= video.currentTime && video.buffered.end(i) > video.currentTime) {
            return video.buffered.end(i) - video.currentTime;
        }
    }
    return 0;
}

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

    // 2. Build combined video+Opus MIME
    var mimeType = this._buildMimeType();
    SP.log.debug("ClientPlayer", "MSE MIME:", mimeType);
    if (!MediaSource.isTypeSupported(mimeType)) {
        throw new Error("MediaSource does not support: " + mimeType);
    }
    this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
    this.sourceBuffer.mode = "segments";

    // 3. Demuxer
    setStatus("Loading decoder...", "#4dabf7", true);
    this.demuxer = new ClientDemuxer(fileUrl, probeData.file_size);
    await this.demuxer.init();

    // Discover subtitle streams (piggyback collection during playback)
    this._subtitleStreamIndices = [];
    this._subtitlePackets = {};
    for (var i = 0; i < this.demuxer.streams.length; i++) {
        var s = this.demuxer.streams[i];
        if (s.codec_type === this.demuxer.libav.AVMEDIA_TYPE_SUBTITLE) {
            this._subtitleStreamIndices.push(s.index);
        }
    }

    // 4. Resolve active audio stream and init the re-encoder
    if (probeData.audio && probeData.audio.length > 0) {
        await this.demuxer.fixAudioParams(probeData);
        this._audioAbsIdx = this.demuxer.getAudioStreamIndex(this.currentAudioTrack);
        if (this._audioAbsIdx < 0) {
            throw new Error("Audio track " + this.currentAudioTrack + " not found in container");
        }
        this.demuxer.setAudioStream(this._audioAbsIdx);
        var audioInfo = probeData.audio[this.currentAudioTrack];
        var extradata = await this.demuxer.getAudioExtradata(this._audioAbsIdx);
        var audioStream = this.demuxer.streams[this._audioAbsIdx];
        this.audioReencoder = new AudioReencoder();
        await this.audioReencoder.init(
            audioInfo.codec,
            audioInfo.sample_rate || 48000,
            audioInfo.channels || 2,
            extradata,
            { num: audioStream.time_base_num, den: audioStream.time_base_den }
        );
    } else {
        this._audioAbsIdx = -1;
        this.demuxer.audioStreamIndex = -1;
    }

    // 5. Muxer (video + Opus audio if present)
    this.muxer = new ClientMuxer(this.demuxer.libav);
    await this.muxer.init(await this._buildStreamInfos());

    // 6. Duration
    if (probeData.duration > 0) {
        this.mediaSource.duration = probeData.duration;
    }

    // 7. Seek handler
    this._setupSeekHandler();

    // 8. Pump
    this.running = true;
    this._pumpPromise = this._pumpLoop();

    SP.log.debug("ClientPlayer", "Started unified video+audio pipeline");
};

ClientPlayer.prototype._pumpLoop = async function() {
    var firstData = true;
    while (this.running) {
        try {
            var ahead = getBufferedAhead(this.video);
            if (ahead > SP.config.CLIENT_BUFFER_AHEAD) {
                setStatus("Client-side", "#51cf66");
                await sleep(1000);
                continue;
            }

            if (firstData) {
                setStatus("Buffering...", "#ffd43b", true);
            }

            // Consume the primer batch from _restartPipeline first — those
            // packets were already read from the demuxer to derive the muxer
            // base PTS, so re-reading here would skip past them.
            var result;
            if (this._primerBatch) {
                result = this._primerBatch;
                this._primerBatch = null;
            } else {
                result = await this.demuxer.readPackets(4 * 1024 * 1024);
            }

            if (!this.running) break;

            if (result.eof && Object.keys(result.packets).length === 0) {
                SP.log.debug("ClientPlayer", "End of stream");
                if (this.mediaSource && this.mediaSource.readyState === "open") {
                    try {
                        // Drain the audio re-encoder first so tail Opus frames
                        // make it into the final mux call alongside the
                        // muxer's lookahead tail.
                        if (this.audioReencoder) {
                            var tailAudio = await this.audioReencoder.flush();
                            if (tailAudio.length > 0 && this._audioAbsIdx >= 0) {
                                var finalBatch = {};
                                finalBatch[this._audioAbsIdx] = tailAudio;
                                var finalFrag = await this.muxer.mux(finalBatch);
                                if (finalFrag && finalFrag.length > 0) {
                                    await waitForSBUpdate(this.sourceBuffer);
                                    this.sourceBuffer.appendBuffer(finalFrag);
                                    await waitForSBUpdate(this.sourceBuffer);
                                }
                            }
                        }
                        var tail = await this.muxer.flush();
                        if (tail && tail.length > 0) {
                            await waitForSBUpdate(this.sourceBuffer);
                            this.sourceBuffer.appendBuffer(tail);
                            await waitForSBUpdate(this.sourceBuffer);
                        }
                    } catch (e) {
                        SP.log.warn("ClientPlayer", "Flush-on-EOF error:", e);
                    }
                    await waitForSBUpdate(this.sourceBuffer);
                    this.mediaSource.endOfStream();
                }
                break;
            }

            // Collect subtitle packets from this batch
            this._collectSubtitlePackets(result.packets);

            // Filter to video + (re-encoded) audio
            var filteredPackets = {};
            var videoIdx = this.demuxer.videoStreamIndex;
            var audioIdx = this._audioAbsIdx;

            if (result.packets[videoIdx]) {
                filteredPackets[videoIdx] = result.packets[videoIdx];
            }
            if (audioIdx >= 0 && this.audioReencoder) {
                if (result.packets[audioIdx] && result.packets[audioIdx].length > 0) {
                    this.audioReencoder.submitPackets(result.packets[audioIdx]);
                }
                var reencoded = this.audioReencoder.drainPackets();
                if (reencoded.length > 0) {
                    filteredPackets[audioIdx] = reencoded;
                }
            }

            if (Object.keys(filteredPackets).length === 0) {
                continue;
            }

            var fmp4Data = await this.muxer.mux(filteredPackets);

            if (!this.running) break;

            if (fmp4Data && fmp4Data.length > 0) {
                await waitForSBUpdate(this.sourceBuffer);
                this.sourceBuffer.appendBuffer(fmp4Data);
                await waitForSBUpdate(this.sourceBuffer);

                this.stats.bytesRead += fmp4Data.length;
                this.stats.packetsRead++;

                if (firstData) {
                    firstData = false;
                    setStatus("Client-side", "#51cf66");
                    if (this.video.paused) {
                        this.video.play().catch(function() {});
                    }
                }

                // If first few appends don't produce playable data, fail fast.
                // Skip while the muxer is still measuring reorder depth.
                var stillWarmingUp = this.muxer && typeof this.muxer.isWarmingUp === "function"
                    && this.muxer.isWarmingUp();
                if (!stillWarmingUp && this.stats.packetsRead <= 3 && this.video.readyState === 0) {
                    this.stats._noDataAppends = (this.stats._noDataAppends || 0) + 1;
                    if (this.stats._noDataAppends >= 3) {
                        throw new Error("First 3 appends produced no playable data");
                    }
                }

                this._evictOldData();
            }

        } catch (err) {
            if (!this.running) break;
            this.stats.errors = (this.stats.errors || 0) + 1;
            SP.log.error("ClientPlayer", "Pump error (" + this.stats.errors + "):", err);

            if (this.stats.errors >= 3) {
                SP.log.error("ClientPlayer", "Too many errors, stopping pipeline");
                setStatus("Decode error", "#ff6b6b");
                break;
            }
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
        } catch (e) { /* may be updating */ }
    }
};

ClientPlayer.prototype._setupSeekHandler = function() {
    var self = this;
    this._seekGeneration = 0;
    this._seekHandler = function() {
        // Always accept seeks — generation counter handles rapid re-entry.
        self._restartPipeline(self.video.currentTime);
    };
    this.video.addEventListener("seeking", this._seekHandler);
};

// Teardown + rebuild the decode/mux pipeline at the given resume time. Shared
// between seek (newAudioAbsIdx=null — keep current audio track) and track
// switch (newAudioAbsIdx=<new abs idx>).
//
// Because the output Opus codecpar is deterministic (always 48 kHz stereo),
// the muxer re-init produces a byte-identical init segment across both flows.
// MSE treats an identical init segment as a no-op, so the SourceBuffer stays.
ClientPlayer.prototype._restartPipeline = async function(resumeTime, newAudioAbsIdx) {
    var generation = ++this._seekGeneration;

    SP.log.debug("ClientPlayer", "Restarting pipeline at", resumeTime.toFixed(2) + "s",
        "(gen " + generation + ")");
    setStatus("Seeking...", "#4dabf7", true);

    // 1. Stop the pump and abort any in-flight range requests.
    this.running = false;
    if (this.demuxer) this.demuxer.abortReads();
    if (this._pumpPromise) await this._pumpPromise.catch(function() {});

    if (generation !== this._seekGeneration) return;

    try {
        // 2. Clear the SourceBuffer.
        try {
            if (this.sourceBuffer.updating) this.sourceBuffer.abort();
        } catch (e) { /* abort may throw in wrong state */ }

        try {
            await waitForSBUpdate(this.sourceBuffer);
            this.sourceBuffer.remove(0, Infinity);
            await waitForSBUpdate(this.sourceBuffer);
        } catch (e) {
            SP.log.warn("ClientPlayer", "SourceBuffer clear error (non-fatal):", e);
        }

        if (generation !== this._seekGeneration) return;

        // 3. Optionally swap the active audio track on the demuxer side.
        if (typeof newAudioAbsIdx === "number" && newAudioAbsIdx >= 0
                && newAudioAbsIdx !== this._audioAbsIdx) {
            this._audioAbsIdx = newAudioAbsIdx;
            this.demuxer.setAudioStream(newAudioAbsIdx);
        }

        // 4. Seek demuxer to nearest keyframe ≤ target.
        await this.demuxer.seek(resumeTime);

        if (generation !== this._seekGeneration) return;

        // 4b. Peek the first batch of packets so we can derive the muxer's
        //     actual normalization base. AVSEEK_FLAG_BACKWARD lands on the
        //     keyframe ≤ resumeTime, so the first video PTS is typically
        //     earlier than resumeTime — and the first AAC frame can lie at a
        //     third position again. The muxer normalizes its first packet's
        //     DTS to 0, so we set timestampOffset to that base instead of
        //     resumeTime, keeping subtitle cues (which carry raw source PTS)
        //     aligned with the audio/video content at the same MSE position.
        var primerBatch = await this.demuxer.readPackets(4 * 1024 * 1024);
        var videoIdx = this.demuxer.videoStreamIndex;
        var audioIdx = this._audioAbsIdx;
        var minPts = Infinity;
        function firstPts(arr) { return (arr && arr.length > 0) ? arr[0].pts : Infinity; }
        if (primerBatch && primerBatch.packets) {
            var vFirst = firstPts(primerBatch.packets[videoIdx]);
            var aFirst = (audioIdx >= 0) ? firstPts(primerBatch.packets[audioIdx]) : Infinity;
            // Both are in their stream's native timebase (each track has its own
            // num/den). Convert to seconds via each track's time_base before
            // taking the min.
            var vStream = this.demuxer.streams[videoIdx];
            var aStream = (audioIdx >= 0) ? this.demuxer.streams[audioIdx] : null;
            var vFirstSec = (isFinite(vFirst) && vStream)
                ? vFirst * vStream.time_base_num / vStream.time_base_den : Infinity;
            var aFirstSec = (isFinite(aFirst) && aStream)
                ? aFirst * aStream.time_base_num / aStream.time_base_den : Infinity;
            minPts = Math.min(vFirstSec, aFirstSec);
        }
        // Fall back to resumeTime if the peek failed to yield any packets — the
        // pump will re-read on its first iteration.
        if (!isFinite(minPts)) minPts = resumeTime;

        // Set timestampOffset to the muxer's expected first-PTS base so MSE
        // position = source content position. Subtitle cues anchored to source
        // PTS now fire when the corresponding A/V content actually plays.
        try { this.sourceBuffer.timestampOffset = minPts; } catch (e) {}

        // 5. Reset audio re-encoder. Opus state is stateful; a half-filled
        //    frame from before the seek would corrupt output.
        if (this.audioReencoder) {
            await this.audioReencoder.reset();
        }

        // 6. Re-init muxer with fresh init segment.
        await this.muxer.destroy();
        this.muxer = new ClientMuxer(this.demuxer.libav);
        await this.muxer.init(await this._buildStreamInfos());

        if (generation !== this._seekGeneration) return;

        // 6b. Hand the peeked primer batch to the pump so it doesn't re-read
        //     (which would skip past these packets) — the pump consumes the
        //     primer on its first iteration before reading anew.
        this._primerBatch = primerBatch;

        // 7. Restart pump.
        this.running = true;
        this._pumpPromise = this._pumpLoop();
    } catch (err) {
        if (generation !== this._seekGeneration) return;
        SP.log.error("ClientPlayer", "Restart error:", err);
        setStatus("Seek error", "#ff6b6b");
        try {
            this.running = true;
            this._pumpPromise = this._pumpLoop();
        } catch (e) {}
    }
};

// Returns true if the switch was applied, false if rejected (e.g. browser
// lacks a decoder for the target codec). The previous pipeline stays intact
// on rejection so the caller can revert its UI to match actual audio.
ClientPlayer.prototype.switchAudioTrack = async function(audioTrackIdx) {
    SP.log.debug("ClientPlayer", "Switching to audio track", audioTrackIdx);

    if (!this.demuxer || !this.probeData.audio || this.probeData.audio.length === 0) {
        return false;
    }

    var newAbsIdx = this.demuxer.getAudioStreamIndex(audioTrackIdx);
    if (newAbsIdx < 0) {
        SP.log.warn("ClientPlayer", "Audio track not found:", audioTrackIdx);
        return false;
    }

    // Swap decoder for the new input codec. Output stays Opus 48 kHz stereo so
    // the muxer's audio codecpar is unchanged and the re-init segment is
    // identical across switches. reconfigureInput verifies codec support
    // before tearing down the existing decoder, so an unsupported track leaves
    // the previous audio pipeline intact.
    var audioInfo = this.probeData.audio[audioTrackIdx];
    var extradata = await this.demuxer.getAudioExtradata(newAbsIdx);
    var audioStream = this.demuxer.streams[newAbsIdx];
    try {
        await this.audioReencoder.reconfigureInput(
            audioInfo.codec,
            audioInfo.sample_rate || 48000,
            audioInfo.channels || 2,
            extradata,
            { num: audioStream.time_base_num, den: audioStream.time_base_den }
        );
    } catch (err) {
        SP.log.error("ClientPlayer", "Audio track switch failed:", err.message || err);
        setStatus("Audio codec not supported: " + audioInfo.codec, "#ff6b6b");
        return false;
    }

    this.currentAudioTrack = audioTrackIdx;
    await this._restartPipeline(this.video.currentTime, newAbsIdx);
    return true;
};

// Collect subtitle packets from a readPackets result into the accumulator.
// Zero-cost when no subtitle streams exist.
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

ClientPlayer.prototype._refreshActiveSubtitle = async function() {
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

    var vtt = await buildVTTFromPackets(pkts, timeBase, subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (this._activeSubtitleTrack + 1);
    attachVTTToVideo(this.video, vtt, label);
};

ClientPlayer.prototype.loadSubtitleTrack = async function(subTrackIndex) {
    if (!this.demuxer) return;

    var subInfo = this.probeData.subtitles[subTrackIndex];
    if (!subInfo) {
        SP.log.warn("ClientPlayer", "Subtitle track not found:", subTrackIndex);
        return;
    }

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
        SP.log.warn("ClientPlayer", "Subtitle stream not found in demuxer");
        return;
    }

    this._activeSubtitleTrack = subTrackIndex;
    this._activeSubtitleAbsIdx = absIndex;
    this._subtitleUpdateCounter = 0;

    var pkts = this._subtitlePackets[absIndex] || [];
    SP.log.debug("ClientPlayer", "Showing subtitle track", subTrackIndex,
        "with", pkts.length, "packets collected so far");

    var stream = this.demuxer.streams[absIndex];
    var timeBase = 1;
    if (stream && stream.time_base_num && stream.time_base_den) {
        timeBase = stream.time_base_num / stream.time_base_den;
    }

    var vtt = await buildVTTFromPackets(pkts, timeBase, subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (subTrackIndex + 1);
    attachVTTToVideo(this.video, vtt, label);
};

// Build stream infos array for muxer initialization. Video comes from the
// demuxer; audio is a freshly-built Opus codecpar (WebCodecs does the encode,
// so libav never sees the audio encoding path).
ClientPlayer.prototype._buildStreamInfos = async function() {
    var streamInfos = [];
    var videoStream = this.demuxer.streams[this.demuxer.videoStreamIndex];
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
    if (this._audioAbsIdx >= 0 && this.audioReencoder) {
        var opusCodecpar = await buildOpusCodecpar(this.demuxer.libav,
            this.audioReencoder.outputChannels, this.audioReencoder.outputSampleRate);
        streamInfos.push({
            inputIndex: this._audioAbsIdx,
            codecpar: opusCodecpar,
            time_base_num: 1,
            time_base_den: this.audioReencoder.outputSampleRate,
            codecName: "opus"
        });
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
        try { this.mediaSource.endOfStream(); } catch (e) {}
    }
    this.mediaSource = null;
    this.sourceBuffer = null;

    this.currentAudioTrack = 0;
    this._audioAbsIdx = -1;
};
