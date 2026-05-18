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
    this._audioAbsIdx = -1;          // absolute stream index for the active audio track
    this._seekHandler = null;
    this._seekGeneration = 0;
    this._pumpPromise = null;
    this._primerBatch = null;        // first post-seek packet batch (peeked to derive muxer base)
    this._loadedFilepath = null;     // saved so watchdog._fullReload can rebuild the pipeline
    this._onUnrecoverable = null;    // optional callback set by player.js for mode-fallback

    // Subordinate concerns kept as separate objects — see end of file.
    this.subtitles = new ClientSubtitleCollector(this);
    this.watchdog = new ClientRecoveryWatchdog(this);
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
    this._loadedFilepath = filepath;
    this.probeData = probeData;
    this.currentAudioTrack = audioTrackIdx || 0;
    this.stats = { bytesRead: 0, packetsRead: 0, startTime: Date.now(), framesDropped: 0 };
    // _needsHardReload is owned by the watchdog and reset on teardown via cleanup() above.

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
    var subIndices = [];
    for (var i = 0; i < this.demuxer.streams.length; i++) {
        var s = this.demuxer.streams[i];
        if (s.codec_type === this.demuxer.libav.AVMEDIA_TYPE_SUBTITLE) {
            subIndices.push(s.index);
        }
    }
    this.subtitles.reset();
    this.subtitles.setStreamIndices(subIndices);

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

    // 9. Ongoing health monitoring. The 20-s startup watchdog in player.js
    //    only catches "never started" failures; this one catches
    //    mid-playback stalls, silent pump exits, SourceBuffer errors, and
    //    `video.error` (which puts the native browser controls in an inert
    //    state — play/pause/seek all become no-ops until the video element
    //    gets a fresh `src`).
    this.watchdog.install();

    SP.log.debug("ClientPlayer", "Started unified video+audio pipeline");
};

ClientPlayer.prototype._pumpLoop = async function() {
    this.watchdog.notifyPumpAlive(true);
    var firstData = true;
    try {
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
            this.subtitles.collect(result.packets);

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
    } finally {
        // Always reflect "pump exited" in both flags so the recovery
        // watchdog can detect "running set but pump silently gone" and
        // attempt restart. Without this, a soft pump death (3 errors then
        // `break`) left `this.running === true`, so external checks
        // couldn't tell the pipeline had stopped feeding the SourceBuffer.
        this.watchdog.notifyPumpAlive(false);
        this.running = false;
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
        // If the video element is in an error state, a soft pipeline restart
        // can't recover — the browser will refuse to play any newly-appended
        // data until `src` is reset. Hand off to the watchdog to do a hard
        // reload (fresh MediaSource + SourceBuffer).
        if (self.video.error) {
            self.watchdog.markNeedsHardReload();
            return;
        }
        // Always accept seeks — generation counter handles rapid re-entry.
        self._restartPipeline(self.video.currentTime);
    };
    this.video.addEventListener("seeking", this._seekHandler);
};

// Tear down the whole pipeline (cleanup) and reload from scratch at
// `resumeTime`. Used by the recovery watchdog when soft `_restartPipeline`
// can't recover — e.g. after `video.error` makes the browser refuse further
// appends, or the SourceBuffer hit an unrecoverable decode failure. Preserves
// audio/subtitle track selection across the rebuild.
ClientPlayer.prototype._fullReload = async function(resumeTime) {
    var probeData = this.probeData;
    var filepath = this._loadedFilepath;
    var audioTrack = this.currentAudioTrack;
    var subtitleTrack = this.subtitles.getActiveTrack();

    if (!filepath || !probeData) {
        throw new Error("ClientPlayer._fullReload: no filepath/probeData to reload from");
    }

    // cleanup() already nulls these out before they're used; capture
    // them above first.
    this.cleanup();
    await this.load(filepath, probeData, audioTrack);

    if (subtitleTrack >= 0) {
        try { await this.loadSubtitleTrack(subtitleTrack); }
        catch (e) { SP.log.warn("ClientPlayer", "Re-attach subtitle failed:", e); }
    }

    if (resumeTime > 0 && isFinite(resumeTime)) {
        // Setting currentTime fires 'seeking' which triggers
        // _restartPipeline — fine because _restartPipeline is cheap when
        // the pipeline just started and the demuxer has minimal in-flight
        // state.
        try { this.video.currentTime = resumeTime; }
        catch (e) { /* may throw if readyState < HAVE_METADATA */ }
    }
    // Best-effort resume — the user may have paused during the recovery
    // gap; play() failure here just leaves the video in paused state
    // which is the expected outcome of a user-paused video.
    this.video.play().catch(function() {});
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
        // Best-effort: try to keep playback alive by restarting the pump from
        // whatever state the demuxer is in. If this throws too, the watchdog
        // will pick up the silent-pump-death pattern next tick.
        this.running = true;
        this._pumpPromise = this._pumpLoop();
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

// Subtitle proxies — real logic lives on `this.subtitles` (see
// ClientSubtitleCollector at the bottom of the file). These exist so
// callers (controls.js, _fullReload) don't need to reach through
// `clientPlayer.subtitles.X`.
ClientPlayer.prototype.loadSubtitleTrack = function(subTrackIndex) {
    return this.subtitles.show(subTrackIndex);
};
ClientPlayer.prototype.clearActiveSubtitle = function() {
    this.subtitles.clearActive();
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

    this.watchdog.teardown();
    this.subtitles.reset();

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


// ===========================================================================
// ClientSubtitleCollector — piggyback subtitle handling for client mode.
//
// During normal playback the pump loop forwards every demuxed packet batch
// here via `collect(packets)`. We accumulate subtitle packets per stream and,
// when the user has an active subtitle selected, rebuild a VTT every 5 batches
// and re-attach it to the <video> via attachVTTToVideo(). This avoids any
// server round-trip for embedded subs in client mode.
// ===========================================================================
function ClientSubtitleCollector(player) {
    this.player = player;
    this._packets = {};        // absStreamIdx -> Packet[]
    this._streamIndices = [];  // absolute indices of subtitle streams
    this._activeTrack = -1;    // probe-level subtitle index (0,1,2...) or -1 = off
    this._activeAbsIdx = -1;   // matching demuxer absolute stream index
    this._updateCounter = 0;
}

ClientSubtitleCollector.prototype.reset = function() {
    this._packets = {};
    this._streamIndices = [];
    this._activeTrack = -1;
    this._activeAbsIdx = -1;
    this._updateCounter = 0;
};

ClientSubtitleCollector.prototype.setStreamIndices = function(indices) {
    this._streamIndices = indices.slice();
};

ClientSubtitleCollector.prototype.getActiveTrack = function() {
    return this._activeTrack;
};

ClientSubtitleCollector.prototype.clearActive = function() {
    this._activeTrack = -1;
    this._activeAbsIdx = -1;
};

// Append subtitle packets from one demuxer batch. If any of the new packets
// belong to the *active* subtitle stream, refresh the VTT every 5 batches —
// the throttle keeps long files from rebuilding VTT on every read.
ClientSubtitleCollector.prototype.collect = function(packets) {
    var dominated = false;
    for (var i = 0; i < this._streamIndices.length; i++) {
        var subIdx = this._streamIndices[i];
        var batch = packets[subIdx];
        if (batch && batch.length > 0) {
            if (!this._packets[subIdx]) this._packets[subIdx] = [];
            for (var j = 0; j < batch.length; j++) {
                this._packets[subIdx].push(batch[j]);
            }
            if (subIdx === this._activeAbsIdx) dominated = true;
        }
    }
    if (dominated && ++this._updateCounter >= 5) {
        this._updateCounter = 0;
        this._refreshActive();
    }
};

// Show the subtitle track at the given probe-level index. Resolves the
// absolute demuxer stream index, captures it for future collect() calls,
// and renders whatever packets have already been collected.
ClientSubtitleCollector.prototype.show = async function(subTrackIndex) {
    var demuxer = this.player.demuxer;
    var probeData = this.player.probeData;
    if (!demuxer) return;

    var subInfo = probeData.subtitles && probeData.subtitles[subTrackIndex];
    if (!subInfo) {
        SP.log.warn("ClientSubtitleCollector", "Subtitle track not found:", subTrackIndex);
        return;
    }

    // Walk the demuxer streams to find the N-th subtitle stream (probe-level
    // indices are dense within the subtitle list, but demuxer indices include
    // video/audio streams interleaved).
    var absIndex = -1;
    var subCount = 0;
    for (var i = 0; i < demuxer.streams.length; i++) {
        var s = demuxer.streams[i];
        if (s.codec_type === demuxer.libav.AVMEDIA_TYPE_SUBTITLE) {
            if (subCount === subTrackIndex) { absIndex = s.index; break; }
            subCount++;
        }
    }

    if (absIndex < 0) {
        SP.log.warn("ClientSubtitleCollector", "Subtitle stream not found in demuxer");
        return;
    }

    this._activeTrack = subTrackIndex;
    this._activeAbsIdx = absIndex;
    this._updateCounter = 0;

    var pkts = this._packets[absIndex] || [];
    SP.log.debug("ClientSubtitleCollector", "Showing track", subTrackIndex,
        "with", pkts.length, "packets collected so far");
    await this._renderVTT(absIndex, subInfo, pkts);
};

ClientSubtitleCollector.prototype._refreshActive = async function() {
    if (this._activeTrack < 0 || this._activeAbsIdx < 0) return;
    var subInfo = this.player.probeData.subtitles[this._activeTrack];
    if (!subInfo) return;
    var pkts = this._packets[this._activeAbsIdx];
    if (!pkts || pkts.length === 0) return;
    await this._renderVTT(this._activeAbsIdx, subInfo, pkts);
};

ClientSubtitleCollector.prototype._renderVTT = async function(absIdx, subInfo, pkts) {
    var stream = this.player.demuxer.streams[absIdx];
    var timeBase = 1;
    if (stream && stream.time_base_num && stream.time_base_den) {
        timeBase = stream.time_base_num / stream.time_base_den;
    }
    var vtt = await buildVTTFromPackets(pkts, timeBase, subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (this._activeTrack + 1);
    attachVTTToVideo(this.player.video, vtt, label);
};


// ===========================================================================
// ClientRecoveryWatchdog — ongoing health monitoring for client mode.
//
// Runs every 2 s after player.load() completes; tears itself down via
// player.cleanup(). Spots four failure modes:
//   1. `video.error` set — browser refuses further playback; needs
//      a fresh MediaSource (hard reload).
//   2. SourceBuffer fired an `error` event — MSE pipeline is corrupt;
//      hard reload.
//   3. Pump silently exited (`_pumpAlive === false` while we believed
//      we were running) — soft restart of the pipeline.
//   4. Playback stalled — `currentTime` not advancing for >8 s while
//      paused === false and buffered-ahead is empty. Soft restart;
//      hard reload if a soft restart didn't unstick it.
// On repeated failure (≥3 hard recoveries or ≥5 soft, within one
// session), give up and call player._onUnrecoverable so the caller can
// fall back to a different playback mode.
// ===========================================================================
function ClientRecoveryWatchdog(player) {
    this.player = player;
    this._timer = null;
    this._pumpAlive = false;
    this._needsHardReload = false;
    this._sbErrorListener = null;
    this._videoErrorListener = null;
}

ClientRecoveryWatchdog.prototype.notifyPumpAlive = function(alive) {
    this._pumpAlive = alive;
};

ClientRecoveryWatchdog.prototype.markNeedsHardReload = function() {
    this._needsHardReload = true;
};

ClientRecoveryWatchdog.prototype.install = function() {
    var self = this;
    var player = this.player;

    if (this._timer) {
        clearInterval(this._timer);
        this._timer = null;
    }

    // Listen for SourceBuffer-level decode failures. The 'error' event
    // can fire after a bad MP4 fragment makes it through `appendBuffer`
    // and trips the segment parser. After this, MSE is unrecoverable
    // without a fresh MediaSource.
    if (player.sourceBuffer && !this._sbErrorListener) {
        this._sbErrorListener = function() {
            SP.log.warn("ClientPlayer", "SourceBuffer error — scheduling hard reload");
            self._needsHardReload = true;
        };
        try {
            player.sourceBuffer.addEventListener("error", this._sbErrorListener);
        } catch (e) { /* SB may be in a weird state already */ }
    }

    // Listen for video-element fatal errors (MEDIA_ERR_DECODE etc.).
    // Once `video.error` is set the native controls go inert.
    if (!this._videoErrorListener) {
        this._videoErrorListener = function() {
            var err = player.video.error;
            SP.log.warn("ClientPlayer", "video.error event",
                err && { code: err.code, message: err.message });
            self._needsHardReload = true;
        };
        player.video.addEventListener("error", this._videoErrorListener);
    }

    var lastCt = player.video.currentTime;
    var lastProgressAt = performance.now();
    var softRecoveries = 0;
    var hardRecoveries = 0;
    var recoveryInFlight = false;

    this._timer = setInterval(function() {
        // Pipeline already torn down — stop polling.
        if (!player.video || !player.mediaSource || !self._timer) return;
        if (recoveryInFlight) return;

        var now = performance.now();
        var ct = player.video.currentTime;

        // 50 ms threshold filters jitter in currentTime updates without
        // flagging a real stall.
        if (Math.abs(ct - lastCt) > 0.05) {
            lastCt = ct;
            lastProgressAt = now;
        }

        var sawHardError = self._needsHardReload || !!player.video.error;
        var pumpDiedSilently = player.running === false && self._pumpAlive === false
            && (player.mediaSource.readyState === "open");
        // `running===false` is also the natural EOF state; distinguish
        // EOF from "pump unexpectedly stopped while still open".
        if (pumpDiedSilently) {
            var dur = player.mediaSource.duration;
            if (isFinite(dur) && dur > 0 && ct >= dur - 0.5) {
                pumpDiedSilently = false;
            }
        }
        var stalledMs = now - lastProgressAt;
        var bufAhead = getBufferedAhead(player.video);
        var stuckPlaying = !player.video.paused && stalledMs > 8000 && bufAhead < 0.5;

        if (sawHardError) {
            hardRecoveries++;
            recoveryInFlight = true;
            self._needsHardReload = false;
            if (hardRecoveries > 3) {
                SP.log.error("ClientPlayer", "Hard recovery exhausted; bailing");
                self.teardown();
                if (typeof player._onUnrecoverable === "function") {
                    player._onUnrecoverable("video.error after " + hardRecoveries + " hard recoveries");
                }
                return;
            }
            SP.log.warn("ClientPlayer", "Hard recovery #" + hardRecoveries + " (fresh MediaSource)");
            setStatus("Recovering…", "#ffd43b", true);
            player._fullReload(ct).then(function() {
                lastProgressAt = performance.now();
                lastCt = player.video.currentTime;
                recoveryInFlight = false;
            }).catch(function(e) {
                SP.log.error("ClientPlayer", "Hard recovery threw:", e);
                recoveryInFlight = false;
            });
            return;
        }

        if (pumpDiedSilently || stuckPlaying) {
            softRecoveries++;
            recoveryInFlight = true;
            if (softRecoveries > 5) {
                SP.log.error("ClientPlayer", "Soft recovery exhausted; escalating to hard reload");
                self._needsHardReload = true;
                recoveryInFlight = false;
                return;
            }
            SP.log.warn("ClientPlayer", "Soft recovery #" + softRecoveries
                + " (" + (pumpDiedSilently ? "pump-dead" : "stalled") + ")");
            setStatus("Recovering…", "#ffd43b", true);
            // _restartPipeline is async but we don't await — kick it off and
            // let the next watchdog tick observe whether it unstuck things.
            Promise.resolve(player._restartPipeline(ct)).then(function() {
                lastProgressAt = performance.now();
                lastCt = player.video.currentTime;
                recoveryInFlight = false;
            }).catch(function() {
                recoveryInFlight = false;
            });
        }
    }, 2000);
};

ClientRecoveryWatchdog.prototype.teardown = function() {
    if (this._timer) {
        clearInterval(this._timer);
        this._timer = null;
    }
    if (this._sbErrorListener && this.player.sourceBuffer) {
        try { this.player.sourceBuffer.removeEventListener("error", this._sbErrorListener); }
        catch (e) { /* SB may be gone */ }
        this._sbErrorListener = null;
    }
    if (this._videoErrorListener) {
        this.player.video.removeEventListener("error", this._videoErrorListener);
        this._videoErrorListener = null;
    }
    this._needsHardReload = false;
    this._pumpAlive = false;
};
