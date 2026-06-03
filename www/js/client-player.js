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

    // Self-healing state (see _startWatchdog / _recover). The pump never dies
    // permanently anymore: any silent stall (error cap, premature EOF, restart
    // throw, wedged playback) is caught by the watchdog and recovered via a
    // re-seek to the current position. onFatal is invoked only after recovery
    // repeatedly fails inside CLIENT_RECOVER_WINDOW_MS — the caller wires it to
    // the transcode fallback.
    this.onFatal = null;
    this._watchdogTimer = null;
    this._recovering = false;
    this._ended = false;             // true once a *legitimate* EOF sealed the MediaSource
    this._lastObservedTime = 0;      // last currentTime the watchdog saw
    this._lastProgressTs = 0;        // wall-clock of the last forward progress
    this._recoveryTimes = [];        // recent recovery timestamps (sliding window)
    this._filepath = null;           // stored so _hardReload can rebuild the pipeline

    // Subordinate concerns kept as separate objects — see end of file.
    this.subtitles = new ClientSubtitleCollector(this);
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
    this._filepath = filepath;       // stored so _hardReload can rebuild the pipeline
    this.probeData = probeData;
    this.currentAudioTrack = audioTrackIdx || 0;
    this.stats = { bytesRead: 0, packetsRead: 0, startTime: Date.now(), framesDropped: 0 };
    this._ended = false;
    this._recovering = false;
    this._recoveryTimes = [];

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
    // Buffering UI hooks: the demuxer retries failed range reads indefinitely
    // rather than faking EOF, so the right UX during a network hiccup is the
    // browser-native buffer-pause feel — same status badge change a stalled
    // HLS player would do. We also log stall start/recovery (with how long the
    // outage lasted) so a copied log shows the network timeline.
    var self = this;
    this.demuxer.onStallStart = function() {
        self._stallStartTs = Date.now();
        SP.log.warn("ClientPlayer", "Network stall — range read retrying ["
            + self._diag() + "]");
        setStatus("Buffering…", "#ffd43b", true);
    };
    this.demuxer.onStallEnd = function() {
        var ms = self._stallStartTs ? (Date.now() - self._stallStartTs) : 0;
        SP.log.info("ClientPlayer", "Network stall recovered after " + ms + "ms");
        self._stallStartTs = 0;
        setStatus("Client-side", "#51cf66");
    };
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

    // 9. Stall watchdog — turns any silent pipeline death into a re-seek+resume.
    this._startWatchdog();

    SP.log.debug("ClientPlayer", "Started unified video+audio pipeline");
};

ClientPlayer.prototype._pumpLoop = async function() {
    var firstData = true;
    SP.log.debug("ClientPlayer", "Pump start [" + this._diag() + "]");
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
                var eofDur = (this.probeData && this.probeData.duration) || 0;
                var eofCt = this.video ? this.video.currentTime : 0;
                var nearEnd = (eofDur <= 0) ||
                    (eofDur - eofCt) <= (SP.config.CLIENT_EOF_GUARD_SEC || 10);
                if (!nearEnd) {
                    // Premature EOF far from the real end — almost always a
                    // transient read fault (flaky range server, proxy 416) or a
                    // corrupt region, not the actual end of the file. Sealing the
                    // MediaSource here would freeze playback for good. Exit the
                    // pump instead; the watchdog re-seeks to currentTime (which
                    // clears libav's EOF state) and resumes.
                    SP.log.warn("ClientPlayer", "Premature EOF at",
                        eofCt.toFixed(1) + "s of " + eofDur + "s — will recover");
                    break;
                }
                SP.log.info("ClientPlayer", "End of stream (real EOF) [" + this._diag() + "]");
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
                this._ended = true;
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
                // A successful append clears the error tally: the threshold
                // below is about *consecutive* failures, not a lifetime count
                // (which would eventually kill any long playback).
                this.stats.errors = 0;

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
            SP.log.error("ClientPlayer", "Pump error #" + this.stats.errors
                + " [" + this._diag() + "]:", err);

            if (this.stats.errors >= 3) {
                // Three *consecutive* failures (the counter resets on every
                // successful append). Exit the pump rather than spinning on a
                // bad batch — the watchdog re-seeks to the current position and
                // restarts a clean pipeline. Repeated failures inside the
                // recovery window escalate to the transcode fallback.
                SP.log.warn("ClientPlayer", "3 consecutive pump errors — exiting for watchdog recovery");
                setStatus("Buffering…", "#ffd43b", true);
                break;
            }
            await sleep(500);
        }
    }
    } finally {
        // Pump must clear `running` even on exception so subsequent
        // cleanup() / _restartPipeline see the right state.
        this.running = false;
        SP.log.debug("ClientPlayer", "Pump exit [" + this._diag()
            + "] ended=" + this._ended + " firstData=" + firstData);
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

// Compact one-line snapshot of the playback state, embedded in log lines at
// every interesting transition so a copied log reads as a timeline of how the
// pipeline got into trouble (buffer drained? pump dead? MSE closed?).
ClientPlayer.prototype._diag = function() {
    var v = this.video;
    var ahead = v ? getBufferedAhead(v) : 0;
    return "ct=" + (v ? v.currentTime.toFixed(2) : "?")
        + " ahead=" + ahead.toFixed(2)
        + " rs=" + (v ? v.readyState : "?")
        + " net=" + (v ? v.networkState : "?")
        + " paused=" + (v ? v.paused : "?")
        + " ms=" + (this.mediaSource ? this.mediaSource.readyState : "none")
        + " running=" + this.running
        + " errs=" + ((this.stats && this.stats.errors) || 0);
};

// ── Self-healing watchdog ──────────────────────────────────────────────────
// Runs while a file is loaded. Each tick it checks whether playback is making
// forward progress; if not — and the stall isn't an expected one (user pause,
// seek, genuine EOF, a restart already in flight) — it recovers by re-seeking
// to the current position. The native <video> "waiting" event already shows
// the spinner, so the user sees: stall → loading → playback resumes.
ClientPlayer.prototype._startWatchdog = function() {
    this._stopWatchdog();
    this._lastObservedTime = this.video ? this.video.currentTime : 0;
    this._lastProgressTs = Date.now();
    var self = this;
    this._watchdogTimer = setInterval(function() { self._watchdogTick(); }, 1000);
};

ClientPlayer.prototype._stopWatchdog = function() {
    if (this._watchdogTimer) {
        clearInterval(this._watchdogTimer);
        this._watchdogTimer = null;
    }
};

ClientPlayer.prototype._watchdogTick = function() {
    var v = this.video;
    if (!v) return;

    var now = Date.now();
    var ct = v.currentTime;
    var advancing = ct > this._lastObservedTime + 0.02;
    if (advancing) {
        this._lastObservedTime = ct;
        this._lastProgressTs = now;
    }

    // A restart already in flight — leave it alone.
    if (this._recovering) { this._lastObservedTime = ct; return; }

    // A fatal element error (MEDIA_ERR_DECODE from a malformed append) pauses
    // the <video>, so it would otherwise look like an intentional pause below.
    // Catch it first — it needs a full MediaSource rebuild (_hardReload), not a
    // re-seek. The view-layer onerror usually fires first; this is the backstop.
    if (v.error) { this._recover("media error " + v.error.code); return; }

    // Stalls we must not fight: nothing is supposed to be playing, or a
    // teardown/restart is already underway.
    if (this._ended || v.ended || v.paused || v.seeking) {
        this._lastObservedTime = ct;
        return;
    }

    var ahead = getBufferedAhead(v);
    var pumpDead = !this.running;
    var stalledMs = now - this._lastProgressTs;

    if (pumpDead && ahead < (SP.config.CLIENT_PUMP_DEAD_REFILL_AHEAD || 2)) {
        // The pump exited (consecutive errors, premature EOF, or a restart
        // throw). Refill just before the existing buffer drains so recovery is
        // as seamless as the network allows.
        this._recover("pump exited");
    } else if (!pumpDead && !advancing && ahead < 0.5
            && stalledMs > (SP.config.CLIENT_STALL_RECOVER_MS || 12000)) {
        // Pump alive but no forward progress and the buffer is empty for a
        // sustained window — the demuxer's own read-retry hasn't resolved it
        // (e.g. a decode wedge, not just a slow network). Force a clean re-seek.
        this._recover("playback wedged");
    }
};

// Restart the pipeline at the current position to recover from a stall. Counts
// recoveries in a sliding window; if they pile up faster than the file can make
// progress, give up and let the caller fall back to the transcode tier.
ClientPlayer.prototype._recover = function(reason) {
    if (this._recovering || this._ended || !this.video) return;

    var now = Date.now();
    var win = SP.config.CLIENT_RECOVER_WINDOW_MS || 30000;
    this._recoveryTimes = (this._recoveryTimes || []).filter(function(t) {
        return now - t < win;
    });
    this._recoveryTimes.push(now);
    if (this._recoveryTimes.length > (SP.config.CLIENT_RECOVER_MAX || 4)) {
        this._giveUp(reason);
        return;
    }

    this._recovering = true;
    SP.log.warn("ClientPlayer", "Auto-recovering (" + reason + ") attempt "
        + this._recoveryTimes.length + "/" + (SP.config.CLIENT_RECOVER_MAX || 4)
        + " in window [" + this._diag() + "]");

    var self = this;
    var resumeTime = this.video.currentTime;
    var settled = function() {
        // Give the freshly-restarted pump a moment to produce data before the
        // next tick judges progress, and re-anchor the progress baseline.
        self._lastObservedTime = self.video ? self.video.currentTime : 0;
        self._lastProgressTs = Date.now();
        self._recovering = false;
    };
    var failed = function() { self._recovering = false; };

    // A fatal element error (MEDIA_ERR_DECODE — e.g. a malformed transmux
    // segment when warmup-measured B-frame reorder depth was too shallow for a
    // later, deeper GOP) can only be cleared by a *new* MediaSource; re-seeking
    // the existing (errored) one is a no-op. Rebuild client-side first — only if
    // that keeps failing does _giveUp drop to the server transcode tier.
    if (this.video.error) {
        this._hardReload(resumeTime).then(settled, failed);
    } else {
        this._restartPipeline(resumeTime, undefined, "Buffering…").then(settled, failed);
    }
};

// Public entry for a fatal <video> error, called from the view-layer onerror
// handler. Routes into the sliding-window recovery (→ _hardReload), so a
// recoverable decode glitch is retried client-side before the view falls back
// to the server transcoder.
ClientPlayer.prototype.handleMediaError = function() {
    var code = (this.video && this.video.error && this.video.error.code) || "?";
    this._recover("media error " + code);
};

// Full client-tier rebuild at resumeTime. Re-runs load() — which tears down the
// errored MediaSource and builds a fresh one (clearing MEDIA_ERR_DECODE), then
// re-inits the demuxer/muxer so a fresh warmup re-measures reorder depth for the
// *current* region of the file (the depth measured at t≈0 can be too shallow for
// a later, deeper B-pyramid GOP — the exact cause of the malformed segment).
// Keeps the client tier instead of dropping to the server transcoder.
ClientPlayer.prototype._hardReload = async function(resumeTime) {
    SP.log.warn("ClientPlayer", "Hard reload (media error) at " + resumeTime.toFixed(1) + "s");
    setStatus("Recovering…", "#ffd43b", true);
    // load() resets _recovering/_recoveryTimes; snapshot the sliding-window
    // count so repeated hard reloads still escalate to _giveUp (→ onFatal →
    // transcode) instead of looping forever.
    var savedTimes = this._recoveryTimes;
    await this.load(this._filepath, this.probeData, this.currentAudioTrack);
    this._recoveryTimes = savedTimes;
    this._recovering = true; // load() cleared it; _recover's .then() resets it
    // Resume at the failed position: setting currentTime fires the seek handler
    // → _restartPipeline seeks the demuxer to the nearest keyframe ≤ resumeTime.
    if (resumeTime > 0) {
        try { this.video.currentTime = resumeTime; } catch (e) {}
    }
    try { this.video.play().catch(function() {}); } catch (e) {}
};

ClientPlayer.prototype._giveUp = function(reason) {
    SP.log.error("ClientPlayer", "Recovery limit exceeded — giving up:", reason);
    this._stopWatchdog();
    this.running = false;
    if (typeof this.onFatal === "function") {
        this.onFatal("Client playback unrecoverable (" + (reason || "repeated stalls") + ")");
    } else {
        setStatus("Playback stalled", "#ff6b6b");
    }
};

ClientPlayer.prototype._setupSeekHandler = function() {
    var self = this;
    this._seekGeneration = 0;
    this._seekHandler = function() {
        // Generation counter inside _restartPipeline handles rapid re-entry.
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
ClientPlayer.prototype._restartPipeline = async function(resumeTime, newAudioAbsIdx, statusText) {
    var generation = ++this._seekGeneration;

    // A restart re-establishes the stream from `resumeTime`, so any prior
    // legitimate-EOF state no longer applies (e.g. recovering after a spurious
    // EOF, or the user seeking backward from the end).
    this._ended = false;

    SP.log.debug("ClientPlayer", "Restarting pipeline at", resumeTime.toFixed(2) + "s",
        "(gen " + generation + ")");
    setStatus(statusText || "Seeking...", statusText ? "#ffd43b" : "#4dabf7", true);

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
// callers (controls.js) don't need to reach through
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
    this._stopWatchdog();
    this._recovering = false;
    this._ended = false;

    this.subtitles.reset();

    // Drop any subtitle <track> elements + disable lingering TextTracks on
    // the <video>. Without this, an old subtitle persists across file changes
    // and tier-switches (client → transcode fallback) until something else
    // re-attaches a track. attachVTTToVideo does this on the next attach,
    // but cleanup() may run with no follow-up attach.
    this.video.querySelectorAll("track").forEach(function(t) { t.remove(); });
    for (var ti = 0; ti < this.video.textTracks.length; ti++) {
        this.video.textTracks[ti].mode = "disabled";
    }

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
    // Cue count is a cheap proxy for "are subtitles keeping up": if the user
    // reports sentences being skipped, compare cue growth here against the
    // demuxer read progress in the surrounding log lines.
    var cueCount = (vtt.match(/-->/g) || []).length;
    SP.log.debug("ClientSubtitle", "VTT rebuilt track=" + this._activeTrack
        + " pkts=" + pkts.length + " cues=" + cueCount + " codec=" + subInfo.codec);
    var label = subInfo.title || subInfo.language || "Track " + (this._activeTrack + 1);
    attachVTTToVideo(this.player.video, vtt, label);
};
