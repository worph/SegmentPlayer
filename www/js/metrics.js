/* Metrics panel — per-mode display with live FPS, client pipeline, HLS, transcode engine */

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
    if (bytes === null || bytes === undefined) return "-";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
    return (bytes / 1073741824).toFixed(2) + " GB";
}

function formatRatio(ratio) {
    return (ratio === null || ratio === undefined) ? "-" : ratio.toFixed(1) + "%";
}

function formatBitrate(bps) {
    if (!bps) return "-";
    if (bps < 1000000) return (bps / 1000).toFixed(0) + " kbps";
    return (bps / 1000000).toFixed(2) + " Mbps";
}

function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) return "-";
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" + s : s);
}

function getRatioClass(ratio) {
    if (!ratio) return "";
    if (ratio >= 70 && ratio <= 80) return "good";
    if (ratio < 70) return "good";
    if (ratio < 90) return "warning";
    return "bad";
}

// ---------------------------------------------------------------------------
// FpsTracker — uses requestVideoFrameCallback (Chrome, Firefox 132+) with
// a getVideoPlaybackQuality polling fallback. Attached once to the page's
// <video> element; survives src changes because the element is stable.
// ---------------------------------------------------------------------------

function FpsTracker(videoEl) {
    this.video = videoEl;
    this._frameTimes = [];  // rolling array of frame-arrival timestamps (ms)
    this._maxEntries = 240; // ~10s at 24fps headroom
    this._stopped = false;
    this._mode = null;      // "rvfc" | "poll"
    this._pollTimer = null;
    this._pollLastTotal = 0;
    this._pollLastAt = 0;
    this._start();
}

FpsTracker.prototype._start = function() {
    var v = this.video;
    if (!v) return;
    var self = this;
    if (typeof v.requestVideoFrameCallback === "function") {
        this._mode = "rvfc";
        var tick = function(now /*, meta*/) {
            if (self._stopped) return;
            var arr = self._frameTimes;
            arr.push(now);
            if (arr.length > self._maxEntries) arr.splice(0, arr.length - self._maxEntries);
            try { v.requestVideoFrameCallback(tick); } catch (e) {}
        };
        try { v.requestVideoFrameCallback(tick); } catch (e) {}
        return;
    }
    // Fallback: poll getVideoPlaybackQuality every 500ms, spread frames over
    // the interval as synthetic timestamps. Less accurate but portable.
    this._mode = "poll";
    this._pollLastTotal = v.getVideoPlaybackQuality ? v.getVideoPlaybackQuality().totalVideoFrames : 0;
    this._pollLastAt = performance.now();
    this._pollTimer = setInterval(function() {
        if (self._stopped || !v.getVideoPlaybackQuality) return;
        var now = performance.now();
        var total = v.getVideoPlaybackQuality().totalVideoFrames;
        var delta = total - self._pollLastTotal;
        if (delta > 0) {
            var dtMs = now - self._pollLastAt;
            var stepMs = dtMs / delta;
            for (var i = 0; i < delta; i++) {
                self._frameTimes.push(self._pollLastAt + stepMs * (i + 1));
            }
            if (self._frameTimes.length > self._maxEntries) {
                self._frameTimes.splice(0, self._frameTimes.length - self._maxEntries);
            }
        }
        self._pollLastTotal = total;
        self._pollLastAt = now;
    }, 500);
};

// Rolling FPS over the last `windowMs` milliseconds (default 1s)
FpsTracker.prototype.current = function(windowMs) {
    if (!windowMs) windowMs = 1000;
    var arr = this._frameTimes;
    if (arr.length < 2) return 0;
    var now = arr[arr.length - 1];
    var cutoff = now - windowMs;
    // Binary search for first entry >= cutoff
    var lo = 0, hi = arr.length;
    while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (arr[mid] < cutoff) lo = mid + 1;
        else hi = mid;
    }
    var count = arr.length - lo;
    if (count < 2) return 0;
    var span = arr[arr.length - 1] - arr[lo];
    if (span <= 0) return 0;
    return ((count - 1) / span) * 1000;
};

FpsTracker.prototype.reset = function() {
    this._frameTimes = [];
};

FpsTracker.prototype.stop = function() {
    this._stopped = true;
    if (this._pollTimer) {
        clearInterval(this._pollTimer);
        this._pollTimer = null;
    }
};

// Lazily initialize a page-lifetime tracker on first panel open.
function ensureFpsTracker() {
    if (!SP.state.fpsTracker && SP.elements && SP.elements.video) {
        SP.state.fpsTracker = new FpsTracker(SP.elements.video);
    }
    return SP.state.fpsTracker;
}

// ---------------------------------------------------------------------------
// Rate tracker — computes bytes/s and packets/s across poll intervals.
// ---------------------------------------------------------------------------

function _computeRate(prev, current, nowMs, lastMs) {
    if (!prev || !lastMs) return null;
    var dt = (nowMs - lastMs) / 1000;
    if (dt <= 0) return null;
    return (current - prev) / dt;
}

// ---------------------------------------------------------------------------
// Mode detection — returns one of "direct" | "client" | "remux" | "transcode"
// Defaults to "direct" if the active mode hasn't been set yet.
// ---------------------------------------------------------------------------

function detectMode() {
    var m = SP.state.activePlaybackMode;
    if (m === "client" || m === "transcode" || m === "remux" || m === "direct") return m;
    if (SP.state.isClientSide) return "client";
    if (SP.state.isTranscoding) return "transcode";
    return "direct";
}

// ---------------------------------------------------------------------------
// Render — one tick of the local poll. Pulls browser-local signals (FPS,
// dropped, buffer, client-pipeline internals, HLS level) and fills the
// common + mode-specific panel sections. Cheap to call every 500ms.
// ---------------------------------------------------------------------------

function renderMetricsLocal() {
    var panel = document.getElementById("metricsPanel");
    if (!panel) return;
    var mode = detectMode();

    // Swap mode class — drives section visibility via CSS
    panel.classList.remove("mode-direct", "mode-client", "mode-remux", "mode-transcode");
    panel.classList.add("mode-" + mode);

    // --- Mode banner ---
    var modeBadge = document.getElementById("metricsModeBadge");
    var modeDetail = document.getElementById("metricsModeDetail");
    if (modeBadge) modeBadge.textContent = mode.toUpperCase();

    var probe = SP.state.probeData || (SP.state.currentFile && SP.state.probeCache ? SP.state.probeCache[SP.state.currentFile] : null);
    if (modeDetail) {
        var parts = [];
        if (probe && probe.video) {
            var vc = (probe.video.codec || "").toUpperCase();
            if (probe.video.profile) vc += " " + probe.video.profile;
            if (probe.video.bit_depth) vc += " " + probe.video.bit_depth + "-bit";
            parts.push(vc);
        }
        if (probe && probe.container) parts.push(probe.container.toUpperCase());
        // Flag the known client-mode HEVC MSE rate cap (see docs/known-issues).
        if (mode === "client" && probe && probe.video &&
                (probe.video.codec === "hevc" || probe.video.codec === "h265")) {
            parts.push("MSE throttled — use Direct for full fps");
        }
        modeDetail.textContent = parts.join(" · ");
    }

    // --- Source section ---
    var videoCodec = probe && probe.video && probe.video.codec
        ? probe.video.codec.toUpperCase() + (probe.video.profile ? " " + probe.video.profile : "")
        : "-";
    var audioTrack = probe && probe.audio && probe.audio[0];
    var audioCodec = audioTrack
        ? audioTrack.codec.toUpperCase() + (audioTrack.channels ? " " + audioTrack.channels + "ch" : "")
        : "-";
    var resolution = probe && probe.video
        ? (probe.video.width + "×" + probe.video.height +
           (probe.video.fps ? " @ " + probe.video.fps.toFixed(2) : ""))
        : "-";
    var container = probe && probe.container ? probe.container.toUpperCase() : "-";

    setText("metricVideoCodec", videoCodec);
    setText("metricAudioCodec", audioCodec);
    setText("metricResolution", resolution);
    setText("metricContainer", container);

    // --- Playback section (common) ---
    var v = SP.elements && SP.elements.video;
    var tracker = ensureFpsTracker();
    if (v) {
        var fps = tracker ? tracker.current(1000) : 0;
        // Expected FPS comes from probe data. Falls back to 24 if unknown so
        // the threshold still works on files where ffprobe couldn't report it.
        var expectedFps = (probe && probe.video && probe.video.fps) || 24;
        if (fps > 0) {
            setText("metricFps", fps.toFixed(1) + " / " + expectedFps.toFixed(2));
            // Red if rendered rate falls below 90% of source — matches the
            // common "noticeable stutter" threshold.
            setClass("metricFps", fps < expectedFps * 0.9 ? "bad" : "good");
        } else {
            setText("metricFps", "- / " + expectedFps.toFixed(2));
            setClass("metricFps", "");
        }

        var q = typeof v.getVideoPlaybackQuality === "function" ? v.getVideoPlaybackQuality() : null;
        if (q && q.totalVideoFrames > 0) {
            var rate = (q.droppedVideoFrames / q.totalVideoFrames) * 100;
            setText("metricDropped", q.droppedVideoFrames + " (" + rate.toFixed(1) + "%)");
            setClass("metricDropped", rate < 1 ? "good" : rate < 5 ? "warning" : "bad");
        } else {
            setText("metricDropped", "-");
            setClass("metricDropped", "");
        }

        var bufferAhead = typeof getBufferedAhead === "function" ? getBufferedAhead(v) : 0;
        setText("metricBuffer", bufferAhead.toFixed(1) + "s");
        setClass("metricBuffer", bufferAhead > 5 ? "good" : bufferAhead > 1 ? "warning" : "bad");

        var ct = v.currentTime, dur = v.duration;
        if (isFinite(dur) && dur > 0) {
            setText("metricProgress", formatTime(ct) + " / " + formatTime(dur));
        } else {
            setText("metricProgress", formatTime(ct));
        }
    }

    // --- Mode-specific sections ---
    if (mode === "client") renderClientSection();
    if (mode === "remux" || mode === "transcode") renderHlsSection();
    // Transcode engine section is populated by the 2s server poll (renderTranscodeSection)
}

function renderClientSection() {
    var cp = SP.state.clientPlayer;
    if (!cp) return;

    var muxer = cp.muxer;
    var demuxer = cp.demuxer;
    var vstats = muxer && typeof muxer.getVideoStats === "function" ? muxer.getVideoStats() : null;
    var dstats = demuxer && typeof demuxer.getStats === "function" ? demuxer.getStats() : null;

    if (vstats) {
        if (vstats.warmingUp) {
            setText("metricClientDepth", "measuring…");
            setClass("metricClientDepth", "warning");
        } else {
            setText("metricClientDepth", String(vstats.reorderDepth));
            setClass("metricClientDepth", "");
        }
        setText("metricClientModal", vstats.modalDuration ? vstats.modalDuration + " ticks" : "-");
        setText("metricClientWarmup", vstats.warmingUp ? "in progress" : "done");
        setClass("metricClientWarmup", vstats.warmingUp ? "warning" : "good");
        var cc = vstats.clampCount || 0;
        setText("metricClientClamps", String(cc));
        setClass("metricClientClamps", cc === 0 ? "good" : cc < 3 ? "warning" : "bad");
        if (vstats.synthesisFallback) {
            setText("metricClientDurRel", "FALLBACK");
            setClass("metricClientDurRel", "bad");
        } else if (vstats.pktDurationSeen && vstats.pktDurationReliable) {
            setText("metricClientDurRel", "demuxer");
            setClass("metricClientDurRel", "good");
        } else {
            setText("metricClientDurRel", "modal");
            setClass("metricClientDurRel", "");
        }
    }

    if (dstats) {
        setText("metricClientBytes",
            formatBytes(dstats.bytesRead) + (dstats.fileSize ? " / " + formatBytes(dstats.fileSize) : ""));
    }

    if (cp.stats) {
        setText("metricClientPackets", cp.stats.packetsRead + " pkts");
    }

    // Ingest rate (bytes/s) — smoothed over a short rolling window to avoid
    // flicker when range requests happen in bursts between polls.
    var now = performance.now();
    var currentBytes = dstats ? dstats.bytesRead : 0;
    var hist = SP.state._metricsHist = SP.state._metricsHist || [];
    hist.push({ bytes: currentBytes, at: now });
    // Keep last 5 samples (~2.5s at 500ms cadence)
    while (hist.length > 5) hist.shift();
    if (hist.length >= 2) {
        var oldest = hist[0];
        var newest = hist[hist.length - 1];
        var dt = (newest.at - oldest.at) / 1000;
        if (dt > 0) {
            var rate = (newest.bytes - oldest.bytes) / dt;
            setText("metricClientRate", formatBytes(Math.max(0, rate)) + "/s");
        } else {
            setText("metricClientRate", "-");
        }
    } else {
        setText("metricClientRate", "-");
    }
}

function renderHlsSection() {
    var hls = SP.state.hls;
    var stats = SP.state.hlsStats || {};
    if (hls && hls.levels && hls.levels.length > 0) {
        var lvlIdx = hls.currentLevel;
        var lvl = lvlIdx >= 0 ? hls.levels[lvlIdx] : null;
        if (lvl) {
            setText("metricHlsLevel", lvl.height + "p" + (hls.autoLevelEnabled !== false && lvlIdx === -1 ? " (auto)" : ""));
            setText("metricHlsBitrate", formatBitrate(lvl.bitrate));
        } else {
            setText("metricHlsLevel", "auto");
            setText("metricHlsBitrate", "-");
        }
    } else {
        setText("metricHlsLevel", "-");
        setText("metricHlsBitrate", "-");
    }
    setText("metricHlsSwitches", String(stats.levelSwitches || 0));
    if (stats.lastFragMs > 0) {
        var bytesLabel = stats.lastFragBytes > 0 ? " · " + formatBytes(stats.lastFragBytes) : "";
        setText("metricHlsFrag", stats.lastFragMs + " ms" + bytesLabel);
    } else {
        setText("metricHlsFrag", "-");
    }
}

// ---------------------------------------------------------------------------
// Transcode engine section — populated by the 2s server poll.
// ---------------------------------------------------------------------------

async function renderTranscodeSection() {
    if (detectMode() !== "transcode") return;
    var data;
    try {
        var response = await fetch("/transcode/metrics");
        if (!response.ok) return;
        data = await response.json();
    } catch (e) {
        return;
    }

    if (data.adaptive_preset) {
        var preset = data.adaptive_preset;
        setText("metricPreset", preset.current_preset);
        var presetIndex = SP.config.PRESETS.indexOf(preset.current_preset);
        var presetPosition = presetIndex >= 0 ? (presetIndex / (SP.config.PRESETS.length - 1)) * 100 : 50;
        var bar = document.getElementById("presetBar");
        if (bar) bar.style.left = presetPosition + "%";
        var up = document.getElementById("presetAdjustUp");
        var down = document.getElementById("presetAdjustDown");
        if (up) up.innerHTML = "&#9650; " + (preset.adjustments_up || 0);
        if (down) down.innerHTML = "&#9660; " + (preset.adjustments_down || 0);
    }

    if (data.adaptive_crf) {
        var crf = data.adaptive_crf;
        setText("metricCRF", "+" + (crf.crf_offset || 0));
        var crfPosition = ((crf.crf_offset || 0) / 7) * 100;
        var crfBar = document.getElementById("crfBar");
        if (crfBar) crfBar.style.left = crfPosition + "%";
        var crfUp = document.getElementById("crfAdjustDown");
        var crfDown = document.getElementById("crfAdjustUp");
        if (crfUp) crfUp.innerHTML = "&#9650; " + (crf.decreases || 0);
        if (crfDown) crfDown.innerHTML = "&#9660; " + (crf.increases || 0);
    }

    var avgRatio = data.transcode_ratio_avg || 0;
    var ratioNeedle = document.getElementById("ratioNeedle");
    var ratioValue = document.getElementById("ratioValue");
    var needlePos = Math.min(100, Math.max(0, avgRatio));
    if (ratioNeedle) ratioNeedle.style.left = needlePos + "%";
    if (ratioValue) {
        ratioValue.textContent = formatRatio(avgRatio);
        ratioValue.className = "ratio-value " + getRatioClass(avgRatio);
    }

    setTextClass("metricRatioAvg", formatRatio(data.transcode_ratio_avg), getRatioClass(data.transcode_ratio_avg));
    setTextClass("metricRatioLast", formatRatio(data.transcode_ratio_last), getRatioClass(data.transcode_ratio_last));
    setTextClass("metricRatioMin", formatRatio(data.transcode_ratio_min), "good");
    setTextClass("metricRatioMax", formatRatio(data.transcode_ratio_max), getRatioClass(data.transcode_ratio_max));
    setTextClass("metricCacheHit", data.cache_hit_rate ? data.cache_hit_rate.toFixed(1) + "%" : "-",
        data.cache_hit_rate > 50 ? "good" : "");
    setText("metricSegments", String(data.total_segments || 0));

    var resEl = document.getElementById("adaptiveQualityRes");
    if (resEl && data.video_codec) resEl.textContent = data.video_codec.toUpperCase();
}

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setClass(id, cls) {
    var el = document.getElementById(id);
    if (!el) return;
    el.className = "metric-value" + (cls ? " " + cls : "");
}

function setTextClass(id, text, cls) {
    var el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = "metric-value" + (cls ? " " + cls : "");
}

// ---------------------------------------------------------------------------
// Panel toggle + polling lifecycle
// ---------------------------------------------------------------------------

function initMetricsToggle() {
    var metricsToggle = document.getElementById("metricsToggle");
    var metricsPanel = document.getElementById("metricsPanel");

    metricsToggle.addEventListener("click", function() {
        this.classList.toggle("active");
        metricsPanel.classList.toggle("active");

        if (metricsPanel.classList.contains("active")) {
            ensureFpsTracker();
            renderMetricsLocal();
            renderTranscodeSection();
            // 500ms cadence for cheap browser-local signals (FPS, buffer, client pipeline, HLS level)
            SP.state.metricsLocalInterval = setInterval(renderMetricsLocal, 500);
            // 2s cadence for the /transcode/metrics server fetch; the handler
            // itself gates on mode, so non-transcode playback makes no requests.
            SP.state.metricsServerInterval = setInterval(renderTranscodeSection, 2000);
        } else {
            if (SP.state.metricsLocalInterval) {
                clearInterval(SP.state.metricsLocalInterval);
                SP.state.metricsLocalInterval = null;
            }
            if (SP.state.metricsServerInterval) {
                clearInterval(SP.state.metricsServerInterval);
                SP.state.metricsServerInterval = null;
            }
        }
    });
}
