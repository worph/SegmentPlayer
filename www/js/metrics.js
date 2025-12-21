/* Metrics panel - transcode stats and adaptive quality display */

function formatRatio(ratio) {
    return ratio ? ratio.toFixed(1) + "%" : "-";
}

function getRatioClass(ratio) {
    if (!ratio) return "";
    if (ratio >= 70 && ratio <= 80) return "good";  // Target range
    if (ratio < 70) return "good";  // Below target is fine
    if (ratio < 90) return "warning";
    return "bad";
}

async function fetchMetrics() {
    try {
        var response = await fetch("/transcode/metrics");
        if (!response.ok) return;

        var data = await response.json();

        // Update adaptive preset display
        if (data.adaptive_preset) {
            var preset = data.adaptive_preset;
            document.getElementById("metricPreset").textContent = preset.current_preset;

            // Position marker on preset bar (0-100%)
            var presetIndex = SP.config.PRESETS.indexOf(preset.current_preset);
            var presetPosition = presetIndex >= 0 ? (presetIndex / (SP.config.PRESETS.length - 1)) * 100 : 50;
            document.getElementById("presetBar").style.left = presetPosition + "%";

            // Update adjustment counts
            document.getElementById("presetAdjustUp").innerHTML = "&#9650; " + (preset.adjustments_up || 0);
            document.getElementById("presetAdjustDown").innerHTML = "&#9660; " + (preset.adjustments_down || 0);
        }

        // Update adaptive CRF display
        if (data.adaptive_crf) {
            var crf = data.adaptive_crf;
            document.getElementById("metricCRF").textContent = "+" + (crf.crf_offset || 0);

            // Position marker on CRF bar (0-100%, where 0=+0, 100=+7)
            var crfPosition = ((crf.crf_offset || 0) / 7) * 100;
            document.getElementById("crfBar").style.left = crfPosition + "%";

            // Update adjustment counts
            document.getElementById("crfAdjustDown").innerHTML = "&#9650; " + (crf.decreases || 0);
            document.getElementById("crfAdjustUp").innerHTML = "&#9660; " + (crf.increases || 0);
        }

        // Update ratio gauge
        var avgRatio = data.transcode_ratio_avg || 0;
        var ratioNeedle = document.getElementById("ratioNeedle");
        var ratioValue = document.getElementById("ratioValue");

        // Clamp ratio to 0-100% for display
        var needlePos = Math.min(100, Math.max(0, avgRatio));
        ratioNeedle.style.left = needlePos + "%";

        ratioValue.textContent = formatRatio(avgRatio);
        ratioValue.className = "ratio-value " + getRatioClass(avgRatio);

        // Update performance metrics
        document.getElementById("metricRatioAvg").textContent = formatRatio(data.transcode_ratio_avg);
        document.getElementById("metricRatioAvg").className = "metric-value " + getRatioClass(data.transcode_ratio_avg);

        document.getElementById("metricRatioLast").textContent = formatRatio(data.transcode_ratio_last);
        document.getElementById("metricRatioLast").className = "metric-value " + getRatioClass(data.transcode_ratio_last);

        document.getElementById("metricRatioMin").textContent = formatRatio(data.transcode_ratio_min);
        document.getElementById("metricRatioMin").className = "metric-value good";

        document.getElementById("metricRatioMax").textContent = formatRatio(data.transcode_ratio_max);
        document.getElementById("metricRatioMax").className = "metric-value " + getRatioClass(data.transcode_ratio_max);

        document.getElementById("metricCacheHit").textContent = data.cache_hit_rate ? data.cache_hit_rate.toFixed(1) + "%" : "-";
        document.getElementById("metricCacheHit").className = "metric-value " + (data.cache_hit_rate > 50 ? "good" : "");

        document.getElementById("metricSegments").textContent = data.total_segments || 0;

        // Update codec info
        document.getElementById("metricVideoCodec").textContent = data.video_codec ? data.video_codec.toUpperCase() : "-";
        document.getElementById("metricAudioCodec").textContent = data.audio_codec ? data.audio_codec.toUpperCase() : "-";

        // Update mode display
        updateQualityDisplay();

    } catch (err) {}
}

function initMetricsToggle() {
    var metricsToggle = document.getElementById("metricsToggle");
    var metricsPanel = document.getElementById("metricsPanel");

    metricsToggle.addEventListener("click", function() {
        this.classList.toggle("active");
        metricsPanel.classList.toggle("active");

        if (metricsPanel.classList.contains("active")) {
            fetchMetrics();
            SP.state.metricsInterval = setInterval(fetchMetrics, 2000);
        } else {
            if (SP.state.metricsInterval) {
                clearInterval(SP.state.metricsInterval);
                SP.state.metricsInterval = null;
            }
        }
    });
}
