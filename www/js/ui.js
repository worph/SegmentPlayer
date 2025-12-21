/* UI state management */

function setStatus(text, color, pulsing) {
    if (pulsing === undefined) pulsing = false;
    SP.elements.statusText.textContent = text;
    SP.elements.statusEl.style.color = color;
    SP.elements.statusDot.classList.toggle("pulse", pulsing);
}

function updateModeDisplay() {
    var modeEl = document.getElementById("metricMode");
    var resEl = document.getElementById("adaptiveQualityRes");

    if (modeEl) {
        if (SP.state.isTranscoding) {
            modeEl.textContent = "Transcode";
            modeEl.className = "metric-value warning";
        } else {
            modeEl.textContent = "Repack";
            modeEl.className = "metric-value good";
        }
    }

    // Update current quality display
    if (resEl) {
        var qualityText;
        if (SP.state.currentResolution === "auto") {
            qualityText = SP.state.actualResolution ? "Auto (" + SP.state.actualResolution + "p)" : "Auto";
        } else if (SP.state.currentResolution === "original") {
            qualityText = "Original";
        } else {
            qualityText = SP.state.currentResolution + "p";
        }
        resEl.textContent = qualityText;
    }
}

function showSubtitleProgress(trackName) {
    SP.elements.subtitleProgress.classList.add("active");
    SP.elements.subtitleProgressFill.style.width = "0%";
    SP.elements.subtitleProgressText.textContent = 'Extracting "' + trackName + '" from video...';

    // Animate progress bar (simulated since we don't have real progress)
    var progress = 0;
    if (SP.state.subtitleProgressInterval) clearInterval(SP.state.subtitleProgressInterval);

    SP.state.subtitleProgressInterval = setInterval(function() {
        // Slow down as we approach 90% (never reach 100% until actually done)
        var remaining = 90 - progress;
        var increment = Math.max(0.5, remaining * 0.1);
        progress = Math.min(90, progress + increment);
        SP.elements.subtitleProgressFill.style.width = progress + "%";

        // Update text based on progress
        if (progress < 30) {
            SP.elements.subtitleProgressText.textContent = 'Extracting "' + trackName + '" from video...';
        } else if (progress < 60) {
            SP.elements.subtitleProgressText.textContent = 'Converting subtitle format...';
        } else {
            SP.elements.subtitleProgressText.textContent = 'Finalizing subtitles...';
        }
    }, 200);
}

function hideSubtitleProgress(success) {
    if (success === undefined) success = true;

    if (SP.state.subtitleProgressInterval) {
        clearInterval(SP.state.subtitleProgressInterval);
        SP.state.subtitleProgressInterval = null;
    }

    if (success) {
        SP.elements.subtitleProgressFill.style.width = "100%";
        SP.elements.subtitleProgressText.textContent = "Subtitles loaded!";
        setTimeout(function() {
            SP.elements.subtitleProgress.classList.remove("active");
        }, 500);
    } else {
        SP.elements.subtitleProgress.classList.remove("active");
    }
}
