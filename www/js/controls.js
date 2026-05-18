/* Controls — audio, subtitle, resolution, mode, download handlers.
 *
 * Each user-facing action exists as a pure setter (setAudioTrack, setSubtitle,
 * setQuality, setMode, downloadCurrent) so the new settings menu can call
 * them directly. The legacy <select> change listeners stay as a thin shim
 * that calls the same setter, keeping the hidden dropdown layer working as
 * the source of truth until phase 6 removes it.
 */

// Helper: get audio index from level URL
function getAudioIdxFromLevel(level) {
    var urlMatch = level.url && level.url[0] ? level.url[0].match(/stream_a(\d+)_/) : null;
    return urlMatch ? parseInt(urlMatch[1]) : 0;
}

// Helper: find levels that use a specific audio track
function getLevelsForAudioTrack(audioIdx) {
    if (!SP.state.hls || !SP.state.hls.levels) return [];
    return SP.state.hls.levels
        .map(function(level, idx) { return { level: level, idx: idx }; })
        .filter(function(item) { return getAudioIdxFromLevel(item.level) === audioIdx; });
}

// Helper: create and append a <track> element pointing at a VTT URL.
// Shared by all subtitle-loading paths (embedded server-extract, transcoded,
// external .srt). Handles the loading spinner and progress banner uniformly.
function attachExternalSubtitle(url, label) {
    SP.elements.subtitleLoading.classList.add("active");
    if (label) showSubtitleProgress(label);
    var track = document.createElement("track");
    track.kind = "subtitles";
    track.src = url;
    track.default = true;
    if (label) track.label = label;
    track.addEventListener("load", function() {
        SP.elements.subtitleLoading.classList.remove("active");
        hideSubtitleProgress(true);
        // Enable by reference: textTracks[0] may be a stale entry from a
        // previously-removed <track> (Chromium retains TextTrack objects
        // after their <track> element is removed).
        if (track.track) track.track.mode = "showing";
    });
    track.addEventListener("error", function() {
        SP.elements.subtitleLoading.classList.remove("active");
        hideSubtitleProgress(false);
    });
    SP.elements.video.appendChild(track);
}

// ─── Pure setters ────────────────────────────────────────────────────────

// Switch the active audio track. `newAudioIdx` is the probe-level index
// (0,1,...). `onRejected` is called if the underlying switch fails (only
// fires for the async client-mode path).
function setAudioTrack(newAudioIdx, onRejected) {
    if (isNaN(newAudioIdx) || newAudioIdx === SP.state.currentAudioIdx) return;

    // Client-side playback: switch audio track via ClientPlayer. The call
    // is async and may reject (e.g. browser lacks a decoder for the target
    // codec). Revert state on rejection so the UI keeps reflecting the
    // actually-playing track.
    if (SP.state.activePlaybackMode === "client" && SP.state.clientPlayer) {
        var prevIdx = SP.state.currentAudioIdx;
        SP.state.currentAudioIdx = newAudioIdx;
        SP.state.clientPlayer.switchAudioTrack(newAudioIdx).then(function(ok) {
            if (!ok) {
                SP.state.currentAudioIdx = prevIdx;
                SP.elements.audioSelect.value = String(prevIdx);
                if (typeof onRejected === "function") onRejected(prevIdx);
            }
        });
        return;
    }

    // Direct mode: use native audioTracks API
    if (!SP.state.hls) {
        if (SP.elements.video.audioTracks && SP.elements.video.audioTracks.length > 1) {
            for (var i = 0; i < SP.elements.video.audioTracks.length; i++) {
                SP.elements.video.audioTracks[i].enabled = (i === newAudioIdx);
            }
            SP.state.currentAudioIdx = newAudioIdx;
        }
        return;
    }

    var currentTime = SP.elements.video.currentTime;
    var wasPlaying = !SP.elements.video.paused;

    // For transcoded mode with muxed audio, we need to find the level
    // with the matching audio index and switch to it
    if (SP.state.activePlaybackMode === "transcode" && SP.state.hls.levels && SP.state.hls.levels.length > 0) {
        var levelsForTrack = getLevelsForAudioTrack(newAudioIdx);

        if (levelsForTrack.length > 0) {
            // Find current resolution to preserve quality
            var currentLevelIdx = SP.state.hls.currentLevel >= 0 ? SP.state.hls.currentLevel : SP.state.hls.loadLevel;
            var currentLevel = SP.state.hls.levels[currentLevelIdx];
            var targetHeight = currentLevel ? currentLevel.height : null;

            // Find best matching level: same resolution, or highest quality
            var targetLevelIdx = levelsForTrack[0].idx;
            if (targetHeight) {
                var exactMatch = levelsForTrack.find(function(item) {
                    return item.level.height === targetHeight;
                });
                if (exactMatch) {
                    targetLevelIdx = exactMatch.idx;
                }
            }

            SP.state.currentAudioIdx = newAudioIdx;

            // Update resolution state BEFORE changing dropdown to avoid triggering resolution handler
            var targetLevel = SP.state.hls.levels[targetLevelIdx];
            if (targetLevel) {
                SP.state.currentResolution = targetLevel.height.toString();
                SP.state.actualResolution = targetLevel.height;
                SP.elements.resolutionSelect.value = targetLevel.height.toString();
                updateQualityDisplay();
            }

            // Lock to this level (disable ABR to prevent switching back to old audio)
            SP.state.hls.currentLevel = targetLevelIdx;

            // Force buffer flush and seek to apply change immediately
            setTimeout(function() {
                SP.elements.video.currentTime = currentTime + 0.1;
                setTimeout(function() {
                    SP.elements.video.currentTime = currentTime;
                    if (wasPlaying) SP.elements.video.play();
                }, 50);
            }, 100);
            return;
        }
    }

    // Fallback: try HLS.js native audio track switching (for non-muxed audio)
    if (SP.state.hls.audioTracks && SP.state.hls.audioTracks.length > newAudioIdx) {
        SP.state.hls.audioTrack = newAudioIdx;
        SP.state.currentAudioIdx = newAudioIdx;
    }
}

// Apply the subtitle selection. `rawVal` is:
//   ""              — turn subtitles off
//   "embedded:N"    — server-extracted (or client-collected) embedded track N
//   numeric string  — transcoded subtitle index, OR HLS subtitle track index
//   any other path  — external .srt under /subs/
// `label` is shown in the <track>.
function setSubtitle(rawVal, label) {
    removeAllTracks(SP.elements.video);
    for (var i = 0; i < SP.elements.video.textTracks.length; i++) {
        SP.elements.video.textTracks[i].mode = "hidden";
    }
    SP.elements.subtitleLoading.classList.remove("active");
    hideSubtitleProgress(false);

    if (rawVal === "" || rawVal == null) {
        // Off — clear active subtitle track in client mode
        if (SP.state.activePlaybackMode === "client" && SP.state.clientPlayer) {
            SP.state.clientPlayer.clearActiveSubtitle();
        }
        return;
    }

    if (SP.state.hls && SP.state.hls.subtitleTracks && SP.state.hls.subtitleTracks.length > 0) {
        SP.state.hls.subtitleTrack = parseInt(rawVal);
        return;
    }

    // Embedded subtitle track
    if (typeof rawVal === "string" && rawVal.indexOf("embedded:") === 0) {
        var subIndex = parseInt(rawVal.split(":")[1]);
        // Client-side mode: use piggybacked subtitle packets (no server needed)
        if (SP.state.activePlaybackMode === "client" && SP.state.clientPlayer) {
            SP.state.clientPlayer.loadSubtitleTrack(subIndex).catch(function(err) {
                SP.log.error("Subtitles", "Client extraction error:", err);
            });
            return;
        }
        // Direct/other modes: extract via server API
        attachExternalSubtitle(
            "/api/subtitle/" + encodeFilePath(SP.state.currentFile) + "/track/" + subIndex + ".vtt",
            label || ("Track " + (subIndex + 1))
        );
        return;
    }

    // Transcoded subtitle track (rawVal is a numeric string)
    var val = parseInt(rawVal);
    if (!isNaN(val) && SP.state.currentTranscodeBase && SP.state.transcodedSubtitleTracks[val]) {
        attachExternalSubtitle(
            SP.state.currentTranscodeBase + "/subs_" + val + ".vtt",
            label || (SP.state.transcodedSubtitleTracks[val].name || "Track " + (val + 1))
        );
        return;
    }

    // External subtitle file (rawVal is a path under /subs/)
    attachExternalSubtitle("/subs/" + rawVal, label || "Subtitle");
}

// Change the streaming quality. `newResolution` is one of:
//   "auto" | "original" | "source" | "1080p" | "720p" | ... | "<height>"
// Only meaningful when hls.js is the active player.
function setQuality(newResolution) {
    if (!SP.state.hls || !SP.state.hls.levels || SP.state.hls.levels.length === 0) return;
    if (newResolution === SP.state.currentResolution) return;

    resetMetrics();
    SP.state.currentResolution = newResolution;
    // Reset actual resolution - will be updated by LEVEL_SWITCHED event
    SP.state.actualResolution = null;
    updateQualityDisplay();

    // For transcoded mode, filter levels by current audio track
    var isTranscoding = SP.state.activePlaybackMode === "transcode";
    var candidateLevels = isTranscoding
        ? getLevelsForAudioTrack(SP.state.currentAudioIdx)
        : SP.state.hls.levels.map(function(level, idx) { return { level: level, idx: idx }; });

    if (newResolution === 'auto') {
        // Only allow auto if there's a single audio track
        var audioIdxSet = {};
        SP.state.hls.levels.forEach(function(level) {
            audioIdxSet[getAudioIdxFromLevel(level)] = true;
        });
        var hasMultipleAudioTracks = isTranscoding && Object.keys(audioIdxSet).length > 1;

        if (hasMultipleAudioTracks) {
            // Pick highest quality level for current audio track instead
            var sorted = candidateLevels.slice().sort(function(a, b) {
                return b.level.height - a.level.height;
            });
            if (sorted.length > 0) {
                SP.state.hls.currentLevel = sorted[0].idx;
            }
        } else {
            SP.state.hls.currentLevel = -1;
        }
    } else {
        var targetHeight = parseInt(newResolution) || 0;
        var levelIdx = -1;

        for (var i = 0; i < candidateLevels.length; i++) {
            var item = candidateLevels[i];
            if (newResolution === 'original' || newResolution === 'source') {
                // Pick highest quality for "original"
                if (levelIdx === -1 || item.level.height > SP.state.hls.levels[levelIdx].height) {
                    levelIdx = item.idx;
                }
            } else if (item.level.height === targetHeight) {
                levelIdx = item.idx;
                break;
            }
        }

        if (levelIdx >= 0) {
            SP.state.hls.currentLevel = levelIdx;
        }
    }
}

// Change the playback mode and re-trigger playback of the current file.
function setMode(newMode) {
    if (newMode === SP.state.playbackMode) return;
    SP.state.playbackMode = newMode;
    localStorage.setItem('sp_playback_mode', newMode);
    updateAutoModeLabel();
    if (SP.state.currentFile) {
        var fileName = SP.state.currentFile.split("/").pop();
        playFile(SP.state.currentFile, fileName);
    }
}

// Download the currently-playing file.
function downloadCurrent() {
    if (!SP.state.currentFile) return;
    var downloadUrl = "/direct/" + encodeFilePath(SP.state.currentFile);
    var a = document.createElement("a");
    a.href = downloadUrl;
    a.download = SP.state.currentFile.split("/").pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function updateAutoModeLabel() {
    var autoOption = SP.elements.modeSelect.querySelector('option[value="auto"]');
    if (!autoOption) return;
    if (SP.state.activePlaybackMode && SP.state.playbackMode === "auto") {
        autoOption.textContent = "Auto (" + SP.state.activePlaybackMode + ")";
    } else {
        autoOption.textContent = "Auto";
    }
}

// ─── Legacy <select> wiring ──────────────────────────────────────────────
// Kept as a thin shim so the hidden legacy dropdowns remain a working
// source of truth. Removed in phase 6 when the menu fully replaces them.

function initAudioControl() {
    SP.elements.audioSelect.addEventListener("change", function() {
        if (this.value === "") return;
        setAudioTrack(parseInt(this.value));
    });
}

function initSubtitleControl() {
    SP.elements.subtitleSelect.addEventListener("change", function() {
        var rawVal = this.value;
        var label = this.options[this.selectedIndex] ? this.options[this.selectedIndex].text : "";
        setSubtitle(rawVal, label);
    });
}

function initResolutionControl() {
    SP.elements.resolutionSelect.addEventListener("change", function() {
        setQuality(this.value);
    });
}

function initDownloadControl() {
    SP.elements.downloadBtn.addEventListener("click", downloadCurrent);
}

function initModeControl() {
    SP.elements.modeSelect.value = SP.state.playbackMode;
    SP.elements.modeSelect.addEventListener("change", function() {
        setMode(this.value);
    });
}

function initControls() {
    initAudioControl();
    initSubtitleControl();
    initResolutionControl();
    initDownloadControl();
    initModeControl();
}
