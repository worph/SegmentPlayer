/* Controls - audio, subtitle, resolution handlers */

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

function initAudioControl() {
    SP.elements.audioSelect.addEventListener("change", function() {
        if (!SP.state.hls || this.value === "") return;

        var newAudioIdx = parseInt(this.value);
        if (newAudioIdx === SP.state.currentAudioIdx) return;

        var currentTime = SP.elements.video.currentTime;
        var wasPlaying = !SP.elements.video.paused;

        // For transcoded mode with muxed audio, we need to find the level
        // with the matching audio index and switch to it
        if (SP.state.isTranscoding && SP.state.hls.levels && SP.state.hls.levels.length > 0) {
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

                // Lock to this level (disable ABR to prevent switching back to old audio)
                SP.state.hls.currentLevel = targetLevelIdx;

                // Update resolution dropdown to match
                var targetLevel = SP.state.hls.levels[targetLevelIdx];
                if (targetLevel) {
                    SP.elements.resolutionSelect.value = targetLevel.height.toString();
                    SP.state.currentResolution = targetLevel.height.toString();
                    SP.state.actualResolution = targetLevel.height;
                    updateModeDisplay();
                }

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
    });
}

function initSubtitleControl() {
    SP.elements.subtitleSelect.addEventListener("change", function() {
        var val = parseInt(this.value);
        SP.elements.video.querySelectorAll("track").forEach(function(t) { t.remove(); });
        for (var i = 0; i < SP.elements.video.textTracks.length; i++) {
            SP.elements.video.textTracks[i].mode = "hidden";
        }
        SP.elements.subtitleLoading.classList.remove("active");
        hideSubtitleProgress(false);

        if (val === -1 || this.value === "" || this.value === "-1") return;

        if (SP.state.hls && SP.state.hls.subtitleTracks && SP.state.hls.subtitleTracks.length > 0) {
            SP.state.hls.subtitleTrack = val;
            return;
        }

        // Show loading indicators for VTT that needs extraction
        SP.elements.subtitleLoading.classList.add("active");

        if (SP.state.currentTranscodeBase && SP.state.transcodedSubtitleTracks[val]) {
            var trackName = SP.state.transcodedSubtitleTracks[val].name || "Track " + (val + 1);
            showSubtitleProgress(trackName);

            var vttUrl = SP.state.currentTranscodeBase + "/subs_" + val + ".vtt";
            var track = document.createElement("track");
            track.kind = "subtitles";
            track.src = vttUrl;
            track.default = true;
            track.label = trackName;
            SP.elements.video.appendChild(track);
            track.addEventListener("load", function() {
                SP.elements.subtitleLoading.classList.remove("active");
                hideSubtitleProgress(true);
                if (SP.elements.video.textTracks.length > 0) {
                    SP.elements.video.textTracks[0].mode = "showing";
                }
            });
            track.addEventListener("error", function() {
                SP.elements.subtitleLoading.classList.remove("active");
                hideSubtitleProgress(false);
            });
            return;
        }

        if (this.value) {
            var subTrackName = this.options[this.selectedIndex].text || "Subtitle";
            showSubtitleProgress(subTrackName);

            var subTrack = document.createElement("track");
            subTrack.kind = "subtitles";
            subTrack.src = "/subs/" + this.value;
            subTrack.default = true;
            SP.elements.video.appendChild(subTrack);
            subTrack.addEventListener("load", function() {
                SP.elements.subtitleLoading.classList.remove("active");
                hideSubtitleProgress(true);
                if (SP.elements.video.textTracks.length > 0) {
                    SP.elements.video.textTracks[0].mode = "showing";
                }
            });
            subTrack.addEventListener("error", function() {
                SP.elements.subtitleLoading.classList.remove("active");
                hideSubtitleProgress(false);
            });
        }
    });

    // TEST: Double-click subtitle dropdown to test progress bar
    SP.elements.subtitleSelect.addEventListener('dblclick', function() {
        showSubtitleProgress("Test Track (Japanese)");
        setTimeout(function() { hideSubtitleProgress(true); }, 5000);
    });
}

function initResolutionControl() {
    SP.elements.resolutionSelect.addEventListener("change", function() {
        if (!SP.state.hls || !SP.state.hls.levels || SP.state.hls.levels.length === 0) return;

        var newResolution = this.value;
        if (newResolution === SP.state.currentResolution) return;

        resetMetrics();
        SP.state.currentResolution = newResolution;
        // Reset actual resolution - will be updated by LEVEL_SWITCHED event
        SP.state.actualResolution = null;
        updateModeDisplay();

        // For transcoded mode, filter levels by current audio track
        var candidateLevels = SP.state.isTranscoding
            ? getLevelsForAudioTrack(SP.state.currentAudioIdx)
            : SP.state.hls.levels.map(function(level, idx) { return { level: level, idx: idx }; });

        if (newResolution === 'auto') {
            // Only allow auto if there's a single audio track
            var audioIdxSet = {};
            SP.state.hls.levels.forEach(function(level) {
                audioIdxSet[getAudioIdxFromLevel(level)] = true;
            });
            var hasMultipleAudioTracks = SP.state.isTranscoding && Object.keys(audioIdxSet).length > 1;

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
    });
}

function initDownloadControl() {
    SP.elements.downloadBtn.addEventListener("click", function() {
        if (SP.state.currentFile) {
            var downloadUrl = "/direct/" + encodeFilePath(SP.state.currentFile);
            var a = document.createElement("a");
            a.href = downloadUrl;
            a.download = SP.state.currentFile.split("/").pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    });
}

function initControls() {
    initAudioControl();
    initSubtitleControl();
    initResolutionControl();
    initDownloadControl();
}
