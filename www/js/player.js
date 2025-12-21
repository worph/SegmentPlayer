/* Player - HLS playback, transcoding fallback */

async function resetMetrics() {
    try {
        await fetch("/transcode/reset-metrics");
    } catch (e) {}
}

async function playFile(filePath, fileName) {
    SP.state.currentFile = filePath;

    // Update URL hash
    updateUrlHash(filePath);
    resetMetrics();

    // Update sidebar active state
    document.querySelectorAll(".file-item").forEach(function(item) {
        item.classList.toggle("active", item.dataset.path === filePath);
    });

    SP.elements.nowPlaying.style.display = "block";
    SP.elements.nowPlayingName.textContent = fileName;
    setStatus("Loading...", "#4dabf7", true);

    SP.elements.downloadBtn.disabled = false;
    SP.elements.downloadBtn.title = "Download: " + fileName;

    if (SP.state.hls) {
        SP.state.hls.destroy();
        SP.state.hls = null;
    }

    SP.elements.audioSelect.innerHTML = '<option value="">Loading...</option>';
    SP.elements.audioSelect.disabled = true;
    SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>';
    SP.elements.subtitleSelect.disabled = true;
    SP.state.currentAudioIdx = 0;

    SP.elements.video.querySelectorAll("track").forEach(function(t) { t.remove(); });

    var videoSrc = "/hls/" + encodeURIComponent(filePath) + "/master.m3u8";

    if (Hls.isSupported()) {
        SP.state.hls = new Hls({
            debug: false,
            enableWorker: true,
            lowLatencyMode: false,
            maxBufferLength: 30,
            maxMaxBufferLength: 60
        });

        SP.state.hls.loadSource(videoSrc);
        SP.state.hls.attachMedia(SP.elements.video);

        SP.state.hls.on(Hls.Events.MANIFEST_PARSED, async function(event, data) {
            setStatus("Ready", "#51cf66");

            if (SP.state.hls.audioTracks && SP.state.hls.audioTracks.length > 0) {
                SP.elements.audioSelect.innerHTML = SP.state.hls.audioTracks.map(function(track, i) {
                    var label = track.name || track.lang || "Track " + (i + 1);
                    return '<option value="' + i + '">' + label + '</option>';
                }).join("");
                SP.elements.audioSelect.disabled = false;
                SP.elements.audioSelect.value = SP.state.hls.audioTrack;
                SP.state.currentAudioIdx = SP.state.hls.audioTrack;
            } else {
                SP.elements.audioSelect.innerHTML = '<option value="">Default</option>';
                SP.elements.audioSelect.disabled = true;
                SP.state.currentAudioIdx = 0;
            }

            var subs = await findSubtitles(filePath);
            if (subs.length > 0) {
                SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>' +
                    subs.map(function(sub) {
                        return '<option value="' + sub.path + '">' + sub.lang + '</option>';
                    }).join("");
                SP.elements.subtitleSelect.disabled = false;
            }

            SP.elements.video.play().catch(function() {});
        });

        SP.state.hls.on(Hls.Events.ERROR, function(event, data) {
            if (data.fatal) {
                switch (data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        tryTranscodedFallback(filePath, fileName);
                        break;
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        SP.state.hls.recoverMediaError();
                        break;
                    default:
                        setStatus("Error", "#ff6b6b");
                }
            }
        });

        SP.state.hls.on(Hls.Events.FRAG_LOADED, function() {
            setStatus("", "#51cf66");
        });

        // Track actual resolution when level changes (for auto mode display)
        SP.state.hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {
            if (SP.state.hls.levels && SP.state.hls.levels[data.level]) {
                SP.state.actualResolution = SP.state.hls.levels[data.level].height;
                updateQualityDisplay();
                // Update Auto option in dropdown
                var autoOption = SP.elements.resolutionSelect.querySelector('option[value="auto"]');
                if (autoOption) {
                    autoOption.textContent = "Auto (" + SP.state.actualResolution + "p)";
                }
            }
        });

    } else if (SP.elements.video.canPlayType("application/vnd.apple.mpegurl")) {
        SP.elements.video.src = videoSrc;
        SP.elements.video.addEventListener("loadedmetadata", function() {
            setStatus("Ready", "#51cf66");
            SP.elements.video.play().catch(function() {});
        });
    } else {
        setStatus("HLS not supported", "#ff6b6b");
    }
}

async function tryTranscodedFallback(filePath, fileName) {
    setStatus("Transcoding...", "#ffd43b", true);

    // Mark as transcode mode
    SP.state.isTranscoding = true;
    updateQualityDisplay();

    var transcodedSrc = "/transcode/" + encodeFilePath(filePath) + "/master.m3u8";

    try {
        var response = await fetch(transcodedSrc);
        if (response.ok) {
            playTranscoded(transcodedSrc, fileName, true);
        } else {
            setStatus("Error", "#ff6b6b");
        }
    } catch (err) {
        setStatus("Error", "#ff6b6b");
    }
}

function parseAndPopulateTracks(manifest) {
    SP.state.transcodedAudioTracks = [];
    SP.state.transcodedSubtitleTracks = [];

    var lines = manifest.split('\n');
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        // Parse audio tracks from EXT-X-MEDIA declarations
        if (line.startsWith('#EXT-X-MEDIA:TYPE=AUDIO')) {
            var name = (line.match(/NAME="([^"]+)"/) || [])[1] || 'Audio';
            var uri = (line.match(/URI="([^"]+)"/) || [])[1] || '';
            SP.state.transcodedAudioTracks.push({ name: name, uri: uri });
        }
        // Parse subtitles from EXT-X-MEDIA declarations
        else if (line.startsWith('#EXT-X-MEDIA:TYPE=SUBTITLES')) {
            var subName = (line.match(/NAME="([^"]+)"/) || [])[1] || 'Subtitle';
            var subUri = (line.match(/URI="([^"]+)"/) || [])[1] || '';
            SP.state.transcodedSubtitleTracks.push({ name: subName, uri: subUri });
        }
    }

    if (SP.state.transcodedAudioTracks.length > 0) {
        SP.elements.audioSelect.innerHTML = SP.state.transcodedAudioTracks.map(function(track, i) {
            return '<option value="' + i + '">' + track.name + '</option>';
        }).join("");
        SP.elements.audioSelect.disabled = false;
        SP.elements.audioSelect.value = "0";
        SP.state.currentAudioIdx = 0;
    }

    if (SP.state.transcodedSubtitleTracks.length > 0) {
        SP.elements.subtitleSelect.innerHTML = '<option value="-1">Off</option>' +
            SP.state.transcodedSubtitleTracks.map(function(track, i) {
                return '<option value="' + i + '">' + track.name + '</option>';
            }).join("");
        SP.elements.subtitleSelect.disabled = false;
    }
}

async function playTranscoded(url, fileName, isActiveTranscode) {
    if (isActiveTranscode === undefined) isActiveTranscode = false;

    if (SP.state.hls) {
        SP.state.hls.destroy();
    }

    setStatus(isActiveTranscode ? "Transcoding..." : "Transcoded", "#ffd43b", isActiveTranscode);

    SP.state.currentTranscodeBase = url.replace("/master.m3u8", "");

    try {
        var response = await fetch(url);
        var manifest = await response.text();
        parseAndPopulateTracks(manifest);
    } catch (e) {}

    SP.elements.resolutionSelect.disabled = false;

    SP.state.hls = new Hls({
        debug: false,
        enableWorker: true,
        maxBufferLength: SP.config.MAX_BUFFER_LENGTH,
        maxMaxBufferLength: SP.config.MAX_BUFFER_LENGTH * 2
    });

    // Load the master playlist - HLS.js will handle audio track switching
    SP.state.hls.loadSource(url);
    SP.state.hls.attachMedia(SP.elements.video);

    SP.state.hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {
        setStatus("", "#51cf66");

        // Populate resolution dropdown from HLS.js levels
        if (SP.state.hls.levels && SP.state.hls.levels.length > 0) {
            // Get unique heights (avoid duplicates from multiple audio tracks)
            var heightSet = {};
            SP.state.hls.levels.forEach(function(level) {
                if (!heightSet[level.height]) {
                    heightSet[level.height] = true;
                }
            });
            var uniqueHeights = Object.keys(heightSet).map(Number).sort(function(a, b) { return b - a; });

            // First option is "Original" (highest quality)
            var originalHeight = uniqueHeights[0];
            SP.elements.resolutionSelect.innerHTML =
                '<option value="auto">Auto</option>' +
                '<option value="' + originalHeight + '">Original (' + originalHeight + 'p)</option>' +
                uniqueHeights.slice(1).map(function(h) {
                    return '<option value="' + h + '">' + h + 'p</option>';
                }).join("");
            SP.elements.resolutionSelect.disabled = false;
            SP.elements.resolutionSelect.value = "auto";
            SP.state.currentResolution = "auto";
            SP.state.actualResolution = null;
            updateQualityDisplay();

            // For multi-audio content, we can't use ABR because it might pick a different audio track
            // Instead, find the highest quality level for audio track 0
            var hasMultipleAudioTracks = SP.state.transcodedAudioTracks && SP.state.transcodedAudioTracks.length > 1;
            if (hasMultipleAudioTracks) {
                // Find levels for audio track 0 and pick the highest quality
                var a0Levels = SP.state.hls.levels
                    .map(function(level, idx) { return { level: level, idx: idx }; })
                    .filter(function(item) {
                        var url = item.level.url && item.level.url[0];
                        return url && url.match(/stream_a0_/);
                    })
                    .sort(function(a, b) { return b.level.height - a.level.height; });

                if (a0Levels.length > 0) {
                    SP.state.hls.currentLevel = a0Levels[0].idx;
                } else {
                    SP.state.hls.currentLevel = -1;
                }
            } else {
                // Single audio track - safe to use ABR
                SP.state.hls.currentLevel = -1;
            }
        }

        // For transcoded mode with muxed audio, keep the dropdown from parseAndPopulateTracks()
        // since HLS.js audioTracks only shows tracks from the current audio group (1 track per group)
        // Skip this block if we already have transcodedAudioTracks parsed from manifest
        if (!SP.state.transcodedAudioTracks || SP.state.transcodedAudioTracks.length === 0) {
            // Populate audio dropdown from HLS.js audio tracks (for non-muxed audio)
            if (SP.state.hls.audioTracks && SP.state.hls.audioTracks.length > 0) {
                SP.elements.audioSelect.innerHTML = SP.state.hls.audioTracks.map(function(track, i) {
                    var label = track.name || track.lang || "Audio " + (i + 1);
                    return '<option value="' + i + '">' + label + '</option>';
                }).join("");
                SP.elements.audioSelect.disabled = false;
                SP.elements.audioSelect.value = SP.state.hls.audioTrack.toString();
                SP.state.currentAudioIdx = SP.state.hls.audioTrack;
            }
        }

        SP.elements.video.play().catch(function() {});
    });

    SP.state.hls.on(Hls.Events.FRAG_LOADING, function() {
        if (isActiveTranscode) {
            setStatus("", "#ffd43b", true);
        }
    });

    SP.state.hls.on(Hls.Events.FRAG_LOADED, function() {
        setStatus("", "#51cf66");
    });

    SP.state.hls.on(Hls.Events.ERROR, function(event, data) {
        if (data.fatal) {
            setStatus("Error", "#ff6b6b");
        }
    });

    // Track actual resolution when level changes (for auto mode display)
    SP.state.hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {
        if (SP.state.hls.levels && SP.state.hls.levels[data.level]) {
            SP.state.actualResolution = SP.state.hls.levels[data.level].height;
            updateQualityDisplay();
            // Update Auto option in dropdown
            var autoOption = SP.elements.resolutionSelect.querySelector('option[value="auto"]');
            if (autoOption) {
                autoOption.textContent = "Auto (" + SP.state.actualResolution + "p)";
            }
        }
    });
}
