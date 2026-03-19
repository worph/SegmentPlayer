/* Player - Direct, HLS remux, and transcoding playback */

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

    // Cleanup existing playback
    if (SP.state.hls) {
        SP.state.hls.destroy();
        SP.state.hls = null;
    }
    SP.elements.video.removeAttribute('src');
    SP.elements.video.load();

    SP.elements.audioSelect.innerHTML = '<option value="">Loading...</option>';
    SP.elements.audioSelect.disabled = true;
    SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>';
    SP.elements.subtitleSelect.disabled = true;
    SP.state.currentAudioIdx = 0;
    SP.state.isTranscoding = false;

    SP.elements.video.querySelectorAll("track").forEach(function(t) { t.remove(); });

    // Probe the file for codec info
    SP.state.probeData = await getProbeData(filePath);

    // Determine playback mode
    var mode;
    if (SP.state.playbackMode === "auto") {
        mode = SP.state.probeData ? detectPlaybackMode(SP.state.probeData) : "remux";
    } else if (SP.state.playbackMode === "direct") {
        mode = "direct";
    } else if (SP.state.playbackMode === "transcode") {
        mode = "transcode";
    } else {
        mode = "remux";
    }

    SP.state.activePlaybackMode = mode;

    // Route to appropriate handler
    switch (mode) {
        case "direct":
            playDirect(filePath, fileName);
            break;
        case "remux":
            playRemux(filePath, fileName);
            break;
        case "transcode":
            var transcodedSrc = "/transcode/" + encodeFilePath(filePath) + "/master.m3u8";
            SP.state.isTranscoding = true;
            updateQualityDisplay();
            playTranscoded(transcodedSrc, fileName, true);
            break;
    }
}

function playDirect(filePath, fileName) {
    var videoSrc = "/direct/" + encodeFilePath(filePath);

    setStatus("Direct", "#51cf66");

    // Disable resolution select — no quality options in direct mode
    SP.elements.resolutionSelect.innerHTML = '<option value="original">Original</option>';
    SP.elements.resolutionSelect.disabled = true;

    // Audio tracks — in direct mode, browser handles audio natively
    if (SP.state.probeData && SP.state.probeData.audio_tracks > 1) {
        // Multiple audio tracks — browser support is inconsistent
        // Show count but note that track switching may not work
        var opts = '';
        for (var i = 0; i < SP.state.probeData.audio_tracks; i++) {
            opts += '<option value="' + i + '">Track ' + (i + 1) + '</option>';
        }
        SP.elements.audioSelect.innerHTML = opts;
        SP.elements.audioSelect.disabled = false;
    } else {
        SP.elements.audioSelect.innerHTML = '<option value="">Default</option>';
        SP.elements.audioSelect.disabled = true;
    }

    // Set video source — native <video> element handles playback
    SP.elements.video.src = videoSrc;

    SP.elements.video.onloadedmetadata = function() {
        setStatus("Direct", "#51cf66");

        // Try to use native audioTracks API if available
        if (SP.elements.video.audioTracks && SP.elements.video.audioTracks.length > 1) {
            var opts = '';
            for (var i = 0; i < SP.elements.video.audioTracks.length; i++) {
                var track = SP.elements.video.audioTracks[i];
                var label = track.label || track.language || 'Track ' + (i + 1);
                opts += '<option value="' + i + '">' + label + '</option>';
            }
            SP.elements.audioSelect.innerHTML = opts;
            SP.elements.audioSelect.disabled = false;
        }

        SP.elements.video.play().catch(function() {});
    };

    SP.elements.video.onerror = function() {
        // Auto-fallback: try remux, then transcode
        console.log("Direct playback failed, falling back...");
        if (SP.state.probeData) {
            var vcodec = SP.state.probeData.video_codec;
            var acodec = SP.state.probeData.audio_codec;
            if ((vcodec === 'h264' || vcodec === 'avc1') && (acodec === 'aac' || acodec === 'mp3')) {
                SP.state.activePlaybackMode = "remux";
                playRemux(filePath, fileName);
                return;
            }
        }
        SP.state.activePlaybackMode = "transcode";
        SP.state.isTranscoding = true;
        updateQualityDisplay();
        tryTranscodedFallback(filePath, fileName);
    };

    // Load external subtitles, then merge with embedded subtitles from probe
    loadExternalSubtitles(filePath).then(function() {
        if (SP.state.probeData && SP.state.probeData.subtitle_list && SP.state.probeData.subtitle_list.length > 0) {
            var embeddedOpts = SP.state.probeData.subtitle_list.map(function(sub) {
                var label = sub.title || (sub.language !== 'und' ? sub.language.toUpperCase() : 'Track ' + (sub.index + 1));
                return '<option value="embedded:' + sub.index + '">' + label + ' (embedded)</option>';
            }).join('');

            // Append embedded subs to whatever external subs are already in the dropdown
            SP.elements.subtitleSelect.innerHTML += embeddedOpts;
            SP.elements.subtitleSelect.disabled = false;
        }
    });
}

async function loadExternalSubtitles(filePath) {
    var subs = await findSubtitles(filePath);
    if (subs.length > 0) {
        SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>' +
            subs.map(function(sub) {
                return '<option value="' + sub.path + '">' + sub.lang + '</option>';
            }).join("");
        SP.elements.subtitleSelect.disabled = false;
    }
}

function playRemux(filePath, fileName) {
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

        // Track actual resolution when level changes
        SP.state.hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {
            if (SP.state.hls.levels && SP.state.hls.levels[data.level]) {
                SP.state.actualResolution = SP.state.hls.levels[data.level].height;
                updateQualityDisplay();
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

    SP.state.isTranscoding = true;
    SP.state.activePlaybackMode = "transcode";
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
        if (line.startsWith('#EXT-X-MEDIA:TYPE=AUDIO')) {
            var name = (line.match(/NAME="([^"]+)"/) || [])[1] || 'Audio';
            var uri = (line.match(/URI="([^"]+)"/) || [])[1] || '';
            SP.state.transcodedAudioTracks.push({ name: name, uri: uri });
        }
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

    SP.state.hls.loadSource(url);
    SP.state.hls.attachMedia(SP.elements.video);

    SP.state.hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {
        setStatus("", "#51cf66");

        if (SP.state.hls.levels && SP.state.hls.levels.length > 0) {
            var heightSet = {};
            SP.state.hls.levels.forEach(function(level) {
                if (!heightSet[level.height]) {
                    heightSet[level.height] = true;
                }
            });
            var uniqueHeights = Object.keys(heightSet).map(Number).sort(function(a, b) { return b - a; });

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

            var hasMultipleAudioTracks = SP.state.transcodedAudioTracks && SP.state.transcodedAudioTracks.length > 1;
            if (hasMultipleAudioTracks) {
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
                SP.state.hls.currentLevel = -1;
            }
        }

        if (!SP.state.transcodedAudioTracks || SP.state.transcodedAudioTracks.length === 0) {
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

    SP.state.hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {
        if (SP.state.hls.levels && SP.state.hls.levels[data.level]) {
            SP.state.actualResolution = SP.state.hls.levels[data.level].height;
            updateQualityDisplay();
            var autoOption = SP.elements.resolutionSelect.querySelector('option[value="auto"]');
            if (autoOption) {
                autoOption.textContent = "Auto (" + SP.state.actualResolution + "p)";
            }
        }
    });
}
