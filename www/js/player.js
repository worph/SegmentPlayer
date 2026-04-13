/* Player - Direct, HLS remux, client-side demux, and transcoding playback */

async function resetMetrics() {
    try {
        await fetch("/transcode/reset-metrics");
    } catch (e) {}
    // Reset local per-session HLS stats so level-switch counts don't leak
    // across file loads.
    SP.state.hlsStats = { levelSwitches: 0, lastFragMs: 0, lastFragBytes: 0, fragLoadedAt: 0 };
}

// Install HLS telemetry hooks on a newly-created hls.js instance. Shared by
// playRemux and playTranscoded.
function attachHlsStatsHooks(hls) {
    if (!hls) return;
    SP.state.hlsStats = SP.state.hlsStats || { levelSwitches: 0, lastFragMs: 0, lastFragBytes: 0, fragLoadedAt: 0 };
    var fragStartMap = {};
    hls.on(Hls.Events.FRAG_LOADING, function(event, data) {
        var url = data && data.frag && data.frag.url;
        if (url) fragStartMap[url] = performance.now();
    });
    hls.on(Hls.Events.FRAG_LOADED, function(event, data) {
        var url = data && data.frag && data.frag.url;
        var startedAt = url ? fragStartMap[url] : undefined;
        if (startedAt) {
            SP.state.hlsStats.lastFragMs = Math.round(performance.now() - startedAt);
            delete fragStartMap[url];
        }
        if (data && data.payload) {
            SP.state.hlsStats.lastFragBytes = data.payload.byteLength || 0;
        } else if (data && data.frag && data.frag.stats && data.frag.stats.total) {
            SP.state.hlsStats.lastFragBytes = data.frag.stats.total;
        }
        SP.state.hlsStats.fragLoadedAt = performance.now();
    });
    hls.on(Hls.Events.LEVEL_SWITCHED, function() {
        SP.state.hlsStats.levelSwitches = (SP.state.hlsStats.levelSwitches || 0) + 1;
    });
}

// Reset all per-file UI selection and playback state to a clean slate.
// Called by playFileSetup before loading a new file and by switchToBrowseMode
// when leaving the player. Returns the new loadToken so callers can capture it
// and bail from stale async work.
function resetPlaybackUIState() {
    // Tear down any active streaming pipeline
    if (SP.state.hls) {
        SP.state.hls.destroy();
        SP.state.hls = null;
    }
    if (SP.state.clientPlayer) {
        SP.state.clientPlayer.cleanup();
        SP.state.clientPlayer = null;
    }

    // Reset video element
    SP.elements.video.removeAttribute('src');
    SP.elements.video.onerror = null;
    SP.elements.video.onloadedmetadata = null;
    SP.elements.video.load();
    removeAllTracks(SP.elements.video);

    // Reset playback flags
    SP.state.isClientSide = false;
    SP.state.isTranscoding = false;

    // Reset audio dropdown to a neutral "Loading..." state. .value is reset
    // explicitly because replacing innerHTML alone leaves the property in
    // an awkward state where the dropdown can still report a stale numeric
    // value from the previous file.
    SP.elements.audioSelect.innerHTML = '<option value="">Loading...</option>';
    SP.elements.audioSelect.disabled = true;
    SP.elements.audioSelect.value = "";
    SP.state.currentAudioIdx = 0;

    // Reset subtitle dropdown to "Off"
    SP.elements.subtitleSelect.innerHTML = '<option value="">Off</option>';
    SP.elements.subtitleSelect.disabled = true;
    SP.elements.subtitleSelect.value = "";

    // Reset resolution dropdown — every tier rebuilds this with its own
    // options, so collapse to a placeholder for now.
    SP.elements.resolutionSelect.innerHTML = '<option value="auto">Auto</option>';
    SP.elements.resolutionSelect.disabled = true;
    SP.elements.resolutionSelect.value = "auto";
    SP.state.currentResolution = "auto";
    SP.state.actualResolution = null;

    // Reset transcoded-track caches (otherwise transcode→remux switch leaves
    // stale entries that the change handler will misuse).
    SP.state.transcodedAudioTracks = [];
    SP.state.transcodedSubtitleTracks = [];
    SP.state.currentTranscodeBase = "";

    // Reset probe caches for the in-flight file
    SP.state.currentProbe = null;
    SP.state.probeData = null;

    // Cancel any in-flight subtitle progress polling
    if (typeof hideSubtitleProgress === "function") {
        hideSubtitleProgress(false);
    }
    if (SP.elements.subtitleLoading) {
        SP.elements.subtitleLoading.classList.remove("active");
    }

    // Bump generation token so stale async work bails out
    SP.state.loadToken = (SP.state.loadToken || 0) + 1;
    return SP.state.loadToken;
}

// Populate the audio dropdown from a list of tracks. Always sets .value and
// SP.state.currentAudioIdx so the UI and state never drift apart.
function setAudioTracks(tracks, getLabel, defaultIdx) {
    if (!tracks || tracks.length === 0) {
        SP.elements.audioSelect.innerHTML = '<option value="">Default</option>';
        SP.elements.audioSelect.disabled = true;
        SP.elements.audioSelect.value = "";
        SP.state.currentAudioIdx = 0;
        return;
    }
    var idx = (typeof defaultIdx === "number" && defaultIdx >= 0 && defaultIdx < tracks.length) ? defaultIdx : 0;
    SP.elements.audioSelect.innerHTML = tracks.map(function(t, i) {
        return '<option value="' + i + '">' + getLabel(t, i) + '</option>';
    }).join("");
    SP.elements.audioSelect.disabled = tracks.length <= 1;
    SP.elements.audioSelect.value = String(idx);
    SP.state.currentAudioIdx = idx;
}

// Populate the subtitle dropdown from a list of {value, label} options.
// Always prepends an "Off" entry and resets .value so the previous video's
// selection cannot leak.
function setSubtitleTracks(options) {
    var html = '<option value="">Off</option>';
    if (options && options.length > 0) {
        html += options.map(function(opt) {
            return '<option value="' + opt.value + '">' + opt.label + '</option>';
        }).join("");
    }
    SP.elements.subtitleSelect.innerHTML = html;
    SP.elements.subtitleSelect.value = "";
    SP.elements.subtitleSelect.disabled = !(options && options.length > 0);
}

// Common setup shared by all playback tiers. Returns the loadToken for this
// load; tier handlers should capture it and check it across await points.
function playFileSetup(filePath, fileName) {
    var token = resetPlaybackUIState();

    SP.state.currentFile = filePath;
    updateUrlHash(filePath);
    resetMetrics();

    document.querySelectorAll(".file-item").forEach(function(item) {
        item.classList.toggle("active", item.dataset.path === filePath);
    });

    SP.elements.nowPlaying.style.display = "block";
    SP.elements.nowPlayingName.textContent = fileName;
    setStatus("Loading...", "#4dabf7", true);

    SP.elements.downloadBtn.disabled = false;
    SP.elements.downloadBtn.title = "Download: " + fileName;

    return token;
}

async function playFile(filePath, fileName) {
    var token = playFileSetup(filePath, fileName);

    // Probe the file for codec info
    var probe = await getProbeData(filePath);

    // If the user clicked another file while the probe was in flight, drop
    // this one — the new load already reset state and is in progress.
    if (token !== SP.state.loadToken) return;

    SP.state.probeData = probe;
    SP.state.currentProbe = probe;

    // Determine playback mode
    var mode;
    if (SP.state.playbackMode === "auto") {
        if (SP.state.probeData) {
            var tier = chooseTier(SP.state.probeData);
            mode = tier === "direct" ? "direct" : tier === "client" ? "client" : tier === "hls" ? "remux" : "transcode";
        } else {
            mode = "remux";
        }
    } else if (SP.state.playbackMode === "direct") {
        mode = "direct";
    } else if (SP.state.playbackMode === "client") {
        mode = "client";
    } else if (SP.state.playbackMode === "transcode") {
        mode = "transcode";
    } else {
        mode = "remux";
    }

    SP.state.activePlaybackMode = mode;
    updateAutoModeLabel();

    // Route to appropriate handler
    switch (mode) {
        case "direct":
            playDirect(filePath, fileName, token);
            break;
        case "client":
            playFileClient(filePath, fileName, SP.state.probeData, token);
            break;
        case "remux":
            playRemux(filePath, fileName, token);
            break;
        case "transcode":
            var transcodedSrc = "/transcode/" + encodeFilePath(filePath) + "/master.m3u8";
            SP.state.isTranscoding = true;
            updateQualityDisplay();
            playTranscoded(transcodedSrc, fileName, true, token);
            break;
    }
}

// Direct mode: native <video> element plays the raw file
function playDirect(filePath, fileName, token) {
    if (token === undefined) token = SP.state.loadToken;
    var videoSrc = "/direct/" + encodeFilePath(filePath);

    setStatus("Direct", "#51cf66");

    // Disable resolution select — no quality options in direct mode
    SP.elements.resolutionSelect.innerHTML = '<option value="original">Original</option>';
    SP.elements.resolutionSelect.disabled = true;
    SP.elements.resolutionSelect.value = "original";

    // Audio: setAudioTracks(null,...) renders the neutral "Default" placeholder
    setAudioTracks(null);

    // Set video source — native <video> element handles playback
    SP.elements.video.src = videoSrc;

    SP.elements.video.onloadedmetadata = function() {
        if (token !== SP.state.loadToken) return;
        setStatus("Direct", "#51cf66");

        // Only expose audio switching if browser supports the audioTracks API
        var native = SP.elements.video.audioTracks;
        if (native && native.length > 1) {
            var tracks = [];
            for (var i = 0; i < native.length; i++) tracks.push(native[i]);
            setAudioTracks(tracks, function(track, i) {
                return track.label || track.language || 'Track ' + (i + 1);
            }, 0);
            // Keep the native enabled flags in sync with the dropdown so the
            // browser actually plays track 0 from the start.
            for (var j = 0; j < native.length; j++) {
                native[j].enabled = (j === 0);
            }
        }

        SP.elements.video.play().catch(function() {});
    };

    SP.elements.video.onerror = function() {
        if (token !== SP.state.loadToken) return;
        // Auto-fallback: try remux, then transcode
        SP.log.warn("Client", "Direct playback failed, falling back...");
        if (SP.state.probeData) {
            var vcodec = SP.state.probeData.video && SP.state.probeData.video.codec;
            var acodec = SP.state.probeData.audio && SP.state.probeData.audio[0] && SP.state.probeData.audio[0].codec;
            if ((vcodec === 'h264' || vcodec === 'avc1') && (acodec === 'aac' || acodec === 'mp3')) {
                SP.state.activePlaybackMode = "remux";
                updateAutoModeLabel();
                playRemux(filePath, fileName, token);
                return;
            }
        }
        SP.state.activePlaybackMode = "transcode";
        updateAutoModeLabel();
        SP.state.isTranscoding = true;
        updateQualityDisplay();
        tryTranscodedFallback(filePath, fileName, token);
    };

    // Load external subtitles, then merge with embedded subtitles from probe.
    // Build a single combined option list and assign once via setSubtitleTracks
    // so we never end up with stale entries from a previous video.
    findSubtitles(filePath).then(function(externalSubs) {
        if (token !== SP.state.loadToken) return;
        var options = [];
        externalSubs.forEach(function(sub) {
            options.push({ value: sub.path, label: sub.lang });
        });
        if (SP.state.probeData && SP.state.probeData.subtitles) {
            SP.state.probeData.subtitles.forEach(function(sub) {
                var label = sub.title || (sub.language !== 'und' ? sub.language.toUpperCase() : 'Track ' + (sub.index + 1));
                options.push({ value: 'embedded:' + sub.index, label: label + ' (embedded)' });
            });
        }
        setSubtitleTracks(options);
    });
}

// Client-side demux + transmux via libav.js + MediaSource
async function playFileClient(filePath, fileName, probeData, token) {
    if (token === undefined) token = SP.state.loadToken;
    setStatus("Analyzing...", "#4dabf7", true);
    SP.state.isClientSide = true;
    SP.state.activePlaybackMode = "client";
    updateQualityDisplay();

    function fallback(reason) {
        if (token !== SP.state.loadToken) return;
        SP.log.warn("Client", "Falling back:", reason);
        SP.state.isClientSide = false;
        if (SP.state.clientPlayer) {
            SP.state.clientPlayer.cleanup();
            SP.state.clientPlayer = null;
        }
        setStatus("Falling back...", "#ffd43b", true);
        // Try Direct mode first (browser may play the file natively even if MSE can't)
        // then fall back to server transcode as last resort
        var vcodec = probeData && probeData.video && probeData.video.codec;
        if (vcodec) {
            var testVideo = document.createElement('video');
            // Test common container MIME types the browser might handle natively
            var mimeTests = ['video/x-matroska', 'video/webm', 'video/mp4'];
            var canDirect = mimeTests.some(function(m) {
                return testVideo.canPlayType(m) !== '';
            });
            if (canDirect) {
                SP.log.warn("Client", "Falling back to Direct mode");
                // Reset video element and wait a tick before Direct mode
                SP.elements.video.removeAttribute('src');
                SP.elements.video.onerror = null;
                SP.elements.video.onloadedmetadata = null;
                SP.elements.video.load();
                SP.state.activePlaybackMode = "direct";
                updateAutoModeLabel();
                setTimeout(function() {
                    if (token !== SP.state.loadToken) return;
                    playDirect(filePath, fileName, token);
                }, 100);
                return;
            }
        }
        SP.log.warn("Client", "Falling back to server transcode");
        tryTranscodedFallback(filePath, fileName, token);
    }

    try {
        // Check prerequisites
        if (typeof ClientPlayer === "undefined") {
            throw new Error("Client-side player module not loaded");
        }
        if (typeof LibAV === "undefined" && typeof loadVendor === "undefined") {
            throw new Error("LibAV vendor library not available");
        }
        if (typeof MediaSource === "undefined") {
            throw new Error("MediaSource API not supported in this browser");
        }
        if (typeof canMSEHandleMP4 === "function" && !canMSEHandleMP4()) {
            throw new Error("Browser MSE does not support MP4 output — client-side transmux requires MSE MP4 support");
        }
        // Check if MSE supports this specific video codec
        if (probeData.video) {
            var testVideoMime = buildMSEMimeType("video", probeData.video);
            if (testVideoMime && !MediaSource.isTypeSupported(testVideoMime)) {
                throw new Error("MSE does not support " + probeData.video.codec + " — use Direct or Transcode mode");
            }
        }

        var player = new ClientPlayer(SP.elements.video);
        SP.state.clientPlayer = player;
        await player.load(filePath, probeData, 0);
        if (token !== SP.state.loadToken) return;
        populateAudioFromProbe(probeData);
        populateSubtitlesFromProbe(probeData, filePath, token);

        // Quality dropdown: only Original in client-side mode
        var h = (probeData.video && probeData.video.height) || "?";
        SP.elements.resolutionSelect.innerHTML = '<option value="original">Original (' + h + 'p)</option>';
        SP.elements.resolutionSelect.disabled = true;
        SP.elements.resolutionSelect.value = "original";
        SP.state.currentResolution = "original";
        SP.state.actualResolution = probeData.video ? probeData.video.height : null;
        updateQualityDisplay();

        setStatus("Client-side", "#51cf66");
        SP.elements.video.play().catch(function() {});

        // Watchdog: if no video frames appear within this window, fallback.
        // For HEVC open-GOP content, the muxer has to wait for the first IDR
        // after the opening one to close the initial fragment (CRAs inside the
        // GOP are deliberately not marked as sync samples to avoid the MSE
        // "RAP with later PTS than dependent non-key" warning). Anime HEVC
        // commonly has 10+ second IDR spacing, so 5 s is too tight.
        var watchdog = setTimeout(function() {
            if (SP.state.clientPlayer && SP.elements.video.readyState < 3) {
                SP.log.warn("Client", "Watchdog: no playable data after 20s");
                fallback("Watchdog timeout — no playable data");
            }
        }, 20000);

        // Clear watchdog once video starts playing
        SP.elements.video.addEventListener("playing", function clearWatchdog() {
            clearTimeout(watchdog);
            SP.elements.video.removeEventListener("playing", clearWatchdog);
        }, { once: true });

    } catch (err) {
        fallback(err.message || err);
    }
}

// Populate audio dropdown from probe data (for client-side mode)
function populateAudioFromProbe(probeData) {
    var tracks = probeData && probeData.audio;
    setAudioTracks(tracks, function(track, i) {
        return track.title || track.language || "Track " + (i + 1);
    }, 0);
}

// Populate subtitle dropdown from probe data (for client-side mode)
async function populateSubtitlesFromProbe(probeData, filePath, token) {
    var options = [];

    // Embedded subtitles from probe
    if (probeData && probeData.subtitles && probeData.subtitles.length > 0) {
        probeData.subtitles.forEach(function(sub, i) {
            var label = sub.title || sub.language || "Track " + (i + 1);
            options.push({ value: 'embedded:' + i, label: label });
        });
    }

    // External subtitle files
    var externalSubs = await findSubtitles(filePath);
    if (token !== undefined && token !== SP.state.loadToken) return;
    externalSubs.forEach(function(sub) {
        options.push({ value: sub.path, label: sub.lang + ' (ext)' });
    });

    setSubtitleTracks(options);
}

// HLS remux via nginx-vod-module (H.264+AAC in compatible containers)
function playRemux(filePath, fileName, token) {
    if (token === undefined) token = SP.state.loadToken;
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
        attachHlsStatsHooks(SP.state.hls);

        SP.state.hls.on(Hls.Events.MANIFEST_PARSED, async function(event, data) {
            if (token !== SP.state.loadToken) return;
            setStatus("Ready", "#51cf66");

            var hlsTracks = SP.state.hls.audioTracks;
            if (hlsTracks && hlsTracks.length > 0) {
                setAudioTracks(hlsTracks, function(t, i) {
                    return t.name || t.lang || "Track " + (i + 1);
                }, SP.state.hls.audioTrack);
            } else {
                setAudioTracks(null);
            }

            var subs = await findSubtitles(filePath);
            if (token !== SP.state.loadToken) return;
            var subOptions = subs.map(function(sub) {
                return { value: sub.path, label: sub.lang };
            });
            setSubtitleTracks(subOptions);

            SP.elements.video.play().catch(function() {});
        });

        var bufferErrorCount = 0;
        var remuxHls = SP.state.hls; // capture reference to THIS instance
        var remuxFallingBack = false;

        SP.state.hls.on(Hls.Events.ERROR, function(event, data) {
            if (remuxFallingBack) return;
            // Ignore events if this HLS instance was replaced by a fallback
            if (SP.state.hls !== remuxHls) return;
            if (token !== SP.state.loadToken) return;

            if (data.fatal) {
                switch (data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        remuxFallingBack = true;
                        tryTranscodedFallback(filePath, fileName, token);
                        break;
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        SP.state.hls.recoverMediaError();
                        break;
                    default:
                        setStatus("Error", "#ff6b6b");
                }
            } else if (data.details === "bufferAppendingError" || data.details === "bufferAppendError") {
                bufferErrorCount++;
                if (bufferErrorCount >= 10 && !remuxFallingBack) {
                    remuxFallingBack = true;
                    SP.log.warn("Remux", "Too many buffer errors (" + bufferErrorCount + "), falling back to transcode");
                    // Destroy synchronously to stop the error flood, then schedule fallback
                    if (SP.state.hls === remuxHls) SP.state.hls = null;
                    remuxHls.destroy();
                    setTimeout(function() {
                        if (token !== SP.state.loadToken) return;
                        tryTranscodedFallback(filePath, fileName, token);
                    }, 0);
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

async function tryTranscodedFallback(filePath, fileName, token) {
    if (token === undefined) token = SP.state.loadToken;
    setStatus("Transcoding...", "#ffd43b", true);

    SP.state.isTranscoding = true;
    SP.state.activePlaybackMode = "transcode";
    updateAutoModeLabel();
    updateQualityDisplay();

    var transcodedSrc = "/transcode/" + encodeFilePath(filePath) + "/master.m3u8";

    try {
        var response = await fetch(transcodedSrc);
        if (token !== SP.state.loadToken) return;
        if (response.ok) {
            playTranscoded(transcodedSrc, fileName, true, token);
        } else {
            setStatus("Error", "#ff6b6b");
        }
    } catch (err) {
        if (token !== SP.state.loadToken) return;
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
        setAudioTracks(SP.state.transcodedAudioTracks, function(t) {
            return t.name;
        }, 0);
    }

    if (SP.state.transcodedSubtitleTracks.length > 0) {
        // Transcoded subtitles use "-1" as the Off sentinel because the change
        // handler treats that value specially. Build the dropdown directly
        // here instead of via setSubtitleTracks.
        SP.elements.subtitleSelect.innerHTML = '<option value="-1">Off</option>' +
            SP.state.transcodedSubtitleTracks.map(function(track, i) {
                return '<option value="' + i + '">' + track.name + '</option>';
            }).join("");
        SP.elements.subtitleSelect.value = "-1";
        SP.elements.subtitleSelect.disabled = false;
    }
}

async function playTranscoded(url, fileName, isActiveTranscode, token) {
    if (isActiveTranscode === undefined) isActiveTranscode = false;
    if (token === undefined) token = SP.state.loadToken;

    if (SP.state.hls) {
        SP.state.hls.destroy();
    }

    setStatus(isActiveTranscode ? "Transcoding..." : "Transcoded", "#ffd43b", isActiveTranscode);

    SP.state.currentTranscodeBase = url.replace("/master.m3u8", "");

    try {
        var response = await fetch(url);
        var manifest = await response.text();
        if (token !== SP.state.loadToken) return;
        parseAndPopulateTracks(manifest);
    } catch (e) {
        if (token !== SP.state.loadToken) return;
    }

    SP.elements.resolutionSelect.disabled = false;

    SP.state.hls = new Hls({
        debug: false,
        enableWorker: true,
        maxBufferLength: SP.config.MAX_BUFFER_LENGTH,
        maxMaxBufferLength: SP.config.MAX_BUFFER_LENGTH * 2
    });

    SP.state.hls.loadSource(url);
    SP.state.hls.attachMedia(SP.elements.video);
    attachHlsStatsHooks(SP.state.hls);

    SP.state.hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {
        if (token !== SP.state.loadToken) return;
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
                setAudioTracks(SP.state.hls.audioTracks, function(t, i) {
                    return t.name || t.lang || "Audio " + (i + 1);
                }, SP.state.hls.audioTrack);
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
