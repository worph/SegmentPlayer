/* Custom player UI — transport bar inside the video container.
 *
 * Owns the bottom control bar (play/pause, seek, time, volume, fullscreen,
 * PiP) and the now-playing/loading overlays. The settings cog is rendered
 * here but its menu logic lives in settings-menu.js.
 *
 * Constraints respected:
 *  - All seeks go through `video.currentTime = X`, which triggers the
 *    existing `seeking` listener in client-player.js:341 and the MediaSource
 *    rebuild path. We never touch `video.src` or any ClientPlayer internals.
 *  - No mutation of SP.state outside what the <video> element already drives.
 */

(function() {
    // ─── Inline SVG icons (kept here so the bar has no external assets) ───
    var ICON = {
        play:    '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
        pause:   '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>',
        volHi:   '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>',
        volMute: '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.17v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>',
        fsEnter: '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>',
        fsExit:  '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/></svg>',
        pip:     '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M19 11h-8v6h8v-6zm4 8V4.98C23 3.88 22.1 3 21 3H3c-1.1 0-2 .88-2 1.98V19c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2zm-2 .02H3V4.97h18v14.05z"/></svg>',
        settings: '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94 0 .31.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>'
    };

    // ─── Module state ───
    var seekDragging = false;
    var preMuteVolume = 1;

    // ─── Public API exposed for player.js / settings-menu.js ───
    window.setNowPlayingTitle = function(name) {
        var el = SP.elements.nowPlayingOverlay;
        if (!el) return;
        if (name) {
            el.textContent = name;
            el.hidden = false;
        } else {
            el.hidden = true;
        }
    };

    window.setPlayerLoading = function(loading) {
        var el = SP.elements.playerLoading;
        if (!el) return;
        el.classList.toggle("is-visible", !!loading);
    };

    window.setPlayerModeBadge = function(mode) {
        var bar = SP.elements.playerControls;
        if (!bar) return;
        var badge = bar.querySelector(".player-controls__mode-badge");
        if (!badge) return;
        badge.dataset.mode = mode || "";
        badge.textContent = mode || "";
    };

    // ─── DOM construction ───
    function buildBar() {
        var bar = SP.elements.playerControls;
        if (!bar) return;
        bar.innerHTML =
            '<div class="player-controls__seek" data-role="seek">' +
                '<div class="player-controls__seek-track">' +
                    '<div class="player-controls__seek-buffered" data-role="buffered"></div>' +
                    '<div class="player-controls__seek-played" data-role="played"></div>' +
                '</div>' +
                '<div class="player-controls__seek-thumb" data-role="thumb"></div>' +
            '</div>' +
            '<div class="player-controls__bar">' +
                '<div class="player-controls__left">' +
                    '<button class="player-btn" data-role="play" aria-label="Play">' + ICON.play + '</button>' +
                    '<div class="player-controls__time">' +
                        '<span data-role="current">0:00</span>' +
                        '<span class="player-controls__time-sep">/</span>' +
                        '<span data-role="total">0:00</span>' +
                    '</div>' +
                    '<span class="player-controls__mode-badge" data-mode=""></span>' +
                    '<span class="player-controls__subtitle-loading" data-role="sub-loading">' +
                        '<span class="mini-spinner"></span><span>Generating...</span>' +
                    '</span>' +
                '</div>' +
                '<div class="player-controls__right">' +
                    '<div class="player-controls__volume" data-role="vol-wrap">' +
                        '<button class="player-btn" data-role="mute" aria-label="Mute">' + ICON.volHi + '</button>' +
                        '<div class="player-controls__volume-slider-wrap">' +
                            '<input type="range" class="player-controls__volume-slider" min="0" max="1" step="0.01" value="1" data-role="vol-slider" aria-label="Volume">' +
                        '</div>' +
                    '</div>' +
                    '<button class="player-btn" data-role="settings" aria-label="Settings" title="Settings">' + ICON.settings + '</button>' +
                    '<button class="player-btn" data-role="pip" aria-label="Picture in picture" title="Picture in picture">' + ICON.pip + '</button>' +
                    '<button class="player-btn" data-role="fullscreen" aria-label="Fullscreen" title="Fullscreen">' + ICON.fsEnter + '</button>' +
                '</div>' +
            '</div>';
    }

    function $(role) {
        return SP.elements.playerControls.querySelector('[data-role="' + role + '"]');
    }

    function fmtTime(t) {
        if (!isFinite(t) || t < 0) t = 0;
        var s = Math.floor(t);
        var h = Math.floor(s / 3600);
        var m = Math.floor((s % 3600) / 60);
        var sec = s % 60;
        var ss = sec < 10 ? "0" + sec : "" + sec;
        if (h > 0) {
            var mm = m < 10 ? "0" + m : "" + m;
            return h + ":" + mm + ":" + ss;
        }
        return m + ":" + ss;
    }

    // ─── Wiring ───
    function wirePlayPause(video) {
        var btn = $("play");
        function refresh() {
            btn.innerHTML = video.paused ? ICON.play : ICON.pause;
            btn.setAttribute("aria-label", video.paused ? "Play" : "Pause");
        }
        btn.addEventListener("click", function() {
            if (video.paused) video.play(); else video.pause();
        });
        video.addEventListener("play", refresh);
        video.addEventListener("pause", refresh);
        refresh();
    }

    function wireTime(video) {
        var current = $("current");
        var total = $("total");
        function refresh() {
            current.textContent = fmtTime(video.currentTime);
            total.textContent = fmtTime(video.duration);
        }
        video.addEventListener("timeupdate", refresh);
        video.addEventListener("durationchange", refresh);
        video.addEventListener("loadedmetadata", refresh);
        refresh();
    }

    function wireSeek(video) {
        var seek = $("seek");
        var played = $("played");
        var buffered = $("buffered");
        var thumb = $("thumb");

        function renderPosition() {
            if (seekDragging) return; // user is dragging — don't fight them
            var d = video.duration;
            if (!isFinite(d) || d <= 0) return;
            var pct = Math.max(0, Math.min(100, (video.currentTime / d) * 100));
            played.style.width = pct + "%";
            thumb.style.left = pct + "%";
        }

        function renderBuffered() {
            var d = video.duration;
            if (!isFinite(d) || d <= 0) return;
            var b = video.buffered;
            if (b.length === 0) { buffered.style.width = "0%"; return; }
            // Use the buffered range that contains currentTime, falling back to the last one.
            var end = b.end(b.length - 1);
            for (var i = 0; i < b.length; i++) {
                if (video.currentTime >= b.start(i) && video.currentTime <= b.end(i)) {
                    end = b.end(i);
                    break;
                }
            }
            buffered.style.width = Math.min(100, (end / d) * 100) + "%";
        }

        function pctFromEvent(e) {
            var rect = seek.getBoundingClientRect();
            var x = (e.clientX != null ? e.clientX : (e.touches && e.touches[0].clientX)) - rect.left;
            return Math.max(0, Math.min(1, x / rect.width));
        }

        function seekTo(e) {
            var d = video.duration;
            if (!isFinite(d) || d <= 0) return;
            var p = pctFromEvent(e);
            // Setting currentTime fires `seeking` → client-mode's listener
            // at client-player.js:341 handles the MediaSource rebuild.
            video.currentTime = p * d;
            // Snap the UI immediately for responsiveness while the seek lands.
            played.style.width = (p * 100) + "%";
            thumb.style.left = (p * 100) + "%";
        }

        seek.addEventListener("mousedown", function(e) {
            seekDragging = true;
            seek.classList.add("is-dragging");
            seekTo(e);
            function onMove(ev) { seekTo(ev); }
            function onUp(ev) {
                seekTo(ev);
                seekDragging = false;
                seek.classList.remove("is-dragging");
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
            }
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        });

        // Touch support — uses Pointer Events fallback via touchstart.
        seek.addEventListener("touchstart", function(e) {
            seekDragging = true;
            seek.classList.add("is-dragging");
            seekTo(e);
            function onMove(ev) { ev.preventDefault(); seekTo(ev); }
            function onEnd() {
                seekDragging = false;
                seek.classList.remove("is-dragging");
                document.removeEventListener("touchmove", onMove);
                document.removeEventListener("touchend", onEnd);
            }
            document.addEventListener("touchmove", onMove, { passive: false });
            document.addEventListener("touchend", onEnd);
        }, { passive: true });

        video.addEventListener("timeupdate", renderPosition);
        video.addEventListener("seeking", renderPosition);
        video.addEventListener("seeked", renderPosition);
        video.addEventListener("progress", renderBuffered);
        video.addEventListener("durationchange", function() { renderPosition(); renderBuffered(); });
        video.addEventListener("loadedmetadata", function() { renderPosition(); renderBuffered(); });
    }

    function wireVolume(video) {
        var muteBtn = $("mute");
        var slider = $("vol-slider");

        function refresh() {
            var muted = video.muted || video.volume === 0;
            muteBtn.innerHTML = muted ? ICON.volMute : ICON.volHi;
            muteBtn.setAttribute("aria-label", muted ? "Unmute" : "Mute");
            slider.value = muted ? 0 : video.volume;
        }

        slider.addEventListener("input", function() {
            var v = parseFloat(slider.value);
            video.volume = v;
            video.muted = (v === 0);
            if (v > 0) preMuteVolume = v;
        });

        muteBtn.addEventListener("click", function() {
            if (video.muted || video.volume === 0) {
                video.muted = false;
                video.volume = preMuteVolume > 0 ? preMuteVolume : 1;
            } else {
                preMuteVolume = video.volume;
                video.muted = true;
            }
        });

        video.addEventListener("volumechange", refresh);
        refresh();
    }

    function wireFullscreen(video) {
        var btn = $("fullscreen");
        var container = SP.elements.videoContainer;

        function refresh() {
            var isFs = document.fullscreenElement === container ||
                       document.webkitFullscreenElement === container;
            btn.innerHTML = isFs ? ICON.fsExit : ICON.fsEnter;
            btn.setAttribute("aria-label", isFs ? "Exit fullscreen" : "Fullscreen");
            container.classList.toggle("is-fullscreen", isFs);
        }

        btn.addEventListener("click", function() {
            var isFs = document.fullscreenElement === container ||
                       document.webkitFullscreenElement === container;
            if (isFs) {
                if (document.exitFullscreen) document.exitFullscreen();
                else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            } else {
                if (container.requestFullscreen) container.requestFullscreen();
                else if (container.webkitRequestFullscreen) container.webkitRequestFullscreen();
            }
        });

        document.addEventListener("fullscreenchange", refresh);
        document.addEventListener("webkitfullscreenchange", refresh);
        refresh();
    }

    function wirePiP(video) {
        var btn = $("pip");
        // Hide on browsers without PiP (Firefox without flag, iOS Safari uses
        // its own UI, etc.) — keep one less broken button than a noisy error.
        if (!document.pictureInPictureEnabled || video.disablePictureInPicture) {
            btn.hidden = true;
            return;
        }
        btn.addEventListener("click", function() {
            if (document.pictureInPictureElement === video) {
                document.exitPictureInPicture().catch(function() {});
            } else {
                video.requestPictureInPicture().catch(function() {});
            }
        });
    }

    function wireClickShield(video) {
        var shield = SP.elements.playerClickShield;
        if (!shield) return;
        var lastClick = 0;
        shield.addEventListener("click", function(e) {
            // Distinguish single from double click: defer the toggle slightly,
            // cancel if a second click arrives.
            var now = Date.now();
            if (now - lastClick < 280) {
                lastClick = 0;
                // Double-click → fullscreen
                var btn = $("fullscreen");
                if (btn) btn.click();
                return;
            }
            lastClick = now;
            setTimeout(function() {
                if (lastClick !== now) return; // a second click came in
                if (video.paused) video.play(); else video.pause();
            }, 280);
        });
    }

    function wireLoadingSpinner(video) {
        var on  = function() { window.setPlayerLoading(true); };
        var off = function() { window.setPlayerLoading(false); };
        video.addEventListener("waiting", on);
        video.addEventListener("stalled", on);
        video.addEventListener("playing", off);
        video.addEventListener("canplay", off);
        video.addEventListener("pause", off);
    }

    // ─── Auto-hide ───
    // Controls fade after 2s of inactivity while playing. Stay visible
    // whenever the video is paused, the mouse is over the bar, or the
    // settings menu is open.
    function wireAutoHide(video, container) {
        var hideTimer = null;
        var bar = SP.elements.playerControls;
        var menu = SP.elements.playerSettingsMenu;

        function show() {
            container.classList.remove("is-idle");
            scheduleHide();
        }
        function hide() {
            if (video.paused) return;
            if (bar && bar.matches(":hover")) return;
            if (menu && menu.classList.contains("is-open")) return;
            container.classList.add("is-idle");
        }
        function scheduleHide() {
            if (hideTimer) clearTimeout(hideTimer);
            hideTimer = setTimeout(hide, 2000);
        }

        container.addEventListener("mousemove", show);
        container.addEventListener("mouseleave", hide);
        if (bar) bar.addEventListener("mouseenter", function() {
            container.classList.remove("is-idle");
            if (hideTimer) clearTimeout(hideTimer);
        });
        if (bar) bar.addEventListener("mouseleave", scheduleHide);
        video.addEventListener("pause", function() {
            container.classList.remove("is-idle");
            if (hideTimer) clearTimeout(hideTimer);
        });
        video.addEventListener("play", scheduleHide);
        show();
    }

    // ─── Keyboard shortcuts ───
    // Document-level listener; ignored when the user is typing in an input
    // or when no file is loaded. Most browsers reserve F/Esc in fullscreen
    // for their own UI, so we preventDefault on our shortcuts.
    function wireKeyboardShortcuts(video, container) {
        function isTyping() {
            var el = document.activeElement;
            if (!el) return false;
            var tag = el.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
            if (el.isContentEditable) return true;
            return false;
        }

        function clampSeek(dt) {
            if (!isFinite(video.duration) || video.duration <= 0) return;
            var t = Math.max(0, Math.min(video.duration, video.currentTime + dt));
            video.currentTime = t;
        }

        function clampVolume(dv) {
            var v = Math.max(0, Math.min(1, video.volume + dv));
            video.volume = v;
            if (v > 0 && video.muted) video.muted = false;
        }

        document.addEventListener("keydown", function(e) {
            if (isTyping()) return;
            // Only handle when a file is loaded (avoid hijacking browse view).
            if (!SP.state.currentFile) return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;

            var key = e.key;
            var handled = true;
            switch (key) {
                case " ":
                case "k":
                case "K":
                    if (video.paused) video.play(); else video.pause();
                    break;
                case "f":
                case "F":
                    $("fullscreen").click();
                    break;
                case "m":
                case "M":
                    $("mute").click();
                    break;
                case "ArrowLeft":  clampSeek(-5); break;
                case "ArrowRight": clampSeek(+5); break;
                case "j": case "J": clampSeek(-10); break;
                case "l": case "L": clampSeek(+10); break;
                case "ArrowUp":   clampVolume(+0.05); break;
                case "ArrowDown": clampVolume(-0.05); break;
                case "c": case "C":
                    if (typeof toggleCaptions === "function") toggleCaptions();
                    break;
                default:
                    if (key >= "0" && key <= "9" && isFinite(video.duration) && video.duration > 0) {
                        video.currentTime = (parseInt(key, 10) / 10) * video.duration;
                    } else {
                        handled = false;
                    }
            }
            if (handled) e.preventDefault();
        });
    }

    function wireSubtitleLoadingMirror() {
        // The existing initSubtitleControl flips classes on the legacy
        // #subtitleLoading span. Mirror that into the bar so the indicator
        // still appears now that the legacy bar is hidden.
        var legacy = SP.elements.subtitleLoading;
        var mirror = $("sub-loading");
        if (!legacy || !mirror) return;
        var mo = new MutationObserver(function() {
            mirror.classList.toggle("is-active", legacy.classList.contains("active"));
        });
        mo.observe(legacy, { attributes: true, attributeFilter: ["class"] });
    }

    // ─── Entry point ───
    window.initPlayerUI = function() {
        var video = SP.elements.video;
        var container = SP.elements.videoContainer;
        if (!video || !SP.elements.playerControls) return;
        buildBar();
        wirePlayPause(video);
        wireTime(video);
        wireSeek(video);
        wireVolume(video);
        wireFullscreen(video);
        wirePiP(video);
        wireClickShield(video);
        wireLoadingSpinner(video);
        wireSubtitleLoadingMirror();
        wireAutoHide(video, container);
        wireKeyboardShortcuts(video, container);
    };
})();
