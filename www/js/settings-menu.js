/* Settings cog menu — Quality, Audio, Subtitles, Mode, Speed, Download.
 *
 * Each page is a list of items rebuilt from SP.state on open, so the menu
 * always reflects current probe data / hls levels / playback rate. Item
 * selection delegates to the pure setters in controls.js (setQuality,
 * setAudioTrack, setSubtitle, setMode, downloadCurrent) — this module
 * never touches playback state directly.
 */

(function() {
    var menuOpen = false;
    var currentPageKey = "root";       // the visible submenu, for refresh()
    var lastSubtitleSelection = null;  // for the captions quick-toggle
    // Active subtitle from the menu's perspective. Synced with the legacy
    // <select> on every menu-driven change so the active-marker reflects
    // the actual choice on re-open.
    var currentSubtitleValue = "";

    function $btn(role) {
        return SP.elements.playerControls
            ? SP.elements.playerControls.querySelector('[data-role="' + role + '"]')
            : null;
    }

    function menuEl() { return SP.elements.playerSettingsMenu; }

    // ─── Page registry — definitions are built lazily on open so they
    //     always reflect the current SP.state. ──────────────────────────
    function buildRootPage() {
        var resolution = currentResolutionLabel();
        var audio = currentAudioLabel();
        var subtitle = currentSubtitleLabel();
        var mode = currentModeLabel();
        var speed = currentSpeedLabel();

        var items = [];
        // Always show Quality and Audio. For tiers without quality variants
        // (client, direct) the submenu shows "Original" only. For files with
        // a single audio track the submenu shows that one track. Keeping the
        // entries always visible avoids the discoverability problem of
        // options that appear/disappear based on hidden state.
        items.push({ label: "Quality", value: resolution, page: "quality" });
        items.push({ label: "Audio", value: audio, page: "audio" });
        items.push({ label: "Subtitles", value: subtitle, page: "subtitles" });
        items.push({ label: "Mode", value: mode, page: "mode" });
        items.push({ label: "Speed", value: speed, page: "speed" });
        items.push({ divider: true });
        items.push({ label: "Download", onSelect: function() { downloadCurrent(); }, terminal: true });
        return { title: null, items: items };
    }

    function buildQualityPage() {
        var items = [];
        // Tiers that don't use hls (client, direct) only have the source
        // stream — show that as a single "Original" entry so the page is
        // never empty. The transcode/remux path with no levels yet shows
        // a Loading placeholder.
        if (!SP.state.hls) {
            var srcHeight = SP.state.probeData && SP.state.probeData.video && SP.state.probeData.video.height;
            var label = srcHeight ? ("Original (" + srcHeight + "p)") : "Original";
            return { title: "Quality", items: [{
                label: label, active: true, onSelect: function() {}
            }], back: "root" };
        }
        if (!SP.state.hls.levels || SP.state.hls.levels.length === 0) {
            return { title: "Quality", items: [{
                label: "Loading qualities…", disabled: true
            }], back: "root" };
        }
        var levels = SP.state.hls.levels;
        var current = SP.state.currentResolution;
        items.push({
            label: "Auto",
            active: current === "auto",
            onSelect: function() { setQuality("auto"); }
        });
        // Deduplicate heights, sort high → low.
        var seen = {};
        var heights = [];
        for (var i = 0; i < levels.length; i++) {
            var h = levels[i].height;
            if (h && !seen[h]) { seen[h] = true; heights.push(h); }
        }
        heights.sort(function(a, b) { return b - a; });
        for (var j = 0; j < heights.length; j++) {
            (function(h) {
                items.push({
                    label: h + "p",
                    active: String(current) === String(h),
                    onSelect: function() { setQuality(String(h)); }
                });
            })(heights[j]);
        }
        return { title: "Quality", items: items, back: "root" };
    }

    function buildAudioPage() {
        var tracks = (SP.state.probeData && SP.state.probeData.audio) || [];
        var current = SP.state.currentAudioIdx;
        var items = tracks.map(function(t, i) {
            var label = trackLabel(t, i);
            return {
                label: label,
                active: i === current,
                onSelect: function() { setAudioTrack(i); }
            };
        });
        return { title: "Audio", items: items, back: "root" };
    }

    function buildSubtitlesPage() {
        var items = [];
        var current = currentSubtitleSelection();
        items.push({
            label: "Off",
            active: current === "",
            onSelect: function() {
                recordSubtitleSelection("", "");
                setSubtitle("", "");
            }
        });
        // HLS native subtitles (remux/transcode tier).
        if (SP.state.hls && SP.state.hls.subtitleTracks && SP.state.hls.subtitleTracks.length > 0) {
            SP.state.hls.subtitleTracks.forEach(function(st, i) {
                var label = st.name || st.lang || ("Track " + (i + 1));
                items.push({
                    label: label,
                    active: String(current) === String(i),
                    onSelect: function() {
                        recordSubtitleSelection(String(i), label);
                        setSubtitle(String(i), label);
                    }
                });
            });
        } else {
            // Embedded (server-extract or client-collected).
            var embedded = (SP.state.probeData && SP.state.probeData.subtitles) || [];
            embedded.forEach(function(s, i) {
                var v = "embedded:" + i;
                var label = trackLabel(s, i);
                items.push({
                    label: label,
                    active: current === v,
                    onSelect: function() {
                        recordSubtitleSelection(v, label);
                        setSubtitle(v, label);
                    }
                });
            });
            // Transcoded subs (when in transcode mode).
            var transcoded = SP.state.transcodedSubtitleTracks || {};
            Object.keys(transcoded).forEach(function(key) {
                var t = transcoded[key];
                var label = (t && t.name) || ("Subtitle " + (parseInt(key) + 1));
                items.push({
                    label: label,
                    active: current === key,
                    onSelect: function() {
                        recordSubtitleSelection(key, label);
                        setSubtitle(key, label);
                    }
                });
            });
        }
        return { title: "Subtitles", items: items, back: "root" };
    }

    function buildModePage() {
        var current = SP.state.playbackMode;
        var modes = [
            { v: "auto",      l: SP.state.activePlaybackMode && current === "auto"
                                  ? "Auto (" + SP.state.activePlaybackMode + ")" : "Auto" },
            { v: "direct",    l: "Direct" },
            { v: "client",    l: "Client" },
            { v: "transcode", l: "Transcode" }
        ];
        var items = modes.map(function(m) {
            return {
                label: m.l,
                active: m.v === current,
                onSelect: function() { setMode(m.v); }
            };
        });
        return { title: "Mode", items: items, back: "root" };
    }

    function buildSpeedPage() {
        var current = SP.elements.video.playbackRate;
        var speeds = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
        var items = speeds.map(function(s) {
            return {
                label: s === 1 ? "Normal" : (s + "×"),
                active: Math.abs(current - s) < 0.001,
                onSelect: function() {
                    SP.elements.video.playbackRate = s;
                }
            };
        });
        return { title: "Speed", items: items, back: "root" };
    }

    var PAGE_BUILDERS = {
        root:      buildRootPage,
        quality:   buildQualityPage,
        audio:     buildAudioPage,
        subtitles: buildSubtitlesPage,
        mode:      buildModePage,
        speed:     buildSpeedPage
    };

    // ─── Label helpers ───
    function trackLabel(t, idx) {
        if (!t) return "Track " + (idx + 1);
        var parts = [];
        if (t.title) parts.push(t.title);
        if (t.language) parts.push("[" + t.language + "]");
        if (t.codec) parts.push(t.codec.toUpperCase());
        return parts.length ? parts.join(" ") : ("Track " + (idx + 1));
    }

    function currentResolutionLabel() {
        // Non-hls tiers (client, direct) — source resolution from probe.
        if (!SP.state.hls) {
            var h = SP.state.probeData && SP.state.probeData.video && SP.state.probeData.video.height;
            return h ? (h + "p") : "Original";
        }
        if (!SP.state.hls.levels || SP.state.hls.levels.length === 0) return "—";
        var r = SP.state.currentResolution || "auto";
        if (r === "auto") return SP.state.actualResolution ? "Auto (" + SP.state.actualResolution + "p)" : "Auto";
        return r + (/^\d+$/.test(r) ? "p" : "");
    }

    function currentAudioLabel() {
        var tracks = (SP.state.probeData && SP.state.probeData.audio) || [];
        var t = tracks[SP.state.currentAudioIdx || 0];
        return trackLabel(t, SP.state.currentAudioIdx || 0);
    }

    function currentSubtitleLabel() {
        var sel = currentSubtitleSelection();
        if (sel === "") return "Off";
        if (typeof sel === "string" && sel.indexOf("embedded:") === 0) {
            var subs = (SP.state.probeData && SP.state.probeData.subtitles) || [];
            var idx = parseInt(sel.split(":")[1]);
            return trackLabel(subs[idx], idx);
        }
        if (SP.state.hls && SP.state.hls.subtitleTracks && SP.state.hls.subtitleTracks.length > 0) {
            var idx2 = parseInt(sel);
            var st = SP.state.hls.subtitleTracks[idx2];
            return st ? (st.name || st.lang || ("Track " + (idx2 + 1))) : "—";
        }
        return "On";
    }

    function currentSubtitleSelection() {
        // Menu-driven changes update currentSubtitleValue. Legacy <select>
        // changes (which still fire while the hidden select exists) also
        // count as truth — fall back to it if the menu hasn't observed
        // a selection yet.
        if (currentSubtitleValue) return currentSubtitleValue;
        var sel = SP.elements.subtitleSelect;
        return sel ? sel.value : "";
    }

    function recordSubtitleSelection(value, label) {
        currentSubtitleValue = value;
        var sel = SP.elements.subtitleSelect;
        if (sel) sel.value = value;
        if (value !== "") lastSubtitleSelection = { value: value, label: label };
    }

    function currentModeLabel() {
        var m = SP.state.playbackMode;
        if (m === "auto" && SP.state.activePlaybackMode) {
            return "Auto (" + SP.state.activePlaybackMode + ")";
        }
        return m ? m.charAt(0).toUpperCase() + m.slice(1) : "Auto";
    }

    function currentSpeedLabel() {
        var r = SP.elements.video.playbackRate;
        if (Math.abs(r - 1) < 0.001) return "Normal";
        return r + "×";
    }

    // ─── Rendering ───
    function renderPage(pageKey) {
        var menu = menuEl();
        var builder = PAGE_BUILDERS[pageKey];
        if (!menu || !builder) return;
        currentPageKey = pageKey;
        var page = builder();
        var html = '<div class="player-settings-menu__page">';
        if (page.title) {
            html += '<div class="player-settings-menu__page-header" data-back="' + (page.back || "root") + '">' +
                        '<span class="player-settings-menu__back">‹</span>' +
                        '<span>' + escapeHtml(page.title) + '</span>' +
                    '</div>';
        }
        html += '<div class="player-settings-menu__items">';
        if (page.items.length === 0) {
            html += '<div class="player-settings-menu__item" style="opacity:0.5;cursor:default">' +
                        '<span class="player-settings-menu__item-label">No options</span>' +
                    '</div>';
        }
        page.items.forEach(function(item, i) {
            if (item.divider) {
                html += '<div class="player-settings-menu__divider"></div>';
                return;
            }
            var cls = "player-settings-menu__item" + (item.active ? " is-active" : "");
            html += '<div class="' + cls + '" tabindex="0" data-item-idx="' + i + '">';
            if (item.page) {
                html += '<span class="player-settings-menu__item-label">' + escapeHtml(item.label) + '</span>';
                if (item.value) html += '<span class="player-settings-menu__item-value">' + escapeHtml(item.value) + '</span>';
                html += '<span class="player-settings-menu__item-chevron">›</span>';
            } else {
                html += '<span class="player-settings-menu__item-check">' + (item.active ? "✓" : "") + '</span>';
                html += '<span class="player-settings-menu__item-label">' + escapeHtml(item.label) + '</span>';
            }
            html += '</div>';
        });
        html += '</div></div>';
        menu.innerHTML = html;
        menu.hidden = false;

        // Wire item clicks. stopPropagation prevents the bubbled click from
        // hitting the document outside-click handler — after a navigate the
        // original element is detached, so menu.contains(e.target) wouldn't
        // recognize it and the menu would close immediately.
        var els = menu.querySelectorAll('.player-settings-menu__item[data-item-idx]');
        els.forEach(function(el) {
            var idx = parseInt(el.dataset.itemIdx);
            var item = page.items[idx];
            if (!item) return;
            el.addEventListener("click", function(e) {
                e.stopPropagation();
                if (item.page) {
                    renderPage(item.page);
                } else if (typeof item.onSelect === "function") {
                    item.onSelect();
                    closeMenu();
                }
            });
            el.addEventListener("keydown", function(e) {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    el.click();
                }
            });
        });
        // Wire back arrow
        var header = menu.querySelector('.player-settings-menu__page-header');
        if (header) {
            header.addEventListener("click", function(e) {
                e.stopPropagation();
                renderPage(header.dataset.back || "root");
            });
        }
        // Auto-focus first item so Enter/keyboard works
        var first = menu.querySelector('.player-settings-menu__item[tabindex]');
        if (first) first.focus();
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // ─── Open/close ───
    function openMenu() {
        if (menuOpen) return;
        menuOpen = true;
        renderPage("root");
        var menu = menuEl();
        // Force a reflow so the browser registers the hidden=false / initial
        // opacity:0 state BEFORE we add is-open. Without this the transition
        // can be skipped (the engine collapses the two style mutations into
        // one paint). Reading offsetHeight is the canonical synchronous-reflow
        // trick. requestAnimationFrame was less reliable — Chrome throttles
        // rAF in backgrounded tabs / headless contexts.
        if (menu) {
            void menu.offsetHeight;
            menu.classList.add("is-open");
        }
        var btn = $btn("settings");
        if (btn) btn.classList.add("is-active");
    }

    function closeMenu() {
        if (!menuOpen) return;
        menuOpen = false;
        var menu = menuEl();
        if (menu) menu.classList.remove("is-open");
        var btn = $btn("settings");
        if (btn) btn.classList.remove("is-active");
        // Hide after transition completes so click-outside on the hidden
        // element doesn't immediately re-open.
        setTimeout(function() {
            if (!menuOpen && menu) menu.hidden = true;
        }, 200);
    }

    function toggleMenu() { menuOpen ? closeMenu() : openMenu(); }

    // Captions quick-toggle — called by the C keyboard shortcut. The
    // dedicated CC button was removed (redundant with the Subtitles menu),
    // but the toggle pattern is still useful from the keyboard.
    window.toggleCaptions = function() {
        var current = currentSubtitleSelection();
        if (current !== "") {
            // Currently on → turn off, remember the choice for re-enable.
            var saved = lastSubtitleSelection;
            recordSubtitleSelection("", "");
            // recordSubtitleSelection clobbers lastSubtitleSelection when
            // value === "", but for the quick-toggle we want to keep the
            // previous choice available — restore it.
            lastSubtitleSelection = saved || { value: current, label: "" };
            setSubtitle("", "");
        } else if (lastSubtitleSelection) {
            // Re-enable previous selection.
            recordSubtitleSelection(lastSubtitleSelection.value, lastSubtitleSelection.label);
            setSubtitle(lastSubtitleSelection.value, lastSubtitleSelection.label);
        } else {
            // No previous selection — open the Subtitles submenu so the
            // user can pick one.
            openMenu();
            setTimeout(function() { renderPage("subtitles"); }, 50);
        }
    };

    // ─── Entry point ───
    window.initSettingsMenu = function() {
        var btn = $btn("settings");
        if (!btn) return;
        btn.addEventListener("click", toggleMenu);

        // Close on Esc.
        document.addEventListener("keydown", function(e) {
            if (e.key === "Escape" && menuOpen) {
                closeMenu();
                e.stopPropagation();
            }
        }, true);

        // Click outside closes the menu.
        document.addEventListener("click", function(e) {
            if (!menuOpen) return;
            var menu = menuEl();
            if (!menu) return;
            if (menu.contains(e.target)) return;
            if (e.target === btn || btn.contains(e.target)) return;
            closeMenu();
        });
    };

    // Re-render the currently-visible page. Called by player.js after
    // hls.js fires MANIFEST_PARSED so the Quality submenu picks up freshly
    // loaded levels without forcing the user to close and re-open.
    window.refreshSettingsMenu = function() {
        if (!menuOpen) return;
        renderPage(currentPageKey);
    };

    // Reset menu state on file switch (called from resetPlaybackUIState).
    window.resetSettingsMenuState = function() {
        lastSubtitleSelection = null;
        currentSubtitleValue = "";
        if (menuOpen) closeMenu();
        var cc = $btn("captions");
        if (cc) cc.classList.remove("is-active");
    };

    // Populators kept as no-ops — the menu reads SP.state on each open,
    // so explicit population isn't needed. Kept for API stability in case
    // player.js calls these.
    window.populateQualityMenu = function() {};
    window.populateAudioMenu   = function() {};
    window.populateSubtitleMenu = function() {};
    window.populateModeMenu    = function() {};
})();
