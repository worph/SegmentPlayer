/* Utility functions */

function formatSize(bytes) {
    if (bytes === 0) return "0 B";
    var k = 1024;
    var sizes = ["B", "KB", "MB", "GB"];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function isVideo(name) {
    var lower = name.toLowerCase();
    return SP.config.videoExtensions.some(function(ext) {
        return lower.endsWith(ext);
    });
}

function isSubtitle(name) {
    var lower = name.toLowerCase();
    return SP.config.subtitleExtensions.some(function(ext) {
        return lower.endsWith(ext);
    });
}

function getIcon(item) {
    if (item.type === "directory") return "📁";
    if (isVideo(item.name)) return "🎬";
    if (isSubtitle(item.name)) return "📝";
    return "📄";
}

function encodeFilePath(path) {
    return path.split('/').map(function(part) {
        return encodeURIComponent(part);
    }).join('/');
}

function extractLang(subName, baseName) {
    var withoutBase = subName.substring(baseName.length);
    var match = withoutBase.match(/[._]([a-z]{2,3})/i);
    return match ? match[1].toUpperCase() : "Unknown";
}

function removeAllTracks(videoEl) {
    videoEl.querySelectorAll("track").forEach(function(t) { t.remove(); });
}

async function getProbeData(filePath) {
    try {
        var response = await fetch("/api/probe/" + encodeFilePath(filePath));
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        return null;
    }
}

/* Quiet-by-default logger.
 * debug/info lines live in a 200-entry ring buffer and stay out of the
 * console unless verbose mode is on (?debug=1 in URL, or localStorage
 * sp_debug=1). warn/error print immediately and flush the buffer as a
 * collapsed group so the lead-up to a failure is visible retroactively.
 *
 * Separately, *every* line (all levels) is mirrored into a larger persistent
 * `history` ring that is NOT cleared by flush(). That's what SP.log.dump()
 * returns and what the "Copy log" button in the metrics panel exports — so a
 * user can grab the lead-up to a stall long after it scrolled out of console.
 */
(function() {
    var RING_SIZE = 200;
    var HISTORY_SIZE = 1000;
    var ring = [];
    var history = [];
    var verbose = false;
    try {
        verbose = new URLSearchParams(location.search).has("debug")
            || localStorage.getItem("sp_debug") === "1";
    } catch (e) {}

    function now() { return Math.round(performance.now()); }

    // Stringify a single log argument for the copyable history. Errors keep
    // their stack; objects are JSON'd (with a circular-safe fallback) so the
    // dump is self-contained text rather than "[object Object]".
    function fmtArg(a) {
        if (a instanceof Error) {
            return (a.name || "Error") + ": " + (a.message || "")
                + (a.stack ? "\n" + a.stack : "");
        }
        if (a !== null && typeof a === "object") {
            try { return JSON.stringify(a); } catch (e) { return String(a); }
        }
        return String(a);
    }

    function pushHistory(level, tag, args) {
        if (history.length >= HISTORY_SIZE) history.shift();
        var msg;
        try { msg = Array.prototype.map.call(args, fmtArg).join(" "); }
        catch (e) { msg = "<unformattable>"; }
        history.push({ t: now(), level: level, tag: tag, msg: msg });
    }

    function push(level, tag, args) {
        if (ring.length >= RING_SIZE) ring.shift();
        ring.push({ t: now(), level: level, tag: tag, args: args });
        pushHistory(level, tag, args);
    }

    function flush() {
        if (ring.length === 0) return;
        var buffered = ring.slice();
        ring.length = 0;
        try {
            console.groupCollapsed("[SP] context before error (" + buffered.length + " lines)");
            for (var i = 0; i < buffered.length; i++) {
                var e = buffered[i];
                var fn = e.level === "error" ? console.error
                    : e.level === "warn" ? console.warn
                    : console.log;
                fn.apply(console, ["[+" + e.t + "ms] [" + e.tag + "]"].concat(e.args));
            }
            console.groupEnd();
        } catch (err) {}
    }

    // Render the persistent history as plain text (oldest → newest).
    function dump() {
        var lines = [];
        for (var i = 0; i < history.length; i++) {
            var e = history[i];
            lines.push("[+" + e.t + "ms] [" + e.level.toUpperCase() + "] ["
                + e.tag + "] " + e.msg);
        }
        return lines.join("\n");
    }

    SP.log = {
        _verbose: verbose,
        debug: function(tag) {
            var args = Array.prototype.slice.call(arguments, 1);
            push("debug", tag, args);
            if (verbose) console.log.apply(console, ["[" + tag + "]"].concat(args));
        },
        info: function(tag) {
            var args = Array.prototype.slice.call(arguments, 1);
            push("info", tag, args);
            if (verbose) console.log.apply(console, ["[" + tag + "]"].concat(args));
        },
        warn: function(tag) {
            var args = Array.prototype.slice.call(arguments, 1);
            pushHistory("warn", tag, args);
            console.warn.apply(console, ["[" + tag + "]"].concat(args));
            flush();
        },
        error: function(tag) {
            var args = Array.prototype.slice.call(arguments, 1);
            pushHistory("error", tag, args);
            console.error.apply(console, ["[" + tag + "]"].concat(args));
            flush();
        },
        flush: flush,
        dump: dump,
        historyLength: function() { return history.length; }
    };

    window.addEventListener("error", flush);
    window.addEventListener("unhandledrejection", flush);
})();

