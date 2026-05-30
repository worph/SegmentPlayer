/* Configuration and global state */

var SP = {
    // State
    state: {
        hls: null,
        currentPath: "",
        currentFile: null,
        currentResolution: "auto",
        actualResolution: null,
        subtitleProgressInterval: null,
        isPlayerMode: false,
        clientPlayer: null,
        probeCache: {},
        currentAudioIdx: 0,
        currentTranscodeBase: "",
        transcodedAudioTracks: [],
        transcodedSubtitleTracks: [],
        metricsLocalInterval: null,
        metricsServerInterval: null,
        playbackMode: localStorage.getItem('sp_playback_mode') || "auto",
        activePlaybackMode: null,
        probeData: null,
        // Monotonically incremented at the start of every playFileSetup. Tier
        // handlers and async callbacks capture this and bail out if it has
        // moved on, so a slow probe/manifest from a previously-clicked file
        // can't pollute the currently-loading one.
        loadToken: 0
    },

    // Configuration
    config: {
        MAX_BUFFER_LENGTH: parseInt(new URLSearchParams(window.location.search).get('buffer') || '300'),
        CLIENT_BUFFER_AHEAD: 30,
        CLIENT_BUFFER_BEHIND: 30,
        // Client-mode self-healing (see ClientPlayer watchdog/_recover).
        CLIENT_STALL_RECOVER_MS: 12000,     // pump alive but wedged this long → force re-seek
        CLIENT_PUMP_DEAD_REFILL_AHEAD: 2,   // pump exited → recover once buffer-ahead drops below this (s)
        CLIENT_EOF_GUARD_SEC: 10,           // EOF farther than this from duration is treated as spurious
        CLIENT_RECOVER_WINDOW_MS: 30000,    // sliding window for recovery-rate escalation
        CLIENT_RECOVER_MAX: 4,              // recoveries allowed in window before falling back to transcode
        CLIENT_MAX_416_RETRIES: 6,          // consecutive in-range 416s tolerated before accepting real EOF
        videoExtensions: [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".m2ts"],
        subtitleExtensions: [".vtt", ".srt", ".ass", ".ssa"],
        PRESETS: ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    },

    // DOM Elements (populated in app.js)
    elements: {}
};
