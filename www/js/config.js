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
        videoExtensions: [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".m2ts"],
        subtitleExtensions: [".vtt", ".srt", ".ass", ".ssa"],
        PRESETS: ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    },

    // DOM Elements (populated in app.js)
    elements: {}
};
