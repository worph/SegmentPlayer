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
        isTranscoding: false,
        isClientSide: false,
        clientPlayer: null,
        currentProbe: null,
        probeCache: {},
        currentAudioIdx: 0,
        currentTranscodeBase: "",
        transcodedAudioTracks: [],
        transcodedSubtitleTracks: [],
        metricsInterval: null,
        playbackMode: localStorage.getItem('sp_playback_mode') || "auto",
        activePlaybackMode: null,
        probeData: null
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
