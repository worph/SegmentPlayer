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
        currentAudioIdx: 0,
        currentTranscodeBase: "",
        transcodedAudioTracks: [],
        transcodedSubtitleTracks: [],
        metricsInterval: null
    },

    // Configuration
    config: {
        MAX_BUFFER_LENGTH: parseInt(new URLSearchParams(window.location.search).get('buffer') || '300'),
        videoExtensions: [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".m2ts"],
        subtitleExtensions: [".vtt", ".srt", ".ass", ".ssa"],
        PRESETS: ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    },

    // DOM Elements (populated in app.js)
    elements: {}
};
