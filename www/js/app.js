/* App initialization - entry point */

(function() {
    // Initialize DOM element references
    function initElements() {
        SP.elements = {
            // Video/Player
            video: document.getElementById("video"),
            audioSelect: document.getElementById("audioSelect"),
            subtitleSelect: document.getElementById("subtitleSelect"),
            resolutionSelect: document.getElementById("resolutionSelect"),
            statusEl: document.getElementById("status"),
            statusDot: document.getElementById("statusDot"),
            statusText: document.getElementById("statusText"),

            // Navigation
            browseView: document.getElementById("browseView"),
            playerView: document.getElementById("playerView"),
            browseBreadcrumb: document.getElementById("browseBreadcrumb"),
            sidebarBreadcrumb: document.getElementById("sidebarBreadcrumb"),
            mediaGrid: document.getElementById("mediaGrid"),
            fileList: document.getElementById("fileList"),
            homeBtn: document.getElementById("homeBtn"),
            browseStats: document.getElementById("browseStats"),

            // Info
            nowPlaying: document.getElementById("nowPlaying"),
            nowPlayingName: document.getElementById("nowPlayingName"),

            // Subtitle progress
            subtitleLoading: document.getElementById("subtitleLoading"),
            subtitleProgress: document.getElementById("subtitleProgress"),
            subtitleProgressFill: document.getElementById("subtitleProgressFill"),
            subtitleProgressText: document.getElementById("subtitleProgressText"),

            // Controls
            downloadBtn: document.getElementById("downloadBtn")
        };
    }

    // Initialize home button
    function initHomeButton() {
        SP.elements.homeBtn.addEventListener("click", function() {
            if (SP.state.isPlayerMode) {
                switchToBrowseMode();
            } else {
                SP.state.currentPath = "";
                loadBrowseView("");
            }
        });
    }

    // Main initialization
    function init() {
        initElements();
        initHomeButton();
        initControls();
        initMetricsToggle();
        initRouter();
    }

    // Start when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
