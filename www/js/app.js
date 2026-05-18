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
            downloadBtn: document.getElementById("downloadBtn"),
            modeSelect: document.getElementById("modeSelect"),

            // Custom player UI (built by player-ui.js / settings-menu.js)
            videoContainer: document.getElementById("videoContainer"),
            playerControls: document.getElementById("playerControls"),
            playerLoading: document.getElementById("playerLoading"),
            playerClickShield: document.getElementById("playerClickShield"),
            playerSettingsMenu: document.getElementById("playerSettingsMenu"),
            nowPlayingOverlay: document.getElementById("nowPlayingOverlay")
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
        initPlayerUI();
        initSettingsMenu();
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
