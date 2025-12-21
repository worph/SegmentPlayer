/* Router - mode switching and URL hash routing */

function switchToPlayerMode() {
    if (SP.state.isPlayerMode) return;
    SP.state.isPlayerMode = true;
    SP.elements.browseView.classList.add("hidden");
    SP.elements.playerView.classList.add("active");
}

function switchToBrowseMode() {
    if (!SP.state.isPlayerMode) return;
    SP.state.isPlayerMode = false;
    SP.elements.playerView.classList.remove("active");
    SP.elements.browseView.classList.remove("hidden");

    // Stop video and clear
    if (SP.state.hls) {
        SP.state.hls.destroy();
        SP.state.hls = null;
    }
    SP.elements.video.src = "";
    SP.state.currentFile = null;

    // Clear URL hash
    history.pushState(null, "", window.location.pathname);

    // Reload browse view with current path
    loadBrowseView(SP.state.currentPath);
}

function updateUrlHash(filePath) {
    var encodedPath = encodeFilePath(filePath);
    history.pushState(null, "", "#/play/" + encodedPath);
}

function handleHashRoute() {
    var hash = window.location.hash;
    if (hash.startsWith("#/play/")) {
        var encodedPath = hash.substring(7);
        var filePath = decodeURIComponent(encodedPath);

        if (filePath) {
            var lastSlash = filePath.lastIndexOf("/");
            var directory = lastSlash > 0 ? filePath.substring(0, lastSlash) : "";
            var fileName = lastSlash > 0 ? filePath.substring(lastSlash + 1) : filePath;

            SP.state.currentPath = directory;
            switchToPlayerMode();
            loadSidebarFiles(directory).then(function() {
                setTimeout(function() {
                    playFile(filePath, fileName);
                }, 100);
            });
        }
    }
}

function initRouter() {
    window.addEventListener("hashchange", handleHashRoute);
    window.addEventListener("popstate", function() {
        if (!window.location.hash.startsWith("#/play/") && SP.state.isPlayerMode) {
            switchToBrowseMode();
        }
    });

    // Initial load
    if (window.location.hash.startsWith("#/play/")) {
        handleHashRoute();
    } else {
        loadBrowseView("");
    }
}
