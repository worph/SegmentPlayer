/* File browser - browse view and sidebar navigation */

function buildBreadcrumb(path, targetId) {
    var parts = path ? path.split("/") : [];
    var html = '<span data-path="">Home</span>';
    var accumulated = "";

    parts.forEach(function(part) {
        accumulated += (accumulated ? "/" : "") + part;
        html += ' / <span data-path="' + accumulated + '">' + part + '</span>';
    });

    var container = document.getElementById(targetId);
    container.innerHTML = html;

    container.querySelectorAll("span").forEach(function(span) {
        span.addEventListener("click", function() {
            SP.state.currentPath = span.dataset.path;
            if (SP.state.isPlayerMode) {
                loadSidebarFiles(SP.state.currentPath);
            } else {
                loadBrowseView(SP.state.currentPath);
            }
        });
    });
}

async function loadBrowseView(path) {
    if (path === undefined) path = "";

    SP.elements.mediaGrid.innerHTML = '<div class="loading"><div class="spinner"></div>Loading media...</div>';
    buildBreadcrumb(path, "browseBreadcrumb");

    try {
        var response = await fetch("/api/files/" + (path ? path + "/" : ""));
        if (!response.ok) throw new Error("Failed to load files");

        var files = await response.json();

        // Sort: folders first, then by name
        files.sort(function(a, b) {
            if (a.type === "directory" && b.type !== "directory") return -1;
            if (a.type !== "directory" && b.type === "directory") return 1;
            return a.name.localeCompare(b.name);
        });

        // Filter to show only videos and folders
        var filtered = files.filter(function(f) {
            return f.type === "directory" || isVideo(f.name);
        });

        // Stats
        var folderCount = filtered.filter(function(f) { return f.type === "directory"; }).length;
        var videoCount = filtered.filter(function(f) { return f.type !== "directory"; }).length;
        SP.elements.browseStats.textContent = folderCount + " folders, " + videoCount + " videos";

        if (filtered.length === 0) {
            SP.elements.mediaGrid.innerHTML =
                '<div class="empty-state" style="grid-column: 1 / -1;">' +
                    '<div class="empty-state-icon">📂</div>' +
                    '<div>No media files found</div>' +
                    '<div style="font-size: 0.85rem; margin-top: 0.5rem; color: #888;">' +
                        'Add video files to your media folder' +
                    '</div>' +
                '</div>';
            return;
        }

        SP.elements.mediaGrid.innerHTML = filtered.map(function(file) {
            var fullPath = path ? path + "/" + file.name : file.name;
            var isFolder = file.type === "directory";

            return '<div class="media-card ' + (isFolder ? 'folder' : '') + '"' +
                     ' data-path="' + fullPath + '"' +
                     ' data-type="' + file.type + '"' +
                     ' data-name="' + file.name + '">' +
                    '<div class="media-card-icon">' + getIcon(file) + '</div>' +
                    '<div class="media-card-info">' +
                        '<div class="media-card-name" title="' + file.name + '">' + file.name + '</div>' +
                        '<div class="media-card-meta">' +
                            (isFolder ? '<span>Folder</span>' : '<span>' + formatSize(file.size || 0) + '</span>') +
                        '</div>' +
                    '</div>' +
                '</div>';
        }).join("");

        // Add click handlers
        SP.elements.mediaGrid.querySelectorAll(".media-card").forEach(function(card) {
            card.addEventListener("click", function() {
                var cardPath = card.dataset.path;
                var cardType = card.dataset.type;
                var cardName = card.dataset.name;

                if (cardType === "directory") {
                    SP.state.currentPath = cardPath;
                    loadBrowseView(cardPath);
                } else if (isVideo(cardName)) {
                    // Switch to player mode and play
                    switchToPlayerMode();
                    loadSidebarFiles(SP.state.currentPath).then(function() {
                        playFile(cardPath, cardName);
                    });
                }
            });
        });

    } catch (err) {
        console.error("Error loading files:", err);
        SP.elements.mediaGrid.innerHTML =
            '<div class="empty-state" style="grid-column: 1 / -1;">' +
                '<div class="empty-state-icon">⚠️</div>' +
                '<div>Error loading files</div>' +
                '<div style="font-size: 0.85rem; margin-top: 0.5rem;">' + err.message + '</div>' +
            '</div>';
    }
}

async function loadSidebarFiles(path) {
    if (path === undefined) path = "";

    SP.elements.fileList.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';
    buildBreadcrumb(path, "sidebarBreadcrumb");

    try {
        var response = await fetch("/api/files/" + (path ? path + "/" : ""));
        if (!response.ok) throw new Error("Failed to load files");

        var files = await response.json();

        files.sort(function(a, b) {
            if (a.type === "directory" && b.type !== "directory") return -1;
            if (a.type !== "directory" && b.type === "directory") return 1;
            return a.name.localeCompare(b.name);
        });

        var filtered = files.filter(function(f) {
            return f.type === "directory" || isVideo(f.name) || isSubtitle(f.name);
        });

        if (filtered.length === 0) {
            SP.elements.fileList.innerHTML =
                '<div class="empty-state">' +
                    '<div class="empty-state-icon">📂</div>' +
                    '<div>No media files</div>' +
                '</div>';
            return;
        }

        SP.elements.fileList.innerHTML = filtered.map(function(file) {
            var fullPath = path ? path + "/" + file.name : file.name;
            var isActive = SP.state.currentFile === fullPath;
            var isFolder = file.type === "directory";

            return '<div class="file-item ' + (isFolder ? 'folder' : '') + ' ' + (isActive ? 'active' : '') + '"' +
                     ' data-path="' + fullPath + '"' +
                     ' data-type="' + file.type + '"' +
                     ' data-name="' + file.name + '">' +
                    '<span class="file-icon">' + getIcon(file) + '</span>' +
                    '<span class="file-name">' + file.name + '</span>' +
                    (file.size ? '<span class="file-size">' + formatSize(file.size) + '</span>' : '') +
                '</div>';
        }).join("");

        SP.elements.fileList.querySelectorAll(".file-item").forEach(function(item) {
            item.addEventListener("click", function() {
                var itemPath = item.dataset.path;
                var itemType = item.dataset.type;
                var itemName = item.dataset.name;

                if (itemType === "directory") {
                    SP.state.currentPath = itemPath;
                    loadSidebarFiles(itemPath);
                } else if (isVideo(itemName)) {
                    playFile(itemPath, itemName);
                }
            });
        });

    } catch (err) {
        console.error("Error loading files:", err);
        SP.elements.fileList.innerHTML =
            '<div class="empty-state">' +
                '<div class="empty-state-icon">⚠️</div>' +
                '<div>Error</div>' +
            '</div>';
    }
}

async function findSubtitles(videoPath) {
    var dir = videoPath.substring(0, videoPath.lastIndexOf("/")) || "";
    var videoName = videoPath.substring(videoPath.lastIndexOf("/") + 1);
    var baseName = videoName.substring(0, videoName.lastIndexOf("."));

    try {
        var response = await fetch("/api/files/" + (dir ? dir + "/" : ""));
        if (!response.ok) return [];

        var files = await response.json();
        return files
            .filter(function(f) { return isSubtitle(f.name) && f.name.startsWith(baseName); })
            .map(function(f) {
                return {
                    name: f.name,
                    path: dir ? dir + "/" + f.name : f.name,
                    lang: extractLang(f.name, baseName)
                };
            });
    } catch (e) {
        return [];
    }
}
