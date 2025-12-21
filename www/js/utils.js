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
