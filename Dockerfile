# SegmentPlayer - On-the-fly HLS streaming with live segment transcoding
# Uses nginx-vod-module base + static FFmpeg 7.x for faster AV1 decode

FROM alfg/nginx-vod-module:latest

LABEL org.opencontainers.image.title="SegmentPlayer"
LABEL org.opencontainers.image.description="On-the-fly HLS streaming with live segment-level transcoding"
LABEL org.opencontainers.image.source="https://github.com/Worph/SegmentPlayer"
LABEL org.opencontainers.image.vendor="Worph"

# Install dependencies and download static FFmpeg 7.x with modern libdav1d
# Static build from johnvansickle.com includes libdav1d 1.4.x
RUN apk add --no-cache \
        python3 \
        curl \
        supervisor \
        xz \
        gettext \
    && curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz \
    && tar xf /tmp/ffmpeg.tar.xz -C /tmp \
    && mv /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg* \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && /usr/local/bin/ffmpeg -version | head -3 \
    && echo "=== Checking libdav1d ===" \
    && /usr/local/bin/ffmpeg -decoders 2>&1 | grep -i dav1d

# Create directory structure
# Client-side demux/decode libraries (libav.js + WebCodecs bridge + sp-audio
# variant) are committed under www/vendor/ and copied in with the rest of www.
RUN mkdir -p /data/www /data/media /data/cache /app /var/log/supervisor

# Copy application files
COPY nginx/nginx.conf /usr/local/nginx/conf/nginx.conf.template
COPY www/ /data/www/
COPY transcoder/ /app/

# Ensure readable permissions (directories need 755, files need 644)
RUN find /data/www -type f -exec chmod 644 {} \; && \
    find /data/www -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \;

# Process management + entrypoint. These are COPY'd from real files (not
# generated with `RUN cat <<'EOF'` heredocs) so the build works on the legacy
# Docker builder too — heredocs require BuildKit and silently produce 0-byte
# files otherwise, which makes the container die with "exec format error".
# .gitattributes pins these to LF so a Windows checkout can't reintroduce CRLF.
COPY supervisord.conf /etc/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment variables
ENV MEDIA_DIR=/data/media \
    CACHE_DIR=/data/cache \
    SEGMENT_DURATION=4 \
    PORT=8080 \
    NGINX_PORT=80 \
    PATH="/usr/local/bin:$PATH"

# Expose default web port
EXPOSE 80

# Health check - uses NGINX_PORT env var
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${NGINX_PORT}/ || exit 1

# Use entrypoint to configure and start
CMD ["/entrypoint.sh"]
