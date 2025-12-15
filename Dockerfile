# SegmentPlayer - On-the-fly HLS streaming with live segment transcoding
# Based on nginx-vod-module with Python transcoder for incompatible codecs

FROM alfg/nginx-vod-module:latest

LABEL org.opencontainers.image.title="SegmentPlayer"
LABEL org.opencontainers.image.description="On-the-fly HLS streaming with live segment-level transcoding"
LABEL org.opencontainers.image.source="https://github.com/Worph/SegmentPlayer"
LABEL org.opencontainers.image.vendor="Worph"

# Install Python and FFmpeg for live transcoding
RUN apk add --no-cache \
    python3 \
    py3-pip \
    ffmpeg \
    curl \
    supervisor

# Create directory structure
RUN mkdir -p /data/www /data/media /data/cache /app /var/log/supervisor

# Copy application files
COPY nginx/nginx.conf /usr/local/nginx/conf/nginx.conf
COPY www/index.html /data/www/index.html
COPY transcoder/server.py /app/server.py

# Create supervisord config for process management
RUN cat > /etc/supervisord.conf <<'EOF'
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
user=root

[program:nginx]
command=/usr/local/nginx/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:transcoder]
command=python3 /app/server.py
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=MEDIA_DIR="/data/media",CACHE_DIR="/data/cache",SEGMENT_DURATION="4",PORT="8080"
EOF

# Environment variables
ENV MEDIA_DIR=/data/media \
    CACHE_DIR=/data/cache \
    SEGMENT_DURATION=4 \
    PORT=8080

# Expose web port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# Use supervisord to manage both processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
