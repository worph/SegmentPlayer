#!/bin/sh
# Substitute environment variables in nginx config, then start supervisord.
envsubst '${NGINX_PORT}' < /usr/local/nginx/conf/nginx.conf.template > /usr/local/nginx/conf/nginx.conf
exec /usr/bin/supervisord -c /etc/supervisord.conf
