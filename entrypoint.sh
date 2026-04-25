#!/bin/sh
set -e

mkdir -p /app/uploads

chown -R 1000:1000 /app/uploads

exec su-exec 1000:1000 "$@"
