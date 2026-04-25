#!/bin/sh
set -e

echo "ddddd"
whoami
id

echo "prep"
mkdir -p /app/uploads

chown -R 1000:1000 /app/uploads

echo "ffff"

echo "start"

exec gosu 1000:1000 "$@"
