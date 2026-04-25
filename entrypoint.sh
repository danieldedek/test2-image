#!/bin/sh
set -e

echo "ddddd"
whoami
id
which gosu || echo "gosu NOT FOUND"

echo "CMD: $@"

exec "$@"
