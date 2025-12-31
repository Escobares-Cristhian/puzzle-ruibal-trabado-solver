#!/bin/sh
set -eu

OUTPUT_DIR="${OUTPUT_FOLDER:-/app/output}"

# Ensure host bind-mount path exists and is writable.
mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"

exec "$@"
