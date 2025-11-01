#!/usr/bin/env bash
# Thin wrapper to keep backward compatibility with docs and scripts
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$SCRIPT_DIR/scripts/setup.sh" ]; then
  exec "$SCRIPT_DIR/scripts/setup.sh" "$@"
else
  echo "Error: scripts/setup.sh not found" >&2
  exit 1
fi

