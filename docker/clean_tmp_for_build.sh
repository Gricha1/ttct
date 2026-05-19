#!/bin/bash
# Удаляет ttct/.tmp (часто создаётся с правами nobody при запуске в контейнере),
# из-за чего docker build выдаёт "can't stat .../.tmp/...". Запускать перед build.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/ttct/.tmp"
if [[ -d "$TMP_DIR" ]]; then
  echo "Removing $TMP_DIR (may need sudo if permission denied)"
  rm -rf "$TMP_DIR" 2>/dev/null || sudo rm -rf "$TMP_DIR"
  echo "Done."
else
  echo "No $TMP_DIR — nothing to clean."
fi
