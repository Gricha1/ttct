#!/usr/bin/env bash
# Alias → train_ttct_minigrid.sh (запуск только из ttct/ttct).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_ttct_minigrid.sh" "$@"
