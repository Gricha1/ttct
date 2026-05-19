#!/usr/bin/env bash
# Canonical overfit check (2L, frozen transformer, balanced batch). Must print SCRIPT_VERSION=3.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/debug_tta_overfit_one_batch.py --dataset "${1:-./dataset/data_debug.pkl}"
