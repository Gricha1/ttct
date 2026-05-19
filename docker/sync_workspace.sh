#!/usr/bin/env bash
# Re-link editable packages after editing ttct / Craftax on the host volume.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/usr/home/workspace}"
export WORKSPACE_ROOT="$WORKSPACE"
source /opt/conda/etc/profile.d/conda.sh
conda activate dynalang

if [[ -f "$WORKSPACE/ttct/setup.py" ]]; then
  pip install --no-build-isolation --no-deps -e "$WORKSPACE/ttct"
fi
craftax="$WORKSPACE/caged_craftext/Craftax"
if [[ -d "$craftax" ]]; then
  pip install --no-build-isolation --no-deps -e "$craftax"
fi
echo "sync_workspace: done"
