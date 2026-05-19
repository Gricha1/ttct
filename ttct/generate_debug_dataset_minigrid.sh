#!/usr/bin/env bash
# MiniGrid debug datasets for TTCT (HazardWorld).
#
# Usage (from ttct/ttct):
#   bash generate_debug_dataset_minigrid.sh 2      # debug2: 10 NL, 1 per traj → data_debug_l2.pkl
#   bash generate_debug_dataset_minigrid.sh 3      # debug3: 10 NL, all pairs → data_debug_l3.pkl
#   bash generate_debug_dataset_minigrid.sh 3 300  # debug3, 300 trajectories
#
# Train:
#   bash train_ttct_minigrid.sh debug2
#   bash train_ttct_minigrid.sh debug3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LEVEL="${1:-2}"
shift || true

ENV_NAME="${ENV_NAME:-MiniGrid-HazardWorld-B-v0}"
MAX_STEPS="${MAX_STEPS:-80}"

case "$LEVEL" in
  2|l2)
    NUM_TRAJ="${1:-200}"
    OUT_PATH="${OUT_PATH:-./dataset/data_debug_l2.pkl}"
    CONSTRAINT_POOL="${CONSTRAINT_POOL:-debug_10}"
    EXPECT_NL=10
    EXPECT_PAIRS_MIN=$((NUM_TRAJ * 8 / 10))
    LEVEL_LABEL="debug2: 10 NL, 1 random constraint per trajectory"
    ;;
  3|l3)
    NUM_TRAJ="${1:-200}"
    OUT_PATH="${OUT_PATH:-./dataset/data_debug_l3.pkl}"
    CONSTRAINT_POOL="${CONSTRAINT_POOL:-debug_10_all}"
    EXPECT_NL=10
    EXPECT_PAIRS_MIN=$((NUM_TRAJ * 10))
    LEVEL_LABEL="debug3: 10 NL, all traj×constraint pairs (like full)"
    ;;
  *)
    echo "ERROR: level must be 2 (debug2) or 3 (debug3), got: ${LEVEL}"
    exit 1
    ;;
esac

echo "====================================================="
echo "RUN: generate_debug_dataset_minigrid"
echo "  ${LEVEL_LABEL}"
echo "  env=${ENV_NAME}"
echo "  trajectories=${NUM_TRAJ}"
echo "  max_steps=${MAX_STEPS}"
echo "  constraint_pool=${CONSTRAINT_POOL}"
echo "  output=${OUT_PATH}"
echo "====================================================="

pip install --no-build-isolation --no-deps -e "${SCRIPT_DIR}" 2>/dev/null || \
  pip install --no-build-isolation --no-deps -e "${SCRIPT_DIR}"

python3 -c "
import ensure_safepo_paths
from generate_dataset_from_paper import _make_hazardworld_env
ensure_safepo_paths.ensure_hazardworld_env('${ENV_NAME}')
env = _make_hazardworld_env('${ENV_NAME}')
env.reset()
env.close()
print('OK: HazardWorld env ready')
"

python3 generate_dataset_from_paper.py \
  --env_name "${ENV_NAME}" \
  --num_trajectories "${NUM_TRAJ}" \
  --max_steps "${MAX_STEPS}" \
  --output_path "${OUT_PATH}" \
  --constraint_pool "${CONSTRAINT_POOL}"

python3 <<PY
import pickle
import sys
import numpy as np
from collections import Counter

out = "${OUT_PATH}"
expect_nl = int("${EXPECT_NL}")
expect_pairs_min = int("${EXPECT_PAIRS_MIN}")
level = "${LEVEL}"
with open(out, "rb") as f:
    data = pickle.load(f)
n = len(data)
nl = Counter(item[4] for item in data)
obs0 = np.asarray(data[0][0][0])
ok_grid = (
    np.allclose(obs0, np.rint(obs0), atol=0.05)
    and obs0[:, :, 0].max() <= 12
    and obs0[:, :, 0].min() >= 0
)
print(f"pairs={n}, unique NL={len(nl)} (pool has {expect_nl} texts)")
print(f"expected pairs >= {expect_pairs_min}")
for text, cnt in sorted(nl.items(), key=lambda x: -x[1]):
    print(f"  [{cnt:4d}] {text[:72]}")
if len(nl) < expect_nl:
    print(f"WARN: fewer than {expect_nl} NL types in data", file=sys.stderr)
if n < expect_pairs_min:
    print(f"WARN: pairs={n} < expected min {expect_pairs_min}", file=sys.stderr)
unique_traj = len({item[0].tobytes() for item in data})
print(f"unique trajectories={unique_traj}")
if level in ("3", "l3") and n < unique_traj * expect_nl * 0.9:
    print("WARN: debug3 expected ~traj*10 pairs", file=sys.stderr)
if not ok_grid:
    print("ERROR: obs are not MiniGrid grids", file=sys.stderr)
    sys.exit(1)
print("OK: debug dataset ready")
PY

echo ""
echo "Done: ${OUT_PATH}"
echo "Train: bash train_ttct_minigrid.sh debug${LEVEL}"
