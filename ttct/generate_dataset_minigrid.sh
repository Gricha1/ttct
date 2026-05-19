#!/usr/bin/env bash
# Generate TTCT dataset (MiniGrid HazardWorld) via generate_dataset_from_paper.py.
#
# Usage (from ttct/ttct):
#   bash generate_dataset_minigrid.sh              # full: paper_full, all pairs, 1000 traj
#   bash generate_dataset_minigrid.sh full_fix     # ~221 NL, 1 random NL per traj, 10000 traj
#
# Override via env:
#   NUM_TRAJ=2000 OUT_PATH=./dataset/data.pkl bash generate_dataset_minigrid.sh
#   CONSTRAINT_POOL=legacy_30 bash generate_dataset_minigrid.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-full}"
case "$MODE" in
  full)
    ENV_NAME="${ENV_NAME:-MiniGrid-HazardWorld-B-v0}"
    NUM_TRAJ="${NUM_TRAJ:-1000}"
    MAX_STEPS="${MAX_STEPS:-200}"
    OUT_PATH="${OUT_PATH:-./dataset/data.pkl}"
    CONSTRAINT_POOL="${CONSTRAINT_POOL:-paper_full}"
    TRAIN_HINT="bash train_ttct_minigrid.sh full"
    ;;
  full_fix)
    ENV_NAME="${ENV_NAME:-MiniGrid-HazardWorld-B-v0}"
    NUM_TRAJ="${NUM_TRAJ:-10000}"
    MAX_STEPS="${MAX_STEPS:-200}"
    OUT_PATH="${OUT_PATH:-./dataset/data_full_fix.pkl}"
    CONSTRAINT_POOL="${CONSTRAINT_POOL:-full_fix}"
    TRAIN_HINT="bash train_ttct_minigrid.sh full_fix"
    ;;
  *)
    echo "ERROR: first argument must be 'full' or 'full_fix', got: ${MODE}"
    echo "  bash generate_dataset_minigrid.sh"
    echo "  bash generate_dataset_minigrid.sh full_fix"
    exit 1
    ;;
esac

# HazardWorld lives in safepo/gym_minigrid (NOT PyPI gym_minigrid)
echo "Installing safepo fork (gym_minigrid + HazardWorld) from ${SCRIPT_DIR} ..."
pip install --no-build-isolation --no-deps -e "${SCRIPT_DIR}"

echo "Checking HazardWorld env registration (${ENV_NAME})..."
python3 -c "
import ensure_safepo_paths
from generate_dataset_from_paper import _make_hazardworld_env
ensure_safepo_paths.ensure_hazardworld_env('${ENV_NAME}')
env = _make_hazardworld_env('${ENV_NAME}')
env.reset()
o, r, term, trunc, info = env.step(env.action_space.sample())
env.close()
print('OK: env ready, step tuple len=5')
"

echo "MiniGrid TTCT dataset [${MODE}]: env=${ENV_NAME} trajectories=${NUM_TRAJ} pool=${CONSTRAINT_POOL}"

python3 generate_dataset_from_paper.py \
  --env_name "${ENV_NAME}" \
  --num_trajectories "${NUM_TRAJ}" \
  --max_steps "${MAX_STEPS}" \
  --output_path "${OUT_PATH}" \
  --constraint_pool "${CONSTRAINT_POOL}"

echo ""
echo "Checking dataset observations (must be integer MiniGrid grid, not synthetic randn)..."
python3 <<PY
import pickle
import sys
import numpy as np
from collections import Counter

out = "${OUT_PATH}"
mode = "${MODE}"
with open(out, "rb") as f:
    data = pickle.load(f)
obs0 = np.asarray(data[0][0][0])
is_int_grid = (
    np.allclose(obs0, np.rint(obs0), atol=0.05)
    and obs0[:, :, 0].max() <= 12
    and obs0[:, :, 0].min() >= 0
)
if not is_int_grid:
    print("ERROR: obs are float noise — env likely failed and synthetic data was used.", file=sys.stderr)
    print(f"  sample min/max: {obs0.min():.3f} / {obs0.max():.3f}", file=sys.stderr)
    print("  Fix: pip install -e .  &&  python3 -c \"import gym_minigrid; import gym; gym.make('${ENV_NAME}')\"", file=sys.stderr)
    sys.exit(1)
print("OK: obs look like MiniGrid object grid (type/color/state).")

if mode == "full_fix":
    n = len(data)
    nl = Counter(item[4] for item in data)
    unique_traj = len({item[0].tobytes() for item in data})
    print(f"pairs={n}, unique trajectories={unique_traj}, unique NL texts={len(nl)}")
    if unique_traj != n:
        print(f"ERROR: expected one row per trajectory, got {n} rows / {unique_traj} trajs", file=sys.stderr)
        sys.exit(1)
    for text, cnt in sorted(nl.items(), key=lambda x: -x[1])[:8]:
        print(f"  [{cnt:4d}] {text[:72]}")
    if len(nl) > 8:
        print(f"  ... and {len(nl) - 8} more NL types")
PY

echo ""
echo "Done. Dataset: ${OUT_PATH}"
echo "Next: ${TRAIN_HINT}"
