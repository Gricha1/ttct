#!/usr/bin/env bash
# Validate PPO-Lag checkpoint: run episode(s), record video, save to folder, send to Comet ML.
# Usage from ttct/ttct:
#   ./run_validate_ppo_lag.sh
#   ./run_validate_ppo_lag.sh /path/to/checkpoint
#   ./run_validate_ppo_lag.sh /path/to/checkpoint --output_dir ./videos --num_episodes 3
# Set COMET_API_KEY in env if needed. Video is logged to project ttct-training.
#
# Default checkpoint: ../runs/single_agent_exp/MiniGrid/our/ppo_lag/seed-000-2026-03-05-20-30-19.255153

set -e
cd "$(dirname "$0")"

DEFAULT_CHECKPOINT="../runs/single_agent_exp/MiniGrid/our/ppo_lag/seed-000-2026-03-05-20-30-19.255153"

# First arg: checkpoint dir if it's a path (not --option)
if [ -n "$1" ] && [ -d "$1" ] && [[ "$1" != --* ]]; then
  CHECKPOINT_DIR="$1"
  shift
else
  CHECKPOINT_DIR="$DEFAULT_CHECKPOINT"
fi

# TTCT base model (required for encoding). Override via env if needed.
export TL_LOADPATH="${TL_LOADPATH:-/usr/home/workspace/ttct/result/2026-02-26-15:33:39/model/checkpoint_latest.pt}"
# Comet ML: нужен для отправки видео в проект ttct-training (запуск будет называться "validation")
export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Error: checkpoint dir not found: $CHECKPOINT_DIR"
  echo "Usage: $0 [checkpoint_dir] [--output_dir DIR] [--num_episodes N] ..."
  exit 1
fi

echo "Checkpoint: $CHECKPOINT_DIR"
echo "TL_LOADPATH: $TL_LOADPATH"
echo ""

exec python safepo/validate_ppo_lag.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --TL_loadpath "$TL_LOADPATH" \
  --use_comet \
  --comet_project_name "ttct-training" \
  "$@"
