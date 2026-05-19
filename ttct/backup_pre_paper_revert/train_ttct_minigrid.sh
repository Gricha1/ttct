#!/usr/bin/env bash
# Train TTCT on MiniGrid HazardWorld (flat 7×7×3 obs, obs_dim=147).
#
# Usage (from ttct/ttct):
#   bash generate_dataset_minigrid.sh
#   bash train_ttct_minigrid.sh
#
# Debug (2 NL, ~40 traj, config matched to scripts/debug_tta_overfit_one_batch.py):
#   bash generate_debug_dataset_minigrid.sh
#   bash train_ttct_minigrid.sh debug
#
# Full run: [DATASET] [BATCH_SIZE] [EPOCHS] [USE_COMET] [LEARNING_RATE] [CA_LOSS_WEIGHT]
# Debug:    [BATCH_SIZE] [EPOCHS] [USE_COMET] [LEARNING_RATE] [CA_LOSS_WEIGHT]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

DEBUG_MODE=false
if [ "${1:-}" = "debug" ]; then
  DEBUG_MODE=true
  shift
fi

if [ "$DEBUG_MODE" = true ]; then
  DATASET="${DATASET:-./dataset/data_debug.pkl}"
  BATCH_SIZE="${1:-16}"
  EPOCHS="${2:-40}"
  USE_COMET="${3:-true}"
  LEARNING_RATE="${4:-3e-4}"
  CA_LOSS_WEIGHT="${5:-0}"
  COMET_NAME="${COMET_NAME:-TTCT MiniGrid DEBUG}"
else
  DATASET="${1:-./dataset/data.pkl}"
  BATCH_SIZE="${2:-64}"
  EPOCHS="${3:-32}"
  USE_COMET="${4:-true}"
  LEARNING_RATE="${5:-1e-4}"
  CA_LOSS_WEIGHT="${6:-0}"
  COMET_NAME="${COMET_NAME:-TTCT MiniGrid}"
fi

if [ ! -f "$DATASET" ]; then
  echo "ERROR: dataset not found: $DATASET"
  if [ "$DEBUG_MODE" = true ]; then
    echo "  bash generate_debug_dataset_minigrid.sh"
  else
    echo "  bash generate_dataset_minigrid.sh"
  fi
  exit 1
fi

if [ "$USE_COMET" = "true" ]; then
  if ! python -c "import comet_ml" 2>/dev/null; then
    echo "Installing comet_ml..."
    pip install comet_ml
  fi
  COMET_FLAG="--use_comet"
else
  COMET_FLAG=""
fi

echo "====================================================="
echo "RUN: train_ttct_minigrid"
if [ "$DEBUG_MODE" = true ]; then
  echo "MODE=debug (2L transformer frozen, 256-dim — same as passing overfit test)"
fi
echo "AGPASS_DATASET_PATH=${DATASET}"
echo "AGPASS_HP_BATCH_SIZE=${BATCH_SIZE}"
echo "AGPASS_HP_EPOCHS=${EPOCHS}"
echo "AGPASS_HP_LEARNING_RATE=${LEARNING_RATE}"
echo "====================================================="

EXTRA_TRAIN_ARGS=()
if [ "$DEBUG_MODE" = true ]; then
  EXTRA_TRAIN_ARGS=(
    --embed_dim 256
    --transformer_width 256
    --transformer_layers 2
    --freeze_trajectory_transformer
    --tta_text_mode unique_nl
    --tta_loss kl
    --tta_skip_inner_ce
    --no_freeze_bert
    --tta_temperature 0.2
    --lr_scheduler none
    --weight_decay 0
  )
fi

python train.py \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --ca_loss_weight "$CA_LOSS_WEIGHT" \
  --tta_loss soft_ce \
  --tta_temperature 0.1 \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --obs_dim 147 \
  --obs_emb_dim 64 \
  --trajectory_length 200 \
  --val_viz_every_epochs 1 \
  --val_viz_every_steps 100 \
  --val_viz_n_violated 2 \
  --val_viz_n_safe 2 \
  --val_viz_frames 8 \
  $COMET_FLAG \
  --comet_project_name "ttct-training" \
  --comet_experiment_name "$COMET_NAME"

echo "Done. Checkpoint: result/<timestamp>/model/checkpoint_latest.pt"
