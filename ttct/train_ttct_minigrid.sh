#!/usr/bin/env bash
# Train TTCT on MiniGrid HazardWorld (flat 7×7×3 obs, obs_dim=147).
#
# Generate data:
#   bash generate_dataset_minigrid.sh              # full → data.pkl
#   bash generate_dataset_minigrid.sh full_fix     # full_fix → data_full_fix.pkl
#   bash generate_debug_dataset_minigrid.sh 2      # debug2 → data_debug_l2.pkl
#   bash generate_debug_dataset_minigrid.sh 3      # debug3 → data_debug_l3.pkl
#
# Train:
#   bash train_ttct_minigrid.sh full               # paper_full (~217 NL, all pairs)
#   bash train_ttct_minigrid.sh full_fix           # ~221 NL, 10000 traj, 1 NL per traj
#   bash train_ttct_minigrid.sh debug2             # 10 NL, 1 constraint per traj
#   bash train_ttct_minigrid.sh debug3             # 10 NL, all pairs (like full)
#
# NEW_ARCH=true (default): 2L/256, transformer_lr=LR*0.1, frozen BERT, no StepLR
# NEW_ARCH=false:          12L/512, uniform LR, StepLR (train.py defaults)
#
# Optional env: NEW_ARCH=false
# Args after mode: [BATCH_SIZE] [EPOCHS] [USE_COMET] [LEARNING_RATE] [CA_LOSS_WEIGHT]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

NEW_ARCH="${NEW_ARCH:-true}"

NEW_ARCH_TRAIN_ARGS=(
  --embed_dim 256
  --transformer_width 256
  --transformer_layers 2
  --tta_loss kl
  --lr_scheduler none
  --weight_decay 0
  --transformer_lr_ratio 0.1
)

_is_new_arch() {
  case "${NEW_ARCH,,}" in
    true|1|yes) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET_MODE="${1:-full}"
case "$DATASET_MODE" in
  full)
    DATASET="${DATASET:-./dataset/data.pkl}"
    COMET_NAME="${COMET_NAME:-TTCT MiniGrid full}"
    shift
    ;;
  debug2)
    DATASET="${DATASET:-./dataset/data_debug_l2.pkl}"
    COMET_NAME="${COMET_NAME:-TTCT MiniGrid debug2}"
    shift
    ;;
  debug3)
    DATASET="${DATASET:-./dataset/data_debug_l3.pkl}"
    COMET_NAME="${COMET_NAME:-TTCT MiniGrid debug3}"
    shift
    ;;
  full_fix)
    DATASET="${DATASET:-./dataset/data_full_fix.pkl}"
    COMET_NAME="${COMET_NAME:-TTCT MiniGrid full_fix}"
    shift
    ;;
  *)
    echo "ERROR: first argument must be 'full', 'full_fix', 'debug2', or 'debug3', got: ${DATASET_MODE}"
    echo "  bash train_ttct_minigrid.sh full"
    echo "  bash train_ttct_minigrid.sh full_fix"
    echo "  bash train_ttct_minigrid.sh debug2"
    echo "  bash train_ttct_minigrid.sh debug3"
    exit 1
    ;;
esac

BATCH_SIZE="${1:-64}"
EPOCHS="${2:-2000}"
USE_COMET="${3:-true}"
LEARNING_RATE="${4:-3e-4}"
CA_LOSS_WEIGHT="${5:-0.01}"

if [ ! -f "$DATASET" ]; then
  echo "ERROR: dataset not found: $DATASET"
  if [ "$DATASET_MODE" = "debug2" ]; then
    echo "  bash generate_debug_dataset_minigrid.sh 2"
  elif [ "$DATASET_MODE" = "debug3" ]; then
    echo "  bash generate_debug_dataset_minigrid.sh 3"
  elif [ "$DATASET_MODE" = "full_fix" ]; then
    echo "  bash generate_dataset_minigrid.sh full_fix"
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

EXTRA_TRAIN_ARGS=()
if _is_new_arch; then
  EXTRA_TRAIN_ARGS=("${NEW_ARCH_TRAIN_ARGS[@]}")
  ARCH_LABEL="NEW_ARCH (2L/256, transformer_lr=LR*0.1)"
else
  ARCH_LABEL="legacy (12L/512, uniform LR, StepLR)"
fi

echo "====================================================="
echo "RUN: train_ttct_minigrid"
echo "  dataset_mode=${DATASET_MODE}  ${ARCH_LABEL}"
echo "  path=${DATASET}"
echo "  batch=${BATCH_SIZE}  epochs=${EPOCHS}  lr=${LEARNING_RATE}"
echo "====================================================="

python train.py \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --ca_loss_weight "$CA_LOSS_WEIGHT" \
  --tta_loss kl \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --obs_dim 147 \
  --obs_emb_dim 64 \
  --trajectory_length 200 \
  --val_viz_every_epochs 1 \
  --val_viz_every_steps 100 \
  --val_viz_n_violated 2 \
  --val_viz_n_safe 2 \
  --val_viz_frames 8 \
  --dataloader_num_workers 0 \
  $COMET_FLAG \
  --comet_project_name "ttct-training" \
  --comet_experiment_name "$COMET_NAME"

echo "Done. Checkpoint: result/<timestamp>/model/checkpoint_latest.pt"
