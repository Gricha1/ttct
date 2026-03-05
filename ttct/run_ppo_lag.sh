#!/usr/bin/env bash
# Run PPO-Lag with TTCT and LoRA fine-tuning of encoders.
# Usage from ttct/ttct:
#   export TL_LOADPATH=/path/to/ttct/result/.../model/checkpoint_latest.pt
#   ./run_ppo_lag.sh
#   ./run_ppo_lag.sh --use_comet --comet_project_name "ttct-training"

set -e
cd "$(dirname "$0")"

export TL_LOADPATH=/usr/home/workspace/ttct/result/2026-02-26-15:33:39/model/checkpoint_latest.pt
BATCH_SIZE="${BATCH_SIZE:-128}"

if [ -z "$TL_LOADPATH" ]; then
  echo "Ошибка: не задан путь к чекпоинту TTCT."
  echo "Задайте переменную TL_LOADPATH или передайте --TL_loadpath ..."
  echo "Пример: export TL_LOADPATH=/path/to/ttct/result/.../model/checkpoint_latest.pt"
  echo "        $0"
  exit 1
fi

set -- \
  --use_predict_cost \
  --use_credit_assignment \
  --lagrangian_multiplier_init 0.1 \
  --batch_size "$BATCH_SIZE" \
  --is_finetune \
  --use_lora \
  --rank 8 \
  "$@"
if [ -n "$TL_LOADPATH" ]; then
  set -- --TL_loadpath "$TL_LOADPATH" "$@"
fi

exec python safepo/ppo_lag.py "$@"
