# TTCT

TTCT: Trajectory-level Textual Constraints Translator

## Installation

To install the required dependencies, run the following command:

`pip install -r requirements.txt`

## Train TTCT

Generate your own dataset followed by article's appendix and put it in `./dataset/data.pkl`. Then train TTCT from the **parent folder** `ttct/`:

```bash
cd ttct
./train_ttct.sh ./dataset/data.pkl 16 32 false
```

Or manually: `python train.py --dataset ./dataset/data.pkl --batch_size 16 --epochs 32`

Script usage: `./train_ttct.sh [DATASET] [BATCH_SIZE] [EPOCHS] [USE_COMET]`. Requires `train.py` in `ttct/`.

## Train Policy

After you train your own TTCT model, you can use TTCT to train a safe agent that constrained by trajectory-level text constrains.

### PPO-Lag with TTCT encoder fine-tuning (LoRA)

As in the paper (Section 5): the trajectory and text encoders are used in two ways — **frozen** (g_T, g_C) for cost prediction, and **trainable** (g*_T, g*_C) with LoRA for encoding history as policy input. To run PPO-Lag with LoRA fine-tuning of the TTCT encoders:

```bash
cd ttct/safepo
python cppo_pid.py \
  --use_predict_cost \
  --use_credit_assignment \
  --lagrangian_multiplier_init 0.1 \
  --TL_loadpath /path/to/ttct/result/.../model/checkpoint_latest.pt \
  --is_finetune \
  --use_lora \
  --rank 8
```

- `--is_finetune`: create a copy of TTCT (EncodeModel) for policy input and fine-tune it; frozen TLmodel is used only for cost prediction. (Default: True)
- `--use_lora`: apply LoRA to EncodeModel; only LoRA parameters are trained. (Default: True)
- `--rank`: LoRA rank (default: 8). Table 3 in the paper uses the same architecture; LoRA is applied to query/value layers in the transformer.

From the project root `ttct/ttct` you can use:

```bash
TL_LOADPATH=/path/to/ttct/model/checkpoint_latest.pt ./run_ppo_lag.sh --is_finetune --use_lora --rank 8
```

(Omitting `--is_finetune` and `--use_lora` also enables them by default in the current config.)

### Other training modes

1. Train with the full TTCT (cost prediction + credit assignment): `python cppo_pid.py --use_predict_cost --use_credit_assignment --lagrangian_multiplier_init=0.1 --TL_loadpath=/your/TTCT/model/path`
2. TTCT without credit assignment: `python cppo_pid.py --use_predict_cost --lagrangian_multiplier_init=0.1 --TL_loadpath=/your/TTCT/model/path`
3. Ground-truth cost (no TTCT): `python cppo_pid.py --lagrangian_multiplier_init=0.1 --TL_loadpath=/dummy/path` (and `--use_pretrained_encoders false` if you do not want to load TTCT)
