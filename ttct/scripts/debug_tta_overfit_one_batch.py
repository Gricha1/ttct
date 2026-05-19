#!/usr/bin/env python3
"""Overfit one balanced batch (paper: B×B + gen_mask(TL) + KL). SCRIPT_VERSION=4"""
SCRIPT_VERSION = 4
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TTCT import TTCT
from utils import KLLoss, gen_mask

LAVA_KEY = "cross lava"
GRASS_KEY = "grass"


def dedupe_one_per_trajectory(items):
    seen = {}
    for item in items:
        obs, act, tls, length, nl = item
        key = np.asarray(obs, dtype=np.float32).tobytes()
        if key not in seen:
            seen[key] = item
    return list(seen.values())


def _is_lava(nl: str) -> bool:
    return LAVA_KEY in nl.lower()


def balanced_batch(data, batch_size: int):
    lava = [x for x in data if _is_lava(x[4])]
    grass = [x for x in data if not _is_lava(x[4])]
    half = batch_size // 2
    if len(lava) < half or len(grass) < half:
        raise RuntimeError(
            f"Need >={half} lava and >={half} grass rows, got {len(lava)} lava, {len(grass)} grass"
        )
    return lava[:half] + grass[:half]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "dataset" / "data_debug.pkl"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--no_dedupe", action="store_true")
    p.add_argument(
        "--full",
        action="store_true",
        help="12L/512 (paper-sized; often fails on 16 samples)",
    )
    p.add_argument(
        "--no_freeze_bert",
        action="store_true",
        help="Разморозить BERT (по умолчанию заморожен).",
    )
    p.add_argument(
        "--train_transformer",
        action="store_true",
        help="Обучать trajectory_transformer (по умолчанию заморожен; на 16 сэмплах часто не сходится).",
    )
    p.add_argument(
        "--lr_transformer",
        type=float,
        default=None,
        help="LR для trajectory_transformer при --train_transformer (default: lr/10).",
    )
    args = p.parse_args()
    if args.lr_transformer is None:
        args.lr_transformer = args.lr / 10.0 if args.train_transformer else args.lr
    args.small = not args.full

    print("=" * 60)
    print(f"debug_tta_overfit_one_batch.py  SCRIPT_VERSION={SCRIPT_VERSION} (paper TTA)")
    freeze_tr = not args.train_transformer
    freeze_bert = not args.no_freeze_bert
    print(
        f"config: layers={2 if args.small else 12}, width={256 if args.small else 512}, "
        f"lr={args.lr}, loss=KL, mask=gen_mask(TL), logits=B×B, "
        f"freeze_bert={freeze_bert}, freeze_transformer={freeze_tr}"
    )
    print("=" * 60)

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    if not args.no_dedupe:
        data = dedupe_one_per_trajectory(data)
    batch = balanced_batch(data, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_len = 200
    ctx = 77
    if args.small:
        tw, layers, ed = 256, 2, 256
    else:
        tw, layers, ed = 512, 12, 512

    model = TTCT(
        embed_dim=ed,
        act_dim=1,
        obs_dim=147,
        obs_emb_dim=64,
        trajectory_length=traj_len,
        context_length=ctx,
        vocab_size=49408,
        transformer_width=tw,
        transformer_heads=8,
        transformer_layers=layers,
        BERT_PATH="bert-base-uncased",
        device=device,
    ).to(device)
    if freeze_bert:
        for param in model.text_model.parameters():
            param.requires_grad = False
    print(f"BERT: {'FROZEN' if freeze_bert else 'TRAINABLE (CPU)'}")
    if freeze_tr:
        for param in model.trajectory_transformer.parameters():
            param.requires_grad = False
    print(
        f"trajectory_transformer: {'FROZEN' if freeze_tr else 'TRAINABLE'} "
        f"({layers} layers, width={tw})"
    )

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    tr_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or name == "logit_scale":
            continue
        if name.startswith("trajectory_transformer."):
            tr_params.append(param)
        else:
            other_params.append(param)
    bert_n = sum(p.numel() for p in model.text_model.parameters() if p.requires_grad)
    print(f"trainable BERT params: {bert_n:,}")
    opt_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": [model.logit_scale], "lr": args.lr * 10.0},
    ]
    if tr_params:
        opt_groups.insert(0, {"params": tr_params, "lr": args.lr_transformer})
        print(f"optimizer: transformer lr={args.lr_transformer}, rest lr={args.lr}")
    opt = torch.optim.Adam(opt_groups, weight_decay=0)
    if args.train_transformer:
        print(
            "WARN: --train_transformer on 16 samples is hard; try without it for sanity check "
            "(frozen transformer usually reaches acc~1 in ~50 steps)."
        )

    obss, acts, tlss, _, nlss = zip(*batch)
    label_counts = Counter("lava" if _is_lava(n) else "grass" for n in nlss)
    print(f"batch labels: {dict(label_counts)}")

    lengths = []
    padded_obss, padded_acts = [], []
    for obs, act in zip(obss, acts):
        lengths.append(len(obs))
        padded_obss.append(
            np.pad(
                obs,
                ((0, traj_len - len(obs)), (0, 0), (0, 0), (0, 0)),
                constant_values=0,
            )
        )
        padded_acts.append(np.pad(act, (0, traj_len - len(act)), constant_values=0))
    lengths = np.array(lengths, dtype=np.int32)
    obs_t = torch.tensor(np.array(padded_obss), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(padded_acts), dtype=torch.float32, device=device)

    ids, masks = [], []
    for sent in nlss:
        enc = tok.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=ctx,
            padding="max_length",
            return_tensors="pt",
        )
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    ids = torch.cat(ids, 0).to(device)
    masks = torch.cat(masks, 0).to(device)

    _, mask_np, mask_count = gen_mask(list(tlss))
    print(f"batch={len(batch)}, gen_mask positives={mask_count:.0f}, shape={mask_np.shape}")
    mask_t = torch.tensor(mask_np, device=device, dtype=torch.float)
    row_is_lava = torch.tensor(
        [_is_lava(n) for n in nlss], device=device, dtype=torch.bool
    )

    def retrieval_acc(logits: torch.Tensor) -> float:
        """Row i: argmax column has same constraint type (lava/grass) as row i."""
        pred_j = logits.argmax(dim=1)
        return (row_is_lava == row_is_lava[pred_j]).float().mean().item()

    loss_traj = KLLoss()
    loss_text = KLLoss()

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(obs_t, act_t, ids, masks, lengths)
        loss = (loss_traj(logits, mask_t) + loss_text(logits.t(), mask_t.t())) / 2
        loss.backward()
        gn = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            acc_r = retrieval_acc(logits)
            pc = Counter(pred.cpu().tolist())
        if step % 50 == 0 or step == args.steps - 1:
            probs = F.softmax(logits, dim=1)
            ent = -(probs * probs.clamp(min=1e-8).log()).sum(1).mean()
            top2 = torch.topk(logits, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).mean()
            print(
                f"step {step:3d} loss={loss.item():.4f} acc_retrieval={acc_r:.3f} "
                f"ent={ent.item():.3f} margin={margin.item():.4f} grad={gn:.2e} pred_cols={dict(pc)}"
            )

    with torch.no_grad():
        logits, _ = model(obs_t, act_t, ids, masks, lengths)
        acc_r = retrieval_acc(logits)
        correct = int(round(acc_r * len(nlss)))
    print(f"final: acc_retrieval={acc_r:.3f} ({correct}/{len(nlss)})")

    if acc_r < 0.9:
        print("FAIL: overfit did not reach acc_retrieval>=0.9 on this batch.")
        if args.train_transformer:
            print("  tip: run without --train_transformer (frozen transformer).")
        sys.exit(1)
    print("OK: one-batch overfit passed.")


if __name__ == "__main__":
    main()
