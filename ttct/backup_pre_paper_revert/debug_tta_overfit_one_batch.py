#!/usr/bin/env python3
"""Overfit one balanced batch (8 lava + 8 grass). SCRIPT_VERSION=3"""
SCRIPT_VERSION = 3
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
from utils import MultiPositiveContrastiveLoss, KLLoss, gen_mask_from_nl

LAVA_KEY = "cross lava"
GRASS_KEY = "grass"


def align_obs_act(obs, act):
    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    if len(obs) == len(act) + 1:
        obs = obs[:-1]
    n = max(min(len(obs), len(act)), 1)
    return obs[:n], act[:n], n


def dedupe_one_per_trajectory(items):
    seen = {}
    for item in items:
        obs, act, tls, length, nl = item
        obs_a, act_a, n = align_obs_act(obs, act)
        key = obs_a.tobytes()
        if key not in seen:
            seen[key] = (obs_a, act_a, tls, n, nl)
    return list(seen.values())


def _is_lava(nl: str) -> bool:
    return LAVA_KEY in nl.lower()


def balanced_batch(data, batch_size: int):
    """Equal lava / grass rows so acc=0.5 majority baseline is obvious."""
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
    p.add_argument("--temperature", type=float, default=0.1, help="soft_ce temperature")
    p.add_argument(
        "--tta_loss",
        type=str,
        default="kl",
        choices=("kl", "soft_ce"),
        help="TTA loss (same as train.py)",
    )
    p.add_argument("--no_dedupe", action="store_true")
    p.add_argument(
        "--train_transformer",
        action="store_true",
        help="Train full 12L transformer (default: frozen, train encoders+projections)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="12L/512 (almost always FAIL on 16 samples; need --allow-full)",
    )
    p.add_argument(
        "--allow-full",
        action="store_true",
        help="Acknowledge that --full usually gives acc~0.5 on this test.",
    )
    args = p.parse_args()
    if args.full and not args.allow_full:
        print("ERROR: --full disabled by default (acc~0.5 on 16 samples).")
        print("  Use default (no flags), or: --full --allow-full --train_transformer")
        sys.exit(2)
    if args.train_transformer and not args.full:
        print("WARN: --train_transformer with 2L frozen is OK; without --full uses small+frozen.")
    args.small = not args.full

    print("=" * 60)
    print(f"debug_tta_overfit_one_batch.py  SCRIPT_VERSION={SCRIPT_VERSION}")
    print(
        f"config: layers={2 if args.small else 12}, width={256 if args.small else 512}, "
        f"freeze_transformer={not args.train_transformer}, lr={args.lr}, T={args.temperature}"
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
    for param in model.text_model.parameters():
        param.requires_grad = False
    freeze_tr = not args.train_transformer
    if freeze_tr:
        for param in model.trajectory_transformer.parameters():
            param.requires_grad = False
    print(
        f"trajectory_transformer: {'FROZEN' if freeze_tr else 'TRAINABLE'} "
        f"({layers} layers, width={tw})"
    )

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and n != "logit_scale"]
    opt = torch.optim.Adam(
        [
            {"params": base_params, "lr": args.lr},
            {"params": [model.logit_scale], "lr": args.lr * 10.0},
        ],
        weight_decay=0,
    )

    obss, acts, _, _, nlss = zip(*batch)
    label_counts = Counter("lava" if _is_lava(n) else "grass" for n in nlss)
    print(f"batch labels: {dict(label_counts)}")

    lengths = []
    padded_obss, padded_acts = [], []
    for obs, act in zip(obss, acts):
        obs_a, act_a, n = align_obs_act(obs, act)
        lengths.append(n)
        padded_obss.append(
            np.pad(
                obs_a,
                ((0, traj_len - n), (0, 0), (0, 0), (0, 0)),
                constant_values=0,
            )
        )
        padded_acts.append(np.pad(act_a, (0, traj_len - n), constant_values=0))
    lengths = np.array(lengths, dtype=np.int32)
    obs_t = torch.tensor(np.array(padded_obss), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(padded_acts), dtype=torch.float32, device=device)
    n_unique_obs = len({obs_t[i, : lengths[i]].cpu().numpy().tobytes() for i in range(len(lengths))})
    print(
        f"unique trajectories in batch: {n_unique_obs}, obs std={obs_t.std().item():.4f}, "
        f"model: width={tw} layers={layers}"
    )

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

    unique_nl = list(dict.fromkeys(nlss))
    print(f"batch={len(batch)}, unique NL={len(unique_nl)}")
    for u in unique_nl:
        print(f"  - {u[:72]}")

    if args.tta_loss == "kl":
        loss_traj = KLLoss()
        loss_text = KLLoss()
    else:
        loss_traj = MultiPositiveContrastiveLoss(args.temperature)
        loss_text = MultiPositiveContrastiveLoss(args.temperature)
    _, mask_np, _ = gen_mask_from_nl(list(nlss))
    mask_t = torch.tensor(mask_np, device=device, dtype=torch.float)
    true_cls = mask_t.argmax(dim=1)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(
            obs_t, act_t, ids, masks, lengths, nl_texts=list(nlss), skip_inner_ce=True
        )
        loss = (
            loss_traj(logits, mask_t) + loss_text(logits.t(), mask_t.t())
        ) / 2
        loss.backward()
        gn = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            acc = (pred == true_cls).float().mean()
            pc = Counter(pred.cpu().tolist())
        if step % 50 == 0 or step == args.steps - 1:
            probs = F.softmax(logits, dim=1)
            ent = -(probs * probs.clamp(min=1e-8).log()).sum(1).mean()
            top2 = torch.topk(logits, k=2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).mean()
            print(
                f"step {step:3d} loss={loss.item():.4f} acc={acc.item():.3f} "
                f"ent={ent.item():.3f} margin={margin.item():.4f} grad={gn:.2e} pred_cols={dict(pc)}"
            )

    with torch.no_grad():
        logits, _ = model(
            obs_t, act_t, ids, masks, lengths, nl_texts=list(nlss), skip_inner_ce=True
        )
        pred = logits.argmax(1)
        acc = (pred == true_cls).float().mean()
        correct = int((pred == true_cls).sum().item())
    print(f"final: acc={acc.item():.3f} ({correct}/{len(true_cls)})")

    if acc.item() < 0.9:
        print(
            "FAIL tips: 1) bash generate_debug_dataset_minigrid.sh  "
            "2) sync TTCT.py  3) try --train_transformer or --full"
        )
        sys.exit(1)
    print("OK: one-batch overfit passed.")


if __name__ == "__main__":
    main()
