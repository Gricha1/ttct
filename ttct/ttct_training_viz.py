"""
Validation visualizations during TTCT training: trajectory frames + NL constraint +
ground-truth violation vs cost-model prediction.
"""
from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

_CONSTRAINT_TEXT_CACHE: Dict[str, Dict[str, dict]] = {}


def _constraint_pool_map(pool: str = "paper_full") -> Dict[str, dict]:
    if pool not in _CONSTRAINT_TEXT_CACHE:
        from generate_dataset_from_paper import generate_all_possible_constraints

        items = generate_all_possible_constraints(pool=pool)
        _CONSTRAINT_TEXT_CACHE[pool] = {c["text"]: c for c in items}
    return _CONSTRAINT_TEXT_CACHE[pool]


def compute_true_step_costs(
    obs: np.ndarray,
    act: np.ndarray,
    length: int,
    nl: str,
    *,
    constraint_pool: str = "paper_full",
) -> np.ndarray:
    """
    Oracle step cost: 1.0 if trajectory prefix [0:t] violates the constraint, else 0.0.
    Matches episodic HazardWorld cost (violation -> cost=1) at trajectory level.
    """
    from generate_dataset_from_paper import check_violation

    length = int(length)
    costs = np.zeros(length, dtype=np.float32)
    pool = _constraint_pool_map(constraint_pool)
    constraint = pool.get(nl, {"text": nl, "type": "quantitative", "params": {}})
    for t in range(1, length + 1):
        traj = {
            "obs": [np.asarray(o) for o in obs[:t]],
            "act": [int(a) for a in act[:t]],
            "length": t,
        }
        if check_violation(traj, constraint):
            costs[t - 1 :] = 1.0
            break
    return costs


def ground_truth_violated(
    obs: np.ndarray,
    act: np.ndarray,
    length: int,
    nl: str,
    *,
    constraint_pool: str = "paper_full",
) -> bool:
    from generate_dataset_from_paper import check_violation

    length = int(length)
    traj = {
        "obs": [np.asarray(o) for o in obs[:length]],
        "act": [int(a) for a in act[:length]],
        "length": length,
    }
    pool = _constraint_pool_map(constraint_pool)
    constraint = pool.get(nl, {"text": nl, "type": "quantitative", "params": {}})
    return bool(check_violation(traj, constraint))


def is_minigrid_object_grid(frame: np.ndarray) -> bool:
    """True if obs looks like HazardWorld/MiniGrid (type,color,state) integer grid."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return False
    if arr.dtype.kind in ("i", "u"):
        arr_i = arr.astype(np.int32)
    else:
        if not np.allclose(arr, np.rint(arr), atol=0.05):
            return False
        arr_i = np.rint(arr).astype(np.int32)
    if arr_i[:, :, 0].min() < 0 or arr_i[:, :, 0].max() > 12:
        return False
    if arr_i[:, :, 1].min() < 0 or arr_i[:, :, 1].max() > 10:
        return False
    return True


# MiniGrid OBJECT_TO_IDX -> RGB (HazardWorld partial view)
_TYPE_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (40, 40, 40),    # unseen
    1: (50, 50, 50),    # empty
    2: (100, 100, 100),  # wall
    3: (220, 220, 220),  # floor
    4: (160, 120, 60),   # door
    5: (255, 255, 0),    # key
    6: (255, 80, 80),    # ball
    7: (255, 165, 0),    # box
    8: (0, 220, 0),      # goal
    9: (255, 40, 40),    # lava
    10: (255, 230, 0),   # agent
    11: (60, 190, 60),   # grass
    12: (60, 120, 255),  # water
}


_GRID_CLASS = None


def _import_fork_grid():
    """Load forked minigrid once; avoid purge/re-register on every val frame."""
    global _GRID_CLASS
    if _GRID_CLASS is not None:
        return _GRID_CLASS
    import ensure_safepo_paths

    ensure_safepo_paths.ensure_hazardworld_env()
    from gym_minigrid.minigrid import Grid

    _GRID_CLASS = Grid
    return Grid


def _sanitize_minigrid_obs(arr: np.ndarray) -> np.ndarray:
    """Clip type/color/state so Grid.decode / colormap never assert."""
    out = np.rint(np.asarray(arr)).astype(np.int32)
    out[..., 0] = np.clip(out[..., 0], 0, 12)
    out[..., 1] = np.clip(out[..., 1], 0, 6)
    out[..., 2] = np.clip(out[..., 2], -3, 2)
    return out


def _infer_agent_dir(actions: np.ndarray, t: int, *, initial_dir: int = 3) -> int:
    """Integrate turn-left/right along the trajectory prefix (MiniGrid actions 0/1)."""
    d = int(initial_dir) % 4
    for a in actions[: int(t) + 1]:
        a = int(a)
        if a == 0:
            d = (d - 1) % 4
        elif a == 1:
            d = (d + 1) % 4
    return d


def _agent_pose_from_obs(arr: np.ndarray) -> Tuple[int, int, int]:
    """
  Agent in encoded partial obs sits on bottom-center empty cell (MiniGrid egocentric view).
  If type==agent (10) is present, use that cell and state channel as direction hint.
    """
    h, w = arr.shape[0], arr.shape[1]
    agent_j, agent_i = w // 2, h - 1
    agent_dir = 3
    where = np.argwhere(arr[:, :, 0] == 10)
    if where.size:
        agent_i, agent_j = int(where[0, 0]), int(where[0, 1])
        st = int(arr[agent_i, agent_j, 2])
        if 0 <= st <= 3:
            agent_dir = st
    return agent_i, agent_j, agent_dir


def _prepare_obs_for_grid_decode(arr: np.ndarray) -> np.ndarray:
    """Grid.decode does not support type 'agent'; map agent cells to empty."""
    out = np.array(arr, copy=True)
    out[out[:, :, 0] == 10] = np.array([1, 0, 0], dtype=out.dtype)
    return out


def _point_in_triangle_mask(
    xf: np.ndarray, yf: np.ndarray, a, b, c
) -> np.ndarray:
    """Barycentric inside-test; xf,yf in [0,1] normalized coords."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v0 = c - a
    v1 = b - a
    v2 = np.stack([xf - a[0], yf - a[1]], axis=-1)
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = (v2[..., 0] * v0[0] + v2[..., 1] * v0[1])
    dot11 = np.dot(v1, v1)
    dot12 = (v2[..., 0] * v1[0] + v2[..., 1] * v1[1])
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.zeros(xf.shape, dtype=bool)
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0) & (v >= 0) & (u + v < 1)


def _draw_agent_triangle_on_tile(tile: np.ndarray, agent_dir: int) -> None:
    """Red agent triangle (numpy only; no gym_minigrid.rendering / OpenGL)."""
    import math

    h, w = tile.shape[0], tile.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    xf = (xx.astype(np.float64) + 0.5) / max(w, 1)
    yf = (yy.astype(np.float64) + 0.5) / max(h, 1)
    tri = ((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
    theta = 0.5 * math.pi * (int(agent_dir) % 4)
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)

    def _rot(px: float, py: float) -> Tuple[float, float]:
        x, y = px - 0.5, py - 0.5
        return 0.5 + x * cos_t - y * sin_t, 0.5 + x * sin_t + y * cos_t

    a = _rot(*tri[0])
    b = _rot(*tri[1])
    c = _rot(*tri[2])
    mask = _point_in_triangle_mask(xf, yf, a, b, c)
    tile[mask] = np.array([255, 0, 0], dtype=tile.dtype)


def _minigrid_obs_to_rgb_simple(
    frame: np.ndarray, tile_size: int = 16, agent_dir: int = 3
) -> np.ndarray:
    """Colormap by object type + red agent triangle (fallback renderer)."""
    arr = _sanitize_minigrid_obs(frame)
    h, w = arr.shape[0], arr.shape[1]
    agent_i, agent_j, dir_hint = _agent_pose_from_obs(arr)
    agent_dir = int(agent_dir) if agent_dir is not None else dir_hint

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            t = int(arr[i, j, 0])
            if t == 10:
                t = 1
            rgb[i, j] = _TYPE_RGB.get(t, (180, 0, 180))

    big = np.repeat(np.repeat(rgb, tile_size, axis=0), tile_size, axis=1)
    y0, y1 = agent_i * tile_size, (agent_i + 1) * tile_size
    x0, x1 = agent_j * tile_size, (agent_j + 1) * tile_size
    tile = big[y0:y1, x0:x1]
    _draw_agent_triangle_on_tile(tile, agent_dir)
    big[y0:y1, x0:x1] = tile
    return big


def _fig_to_rgb(fig) -> np.ndarray:
    """Matplotlib 3.8+ removed tostring_rgb; use buffer_rgba when available."""
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba())
        rgb = buf[:, :, :3].copy()
        return rgb
    if hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(int(h), int(w), 3).copy()
    raise RuntimeError("Cannot export matplotlib figure to RGB array")


def _float_obs_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Fallback when obs are not a MiniGrid grid (e.g. synthetic randn dataset)."""
    x = np.asarray(frame, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (H,W,C), got {x.shape}")
    rgb = np.zeros(x.shape, dtype=np.float32)
    for c in range(3):
        ch = x[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        if hi > lo + 1e-8:
            rgb[:, :, c] = (ch - lo) / (hi - lo)
        else:
            rgb[:, :, c] = 0.5
    return (rgb * 255.0).astype(np.uint8)


def _minigrid_obs_to_rgb(
    frame: np.ndarray, tile_size: int = 16, agent_dir: int = 3
) -> np.ndarray:
    """
    MiniGrid obs are (H,W,3) object grids (type, color, state), not RGB.
    Decode + render via forked gym_minigrid Grid.render() with agent triangle.
    """
    Grid = _import_fork_grid()

    arr = _sanitize_minigrid_obs(frame)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3) MiniGrid grid, got {arr.shape}")
    agent_i, agent_j, dir_hint = _agent_pose_from_obs(arr)
    agent_dir = int(agent_dir) if agent_dir is not None else dir_hint
    arr_dec = _prepare_obs_for_grid_decode(arr)

    grid, vis_mask = Grid.decode(arr_dec)
    agent_pos = (agent_j, agent_i)
    img = grid.render(
        tile_size=tile_size,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        highlight_mask=np.zeros_like(vis_mask, dtype=bool),
    )
    out = np.asarray(img, dtype=np.uint8)
    if out.size == 0 or out.max() == 0:
        raise RuntimeError("Grid.render returned empty image")
    return out


def _obs_to_rgb(
    frame: np.ndarray, *, prefer_simple: bool = False, agent_dir: int = 3
) -> Tuple[np.ndarray, str]:
    """
    RGB uint8 for display.
    Returns (image, mode) where mode is 'grid' | 'grid-simple' | 'heatmap' | 'rgb'.
    """
    x = np.asarray(frame)
    if x.ndim == 3 and x.shape[-1] == 3 and is_minigrid_object_grid(x):
        # Default: colormap + numpy triangle (stable in Docker/headless).
        if prefer_simple:
            return _minigrid_obs_to_rgb_simple(x, agent_dir=agent_dir), "grid-simple"
        try:
            return _minigrid_obs_to_rgb(x, agent_dir=agent_dir), "grid"
        except Exception:
            return _minigrid_obs_to_rgb_simple(x, agent_dir=agent_dir), "grid-simple"
    if x.ndim == 3 and x.shape[-1] == 3:
        return _float_obs_to_rgb(x), "heatmap"
    x = np.asarray(frame, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected (H,W,C) frame, got shape {x.shape}")
    if x.max() <= 1.0 + 1e-6:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8), "rgb"


def dataset_uses_real_minigrid_obs(dataset: Sequence, n_check: int = 8) -> bool:
    """Sample dataset; False if obs look like synthetic randn (env failed at gen time)."""
    n = min(n_check, len(dataset))
    if n == 0:
        return False
    ok = 0
    for i in range(n):
        obs, *_ = dataset[i]
        if is_minigrid_object_grid(np.asarray(obs[0])):
            ok += 1
    return ok >= max(1, n // 2)


def _subsample_indices(length: int, n_frames: int) -> List[int]:
    length = max(1, int(length))
    n_frames = max(1, int(n_frames))
    if length <= n_frames:
        return list(range(length))
    return [int(round(i * (length - 1) / (n_frames - 1))) for i in range(n_frames)]


@torch.no_grad()
def predict_cost_details(
    model: torch.nn.Module,
    tokenizer: Any,
    obs: np.ndarray,
    act: np.ndarray,
    length: int,
    nl: str,
    *,
    context_length: int = 77,
    is_predict_cost: bool = True,
    constraint_pool: str = "paper_full",
) -> Dict[str, float]:
    """Per-step predicted + oracle true costs; violation if max(step_cost) > 0."""
    length = int(length)
    obs_list = obs[:length]
    act_list = act[:length]
    true_step_costs = compute_true_step_costs(
        obs, act, length, nl, constraint_pool=constraint_pool
    )
    text_feat = model.test_encode_text([nl])
    episodic = model.episodic_cost_layer(text_feat.detach()).reshape(-1)
    episodic_val = float(episodic[0].item()) if episodic.numel() else float("nan")

    # Val viz: regression sigmoid only (no cosine>threshold -> hard 1.0 spikes).
    out = model.get_cost(
        [obs_list], [act_list], text_feat, is_predict_cost, apply_cosine_threshold=False
    )
    step_costs = out.detach().float().cpu().numpy().reshape(-1)
    step_costs = step_costs[:length] if step_costs.size >= length else step_costs

    pred_max = float(np.max(step_costs)) if step_costs.size else 0.0
    pred_min = float(np.min(step_costs)) if step_costs.size else 0.0
    pred_mean = float(np.mean(step_costs)) if step_costs.size else 0.0
    pred_sum = float(np.sum(np.maximum(step_costs, 0.0))) if step_costs.size else 0.0
    pred_violated = pred_max > 0.5 or float(episodic_val) > 0.5
    true_max = float(np.max(true_step_costs)) if true_step_costs.size else 0.0
    true_mean = float(np.mean(true_step_costs)) if true_step_costs.size else 0.0

    return {
        "episodic_head": episodic_val,
        "step_cost_max": pred_max,
        "step_cost_min": pred_min,
        "step_cost_mean": pred_mean,
        "step_cost_sum_pos": pred_sum,
        "pred_violated": pred_violated,
        "step_costs": step_costs,
        "true_step_costs": true_step_costs,
        "true_step_max": true_max,
        "true_step_mean": true_mean,
        "true_violated": true_max > 0.0,
    }


def _aggregate_cost_metrics(preds: List[Dict[str, Any]]) -> Dict[str, float]:
    """Pool step costs and episodic heads from predict_cost_details runs."""
    if not preds:
        return {}
    all_steps = np.concatenate(
        [np.asarray(p["step_costs"], dtype=np.float32).reshape(-1) for p in preds if p.get("step_costs") is not None]
    )
    episodic = np.asarray([p["episodic_head"] for p in preds], dtype=np.float32)
    out: Dict[str, float] = {
        "cost_episodic_mean": float(np.mean(episodic)),
        "cost_episodic_max": float(np.max(episodic)),
        "cost_episodic_min": float(np.min(episodic)),
    }
    if all_steps.size:
        out.update(
            {
                "cost_step_mean": float(np.mean(all_steps)),
                "cost_step_max": float(np.max(all_steps)),
                "cost_step_min": float(np.min(all_steps)),
            }
        )
    return out


def _log_cost_metrics(
    metrics: Dict[str, float],
    *,
    writer: Any,
    comet_experiment: Any,
    global_step: int,
    prefix: str = "val_viz",
) -> None:
    for key, val in metrics.items():
        if writer is not None:
            writer.add_scalar(f"{prefix}/{key}", val, global_step)
        if comet_experiment is not None:
            comet_experiment.log_metric(f"{prefix}/{key}", val, step=global_step)


def _ensure_cost_defaults(model: torch.nn.Module, threshold: float, episodic_cost_value: float) -> None:
    if getattr(model, "threshold", None) is None:
        model.threshold = threshold
    if getattr(model, "episodic_cost_value", None) is None:
        model.episodic_cost_value = episodic_cost_value


def select_balanced_samples(
    dataset: Sequence,
    n_violated: int,
    n_safe: int,
    *,
    constraint_pool: str = "paper_full",
    seed: int = 0,
) -> List[Tuple[int, bool]]:
    """Returns list of (index, gt_violated) with up to n_violated + n_safe items."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    picked_v: List[Tuple[int, bool]] = []
    picked_s: List[Tuple[int, bool]] = []

    n_skip_errors = 0
    for idx in indices:
        obs, act, _tls, length, nl = dataset[int(idx)]
        try:
            gt = ground_truth_violated(obs, act, length, nl, constraint_pool=constraint_pool)
        except Exception:
            n_skip_errors += 1
            continue
        if gt and len(picked_v) < n_violated:
            picked_v.append((int(idx), True))
        elif not gt and len(picked_s) < n_safe:
            picked_s.append((int(idx), False))
        if len(picked_v) >= n_violated and len(picked_s) >= n_safe:
            break

    out = picked_v + picked_s
    if not out and n_skip_errors > 0:
        raise RuntimeError(
            f"select_balanced_samples: all {n_skip_errors} candidates failed GT check "
            f"(import generate_dataset_from_paper / constraint pool?)"
        )
    return out


def render_validation_figure(
    obs: np.ndarray,
    act: np.ndarray,
    length: int,
    nl: str,
    gt_violated: bool,
    pred: Dict[str, Any],
    *,
    n_frames: int = 8,
    title: str = "",
) -> np.ndarray:
    """Returns RGB image (H,W,3) uint8 for logging."""
    length = int(length)
    frame_ids = _subsample_indices(length, n_frames)
    n_cols = len(frame_ids)

    fig = plt.figure(figsize=(1.9 * n_cols, 7.5), dpi=110)
    gs = GridSpec(
        3,
        n_cols,
        figure=fig,
        height_ratios=[2.4, 1.1, 1.3],
        hspace=0.45,
        wspace=0.12,
        top=0.96,
        bottom=0.06,
        left=0.04,
        right=0.98,
    )

    render_mode_note = pred.get("_render_mode_note", "")

    for j, t in enumerate(frame_ids):
        ax = fig.add_subplot(gs[0, j])
        try:
            agent_dir = _infer_agent_dir(act, t)
            rgb, mode = _obs_to_rgb(obs[t], prefer_simple=True, agent_dir=agent_dir)
            ax.imshow(rgb, interpolation="nearest")
            suffix = "" if mode == "grid" else f" [{mode}]"
            ax.set_title(f"t={t}{suffix}", fontsize=8)
        except Exception as exc:
            ax.imshow(np.zeros((32, 32, 3), dtype=np.uint8))
            ax.set_title(f"t={t} (err)", fontsize=8)
            ax.text(0.5, 0.5, str(exc)[:50], ha="center", va="center", fontsize=5, color="red")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_cost = fig.add_subplot(gs[1, :])
    step_costs = np.asarray(pred.get("step_costs", []), dtype=np.float32).reshape(-1)[:length]
    true_costs = np.asarray(pred.get("true_step_costs", []), dtype=np.float32).reshape(-1)[:length]
    xs = np.arange(max(step_costs.size, true_costs.size, 1))
    if step_costs.size:
        ax_cost.plot(
            np.arange(step_costs.size),
            step_costs,
            color="#c0392b",
            linewidth=1.5,
            marker=".",
            markersize=3,
            label="predicted (regression)",
        )
    if true_costs.size:
        ax_cost.plot(
            np.arange(true_costs.size),
            true_costs,
            color="#2980b9",
            linewidth=1.8,
            drawstyle="steps-post",
            label="true (oracle)",
        )
    ax_cost.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    y_vals = []
    if step_costs.size:
        y_vals.extend(step_costs.tolist())
    if true_costs.size:
        y_vals.extend(true_costs.tolist())
    if y_vals:
        y_hi, y_lo = float(max(y_vals)), float(min(y_vals))
        pad = max(0.05, 0.08 * (y_hi - y_lo + 1e-6))
        ax_cost.set_ylim(y_lo - pad, y_hi + pad)
    ax_cost.set_xlim(0, max(length - 1, 1))
    ax_cost.set_xlabel("step", fontsize=9)
    ax_cost.set_ylabel("cost", fontsize=9)
    ax_cost.legend(loc="upper right", fontsize=8)
    ax_cost.grid(True, alpha=0.25)

    gt_s = "VIOLATION" if gt_violated else "SAFE"
    pr_s = "VIOLATION" if pred.get("pred_violated") else "SAFE"
    match = "OK" if bool(gt_violated) == bool(pred.get("pred_violated")) else "MISMATCH"
    wrapped = "\n".join(textwrap.wrap(str(nl), width=90))
    info_lines = [
        title,
        render_mode_note,
        "",
        "Constraint:",
        wrapped,
        "",
        f"Ground truth: {gt_s}",
        f"Predicted:    {pr_s}  ({match})",
        (
            f"episodic_pred={pred.get('episodic_head', float('nan')):.4f}   "
            f"pred min/mean/max="
            f"{pred.get('step_cost_min', 0):.4f}/"
            f"{pred.get('step_cost_mean', 0):.4f}/"
            f"{pred.get('step_cost_max', 0):.4f}   "
            f"true mean/max="
            f"{pred.get('true_step_mean', 0):.4f}/"
            f"{pred.get('true_step_max', 0):.4f}"
        ),
    ]
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis("off")
    ax_info.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        linespacing=1.35,
    )

    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def run_epoch_validation_viz(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Sequence,
    *,
    out_dir: str,
    epoch: int,
    tag_prefix: Optional[str] = None,
    device: torch.device,
    context_length: int = 77,
    n_violated: int = 2,
    n_safe: int = 2,
    n_frames: int = 8,
    constraint_pool: str = "paper_full",
    threshold: float = 5.5,
    episodic_cost_value: float = 1.0,
    is_predict_cost: bool = True,
    seed: int = 0,
    writer: Any = None,
    comet_experiment: Any = None,
    global_step: int = 0,
) -> Dict[str, float]:
    """
    Save PNG panels and optional TensorBoard / Comet images.
    Returns summary metrics (accuracy on picked samples).
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    _ensure_cost_defaults(model, threshold, episodic_cost_value)

    picks = select_balanced_samples(
        dataset,
        n_violated,
        n_safe,
        constraint_pool=constraint_pool,
        seed=seed + epoch,
    )
    file_tag = tag_prefix if tag_prefix is not None else f"epoch{epoch:03d}"

    render_mode_note = ""
    if not dataset_uses_real_minigrid_obs(dataset):
        render_mode_note = (
            "WARN: dataset obs are NOT MiniGrid grids (likely synthetic randn). "
            "Regenerate: pip install -e . && bash generate_dataset_minigrid.sh"
        )
        import warnings

        warnings.warn(render_mode_note)

    if not picks:
        return {"val_viz_n": 0.0, "val_viz_acc": float("nan"), "val_viz_dir": out_dir, "val_viz_tag": file_tag}

    correct = 0
    saved_paths: List[str] = []
    pred_records: List[Dict[str, Any]] = []
    for k, (idx, gt) in enumerate(picks):
        obs, act, _tls, length, nl = dataset[idx]
        pred = predict_cost_details(
            model,
            tokenizer,
            obs,
            act,
            length,
            nl,
            context_length=context_length,
            is_predict_cost=is_predict_cost,
            constraint_pool=constraint_pool,
        )
        pred["_render_mode_note"] = render_mode_note
        pred_records.append(pred)
        pr = bool(pred["pred_violated"])
        if pr == gt:
            correct += 1

        tag = "violated" if gt else "safe"
        title = f"{file_tag} sample={k} idx={idx} ({tag})"
        img = render_validation_figure(
            obs, act, length, nl, gt, pred, n_frames=n_frames, title=title
        )

        fname = os.path.join(out_dir, f"{file_tag}_{tag}_{k:02d}.png")
        plt.imsave(fname, img)
        saved_paths.append(fname)

        if writer is not None:
            writer.add_image(
                f"val_viz/{file_tag}/{tag}_{k}",
                img.transpose(2, 0, 1),
                global_step,
            )

        if comet_experiment is not None:
            try:
                comet_experiment.log_image(
                    img,
                    name=f"val_viz/{file_tag}_{tag}_{k}",
                    step=global_step,
                )
            except Exception as exc:
                import warnings

                warnings.warn(f"Comet log_image failed: {exc}")

    acc = correct / len(picks)
    cost_metrics = _aggregate_cost_metrics(pred_records)
    if cost_metrics:
        _log_cost_metrics(
            cost_metrics,
            writer=writer,
            comet_experiment=comet_experiment,
            global_step=global_step,
        )

    return {
        "val_viz_n": float(len(picks)),
        "val_viz_acc": float(acc),
        "val_viz_dir": out_dir,
        "val_viz_tag": file_tag,
        "val_viz_paths": saved_paths,
        **cost_metrics,
    }
