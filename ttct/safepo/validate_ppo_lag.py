#!/usr/bin/env python3
"""
Validation script for PPO-Lag (MiniGrid): load checkpoint, run episode(s), record video.
Usage:
  python safepo/validate_ppo_lag.py --checkpoint_dir ../runs/.../seed-000-2026-03-05-20-30-19.255153
  python safepo/validate_ppo_lag.py --checkpoint_dir /path/to/run --output_dir ./videos
"""

from __future__ import annotations
import argparse
import os.path as osp
import sys

# sys.path: safepo first (utils, common), then ttct_root (TTCT). ttct_root has utils.py which would shadow safepo/utils/
_script_dir = osp.dirname(osp.abspath(__file__))
_safepo_dir = _script_dir
_ttct_root = osp.dirname(_safepo_dir)
for d in (_ttct_root, _safepo_dir):
    if d in sys.path:
        sys.path.remove(d)
sys.path.insert(0, _ttct_root)
sys.path.insert(0, _safepo_dir)

import json
import glob
import os
import os.path as osp
import sys
from copy import deepcopy

import numpy as np
import torch
import loralib as lora
import gym
import gym_minigrid  # registers HazardWorld envs

from TTCT import TTCT
from common.model import ActorVCriticTrajectory
from utils.async_vector_env import CostInInfoWrapper


# Shared constants with ppo_lag
embed_dim = 512
trajectory_length = 200
context_length = 77
vocab_size = 49408
transformer_width = 512
transformer_heads = 8
transformer_layers = 12
obs_dim = 147
obs_emb_dim = 64
act_dim = 1
isaac_gym_specific_cfg = {
    "hidden_sizes": [256, 128, 128, 64],
    "threshold_Mini": 7.55,
    "threshold_Goal": 5.5,
    "cost_value": 1.0,
}


def _make_minigrid(name):
    try:
        return gym.make(name, disable_env_checker=True)
    except TypeError:
        return gym.make(name)


def _wrap_minigrid(name):
    return CostInInfoWrapper(_make_minigrid(name))


def _draw_constraint_on_frame(frame, constraint_text, cumulative_reward=None, cumulative_cost=None):
    """Draw constraint (mission) on the left, Reward and Cost on the right. Returns frame with text overlay."""
    if not hasattr(frame, "shape"):
        return frame
    frame = np.asarray(frame).astype(np.uint8).copy()
    h, w = frame.shape[:2]
    # If frame is tiny (e.g. raw obs 7x7), scale up so text fits
    min_side = 160
    if h < min_side or w < min_side:
        scale_factor = max(min_side // h, min_side // w, 2)
        frame = np.repeat(np.repeat(frame, scale_factor, axis=0), scale_factor, axis=1)
        h, w = frame.shape[:2]
    text = str(constraint_text).strip() if constraint_text else ""
    # Try cv2 first
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(0.7, w / 200.0))
        thick = max(1, int(scale * 2))
        # Left: constraint (mission)
        if text:
            words = text.split()
            lines = []
            line = ""
            for wd in words:
                if len(line) + len(wd) + 1 <= 30:
                    line = (line + " " + wd).strip() if line else wd
                else:
                    if line:
                        lines.append(line)
                    line = wd
            if line:
                lines.append(line)
            y = max(20, int(h * 0.04))
            for ln in lines[:4]:
                (tw, th), _ = cv2.getTextSize(ln, font, scale, thick)
                if y + th > h - 2:
                    break
                cv2.rectangle(frame, (0, y - th - 2), (min(w, tw + 4), y + 2), (0, 0, 0), -1)
                cv2.putText(frame, ln, (2, y), font, scale, (255, 255, 255), thick)
                y += th + 4
        # Right: Reward and Cost
        if cumulative_reward is not None or cumulative_cost is not None:
            right_lines = []
            if cumulative_reward is not None:
                right_lines.append("Reward: %.2f" % cumulative_reward)
            if cumulative_cost is not None:
                right_lines.append("Cost: %.2f" % cumulative_cost)
            x0 = w - 10
            y0 = max(20, int(h * 0.04))
            for i, ln in enumerate(right_lines):
                (tw, th), _ = cv2.getTextSize(ln, font, scale, thick)
                x = x0 - tw
                y = y0 + i * (th + 6)
                if y + th > h - 2:
                    break
                cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
                cv2.putText(frame, ln, (x, y), font, scale, (255, 255, 0) if "Reward" in ln else (255, 180, 180), thick)
        return frame
    except Exception:
        pass
    # Fallback: PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(12, min(24, w // 10)))
        except Exception:
            font = ImageFont.load_default()
        y = 5
        if text:
            for ln in (text[:80] if len(text) > 80 else text).split()[:5]:
                draw.text((5, y), ln[:40], fill=(255, 255, 255), font=font)
                y += 18
        if cumulative_reward is not None or cumulative_cost is not None:
            ry = 5
            if cumulative_reward is not None:
                draw.text((w - 120, ry), "Reward: %.2f" % cumulative_reward, fill=(255, 255, 0), font=font)
                ry += 20
            if cumulative_cost is not None:
                draw.text((w - 120, ry), "Cost: %.2f" % cumulative_cost, fill=(255, 200, 200), font=font)
        return np.array(pil_img)
    except Exception:
        return frame


def _flatten_obs(obs, obs_dim=147):
    """Convert MiniGrid obs (dict with 'image' or array) to flat vector of size obs_dim for TTCT."""
    if isinstance(obs, dict) and "image" in obs:
        arr = np.asarray(obs["image"], dtype=np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32)
    flat = arr.ravel()
    if flat.size != obs_dim:
        flat = np.pad(flat, (0, max(0, obs_dim - flat.size)), mode="constant", constant_values=0)[:obs_dim]
    return flat


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens[:-1]:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def lora_model(model, rank):
    alpha = 16
    layer_names_dict = model.state_dict().keys()
    module_list = list({".".join(key.split(".")[:-1]) for key in layer_names_dict})
    for submodule_key in module_list:
        if submodule_key.split(".")[-1] in ["query", "value"]:
            module_state_dict = model.get_submodule(submodule_key).state_dict()
            submodule = model.get_submodule(submodule_key)
            lora_layer = lora.Linear(
                submodule.in_features,
                submodule.out_features,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
            )
            lora_layer.load_state_dict(module_state_dict, strict=False)
            _set_module(model, submodule_key, lora_layer)


def load_from_save(tlmodel, name):
    if not os.path.exists(name):
        raise FileNotFoundError(f"Model file not found: {name}")
    with open(name, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    tlmodel.load_state_dict(state_dict, strict=True)


def find_latest_checkpoint(torch_save_dir, prefix):
    """Find checkpoint with highest epoch number. E.g. model50.pt -> 50."""
    pattern = osp.join(torch_save_dir, prefix + "*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    best_epoch = -1
    best_path = None
    for f in files:
        base = osp.basename(f)
        # model50.pt or Encodemodel50.pt
        stem = base.replace(prefix, "").replace(".pt", "").strip()
        try:
            epoch = int(stem) if stem else 0
        except ValueError:
            epoch = 0
        if epoch >= best_epoch:
            best_epoch = epoch
            best_path = f
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Validate PPO-Lag checkpoint, record video")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to run directory (e.g. ../runs/.../seed-000-2026-03-05-...)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save video. Default: <checkpoint_dir>/validation_videos")
    parser.add_argument("--TL_loadpath", type=str, default=None,
                        help="Override TTCT model path (from config if not set)")
    parser.add_argument("--env_name", type=str, default="MiniGrid-HazardWorld-B-v0",
                        help="Minigrid env for validation")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for env")
    parser.add_argument("--use_comet", action="store_true",
                        help="Log video and metrics to Comet ML (project ttct-training)")
    parser.add_argument("--comet_project_name", type=str, default="ttct-training",
                        help="Comet ML project name")
    args = parser.parse_args()

    checkpoint_dir = osp.abspath(args.checkpoint_dir)
    if not osp.isdir(checkpoint_dir):
        print(f"Error: checkpoint_dir does not exist: {checkpoint_dir}")
        sys.exit(1)

    config_path = osp.join(checkpoint_dir, "config.json")
    if not osp.isfile(config_path):
        print(f"Error: config.json not found in {checkpoint_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    TL_loadpath = args.TL_loadpath or config.get("TL_loadpath", "") or os.environ.get("TL_LOADPATH", "")
    if not TL_loadpath or not osp.isfile(TL_loadpath):
        # Search result/ for checkpoint_latest.pt
        for search_root in [osp.join(_ttct_root, "..", "result"), osp.join(osp.dirname(checkpoint_dir), "..", "..", "..", "result")]:
            search_root = osp.abspath(search_root)
            if osp.isdir(search_root):
                for c in glob.glob(osp.join(search_root, "**/checkpoint_latest.pt"), recursive=True):
                    TL_loadpath = c
                    break
            if TL_loadpath and osp.isfile(TL_loadpath):
                break
        if not TL_loadpath or not osp.isfile(TL_loadpath):
            print("Error: TL_loadpath (TTCT model) not found.")
            print("  Set env TL_LOADPATH or --TL_loadpath /path/to/checkpoint_latest.pt")
            sys.exit(1)

    output_dir = args.output_dir or osp.join(checkpoint_dir, "validation_videos")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    comet_exp = None
    if args.use_comet:
        # Default API key if not in env (same as run_ppo_lag.sh), so Comet works when run without .sh
        if not os.environ.get("COMET_API_KEY"):
            os.environ["COMET_API_KEY"] = "3OfuYHwcRgIwG7DzgzJ190igY"
        try:
            import comet_ml
            comet_exp = comet_ml.Experiment(project_name=args.comet_project_name)
            comet_exp.set_name("validation")
            comet_exp.log_parameter("checkpoint_dir", checkpoint_dir)
            comet_exp.log_parameter("env_name", args.env_name)
            comet_exp.log_parameter("script", "validate_ppo_lag")
            print("Comet ML: experiment 'validation' started (project=%s)" % args.comet_project_name)
        except Exception as e:
            print("Comet ML init failed: %s" % e)
            import traceback
            traceback.print_exc()
            comet_exp = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_full = isaac_gym_specific_cfg

    # Build TTCT / EncodeModel
    is_finetune = config.get("is_finetune", True)
    use_lora = config.get("use_lora", True)
    rank = config.get("rank", 8)

    EncodeModel = None
    if is_finetune:
        EncodeModel = TTCT(
            embed_dim=embed_dim,
            trajectory_length=trajectory_length,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            act_dim=act_dim,
            BERT_PATH="bert-base-uncased",
            device=device,
            obs_emb_dim=obs_emb_dim,
            obs_dim=obs_dim,
            threshold=config_full["threshold_Mini"],
            episodic_cost_value=config_full["cost_value"],
        )
        load_from_save(EncodeModel, TL_loadpath)
        EncodeModel = EncodeModel.to(device)
        if use_lora:
            lora_model(EncodeModel, rank)
            lora.mark_only_lora_as_trainable(EncodeModel)
        EncodeModel = EncodeModel.to(device)
        EncodeModel.eval()

    TLmodel = TTCT(
        embed_dim=embed_dim,
        trajectory_length=trajectory_length,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        act_dim=act_dim,
        BERT_PATH="bert-base-uncased",
        device=device,
        obs_emb_dim=obs_emb_dim,
        obs_dim=obs_dim,
        threshold=config_full["threshold_Mini"],
        episodic_cost_value=config_full["cost_value"],
    )
    load_from_save(TLmodel, TL_loadpath)
    TLmodel = TLmodel.to(device)
    TLmodel.eval()

    # Build policy
    policy = ActorVCriticTrajectory(
        obs_dim=obs_dim,
        trajectory_dim=embed_dim,
        text_dim=embed_dim,
        act_dim=7,
        hidden_sizes=config_full["hidden_sizes"],
        is_discrete=True,
    ).to(device)

    torch_save_dir = osp.join(checkpoint_dir, "torch_save")
    model_path = find_latest_checkpoint(torch_save_dir, "model")
    enc_path = find_latest_checkpoint(torch_save_dir, "Encodemodel")

    if not model_path:
        print(f"Error: no model*.pt found in {torch_save_dir}")
        sys.exit(1)

    # Infer act_dim from checkpoint (may differ from default MiniGrid=7, e.g. SafetyGym=4)
    ckpt_sd = torch.load(model_path, map_location=device)
    if "log_std" in ckpt_sd:
        act_dim_ckpt = int(ckpt_sd["log_std"].shape[0])
    else:
        for k in ("mean.4.bias", "mean.4.weight"):
            if k in ckpt_sd:
                act_dim_ckpt = int(ckpt_sd[k].shape[0])
                break
        else:
            act_dim_ckpt = 7
    print(f"Inferred act_dim={act_dim_ckpt} from checkpoint")

    policy = ActorVCriticTrajectory(
        obs_dim=obs_dim,
        trajectory_dim=embed_dim,
        text_dim=embed_dim,
        act_dim=act_dim_ckpt,
        hidden_sizes=config_full["hidden_sizes"],
        is_discrete=True,
    ).to(device)

    policy.actor.load_state_dict(ckpt_sd, strict=True)
    policy.eval()

    if is_finetune and enc_path:
        lora_sd = torch.load(enc_path, map_location=device)
        policy_enc = EncodeModel
        policy_enc.load_state_dict(lora_sd, strict=False)

    # Create env
    video_env = _wrap_minigrid(args.env_name)
    video_env.seed(args.seed)

    try:
        import imageio
    except ImportError:
        print("Error: imageio required. pip install imageio imageio-ffmpeg")
        sys.exit(1)

    all_metrics = []
    for ep in range(args.num_episodes):
        reset_out = video_env.reset()
        if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 2:
            vid_obs, vid_info = reset_out[0], reset_out[1]
        else:
            vid_obs, vid_info = reset_out, {}
        vid_mission = [vid_info.get("mission", "") if isinstance(vid_info, dict) else ""]
        current_mission = vid_mission[0] if vid_mission else ""
        vid_obslist = [[_flatten_obs(vid_obs, obs_dim)]]
        vid_actlist = [[-1]]
        vid_lengths = [1]

        with torch.no_grad():
            emb_m = TLmodel.test_encode_text(vid_mission)
            if is_finetune:
                finetune_m = EncodeModel.test_encode_text(vid_mission)

        frames = []
        vid_done = False
        vid_rew, vid_cost, vid_len = 0.0, 0.0, 0
        # TTCT.test_encode expects len(obs) <= trajectory_length; we start from 1 and add one per step,
        # so use at most trajectory_length - 1 steps to keep length <= trajectory_length.
        max_vid_steps = max(1, trajectory_length - 1)

        while not vid_done and vid_len < max_vid_steps:
            # TTCT.test_encode pads to max(lengths); pass last cap steps and set lengths to actual lengths to avoid negative padding
            cap = trajectory_length
            obs_capped = [vid_obslist[0][-cap:]]
            act_capped = [vid_actlist[0][-cap:]]
            len_capped = [len(obs_capped[0])]
            if is_finetune:
                obsw = EncodeModel.test_encode(obs_capped, act_capped, len_capped, finetune_m)
            else:
                obsw = TLmodel.test_encode(obs_capped, act_capped, len_capped, emb_m)
            obsw = torch.as_tensor(obsw, dtype=torch.float32, device=device)
            if obsw.dim() == 1:
                obsw = obsw.unsqueeze(0)
            act, _, _, _ = policy.step(obsw, deterministic=True)
            action = act.detach().squeeze().cpu().numpy()
            action_val = int(np.asarray(action).flat[0]) if np.asarray(action).size else 0
            out = video_env.step(action_val)
            if len(out) == 5:
                next_obs, reward, term, trunc, info = out
                cost = info.get("cost", 0.0) if isinstance(info, dict) else 0.0
            else:
                next_obs, reward, cost, term, trunc, info = out
            if isinstance(info, dict) and info.get("mission"):
                current_mission = info["mission"]
            vid_rew += float(reward)
            vid_cost += float(cost)
            vid_len += 1
            try:
                frame = video_env.render(mode="rgb_array")
                if frame is None:
                    frame = video_env.render()
                if frame is not None:
                    if not hasattr(frame, "shape") and hasattr(frame, "size"):
                        frame = np.array(frame)
                    if hasattr(frame, "shape") and len(frame.shape) >= 2:
                        frame = np.asarray(frame).astype(np.uint8)
                        frame = _draw_constraint_on_frame(
                            frame, current_mission,
                            cumulative_reward=vid_rew, cumulative_cost=vid_cost,
                        )
                        frames.append(frame)
            except Exception:
                pass
            vid_done = term or trunc
            if not vid_done:
                vid_obslist[0].append(_flatten_obs(next_obs, obs_dim))
                vid_actlist[0].append(action_val)
                vid_lengths[0] += 1

        video_env.close()

        metrics = {"EpRet": vid_rew, "EpCost": vid_cost, "EpLen": vid_len}
        all_metrics.append(metrics)
        print(f"Episode {ep + 1}/{args.num_episodes}: reward={vid_rew:.2f}, cost={vid_cost:.2f}, len={vid_len}")

        if frames:
            video_path = osp.join(output_dir, f"trajectory_ep{ep + 1}.mp4")
            imageio.mimsave(video_path, frames, fps=10, codec="libx264")
            print(f"  Video saved: {video_path}")
            if comet_exp is not None:
                try:
                    comet_exp.log_video(
                        file=video_path,
                        name="validation_trajectory_ep%d" % (ep + 1),
                        step=ep + 1,
                        format="mp4",
                    )
                    comet_exp.log_metrics(
                        {"Validation/EpRet": vid_rew, "Validation/EpCost": vid_cost, "Validation/EpLen": vid_len},
                        step=ep + 1,
                    )
                    print("  Video and metrics sent to Comet ML")
                except Exception as e:
                    print("  Comet log_video failed: %s" % e)
                    import traceback
                    traceback.print_exc()
        else:
            print("  Warning: no frames collected (render may not support rgb_array)")

    if all_metrics:
        avg_r = np.mean([m["EpRet"] for m in all_metrics])
        avg_c = np.mean([m["EpCost"] for m in all_metrics])
        avg_l = np.mean([m["EpLen"] for m in all_metrics])
        print(f"\nValidation summary: EpRet={avg_r:.2f}, EpCost={avg_c:.2f}, EpLen={avg_l:.1f}")
    if comet_exp is not None:
        try:
            comet_exp.end()
            print("Comet ML experiment ended.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
