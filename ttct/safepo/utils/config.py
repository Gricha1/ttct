# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
# Config for PPO-Lag / TTCT single-agent training.
# Adds Comet ML options: --use_comet, --comet_project_name, --comet_workspace.

import argparse
import os

isaac_gym_map = {}  # MiniGrid and SafetyGym are not in Isaac Gym


def single_agent_args():
    parser = argparse.ArgumentParser(description="PPO-Lag / TTCT single-agent")
    # Environment
    parser.add_argument("--task", type=str, default="MiniGrid")
    parser.add_argument("--experiment", type=str, default="single_agent_exp")
    parser.add_argument("--is_lava", action="store_true")
    parser.add_argument("--num_envs", type=int, default=3, help="Fewer envs = less GPU memory (trajectory batch)")
    # Craftax/CrafText (when --task Craftax)
    parser.add_argument("--craftext_settings", type=str, default="achievements_safe_budget_energy")
    parser.add_argument("--craftext_env_name", type=str, default="Craftax-Classic-Pixels-v1-Text")
    parser.add_argument("--constraint_text", type=str, default="You must maintain your energy level at or above 8.")
    # Training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=16384)
    parser.add_argument("--total_steps", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=128, help="PPO minibatch size")
    parser.add_argument("--cost_limit", type=float, default=0.5)
    parser.add_argument("--lagrangian_multiplier_init", type=float, default=0.1)
    parser.add_argument("--lagrangian_multiplier_lr", type=float, default=0.035)
    # MiniGrid HazardWorld: some tasks terminate the episode immediately when cost becomes 1.
    # This switch allows continuing the episode after the first cost violation.
    parser.add_argument(
        "--ignore_cost_termination",
        action="store_true",
        default=False,
        help="If set, override env terminated signal when cost==1 (MiniGrid HazardWorld).",
    )
    # TTCT / cost
    parser.add_argument("--use_predict_cost", action="store_true", default=True)
    parser.add_argument("--use_credit_assignment", action="store_true", default=True)
    parser.add_argument("--TL_loadpath", type=str, default="")
    parser.add_argument("--language_model", type=str, default="TLmodel")
    parser.add_argument("--use_pretrained_encoders", type=str, default="true")
    parser.add_argument("--is_finetune", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--rank", type=int, default=8)
    # Logging / run
    parser.add_argument("--log_dir", type=str, default="../runs")
    parser.add_argument("--write_terminal", action="store_true", default=True)
    parser.add_argument("--use_eval", action="store_true", default=False)
    # Comet ML
    parser.add_argument("--use_comet", action="store_true", default=False, help="Enable Comet ML logging")
    parser.add_argument("--comet_project_name", type=str, default="ttct-training", help="Comet ML project name")
    parser.add_argument("--comet_workspace", type=str, default=None, help="Comet ML workspace (optional)")
    parser.add_argument(
        "--comet_experiment_name",
        type=str,
        default="ppo lag minigrid",
        help="Comet ML experiment name (optional).",
    )
    # Device / misc
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--randomize", action="store_true", default=False)

    args = parser.parse_args()
    if args.use_pretrained_encoders in ("true", "1", "yes"):
        args.use_pretrained_encoders = True
    elif args.use_pretrained_encoders in ("false", "0", "no"):
        args.use_pretrained_encoders = False
    return args, None
