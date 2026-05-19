Backup of debug-optimized TTCT (2026-05-18) before paper revert.

Restore:
  cp backup_pre_paper_revert/{TTCT.py,train.py,utils.py,generate_dataset_from_paper.py,train_ttct_minigrid.sh} .
  cp backup_pre_paper_revert/debug_tta_overfit_one_batch.py scripts/

Key features in this backup:
- unique_nl + gen_mask_from_nl (B×U logits)
- traj_input_ln, padding mask, ReLU obs encoder
- optional freeze_trajectory_transformer, skip_inner_ce
- align_obs_act in train loop
