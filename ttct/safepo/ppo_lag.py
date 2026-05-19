# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
import os

# Transformers (via TTCT) can import JAX before Craftax env inits; set platform before any of that loads.
# Default: JAX on GPU. Set CRAFTEXT_JAX_CPU=1 to force CPU if PyTorch+JAX on one GPU misbehaves.
_cjx = os.environ.get("CRAFTEXT_JAX_CPU", "0").strip().lower()
if _cjx not in ("0", "false", "no", "off", "gpu", ""):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

from copy import deepcopy
from tqdm import tqdm
import datetime
from torch.distributions import Categorical
from torch.distributions import Normal
import random
import sys
import time
from collections import deque
from TTCT import TTCT
from craftext_pixel_encoder import CraftextPixelEncoder
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import gym
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import loralib as lora
import sys
from common.buffer import VectorizedOnPolicyBuffer
from common.lagrange import Lagrange
from common.logger import EpochLogger, convert_json
from common.model import ActorVCriticTrajectory
from utils.config import single_agent_args, isaac_gym_map
from utils.util import BufferDataset
from utils.async_vector_env import AsyncVectorEnv, CostInInfoWrapper, IgnoreCostTerminationWrapper
from utils.craftext_gym_env import CraftaxCMDPGymEnv
from utils.simple_vector_env import SimpleVectorEnv
import gym_minigrid
import safety_gymnasium



default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [256, 128, 128, 64],
    'r_gamma': 0.95,
    'c_gamma': 0.95,
    'threshold_Mini':7.55,
    'threshold_Goal':5.5,
    'cost_value':1.0,
    'batch_size':512,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'learning_rate':3e-4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
}


def _copy_rollout_obs(x):
    """Avoid deepcopy on large pixel frames (63×63×3); keeps rollout Python overhead low."""
    if isinstance(x, np.ndarray):
        # np.asarray(..., copy=) requires NumPy 2+; np.array(..., copy=True) works on 1.x/2.x
        return np.array(x, dtype=np.float32, copy=True)
    return deepcopy(x)


def _slice_for_ttct_encode(obslist, actlist, lengths, max_len: int):
    """
    Only the last max_len frames go into TTCT.test_encode (CNN+transformer scale with T).
    Full obslist/actlist stay intact for buffer storage and get_cost.
    max_len <= 0: no slicing (full history each encode).
    """
    if max_len <= 0:
        return obslist, actlist, lengths
    out_o, out_a, out_l = [], [], []
    for i in range(len(obslist)):
        lo = obslist[i]
        la = actlist[i]
        n = len(lo)
        if n <= max_len:
            out_o.append(lo)
            out_a.append(la)
            out_l.append(int(lengths[i]) if i < len(lengths) else n)
        else:
            out_o.append(lo[-max_len:])
            out_a.append(la[-max_len:])
            out_l.append(max_len)
    return out_o, out_a, out_l


def _buffer_histories(obslist, actlist):
    """Light copies for buffer.store (was deepcopy of nested lists every env-step)."""
    oo = [[_copy_rollout_obs(f) for f in traj[:-1]] for traj in obslist]
    aa = [list(traj[:-1]) for traj in actlist]
    return oo, aa


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def lora_model(model,rank):
    rank=rank
    alpha = 16
    layer_names_dict = model.state_dict().keys()
    module_list = []
    for key in layer_names_dict:
        module_list.append('.'.join(key.split('.')[:-1]))
    for submodule_key in module_list:
        if submodule_key.split('.')[-1] in ["query", "value"]:
            module_state_dict = model.get_submodule(submodule_key).state_dict()
            submodule = model.get_submodule(submodule_key)
            lora_layer = lora.Linear(
                submodule.in_features,
                submodule.out_features,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1
            )
            lora_layer.load_state_dict(module_state_dict,strict=False)
            _set_module(model, submodule_key, lora_layer)
    

def load_from_save(tlmodel, name, strict: bool = True):
    model_path = name
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please provide a valid path to a trained TTCT model using --TL-loadpath."
        )
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")
    if strict:
        tlmodel.load_state_dict(state_dict, strict=True)
    else:
        cur = tlmodel.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in cur and cur[k].shape == v.shape}
        cur.update(matched)
        tlmodel.load_state_dict(cur, strict=False)
    

def main(args, cfg_env=None):
    if args.task == "MiniGrid":
        act_dim=1
        obs_dim=147
        obs_emb_dim=64
    elif args.task == "Craftax":
        act_dim = 1
        _ih, _iw = 63, 63
        obs_dim = CraftextPixelEncoder.output_dim(_ih, _iw)
        obs_emb_dim = obs_dim
    else:
        act_dim=2
        obs_dim=60
        obs_emb_dim=256
        
    embed_dim=512
    trajectory_length=200
    context_length=77
    vocab_size=49408
    config = isaac_gym_specific_cfg
    transformer_width=512
    transformer_heads=8
    transformer_layers=12
    BERT_PATH='./bert-base-uncased'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Helps cuDNN pick algorithms; reduces sporadic "Unable to find a valid cuDNN algorithm" under some drivers/VRAM states.
        torch.backends.cudnn.benchmark = True
        if os.environ.get("TTCT_CUDNN_DETERMINISTIC", "").strip() in ("1", "true", "yes"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if os.environ.get("TTCT_DISABLE_CUDNN", "").strip() in ("1", "true", "yes"):
            torch.backends.cudnn.enabled = False
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda" and os.environ.get("TTCT_ALLOW_TF32", "").strip() not in ("1", "true", "yes"):
        # Avoid CUDNN_STATUS_INTERNAL_ERROR / bad heuristics on some GPU + driver + PyTorch builds (PyTorch suggests this for similar failures).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda" and torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    use_predict_cost=args.use_predict_cost
    language_model= args.language_model
    if args.is_lava:
        config['threshold_Mini']=5.0
    _ttct_pixel_kw = {}
    _tl_load_strict = True
    if args.task == "Craftax":
        _ttct_pixel_kw = dict(use_pixel_encoder=True, image_hw=(63, 63))
        _tl_load_strict = False
    if language_model=="TLmodel":
        TL_loadpath=args.TL_loadpath
        if args.is_finetune:
            EncodeModel=TTCT(
                    embed_dim=embed_dim,
                    trajectory_length=trajectory_length,
                    context_length=context_length,
                    vocab_size=vocab_size,
                    transformer_width=transformer_width,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    act_dim=act_dim,
                    BERT_PATH='bert-base-uncased',
                    device=device,
                    obs_emb_dim=obs_emb_dim,
                    obs_dim=obs_dim,
                    threshold=config['threshold_Mini'] if args.task == "MiniGrid" else config['threshold_Goal'],
                    episodic_cost_value=config['cost_value'],
                    **_ttct_pixel_kw,
                )
            if args.use_pretrained_encoders:
                load_from_save(EncodeModel, TL_loadpath, strict=_tl_load_strict)
            if args.use_lora:
                lora_model(EncodeModel,args.rank)
                lora.mark_only_lora_as_trainable(EncodeModel)
            if args.task == "Craftax":
                for _n, _p in EncodeModel.named_parameters():
                    if _n.startswith("pixel_encoder.") or _n.startswith("embedding_act.") or _n.startswith("obs_encoder_linear."):
                        _p.requires_grad = True
            EncodeModel=EncodeModel.to(device)
        TLmodel=TTCT(
                    embed_dim=embed_dim,
                    trajectory_length=trajectory_length,
                    context_length=context_length,
                    vocab_size=vocab_size,
                    transformer_width=transformer_width,
                    transformer_heads=transformer_heads,
                    transformer_layers=transformer_layers,
                    act_dim=act_dim,
                    BERT_PATH='bert-base-uncased',
                    device=device,
                    obs_emb_dim=obs_emb_dim,
                    obs_dim=obs_dim,
                    threshold=config['threshold_Mini'] if args.task == "MiniGrid" else config['threshold_Goal'],
                    episodic_cost_value=config['cost_value'],
                    **_ttct_pixel_kw,
                )
        if args.use_pretrained_encoders:
            load_from_save(TLmodel, TL_loadpath, strict=_tl_load_strict)
        elif use_predict_cost:
            print(f"WARNING: --use-predict-cost is enabled but --use_pretrained_encoders is False.")
            print(f"         The TTCT model will use random weights, which may not work correctly.")
            print(f"         Consider training a TTCT model first or setting --use_pretrained_encoders=True with a valid --TL-loadpath.")
        TLmodel=TLmodel.to(device)
        TLmodel.eval()
        
    elif language_model=='Bert':
        TLmodel=TTCT(
                embed_dim=embed_dim,
                trajectory_length=trajectory_length,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
                act_dim=act_dim,
                BERT_PATH='bert-base-uncased',
                device=device,
                obs_emb_dim=obs_emb_dim,
                obs_dim=obs_dim,
            )
    else:
        raise NotImplementedError
    
    if args.is_finetune:
        EncodeModel.train()
    
    if args.task == "MiniGrid":
        def _make_minigrid(name):
            try:
                # Newer gym versions: disable_env_checker to allow 6-element step from env
                return gym.make(name, disable_env_checker=True)
            except TypeError:
                # Older gym versions without disable_env_checker
                return gym.make(name)

        def _wrap_minigrid(name):
            env = _make_minigrid(name)
            if getattr(args, "ignore_cost_termination", False):
                env = IgnoreCostTerminationWrapper(env, cost_threshold=1.0)
            return CostInInfoWrapper(env)
        envB=[lambda: _wrap_minigrid('MiniGrid-HazardWorld-B-v0') for _ in range(args.num_envs//3)]
        envS=[lambda: _wrap_minigrid('MiniGrid-HazardWorld-S-v0') for _ in range(args.num_envs//3)]
        envL=[lambda: _wrap_minigrid('MiniGrid-HazardWorld-L-v0') for _ in range(args.num_envs-(args.num_envs//3)*2)]
        allenv=envB+envS+envL
        if args.is_lava:
            allenv=[lambda: _wrap_minigrid('MiniGrid-HazardWorld-LavaWall-v0') for _ in range(args.num_envs)]
        env = AsyncVectorEnv(allenv)   
    elif args.task == "Craftax":
        # Use gym-style wrapper around Craftax CMDP env, returning 6-tuple step with cost.
        # CostInInfoWrapper is not needed because our env already returns (obs, reward, cost, term, trunc, info).
        def _make_craftax():
            return CraftaxCMDPGymEnv(
                env_name=getattr(args, "craftext_env_name", "Craftax-Classic-Pixels-v1-Text"),
                craftext_settings=getattr(args, "craftext_settings", "achievements_safe_budget_energy"),
                seed=int(args.seed),
                max_episode_steps=199,
                constraint_text=getattr(args, "constraint_text", "You must maintain your energy level at or above 8."),
            )
        allenv = [lambda: _make_craftax() for _ in range(args.num_envs)]
        # Avoid multiprocessing for JAX Craftax envs (can crash in subprocess).
        env = SimpleVectorEnv(allenv)
    elif args.task == "SafetyRacecarGoal2-v0":
        configB={"env_type":'budgetary','agent_name':'Racecar'}
        configR={"env_type":'relational','agent_name':'Racecar'}
        envB=[lambda: safety_gymnasium.make('SafetyRacecarGoal2-v0',max_episode_steps=199,render_mode='rgb_array',camera_name="human",width=256,height=256,config=configB) for _ in range(args.num_envs//2)]
        envR=[lambda: safety_gymnasium.make('SafetyRacecarGoal2-v0',max_episode_steps=199,render_mode='rgb_array',camera_name="human",width=256,height=256,config=configR) for _ in range(args.num_envs//2)]
        allenv=envB+envR
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv(allenv)
    else:
        raise NotImplementedError
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_space = torch.zeros((embed_dim+embed_dim+obs_dim,))
    if args.task != "MiniGrid":
        config["steps_per_epoch"]=config["steps_per_epoch"]//4
    # allow overriding batch size from CLI
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["batch_size"] = args.batch_size
    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    VALIDATION_INTERVAL_STEPS = 50_000
    next_validation_at = VALIDATION_INTERVAL_STEPS
    # create the actor-critic module
    policy = ActorVCriticTrajectory(
        obs_dim=obs_dim,
        trajectory_dim=embed_dim,
        text_dim=embed_dim,
        act_dim=act_space.n if args.task in ("MiniGrid", "Craftax") else act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        is_discrete=(args.task in ("MiniGrid", "Craftax"))
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=config['learning_rate'])
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=config['learning_rate']
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=config['learning_rate']
    )
    if args.is_finetune:
        Encode_optimizer = torch.optim.Adam(EncodeModel.parameters(), lr=1e-5)
        Encode_scheduler = LinearLR(
            Encode_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=epochs,
            verbose=False,
        )
    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        r_gamma=config["r_gamma"],
        c_gamma=config["c_gamma"],
    )
    _raw_ttct_enc = os.environ.get(
        "TTCT_PPO_ENCODE_MAXLEN", "64" if args.task == "Craftax" else "0"
    ).strip()
    _ttct_encode_maxlen = 0 if _raw_ttct_enc == "" else max(0, int(_raw_ttct_enc))
    # setup lagrangian multiplier
    lagrange = Lagrange(
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )

    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    use_comet = getattr(args, "use_comet", False)
    comet_experiment = None
    if use_comet:
        try:
            import comet_ml
            comet_project = getattr(args, "comet_project_name", "ttct_training")
            comet_workspace = getattr(args, "comet_workspace", None)
            comet_experiment = comet_ml.Experiment(project_name=comet_project, workspace=comet_workspace)
            comet_name = getattr(args, "comet_experiment_name", None)
            if comet_name:
                try:
                    comet_experiment.set_name(comet_name)
                except Exception:
                    pass
            params = dict(convert_json(dict_args))
            comet_experiment.log_parameters(params)
            if torch.cuda.is_available():
                comet_experiment.log_parameter("gpu_name", torch.cuda.get_device_name(0))
        except Exception:
            use_comet = False
            comet_experiment = None
            import traceback
            traceback.print_exc()
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
        use_comet=use_comet,
        comet_experiment=comet_experiment,
    )
    if use_comet and comet_experiment is not None:
        logger.log("Comet ML: experiment started (project=%s)" % getattr(args, "comet_project_name", "ttct-training"), color="green")
    rew_deque = deque(maxlen=50)
    train_cost_deque = deque(maxlen=50)
    true_cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    if args.is_finetune:
        logger.setup_torch_saver1(EncodeModel)
    logger.log("Start with training.")
    if args.task == "Craftax":
        logger.log(
            "Craftax rollout: TTCT_PPO_ENCODE_MAXLEN=%d (last T frames for policy TTCT encode; 0=full; set env to tune speed/quality)"
            % _ttct_encode_maxlen,
            color="cyan",
        )
    actlist=[[] for i in range(args.num_envs)]
    obslist=[[] for i in range(args.num_envs)]
    truecostlist=[[] for i in range(args.num_envs)]
    predictcostlist=[[] for i in range(args.num_envs)]
    obs, info = env.reset()
    if act_dim==1:
        act=-1
    elif act_dim==2:
        act=(-1,-1)
    else:
        raise ValueError("act_dim should be 1 or 2")
    mission=[]
    if args.task in ('MiniGrid', 'Craftax'):
        for idx in range(args.num_envs):
            mission.append(info[idx]['mission'])
    else:
        mission = info['mission']
        
    with torch.no_grad():
        emb_mission=TLmodel.test_encode_text(mission)
    if args.is_finetune:
        with torch.no_grad():
            finetune_mission=EncodeModel.test_encode_text(mission)
    for index,item in enumerate(obs):
        obslist[index].append(_copy_rollout_obs(item))
        actlist[index].append(act)
    lengths=[1 for i in range(args.num_envs)]
    # if use_predict_cost:
    _o0, _a0, _l0 = _slice_for_ttct_encode(obslist, actlist, lengths, _ttct_encode_maxlen)
    if args.is_finetune:
        with torch.no_grad():
            obswithconstraint=EncodeModel.test_encode(_o0,_a0,_l0,finetune_mission)
    else:
        with torch.no_grad():
            obswithconstraint = TLmodel.test_encode(_o0,_a0,_l0,emb_mission)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret,ep_cost_train,ep_cost_true, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    
    
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in tqdm(range(local_steps_per_epoch)):
            with torch.no_grad(): 
                act, log_prob, value_r, value_c = policy.step(obswithconstraint, deterministic=False)
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, true_cost, terminated, truncated, info = env.step(action)
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if done or time_out:
                    if args.task in ("MiniGrid", "Craftax"):
                        final_obs = info[idx]["final_observation"]
                    else:
                        final_obs = info["final_observation"][idx]
                    obslist[idx].append(_copy_rollout_obs(final_obs))
                else:
                    obslist[idx].append(_copy_rollout_obs(next_obs[idx]))
                actlist[idx].append(action[idx])
                lengths[idx] += 1
            _oe, _ae, _le = _slice_for_ttct_encode(obslist, actlist, lengths, _ttct_encode_maxlen)
            if args.is_finetune:
                with torch.no_grad():
                    next_obswithconstraint=EncodeModel.test_encode(_oe,_ae,_le,finetune_mission)
            cost_train = true_cost
            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost_train += cost_train
            ep_cost_true += true_cost.cpu().numpy() if args.task in isaac_gym_map.keys() else true_cost
            ep_len += 1
            next_obswithconstraint, reward, cost_train, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obswithconstraint, reward, cost_train, terminated, truncated)
            )
            _obs_buf, _act_buf = _buffer_histories(obslist, actlist)
            buffer.store(
                obs=obswithconstraint,
                act=torch.tensor(action),
                obslist=_obs_buf,
                actlist=_act_buf,
                lengths=[item-1 for item in lengths],
                mission=list(mission),
                reward=reward,
                cost=cost_train,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )
            
            obs = next_obs
            obswithconstraint=next_obswithconstraint
            epoch_end = steps >= local_steps_per_epoch - 1
            is_change=False
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obswithconstraint[idx].unsqueeze(0).to(device), deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obswithconstraint[idx].unsqueeze(0).to(device), deterministic=False
                                )
                        last_value_r = last_value_r
                        last_value_c = last_value_c
                    predict_cost=buffer.finish_path(
                                last_value_r=last_value_r, last_value_c=last_value_c, idx=idx,TL_condition=use_predict_cost,TLmodel=TLmodel,use_cost_prediction=args.use_credit_assignment,
                                obslist=obslist[idx],actlist=actlist[idx],emb_mission=emb_mission[idx]
                            )
                    if done or time_out:
                        is_change=True
                        if args.task in ("MiniGrid", "Craftax"):
                            mission[idx] = info[idx]["mission"]
                        else:
                            mission[idx] = info["mission"][idx]
                        lengths[idx] = 1
                        truecostlist[idx]=[0]
                        predictcostlist[idx]=[0]
                        obslist[idx] = [_copy_rollout_obs(obs[idx])]
                        if act_dim==1:
                            act=-1
                        elif act_dim==2:
                            act=(-1,-1)
                        actlist[idx] = [act]
                        if use_predict_cost:
                            train_cost_deque.append(predict_cost.cpu().numpy().sum())
                        else:
                            train_cost_deque.append(ep_cost_train[idx])
                        rew_deque.append(ep_ret[idx])
                        true_cost_deque.append(ep_cost_true[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCostTrain": np.mean(train_cost_deque),
                                "Metrics/EpCostTrue": np.mean(true_cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost_train[idx] = 0.0
                        ep_cost_true[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False
                    
            if is_change:
                with torch.no_grad():
                    emb_mission=TLmodel.test_encode_text(mission)
                if args.is_finetune:
                    with torch.no_grad():
                        finetune_mission=EncodeModel.test_encode_text(mission)
                        _or, _ar, _lr = _slice_for_ttct_encode(obslist, actlist, lengths, _ttct_encode_maxlen)
                        obswithconstraint=EncodeModel.test_encode(_or,_ar,_lr,finetune_mission)
                else:
                    with torch.no_grad():
                        _or, _ar, _lr = _slice_for_ttct_encode(obslist, actlist, lengths, _ttct_encode_maxlen)
                        obswithconstraint=TLmodel.test_encode(_or,_ar,_lr,emb_mission)        
        rollout_end_time = time.time()
        total_env_steps = (epoch + 1) * steps_per_epoch
        # Validation every 50k env steps: run for each passed milestone (шаги не ровно 50k — проверяем все пороги)
        comet_exp = getattr(logger, "comet_experiment", None)
        while (args.task == "MiniGrid" and total_env_steps >= next_validation_at and comet_exp is not None):
            validation_step = next_validation_at
            next_validation_at += VALIDATION_INTERVAL_STEPS
            try:
                logger.log("Validation at step %d (total_env_steps=%d)..." % (validation_step, total_env_steps), color="cyan")
                video_env = _wrap_minigrid('MiniGrid-HazardWorld-B-v0')()
                reset_out = video_env.reset()
                if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 2:
                    vid_obs, vid_info = reset_out[0], reset_out[1]
                else:
                    vid_obs, vid_info = reset_out, {}
                vid_mission = [vid_info.get('mission', '') if isinstance(vid_info, dict) else '']
                vid_obslist = [deepcopy(vid_obs)]
                vid_actlist = [[-1]]
                vid_lengths = [1]
                with torch.no_grad():
                    emb_m = TLmodel.test_encode_text(vid_mission)
                    if args.is_finetune:
                        finetune_m = EncodeModel.test_encode_text(vid_mission)
                frames = []
                vid_done = False
                vid_rew, vid_cost, vid_len = 0.0, 0.0, 0
                max_vid_steps = 500
                while not vid_done and vid_len < max_vid_steps:
                    if args.is_finetune:
                        obsw = EncodeModel.test_encode(vid_obslist, vid_actlist, vid_lengths, finetune_m)
                    else:
                        obsw = TLmodel.test_encode(vid_obslist, vid_actlist, vid_lengths, emb_m)
                    obsw = torch.as_tensor(obsw, dtype=torch.float32, device=device)
                    if obsw.dim() == 1:
                        obsw = obsw.unsqueeze(0)
                    act, _, _, _ = policy.step(obsw, deterministic=True)
                    action = act.detach().squeeze().cpu().numpy()
                    if np.isscalar(action):
                        action = np.array([action])
                    out = video_env.step(int(action[0]) if action.size else 0)
                    if len(out) == 5:
                        next_obs, reward, term, trunc, info = out
                        cost = info.get('cost', 0.0) if isinstance(info, dict) else 0.0
                    else:
                        next_obs, reward, cost, term, trunc, info = out
                    vid_rew += float(reward)
                    vid_cost += float(cost)
                    vid_len += 1
                    try:
                        frame = video_env.render(mode='rgb_array')
                        if frame is None:
                            frame = video_env.render()
                        if frame is not None:
                            if not hasattr(frame, 'shape') and hasattr(frame, 'size'):
                                frame = np.array(frame)
                            if hasattr(frame, 'shape') and len(frame.shape) >= 2:
                                frames.append(frame)
                    except Exception:
                        pass
                    vid_done = term or trunc
                    if not vid_done:
                        vid_obslist[0].append(deepcopy(next_obs))
                        vid_actlist[0].append(int(action[0]) if action.size else 0)
                        vid_lengths[0] += 1
                video_env.close()
                if frames:
                    import tempfile
                    try:
                        import imageio
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                            video_path = f.name
                        # Ensure uint8 (H,W,C) for imageio
                        frames_np = [np.asarray(f).astype(np.uint8) if f.dtype != np.uint8 else f for f in frames]
                        imageio.mimsave(video_path, frames_np, fps=10, codec='libx264')
                        logger.log_video(video_path, step=validation_step, name="eval_50k")
                        try:
                            os.unlink(video_path)
                        except Exception:
                            pass
                        logger.log("Validation: video logged (%d frames) at step %d" % (len(frames), validation_step), color="green")
                    except ImportError:
                        logger.log("Validation: imageio not installed, skipping video (metrics still logged)", color="yellow")
                    except Exception as e:
                        logger.log("Validation: video save failed: %s" % e, color="red")
                else:
                    logger.log("Validation: no frames from render (video not recorded) at step %d" % validation_step, color="yellow")
                comet_exp.log_metrics({
                    "Validation/EpRet": vid_rew,
                    "Validation/EpCost": vid_cost,
                    "Validation/EpLen": vid_len,
                }, step=validation_step)
            except Exception:
                import traceback
                traceback.print_exc()
                logger.log("Validation failed at step %d" % validation_step, color="red")
        eval_start_time = time.time()
        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCostTrain")
        lagrange.update_lagrange_multiplier(ep_costs)
        
        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])
        if args.task == "MiniGrid":
            old_distribution=Categorical(old_distribution)
        # comnpute advantage
        advantage = data["adv_r"] - lagrange._lagrangian_multiplier * data["adv_c"]
        advantage /= (lagrange._lagrangian_multiplier + 1)

        dataloader = DataLoader(
            dataset=BufferDataset(
                data["obslist"],
                data["actlist"],
                data["obs"],
                data["mission"],
                data["lengths"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,collate_fn=lambda x:x
        )
        update_counts = 0
        final_kl=None
        for _ in range(config["learning_iters"]):
            for traindata in tqdm(dataloader):
                (
                    obslist_b,
                    actlist_b,
                    obs_b,
                    mission_b,
                    lengths_b,
                    act_b,
                    log_prob_b,
                    target_value_r_b,
                    target_value_c_b,
                    adv_b
                ) = list(zip(*traindata))
                if args.is_finetune and update_counts==0:
                    Encode_optimizer.zero_grad()
                    text_featrues=EncodeModel.test_encode_text(mission_b)
                    _obl = list(obslist_b)
                    _abl = list(actlist_b)
                    _lbl = list(lengths_b)
                    _osl, _asl, _lsl = _slice_for_ttct_encode(_obl, _abl, _lbl, _ttct_encode_maxlen)
                    obs_b=EncodeModel.test_encode(_osl, _asl, _lsl, text_featrues)
                else:
                    obs_b=torch.cat([ele.unsqueeze(0) for ele in obs_b],dim=0).to(device)
                reward_critic_optimizer.zero_grad()
                target_value_r_b=torch.tensor(target_value_r_b).to(device)
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                target_value_c_b=torch.tensor(target_value_c_b).to(device)
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                
                if args.task == "MiniGrid":
                    act_b=torch.tensor(act_b).to(device)
                    action_probs = policy.actor(obs_b)
                    distribution=Categorical(action_probs)
                    log_prob = distribution.log_prob(act_b)
                else:
                    act_b_fix=torch.cat([item.unsqueeze(0) for item in act_b],dim=0).to(device)
                    distribution = policy.actor(obs_b)
                    log_prob = distribution.log_prob(act_b_fix).sum(dim=-1)
                log_prob_b=torch.tensor(log_prob_b).to(device)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                adv_b=torch.tensor(adv_b).to(device)
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                total_loss = loss_pi + 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi + loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()
                if args.is_finetune and update_counts==0:
                    Encode_optimizer.step()
                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                    }
                )

            new_distribution = policy.actor(data["obs"])
            if args.task == "MiniGrid":
                new_distribution=Categorical(new_distribution)
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .mean()
                .item()
            )
            print(kl)
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()
        if args.is_finetune:
            Encode_scheduler.step()
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCostTrain")
            logger.log_tabular("Metrics/EpCostTrue")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            # logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LagragianMultiplier", lagrange._lagrangian_multiplier)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())

            logger.env_step = (epoch + 1) * steps_per_epoch
            logger.dump_tabular()
            if (epoch+1) % 50 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                logger.torch_save1(itr=epoch)
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    
    relpath = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    if args.use_predict_cost:
        if args.use_credit_assignment:
            exp_fold="our"
        else:
            exp_fold="without_credit_assignment"
    else:
        exp_fold="standard"
    if args.is_lava:
        args.log_dir = os.path.join(args.log_dir, "lava", args.task, exp_fold , algo, relpath)
    else:
        args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, exp_fold , algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
