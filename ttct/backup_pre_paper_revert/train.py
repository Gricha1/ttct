import numpy as np
import torch
from loguru import logger
from torch import optim
from utils import (
    MultiPositiveContrastiveLoss,
    align_obs_act,
    gen_mask,
    gen_mask_from_nl,
    KLLoss,
    split_dataset,
)
from torch.optim import lr_scheduler
from TTCT import TTCT
from craftext_pixel_encoder import CraftextPixelEncoder
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Tuple


def contrastive_argmax_row_accuracy(
    logits: np.ndarray, mask: np.ndarray
) -> float:
    """
    Траектория i -> j* = argmax_j logits[i,j]. Верно, если mask[i, j*] == 1
    (выбранный «текст» в батчевой матрице совпадает с разрешённой парой по TL).
    """
    if logits is None or mask is None or logits.size == 0:
        return float("nan")
    if logits.shape != mask.shape or logits.ndim != 2:
        return float("nan")
    b, c = logits.shape
    if b == 0 or c == 0:
        return float("nan")
    good = 0
    used = 0
    for i in range(b):
        if not np.any(mask[i] >= 0.5):
            continue
        j_star = int(np.argmax(logits[i]))
        used += 1
        if mask[i, j_star] >= 0.5:
            good += 1
    if used == 0:
        return float("nan")
    return float(good) / used


def contrastive_argmax_col_accuracy(
    logits: np.ndarray, mask: np.ndarray
) -> float:
    """
    Текст j -> i* = argmax_i logits[i,j]. Верно, если mask[i*, j] == 1
    (симметричная метрика «текст выбрал свою траекторию»).
    """
    if logits is None or mask is None or logits.size == 0:
        return float("nan")
    if logits.shape != mask.shape or logits.ndim != 2:
        return float("nan")
    b, c = logits.shape
    if b == 0 or c == 0:
        return float("nan")
    good = 0
    used = 0
    for j in range(c):
        if not np.any(mask[:, j] >= 0.5):
            continue
        i_star = int(np.argmax(logits[:, j]))
        used += 1
        if mask[i_star, j] >= 0.5:
            good += 1
    if used == 0:
        return float("nan")
    return float(good) / used


def roc_auc_binary_maybe(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    ROC-AUC по бинарным меткам; None если в батче один класс или константные скоры
    (часто при маске «все пары положительные» — не подменяем 0.5, чтобы не вводить в заблуждение).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1e4, neginf=-1e4)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    if np.unique(y_score).size < 2:
        return None
    try:
        v = float(roc_auc_score(y_true, y_score))
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except ValueError:
        return None


def diagonal_retrieval_accuracy(logits: np.ndarray) -> float:
    """
    In-batch «CLIP»: траектория i должна иметь максимальный логит в колонке i
    (тот же индекс, что и свой текст в батче). Осмысленно даже при одинаковом NL,
    если эмбеддинги траекторий различаются.
    """
    if logits is None or logits.ndim != 2 or logits.size == 0:
        return float("nan")
    b, c = logits.shape
    if b != c or b == 0:
        return float("nan")
    good = sum(1 for i in range(b) if int(np.argmax(logits[i])) == i)
    return float(good) / float(b)


def logits_numpy_sanitized_for_metrics(
    logits_t: torch.Tensor, clip_abs: float = 1e4
) -> Tuple[np.ndarray, bool]:
    """
    Для AUC/accuracy: убираем NaN/Inf и клипаем — иначе на ранних шагах logit_scale даёт inf,
    sklearn и argmax дают nan, а KLDiv пишет «NaN or Inf in input tensor».
    На обучение/градиенты это не влияет (только снимок после forward).
    """
    x = logits_t.detach().cpu().float().numpy()
    had_bad = bool(np.isnan(x).any() or np.isinf(x).any())
    x = np.nan_to_num(x, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    x = np.clip(x, -clip_abs, clip_abs).astype(np.float32, copy=False)
    return x, had_bad


_GRAD_NORM_GROUP_PREFIXES = (
    ("bert", ("text_model.",)),
    ("text_projection", ("text_projection",)),
    ("trajectory_transformer", ("trajectory_transformer.",)),
    (
        "obs_encoder",
        ("obs_encoder", "obs_encoder_linear", "pixel_encoder", "embedding_act"),
    ),
    ("cost_heads", ("cost_assignment_layer", "episodic_cost_layer")),
    ("logit_scale", ("logit_scale",)),
)


def _grad_norm_param_group(param_name: str) -> str:
    for group, prefixes in _GRAD_NORM_GROUP_PREFIXES:
        if any(param_name.startswith(p) or param_name == p for p in prefixes):
            return group
    return "other"


def _collect_grad_norms_by_group(model: torch.nn.Module) -> Tuple[float, Dict[str, float]]:
    """L2 norm of gradients per module group and total (after backward)."""
    per_group_sq: Dict[str, float] = {}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g_norm = float(param.grad.detach().data.norm(2).item())
        if not np.isfinite(g_norm):
            continue
        group = _grad_norm_param_group(name)
        per_group_sq[group] = per_group_sq.get(group, 0.0) + g_norm * g_norm
        total_sq += g_norm * g_norm
    per_group = {k: float(np.sqrt(v)) for k, v in per_group_sq.items()}
    total = float(np.sqrt(total_sq)) if total_sq > 0 else 0.0
    return total, per_group


def measure_loss_grad_norms(
    model: torch.nn.Module,
    loss: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    return _collect_grad_norms_by_group(model)


def _log_train_grad_metrics(
    metrics: Dict[str, float],
    *,
    writer: SummaryWriter,
    step: int,
    comet_experiment=None,
) -> None:
    """TensorBoard train_grad/* and Comet train_grad/* (not mixed with train/loss)."""
    for key, val in metrics.items():
        writer.add_scalar(f"train_grad/{key}", val, step)
        if comet_experiment is not None:
            comet_experiment.log_metric(f"train_grad/{key}", val, step=step)


def log_component_grad_norms(
    model: torch.nn.Module,
    tta_loss: torch.Tensor,
    ca_weighted_loss: torch.Tensor,
    *,
    writer: SummaryWriter,
    step: int,
    comet_experiment=None,
) -> Dict[str, float]:
    """
    Separate backward passes for TTA vs weighted CA (debug only).
    Caller must zero_grad() before the training backward.
    """
    tta_total, tta_groups = measure_loss_grad_norms(
        model, tta_loss, retain_graph=True
    )
    ca_total, ca_groups = measure_loss_grad_norms(
        model, ca_weighted_loss, retain_graph=True
    )

    metrics: Dict[str, float] = {
        "tta_total": tta_total,
        "ca_weighted_total": ca_total,
    }
    if tta_total > 1e-12:
        metrics["ratio_ca_over_tta"] = ca_total / tta_total

    all_groups = set(tta_groups) | set(ca_groups)
    for group in sorted(all_groups):
        metrics[f"tta_{group}"] = tta_groups.get(group, 0.0)
        metrics[f"ca_{group}"] = ca_groups.get(group, 0.0)
        tta_g = tta_groups.get(group, 0.0)
        if tta_g > 1e-12:
            metrics[f"ratio_ca_over_tta_{group}"] = ca_groups.get(group, 0.0) / tta_g

    _log_train_grad_metrics(
        metrics, writer=writer, step=step, comet_experiment=comet_experiment
    )
    return metrics


def log_combined_grad_norms(
    model: torch.nn.Module,
    *,
    writer: SummaryWriter,
    step: int,
    comet_experiment=None,
    prefix: str = "combined",
) -> Dict[str, float]:
    """Grad norms after loss.backward() (actual training step, before clip)."""
    total, groups = _collect_grad_norms_by_group(model)
    metrics: Dict[str, float] = {f"{prefix}_total": total}
    for group, norm in groups.items():
        metrics[f"{prefix}_{group}"] = norm
    _log_train_grad_metrics(
        metrics, writer=writer, step=step, comet_experiment=comet_experiment
    )
    return metrics


def _tta_prepare_mask_and_forward(
    model: TTCT,
    observations: torch.Tensor,
    acts: torch.Tensor,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    lengths,
    TLss,
    NLss,
    tta_text_mode: str,
    device: torch.device,
    skip_inner_ce: bool = False,
):
    fwd_kw = dict(skip_inner_ce=skip_inner_ce)
    if tta_text_mode == "unique_nl":
        _, mask, count = gen_mask_from_nl(NLss)
        logits, ca = model(
            observations,
            acts,
            input_ids,
            attention_masks,
            lengths,
            nl_texts=NLss,
            **fwd_kw,
        )
    else:
        _, mask, count = gen_mask(TLss)
        logits, ca = model(
            observations,
            acts,
            input_ids,
            attention_masks,
            lengths,
            **fwd_kw,
        )
    mask_t = torch.tensor(mask, device=device, dtype=torch.float)
    return logits, ca, mask_t, count


def _compute_tta_loss(
    logits: torch.Tensor, mask: torch.Tensor, loss_traj, loss_text
):
    tta_traj = loss_traj(logits, mask)
    tta_text = loss_text(logits.t(), mask.t())
    return (tta_traj + tta_text) / 2, tta_traj, tta_text


def _tta_batch_diagnostics(
    logits: torch.Tensor, mask: torch.Tensor, temperature: float = 0.07
) -> Dict[str, float]:
    """
    TTA stuck near ln(batch_cols) usually means uniform softmax (random retrieval).
    pred_entropy_frac -> 1.0: predictions nearly uniform.
    """
    import math

    import torch.nn.functional as F

    with torch.no_grad():
        logits_f = logits.detach().float()
        b, c = logits_f.shape
        if b == 0 or c == 0:
            return {}
        probs = F.softmax(logits_f / max(temperature, 1e-6), dim=1)
        ent = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)
        max_ent = math.log(max(c, 2))
        mm = mask.detach().float().cpu().numpy()
        pos_per_row = mm.sum(axis=1)
        rows = F.normalize(logits_f, dim=1)
        gram = rows @ rows.t()
        eye = torch.eye(b, device=gram.device, dtype=torch.bool)
        off = gram[~eye]
        out = {
            "tta/pred_entropy_frac": float((ent.mean() / max_ent).item()),
            "tta/random_guess_ln_cols": float(math.log(max(c, 2))),
            "tta/mask_positives_per_row": float(pos_per_row.mean()),
            "tta/mask_cols": float(c),
            "tta/logits_std_row": float(logits_f.std(dim=1).mean().item()),
            "tta/logits_max_abs": float(logits_f.abs().max().item()),
            "tta/logits_row_offdiag_std": float(off.std().item()) if off.numel() else 0.0,
        }
        m = mask.detach().float()
        pos_logits = logits_f[m >= 0.5]
        neg_logits = logits_f[m < 0.5]
        if pos_logits.numel():
            out["tta/logits_pos_mean"] = float(pos_logits.mean().item())
        if neg_logits.numel():
            out["tta/logits_neg_mean"] = float(neg_logits.mean().item())
        if pos_logits.numel() and neg_logits.numel():
            out["tta/logits_margin_pos_minus_neg"] = float(
                pos_logits.mean().item() - neg_logits.mean().item()
            )
        if c > 1:
            pred = logits_f.argmax(dim=1)
            true = m.argmax(dim=1)
            out["tta/retrieval_acc"] = float((pred == true).float().mean().item())
            top2 = torch.topk(logits_f, k=min(2, c), dim=1).values
            if top2.size(1) >= 2:
                out["tta/logits_gap_top1_top2"] = float(
                    (top2[:, 0] - top2[:, 1]).mean().item()
                )
        if b == c:
            diag = torch.diag(gram)
            out["tta/logits_diag_mean"] = float(diag.mean().item())
            out["tta/logits_offdiag_mean"] = float(off.mean().item()) if off.numel() else 0.0
        return out


def _logits_contrastive_diagnostics(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Positive vs negative logits (mask-aligned; works for B×B and B×U)."""
    with torch.no_grad():
        x = logits.detach().float()
        if x.ndim != 2 or x.numel() == 0:
            return {}
        b, c = x.shape
        out: Dict[str, float] = {
            "logits/std": float(x.std().item()),
            "logits/max_abs": float(x.abs().max().item()),
        }
        if mask is not None and mask.shape == x.shape:
            m = mask.detach().float()
            pos = x[m >= 0.5]
            neg = x[m < 0.5]
            if pos.numel():
                out["logits/pos_mean"] = float(pos.mean().item())
                out["logits/diag_mean"] = out["logits/pos_mean"]
            if neg.numel():
                out["logits/neg_mean"] = float(neg.mean().item())
                out["logits/offdiag_mean"] = out["logits/neg_mean"]
            if pos.numel() and neg.numel():
                margin = float(pos.mean().item() - neg.mean().item())
                out["logits/margin_pos_minus_neg"] = margin
                out["logits/margin_diag_minus_off"] = margin
            return out
        if b != c:
            return out
        diag = torch.diag(x)
        off_mask = ~torch.eye(b, device=x.device, dtype=torch.bool)
        off = x[off_mask]
        out["logits/diag_mean"] = float(diag.mean().item())
        out["logits/offdiag_mean"] = float(off.mean().item()) if off.numel() else 0.0
        out["logits/margin_diag_minus_off"] = float(diag.mean().item() - off.mean().item()) if off.numel() else 0.0
    return out


# Comet ML для логирования экспериментов
try:
    from comet_ml import Experiment
    COMET_ML_AVAILABLE = True
except ImportError:
    COMET_ML_AVAILABLE = False
    print("⚠️  Comet ML не установлен. Установите: pip install comet_ml")

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--act_dim', type=int, default=1, help='Action dimension')
parser.add_argument('--context_length', type=int, default=77, help='Context length')
parser.add_argument('--obs_dim', type=int, default=147, help='Flat observation dim (linear encoder path only)')
parser.add_argument('--obs_emb_dim', type=int, default=64, help='Linear obs hidden dim (ignored when --use_pixel_encoder)')
parser.add_argument(
    '--use_pixel_encoder',
    action='store_true',
    help='Craftax-style CNN on (H,W,3) frames; dataset frames must be e.g. 63x63x3 (see caged_craftext gen --full_resolution).',
)
parser.add_argument('--image_h', type=int, default=63, help='Pixel encoder input height (with --use_pixel_encoder)')
parser.add_argument('--image_w', type=int, default=63, help='Pixel encoder input width (with --use_pixel_encoder)')
parser.add_argument('--vocab_size', type=int, default=49408, help='Vocabulary size')
parser.add_argument('--trajectory_length', type=int, default=200, help='Trajectory length')
parser.add_argument('--transformer_width', type=int, default=512, help='Transformer width')
parser.add_argument('--transformer_heads', type=int, default=8, help='Transformer heads')
parser.add_argument('--transformer_layers', type=int, default=12, help='Transformer layers')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=194, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.001,
    help='Adam weight decay (0 удобно для overfit на debug).',
)
parser.add_argument(
    '--lr_scheduler',
    type=str,
    default='step',
    choices=('step', 'none'),
    help='step: StepLR каждые lr_step_size эпох; none: постоянный LR.',
)
parser.add_argument(
    '--lr_step_size',
    type=int,
    default=10,
    help='StepLR: уменьшить LR каждые N эпох (на debug ~4 шага/эпоху → step 40 при N=10).',
)
parser.add_argument(
    '--lr_gamma',
    type=float,
    default=0.1,
    help='StepLR: множитель LR при decay.',
)
parser.add_argument('--dataset', type=str, default="./dataset/data.pkl")
parser.add_argument('--use_comet', action='store_true', help='Использовать Comet ML для логирования')
parser.add_argument('--comet_project_name', type=str, default='TTCT-Training', help='Имя проекта в Comet ML')
parser.add_argument('--comet_workspace', type=str, default=None, help='Workspace в Comet ML (опционально)')
parser.add_argument('--comet_experiment_name', type=str, default=None, help='Имя эксперимента в Comet ML (опционально)')
parser.add_argument(
    '--val_viz_every_epochs',
    type=int,
    default=1,
    help='Каждые N эпох: картинки val (trajectory + NL + cost model). 0 = выкл.',
)
parser.add_argument(
    '--val_viz_every_steps',
    type=int,
    default=0,
    help='Дополнительно: картинки каждые N train-шагов (0=только по эпохам).',
)
parser.add_argument('--val_viz_n_violated', type=int, default=2, help='Сколько примеров с GT violation на эпоху')
parser.add_argument('--val_viz_n_safe', type=int, default=2, help='Сколько примеров без нарушения на эпоху')
parser.add_argument('--val_viz_frames', type=int, default=8, help='Кадров траектории на картинке')
parser.add_argument(
    '--val_viz_constraint_pool',
    type=str,
    default='paper_full',
    choices=('paper_full', 'legacy_30'),
    help='Пул ограничений для oracle GT (check_violation)',
)
parser.add_argument('--cost_threshold', type=float, default=5.5, help='Порог cosine для get_cost (как в PPO-Lag)')
parser.add_argument('--episodic_cost_value', type=float, default=1.0, help='Стоимость при срабатывании порога')
parser.add_argument(
    '--ca_loss_weight',
    type=float,
    default=0.001,
    help='Множитель CA loss в суммарном loss (loss = TTA + ca_loss_weight * CA)',
)
parser.add_argument(
    '--grad_norm_log_every',
    type=int,
    default=10,
    help='Каждые N шагов: нормы градиентов TTA vs CA по группам (0=выкл). +2 backward/шаг.',
)
parser.add_argument(
    '--no_freeze_bert',
    action='store_false',
    dest='freeze_bert',
    help='Разморозить BERT (по умолчанию BERT заморожен).',
)
parser.set_defaults(freeze_bert=True)
parser.add_argument(
    '--freeze_trajectory_transformer',
    action='store_true',
    help='Не обучать trajectory_transformer (для debug / малых датасетов).',
)
parser.add_argument(
    '--tta_temperature',
    type=float,
    default=0.1,
    help='Температура softmax для TTA (MultiPositiveContrastiveLoss).',
)
parser.add_argument(
    '--tta_text_mode',
    type=str,
    default='inbatch',
    choices=('inbatch', 'unique_nl'),
    help='inbatch: B×B logits + TL mask; unique_nl: B×U logits, one-hot by NL (debug / few tasks)',
)
parser.add_argument(
    '--tta_skip_inner_ce',
    action='store_true',
    help='Не добавлять trajectory_inner_loss в CA (полезно при ca_loss_weight=0).',
)
parser.add_argument(
    '--tta_loss',
    type=str,
    default='soft_ce',
    choices=('soft_ce', 'kl'),
    help='TTA: soft_ce (mask target) или kl (KLLoss, как в оригинальном коде).',
)

args = parser.parse_args()

if __name__ == '__main__':
    # Create a SummaryWriter for logging
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'./result/{current_time}/log/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Инициализация Comet ML
    comet_experiment = None
    if args.use_comet and COMET_ML_AVAILABLE:
        try:
            comet_experiment = Experiment(
                project_name=args.comet_project_name,
                workspace=args.comet_workspace,
                auto_param_logging=False,  # Будем логировать параметры вручную
                auto_metric_logging=False,  # Будем логировать метрики вручную
                log_code=False,
            )
            print("✅ Comet ML инициализирован")
        except Exception as e:
            print(f"⚠️  Ошибка инициализации Comet ML: {e}")
            print("   Продолжаем без Comet ML")
            comet_experiment = None
    elif args.use_comet and not COMET_ML_AVAILABLE:
        print("⚠️  Comet ML запрошен, но не установлен. Продолжаем без Comet ML")
    
    # Логирование гиперпараметров в Comet ML
    if comet_experiment:
        hyperparams = {
            'embed_dim': args.embed_dim,
            'act_dim': args.act_dim,
            'context_length': args.context_length,
            'obs_dim': args.obs_dim,
            'obs_emb_dim': args.obs_emb_dim,
            'use_pixel_encoder': args.use_pixel_encoder,
            'image_h': args.image_h,
            'image_w': args.image_w,
            'vocab_size': args.vocab_size,
            'trajectory_length': args.trajectory_length,
            'transformer_width': args.transformer_width,
            'transformer_heads': args.transformer_heads,
            'transformer_layers': args.transformer_layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'lr_scheduler': args.lr_scheduler,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma,
            'ca_loss_weight': args.ca_loss_weight,
            'grad_norm_log_every': args.grad_norm_log_every,
            'freeze_bert': args.freeze_bert,
            'freeze_trajectory_transformer': args.freeze_trajectory_transformer,
            'tta_loss': args.tta_loss,
            'tta_text_mode': args.tta_text_mode,
            'tta_skip_inner_ce': args.tta_skip_inner_ce,
            'tta_temperature': args.tta_temperature,
            'dataset': args.dataset,
            'device': str(device),
        }
        
        # Добавляем информацию о GPU если доступно
        if torch.cuda.is_available():
            hyperparams['gpu_name'] = torch.cuda.get_device_name(0)
            hyperparams['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        comet_experiment.log_parameters(hyperparams)
        if args.comet_experiment_name:
            comet_experiment.set_name(args.comet_experiment_name)
        else:
            comet_experiment.set_name(f"TTCT-{current_time}")
        print(f"📊 Comet ML: эксперимент '{comet_experiment.get_name()}' создан")
    
    # Очистка кэша GPU перед началом обучения
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Свободно: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Предупреждение о большом batch_size
        if args.batch_size > 32:
            print(f"⚠️  ВНИМАНИЕ: batch_size={args.batch_size} может быть слишком большим для вашей GPU!")
            print(f"   Рекомендуется использовать --batch_size 16 или меньше")
    trajectory_length = args.trajectory_length
    context_length = args.context_length
    if args.use_pixel_encoder:
        ih, iw = int(args.image_h), int(args.image_w)
        cnn_out = CraftextPixelEncoder.output_dim(ih, iw)
        ttct_pixel_kw = dict(
            use_pixel_encoder=True,
            image_hw=(ih, iw),
            image_c=3,
        )
        obs_dim_m, obs_emb_m = cnn_out, cnn_out
    else:
        ttct_pixel_kw = {}
        obs_dim_m, obs_emb_m = args.obs_dim, args.obs_emb_dim

    model = TTCT(
        embed_dim=args.embed_dim,
        trajectory_length=args.trajectory_length,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        transformer_width=args.transformer_width,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        act_dim=args.act_dim,
        obs_dim=obs_dim_m,
        obs_emb_dim=obs_emb_m,
        BERT_PATH='bert-base-uncased',
        device=device,
        threshold=args.cost_threshold,
        episodic_cost_value=args.episodic_cost_value,
        **ttct_pixel_kw,
    ).to(device)

    if args.freeze_bert:
        for param in model.text_model.parameters():
            param.requires_grad = False
        logger.info("BERT frozen (train text_projection + trajectory path)")
    if args.freeze_trajectory_transformer:
        for param in model.trajectory_transformer.parameters():
            param.requires_grad = False
        logger.info("trajectory_transformer frozen (forward only)")

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_n:,}")

    if args.use_pixel_encoder and args.batch_size > 24:
        logger.warning(
            f"batch_size={args.batch_size} with 63×63 pixel trajectories often causes CUDA OOM during backward; "
            f"try --batch_size 8..16 or set BATCH_SIZE in train_ttct_budget_energy.sh (default full-res is 12)."
        )

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
        logger.info(
            f"LR scheduler: StepLR every {args.lr_step_size} epochs, gamma={args.lr_gamma}"
        )
    else:
        logger.info("LR scheduler: disabled (constant learning rate)")

    if args.tta_loss == "kl":
        loss_trajectory = KLLoss()
        loss_text = KLLoss()
        logger.info("TTA loss: KL on mask (KLLoss)")
    else:
        loss_trajectory = MultiPositiveContrastiveLoss(args.tta_temperature)
        loss_text = MultiPositiveContrastiveLoss(args.tta_temperature)
        logger.info(f"TTA loss: soft_ce on mask, temperature={args.tta_temperature}")
    total_step=0
    curr_total_loss=0
    curr_auc=0
    curr_TTA_loss=0
    curr_CA_loss=0
    curr_CA_loss_weighted=0
    ca_loss_weight = float(args.ca_loss_weight)
    logger.info(
        f'loss = TTA({args.tta_loss}, text={args.tta_text_mode}) + {ca_loss_weight} * CA ; '
        f'freeze_bert={args.freeze_bert}, freeze_traj_transformer={args.freeze_trajectory_transformer}'
    )
    if args.grad_norm_log_every > 0:
        logger.info(
            f'grad norms: every {args.grad_norm_log_every} steps -> TensorBoard/Comet train_grad/*'
        )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    trainset,testset=split_dataset(args.dataset)
    # Уменьшаем num_workers для экономии памяти
    num_workers = min(4, os.cpu_count() or 1)  # Ограничиваем количество воркеров
    dataloader_train=torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,collate_fn=lambda x:x)
    dataloader_test=torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,collate_fn=lambda x:x)

    steps_per_epoch = max(1, len(dataloader_train))
    viz_dir = f'./result/{current_time}/val_viz'
    logger.info(
        f'val_viz: PNG -> {viz_dir}/ ; train steps/epoch ~{steps_per_epoch} '
        f'(картинки по эпохам после ~{steps_per_epoch} шагов на оси train/*)'
    )
    if args.val_viz_every_steps > 0:
        logger.info(f'val_viz: additionally every {args.val_viz_every_steps} train steps')
    elif args.val_viz_every_epochs > 0:
        logger.info(
            'val_viz: only at end of each epoch (not every train step). '
            'Set --val_viz_every_steps 1000 for mid-epoch panels.'
        )

    def _run_val_viz(phase_dataset, epoch_idx: int, tag_prefix: str, step_for_log: int) -> None:
        if args.val_viz_every_epochs <= 0 and args.val_viz_every_steps <= 0:
            return
        from ttct_training_viz import run_epoch_validation_viz

        was_training = model.training
        try:
            viz_metrics = run_epoch_validation_viz(
                model,
                tokenizer,
                phase_dataset,
                out_dir=viz_dir,
                epoch=epoch_idx,
                tag_prefix=tag_prefix,
                device=device,
                context_length=context_length,
                n_violated=args.val_viz_n_violated,
                n_safe=args.val_viz_n_safe,
                n_frames=args.val_viz_frames,
                constraint_pool=args.val_viz_constraint_pool,
                threshold=args.cost_threshold,
                episodic_cost_value=args.episodic_cost_value,
                is_predict_cost=True,
                seed=42,
                writer=writer,
                comet_experiment=comet_experiment,
                global_step=step_for_log,
            )
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

        n_viz = int(viz_metrics.get("val_viz_n", 0))
        if n_viz > 0:
            acc_viz = viz_metrics["val_viz_acc"]
            paths = viz_metrics.get("val_viz_paths") or []
            logger.info(
                f'Val viz [{tag_prefix}]: {n_viz} panels, acc={acc_viz:.2f} -> {viz_dir}/'
            )
            cost_log_parts = []
            for key in (
                "cost_step_min",
                "cost_step_mean",
                "cost_step_max",
                "cost_episodic_min",
                "cost_episodic_mean",
                "cost_episodic_max",
            ):
                if key in viz_metrics and viz_metrics[key] == viz_metrics[key]:
                    cost_log_parts.append(f"{key}={viz_metrics[key]:.4f}")
            if cost_log_parts:
                logger.info(f"  cost: {', '.join(cost_log_parts)}")
            for p in paths[:4]:
                logger.info(f'  saved: {p}')
            writer.add_scalar("val_viz/accuracy", acc_viz, step_for_log)
            if comet_experiment:
                comet_experiment.log_metric("val_viz/accuracy", acc_viz, step=step_for_log)
        else:
            logger.warning(
                f'Val viz [{tag_prefix}]: no panels saved (check {viz_dir}, constraint pool, matplotlib)'
            )

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader_train, 0):
            model.train()
            transposed_data = list(zip(*data)) 
            obss = transposed_data[0]
            acts_raw = transposed_data[1]
            lengths_list = []
            padded_obss = []
            padded_acts = []
            for obs, act in zip(obss, acts_raw):
                obs_a, act_a, n = align_obs_act(obs, act)
                lengths_list.append(n)
                padded_obss.append(
                    np.pad(
                        obs_a,
                        ((0, trajectory_length - n), (0, 0), (0, 0), (0, 0)),
                        constant_values=0,
                    )
                )
                padded_acts.append(
                    np.pad(act_a, (0, trajectory_length - n), constant_values=0)
                )
            lengths = np.array(lengths_list, dtype=np.int32)
            padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
            acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
            TLss = list(transposed_data[2])
            observations = padded_obss.to(device, non_blocking=True)
            NLss=list(transposed_data[4])
            input_ids = []
            attention_masks = []
            for sent in NLss:
                encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
            attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
            step_grad_metrics: Optional[Dict[str, float]] = None
            logits_per_trajectory, CA_loss_raw, mask, _mask_count = _tta_prepare_mask_and_forward(
                model,
                observations,
                acts,
                input_ids,
                attention_masks,
                lengths,
                TLss,
                NLss,
                args.tta_text_mode,
                device,
                skip_inner_ce=args.tta_skip_inner_ce,
            )
            CA_loss_weighted = CA_loss_raw * ca_loss_weight
            TTA_loss, TTA_loss_traj, TTA_loss_text = _compute_tta_loss(
                logits_per_trajectory, mask, loss_trajectory, loss_text
            )
            loss = TTA_loss + CA_loss_weighted

            if (
                args.grad_norm_log_every > 0
                and total_step % args.grad_norm_log_every == 0
            ):
                step_grad_metrics = log_component_grad_norms(
                    model,
                    TTA_loss,
                    CA_loss_weighted,
                    writer=writer,
                    step=total_step,
                    comet_experiment=comet_experiment,
                )
                logit_scale_val = float(
                    model.logit_scale.exp().clamp(min=1e-3, max=100.0).detach().item()
                )
                writer.add_scalar("Metrics/logit_scale", logit_scale_val, total_step)
                if comet_experiment:
                    comet_experiment.log_metric(
                        "train/logit_scale", logit_scale_val, step=total_step
                    )
                for k, v in _logits_contrastive_diagnostics(
                    logits_per_trajectory, mask
                ).items():
                    writer.add_scalar(k.replace("logits/", "Logits/"), v, total_step)
                    if comet_experiment:
                        comet_experiment.log_metric(
                            f"train/{k.replace('logits/', 'logits_')}", v, step=total_step
                        )
                writer.add_scalar("Loss/Train_TTA_traj", TTA_loss_traj.item(), total_step)
                writer.add_scalar("Loss/Train_TTA_text", TTA_loss_text.item(), total_step)
                if comet_experiment:
                    comet_experiment.log_metric(
                        "train/loss_TTA_traj", TTA_loss_traj.item(), step=total_step
                    )
                    comet_experiment.log_metric(
                        "train/loss_TTA_text", TTA_loss_text.item(), step=total_step
                    )

            # Проверка на NaN/Inf в loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'NaN/Inf detected in loss at step {total_step}, skipping this batch')
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if (
                args.grad_norm_log_every > 0
                and total_step % args.grad_norm_log_every == 0
            ):
                log_combined_grad_norms(
                    model,
                    writer=writer,
                    step=total_step,
                    comet_experiment=comet_experiment,
                    prefix="combined",
                )
            # Gradient clipping для предотвращения взрывающихся градиентов
            grad_clip_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            if (
                args.grad_norm_log_every > 0
                and total_step % args.grad_norm_log_every == 0
            ):
                writer.add_scalar("train_grad/grad_clip_norm", float(grad_clip_norm), total_step)
                if comet_experiment:
                    comet_experiment.log_metric(
                        "train_grad/grad_clip_norm", float(grad_clip_norm), step=total_step
                    )
            optimizer.step()
            curr_total_loss+=loss.item()
            curr_TTA_loss+=TTA_loss.item()
            curr_CA_loss += CA_loss_raw.item()
            curr_CA_loss_weighted += CA_loss_weighted.item()
            if total_step % 10 == 0:
                mask_cpu = mask.cpu()
                mm = mask_cpu.numpy()
                lm, had_bad_logits = logits_numpy_sanitized_for_metrics(
                    logits_per_trajectory, clip_abs=1e4
                )
                if had_bad_logits and (total_step % 100 == 0):
                    logger.warning(
                        f'Logits contained NaN/Inf at step {total_step} (this warning at most every 100 steps); '
                        f'metrics AUC/accuracy use sanitized+clipped copy (not used in loss).'
                    )

                y_true = mm.flatten()
                y_pred = lm.flatten()
                roc_auc = roc_auc_binary_maybe(y_true, y_pred)
                diag_acc = diagonal_retrieval_accuracy(lm)
                mask_mean = float(np.mean(mm))
                mask_all_pos = bool(np.all(mm >= 0.5))
                tta_diag = _tta_batch_diagnostics(
                    logits_per_trajectory, mask, temperature=args.tta_temperature
                )
                for k, v in tta_diag.items():
                    writer.add_scalar(k.replace("tta/", "TTA/"), v, total_step)
                    if comet_experiment:
                        comet_experiment.log_metric(
                            f"train/{k.replace('tta/', 'tta_')}", v, step=total_step
                        )

                try:
                    acc_row = contrastive_argmax_row_accuracy(lm, mm)
                    acc_col = contrastive_argmax_col_accuracy(lm, mm)
                except Exception as e:
                    acc_row = 0.0
                    acc_col = 0.0
                    logger.warning(f'Error computing contrastive accuracy at step {total_step}: {e}')
                if isinstance(acc_row, float) and np.isnan(acc_row):
                    acc_row = 0.0
                if isinstance(acc_col, float) and np.isnan(acc_col):
                    acc_col = 0.0
                if isinstance(diag_acc, float) and np.isnan(diag_acc):
                    diag_acc = 0.0
                auc_str = f'{roc_auc:.4f}' if roc_auc is not None else 'n/a'
                avg_loss = curr_total_loss / 10
                avg_TTA_loss = curr_TTA_loss/10
                avg_CA_loss = curr_CA_loss / 10
                avg_CA_loss_weighted = curr_CA_loss_weighted / 10
                trivial = ' (mask all 1s: row/col acc trivial)' if mask_all_pos else ''
                grad_note = ""
                if step_grad_metrics:
                    tta_gn = step_grad_metrics.get("tta_total", 0.0)
                    ca_gn = step_grad_metrics.get("ca_weighted_total", 0.0)
                    ratio = step_grad_metrics.get("ratio_ca_over_tta", float("inf"))
                    tta_bert = step_grad_metrics.get("tta_bert", 0.0)
                    grad_note = (
                        f", grad||TTA={tta_gn:.2e} grad||CA_w={ca_gn:.2e} "
                        f"ratio={ratio:.2f} tta_bert={tta_bert:.2e}"
                    )

                logger.info(
                    f'Epoch: {epoch}, Loss: {avg_loss}, TTA_loss:{avg_TTA_loss}, '
                    f'CA_loss:{avg_CA_loss}, CA_weighted:{avg_CA_loss_weighted}, '
                    f'AUC: {auc_str}, diag_acc: {diag_acc:.4f}, mask_mean: {mask_mean:.3f}, '
                    f'acc_row: {acc_row}, acc_col: {acc_col}{trivial}{grad_note}'
                )
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_step)
                writer.add_scalar('Loss/Train_total', avg_loss, total_step)
                writer.add_scalar('Loss/Train_CA', avg_CA_loss, total_step)
                writer.add_scalar('Loss/Train_CA_weighted', avg_CA_loss_weighted, total_step)
                writer.add_scalar('Loss/Train_TTA', avg_TTA_loss, total_step)
                writer.add_scalar('Metrics/mask_mean', mask_mean, total_step)
                writer.add_scalar('Metrics/mask_all_positive', float(mask_all_pos), total_step)
                writer.add_scalar('Accuracy/Train_diag', diag_acc, total_step)
                if roc_auc is not None:
                    writer.add_scalar('AUC', roc_auc, total_step)
                if not (isinstance(acc_row, float) and np.isnan(acc_row)):
                    writer.add_scalar('Accuracy/Train_row', acc_row, total_step)
                if not (isinstance(acc_col, float) and np.isnan(acc_col)):
                    writer.add_scalar('Accuracy/Train_col', acc_col, total_step)
                if not (isinstance(acc_row, float) and np.isnan(acc_row)) and not (
                    isinstance(acc_col, float) and np.isnan(acc_col)
                ):
                    writer.add_scalar('Accuracy/Train_mean', (acc_row + acc_col) / 2.0, total_step)
                
                # Логирование в Comet ML
                if comet_experiment:
                    comet_experiment.log_metric('train/learning_rate', optimizer.param_groups[0]['lr'], step=total_step)
                    comet_experiment.log_metric('train/loss_total', avg_loss, step=total_step)
                    comet_experiment.log_metric('train/loss_CA', avg_CA_loss, step=total_step)
                    comet_experiment.log_metric(
                        'train/loss_CA_weighted', avg_CA_loss_weighted, step=total_step
                    )
                    comet_experiment.log_metric('train/loss_TTA', avg_TTA_loss, step=total_step)
                    if roc_auc is not None:
                        comet_experiment.log_metric('train/auc', roc_auc, step=total_step)
                    comet_experiment.log_metric('train/auc_skipped', float(roc_auc is None), step=total_step)
                    comet_experiment.log_metric('train/mask_mean', mask_mean, step=total_step)
                    comet_experiment.log_metric('train/mask_all_positive', float(mask_all_pos), step=total_step)
                    comet_experiment.log_metric('train/diag_accuracy', diag_acc, step=total_step)
                    comet_experiment.log_metric('train/accuracy', acc_row, step=total_step)
                    comet_experiment.log_metric('train/accuracy_text', acc_col, step=total_step)
                    if not (isinstance(acc_row, float) and np.isnan(acc_row)) and not (
                        isinstance(acc_col, float) and np.isnan(acc_col)
                    ):
                        comet_experiment.log_metric('train/accuracy_mean', (acc_row + acc_col) / 2.0, step=total_step)
                    comet_experiment.log_metric('train/epoch', epoch, step=total_step)
                    
                    # Логирование использования GPU памяти (если доступно)
                    if torch.cuda.is_available():
                        comet_experiment.log_metric('system/gpu_memory_allocated', 
                                                   torch.cuda.memory_allocated(0) / 1024**3, step=total_step)
                        comet_experiment.log_metric('system/gpu_memory_reserved', 
                                                   torch.cuda.memory_reserved(0) / 1024**3, step=total_step)
                
                curr_total_loss, curr_auc, curr_TTA_loss, curr_CA_loss, curr_CA_loss_weighted = (
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            total_step += 1

            if (
                args.val_viz_every_steps > 0
                and total_step > 0
                and total_step % args.val_viz_every_steps == 0
            ):
                with torch.no_grad():
                    _run_val_viz(
                        trainset,
                        epoch,
                        tag_prefix=f"step{total_step:06d}",
                        step_for_log=total_step,
                    )
            
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_roc_auc=0.0
            valid_auc_steps=0
            test_step=0
            test_TTA_loss=0
            test_CA_loss=0
            test_CA_loss_weighted=0
            test_acc_row_sum = 0.0
            test_acc_col_sum = 0.0
            test_acc_batch_count = 0
            test_diag_sum = 0.0
            test_mask_mean_sum = 0.0
            for i, data in enumerate(dataloader_test, 0):
                transposed_data = list(zip(*data)) 
                obss = transposed_data[0]
                acts_raw = transposed_data[1]
                lengths_list = []
                padded_obss = []
                padded_acts = []
                for obs, act in zip(obss, acts_raw):
                    obs_a, act_a, n = align_obs_act(obs, act)
                    lengths_list.append(n)
                    padded_obss.append(
                        np.pad(
                            obs_a,
                            ((0, trajectory_length - n), (0, 0), (0, 0), (0, 0)),
                            constant_values=0,
                        )
                    )
                    padded_acts.append(
                        np.pad(act_a, (0, trajectory_length - n), constant_values=0)
                    )
                lengths = np.array(lengths_list, dtype=np.int32)
                padded_obss = torch.tensor(np.array(padded_obss), dtype=torch.float32).to(device, non_blocking=True)
                acts = torch.tensor(np.array(padded_acts), dtype=torch.float32).to(device, non_blocking=True)
                TLss = list(transposed_data[2])
                observations = padded_obss.to(device, non_blocking=True)
                NLss=list(transposed_data[4])
                input_ids = []
                attention_masks = []
                for sent in NLss:
                    encoded_dict=tokenizer.encode_plus(sent, add_special_tokens=True, max_length=context_length, padding='max_length', return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
                    input_ids.append(encoded_dict['input_ids'])
                    attention_masks.append(encoded_dict['attention_mask'])
                input_ids = torch.cat(input_ids, dim=0).to(device, non_blocking=True)
                attention_masks = torch.cat(attention_masks, dim=0).to(device, non_blocking=True)
                logits_per_trajectory, CA_loss_raw, mask, _mask_count = _tta_prepare_mask_and_forward(
                    model,
                    observations,
                    acts,
                    input_ids,
                    attention_masks,
                    lengths,
                    TLss,
                    NLss,
                    args.tta_text_mode,
                    device,
                    skip_inner_ce=args.tta_skip_inner_ce,
                )
                CA_loss_weighted = CA_loss_raw * ca_loss_weight
                TTA_loss, _, _ = _compute_tta_loss(
                    logits_per_trajectory, mask, loss_trajectory, loss_text
                )
                loss_eval = TTA_loss + CA_loss_weighted
                mask_cpu = mask.cpu()
                mm = mask_cpu.numpy()
                lm, _ = logits_numpy_sanitized_for_metrics(
                    logits_per_trajectory, clip_abs=1e4
                )
                _auc_b = roc_auc_binary_maybe(mm.flatten(), lm.flatten())
                if _auc_b is not None:
                    test_roc_auc += _auc_b
                    valid_auc_steps += 1
                try:
                    ar = contrastive_argmax_row_accuracy(lm, mm)
                    ac = contrastive_argmax_col_accuracy(lm, mm)
                except Exception:
                    ar, ac = 0.0, 0.0
                if isinstance(ar, float) and np.isnan(ar):
                    ar = 0.0
                if isinstance(ac, float) and np.isnan(ac):
                    ac = 0.0
                test_acc_row_sum += ar
                test_acc_col_sum += ac
                test_acc_batch_count += 1
                _d = diagonal_retrieval_accuracy(lm)
                if not (isinstance(_d, float) and np.isnan(_d)):
                    test_diag_sum += float(_d)
                test_mask_mean_sum += float(np.mean(mm))
                test_step+=1
                test_loss += loss_eval.item()
                test_CA_loss += CA_loss_raw.item()
                test_CA_loss_weighted += CA_loss_weighted.item()
                test_TTA_loss+=TTA_loss.item() 
            writer.add_scalar('Test_learning_rate',optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Loss/Test_all', test_loss/test_step, epoch)
            writer.add_scalar('Loss/Test_CA', test_CA_loss / test_step, epoch)
            writer.add_scalar('Loss/Test_CA_weighted', test_CA_loss_weighted / test_step, epoch)
            writer.add_scalar('Loss/Test_TTA', test_TTA_loss/test_step, epoch)
            test_auc_value = float("nan") if valid_auc_steps == 0 else (test_roc_auc / valid_auc_steps)
            writer.add_scalar('Test_AUC', test_auc_value, epoch)
            if test_step > 0:
                writer.add_scalar('Accuracy/Test_diag', test_diag_sum / test_step, epoch)
                writer.add_scalar('Metrics/Test_mask_mean', test_mask_mean_sum / test_step, epoch)
                writer.add_scalar(
                    'Metrics/Test_auc_batches_valid_frac',
                    valid_auc_steps / max(test_step, 1),
                    epoch,
                )
            test_acc_row_mean = float("nan")
            test_acc_col_mean = float("nan")
            if test_acc_batch_count > 0:
                test_acc_row_mean = test_acc_row_sum / test_acc_batch_count
                test_acc_col_mean = test_acc_col_sum / test_acc_batch_count
                writer.add_scalar('Accuracy/Test_row', test_acc_row_mean, epoch)
                writer.add_scalar('Accuracy/Test_col', test_acc_col_mean, epoch)
                writer.add_scalar('Accuracy/Test_mean', (test_acc_row_mean + test_acc_col_mean) / 2.0, epoch)
            
            # Логирование тестовых метрик в Comet ML
            if comet_experiment:
                comet_experiment.log_metric('test/learning_rate', optimizer.param_groups[0]['lr'], step=epoch)
                comet_experiment.log_metric('test/loss_total', test_loss/test_step, step=epoch)
                comet_experiment.log_metric('test/loss_CA', test_CA_loss / test_step, step=epoch)
                comet_experiment.log_metric(
                    'test/loss_CA_weighted', test_CA_loss_weighted / test_step, step=epoch
                )
                comet_experiment.log_metric('test/loss_TTA', test_TTA_loss/test_step, step=epoch)
                if valid_auc_steps > 0:
                    comet_experiment.log_metric('test/auc', test_auc_value, step=epoch)
                comet_experiment.log_metric(
                    'test/auc_batches_valid_frac', valid_auc_steps / max(test_step, 1), step=epoch
                )
                if test_step > 0:
                    comet_experiment.log_metric(
                        'test/diag_accuracy', test_diag_sum / test_step, step=epoch
                    )
                    comet_experiment.log_metric(
                        'test/mask_mean', test_mask_mean_sum / test_step, step=epoch
                    )
                if test_acc_batch_count > 0:
                    comet_experiment.log_metric('test/accuracy', test_acc_row_mean, step=epoch)
                    comet_experiment.log_metric('test/accuracy_text', test_acc_col_mean, step=epoch)
                    comet_experiment.log_metric(
                        'test/accuracy_mean', (test_acc_row_mean + test_acc_col_mean) / 2.0, step=epoch
                    )
                logger.info(f'Comet ML: Эпоха {epoch} залогирована')

            if args.val_viz_every_epochs > 0 and (
                epoch % args.val_viz_every_epochs == 0 or epoch == args.epochs - 1
            ):
                _run_val_viz(
                    testset,
                    epoch,
                    tag_prefix=f"epoch{epoch:03d}",
                    step_for_log=total_step,
                )
        
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/learning_rate", lr_now, total_step)
        if comet_experiment:
            comet_experiment.log_metric("train/learning_rate", lr_now, step=total_step)
        logger.info(f"Epoch {epoch} done, lr={lr_now:.2e}")
        checkpoint_dir = f'./result/{current_time}/model/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Сохраняем только последний чекпоинт (удаляем предыдущий для экономии места)
        checkpoint_path = f'./result/{current_time}/model/checkpoint_latest.pt'
        
        # Удаляем предыдущий чекпоинт, если он существует
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить предыдущий чекпоинт: {e}")
        
        # Сохраняем текущий чекпоинт (перезаписываем предыдущий)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Чекпоинт сохранен: {checkpoint_path}')
    
    writer.close()
    
    # Финальный чекпоинт уже сохранен как checkpoint_latest.pt
    checkpoint_path = f'./result/{current_time}/model/checkpoint_latest.pt'
    
    # Логируем финальный чекпоинт в Comet ML (только один раз в конце)
    if comet_experiment:
        if os.path.exists(checkpoint_path):
            comet_experiment.log_asset(checkpoint_path, file_name='checkpoint_latest.pt')
            logger.info(f'Comet ML: Финальный чекпоинт залогирован')
        comet_experiment.end()
        print(f"✅ Comet ML: эксперимент завершен")
    
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Чекпоинт: {checkpoint_path}")