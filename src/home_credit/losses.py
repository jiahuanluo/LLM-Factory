"""Loss functions for HomeCreditModel pretrain + finetune."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pretrain_loss(pred_dict: dict) -> tuple[torch.Tensor, dict]:
    """Combine per-branch mask reconstruction losses.

    Numeric branches (bureau/prev/card): MSE against standardized targets.
    Text branch: CE against token IDs.

    All losses are equally weighted (per spec §5). Returns (total_loss, log_dict).
    """
    losses = {}
    if 'bureau_pred' in pred_dict:
        losses['bureau'] = F.mse_loss(pred_dict['bureau_pred'], pred_dict['bureau_target'])
    if 'prev_pred' in pred_dict:
        losses['prev'] = F.mse_loss(pred_dict['prev_pred'], pred_dict['prev_target'])
    if 'card_pred' in pred_dict:
        losses['card'] = F.mse_loss(pred_dict['card_pred'], pred_dict['card_target'])
    if 'text_logits' in pred_dict:
        losses['text'] = F.cross_entropy(
            pred_dict['text_logits'], pred_dict['text_target'],
        )

    if not losses:
        # Batch had nothing to mask (unlikely but possible if all branches empty)
        # Return zero with grad to keep backward happy
        zero = torch.tensor(0.0, device=next(iter(pred_dict.values())).device, requires_grad=True)
        return zero, {}

    total = sum(losses.values())
    log = {f'loss/{k}': v.detach().item() for k, v in losses.items()}
    log['loss/total'] = total.item()
    return total, log


def finetune_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """BCE with optional class-imbalance weighting.

    Args:
        logits: [B, 1] or [B]
        target: [B] long (0 or 1)
        pos_weight: weight for positive class (use ratio neg/pos for imbalance)
    """
    logits = logits.squeeze(-1) if logits.dim() == 2 and logits.shape[-1] == 1 else logits
    target_f = target.float()
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target_f, pos_weight=pw)
    return F.binary_cross_entropy_with_logits(logits, target_f)


@torch.no_grad()
def compute_auc(logits: torch.Tensor, target: torch.Tensor) -> float:
    """ROC AUC. Returns 0.5 if only one class present."""
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(logits.squeeze(-1) if logits.dim() == 2 else logits).cpu().numpy()
    y = target.cpu().numpy()
    if len(set(y.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(y, probs))


@torch.no_grad()
def compute_ks(logits: torch.Tensor, target: torch.Tensor) -> float:
    """KS statistic (max TPR - FPR gap)."""
    import numpy as np
    probs = torch.sigmoid(logits.squeeze(-1) if logits.dim() == 2 else logits).cpu().numpy()
    y = target.cpu().numpy()
    if len(set(y.tolist())) < 2:
        return 0.0
    order = np.argsort(-probs)
    y_sorted = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))
