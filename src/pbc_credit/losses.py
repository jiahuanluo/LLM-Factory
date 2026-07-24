"""PBC losses: pretrain (multi-branch mask) + finetune (BCE)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pretrain_loss(out: dict, mlm_weight: float = 1.0, struct_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """各分支 mask 损失等权求和（含 text MLM）。

    out 中可能包含：
      account_numeric_pred / account_numeric_target  [N, F]
      account_paystate_pred / account_paystate_target [N, 60]
      query_numeric_pred / query_numeric_target [N, 1]
      public_numeric_pred / public_numeric_target [N, 2]
      summary_numeric_pred / summary_numeric_target [B, F]
      text_logits [B, L, V] / text_target [B, L] / text_mask_pos [B, L] bool
    """
    components = {}
    total = torch.zeros(1, device=_first_device(out), dtype=torch.float32)
    has_any = False

    for key in list(out.keys()):
        if key.endswith('_pred'):
            prefix = key[:-5]  # remove '_pred'
            tgt_key = f'{prefix}_target'
            if tgt_key not in out:
                continue
            pred, tgt = out[key], out[tgt_key]
            if pred.shape[0] == 0:
                continue
            if 'paystate' in prefix:
                loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                       tgt.reshape(-1).long(),
                                       ignore_index=0)
            else:
                loss = F.mse_loss(pred.float(), tgt.float())
            components[f'loss/{prefix}'] = loss.item()
            total = total + struct_weight * loss
            has_any = True

    # text MLM loss
    if 'text_logits' in out and 'text_target' in out:
        mask_pos = out.get('text_mask_pos')
        logits = out['text_logits']
        target = out['text_target']
        if mask_pos is not None and mask_pos.any():
            pred = logits[mask_pos]        # [N_masked, V]
            tgt = target[mask_pos]          # [N_masked]
            mlm = F.cross_entropy(pred, tgt.long())
            components['loss/text_mlm'] = mlm.item()
            total = total + mlm_weight * mlm
            has_any = True

    if not has_any:
        total = torch.zeros(1, device=total.device, dtype=torch.float32, requires_grad=True)
    components['loss/total'] = total.item()
    return total.mean(), components


def finetune_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """BCE with pos_weight for class imbalance."""
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target.float(), pos_weight=pw)


def _first_device(d: dict) -> torch.device:
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device('cpu')


# ============================================================
# Metrics
# ============================================================

def compute_auc(probs: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(target.cpu().numpy(), probs.cpu().numpy()))
    except Exception:
        return 0.5


def compute_ks(probs: torch.Tensor, target: torch.Tensor) -> float:
    """KS = max |F_positive(x) - F_negative(x)|."""
    import numpy as np
    p = probs.cpu().numpy()
    y = target.cpu().numpy()
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    cdf_pos = np.searchsorted(np.sort(pos), np.linspace(0, 1, 200), side='right') / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), np.linspace(0, 1, 200), side='right') / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))
