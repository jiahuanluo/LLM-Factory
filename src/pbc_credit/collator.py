"""PBC collator：把 list[sample] 填充成 batch dict。"""
from __future__ import annotations

import torch

from .fields import ACCOUNT_TYPES


def pad_2d(tensors: list[torch.Tensor], masks: list[torch.Tensor], pad_value: float = 0.0):
    """Pad list of [N_i, F] → [B, N_max, F]. Masks [N_i] → [B, N_max]."""
    B = len(tensors)
    N_max = max((t.shape[0] for t in tensors), default=0)
    if N_max == 0:
        F = tensors[0].shape[1] if tensors and tensors[0].dim() >= 2 else 0
        return (
            torch.zeros(B, 0, F, dtype=tensors[0].dtype if tensors else torch.float32),
            torch.zeros(B, 0, dtype=masks[0].dtype if masks else torch.long),
        )
    F = tensors[0].shape[1]
    dtype = tensors[0].dtype
    mtype = masks[0].dtype if masks else torch.long
    padded_t = torch.full((B, N_max, F), float(pad_value), dtype=dtype)
    padded_m = torch.zeros(B, N_max, dtype=mtype)
    for i, (t, m) in enumerate(zip(tensors, masks)):
        n = t.shape[0]
        if n > 0:
            padded_t[i, :n] = t
            padded_m[i, :n] = m
    return padded_t, padded_m


def pad_paystate(tensors: list[torch.Tensor], masks: list[torch.Tensor]):
    """Paystate: [N_i, 60] → [B, N_max, 60]."""
    B = len(tensors)
    N_max = max((t.shape[0] for t in tensors), default=0)
    if N_max == 0:
        return torch.zeros(B, 0, 60, dtype=torch.long)
    padded = torch.zeros(B, N_max, 60, dtype=torch.long)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n > 0:
            padded[i, :n] = t
    return padded


class PbcCollator:
    """Batch samples to model-ready dict（user + 6 类账户）。"""

    def __call__(self, samples: list[dict]) -> dict:
        batch: dict = {}

        # 1. user (fixed)
        batch['user_numeric'] = torch.stack([s['user_numeric'] for s in samples])
        batch['user_cat_ids'] = torch.stack([s['user_cat_ids'] for s in samples])
        batch['user_cat_mask'] = torch.stack([s['user_cat_mask'] for s in samples])

        # 2. accounts (variable, 6 types: d1/r1/r2/r3/r4/c1)
        for t_lower in [t.lower() for t in ACCOUNT_TYPES]:
            num_t = [s[f'{t_lower}_numeric'] for s in samples]
            mask_t = [s[f'{t_lower}_mask'] for s in samples]
            cat_t = [s[f'{t_lower}_cat_ids'] for s in samples]
            pay_t = [s[f'{t_lower}_paystate'] for s in samples]

            padded_num, padded_mask = pad_2d(num_t, mask_t, pad_value=0.0)
            padded_cat, _ = pad_2d(cat_t, mask_t, pad_value=0)
            padded_pay = pad_paystate(pay_t, mask_t)

            batch[f'{t_lower}_numeric'] = padded_num
            batch[f'{t_lower}_cat_ids'] = padded_cat
            batch[f'{t_lower}_paystate'] = padded_pay
            batch[f'{t_lower}_mask'] = padded_mask

        # 3. target (optional)
        if 'target' in samples[0]:
            batch['target'] = torch.stack([s['target'] for s in samples]).squeeze(-1)

        return batch
