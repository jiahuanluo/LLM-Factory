"""PBC collator：把 list[sample] 填充成 batch dict。"""
from __future__ import annotations

import torch


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


def pad_1d(tensors: list[torch.Tensor], pad_value: int = 0):
    """Pad list of [L_i] long tensors to [B, L_max]."""
    B = len(tensors)
    L_max = max((t.shape[0] for t in tensors), default=0)
    if L_max == 0:
        return torch.zeros(B, 0, dtype=tensors[0].dtype if tensors else torch.long)
    dtype = tensors[0].dtype
    padded = torch.full((B, L_max), pad_value, dtype=dtype)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n > 0:
            padded[i, :n] = t
    return padded


class PbcCollator:
    """Batch samples to model-ready dict."""

    def __call__(self, samples: list[dict]) -> dict:
        batch: dict = {}

        # 1. user (fixed)
        batch['user_numeric'] = torch.stack([s['user_numeric'] for s in samples])
        batch['user_cat_ids'] = torch.stack([s['user_cat_ids'] for s in samples])
        batch['user_cat_mask'] = torch.stack([s['user_cat_mask'] for s in samples])

        # 2. summary (fixed)
        batch['summary_numeric'] = torch.stack([s['summary_numeric'] for s in samples])
        batch['summary_cat_ids'] = torch.stack([s['summary_cat_ids'] for s in samples])
        batch['summary_cat_mask'] = torch.stack([s['summary_cat_mask'] for s in samples])

        # 3. accounts (variable, 5 types)
        for t_lower in ['d1', 'r1', 'r2', 'r3', 'r4']:
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

        # 4. queries
        q_num = [s['query_numeric'] for s in samples]
        q_mask = [s['query_mask'] for s in samples]
        q_cat = [s['query_cat_ids'] for s in samples]
        padded_qnum, padded_qmask = pad_2d(q_num, q_mask, pad_value=0.0)
        padded_qcat, _ = pad_2d(q_cat, q_mask, pad_value=0)
        batch['query_numeric'] = padded_qnum
        batch['query_cat_ids'] = padded_qcat
        batch['query_mask'] = padded_qmask

        # 5. publics
        p_num = [s['public_numeric'] for s in samples]
        p_mask = [s['public_mask'] for s in samples]
        p_cat = [s['public_cat_ids'] for s in samples]
        padded_pnum, padded_pmask = pad_2d(p_num, p_mask, pad_value=0.0)
        padded_pcat, _ = pad_2d(p_cat, p_mask, pad_value=0)
        batch['public_numeric'] = padded_pnum
        batch['public_cat_ids'] = padded_pcat
        batch['public_mask'] = padded_pmask

        # 6. target (optional)
        if 'target' in samples[0]:
            batch['target'] = torch.stack([s['target'] for s in samples]).squeeze(-1)

        # 7. text (pbc_text, 变长 pad 到 batch max)
        if 'text_input_ids' in samples[0]:
            text_ids = [s['text_input_ids'] for s in samples]
            text_masks = [s['text_attention_mask'] for s in samples]
            batch['text_input_ids'] = pad_1d(text_ids, pad_value=0)
            batch['text_attention_mask'] = pad_1d(text_masks, pad_value=0)

        return batch
