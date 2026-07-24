"""HomeCreditCollator — batch a list of sample dicts into model-ready tensors."""
from __future__ import annotations

import torch


def pad_2d(tensors: list[torch.Tensor], masks: list[torch.Tensor], pad_value: float = 0.0):
    """Pad list of [N_i, F] tensors to [B, N_max, F].

    Args:
        tensors: list of [N_i, F] float tensors
        masks: list of [N_i] long tensors (1=valid, 0=pad)
        pad_value: fill value for padded positions

    Returns:
        padded_tensors: [B, N_max, F]
        padded_masks:   [B, N_max]
    """
    B = len(tensors)
    N_max = max((t.shape[0] for t in tensors), default=0)
    if N_max == 0:
        # All empty in this batch
        F = tensors[0].shape[1] if tensors else 0
        return (
            torch.zeros(B, 0, F, dtype=tensors[0].dtype if tensors else torch.float32),
            torch.zeros(B, 0, dtype=masks[0].dtype if masks else torch.long),
        )
    F = tensors[0].shape[1]
    dtype = tensors[0].dtype
    mtype = masks[0].dtype
    padded_t = torch.full((B, N_max, F), float(pad_value), dtype=dtype)
    padded_m = torch.zeros(B, N_max, dtype=mtype)
    for i, (t, m) in enumerate(zip(tensors, masks)):
        n = t.shape[0]
        if n > 0:
            padded_t[i, :n] = t
            padded_m[i, :n] = m
    return padded_t, padded_m


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


class HomeCreditCollator:
    """Batch samples into model-ready dict.

    Fixed fields: stacked.
    Variable fields: padded via pad_2d / pad_1d.
    Optional target: stacked if present.
    """

    def __call__(self, samples: list[dict]) -> dict:
        batch: dict = {}

        # 1. User profile (fixed)
        batch['user_numeric'] = torch.stack([s['user_numeric'] for s in samples])
        batch['user_cat_ids'] = torch.stack([s['user_cat_ids'] for s in samples])
        batch['user_cat_mask'] = torch.stack([s['user_cat_mask'] for s in samples])

        # 2. Query (fixed, length 6)
        batch['query_features'] = torch.stack([s['query_features'] for s in samples])
        batch['query_mask'] = torch.stack([s['query_mask'] for s in samples])

        # 3. Sub-table branches (variable length, pad to batch max)
        for branch in ['bureau', 'prev', 'card']:
            numeric_tensors = [s[f'{branch}_numeric'] for s in samples]
            mask_tensors = [s[f'{branch}_mask'] for s in samples]
            cat_tensors = [s[f'{branch}_cat_ids'] for s in samples]

            padded_num, padded_mask = pad_2d(numeric_tensors, mask_tensors, pad_value=0.0)

            # Cat ids have shape [N_i, F_cat]; pad along N dimension
            # Use same pad_2d but it expects masks to align — just call directly
            padded_cat, _ = pad_2d(cat_tensors, mask_tensors, pad_value=0)

            batch[f'{branch}_numeric'] = padded_num
            batch[f'{branch}_cat_ids'] = padded_cat
            batch[f'{branch}_mask'] = padded_mask

        # 4. Text (variable length, pad to batch max)
        text_ids = [s['text_input_ids'] for s in samples]
        text_masks = [s['text_attention_mask'] for s in samples]
        batch['text_input_ids'] = pad_1d(text_ids, pad_value=0)
        batch['text_attention_mask'] = pad_1d(text_masks, pad_value=0)

        # 5. Target (optional)
        if 'target' in samples[0]:
            batch['target'] = torch.stack([s['target'] for s in samples]).squeeze(-1)  # [B]

        return batch
