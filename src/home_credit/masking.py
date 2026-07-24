"""Generate mask positions for pretrain mode.

Three transaction branches (bureau/prev/card) get row-level masks (15%).
Text gets token-level mask (15%, avoiding special tokens).
User profile + query are NOT masked (per spec §5 — fixed-dim fields mask
trivially and risk learning local-correlation shortcuts).
"""
from __future__ import annotations

import torch


def generate_branch_mask(pad_mask: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
    """For each valid position (pad_mask==1), independently decide whether to mask.

    Args:
        pad_mask: [B, N] long, 1=valid row, 0=pad
        mask_ratio: probability of masking each valid row

    Returns:
        masked_pos: [B, N] bool, True at rows to predict
    """
    rand = torch.rand(pad_mask.shape, device=pad_mask.device, dtype=torch.float32)
    return (rand < mask_ratio) & (pad_mask == 1)


def generate_text_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_ratio: float = 0.15,
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103),
) -> torch.Tensor:
    """Mask 15% of non-special tokens.

    Default special_token_ids covers: pad (0), [UNK] (100), [CLS] (101),
    [SEP] (102), [MASK] (103) — BERT vocab conventions used by neobert.
    """
    valid = attention_mask == 1
    for sid in special_token_ids:
        valid = valid & (input_ids != sid)
    rand = torch.rand(input_ids.shape, device=input_ids.device, dtype=torch.float32)
    return (rand < mask_ratio) & valid


def add_masks_to_batch(
    batch: dict,
    mask_ratio: float = 0.15,
    text_special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103),
) -> dict:
    """Return a new batch dict with *_masked_pos tensors added.

    Generated:
        bureau_masked_pos, prev_masked_pos, card_masked_pos — [B, N] bool
        text_masked_pos — [B, L] bool
    """
    out = dict(batch)
    for branch in ['bureau', 'prev', 'card']:
        key = f'{branch}_mask'
        if key in out:
            out[f'{branch}_masked_pos'] = generate_branch_mask(out[key], mask_ratio)
    if 'text_input_ids' in out:
        out['text_masked_pos'] = generate_text_mask(
            out['text_input_ids'], out['text_attention_mask'],
            mask_ratio=mask_ratio, special_token_ids=text_special_token_ids,
        )
    return out
