"""PBC masking：对 user / accounts（numeric + paystate）分支做联合 mask。"""
from __future__ import annotations

import torch

from .fields import ACCOUNT_TYPES


def _mask_branch(numeric: torch.Tensor, mask: torch.Tensor, mask_ratio: float):
    """对 [B, N, F] 的 numeric 按 mask==1 的位置随机 mask。

    Returns:
      masked_numeric: 用 0 替换被 mask 的行的 numeric
      masked_pos: [B, N] bool, True = 被 mask 的位置
    """
    if numeric.shape[1] == 0:
        return numeric, torch.zeros_like(mask, dtype=torch.bool)
    valid = (mask == 1)  # [B, N]
    rand = torch.rand(numeric.shape[:2], device=numeric.device)  # [B, N]
    masked_pos = valid & (rand < mask_ratio)
    # numeric 用 0 替换 masked 行
    mp_num = masked_pos.unsqueeze(-1).expand_as(numeric)
    masked_numeric = numeric.masked_fill(mp_num, 0.0)
    return masked_numeric, masked_pos


def add_masks_to_batch(batch: dict, mask_ratio: float = 0.15) -> dict:
    """对 batch 中各分支应用 mask，附加 *_masked_pos 和 *_raw_target 字段。

    重要：mask 后原始值会丢失，所以预先把 target 存到 *_raw 字段，
    _forward_pretrain 必须读 *_raw，不能读 batch[f'{x}_numeric']（那是 masked 后的）。
    """
    out = dict(batch)

    # user numeric（固定 [B, 18]，按特征位 mask）
    u_num = batch.get('user_numeric')
    if u_num is not None:
        out['user_numeric_raw'] = u_num.clone()
        rand_u = torch.rand_like(u_num)
        u_pos = rand_u < mask_ratio
        out['user_numeric'] = u_num.masked_fill(u_pos, 0.0)
        out['user_masked_pos'] = u_pos

    # accounts numeric + paystate（6 类共享同一套逻辑）
    for t in [t.lower() for t in ACCOUNT_TYPES]:
        num = batch.get(f'{t}_numeric')
        mask = batch.get(f'{t}_mask')
        if num is None or mask is None:
            continue
        # 先存 raw target
        out[f'{t}_numeric_raw'] = num.clone()
        masked_num, pos = _mask_branch(num, mask, mask_ratio)
        out[f'{t}_numeric'] = masked_num
        out[f'{t}_masked_pos'] = pos

        # paystate：独立 mask（PAD 位 0 不参与）
        pay = batch.get(f'{t}_paystate')
        if pay is not None and pay.shape[1] > 0:
            out[f'{t}_paystate_raw'] = pay.clone()
            valid_pay = (pay != 0)
            rand_pay = torch.rand_like(pay, dtype=torch.float32)
            pos_pay = valid_pay & (rand_pay < mask_ratio)
            out[f'{t}_paystate'] = pay.masked_fill(pos_pay, 0)
            out[f'{t}_paystate_masked_pos'] = pos_pay

    return out
