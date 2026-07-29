"""PBC masking：对 numeric / paystate / query / public / summary 分支做联合 mask。"""
from __future__ import annotations

import torch


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

    重要：mask 后原始值会丢失，所以预先把 target 存到 *_raw_target 字段，
    _forward_pretrain 必须读 *_raw_target，不能读 batch[f'{x}_numeric']（那是 masked 后的）。
    """
    out = dict(batch)

    # accounts numeric + paystate
    for t in ['d1', 'r1', 'r2', 'r3', 'r4']:
        num = batch.get(f'{t}_numeric')
        mask = batch.get(f'{t}_mask')
        if num is None or mask is None:
            continue
        # 先存 raw target
        out[f'{t}_numeric_raw'] = num.clone()
        masked_num, pos = _mask_branch(num, mask, mask_ratio)
        out[f'{t}_numeric'] = masked_num
        out[f'{t}_masked_pos'] = pos

        # paystate：独立 mask
        pay = batch.get(f'{t}_paystate')
        if pay is not None and pay.shape[1] > 0:
            out[f'{t}_paystate_raw'] = pay.clone()
            valid_pay = (pay != 0)
            rand_pay = torch.rand_like(pay, dtype=torch.float32)
            pos_pay = valid_pay & (rand_pay < mask_ratio)
            out[f'{t}_paystate'] = pay.masked_fill(pos_pay, 0)
            out[f'{t}_paystate_masked_pos'] = pos_pay

    # queries
    q_num = batch.get('query_numeric')
    q_mask = batch.get('query_mask')
    if q_num is not None:
        out['query_numeric_raw'] = q_num.clone()
        masked_qn, q_pos = _mask_branch(q_num, q_mask, mask_ratio)
        out['query_numeric'] = masked_qn
        out['query_masked_pos'] = q_pos

    # publics
    p_num = batch.get('public_numeric')
    p_mask = batch.get('public_mask')
    if p_num is not None:
        out['public_numeric_raw'] = p_num.clone()
        masked_pn, p_pos = _mask_branch(p_num, p_mask, mask_ratio)
        out['public_numeric'] = masked_pn
        out['public_masked_pos'] = p_pos

    # summary numeric
    s_num = batch.get('summary_numeric')
    if s_num is not None:
        out['summary_numeric_raw'] = s_num.clone()
        rand_s = torch.rand_like(s_num)
        s_pos = rand_s < mask_ratio
        out['summary_numeric'] = s_num.masked_fill(s_pos, 0.0)
        out['summary_masked_pos'] = s_pos

    return out
