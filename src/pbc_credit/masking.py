"""PBC masking：对 numeric / paystate / query / public / summary / text 分支做联合 mask。"""
from __future__ import annotations

import torch


# gte-new tokenizer 特殊 token id（在 0-103 范围，BERT 系约定）
TEXT_SPECIAL_IDS = (0, 100, 101, 102, 103)  # PAD / UNK / CLS / SEP / MASK


def mask_text(input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              mask_token_id: int = 103,
              mask_ratio: float = 0.15,
              vocab_size: int = 30528) -> tuple[torch.Tensor, torch.Tensor]:
    """对 text token 做 MLM mask（BERT 约定：跳过 special + pad）。

    Returns:
      masked_ids: 替换 [MASK] 后的 input_ids
      mask_pos: [B, L] bool，True = 被 mask 的位置
    """
    rand = torch.rand_like(input_ids, dtype=torch.float32)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in TEXT_SPECIAL_IDS:
        special_mask |= (input_ids == sid)
    valid = (attention_mask == 1) & (~special_mask)
    mask_pos = valid & (rand < mask_ratio)
    masked_ids = input_ids.clone()
    masked_ids[mask_pos] = mask_token_id
    return masked_ids, mask_pos


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

    # text MLM mask
    text_ids = batch.get('text_input_ids')
    text_mask = batch.get('text_attention_mask')
    if text_ids is not None and text_mask is not None:
        out['text_input_ids_raw'] = text_ids.clone()
        masked_ids, text_pos = mask_text(text_ids, text_mask, mask_ratio=mask_ratio)
        out['text_input_ids'] = masked_ids
        out['text_masked_pos'] = text_pos

    return out
