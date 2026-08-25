"""PBC Dataset：读 SQL 管道产出的 JSONL。

每行一个 JSON：
  {"reportsn": "...", "pbc_struct": "<stringified flat sample dict>", "label": 0/1?}
预训练样本不带 label（或 label 为 null）。

pbc_struct 由 scripts/postprocess_pbc_struct.py 从 Spark 物化表 dump 转换而来，
cat / paystate 的 encode 均已在 SQL 端完成；本类负责把（嵌套）list 转回
torch tensor 并补齐空 2D 字段（如 d1_numeric=[] → (0, 10)）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .fields import (
    ACCOUNT_TYPES, USER_NUMERIC_DIM, USER_CAT_DIM,
    ACCOUNT_NUMERIC_DIM, ACCOUNT_CAT_DIM, PAYSTATE_LEN,
)


# 2D 字段的列数（变长 N 维 → N x cols）。空数组反序列化后要 reshape 成 (0, cols)。
_2D_COL_COUNTS: dict[str, int] = {}
for _t in ACCOUNT_TYPES:
    _t = _t.lower()
    _2D_COL_COUNTS[f'{_t}_numeric'] = ACCOUNT_NUMERIC_DIM   # 10
    _2D_COL_COUNTS[f'{_t}_cat_ids'] = ACCOUNT_CAT_DIM       # 13
    _2D_COL_COUNTS[f'{_t}_cat_mask'] = ACCOUNT_CAT_DIM
    _2D_COL_COUNTS[f'{_t}_paystate'] = PAYSTATE_LEN         # 60


def _is_numeric_field(name: str) -> bool:
    return name.endswith('_numeric') or name == 'target'


def _to_tensor(name: str, value):
    """List → tensor；按字段名判 dtype；空 2D 字段 reshape 成 (0, cols)。"""
    if isinstance(value, (int, float)):
        return torch.tensor(value, dtype=torch.float32 if _is_numeric_field(name) else torch.long)
    if not isinstance(value, list):
        # 非数值（str 等），原样返回
        return value
    dtype = torch.float32 if _is_numeric_field(name) else torch.long
    t = torch.tensor(value, dtype=dtype)
    # 空 2D 字段：JSON 反序列化得到 shape (0,)，要补成 (0, cols)
    if name in _2D_COL_COUNTS and t.dim() == 1 and t.shape[0] == 0:
        t = t.reshape(0, _2D_COL_COUNTS[name])
    return t


class PbcDataset(Dataset):
    def __init__(self, path: str | Path, pretrain_mode: bool = False):
        self.path = Path(path)
        self.pretrain_mode = pretrain_mode
        self.samples: list[dict] = []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                outer = json.loads(line)
                struct_str = outer.get('pbc_struct')
                if struct_str is None:
                    raise ValueError(
                        f'JSONL line missing pbc_struct field: {line[:120]}'
                    )
                raw = json.loads(struct_str)
                sample = {k: _to_tensor(k, v) for k, v in raw.items()}
                sample['report_id'] = outer.get('reportsn', '')
                if not pretrain_mode and 'label' in outer and outer['label'] is not None:
                    sample['target'] = torch.tensor([float(outer['label'])], dtype=torch.float32)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
