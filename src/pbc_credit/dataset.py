"""PBC Dataset：读生产 JSONL 格式。

每行一个 JSON：
  {"biz_sno": "...", "pbcg2_json": "<stringified sample dict>", "label": 0/1}
预训练样本不带 label（或 label 为 null）。

pbcg2_json 解出来是 post-encode 的 sample dict，所有 tensor 字段以（嵌套）list
表示；本类负责把它们转回 torch tensor 并补齐空 2D 字段（如 d1_numeric=[] → (0, F)）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# 2D 字段的列数（变长 N 维 → N x cols）。空数组反序列化后要 reshape 成 (0, cols)。
_2D_COL_COUNTS: dict[str, int] = {}
for _t in ('d1', 'r1', 'r2', 'r3', 'r4'):
    _2D_COL_COUNTS[f'{_t}_numeric'] = 8
    _2D_COL_COUNTS[f'{_t}_cat_ids'] = 6
    _2D_COL_COUNTS[f'{_t}_cat_mask'] = 6
    _2D_COL_COUNTS[f'{_t}_paystate'] = 60
_2D_COL_COUNTS['query_numeric'] = 1
_2D_COL_COUNTS['query_cat_ids'] = 2
_2D_COL_COUNTS['query_cat_mask'] = 2
_2D_COL_COUNTS['public_numeric'] = 2
_2D_COL_COUNTS['public_cat_ids'] = 1
_2D_COL_COUNTS['public_cat_mask'] = 1


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
                raw = json.loads(outer['pbcg2_json'])
                sample = {k: _to_tensor(k, v) for k, v in raw.items()}
                sample['report_id'] = outer.get('biz_sno', '')
                if not pretrain_mode and 'label' in outer and outer['label'] is not None:
                    sample['target'] = torch.tensor([float(outer['label'])], dtype=torch.float32)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
