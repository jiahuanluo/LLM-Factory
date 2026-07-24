"""PBC Dataset：流式 pickle 读取（每条 sample 一个 pickle.dump）。"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PbcDataset(Dataset):
    def __init__(self, path: str | Path, pretrain_mode: bool = False):
        self.path = Path(path)
        self.pretrain_mode = pretrain_mode
        self.samples: list[dict] = []
        with open(self.path, 'rb') as f:
            while True:
                try:
                    self.samples.append(pickle.load(f))
                except EOFError:
                    break
        if pretrain_mode:
            for s in self.samples:
                s.pop('target', None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
