"""HomeCreditDataset — indexes into a streaming-pickle samples file."""
from __future__ import annotations

import pickle
from pathlib import Path

from torch.utils.data import Dataset


class HomeCreditDataset(Dataset):
    """Reads a samples_*.pkl produced by sample_builder.

    The file is a pickle STREAM: one dict per pickle.dump call. This avoids
    materializing the whole list during build and avoids a double-buffer when
    loading (we read records sequentially into one list).
    """

    def __init__(self, path: str | Path, pretrain_mode: bool = False):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.pretrain_mode = pretrain_mode
        self.samples = []
        with open(path, 'rb') as f:
            while True:
                try:
                    self.samples.append(pickle.load(f))
                except EOFError:
                    break
                except Exception as e:
                    # Partial record at end (e.g., process died mid-write)
                    print(f'  [HomeCreditDataset] stopping at corrupt/short record: {e}')
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.pretrain_mode and 'target' in s:
            s = {k: v for k, v in s.items() if k != 'target'}
        return s
