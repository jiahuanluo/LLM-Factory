"""Pretrain HomeCreditModel with multi-modal joint mask reconstruction.

Usage:
  python run_home_credit_pretrain.py configs/hc_pretrain.yaml

YAML config keys:
  train_samples: data/home-credit/processed/samples_train.pkl       # labeled
  test_samples:  data/home-credit/processed/samples_test_unlabeled.pkl  # unlabeled (combined with train)
  cat_vocab:     data/home-credit/processed/cat_vocab.json
  output_dir:    output/hc_pretrained
  epochs: 2
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 0.01
  mask_ratio: 0.15
  d: 128
  n_heads: 4
  n_layers: 2
  dropout: 0.1
  num_workers: 4
  log_every: 50
  save_every_epochs: 1
  seed: 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from home_credit.collator import HomeCreditCollator
from home_credit.dataset import HomeCreditDataset
from home_credit.losses import pretrain_loss
from home_credit.masking import add_masks_to_batch
from home_credit.model import HomeCreditModel, HomeCreditModelConfig


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_vocab_sizes(path: str) -> dict:
    with open(path) as f:
        v = json.load(f)
    return {
        table: {f: max(m.values()) + 2 if m else 2 for f, m in fields.items()}
        for table, fields in v.items()
    }


class MaskingCollator:
    """Wraps HomeCreditCollator and applies add_masks_to_batch after collation."""

    def __init__(self, mask_ratio: float = 0.15):
        self.base = HomeCreditCollator()
        self.mask_ratio = mask_ratio

    def __call__(self, samples):
        batch = self.base(samples)
        # Drop 'target' if present (pretrain doesn't use it)
        batch.pop('target', None)
        return add_masks_to_batch(batch, mask_ratio=self.mask_ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config path')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    print(f'=== Config ===\n{yaml.safe_dump(cfg, default_flow_style=False)}')

    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))
    random.seed(cfg.get('seed', 42))

    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Data ===
    print('=== Loading datasets ===')
    paths = cfg.get('pretrain_samples') or [
        cfg['train_samples'], cfg.get('test_samples'),
    ]
    paths = [p for p in paths if p]
    datasets = []
    for p in paths:
        ds = HomeCreditDataset(p, pretrain_mode=True)  # drop 'target' field
        print(f'  {p}: {len(ds):,} samples')
        datasets.append(ds)
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f'  combined: {len(combined):,} samples')

    collator = MaskingCollator(mask_ratio=cfg.get('mask_ratio', 0.15))
    loader = DataLoader(
        combined,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    # === Model ===
    vocab_sizes = load_vocab_sizes(cfg['cat_vocab'])
    model_cfg = HomeCreditModelConfig(
        d=cfg.get('d', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 2),
        dropout=cfg.get('dropout', 0.1),
        user_cat_field_sizes=vocab_sizes['user'],
        bureau_cat_field_sizes=vocab_sizes['bureau'],
        prev_cat_field_sizes=vocab_sizes['prev'],
        card_cat_field_sizes=vocab_sizes['card'],
        text_vocab_size=cfg.get('text_vocab_size', 30522),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HomeCreditModel(model_cfg, pretrain_mode=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    # === Optimizer ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    total_steps = len(loader) * cfg['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], total_steps=total_steps,
        pct_start=0.05,
    )

    # === Train ===
    log_path = out_dir / 'pretrain.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f'=== Training: {cfg["epochs"]} epochs x {len(loader)} steps = {total_steps} steps ===')

    model.train()
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        epoch_components = {}
        n_batches = 0

        for batch in loader:
            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(batch)
            loss, log_dict = pretrain_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k, v in log_dict.items():
                epoch_components[k] = epoch_components.get(k, 0.0) + v
            n_batches += 1
            global_step += 1

            if global_step % cfg.get('log_every', 50) == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta_min = (total_steps - global_step) / max(steps_per_sec, 1e-9) / 60
                cur_lr = scheduler.get_last_lr()[0]
                components_str = " ".join(
                    f"{k.split('/')[-1]}={v:.3f}"
                    for k, v in log_dict.items() if k != 'loss/total'
                )
                log(f'  [ep{epoch} step {global_step}/{total_steps}] '
                    f'loss={loss.item():.4f}  {components_str}  '
                    f'lr={cur_lr:.2e}  sps={steps_per_sec:.1f}  eta={eta_min:.1f}m')

        avg_loss = epoch_loss / n_batches
        avg_components = {k: v / n_batches for k, v in epoch_components.items()}
        comp_str = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in avg_components.items() if k != 'loss/total')
        log(f'=== Epoch {epoch} done: avg_loss={avg_loss:.4f}  {comp_str} ===')

        # Save encoder + interactive weights (exclude task-specific mask heads)
        if (epoch + 1) % cfg.get('save_every_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            ckpt_path = out_dir / f'encoder_state_ep{epoch+1}.pt'
            encoder_state = {
                k: v for k, v in model.state_dict().items()
                if not k.endswith('_mask_head.weight') and not k.endswith('_mask_head.bias')
            }
            torch.save({
                'model_state': encoder_state,
                'config': {**cfg, 'vocab_sizes': vocab_sizes},
                'epoch': epoch + 1,
            }, ckpt_path)
            log(f'  saved {ckpt_path}')

            # Also save as the canonical "encoder_state.pt"
            torch.save({
                'model_state': encoder_state,
                'config': {**cfg, 'vocab_sizes': vocab_sizes},
                'epoch': epoch + 1,
            }, out_dir / 'encoder_state.pt')

    log_file.close()
    print(f'\n=== Pretrain done. Total time: {(time.time()-start_time)/60:.1f} min ===')
    print(f'Encoder weights: {out_dir / "encoder_state.pt"}')


if __name__ == '__main__':
    main()
