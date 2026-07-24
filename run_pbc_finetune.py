"""Finetune PbcCreditModel for binary classification.

Usage:
  python run_pbc_finetune.py configs/pbc_finetune.yaml
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import PbcDataset
from pbc_credit.fields import (
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS,
    SUMMARY_TABLES, PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE,
)
from pbc_credit.losses import finetune_loss, compute_auc, compute_ks
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig
from run_pbc_pretrain import build_model_cfg


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch)
        loss = finetune_loss(logits, batch['target'], pos_weight=pos_weight)
        probs = torch.sigmoid(logits.squeeze(-1))
        all_probs.append(probs.cpu())
        all_labels.append(batch['target'].cpu())
        total_loss += loss.item() * len(probs)
        n += len(probs)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return {
        'loss': total_loss / max(n, 1),
        'auc': compute_auc(probs, labels),
        'ks': compute_ks(probs, labels),
    }


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
    train_ds = PbcDataset(cfg['train_samples'])
    val_ds = PbcDataset(cfg['val_samples'])
    print(f'  train: {len(train_ds):,}')
    print(f'  val:   {len(val_ds):,}')

    collator = PbcCollator()
    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        collate_fn=collator, num_workers=cfg.get('num_workers', 0),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
        collate_fn=collator, num_workers=cfg.get('num_workers', 0),
        pin_memory=True,
    )

    # === Model ===
    with open(cfg['cat_vocab']) as f:
        vocab = json.load(f)
    model_cfg = build_model_cfg(cfg, vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PbcCreditModel(model_cfg, pretrain_mode=False).to(device)

    # Load pretrained
    init_from = cfg.get('init_from_pretrain')
    if init_from and Path(init_from).exists():
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        state = ckpt['model_state']
        own = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(loaded)
        model.load_state_dict(own)
        print(f'=== Loaded {len(loaded)}/{len(own)} tensors from {init_from} ===')
    else:
        print('=== No pretrain init; training from scratch ===')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.98), eps=1e-6,
    )
    total_steps = len(train_loader) * cfg['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], total_steps=total_steps, pct_start=0.1,
    )

    log_path = out_dir / 'finetune.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f'=== Training: {cfg["epochs"]} epochs x {len(train_loader)} steps = {total_steps} steps ===')

    best_auc = -1.0
    global_step = 0
    eval_every = cfg.get('eval_per_epoch_steps', 100)
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss = finetune_loss(logits, batch['target'], pos_weight=cfg.get('pos_weight', 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg.get('log_every', 20) == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                eta_min = (total_steps - global_step) / max(sps, 1e-9) / 60
                cur_lr = scheduler.get_last_lr()[0]
                log(f'  [ep{epoch} step {global_step}/{total_steps}] '
                    f'loss={loss.item():.4f}  lr={cur_lr:.2e}  sps={sps:.1f}  eta={eta_min:.1f}m')

            if global_step % eval_every == 0:
                metrics = evaluate(model, val_loader, device, cfg.get('pos_weight', 1.0))
                log(f'  [eval step {global_step}] loss={metrics["loss"]:.4f} '
                    f'auc={metrics["auc"]:.4f}  ks={metrics["ks"]:.4f}')
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    torch.save({
                        'model_state': model.state_dict(),
                        'metrics': metrics,
                        'step': global_step,
                    }, out_dir / 'best.pt')
                    log(f'    ★ new best AUC={best_auc:.4f} saved')
                model.train()

        avg = epoch_loss / max(n_batches, 1)
        log(f'=== Epoch {epoch} done: avg_loss={avg:.4f} ===')

    # final eval
    metrics = evaluate(model, val_loader, device, cfg.get('pos_weight', 1.0))
    log(f'=== Final eval: loss={metrics["loss"]:.4f} auc={metrics["auc"]:.4f} '
        f'ks={metrics["ks"]:.4f} (best AUC={best_auc:.4f}) ===')
    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump({'final': metrics, 'best_auc': best_auc}, f, indent=2)

    log_file.close()
    print(f'\n=== Finetune done. Total: {(time.time()-start_time)/60:.1f} min ===')


if __name__ == '__main__':
    main()
