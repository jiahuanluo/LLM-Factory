"""Finetune HomeCreditModel for TARGET classification.

Usage:
  python run_home_credit_finetune.py configs/hc_finetune.yaml

YAML config keys:
  train_samples:        data/home-credit/processed/samples_train.pkl
  val_samples:          data/home-credit/processed/samples_val.pkl
  cat_vocab:            data/home-credit/processed/cat_vocab.json
  init_from_pretrain:   output/hc_pretrained/encoder_state.pt  # or null
  output_dir:           output/hc_finetuned
  epochs: 5
  batch_size: 64
  lr: 5.0e-5
  weight_decay: 0.01
  pos_weight: 11.4        # BCE pos weight (neg/pos ratio)
  d: 128
  n_heads: 4
  n_layers: 2
  dropout: 0.1
  num_workers: 4
  eval_per_epoch_steps: 2000
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from home_credit.collator import HomeCreditCollator
from home_credit.dataset import HomeCreditDataset
from home_credit.losses import compute_auc, compute_ks, finetune_loss
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


def evaluate(model, loader, device, pos_weight=None):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(batch)
            loss = finetune_loss(logits, batch['target'], pos_weight=pos_weight)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu())
            all_targets.append(batch['target'].detach().cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return {
        'loss': total_loss / max(n_batches, 1),
        'auc': compute_auc(logits, targets),
        'ks': compute_ks(logits, targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
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
    train_ds = HomeCreditDataset(cfg['train_samples'])
    val_ds = HomeCreditDataset(cfg['val_samples'])
    print(f'  train: {len(train_ds):,}  val: {len(val_ds):,}')

    collator = HomeCreditCollator()
    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        collate_fn=collator, num_workers=cfg.get('num_workers', 4),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'] * 2, shuffle=False,
        collate_fn=collator, num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
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
    model = HomeCreditModel(model_cfg, pretrain_mode=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    # === Load pretrained encoder weights ===
    pretrain_path = cfg.get('init_from_pretrain')
    if pretrain_path:
        pretrain_path = Path(pretrain_path)
        if pretrain_path.exists():
            print(f'=== Loading pretrained encoders from {pretrain_path} ===')
            ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
            state = ckpt['model_state']
            # Filter to keys that match (interactive/top are new in finetune and won't be in pretrain)
            own_state = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            for k, v in state.items():
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)
            model.load_state_dict(own_state)
            print(f'  loaded {len(loaded_keys)} tensors, skipped {len(skipped_keys)}')
            if skipped_keys[:5]:
                print(f'  skip samples: {skipped_keys[:5]}')
        else:
            print(f'  WARNING: pretrain path {pretrain_path} does not exist; training from scratch')

    # === Optimizer ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    total_steps = len(train_loader) * cfg['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], total_steps=total_steps, pct_start=0.05,
    )

    # === Train ===
    log_path = out_dir / 'finetune.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    pos_weight = cfg.get('pos_weight', 11.4)
    log(f'=== Training: {cfg["epochs"]} epochs x {len(train_loader)} steps = {total_steps} steps ===')
    log(f'    pos_weight={pos_weight}')

    best_auc = -1.0
    best_path = out_dir / 'best_model.pt'
    global_step = 0
    start_time = time.time()
    eval_per_steps = cfg.get('eval_per_epoch_steps', 2000)

    history = []

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss = finetune_loss(logits, batch['target'], pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg.get('log_every', 50) == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                eta_min = (total_steps - global_step) / max(sps, 1e-9) / 60
                cur_lr = scheduler.get_last_lr()[0]
                log(f'  [ep{epoch} step {global_step}/{total_steps}] '
                    f'loss={loss.item():.4f}  lr={cur_lr:.2e}  sps={sps:.1f}  eta={eta_min:.1f}m')

            if global_step % eval_per_steps == 0:
                metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
                log(f'  [ep{epoch} step {global_step}] VAL  loss={metrics["loss"]:.4f}  '
                    f'auc={metrics["auc"]:.4f}  ks={metrics["ks"]:.4f}')
                history.append({'step': global_step, 'epoch': epoch, **metrics})
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    torch.save({
                        'model_state': model.state_dict(),
                        'config': {**cfg, 'vocab_sizes': vocab_sizes},
                        'epoch': epoch,
                        'step': global_step,
                        'metrics': metrics,
                    }, best_path)
                    log(f'    ↑ new best AUC={best_auc:.4f}, saved to {best_path}')
                model.train()

        log(f'=== Epoch {epoch} done: avg_train_loss={epoch_loss/max(n_batches,1):.4f} ===')

    # Final eval
    final_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
    log(f'\n=== Final VAL metrics: loss={final_metrics["loss"]:.4f}  '
        f'auc={final_metrics["auc"]:.4f}  ks={final_metrics["ks"]:.4f} ===')

    # Save final + reload best for predictions
    final_path = out_dir / 'final_model.pt'
    torch.save({
        'model_state': model.state_dict(),
        'config': {**cfg, 'vocab_sizes': vocab_sizes},
        'epoch': cfg['epochs'],
        'step': global_step,
        'metrics': final_metrics,
    }, final_path)

    # Predict using best model
    if best_path.exists():
        log(f'\n=== Reloading best model from {best_path} ===')
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state'])
        log(f'  best was at step {best_ckpt["step"]} with AUC={best_ckpt["metrics"]["auc"]:.4f}')

    model.eval()
    all_logits = []
    all_targets = []
    all_sk_ids = []
    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(batch_gpu).cpu()
            all_logits.append(logits)
            all_targets.append(batch['target'])
    logits = torch.cat(all_logits).squeeze(-1)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits)

    import csv
    pred_path = out_dir / 'val_predictions.csv'
    with open(pred_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sk_id_curr', 'target', 'prediction'])
        # val_loader iterates in original order of samples_val.pkl, which is the
        # same order as split.json's val_sk_id_curr
        with open('data/home-credit/processed/split.json') as sj:
            split = json.load(sj)
        val_ids = split['val_sk_id_curr']
        assert len(val_ids) == len(probs), f'{len(val_ids)} vs {len(probs)}'
        for sk_id, t, p in zip(val_ids, targets.tolist(), probs.tolist()):
            w.writerow([sk_id, int(t), float(p)])
    log(f'  wrote predictions to {pred_path}')

    # Final results
    eval_results = {
        'best_val_auc': best_auc,
        'final_val_auc': final_metrics['auc'],
        'final_val_ks': final_metrics['ks'],
        'final_val_loss': final_metrics['loss'],
        'history': history,
    }
    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    log(f'\n=== eval_results.json written ===')
    log(f'    best_val_auc={best_auc:.4f}  final_val_auc={final_metrics["auc"]:.4f}  '
        f'ks={final_metrics["ks"]:.4f}')

    log_file.close()
    print(f'\n=== Finetune done. Total time: {(time.time()-start_time)/60:.1f} min ===')


if __name__ == '__main__':
    main()
