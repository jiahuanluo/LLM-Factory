"""Pretrain PbcCreditModel with multi-modal joint mask reconstruction.

Usage:
  python run_pbc_pretrain.py configs/pbc_pretrain.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import PbcDataset
from pbc_credit.fields import (
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS,
    SUMMARY_TABLES, PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE,
)
from pbc_credit.losses import pretrain_loss, EMANormalizer
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MaskingCollator:
    def __init__(self, mask_ratio: float = 0.15):
        self.base = PbcCollator()
        self.mask_ratio = mask_ratio

    def __call__(self, samples):
        batch = self.base(samples)
        batch.pop('target', None)
        return add_masks_to_batch(batch, mask_ratio=self.mask_ratio)


@torch.no_grad()
def evaluate_pretrain(model, val_loader, device, mask_seed: int = 12345) -> dict:
    """在 val 上跑 reconstruction，返回每分支 loss + paystate perplexity。

    mask_seed 固定 → 每次 eval 的 mask pattern 相同 → 指标跨 step 可直接比较。
    """
    model.eval()
    torch.manual_seed(mask_seed)
    branch_sum: dict[str, float] = {}
    branch_n: dict[str, int] = {}
    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        for k in list(out.keys()):
            if not k.endswith('_pred'):
                continue
            prefix = k[:-5]
            tgt_key = f'{prefix}_target'
            if tgt_key not in out:
                continue
            pred, tgt = out[k], out[tgt_key]
            if pred.shape[0] == 0:
                continue
            if 'paystate' in prefix:
                loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                       tgt.reshape(-1).long(), ignore_index=0)
            else:
                loss = F.mse_loss(pred.float(), tgt.float())
            branch_sum[prefix] = branch_sum.get(prefix, 0.0) + loss.item()
            branch_n[prefix] = branch_n.get(prefix, 0) + 1
    metrics = {f'val/{k}': v / max(branch_n[k], 1) for k, v in branch_sum.items()}
    paystate_losses = [metrics[f'val/{k}'] for k in branch_sum if 'paystate' in k]
    if paystate_losses:
        metrics['val/paystate_ppl'] = float(math.exp(sum(paystate_losses) / len(paystate_losses)))
    metrics['val/total'] = sum(metrics[f'val/{k}'] for k in branch_sum)
    model.train()
    return metrics


def build_model_cfg(cfg: dict, vocab: dict) -> PbcCreditModelConfig:
    # user cat tables: {field_name: vocab_size}
    user_tables = {}
    for _path, t in USER_CAT_FIELDS:
        if t:
            user_tables[t] = len(vocab.get('user', {}).get(t, {'<UNK>': 0})) + 1
    # summary cat tables
    summary_tables = {}
    for _n, _l, _nf, cats in SUMMARY_TABLES:
        for _f, t in cats:
            if t and t not in summary_tables:
                summary_tables[t] = len(vocab.get('summary', {}).get(t, {'<UNK>': 0})) + 1
    # account cat tables
    acc_tables = {}
    for _f, t in ACCOUNT_CAT_FIELDS:
        if t:
            acc_tables[t] = len(vocab.get('account', {}).get(t, {'<UNK>': 0})) + 1
    # query cat tables
    q_tables = {}
    for _f, t in QUERY_CAT_FIELDS:
        if t:
            q_tables[t] = len(vocab.get('query', {}).get(t, {'<UNK>': 0})) + 1

    # summary numeric 维度
    n_sum_num = 0
    for _name, is_list, nums, _c in SUMMARY_TABLES:
        n_sum_num += (1 if is_list else 0) + len(nums)

    return PbcCreditModelConfig(
        d=cfg.get('d', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 2),
        dropout=cfg.get('dropout', 0.1),
        user_numeric_dim=10,
        user_cat_tables=user_tables,
        summary_numeric_dim=n_sum_num,
        summary_cat_tables=summary_tables,
        account_numeric_dim=8,  # len(ACCOUNT_NUMERIC_FIELDS)
        account_cat_tables=acc_tables,
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
        query_numeric_dim=1,
        query_cat_tables=q_tables,
        public_type_vocab_size=PUBLIC_TYPE_VOCAB_SIZE,
    )


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
    paths = [p for p in cfg['pretrain_samples'] if p]
    datasets = []
    for p in paths:
        ds = PbcDataset(p, pretrain_mode=True)
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
        num_workers=cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=True,
    )

    # === Val loader（独立 split，固定 mask seed 跨 step 可比） ===
    val_loader = None
    if cfg.get('val_samples'):
        val_ds = PbcDataset(cfg['val_samples'], pretrain_mode=True)
        print(f'  val: {len(val_ds):,} samples')
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.get('eval_batch_size', cfg['batch_size']) * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.get('num_workers', 0),
            pin_memory=True,
        )

    # === Model ===
    with open(cfg['cat_vocab']) as f:
        vocab = json.load(f)
    model_cfg = build_model_cfg(cfg, vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PbcCreditModel(model_cfg, pretrain_mode=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    # === Loss normalizer（分支 EMA 权重，让 accounts(10 项) 不主导 query/public/summary） ===
    normalizer = EMANormalizer(decay=cfg.get('ema_decay', 0.99))

    # === Optim ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.98), eps=1e-6,
    )
    total_steps = len(loader) * cfg['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'], total_steps=total_steps, pct_start=0.05,
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
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        epoch_components: dict = {}
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(batch)
            loss, log_dict = pretrain_loss(out, normalizer=normalizer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            for k, v in log_dict.items():
                epoch_components[k] = epoch_components.get(k, 0.0) + v
            n_batches += 1
            global_step += 1

            if global_step % cfg.get('log_every', 20) == 0:
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                eta_min = (total_steps - global_step) / max(sps, 1e-9) / 60
                cur_lr = scheduler.get_last_lr()[0]
                comp_str = ' '.join(
                    f"{k.split('/')[-1]}={v:.3f}"
                    for k, v in log_dict.items()
                    if k != 'loss/total' and k.startswith('loss/')
                )
                log(f'  [ep{epoch} step {global_step}/{total_steps}] '
                    f'loss={loss.item():.4f}  {comp_str}  '
                    f'lr={cur_lr:.2e}  sps={sps:.1f}  eta={eta_min:.1f}m')

            # === Val eval ===
            if val_loader is not None and global_step % cfg.get('eval_per_steps', 50) == 0:
                metrics = evaluate_pretrain(model, val_loader, device)
                comp_str = ' '.join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in metrics.items()
                    if k != 'val/total'
                )
                log(f'  [val step {global_step}] total={metrics["val/total"]:.4f}  {comp_str}')
                if metrics['val/total'] < best_val_loss:
                    best_val_loss = metrics['val/total']
                    state = {
                        k: v for k, v in model.state_dict().items()
                        if not k.endswith('_mask_head.weight') and not k.endswith('_mask_head.bias')
                    }
                    torch.save({
                        'model_state': state,
                        'config': {**cfg},
                        'model_cfg': model_cfg.__dict__,
                        'val_metrics': metrics,
                        'step': global_step,
                    }, out_dir / 'best_encoder.pt')
                    log(f'    ★ new best val_total={best_val_loss:.4f} saved → best_encoder.pt')
                model.train()

        avg = epoch_loss / max(n_batches, 1)
        avg_comp = {k: v / max(n_batches, 1) for k, v in epoch_components.items()}
        comp_str = ' '.join(f"{k.split('/')[-1]}={v:.4f}"
                            for k, v in avg_comp.items()
                            if k != 'loss/total' and k.startswith('loss/'))
        log(f'=== Epoch {epoch} done: avg_loss={avg:.4f}  {comp_str} ===')

        # save encoder
        if (epoch + 1) % cfg.get('save_every_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            ckpt = out_dir / f'encoder_state_ep{epoch+1}.pt'
            state = {
                k: v for k, v in model.state_dict().items()
                if not k.endswith('_mask_head.weight') and not k.endswith('_mask_head.bias')
            }
            torch.save({
                'model_state': state,
                'config': {**cfg},
                'model_cfg': model_cfg.__dict__,
            }, ckpt)
            torch.save({
                'model_state': state,
                'config': {**cfg},
                'model_cfg': model_cfg.__dict__,
            }, out_dir / 'encoder_state.pt')
            log(f'  saved {ckpt}')

    log_file.close()
    print(f'\n=== Pretrain done. Total: {(time.time()-start_time)/60:.1f} min ===')


if __name__ == '__main__':
    main()
