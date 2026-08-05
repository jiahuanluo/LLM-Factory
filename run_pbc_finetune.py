"""Finetune PbcCreditModel via HF Trainer (multi-GPU via torchrun).

Usage:
  python run_pbc_finetune.py configs/pbc_finetune.yaml
  torchrun --nproc_per_node=8 run_pbc_finetune.py configs/pbc_finetune.yaml
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import PbcDataset
from pbc_credit.losses import finetune_loss, compute_auc, compute_ks
from pbc_credit.model import PbcCreditModel
from run_pbc_pretrain import build_model_cfg


# ============================================================
# Arguments
# ============================================================

@dataclass
class PbcFinetuneModelArguments:
    d: int = field(default=256)
    n_heads: int = field(default=8)
    n_layers: int = field(default=4)
    top_hidden: int = field(default=512, metadata={"help": "顶层 trunk head 隐藏维"})
    top_n_layers: int = field(default=2, metadata={"help": "顶层 trunk transformer 层数"})
    top_n_heads: int = field(default=8, metadata={"help": "顶层 trunk attention 头数"})
    dropout: float = field(default=0.1)
    init_from_pretrain: str | None = field(default=None,
        metadata={"help": "Path to encoder_state.pt from pretrain"})
    pos_weight: float = field(default=4.0)


@dataclass
class PbcDataArguments:
    train_samples: str = field(default=None)
    val_samples: str = field(default=None)
    cat_vocab: str = field(default=None)


# ============================================================
# Eval helper (reused from old script)
# ============================================================

@torch.no_grad()
def evaluate_finetune(model, val_loader, device, pos_weight: float) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(batch)
        loss = finetune_loss(logits, batch['target'], pos_weight=pos_weight)
        probs = torch.sigmoid(logits.squeeze(-1))
        all_probs.append(probs.cpu())
        all_labels.append(batch['target'].cpu())
        total_loss += loss.item() * len(probs)
        n += len(probs)
    probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    model.train()
    return {
        'loss': total_loss / max(n, 1),
        'auc': compute_auc(probs, labels),
        'ks': compute_ks(probs, labels),
    }


# ============================================================
# Trainer subclass
# ============================================================

class PbcFineTrainer(Trainer):
    def __init__(self, *args, pos_weight: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(inputs)
        target = inputs['target']
        loss = finetune_loss(logits, target, pos_weight=self.pos_weight)
        if return_outputs:
            return loss, logits
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            return {}
        loader = self.get_eval_dataloader(eval_ds)
        metrics = evaluate_finetune(self.model, loader, self.args.device, self.pos_weight)
        out = {f'{metric_key_prefix}_{k}': v for k, v in metrics.items()}
        self.log(out)
        return out


# ============================================================
# Main
# ============================================================

def main():
    parser = HfArgumentParser((PbcFinetuneModelArguments, PbcDataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(data_args.cat_vocab) as f:
        vocab = json.load(f)
    model_cfg = build_model_cfg(model_args, vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PbcCreditModel(model_cfg, pretrain_mode=False).to(device)

    # Load pretrained encoder
    if model_args.init_from_pretrain and Path(model_args.init_from_pretrain).exists():
        ckpt = torch.load(model_args.init_from_pretrain, map_location=device, weights_only=False)
        state = ckpt['model_state']
        own = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(loaded)
        model.load_state_dict(own)
        print(f'=== Loaded {len(loaded)}/{len(own)} tensors from {model_args.init_from_pretrain} ===')
    else:
        print('=== No pretrain init; training from scratch ===')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    # === Datasets ===
    train_ds = PbcDataset(data_args.train_samples, pretrain_mode=False)
    eval_ds = PbcDataset(data_args.val_samples, pretrain_mode=False)
    print(f'  train: {len(train_ds):,}')
    print(f'  val:   {len(eval_ds):,}')

    trainer = PbcFineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PbcCollator(),
        pos_weight=model_args.pos_weight,
    )

    last_ckpt = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        from transformers.trainer_utils import get_last_checkpoint
        try:
            last_ckpt = get_last_checkpoint(training_args.output_dir)
            if last_ckpt:
                print(f'=== Resuming from {last_ckpt} ===')
        except Exception:
            pass

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()
    trainer.log_metrics('train', train_result.metrics)
    trainer.save_metrics('train', train_result.metrics)
    trainer.save_state()

    if eval_ds is not None:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == '__main__':
    main()
