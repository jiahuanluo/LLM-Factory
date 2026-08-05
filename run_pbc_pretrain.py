"""Pretrain PbcCreditModel via HF Trainer (multi-GPU via torchrun).

Usage:
  python run_pbc_pretrain.py configs/pbc_pretrain.yaml
  torchrun --nproc_per_node=8 run_pbc_pretrain.py configs/pbc_pretrain.yaml
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import PbcDataset
from pbc_credit.fields import (
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS,
    SUMMARY_TABLES, OBLIGATIONS_CAT_FIELDS,
    PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE, OBLIGATION_TYPE_VOCAB_SIZE,
)
from pbc_credit.losses import pretrain_loss, EMANormalizer
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig


# ============================================================
# Arguments
# ============================================================

@dataclass
class PbcModelArguments:
    d: int = field(default=256)
    n_heads: int = field(default=8)
    n_layers: int = field(default=4)
    top_hidden: int = field(default=512, metadata={"help": "顶层 trunk head 隐藏维"})
    top_n_layers: int = field(default=2, metadata={"help": "顶层 trunk transformer 层数"})
    top_n_heads: int = field(default=8, metadata={"help": "顶层 trunk attention 头数"})
    dropout: float = field(default=0.1)
    mask_ratio: float = field(default=0.15)
    ema_decay: float = field(default=0.99, metadata={"help": "EMA decay for branch loss normalizer"})
    mask_seed: int = field(default=12345, metadata={"help": "Fixed seed for val mask reproducibility"})
    contrastive_weight: float = field(default=0.1, metadata={"help": "Weight for cross-view consistency loss (0 disables)"})


@dataclass
class PbcDataArguments:
    pretrain_samples: list[str] = field(default_factory=list, metadata={"nargs": "+", "help": "JSONL files"})
    val_samples: str | None = field(default=None)
    cat_vocab: str = field(default=None)


# ============================================================
# Collator + helpers
# ============================================================

class MaskingCollator:
    def __init__(self, mask_ratio: float = 0.15):
        self.base = PbcCollator()
        self.mask_ratio = mask_ratio

    def __call__(self, samples):
        batch = self.base(samples)
        batch.pop('target', None)
        return add_masks_to_batch(batch, mask_ratio=self.mask_ratio)


def build_model_cfg(model_args: PbcModelArguments, vocab: dict) -> PbcCreditModelConfig:
    user_tables = {}
    for _path, t in USER_CAT_FIELDS:
        if t:
            user_tables[t] = len(vocab.get('user', {}).get(t, {'<UNK>': 0})) + 1
    summary_tables = {}
    for _n, _l, _nf, cats in SUMMARY_TABLES:
        for _f, t in cats:
            if t and t not in summary_tables:
                summary_tables[t] = len(vocab.get('summary', {}).get(t, {'<UNK>': 0})) + 1
    acc_tables = {}
    for _f, t in ACCOUNT_CAT_FIELDS:
        if t:
            acc_tables[t] = len(vocab.get('account', {}).get(t, {'<UNK>': 0})) + 1
    q_tables = {}
    for _f, t in QUERY_CAT_FIELDS:
        if t:
            q_tables[t] = len(vocab.get('query', {}).get(t, {'<UNK>': 0})) + 1
    obl_tables = {}
    for _ot, _f, t in OBLIGATIONS_CAT_FIELDS:
        if t and t not in obl_tables:
            obl_tables[t] = len(vocab.get('obligation', {}).get(t, {'<UNK>': 0})) + 1

    n_sum_num = 0
    for _name, is_list, nums, _c in SUMMARY_TABLES:
        n_sum_num += (1 if is_list else 0) + len(nums)

    return PbcCreditModelConfig(
        d=model_args.d,
        n_heads=model_args.n_heads,
        n_layers=model_args.n_layers,
        top_hidden=model_args.top_hidden,
        top_n_layers=model_args.top_n_layers,
        top_n_heads=model_args.top_n_heads,
        dropout=model_args.dropout,
        user_numeric_dim=18,                          # 13 base (含 3 稳定性) + score 5
        user_cat_tables=user_tables,
        summary_numeric_dim=n_sum_num,
        summary_cat_tables=summary_tables,
        account_numeric_dim=15,                       # 8 base + specialTrades 2 + age 1 + 4 ratio
        account_cat_tables=acc_tables,
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
        query_numeric_dim=1,
        query_cat_tables=q_tables,
        public_type_vocab_size=PUBLIC_TYPE_VOCAB_SIZE,
        obligation_type_vocab_size=OBLIGATION_TYPE_VOCAB_SIZE,
        obligation_cat_tables=obl_tables,
    )


@torch.no_grad()
def evaluate_pretrain(model, val_loader, device, mask_seed: int = 12345) -> dict:
    """Run reconstruction eval on val; fixed mask seed → cross-step comparable."""
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
    metrics = {f'{k}': v / max(branch_n[k], 1) for k, v in branch_sum.items()}
    paystate_losses = [metrics[k] for k in branch_sum if 'paystate' in k]
    if paystate_losses:
        metrics['paystate_ppl'] = float(math.exp(sum(paystate_losses) / len(paystate_losses)))
    metrics['total'] = sum(metrics[k] for k in branch_sum)
    model.train()
    return metrics


# ============================================================
# Trainer subclass
# ============================================================

class PbcPreTrainer(Trainer):
    """HF Trainer for PBC pretrain: dict-batch forward + EMA loss normalizer
    + custom eval (per-branch reconstruction loss + paystate perplexity)
    + P0 B1: contrastive consistency (double-forward, dropout as view augmentation)."""

    def __init__(self, *args, normalizer: EMANormalizer | None = None,
                 mask_seed: int = 12345, contrastive_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = normalizer or EMANormalizer()
        self.mask_seed = mask_seed
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # View 1: reconstruction
        outputs1 = model(inputs)
        loss1, _ = pretrain_loss(outputs1, normalizer=self.normalizer)

        # View 2: 同一 inputs 再 forward 一次；dropout 提供随机性作为不同 view
        # 加 contrastive consistency: 两个 view 的 summary_emb 应趋近
        if self.contrastive_weight > 0 and self.model.training:
            outputs2 = model(inputs)
            loss2, _ = pretrain_loss(outputs2, normalizer=self.normalizer)
            emb1 = outputs1.get('summary_emb')
            emb2 = outputs2.get('summary_emb')
            if emb1 is not None and emb2 is not None:
                # 排除 NaN / 全 0（如全空 batch）
                if emb1.shape[0] > 0 and emb2.shape[0] > 0:
                    cos = F.cosine_similarity(emb1, emb2, dim=-1).clamp(-1.0, 1.0)
                    consistency = (1.0 - cos).mean()
                    loss = (loss1 + loss2) * 0.5 + self.contrastive_weight * consistency
                else:
                    loss = (loss1 + loss2) * 0.5
            else:
                loss = (loss1 + loss2) * 0.5
        else:
            loss = loss1

        if return_outputs:
            return loss, outputs1
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            return {}
        loader = self.get_eval_dataloader(eval_ds)
        metrics = evaluate_pretrain(self.model, loader, self.args.device,
                                     mask_seed=self.mask_seed)
        out = {f'{metric_key_prefix}_{k}': v for k, v in metrics.items()}
        self.log(out)
        return out

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        target_dir = output_dir or self.args.output_dir
        os.makedirs(target_dir, exist_ok=True)
        # Encoder-only ckpt（strip mask heads），保持 finetune 加载兼容
        state = {
            k: v for k, v in self.model.state_dict().items()
            if not k.endswith('_mask_head.weight') and not k.endswith('_mask_head.bias')
        }
        torch.save({
            'model_state': state,
            'model_cfg': self.model.config.__dict__,
        }, os.path.join(target_dir, 'encoder_state.pt'))


# ============================================================
# Main
# ============================================================

def main():
    parser = HfArgumentParser((PbcModelArguments, PbcDataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        model_args, data_args, training_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(data_args.cat_vocab) as f:
        vocab = json.load(f)
    model_cfg = build_model_cfg(model_args, vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PbcCreditModel(model_cfg, pretrain_mode=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'=== Model: {n_params:,} params ({n_params/1e6:.2f}M), device={device} ===')

    # === Datasets ===
    print('=== Loading datasets ===')
    paths = [p for p in data_args.pretrain_samples if p]
    datasets = []
    for p in paths:
        ds = PbcDataset(p, pretrain_mode=True)
        print(f'  {p}: {len(ds):,} samples')
        datasets.append(ds)
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f'  train combined: {len(train_ds):,} samples')
    eval_ds = None
    if data_args.val_samples:
        eval_ds = PbcDataset(data_args.val_samples, pretrain_mode=True)
        print(f'  val: {len(eval_ds):,} samples')

    collator = MaskingCollator(mask_ratio=model_args.mask_ratio)

    trainer = PbcPreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        normalizer=EMANormalizer(decay=model_args.ema_decay),
        mask_seed=model_args.mask_seed,
        contrastive_weight=model_args.contrastive_weight,
    )

    # === Train ===
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

    # === Final eval ===
    if eval_ds is not None:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)


if __name__ == '__main__':
    main()
