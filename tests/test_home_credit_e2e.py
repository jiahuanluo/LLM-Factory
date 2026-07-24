"""End-to-end shape check: real data → collator → model (finetune mode)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import DataLoader

from home_credit.dataset import HomeCreditDataset
from home_credit.collator import HomeCreditCollator
from home_credit.model import HomeCreditModel, HomeCreditModelConfig


def load_vocab():
    with open('data/home-credit/processed/cat_vocab.json') as f:
        v = json.load(f)
    # v: {table: {field: {value: id}}}, id starts at 1
    # Embedding size per field = max id + 2 (id 0 reserved for <UNK>, +1 because embedding is [0, max_id])
    sizes = {}
    for table, fields in v.items():
        sizes[table] = {
            f: (max(field_map.values()) + 2 if field_map else 2)
            for f, field_map in fields.items()
        }
    return sizes


def main():
    print('Loading val set (smallest) for fast iteration...')
    ds = HomeCreditDataset('data/home-credit/processed/samples_val.pkl')
    print(f'  val samples: {len(ds):,}')

    # Subsample 64 for one batch
    small = torch.utils.data.Subset(ds, list(range(64)))
    loader = DataLoader(small, batch_size=16, collate_fn=HomeCreditCollator(), shuffle=False)

    # Load actual vocab sizes for model config
    vocab_sizes = load_vocab()
    cfg = HomeCreditModelConfig(
        user_cat_field_sizes=vocab_sizes['user'],
        bureau_cat_field_sizes=vocab_sizes['bureau'],
        prev_cat_field_sizes=vocab_sizes['prev'],
        card_cat_field_sizes=vocab_sizes['card'],
    )
    print(f'\nModel config:')
    print(f'  d={cfg.d}, heads={cfg.n_heads}, layers={cfg.n_layers}')
    print(f'  user_cat fields: {len(cfg.user_cat_field_sizes)}, total emb params: {sum((s+1)*cfg.user_cat_embed_dim for s in cfg.user_cat_field_sizes.values()):,}')

    model = HomeCreditModel(cfg, pretrain_mode=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  total params: {n_params:,} ({n_params/1e6:.2f}M)')

    print('\nForward on 1 batch (B=16)...')
    batch = next(iter(loader))
    print(f'  batch keys: {sorted(batch.keys())}')
    print(f'  shapes: user_numeric={tuple(batch["user_numeric"].shape)}, '
          f'bureau_num={tuple(batch["bureau_numeric"].shape)}, '
          f'prev_num={tuple(batch["prev_numeric"].shape)}, '
          f'card_num={tuple(batch["card_numeric"].shape)}, '
          f'text={tuple(batch["text_input_ids"].shape)}')

    model.eval()
    with torch.no_grad():
        logit = model(batch)
    print(f'  logit shape: {tuple(logit.shape)}  (expected: (16, 1))')
    assert logit.shape == (16, 1), f'expected (16, 1), got {logit.shape}'

    print('\nChecking differentiability (one backward step)...')
    model.train()
    logit = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit.squeeze(-1), batch['target'].float(),
    )
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f'  loss={loss.item():.4f}  grad_norm={grad_norm:.2f}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
