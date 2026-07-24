"""Test G004: masking + pretrain-mode model forward on real val data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, Subset

from home_credit.dataset import HomeCreditDataset
from home_credit.collator import HomeCreditCollator
from home_credit.masking import (
    generate_branch_mask, generate_text_mask, add_masks_to_batch,
)
from home_credit.model import HomeCreditModel, HomeCreditModelConfig
from test_home_credit_e2e import load_vocab


def test_generate_branch_mask_ratio():
    """Check actual mask ratio is close to 0.15."""
    torch.manual_seed(42)
    pad_mask = torch.ones(16, 100, dtype=torch.long)
    # Pad some positions
    pad_mask[:, 80:] = 0
    masked = generate_branch_mask(pad_mask, mask_ratio=0.15)
    assert masked.dtype == torch.bool
    assert masked.shape == pad_mask.shape
    # Should never mask padded positions
    assert (~masked[:, 80:]).all()
    # Mask ratio among valid positions should be close to 0.15
    actual_ratio = masked[:, :80].float().mean().item()
    assert 0.10 < actual_ratio < 0.20, f'expected ~0.15, got {actual_ratio:.3f}'
    print(f'  branch mask ratio: {actual_ratio:.3f}  OK')


def test_generate_text_mask_specials():
    """Check text mask never touches special tokens."""
    torch.manual_seed(0)
    input_ids = torch.randint(104, 30000, (8, 64))
    # Place specials: 0=pad, 100=UNK, 101=CLS, 102=SEP, 103=MASK
    input_ids[:, 0] = 101
    input_ids[:, -1] = 102
    input_ids[0, 5] = 100
    input_ids[1, 10] = 103
    input_ids[:, 40:] = 0  # padded
    attention_mask = (input_ids != 0).long()

    masked = generate_text_mask(input_ids, attention_mask, mask_ratio=0.20)
    assert masked.dtype == torch.bool
    # No special token should be masked
    for sid in [100, 101, 102, 103]:
        assert (~masked[input_ids == sid]).all(), f'special token {sid} got masked'
    # No padded position
    assert (~masked[attention_mask == 0]).all()
    print(f'  text mask special-token avoidance OK')


def test_pretrain_forward():
    """End-to-end: collator + add_masks_to_batch + pretrain model."""
    ds = HomeCreditDataset('data/home-credit/processed/samples_val.pkl')
    loader = DataLoader(
        Subset(ds, list(range(32))),
        batch_size=16, collate_fn=HomeCreditCollator(), shuffle=False,
    )

    vocab = load_vocab()
    cfg = HomeCreditModelConfig(
        user_cat_field_sizes=vocab['user'],
        bureau_cat_field_sizes=vocab['bureau'],
        prev_cat_field_sizes=vocab['prev'],
        card_cat_field_sizes=vocab['card'],
    )
    model = HomeCreditModel(cfg, pretrain_mode=True)
    print(f'  pretrain model params: {sum(p.numel() for p in model.parameters()):,}')

    batch = next(iter(loader))
    batch_masked = add_masks_to_batch(batch, mask_ratio=0.15)

    # Sanity: mask ratios
    for branch in ['bureau', 'prev', 'card']:
        mp = batch_masked[f'{branch}_masked_pos']
        valid = batch_masked[f'{branch}_mask']
        if valid.sum() > 0:
            ratio = (mp & (valid == 1)).sum().item() / valid.sum().item()
            print(f'  {branch} mask ratio: {ratio:.3f}  (valid rows: {valid.sum().item()})')
    text_mp = batch_masked['text_masked_pos']
    text_valid = batch_masked['text_attention_mask']
    text_ratio = (text_mp & (text_valid == 1)).sum().item() / text_valid.sum().item()
    print(f'  text mask ratio: {text_ratio:.3f}')

    model.eval()
    with torch.no_grad():
        out_preview = model(batch_masked)

    # Backward requires train mode + grad
    model.train()
    out = model(batch_masked)

    print(f'\n  pretrain output keys: {sorted(out.keys())}')
    for k in out:
        if isinstance(out[k], torch.Tensor):
            print(f'    {k}: shape={tuple(out[k].shape)} mean={out[k].float().mean().item():.3f}')

    # Verify shapes: predictions should match targets
    for branch in ['bureau', 'prev', 'card']:
        if f'{branch}_pred' in out:
            assert out[f'{branch}_pred'].shape == out[f'{branch}_target'].shape, \
                f'{branch} pred/target shape mismatch'

    # Test loss computation
    losses = {}
    if 'bureau_pred' in out:
        losses['bureau'] = torch.nn.functional.mse_loss(out['bureau_pred'], out['bureau_target'])
    if 'prev_pred' in out:
        losses['prev'] = torch.nn.functional.mse_loss(out['prev_pred'], out['prev_target'])
    if 'card_pred' in out:
        losses['card'] = torch.nn.functional.mse_loss(out['card_pred'], out['card_target'])
    if 'text_logits' in out:
        losses['text'] = torch.nn.functional.cross_entropy(
            out['text_logits'], out['text_target'],
        )
    print(f'\n  losses: {[(k, f"{v.item():.3f}") for k, v in losses.items()]}')

    total = sum(losses.values())
    total.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f'  total loss: {total.item():.3f}  grad_norm: {grad_norm:.2f}')

    print('\nAll pretrain checks passed.')


if __name__ == '__main__':
    print('Testing generate_branch_mask ratio:')
    test_generate_branch_mask_ratio()
    print('\nTesting generate_text_mask specials:')
    test_generate_text_mask_specials()
    print('\nTesting pretrain forward E2E:')
    test_pretrain_forward()
