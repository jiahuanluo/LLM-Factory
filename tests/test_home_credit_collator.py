"""Sanity test for HomeCreditCollator with mock samples. No data dependency."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from home_credit.collator import HomeCreditCollator, pad_2d, pad_1d


def make_mock_sample(sk, n_bureau, n_prev, n_card, l_text, target=True):
    """Build a mock sample matching sample_builder output format."""
    s = {
        'sk_id_curr': sk,
        'user_numeric': torch.randn(62),
        'user_cat_ids': torch.randint(0, 10, (16,)),
        'user_cat_mask': torch.ones(16, dtype=torch.long),
        'query_features': torch.randn(6, 1),
        'query_mask': torch.ones(6, dtype=torch.long),
        'bureau_numeric': torch.randn(n_bureau, 12),
        'bureau_cat_ids': torch.randint(0, 5, (n_bureau, 3)),
        'bureau_mask': torch.ones(n_bureau, dtype=torch.long),
        'prev_numeric': torch.randn(n_prev, 19),
        'prev_cat_ids': torch.randint(0, 10, (n_prev, 16)),
        'prev_mask': torch.ones(n_prev, dtype=torch.long),
        'card_numeric': torch.randn(n_card, 20),
        'card_cat_ids': torch.randint(0, 3, (n_card, 1)),
        'card_mask': torch.ones(n_card, dtype=torch.long),
        'text_input_ids': torch.randint(0, 1000, (l_text,)),
        'text_attention_mask': torch.ones(l_text, dtype=torch.long),
    }
    if target:
        s['target'] = torch.tensor([sk % 2], dtype=torch.long)
    return s


def test_pad_2d_basic():
    a = torch.randn(3, 5)
    b = torch.randn(1, 5)
    masks = [torch.ones(3, dtype=torch.long), torch.ones(1, dtype=torch.long)]
    padded, padded_mask = pad_2d([a, b], masks)
    assert padded.shape == (2, 3, 5), f'expected (2,3,5), got {padded.shape}'
    assert padded_mask.shape == (2, 3)
    # Second sample's last 2 rows should be padding
    assert padded_mask[1, 1] == 0
    assert padded_mask[1, 0] == 1
    print('  pad_2d_basic OK')


def test_pad_2d_all_empty():
    a = torch.zeros(0, 5)
    b = torch.zeros(0, 5)
    masks = [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)]
    padded, padded_mask = pad_2d([a, b], masks)
    assert padded.shape == (2, 0, 5)
    assert padded_mask.shape == (2, 0)
    print('  pad_2d_all_empty OK')


def test_pad_1d_basic():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    padded = pad_1d([a, b])
    assert padded.shape == (2, 3)
    assert padded[1, 2] == 0  # pad
    print('  pad_1d_basic OK')


def test_collator_mixed_batch():
    """Batch with varying branch sizes."""
    samples = [
        make_mock_sample(1, n_bureau=5, n_prev=2, n_card=10, l_text=30),
        make_mock_sample(2, n_bureau=0, n_prev=0, n_card=0, l_text=10),
        make_mock_sample(3, n_bureau=20, n_prev=10, n_card=50, l_text=64),
    ]
    collator = HomeCreditCollator()
    batch = collator(samples)

    B = 3
    assert batch['user_numeric'].shape == (B, 62)
    assert batch['user_cat_ids'].shape == (B, 16)
    assert batch['user_cat_mask'].shape == (B, 16)
    assert batch['query_features'].shape == (B, 6, 1)
    assert batch['query_mask'].shape == (B, 6)

    # bureau: max=20
    assert batch['bureau_numeric'].shape == (B, 20, 12)
    assert batch['bureau_cat_ids'].shape == (B, 20, 3)
    assert batch['bureau_mask'].shape == (B, 20)
    # Sample 1 had 0 bureau rows → all padded
    assert batch['bureau_mask'][1].sum() == 0
    # Sample 0 had 5 bureau rows
    assert batch['bureau_mask'][0].sum() == 5

    # prev: max=10
    assert batch['prev_numeric'].shape == (B, 10, 19)

    # card: max=50
    assert batch['card_numeric'].shape == (B, 50, 20)

    # text: max=64
    assert batch['text_input_ids'].shape == (B, 64)
    assert batch['text_attention_mask'].shape == (B, 64)
    assert batch['text_attention_mask'][0].sum() == 30

    # target
    assert batch['target'].shape == (B,)
    print('  collator_mixed_batch OK')


def test_collator_no_target():
    """Pretrain mode: no target in samples."""
    samples = [
        make_mock_sample(1, 5, 2, 10, 30, target=False),
        make_mock_sample(2, 3, 1, 5, 20, target=False),
    ]
    collator = HomeCreditCollator()
    batch = collator(samples)
    assert 'target' not in batch
    print('  collator_no_target OK')


def test_dataloader_integration():
    """End-to-end: Dataset + DataLoader + Collator."""
    from torch.utils.data import DataLoader
    from home_credit.dataset import HomeCreditDataset

    # Build mock samples list, save as STREAMING pickle (one record per dump)
    import tempfile, pickle
    samples = [make_mock_sample(i, n_bureau=i % 5, n_prev=i % 3, n_card=i % 8, l_text=20 + i % 30)
               for i in range(20)]

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        for s in samples:
            pickle.dump(s, f, protocol=4)
        tmp_path = f.name

    ds = HomeCreditDataset(tmp_path)
    assert len(ds) == 20
    assert ds[0]['sk_id_curr'] == 0

    loader = DataLoader(ds, batch_size=8, collate_fn=HomeCreditCollator(), shuffle=False)
    for i, batch in enumerate(loader):
        assert batch['user_numeric'].shape[0] <= 8
    print('  dataloader_integration OK')

    Path(tmp_path).unlink()


if __name__ == '__main__':
    print('Testing pad helpers:')
    test_pad_2d_basic()
    test_pad_2d_all_empty()
    test_pad_1d_basic()

    print('\nTesting collator:')
    test_collator_mixed_batch()
    test_collator_no_target()

    print('\nTesting integration:')
    test_dataloader_integration()

    print('\nAll tests passed.')
