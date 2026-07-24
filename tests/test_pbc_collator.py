"""PBC collator 单元测试：padding 正确性。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator, pad_2d


def _fake_sample(n_d1, n_q, n_p):
    """构造一个最小合法的 sample。ACCOUNT_NUMERIC_FIELDS=8, ACCOUNT_CAT_FIELDS=6。"""
    return {
        'user_numeric': torch.zeros(10),
        'user_cat_ids': torch.zeros(11, dtype=torch.long),
        'user_cat_mask': torch.ones(11, dtype=torch.long),
        'summary_numeric': torch.zeros(36),
        'summary_cat_ids': torch.zeros(10, dtype=torch.long),
        'summary_cat_mask': torch.zeros(10, dtype=torch.long),
        'd1_numeric': torch.zeros(n_d1, 8),
        'd1_cat_ids': torch.zeros(n_d1, 6, dtype=torch.long),
        'd1_paystate': torch.zeros(n_d1, 60, dtype=torch.long),
        'd1_mask': torch.ones(n_d1, dtype=torch.long),
        'r1_numeric': torch.zeros(0, 8),
        'r1_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r1_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r1_mask': torch.zeros(0, dtype=torch.long),
        'r2_numeric': torch.zeros(0, 8),
        'r2_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r2_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r2_mask': torch.zeros(0, dtype=torch.long),
        'r3_numeric': torch.zeros(0, 8),
        'r3_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r3_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r3_mask': torch.zeros(0, dtype=torch.long),
        'r4_numeric': torch.zeros(0, 8),
        'r4_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r4_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r4_mask': torch.zeros(0, dtype=torch.long),
        'query_numeric': torch.zeros(n_q, 1),
        'query_cat_ids': torch.zeros(n_q, 2, dtype=torch.long),
        'query_mask': torch.ones(n_q, dtype=torch.long),
        'public_numeric': torch.zeros(n_p, 2),
        'public_cat_ids': torch.zeros(n_p, 1, dtype=torch.long),
        'public_mask': torch.ones(n_p, dtype=torch.long),
        'target': torch.tensor([0.0]),
    }


def test_pad_2d_basic():
    """pad_2d 对变长 [N_i, F] 正确填充。"""
    ts = [torch.zeros(3, 5), torch.zeros(5, 5), torch.zeros(1, 5)]
    ms = [torch.ones(3), torch.ones(5), torch.ones(1)]
    pt, pm = pad_2d(ts, ms, pad_value=0.0)
    assert pt.shape == (3, 5, 5), pt.shape
    assert pm.shape == (3, 5)
    assert pm[2].sum().item() == 1


def test_collator_pads_variable_accounts():
    """batch 中不同样本 d1 账户数不同，collator 正确 pad。"""
    samples = [_fake_sample(2, 5, 0), _fake_sample(4, 3, 0), _fake_sample(0, 10, 0)]
    batch = PbcCollator()(samples)
    assert batch['d1_numeric'].shape == (3, 4, 8)
    assert batch['d1_mask'].shape == (3, 4)
    assert batch['d1_mask'][2].sum().item() == 0
    assert batch['d1_mask'][0].sum().item() == 2


def test_collator_handles_empty_branch():
    """某分支全空时（如所有样本都无 r1），collator 不崩。"""
    samples = [_fake_sample(2, 5, 0), _fake_sample(3, 2, 0)]
    batch = PbcCollator()(samples)
    assert batch['r1_numeric'].shape == (2, 0, 8)
    assert batch['r1_mask'].shape == (2, 0)


def test_collator_target_stacked():
    samples = [_fake_sample(1, 1, 0), _fake_sample(2, 1, 0)]
    batch = PbcCollator()(samples)
    assert batch['target'].shape == (2,)


if __name__ == '__main__':
    test_pad_2d_basic()
    test_collator_pads_variable_accounts()
    test_collator_handles_empty_branch()
    test_collator_target_stacked()
    print('✓ test_pbc_collator passed')
