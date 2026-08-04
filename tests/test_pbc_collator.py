"""PBC collator 单元测试：padding 正确性。

新 fixture 对齐生产数据：6 类账户（含 C1）+ obligations 模态；account_numeric_dim=10；
user_numeric_dim=14。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator, pad_2d


def _fake_sample(n_d1, n_q, n_p, n_o=2, n_c1=0):
    """构造一个最小合法的 sample（含 6 类账户 + obligations）。"""
    sample = {
        'user_numeric': torch.zeros(18),
        'user_cat_ids': torch.zeros(12, dtype=torch.long),
        'user_cat_mask': torch.ones(12, dtype=torch.long),
        'summary_numeric': torch.zeros(36),
        'summary_cat_ids': torch.zeros(10, dtype=torch.long),
        'summary_cat_mask': torch.zeros(10, dtype=torch.long),
        'query_numeric': torch.zeros(n_q, 1),
        'query_cat_ids': torch.zeros(n_q, 2, dtype=torch.long),
        'query_mask': torch.ones(n_q, dtype=torch.long),
        'public_numeric': torch.zeros(n_p, 2),
        'public_cat_ids': torch.zeros(n_p, 1, dtype=torch.long),
        'public_mask': torch.ones(n_p, dtype=torch.long),
        'obligation_numeric': torch.zeros(n_o, 2),
        'obligation_cat_ids': torch.zeros(n_o, 7, dtype=torch.long),
        'obligation_cat_mask': torch.zeros(n_o, 7, dtype=torch.long),
        'obligation_mask': torch.ones(n_o, dtype=torch.long),
        'target': torch.tensor([0.0]),
    }
    # 6 类账户（d1/r1/r2/r3/r4/c1）共享同结构；account_numeric=15, account_cat=9
    for t in ('d1', 'r1', 'r2', 'r3', 'r4', 'c1'):
        n = n_d1 if t == 'd1' else (n_c1 if t == 'c1' else 0)
        sample[f'{t}_numeric'] = torch.zeros(n, 15)
        sample[f'{t}_cat_ids'] = torch.zeros(n, 9, dtype=torch.long)
        sample[f'{t}_paystate'] = torch.zeros(n, 60, dtype=torch.long)
        sample[f'{t}_mask'] = torch.ones(n, dtype=torch.long) if n > 0 else torch.zeros(0, dtype=torch.long)
    return sample


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
    assert batch['d1_numeric'].shape == (3, 4, 15)
    assert batch['d1_mask'].shape == (3, 4)
    assert batch['d1_mask'][2].sum().item() == 0
    assert batch['d1_mask'][0].sum().item() == 2


def test_collator_handles_empty_branch():
    """某分支全空时（如所有样本都无 r1/c1），collator 不崩。"""
    samples = [_fake_sample(2, 5, 0), _fake_sample(3, 2, 0)]
    batch = PbcCollator()(samples)
    assert batch['r1_numeric'].shape == (2, 0, 15)
    assert batch['c1_numeric'].shape == (2, 0, 15)
    assert batch['r1_mask'].shape == (2, 0)


def test_collator_obligations_padded():
    """obligations 变长 pad 正确。"""
    s1 = _fake_sample(1, 1, 0, n_o=3)
    s2 = _fake_sample(1, 1, 0, n_o=5)
    s3 = _fake_sample(1, 1, 0, n_o=0)
    batch = PbcCollator()([s1, s2, s3])
    assert batch['obligation_numeric'].shape == (3, 5, 2)
    assert batch['obligation_cat_ids'].shape == (3, 5, 7)
    assert batch['obligation_mask'][2].sum().item() == 0  # s3 has 0 obligations


def test_collator_target_stacked():
    samples = [_fake_sample(1, 1, 0), _fake_sample(2, 1, 0)]
    batch = PbcCollator()(samples)
    assert batch['target'].shape == (2,)


if __name__ == '__main__':
    test_pad_2d_basic()
    test_collator_pads_variable_accounts()
    test_collator_handles_empty_branch()
    test_collator_obligations_padded()
    test_collator_target_stacked()
    print('✓ test_pbc_collator passed')
