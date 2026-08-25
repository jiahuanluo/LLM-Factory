"""PbcCollator 测试：padding 形状 + 空账户边界（新 SQL 管道格式）。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import _to_tensor, _2D_COL_COUNTS
from pbc_credit.fields import (
    ACCOUNT_TYPES, USER_NUMERIC_DIM, USER_CAT_DIM,
    ACCOUNT_NUMERIC_DIM, ACCOUNT_CAT_DIM, PAYSTATE_LEN,
)


def make_sample(counts: dict[str, int], seed: int = 0) -> dict:
    """counts: {type: N}，构造一个新格式样本（tensor 形态，同 PbcDataset 输出）。"""
    g = torch.Generator().manual_seed(seed)
    sample = {
        'user_numeric': torch.randn(USER_NUMERIC_DIM, generator=g),
        'user_cat_ids': torch.randint(1, 5, (USER_CAT_DIM,), generator=g),
        'user_cat_mask': torch.randint(0, 2, (USER_CAT_DIM,), generator=g),
    }
    for t in [x.lower() for x in ACCOUNT_TYPES]:
        n = counts.get(t, 0)
        sample[f'{t}_numeric'] = torch.randn(n, ACCOUNT_NUMERIC_DIM, generator=g)
        sample[f'{t}_cat_ids'] = torch.randint(1, 5, (n, ACCOUNT_CAT_DIM), generator=g)
        sample[f'{t}_cat_mask'] = torch.ones(n, ACCOUNT_CAT_DIM, dtype=torch.long)
        sample[f'{t}_paystate'] = torch.randint(1, 20, (n, PAYSTATE_LEN), generator=g)
        sample[f'{t}_mask'] = torch.ones(n, dtype=torch.long)
    return sample


def test_collator_shapes():
    samples = [
        make_sample({'d1': 3, 'r2': 5}, seed=1),
        make_sample({'d1': 1, 'r2': 2, 'c1': 1}, seed=2),
    ]
    batch = PbcCollator()(samples)

    assert batch['user_numeric'].shape == (2, USER_NUMERIC_DIM)
    assert batch['user_cat_ids'].shape == (2, USER_CAT_DIM)
    assert batch['user_cat_mask'].shape == (2, USER_CAT_DIM)

    assert batch['d1_numeric'].shape == (2, 3, ACCOUNT_NUMERIC_DIM)
    assert batch['d1_cat_ids'].shape == (2, 3, ACCOUNT_CAT_DIM)
    assert batch['d1_paystate'].shape == (2, 3, PAYSTATE_LEN)
    assert batch['d1_mask'].shape == (2, 3)
    # sample 2 只有 1 个 d1，pad 位 mask=0 且数值为 0
    assert batch['d1_mask'][1, 1:].sum() == 0
    assert batch['d1_numeric'][1, 1:].abs().sum() == 0

    assert batch['r2_numeric'].shape == (2, 5, ACCOUNT_NUMERIC_DIM)
    # 两样本都没有 r3/r4 → 0 长度维度
    assert batch['r3_numeric'].shape == (2, 0, ACCOUNT_NUMERIC_DIM)
    assert batch['r4_paystate'].shape == (2, 0, PAYSTATE_LEN)
    # c1 只有 sample 2 有
    assert batch['c1_mask'][0].sum() == 0
    assert batch['c1_mask'][1].sum() == 1


def test_collator_all_empty_accounts():
    """全空账户不崩（生产常态）。"""
    samples = [make_sample({}, seed=s) for s in range(3)]
    batch = PbcCollator()(samples)
    for t in [x.lower() for x in ACCOUNT_TYPES]:
        assert batch[f'{t}_numeric'].shape == (3, 0, ACCOUNT_NUMERIC_DIM)
        assert batch[f'{t}_mask'].shape == (3, 0)


def test_to_tensor_empty_2d_reshape():
    """JSON 反序列化的空 2D 字段 [] 应 reshape 成 (0, cols)。"""
    for name, cols in _2D_COL_COUNTS.items():
        t = _to_tensor(name, [])
        assert t.shape == (0, cols), f'{name}: {t.shape} != (0, {cols})'
    # 非 2D 字段不受影响
    assert _to_tensor('user_numeric', [1.0] * USER_NUMERIC_DIM).shape == (USER_NUMERIC_DIM,)


def test_to_tensor_dtypes():
    num = _to_tensor('d1_numeric', [[1.5] * ACCOUNT_NUMERIC_DIM])
    cat = _to_tensor('d1_cat_ids', [[1] * ACCOUNT_CAT_DIM])
    pay = _to_tensor('d1_paystate', [[1] * PAYSTATE_LEN])
    assert num.dtype == torch.float32
    assert cat.dtype == torch.long
    assert pay.dtype == torch.long


if __name__ == '__main__':
    test_collator_shapes()
    test_collator_all_empty_accounts()
    test_to_tensor_empty_2d_reshape()
    test_to_tensor_dtypes()
    print('✓ test_pbc_collator passed')
