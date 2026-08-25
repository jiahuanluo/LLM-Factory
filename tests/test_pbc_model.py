"""PBC model 单元测试：前向 + 反向 + 边界情况（C1=0、全空账户等，新 SQL 管道格式）。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.fields import (
    ACCOUNT_TYPES, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS,
    USER_NUMERIC_DIM, USER_CAT_DIM, ACCOUNT_NUMERIC_DIM, ACCOUNT_CAT_DIM,
    PAYSTATE_VOCAB_SIZE, PAYSTATE_LEN,
)
from pbc_credit.losses import pretrain_loss, finetune_loss
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig


def make_vocab(min_values: int = 4) -> dict:
    """合成 vocab：每张码值表 UNK + 若干值。"""
    vocab = {'user': {}, 'account': {}}
    for _f, table in USER_CAT_FIELDS:
        vocab['user'][table] = {'<UNK>': 0, **{str(v): v for v in range(1, min_values + 1)}}
    for _f, table in ACCOUNT_CAT_FIELDS:
        vocab['account'][table] = {'<UNK>': 0, **{str(v): v for v in range(1, min_values + 1)}}
    return vocab


def build_cfg(vocab: dict, d: int = 32, n_heads: int = 4, n_layers: int = 1) -> PbcCreditModelConfig:
    return PbcCreditModelConfig(
        d=d, n_heads=n_heads, n_layers=n_layers, dropout=0.0, top_hidden=64,
        user_numeric_dim=USER_NUMERIC_DIM,
        user_cat_tables={t: len(vocab['user'][t]) + 1 for _f, t in USER_CAT_FIELDS},
        account_numeric_dim=ACCOUNT_NUMERIC_DIM,
        account_cat_tables={t: len(vocab['account'][t]) + 1 for _f, t in ACCOUNT_CAT_FIELDS},
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
    )


def make_sample(counts: dict[str, int], seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sample = {
        'user_numeric': torch.randn(USER_NUMERIC_DIM, generator=g),
        'user_cat_ids': torch.randint(1, 4, (USER_CAT_DIM,), generator=g),
        'user_cat_mask': torch.randint(0, 2, (USER_CAT_DIM,), generator=g),
    }
    for t in [x.lower() for x in ACCOUNT_TYPES]:
        n = counts.get(t, 0)
        sample[f'{t}_numeric'] = torch.randn(n, ACCOUNT_NUMERIC_DIM, generator=g)
        sample[f'{t}_cat_ids'] = torch.randint(1, 4, (n, ACCOUNT_CAT_DIM), generator=g)
        sample[f'{t}_cat_mask'] = torch.ones(n, ACCOUNT_CAT_DIM, dtype=torch.long)
        sample[f'{t}_paystate'] = torch.randint(2, 20, (n, PAYSTATE_LEN), generator=g)
        sample[f'{t}_mask'] = torch.ones(n, dtype=torch.long)
    return sample


def _load_samples(specs: list[dict[str, int]]):
    return [make_sample(s, seed=i) for i, s in enumerate(specs)]


def test_finetune_forward_backward():
    """finetune 模式下前向输出 [B,1]；反向更新参数。"""
    samples = _load_samples([{'d1': 2, 'r2': 3}, {'d1': 1, 'c1': 1}, {}, {'r2': 2}])
    model = PbcCreditModel(build_cfg(make_vocab()), pretrain_mode=False)

    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (4, 1), logits.shape
    assert not torch.isnan(logits).any()

    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = finetune_loss(logits, target, pos_weight=4.0)
    loss.backward()
    assert model.user_encoder.norm.weight.grad is not None


def test_pretrain_forward_backward():
    """pretrain 模式：mask 重建头输出 + 损失可反传。"""
    samples = _load_samples([{'d1': 2, 'r2': 3}, {'d1': 3}])
    model = PbcCreditModel(build_cfg(make_vocab()), pretrain_mode=True)

    batch = add_masks_to_batch(PbcCollator()(samples), mask_ratio=0.5)
    out = model(batch)

    keys_str = ' '.join(out.keys())
    assert 'acc_d1_numeric_pred' in keys_str, 'd1 numeric 重建头缺失'
    assert 'acc_d1_paystate_pred' in keys_str, 'paystate 重建头缺失'
    assert 'user_numeric_pred' in keys_str, 'user numeric 重建头缺失'
    assert 'user_emb' in out, 'user_emb（contrastive 用）缺失'

    loss, comps = pretrain_loss(out)
    assert not torch.isnan(loss)
    loss.backward()
    assert not torch.isnan(loss)


def test_handles_empty_accounts():
    """所有账户分支为空时不崩（生产中 N=0 是常态）。"""
    samples = _load_samples([{}, {}])
    model = PbcCreditModel(build_cfg(make_vocab()), pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_handles_single_account():
    """每个分支只有 1 个账户时不崩（边界情况）。"""
    samples = _load_samples([{t: 1 for t in ['d1', 'r2']}, {'d1': 1}])
    model = PbcCreditModel(build_cfg(make_vocab()), pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_param_count_scaled():
    """默认配置参数量应与设计一致（user+accounts 双模态，d=256/4 层）。"""
    cfg = PbcCreditModelConfig()  # 默认 d=256, n_layers=4
    cfg.user_cat_tables = {t: 30 for _f, t in USER_CAT_FIELDS}
    cfg.account_cat_tables = {t: 30 for _f, t in ACCOUNT_CAT_FIELDS}
    model = PbcCreditModel(cfg, pretrain_mode=False)
    n = sum(p.numel() for p in model.parameters())
    assert n > 5_000_000, f'params {n:,} < 5M，模型意外缩水'


if __name__ == '__main__':
    test_finetune_forward_backward()
    test_pretrain_forward_backward()
    test_handles_empty_accounts()
    test_handles_single_account()
    test_param_count_scaled()
    print('✓ test_pbc_model passed')
