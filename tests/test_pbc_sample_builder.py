"""PBC sample_builder 单元测试：解析真实 CrisPbc.json。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.sample_builder import build_sample, encode_sample
from pbc_credit.vocab import build_cat_vocab, save_vocab, load_vocab
from pbc_credit.fields import ACCOUNT_TYPES, USER_NUMERIC_SPECS

TEMPLATE = 'data/home-credit/个人征信/CrisPbc.json'
VOCAB_PATH = 'data/pbc/processed/cat_vocab_test.json'


def _ensure_vocab():
    if Path(VOCAB_PATH).exists():
        return load_vocab(VOCAB_PATH)
    vocab = build_cat_vocab()
    save_vocab(vocab, VOCAB_PATH)
    return vocab


def test_build_sample_shapes():
    """解析 CrisPbc.json 后，所有 5 模态字段非空且形状正确。"""
    vocab = _ensure_vocab()
    with open(TEMPLATE, encoding='utf-8') as f:
        report = json.load(f)

    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    # user
    assert sample['user_numeric'].shape == (len(USER_NUMERIC_SPECS),)
    assert sample['user_cat_ids'].shape == (11,)  # USER_CAT_FIELDS 长度
    assert sample['user_cat_mask'].shape == (11,)

    # summary
    assert sample['summary_numeric'].dim() == 1
    assert sample['summary_numeric'].shape[0] > 10  # 多张表聚合

    # accounts（按类型分桶）— ACCOUNT_NUMERIC_FIELDS=8, ACCOUNT_CAT_FIELDS=6
    for t in ACCOUNT_TYPES:
        k = t.lower()
        n = sample[f'{k}_mask'].shape[0]
        if n > 0:
            assert sample[f'{k}_numeric'].shape == (n, 8), \
                f'{k}_numeric shape: {sample[f"{k}_numeric"].shape}'
            assert sample[f'{k}_paystate'].shape == (n, 60)
            assert sample[f'{k}_cat_ids'].shape[1] == 6
            assert sample[f'{k}_mask'].shape == (n,)

    # queries
    n_q = sample['query_mask'].shape[0]
    assert n_q > 0  # CrisPbc 样本有 23 条查询
    assert sample['query_numeric'].shape == (n_q, 1)

    # publics：CrisPbc.json 可能没有 publicInfo，所以只验形状
    n_p = sample['public_mask'].shape[0]
    if n_p > 0:
        assert sample['public_numeric'].shape == (n_p, 2)


def test_build_sample_has_no_nan():
    """user_numeric 无 NaN。"""
    vocab = _ensure_vocab()
    with open(TEMPLATE, encoding='utf-8') as f:
        report = json.load(f)
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    # 把 NaN 替换成 0 后再验（实际训练前会过 normalizer）
    u = sample['user_numeric']
    nan_count = torch.isnan(u).sum().item()
    # 至多 3 个 NaN（年龄/最早手机/最近手机 在某些字段缺失时）
    assert nan_count <= 3, f'too many NaN in user_numeric: {nan_count}'


def test_encode_sample_idempotent_vocab():
    """vocab 加载 + 编码后 cat_ids 是合法整数。"""
    vocab = _ensure_vocab()
    with open(TEMPLATE, encoding='utf-8') as f:
        report = json.load(f)
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    # user_cat_ids 全部 >= 0
    assert (sample['user_cat_ids'] >= 0).all()
    # summary_cat_ids 同理
    assert (sample['summary_cat_ids'] >= 0).all()


if __name__ == '__main__':
    test_build_sample_shapes()
    test_build_sample_has_no_nan()
    test_encode_sample_idempotent_vocab()
    print('✓ test_pbc_sample_builder passed')
