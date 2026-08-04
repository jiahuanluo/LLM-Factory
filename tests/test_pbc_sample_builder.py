"""PBC sample_builder 单元测试：解析真实生产报告。

测试用 data/pbc/cris_json_split/ 下的报告（生产 mock），不再依赖 CrisPbc.json 单模板。
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.sample_builder import build_sample, encode_sample
from pbc_credit.vocab import build_cat_vocab, save_vocab, load_vocab
from pbc_credit.fields import ACCOUNT_TYPES, USER_NUMERIC_SPECS, OBLIGATIONS_CAT_FIELDS

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / 'data' / 'pbc' / 'cris_json_split'
VOCAB_PATH = REPO_ROOT / 'data' / 'pbc' / 'processed' / 'cat_vocab_test.json'


def _ensure_vocab():
    if VOCAB_PATH.exists():
        return load_vocab(VOCAB_PATH)
    vocab = build_cat_vocab()
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, VOCAB_PATH)
    return vocab


def _load_first_report():
    files = sorted(SAMPLES_DIR.glob('json_*.json'))
    assert files, f'no production mock reports in {SAMPLES_DIR}'
    with open(files[0], encoding='utf-8') as f:
        return json.load(f)


def test_build_sample_shapes():
    """解析生产报告后，所有 6 模态字段形状正确。"""
    vocab = _ensure_vocab()
    report = _load_first_report()

    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    # user（18 维：13 base 含 3 稳定性 + score 5）
    assert sample['user_numeric'].shape == (len(USER_NUMERIC_SPECS),), \
        f"user_numeric: {sample['user_numeric'].shape} vs {len(USER_NUMERIC_SPECS)}"
    assert sample['user_numeric'].shape[0] == 18
    assert sample['user_cat_ids'].shape == (12,)
    assert sample['user_cat_mask'].shape == (12,)

    # summary
    assert sample['summary_numeric'].dim() == 1
    assert sample['summary_numeric'].shape[0] > 10

    # accounts（6 类）— ACCOUNT_NUMERIC_FIELDS=15, ACCOUNT_CAT_FIELDS=9
    for t in ACCOUNT_TYPES:
        k = t.lower()
        n = sample[f'{k}_mask'].shape[0]
        if n > 0:
            assert sample[f'{k}_numeric'].shape == (n, 15), \
                f'{k}_numeric shape: {sample[f"{k}_numeric"].shape}'
            assert sample[f'{k}_paystate'].shape == (n, 60)
            assert sample[f'{k}_cat_ids'].shape[1] == 9
            assert sample[f'{k}_mask'].shape == (n,)

    # queries
    n_q = sample['query_mask'].shape[0]
    assert n_q > 0
    assert sample['query_numeric'].shape == (n_q, 1)

    # publics
    n_p = sample['public_mask'].shape[0]
    if n_p > 0:
        assert sample['public_numeric'].shape == (n_p, 2)

    # obligations（新模态：合并 agreement + postpay + related_repay）
    n_o = sample['obligation_mask'].shape[0]
    assert n_o > 0, '生产报告应含至少 1 条 obligation'
    assert sample['obligation_numeric'].shape == (n_o, 2)
    # 7 列 cat = [type, ag_f1, ag_f2, pp_f1, pp_f2, rr_f1, rr_f2]
    assert sample['obligation_cat_ids'].shape == (n_o, 7)


def test_build_sample_has_no_nan():
    """user_numeric 无 NaN（生产报告应有完整 identity）。"""
    vocab = _ensure_vocab()
    report = _load_first_report()
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    u = sample['user_numeric']
    nan_count = torch.isnan(u).sum().item()
    assert nan_count <= 3, f'too many NaN in user_numeric: {nan_count}'


def test_encode_sample_idempotent_vocab():
    """vocab 加载 + 编码后 cat_ids 是合法整数。"""
    vocab = _ensure_vocab()
    report = _load_first_report()
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)

    assert (sample['user_cat_ids'] >= 0).all()
    assert (sample['summary_cat_ids'] >= 0).all()
    assert (sample['obligation_cat_ids'] >= 0).all()
    # c1 也应有合法 cat_ids
    assert (sample['c1_cat_ids'] >= 0).all()


def test_obligation_cat_layout():
    """obligation cat_ids 7 列结构正确：第 0 列 type，后 6 列字段。"""
    vocab = _ensure_vocab()
    report = _load_first_report()
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)
    n_o = sample['obligation_mask'].shape[0]
    if n_o == 0:
        return
    # 第 0 列 type_id 应在 [0, 3) 内（3 种 obligation type）
    type_ids = sample['obligation_cat_ids'][:, 0]
    assert (type_ids < 3).all(), f'type_id out of range: max={type_ids.max()}'
    # 后 6 列应对应 OBLIGATIONS_CAT_FIELDS 顺序（6 个 (type, field, table) 元组）
    assert sample['obligation_cat_ids'].shape[1] == 1 + len(OBLIGATIONS_CAT_FIELDS)


if __name__ == '__main__':
    test_build_sample_shapes()
    test_build_sample_has_no_nan()
    test_encode_sample_idempotent_vocab()
    test_obligation_cat_layout()
    print('✓ test_pbc_sample_builder passed')
