"""端到端测试：用真实生产报告 + 多样化报告验证整个 pipeline。

覆盖：
1. 598 份生产 mock（data/pbc/cris_json_split/）→ build_sample + encode_sample 不报错
2. 多样化报告（data/pbc/cris_json_diverse/，含 13 种扰动）→ 不报错
3. 用真实报告 batch → pretrain forward + loss.backward
4. 用真实报告 batch → finetune forward + loss.backward
"""
import json
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.fields import (
    PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE, OBLIGATION_TYPE_VOCAB_SIZE,
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS,
    SUMMARY_TABLES, OBLIGATIONS_CAT_FIELDS,
)
from pbc_credit.losses import pretrain_loss, finetune_loss
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig
from pbc_credit.sample_builder import build_sample, encode_sample
from pbc_credit.vocab import build_cat_vocab, load_vocab

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_DIR = REPO_ROOT / 'data' / 'pbc' / 'cris_json_split'
DIVERSE_DIR = REPO_ROOT / 'data' / 'pbc' / 'cris_json_diverse'


def _load_vocab():
    p = REPO_ROOT / 'data' / 'pbc' / 'processed' / 'cat_vocab.json'
    if p.exists():
        return load_vocab(p)
    return build_cat_vocab()


def _build_cfg(vocab):
    user_tables = {t: len(vocab.get('user', {}).get(t, {'<UNK>': 0})) + 1
                   for _p, t in USER_CAT_FIELDS if t}
    summary_tables = {}
    for _n, _l, _nf, cats in SUMMARY_TABLES:
        for _f, t in cats:
            if t and t not in summary_tables:
                summary_tables[t] = len(vocab.get('summary', {}).get(t, {'<UNK>': 0})) + 1
    acc_tables = {t: len(vocab.get('account', {}).get(t, {'<UNK>': 0})) + 1
                  for _f, t in ACCOUNT_CAT_FIELDS if t}
    q_tables = {t: len(vocab.get('query', {}).get(t, {'<UNK>': 0})) + 1
                for _f, t in QUERY_CAT_FIELDS if t}
    obl_tables = {}
    for _ot, _f, t in OBLIGATIONS_CAT_FIELDS:
        if t and t not in obl_tables:
            obl_tables[t] = len(vocab.get('obligation', {}).get(t, {'<UNK>': 0})) + 1
    n_sum_num = sum((1 if is_list else 0) + len(nums)
                    for _name, is_list, nums, _c in SUMMARY_TABLES)
    return PbcCreditModelConfig(
        d=32, n_heads=4, n_layers=1, dropout=0.0, top_hidden=64,
        user_numeric_dim=18,
        user_cat_tables=user_tables,
        summary_numeric_dim=n_sum_num,
        summary_cat_tables=summary_tables,
        account_numeric_dim=15,
        account_cat_tables=acc_tables,
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
        query_numeric_dim=1,
        query_cat_tables=q_tables,
        public_type_vocab_size=PUBLIC_TYPE_VOCAB_SIZE,
        obligation_type_vocab_size=OBLIGATION_TYPE_VOCAB_SIZE,
        obligation_cat_tables=obl_tables,
    )


def _build_samples(files, vocab):
    out = []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        s = build_sample(r, vocab)
        encode_sample(s, vocab)
        out.append(s)
    return out


@pytest.mark.skipif(not PROD_DIR.exists(), reason='production mock reports missing')
def test_parse_all_production_reports():
    """598 份生产 mock 全部能解析。"""
    vocab = _load_vocab()
    files = sorted(PROD_DIR.glob('json_*.json'))
    n_ok = 0
    n_err = 0
    for f in files:
        try:
            with open(f) as fp:
                r = json.load(fp)
            s = build_sample(r, vocab)
            encode_sample(s, vocab)
            n_ok += 1
        except Exception:
            n_err += 1
    assert n_err == 0, f'{n_err}/{len(files)} production reports failed to parse'
    assert n_ok >= 500, f'expected >=500 production reports, got {n_ok}'


@pytest.mark.skipif(not DIVERSE_DIR.exists(), reason='diverse reports missing')
def test_parse_diverse_reports():
    """多样化报告（13 种扰动）全部能解析。"""
    vocab = _load_vocab()
    files = sorted(DIVERSE_DIR.glob('*.json'))
    if not files:
        pytest.skip('no diverse reports; run scripts/generate_diverse_reports.py')
    n_ok = 0
    n_err = 0
    errors = []
    for f in files:
        try:
            with open(f) as fp:
                r = json.load(fp)
            s = build_sample(r, vocab)
            encode_sample(s, vocab)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if len(errors) < 5:
                errors.append((f.name, type(e).__name__, str(e)[:80]))
    assert n_err == 0, f'{n_err}/{len(files)} diverse reports failed: {errors}'


@pytest.mark.skipif(not PROD_DIR.exists(), reason='production mock reports missing')
def test_e2e_pretrain_on_real_data():
    """端到端：真实报告 → batch → pretrain forward + backward。"""
    vocab = _load_vocab()
    files = sorted(PROD_DIR.glob('json_*.json'))[:4]
    samples = _build_samples(files, vocab)

    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=True)
    batch = add_masks_to_batch(PbcCollator()(samples), mask_ratio=0.3)
    out = model(batch)
    assert any(k.endswith('_pred') for k in out)
    loss, _ = pretrain_loss(out)
    loss.backward()
    assert not torch.isnan(loss)


@pytest.mark.skipif(not PROD_DIR.exists(), reason='production mock reports missing')
def test_e2e_finetune_on_real_data():
    """端到端：真实报告 → batch → finetune forward + backward。"""
    vocab = _load_vocab()
    files = sorted(PROD_DIR.glob('json_*.json'))[:4]
    samples = _build_samples(files, vocab)

    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (4, 1)
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = finetune_loss(logits, target, pos_weight=4.0)
    loss.backward()
    assert not torch.isnan(loss)


@pytest.mark.skipif(not DIVERSE_DIR.exists(), reason='diverse reports missing')
def test_e2e_finetune_on_diverse_data():
    """端到端：多样化报告 → batch → finetune forward。

    覆盖极端情况：heavy_accounts（N=30+）、heavy_queries（N=400+）、
    drop_summary、drop_publicinfo 等。
    """
    vocab = _load_vocab()
    files = sorted(DIVERSE_DIR.glob('*.json'))[:6]
    samples = _build_samples(files, vocab)

    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (6, 1)
    assert not torch.isnan(logits).any()


if __name__ == '__main__':
    test_parse_all_production_reports()
    test_parse_diverse_reports()
    test_e2e_pretrain_on_real_data()
    test_e2e_finetune_on_real_data()
    test_e2e_finetune_on_diverse_data()
    print('✓ test_pbc_pipeline_e2e passed')
