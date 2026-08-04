"""PBC model 单元测试：前向 + 反向 + 边界情况（C1=0, obligations=0, 等）。

测试 fixture 直接用生产报告（cris_json_split/json_*.json）→ build_sample → collate，
保证测试覆盖真实数据形态。
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.dataset import _to_tensor
from pbc_credit.fields import (
    PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE,
    OBLIGATION_TYPE_VOCAB_SIZE,
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS,
    SUMMARY_TABLES, OBLIGATIONS_CAT_FIELDS,
)
from pbc_credit.losses import pretrain_loss, finetune_loss
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig
from pbc_credit.sample_builder import build_sample, encode_sample
from pbc_credit.vocab import build_cat_vocab


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / 'data' / 'pbc' / 'cris_json_split'


def _load_vocab():
    return build_cat_vocab()


def _load_real_samples(n: int = 4):
    """从前 n 份生产报告构建 samples。"""
    vocab = _load_vocab()
    files = sorted(SAMPLES_DIR.glob('json_*.json'))[:n]
    samples = []
    for f in files:
        with open(f) as fp:
            report = json.load(fp)
        s = build_sample(report, vocab)
        encode_sample(s, vocab)
        samples.append(s)
    return samples, vocab


def _build_cfg(vocab):
    """从真实 vocab 构建 cfg。"""
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


def test_finetune_forward_backward():
    """finetune 模式下前向输出 [B,1]；反向更新参数。"""
    samples, vocab = _load_real_samples(4)
    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=False)

    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (4, 1), logits.shape

    # 反向
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = finetune_loss(logits, target, pos_weight=4.0)
    loss.backward()
    assert model.user_encoder.norm.weight.grad is not None


def test_pretrain_forward_backward():
    """pretrain 模式下 mask 损失可反传。"""
    samples, vocab = _load_real_samples(4)
    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=True)

    base_batch = PbcCollator()(samples)
    batch = add_masks_to_batch(base_batch, mask_ratio=0.5)
    out = model(batch)

    # 至少有一个分支的 pred
    assert any(k.endswith('_pred') for k in out), 'pretrain out should have pred keys'
    # 应该包含 c1 和 obligation 的 pred
    keys_str = ' '.join(out.keys())
    assert 'acc_c1_' in keys_str, 'c1 pretrain head missing'
    assert 'obligation_numeric_pred' in keys_str, 'obligation pretrain head missing'

    loss, comps = pretrain_loss(out)
    assert not torch.isnan(loss)
    loss.backward()
    assert not torch.isnan(loss)


def test_handles_empty_accounts():
    """所有账户分支为空时不崩（生产中 N=0 是常态）。"""
    samples, vocab = _load_real_samples(2)
    # 把所有 account 字段清空
    for s in samples:
        for t in ('d1', 'r1', 'r2', 'r3', 'r4', 'c1'):
            s[f'{t}_numeric'] = s[f'{t}_numeric'][:0]
            s[f'{t}_cat_ids'] = s[f'{t}_cat_ids'][:0]
            s[f'{t}_cat_mask'] = s[f'{t}_cat_mask'][:0]
            s[f'{t}_paystate'] = s[f'{t}_paystate'][:0]
            s[f'{t}_mask'] = torch.zeros(0, dtype=torch.long)
        # 清空 obligations + queries + publics
        for k in ('obligation', 'query', 'public'):
            s[f'{k}_numeric'] = s[f'{k}_numeric'][:0]
            s[f'{k}_cat_ids'] = s[f'{k}_cat_ids'][:0]
            s[f'{k}_cat_mask'] = s[f'{k}_cat_mask'][:0]
            s[f'{k}_mask'] = torch.zeros(0, dtype=torch.long)

    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_handles_single_account():
    """每个分支只有 1 个账户时不崩（边界情况）。"""
    samples, vocab = _load_real_samples(2)
    for s in samples:
        for t in ('d1', 'r1', 'r2', 'r3', 'r4', 'c1'):
            s[f'{t}_numeric'] = s[f'{t}_numeric'][:1]
            s[f'{t}_cat_ids'] = s[f'{t}_cat_ids'][:1]
            s[f'{t}_cat_mask'] = s[f'{t}_cat_mask'][:1]
            s[f'{t}_paystate'] = s[f'{t}_paystate'][:1]
            s[f'{t}_mask'] = torch.ones(1, dtype=torch.long)

    cfg = _build_cfg(vocab)
    model = PbcCreditModel(cfg, pretrain_mode=False)
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_dataset_roundtrip():
    """JSONL 序列化/反序列化往返测试。"""
    from pbc_credit.sample_builder import to_jsonable, from_jsonable
    samples, _ = _load_real_samples(1)
    s = samples[0]
    js = to_jsonable(s)
    # 模拟 jsonl 序列化
    serialized = json.dumps(js, ensure_ascii=False)
    deserialized = json.loads(serialized)
    restored = {k: _to_tensor(k, v) for k, v in deserialized.items()}

    # 关键字段 shape 应该一致
    for k in ('user_numeric', 'summary_numeric', 'd1_numeric',
              'c1_numeric', 'obligation_numeric', 'query_numeric'):
        assert restored[k].shape == s[k].shape, f'{k}: {restored[k].shape} vs {s[k].shape}'


if __name__ == '__main__':
    test_finetune_forward_backward()
    test_pretrain_forward_backward()
    test_handles_empty_accounts()
    test_handles_single_account()
    test_dataset_roundtrip()
    print('✓ test_pbc_model passed')
