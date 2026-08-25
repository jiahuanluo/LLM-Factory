"""端到端测试：SQL 物化表 dump → postprocess → PbcDataset → collator → model。

数据形态对齐 scripts/postprocess_pbc_struct.py 的输出契约：
  {"reportsn": "...", "pbc_struct": "<stringified flat sample>"}
不再依赖本地 mock JSON 报告（旧 sample_builder 路径已废弃，数据由 SQL 产出）。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from postprocess_pbc_struct import transform_sample  # noqa: E402

from pbc_credit.collator import PbcCollator  # noqa: E402
from pbc_credit.dataset import PbcDataset  # noqa: E402
from pbc_credit.fields import (  # noqa: E402
    ACCOUNT_TYPES, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS,
    USER_NUMERIC_DIM, USER_CAT_DIM, ACCOUNT_NUMERIC_DIM, ACCOUNT_CAT_DIM,
    PAYSTATE_LEN, PAYSTATE_VOCAB_SIZE,
)
from pbc_credit.losses import pretrain_loss, finetune_loss  # noqa: E402
from pbc_credit.masking import add_masks_to_batch  # noqa: E402
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig  # noqa: E402
from pbc_credit.vocab import build_cat_vocab  # noqa: E402


def _make_sql_struct(counts: dict[str, int]) -> dict:
    """构造 postprocess 输入侧（SQL TO_JSON 输出）的内层 struct。"""
    struct = {
        'user_numeric': [round(0.1 * i, 2) for i in range(USER_NUMERIC_DIM)],
        'user_cat_ids': [1 + (i % 4) for i in range(USER_CAT_DIM)],
        'user_cat_mask': [i % 2 for i in range(USER_CAT_DIM)],
    }
    for t in [x.lower() for x in ACCOUNT_TYPES]:
        n = counts.get(t, 0)
        struct[t] = [
            {
                'numeric': [round(0.1 * j, 2) for j in range(ACCOUNT_NUMERIC_DIM)],
                'cat_ids': [1 + (j % 4) for j in range(ACCOUNT_CAT_DIM)],
                'cat_mask': [1] * ACCOUNT_CAT_DIM,
                'paystate': [0] * 20 + [1 + (j % 19) for j in range(PAYSTATE_LEN - 20)],
            }
            for _ in range(n)
        ]
    return struct


def _make_vocab() -> dict:
    vocab = {'user': {}, 'account': {}}
    for _f, table in USER_CAT_FIELDS:
        vocab['user'][table] = {'<UNK>': 0, **{str(v): v for v in range(1, 5)}}
    for _f, table in ACCOUNT_CAT_FIELDS:
        vocab['account'][table] = {'<UNK>': 0, **{str(v): v for v in range(1, 5)}}
    return vocab


def _build_cfg(vocab: dict) -> PbcCreditModelConfig:
    return PbcCreditModelConfig(
        d=32, n_heads=4, n_layers=1, dropout=0.0, top_hidden=64,
        user_numeric_dim=USER_NUMERIC_DIM,
        user_cat_tables={t: len(vocab['user'][t]) + 1 for _f, t in USER_CAT_FIELDS},
        account_numeric_dim=ACCOUNT_NUMERIC_DIM,
        account_cat_tables={t: len(vocab['account'][t]) + 1 for _f, t in ACCOUNT_CAT_FIELDS},
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
    )


def _write_train_jsonl(path: Path, specs: list[dict[str, int]]):
    """构造 SQL dump → 过 postprocess → 训练 JSONL（同一契约）。"""
    with open(path, 'w', encoding='utf-8') as f:
        for i, counts in enumerate(specs):
            flat = transform_sample(_make_sql_struct(counts))
            f.write(json.dumps({
                'reportsn': f'R{i:04d}',
                'pbc_struct': json.dumps(flat, ensure_ascii=False),
            }, ensure_ascii=False) + '\n')


def test_postprocess_transform_contract():
    """postprocess 输出字段契约：扁平 5 数组 per 类型 + user 3 数组。"""
    flat = transform_sample(_make_sql_struct({'d1': 2, 'r2': 1}))
    for key in ('user_numeric', 'user_cat_ids', 'user_cat_mask'):
        assert key in flat
    for t in [x.lower() for x in ACCOUNT_TYPES]:
        assert f'{t}_numeric' in flat
        assert f'{t}_cat_ids' in flat
        assert f'{t}_cat_mask' in flat
        assert f'{t}_paystate' in flat
        assert f'{t}_mask' in flat
    assert len(flat['user_numeric']) == USER_NUMERIC_DIM
    assert len(flat['d1_numeric']) == 2
    assert len(flat['d1_numeric'][0]) == ACCOUNT_NUMERIC_DIM
    assert len(flat['d1_paystate'][0]) == PAYSTATE_LEN
    assert flat['d1_mask'] == [1, 1]
    assert flat['r3_numeric'] == []  # 空类型 → 空数组（dataset 会 reshape (0, cols)）


def test_dataset_from_postprocess_output():
    """postprocess 输出 JSONL → PbcDataset：字段 shape / dtype / report_id。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'train.jsonl'
        _write_train_jsonl(p, [{'d1': 2, 'r2': 1}, {}])
        ds = PbcDataset(p, pretrain_mode=True)
        assert len(ds) == 2
        s0, s1 = ds[0], ds[1]
        assert s0['user_numeric'].shape == (USER_NUMERIC_DIM,)
        assert s0['d1_numeric'].shape == (2, ACCOUNT_NUMERIC_DIM)
        assert s0['d1_paystate'].shape == (2, PAYSTATE_LEN)
        assert s1['d1_numeric'].shape == (0, ACCOUNT_NUMERIC_DIM)  # 空类型 reshape
        assert s0['report_id'] == 'R0000'
        # label 模式
        with open(p, encoding='utf-8') as f:
            lines = [json.loads(x) for x in f]
        for i, line in enumerate(lines):
            line['label'] = i % 2
        p2 = Path(tmp) / 'train_label.jsonl'
        with open(p2, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        ds2 = PbcDataset(p2, pretrain_mode=False)
        assert torch.allclose(ds2[0]['target'], torch.tensor([0.0]))
        assert torch.allclose(ds2[1]['target'], torch.tensor([1.0]))


def test_e2e_pretrain_and_finetune():
    """全链路：JSONL → dataset → collator → pretrain/finetune forward+backward。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'train.jsonl'
        _write_train_jsonl(p, [{'d1': 2, 'r2': 3}, {'d1': 1, 'c1': 1}, {}, {'r2': 2}])
        ds = PbcDataset(p, pretrain_mode=True)
        samples = list(ds)
        vocab = _make_vocab()
        cfg = _build_cfg(vocab)

        # pretrain
        model = PbcCreditModel(cfg, pretrain_mode=True)
        batch = add_masks_to_batch(PbcCollator()(samples), mask_ratio=0.3)
        out = model(batch)
        assert any(k.endswith('_pred') for k in out)
        loss, _ = pretrain_loss(out)
        loss.backward()
        assert not torch.isnan(loss)

        # finetune
        model = PbcCreditModel(cfg, pretrain_mode=False)
        batch = PbcCollator()(samples)
        logits = model(batch)
        assert logits.shape == (4, 1)
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = finetune_loss(logits, target, pos_weight=4.0)
        loss.backward()
        assert not torch.isnan(loss)


def test_vocab_build_from_sql_dump():
    """SQL dump JSONL → build_cat_vocab：id 与 SQL ROW_NUMBER 规则一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'dump.jsonl'
        rows = [
            {"section": "user", "code_table": "性别代码表", "code_value": "2"},
            {"section": "user", "code_table": "性别代码表", "code_value": "1"},
            {"section": "user", "code_table": "性别代码表", "code_value": "1"},  # 去重
            {"section": "account", "code_table": "机构类型代码", "code_value": "9"},
            {"section": "account", "code_table": "机构类型代码", "code_value": "1"},
        ]
        with open(p, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        vocab = build_cat_vocab(str(p))
        assert vocab['user']['性别代码表'] == {'<UNK>': 0, '1': 1, '2': 2}
        assert vocab['account']['机构类型代码'] == {'<UNK>': 0, '1': 1, '9': 2}


def test_postprocess_script_cli():
    """postprocess_pbc_struct.py CLI 冒烟（pandarallel 并行路径）。"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'dump.jsonl'
        out = Path(tmp) / 'train.jsonl'
        with open(src, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'reportsn': 'R0001',
                                'pbc_struct': json.dumps(_make_sql_struct({'d1': 1}),
                                                         ensure_ascii=False)},
                               ensure_ascii=False) + '\n')
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'scripts' / 'postprocess_pbc_struct.py'),
             str(src), 'ignored', str(out)],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        with open(out, encoding='utf-8') as f:
            rec = json.loads(f.readline())
        assert rec['reportsn'] == 'R0001'
        assert 'd1_paystate' in json.loads(rec['pbc_struct'])


if __name__ == '__main__':
    test_postprocess_transform_contract()
    test_dataset_from_postprocess_output()
    test_e2e_pretrain_and_finetune()
    test_vocab_build_from_sql_dump()
    test_postprocess_script_cli()
    print('✓ test_pbc_pipeline_e2e passed')
