"""基于 CrisPbc.json 模板造假数据集，用于本地端到端跑通。

输出：
  data/pbc/processed/samples_train.pkl
  data/pbc/processed/samples_val.pkl
  data/pbc/processed/samples_test_unlabeled.pkl
  data/pbc/processed/cat_vocab.json

每条 sample 同时包含：
  - pbc_struct 各模态 tensor（src/pbc_credit/sample_builder）
  - pbc_text 的 tokenized 结果（text_input_ids + text_attention_mask）

用法：
  python scripts/prepare_pbc_samples.py --template data/home-credit/个人征信/CrisPbc.json --n 1000
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.sample_builder import build_sample, encode_sample
from pbc_credit.vocab import build_cat_vocab, save_vocab, load_vocab
from pbc_credit.fields import get_path


# ============================================================
# pbc_text 拼接（生产侧由 Spark 做，这里本地造假参考）
# ============================================================

def build_pbc_text(report: dict, max_len: int = 512) -> str:
    """把 CrisPbc.json 的关键字段拼成 pbc_text 文本。

    生产端你们已有自己的拼接逻辑，这个只是本地 smoke test 用。
    """
    parts = []

    idt = get_path(report, 'personInfo.identity', {}) or {}
    if idt:
        parts.append(f"[PERSON] 性别{idt.get('pb01ad01', '')} 学历{idt.get('pb01ad02', '')} "
                     f"学位{idt.get('pb01ad03', '')} 就业{idt.get('pb01ad04', '')}")

    marriage = get_path(report, 'personInfo.marriage', {}) or {}
    if marriage:
        parts.append(f"[MARRIAGE] 状态{marriage.get('pb020d01', '')}")

    for prof in (get_path(report, 'personInfo.professionals', []) or [])[:1]:
        parts.append(f"[JOB] 单位{prof.get('pb040q01', '')} 行业{prof.get('pb040d03', '')} "
                     f"职务{prof.get('pb040d05', '')}")

    # 账户
    for a in (get_path(report, 'accountInfos', []) or []):
        b = a.get('accountBasic', {}) or {}
        parts.append(f"[ACCT] 类型{b.get('pd01ad01', '')} 业务{b.get('pd01ad02', '')} "
                     f"发放{b.get('pd01aj01', '')} 余额{b.get('pd01aj02', '')} "
                     f"发放日{b.get('pd01ar01', '')}")
        # 最近 24 月还款状态
        s24 = get_path(a, 'latest24PayState.latest24state')
        if s24:
            parts.append(f"[PAYSTATE24] {s24}")

    # 概要
    sinfo = get_path(report, 'summaryInfo', {}) or {}
    if sinfo.get('querySummary'):
        qs = sinfo['querySummary']
        parts.append(f"[QUERY_SUM] 近1月贷款{qs.get('pc05bs01', '')}次 "
                     f"近2年贷后{qs.get('pc05bs06', '')}次")

    # 查询记录
    for q in (get_path(report, 'queryRecords', []) or [])[:5]:
        parts.append(f"[QUERY] 机构{q.get('ph010d01', '')} 原因{q.get('ph010q03', '')} "
                     f"日期{q.get('ph010r01', '')}")

    text = ' '.join(parts)
    return text[:max_len * 4]  # 粗糙截断（按字符）


def get_text_tokenizer(name: str = 'models/gte-new'):
    """加载 gte-new tokenizer（生产端你们可能用别的，这里复用项目内模型）。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def perturb_report(template: dict, idx: int, seed: int) -> dict:
    """对模板做随机扰动：改 reportsn + shuffle 账户 + 随机删除部分账户。"""
    rng = random.Random(seed + idx)
    r = copy.deepcopy(template)
    r['reportsn'] = f'mock_{idx:010d}_{rng.randint(10000, 99999)}'
    r['name'] = f'测试{idx:04d}'

    accs = r.get('accountInfos', [])
    rng.shuffle(accs)
    # 随机保留 2~8 个账户
    keep = rng.randint(min(2, len(accs)), min(8, len(accs)))
    r['accountInfos'] = accs[:keep]

    # 随机改部分金额数值（±10%）
    for a in r['accountInfos']:
        basic = a.get('accountBasic', {})
        for k in ('pd01aj01', 'pd01aj02', 'pd01aj03'):
            v = basic.get(k)
            if v and v != '':
                try:
                    fv = float(v)
                    basic[k] = str(int(fv * rng.uniform(0.9, 1.1)))
                except (TypeError, ValueError):
                    pass

    # shuffle queryRecords，随机保留
    qs = r.get('queryRecords', [])
    rng.shuffle(qs)
    r['queryRecords'] = qs[:rng.randint(min(3, len(qs)), min(20, len(qs)))]

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', default='data/home-credit/个人征信/CrisPbc.json')
    parser.add_argument('--out_dir', default='data/pbc/processed')
    parser.add_argument('--n', type=int, default=1000, help='总样本数')
    parser.add_argument('--pos_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--codetable', default='data/home-credit/个人征信/个人征信码值表.xlsx')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. 构建 vocab ===
    vocab_path = out_dir / 'cat_vocab.json'
    if not vocab_path.exists():
        print(f'=== 构建 cat_vocab from {args.codetable} ===')
        # 重定向路径让 build_cat_vocab 能找到码值表
        import openpyxl
        wb = openpyxl.load_workbook(args.codetable, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        tables = {}
        for row in ws.iter_rows(values_only=True):
            if not row or row[0] is None:
                continue
            name = str(row[0]).strip()
            code = row[1]
            if code is None or name in ('关键字',):
                continue
            tables.setdefault(name, {}).setdefault('<UNK>', 0)
            code = str(code).strip()
            if code not in tables[name]:
                tables[name][code] = len(tables[name])
        # 用 build_cat_vocab 逻辑（传入已有 tables）
        from pbc_credit.vocab import collect_used_tables
        used = collect_used_tables()
        vocab = {
            branch: {name: tables.get(name, {'<UNK>': 0})
                     for name in names if name}
            for branch, names in used.items()
        }
        from pbc_credit.fields import PAYSTATE_VOCAB, PUBLIC_TYPE_VOCAB
        vocab['paystate'] = {'<all>': PAYSTATE_VOCAB}
        vocab['public_type'] = {'<all>': PUBLIC_TYPE_VOCAB}
        save_vocab(vocab, vocab_path)
        print(f'  saved → {vocab_path}')
    else:
        vocab = load_vocab(vocab_path)
        print(f'loaded vocab: {vocab_path}')

    # === 2. 读模板 + 造假 ===
    with open(args.template, encoding='utf-8') as f:
        template = json.load(f)

    n = args.n
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    splits = {
        'train': [(i, 1 if random.random() < args.pos_ratio else 0) for i in range(n_train)],
        'val': [(i + n_train, 1 if random.random() < args.pos_ratio else 0)
                for i in range(n_val)],
        'test_unlabeled': [(i + n_train + n_val, None)
                           for i in range(n_test)],
    }

    print(f'=== 造假 {n} 样本：train={n_train} val={n_val} test={n_test} ===')

    # 加载 gte-new tokenizer（造假含 pbc_text）
    try:
        tokenizer = get_text_tokenizer()
        print(f'loaded tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}')
    except Exception as e:
        print(f'WARN: 无法加载 gte-new tokenizer ({e})，sample 不含 text_input_ids')
        tokenizer = None

    for split, items in splits.items():
        out_path = out_dir / f'samples_{split}.pkl'
        with open(out_path, 'wb') as f:
            for global_idx, label in items:
                r = perturb_report(template, global_idx, args.seed)
                sample = build_sample(r, vocab)
                encode_sample(sample, vocab)
                # 造假 pbc_text
                if tokenizer is not None:
                    text = build_pbc_text(r, max_len=128)
                    enc = tokenizer(text, max_length=128, truncation=True,
                                    padding=False, return_tensors=None)
                    sample['text_input_ids'] = torch.tensor(enc['input_ids'], dtype=torch.long)
                    sample['text_attention_mask'] = torch.tensor(enc['attention_mask'], dtype=torch.long)
                if label is not None:
                    sample['target'] = torch.tensor([float(label)], dtype=torch.float32)
                pickle.dump(sample, f)
        print(f'  wrote {len(items)} → {out_path}')

    print(f'\n=== Done. Samples in {out_dir} ===')


if __name__ == '__main__':
    main()
