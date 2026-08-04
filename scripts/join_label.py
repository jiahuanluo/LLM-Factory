"""读 pbc_struct JSONL + 加 label → 训练用 JSONL。

生产中 label 来自业务表（按 biz_sno join），本脚本用于本地 dev：
  - 默认：随机 label（pos_ratio=0.2，无生产意义）
  - 可选：从外部 label 文件读取真实 label（label_file 字段：biz_sno,label）
  - 可选：scenario 扰动（stress test 用，生产中 Spark 不会做）

输出（生产 JSONL 格式，每行 {biz_sno, pbc_struct, label}）：
  data/pbc/processed/train.jsonl
  data/pbc/processed/val.jsonl

用法：
  # 默认随机 label
  python scripts/join_label.py --val_ratio 0.15

  # 用真实 label 文件
  python scripts/join_label.py --label_file labels.csv

  # 含 scenario 扰动（仅本地 stress test）
  python scripts/join_label.py --apply_scenario
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.dataset import _to_tensor
from pbc_credit.sample_builder import build_sample, encode_sample, to_jsonable
from pbc_credit.fields import ACCOUNT_TYPES


# Scenario 系统：仅用于 stress test 验证 collator/model 在 N=0/N=heavy 时不崩
SCENARIOS = [
    'balanced', 'balanced', 'balanced', 'balanced',
    'minimal',
    'no_accounts', 'no_d1', 'no_r2', 'no_r3', 'no_r4', 'no_c1',
    'no_queries', 'no_publics', 'no_obligations',
    'single_account', 'single_query',
    'heavy_accounts', 'heavy_queries', 'heavy_publics', 'heavy_obligations',
]


def _acc_type(a: dict) -> str:
    return (a.get('accountBasic', {}).get('pd01ad01', '') or '').strip()


def _dup_to_n(items: list, target: int, rng: random.Random) -> list:
    if not items:
        return []
    out = list(items)
    while len(out) < target:
        out.append(copy.deepcopy(rng.choice(items)))
    return out


def perturb_struct(struct: dict, idx: int, seed: int, scenario: str = 'balanced') -> dict:
    """对 decode 后的 sample dict 应用 scenario 扰动（仅 stress test）。

    注意：pbc_struct 是已经 encode 过的张量 dict，要扰动 list 字段需先 decode。
    简化做法：直接对 list 字段做切片/duplicate。
    """
    rng = random.Random(seed + idx)
    r = copy.deepcopy(struct)
    # reportsn 加 idx 后缀
    if 'report_id' in r:
        r['report_id'] = f'{r["report_id"]}_{idx:06d}'

    # accounts：先 flatten 再按 scenario 切片
    for t_lower in ('d1', 'r1', 'r2', 'r3', 'r4', 'c1'):
        if t_lower not in r:
            continue
        # 只能从 list 字段长度判断 N
        n = len(r.get(f'{t_lower}_numeric', []))
        if n == 0:
            continue
        keep_n = n  # default balanced keep all
        if scenario == 'no_accounts':
            keep_n = 0
        elif scenario in ('no_d1', 'no_r2', 'no_r3', 'no_r4', 'no_c1'):
            if scenario.endswith(t_lower):
                keep_n = 0
        elif scenario == 'single_account':
            keep_n = min(1, n)
        elif scenario == 'minimal':
            keep_n = rng.randint(0, 1)
        elif scenario == 'heavy_accounts':
            target = rng.randint(15, 25)
            keep_n = target  # 需 duplicate（见下）
        elif scenario == 'balanced':
            keep_n = rng.randint(min(2, n), min(8, n))
        # apply：切片或 duplicate
        if keep_n > n:  # duplicate
            for k in (f'{t_lower}_numeric', f'{t_lower}_cat_ids',
                      f'{t_lower}_cat_mask', f'{t_lower}_paystate'):
                if k in r and isinstance(r[k], list):
                    src = r[k]
                    while len(r[k]) < keep_n:
                        r[k].append(copy.deepcopy(src[rng.randint(0, max(0, len(src)-1))]))
                    if r[k][0] and isinstance(r[k][0], list):
                        # 2D，mask 同步
                        pass
            mask_field = f'{t_lower}_mask'
            if mask_field in r:
                while len(r[mask_field]) < keep_n:
                    r[mask_field].append(1)
        elif keep_n < n:
            for k in (f'{t_lower}_numeric', f'{t_lower}_cat_ids',
                      f'{t_lower}_cat_mask', f'{t_lower}_paystate', f'{t_lower}_mask'):
                if k in r and isinstance(r[k], list):
                    r[k] = r[k][:keep_n]
            if f'{t_lower}_mask' in r and isinstance(r[f'{t_lower}_mask'], list):
                r[f'{t_lower}_mask'] = [1] * keep_n if keep_n > 0 else []

    # queries / publics / obligations 切片（同样逻辑）
    for branch, fields in [
        ('query', ['query_numeric', 'query_cat_ids', 'query_mask']),
        ('public', ['public_numeric', 'public_cat_ids', 'public_mask']),
        ('obligation', ['obligation_numeric', 'obligation_cat_ids', 'obligation_mask']),
    ]:
        nums_key = f'{branch}_numeric'
        if nums_key not in r:
            continue
        n = len(r[nums_key])
        if n == 0:
            continue
        if scenario == f'no_{branch}s' or (branch == 'obligation' and scenario == 'no_obligations'):
            for k in fields:
                if k in r:
                    r[k] = []
        elif scenario == f'single_{branch}':
            for k in fields:
                if k in r and isinstance(r[k], list):
                    r[k] = r[k][:1]
        elif scenario == f'heavy_{branch}s':
            target = rng.randint(30, 60) if branch == 'query' else rng.randint(10, 20)
            for k in fields:
                if k in r and isinstance(r[k], list):
                    src = r[k]
                    while len(r[k]) < target:
                        r[k].append(copy.deepcopy(src[rng.randint(0, len(src)-1)] if src else []))
        elif scenario == 'minimal':
            for k in fields:
                if k in r and isinstance(r[k], list):
                    r[k] = r[k][:rng.randint(0, 1)]
        elif scenario == 'balanced':
            if branch == 'query':
                keep = rng.randint(min(3, n), min(20, n))
            elif branch == 'public':
                keep = rng.randint(1, min(3, n))
            else:
                keep = rng.randint(1, min(6, n))
            for k in fields:
                if k in r and isinstance(r[k], list):
                    r[k] = r[k][:keep]
        # 同步 mask
        mask_key = f'{branch}_mask'
        if mask_key in r and isinstance(r[mask_key], list):
            n_new = len(r[nums_key])
            r[mask_key] = [1] * n_new

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/pbc/processed/pbc_struct.jsonl',
                        help='simulate_spark_parse.py 输出')
    parser.add_argument('--out_dir', default='data/pbc/processed')
    parser.add_argument('--label_file', default=None,
                        help='外部 label CSV/JSONL（biz_sno,label）；不传则随机')
    parser.add_argument('--pos_ratio', type=float, default=0.2,
                        help='随机 label 正样本率（label_file 未传时）')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--apply_scenario', action='store_true',
                        help='开启 scenario 扰动（仅本地 stress test，生产不开）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # === 1. 读 pbc_struct JSONL ===
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f'{in_path} not found; run simulate_spark_parse.py first')
    print(f'=== 加载 {in_path} ===')
    samples: list[dict] = []  # 每条 {biz_sno, pbc_struct dict}
    with open(in_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outer = json.loads(line)
            biz_sno = outer['biz_sno']
            struct = json.loads(outer['pbc_struct'])
            samples.append({'biz_sno': biz_sno, 'struct': struct})
    print(f'  loaded {len(samples)} samples')

    # === 2. 读外部 label（可选）===
    label_map: dict[str, int] = {}
    if args.label_file:
        lp = Path(args.label_file)
        if lp.suffix == '.csv':
            import csv
            with open(lp, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_map[row['biz_sno']] = int(row['label'])
        else:
            with open(lp, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    label_map[row['biz_sno']] = int(row['label'])
        print(f'  loaded {len(label_map)} labels from {lp}')

    # === 3. 切 train/val ===
    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = int(len(samples) * args.val_ratio)
    val_idx = set(indices[:n_val])
    splits = {
        'train': [s for i, s in enumerate(samples) if i not in val_idx],
        'val':   [s for i, s in enumerate(samples) if i in val_idx],
    }
    print(f'  train={len(splits["train"])} val={len(splits["val"])}')

    # === 4. 写 train/val JSONL ===
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    stats: dict[str, Counter] = {'train': Counter(), 'val': Counter()}

    for split, items in splits.items():
        out_path = out_dir / f'{split}.jsonl'
        n_written = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(items):
                struct = item['struct']
                biz_sno = item['biz_sno']
                if args.apply_scenario:
                    scenario = SCENARIOS[i % len(SCENARIOS)]
                    struct = perturb_struct(struct, i, args.seed, scenario)
                    stats[split][scenario] += 1
                else:
                    stats[split]['no_perturb'] += 1

                # label
                if biz_sno in label_map:
                    label = label_map[biz_sno]
                else:
                    label = 1 if random.random() < args.pos_ratio else 0

                f.write(json.dumps({
                    'biz_sno': biz_sno,
                    'pbc_struct': json.dumps(struct, ensure_ascii=False),
                    'label': int(label),
                }, ensure_ascii=False) + '\n')
                n_written += 1
        print(f'  wrote {n_written} → {out_path}')
        print(f'  {split} 分布: {dict(stats[split])}')


if __name__ == '__main__':
    main()
