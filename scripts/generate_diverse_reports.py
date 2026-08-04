"""生成多样化生产报告（用于压力测试）。

数据源：data/pbc/cris_json_split/*.json（598 份模板化 mock）
输出：data/pbc/cris_json_diverse/*.json（带强扰动的多样化报告）

扰动维度（每条随机选 1-2 个应用）：
  - heavy_accounts: 复制账户到 30-100 条
  - heavy_queries: 复制查询到 100-500 条
  - heavy_obligations: agreement/postpay/related_repay 各 10-30 条
  - heavy_publics: 每个 publicInfo 子表 5-15 条
  - drop_branch: 移除某个 top-level 分支（summaryInfo/publicInfo/queryRecords 等）
  - vary_amounts: pd01aj01/j02 等金额字段乘以 0.01-100 倍
  - vary_codes: 把 cat 字段值替换成码值表内的随机其它码
  - empty_account: 把某些账户的 numeric 字段置空
  - future_dates: 把查询日期改成未来 1-365 天
  - extreme_amount: 故意造 1e10 / 负数金额

用法：
  python scripts/generate_diverse_reports.py --n 100
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


PERTURBATIONS = [
    'heavy_accounts', 'heavy_queries', 'heavy_obligations', 'heavy_publics',
    'drop_summary', 'drop_publicinfo', 'drop_queryrecords',
    'vary_amounts', 'vary_codes', 'empty_numeric',
    'future_dates', 'extreme_amount', 'mixed',
]


def _deep_copy_list(items: list, target: int, rng: random.Random) -> list:
    if not items:
        return []
    out = list(items)
    while len(out) < target:
        out.append(copy.deepcopy(rng.choice(items)))
    return out


def perturb(report: dict, idx: int, seed: int) -> tuple[dict, list[str]]:
    """返回 (扰动后 report, 应用过的 perturbation list)。"""
    rng = random.Random(seed + idx)
    r = copy.deepcopy(report)
    r['reportsn'] = (r.get('reportsn') or '') + f'_div{idx:04d}_{rng.randint(10000, 99999)}'

    # 每条随机选 1-3 个 perturbation
    n_perturb = rng.randint(1, 3)
    chosen = rng.sample(PERTURBATIONS, n_perturb)

    for p in chosen:
        if p == 'heavy_accounts':
            accs = r.get('accountInfos', [])
            if accs:
                target = rng.randint(30, 100)
                r['accountInfos'] = _deep_copy_list(accs, target, rng)
        elif p == 'heavy_queries':
            qs = r.get('queryRecords', [])
            if qs:
                target = rng.randint(100, 500)
                r['queryRecords'] = _deep_copy_list(qs, target, rng)
        elif p == 'heavy_obligations':
            for key in ('agreementInfos', 'postpays', 'relatedRepayDutyInfos'):
                items = r.get(key) or []
                if items:
                    r[key] = _deep_copy_list(items, rng.randint(10, 30), rng)
        elif p == 'heavy_publics':
            pinfo = r.get('publicInfo', {}) or {}
            for k, v in list(pinfo.items()):
                if isinstance(v, list) and v:
                    pinfo[k] = _deep_copy_list(v, rng.randint(5, 15), rng)
            r['publicInfo'] = pinfo
        elif p == 'drop_summary':
            r.pop('summaryInfo', None)
        elif p == 'drop_publicinfo':
            r.pop('publicInfo', None)
        elif p == 'drop_queryrecords':
            r['queryRecords'] = []
        elif p == 'vary_amounts':
            for acc in r.get('accountInfos', []):
                basic = acc.get('accountBasic', {})
                for k in ('pd01aj01', 'pd01aj02', 'pd01aj03'):
                    v = basic.get(k)
                    if v and v != '':
                        try:
                            fv = float(v) * rng.uniform(0.01, 100.0)
                            basic[k] = str(int(fv))
                        except (TypeError, ValueError):
                            pass
        elif p == 'vary_codes':
            # 随机替换若干 cat 字段为 0-99 随机整数（模拟未见过的码值）
            for acc in r.get('accountInfos', []):
                basic = acc.get('accountBasic', {})
                for k in ('pd01ad02', 'pd01ad06', 'pd01ad07'):
                    if rng.random() < 0.3:
                        basic[k] = str(rng.randint(50, 200))
            for q in r.get('queryRecords', []):
                if rng.random() < 0.3:
                    q['ph010q03'] = str(rng.randint(1, 30))
        elif p == 'empty_numeric':
            # 把某些账户的 numeric 字段置空字符串
            for acc in rng.sample(r.get('accountInfos', []),
                                   min(2, len(r.get('accountInfos', [])))):
                basic = acc.get('accountBasic', {})
                for k in ('pd01aj01', 'pd01aj02'):
                    if rng.random() < 0.5:
                        basic[k] = ''
        elif p == 'future_dates':
            from datetime import datetime, timedelta
            base = datetime(2025, 6, 1) + timedelta(days=rng.randint(1, 365))
            for q in r.get('queryRecords', []):
                if rng.random() < 0.3:
                    q['ph010r01'] = base.strftime('%Y-%m-%d')
        elif p == 'extreme_amount':
            # 故意造极端金额
            for acc in rng.sample(r.get('accountInfos', []),
                                   min(1, len(r.get('accountInfos', [])))):
                basic = acc.get('accountBasic', {})
                if rng.random() < 0.5:
                    basic['pd01aj01'] = str(rng.choice([int(1e10), -1, 0, 999999999]))
        elif p == 'mixed':
            # 不做扰动，但保留分支以触发 baseline 路径
            pass

    return r, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='data/pbc/cris_json_split')
    parser.add_argument('--out_dir', default='data/pbc/cris_json_diverse')
    parser.add_argument('--n', type=int, default=100, help='生成多少份多样化报告')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(in_dir.glob('*.json'))
    if not sources:
        raise FileNotFoundError(f'no .json in {in_dir}')
    print(f'=== 从 {in_dir} 加载 {len(sources)} 份源报告 ===')

    from collections import Counter
    perturb_count = Counter()

    for i in range(args.n):
        src_path = sources[i % len(sources)]
        with open(src_path, encoding='utf-8') as f:
            base = json.load(f)
        r, applied = perturb(base, i, args.seed)
        for p in applied:
            perturb_count[p] += 1
        out_path = out_dir / f'diverse_{i:04d}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(r, f, ensure_ascii=False)

    print(f'=== Done. {args.n} diverse reports → {out_dir} ===')
    print(f'  perturb 分布: {dict(perturb_count)}')


if __name__ == '__main__':
    main()
