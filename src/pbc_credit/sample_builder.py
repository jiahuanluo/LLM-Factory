"""PBC sample builder：CrisPbc.json → flat sample dict.

一个 sample 对应一份征信报告（一个 reportsn），输出 5 模态张量。

用法：
  python -m pbc_credit.sample_builder --input xxx.json --output samples/
"""
from __future__ import annotations

import argparse
import datetime
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .fields import (
    get_path,
    USER_CAT_FIELDS, USER_NUMERIC_SPECS,
    SUMMARY_TABLES,
    ACCOUNT_TYPES, ACCOUNT_CAT_FIELDS, ACCOUNT_NUMERIC_FIELDS,
    QUERY_CAT_FIELDS, QUERY_NUMERIC_FIELDS,
    PUBLIC_TYPES, PUBLIC_TYPE_VOCAB,
    PAYSTATE_VOCAB, PAYSTATE_VOCAB_SIZE,
)
from .vocab import build_cat_vocab, save_vocab, load_vocab, encode_value


# ============================================================
# 日期解析
# ============================================================

def parse_date(s) -> datetime.datetime | None:
    """支持 'YYYY-MM-DD' / 'YYYY-MM' / 'YYYYMMDD' / 'YYYYMMDDHHMMSS'."""
    if s is None or s == '' or not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m', '%Y%m%d', '%Y%m%d%H%M%S'):
        try:
            return datetime.datetime.strptime(s[:len(fmt)] if 'T' in fmt else s, fmt)
        except ValueError:
            continue
    return None


def years_since(dt: datetime.datetime | None, ref: datetime.datetime | None = None) -> float:
    if dt is None:
        return float('nan')
    if ref is None:
        ref = datetime.datetime.now()
    return (ref - dt).total_seconds() / (365.25 * 86400)


def days_since(dt: datetime.datetime | None, ref: datetime.datetime | None = None) -> float:
    if dt is None:
        return float('nan')
    if ref is None:
        ref = datetime.datetime.now()
    return (ref - dt).total_seconds() / 86400


def _report_ref_date(report: dict) -> datetime.datetime:
    """从报告头部提取参考日期（生产环境用报告生成日期，不再 hardcode）。"""
    for path in ('tranDate', 'reportTime', 'header.request.tranDate'):
        v = get_path(report, path)
        dt = parse_date(v)
        if dt:
            return dt
    return datetime.datetime.now()


# ============================================================
# Normalizer（z-score，只对 user_numeric；account/query 用 LayerNorm）
# ============================================================

class Normalizer:
    """保存 user_numeric 字段的 mean/std，apply 时对缺失填 0。"""
    def __init__(self):
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, samples: list[dict]):
        if not samples:
            return
        specs = [name for name, _ in USER_NUMERIC_SPECS]
        for i, name in enumerate(specs):
            vals = [s['user_numeric'][i].item() for s in samples
                    if not np.isnan(s['user_numeric'][i].item())]
            if len(vals) < 2:
                self.stats[name] = (0.0, 1.0)
                continue
            mu, sigma = float(np.mean(vals)), float(np.std(vals) + 1e-6)
            self.stats[name] = (mu, sigma)

    def apply_to_list(self, samples: list[dict]):
        specs = [name for name, _ in USER_NUMERIC_SPECS]
        for s in samples:
            for i, name in enumerate(specs):
                mu, sigma = self.stats.get(name, (0.0, 1.0))
                v = s['user_numeric'][i].item()
                if np.isnan(v):
                    s['user_numeric'][i] = 0.0
                else:
                    s['user_numeric'][i] = (v - mu) / sigma


# ============================================================
# 分支解析
# ============================================================

def _safe_float(v, default=float('nan')) -> float:
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def build_user(report: dict, ref: datetime.datetime) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (numeric[U_n], cat_ids[U_c], cat_mask[U_c])。"""
    # numeric
    identity = get_path(report, 'personInfo.identity', {}) or {}
    mobiles = get_path(report, 'personInfo.identity.mobiles', []) or []
    residences = get_path(report, 'personInfo.residences', []) or []
    professionals = get_path(report, 'personInfo.professionals', []) or []
    marriage = get_path(report, 'personInfo.marriage', {}) or {}
    identity_others = get_path(report, 'header.identityOthers', []) or []

    dob = parse_date(identity.get('pb01ar01'))
    age = years_since(dob, ref) if dob else float('nan')

    latest_mobile = None
    earliest_mobile = None
    for m in mobiles:
        d = parse_date(m.get('pb01br01'))
        if d:
            if latest_mobile is None or d > latest_mobile:
                latest_mobile = d
            if earliest_mobile is None or d < earliest_mobile:
                earliest_mobile = d

    employer_year = None
    if professionals:
        y = professionals[0].get('pb040r01')
        if y:
            try:
                employer_year = datetime.datetime(int(y), 1, 1)
            except (ValueError, TypeError):
                pass

    numeric_values = [
        age,
        float(len(mobiles)),
        float(len(residences)),
        float(len(professionals)),
        1.0 if marriage else 0.0,
        years_since(earliest_mobile, ref),
        years_since(latest_mobile, ref),
        years_since(employer_year, ref),
        1.0 if identity.get('pb01aq01') else 0.0,
        float(len(identity_others)),
    ]
    numeric = torch.tensor(numeric_values, dtype=torch.float32)

    cat_ids = []
    cat_mask = []
    for path, _table in USER_CAT_FIELDS:
        v = get_path(report, path)
        cat_ids.append(v)
        cat_mask.append(0 if (v is None or v == '') else 1)
    return numeric, cat_ids, cat_mask


def build_summary(report: dict) -> tuple[list[float], list[Any], list[int]]:
    """返回 (numeric_values, cat_values, cat_mask)。"""
    import math
    sinfo = get_path(report, 'summaryInfo', {}) or {}
    nums: list[float] = []
    cats: list[Any] = []
    cmask: list[int] = []

    def _tame(v: float) -> float:
        """大数值压缩：|v|>1000 走 log1p（保留正负号）。"""
        if v is None or (isinstance(v, float) and (v != v)):  # NaN
            return 0.0
        if abs(v) > 1000:
            return math.copysign(math.log1p(abs(v)), v)
        return float(v)

    for name, is_list, num_fields, cat_fields in SUMMARY_TABLES:
        node = sinfo.get(name)
        if node is None:
            for _ in num_fields:
                nums.append(0.0)
            for _ in cat_fields:
                cats.append(None)
                cmask.append(0)
            continue

        if is_list:
            items = node if isinstance(node, list) else []
            nums.append(float(len(items)))
            for nf in num_fields:
                total = sum(_safe_float(it.get(nf), 0.0) for it in items)
                nums.append(_tame(total))
            for cf, _t in cat_fields:
                cats.append(items[0].get(cf) if items else None)
                cmask.append(1 if items else 0)
        else:
            for nf in num_fields:
                nums.append(_tame(_safe_float(node.get(nf))))
            for cf, _t in cat_fields:
                cats.append(node.get(cf))
                cmask.append(0 if node.get(cf) in (None, '') else 1)

    return nums, cats, cmask


def _parse_paystate_to_60(report_account: dict) -> list[int]:
    """从 latest5year 或 latest24PayState 提取 60 月还款状态 id 序列。"""
    # 优先 latest5year.latest5yearDetails（list of dict）
    det = get_path(report_account, 'latest5year.latest5yearDetails')
    if det and isinstance(det, list) and len(det) > 0:
        states = []
        for row in det:
            s = row.get('pd01ed01', '')
            ch = str(s).strip()[:1].upper() if s else ''
            states.append(ch)
        # take last 60
        states = states[-60:]
        # pad left to 60
        states = ['<PAD>'] * (60 - len(states)) + states
        return [PAYSTATE_VOCAB.get(ch if ch != '<PAD>' else '<PAD>', 1) for ch in states]

    # fallback: latest24PayState.latest24state (24-char string)
    s24 = get_path(report_account, 'latest24PayState.latest24state')
    if s24 and isinstance(s24, str):
        chars = list(s24.upper())[-24:]
        chars = ['<PAD>'] * (24 - len(chars)) + chars
        # left pad to 60 with PAD
        chars = ['<PAD>'] * (60 - len(chars)) + chars
        return [PAYSTATE_VOCAB.get(c, 1) if c != '<PAD>' else 0 for c in chars]

    # 全空
    return [0] * 60


def build_accounts(report: dict) -> dict[str, dict]:
    """按 pd01ad01 分桶返回 5 类账户。"""
    accs = get_path(report, 'accountInfos', []) or []
    by_type: dict[str, list[dict]] = {t: [] for t in ACCOUNT_TYPES}
    for a in accs:
        basic = a.get('accountBasic', {}) or {}
        t = basic.get('pd01ad01', '').strip()
        if t in by_type:
            by_type[t].append(a)

    result: dict[str, dict] = {}
    for t, items in by_type.items():
        n = len(items)
        numeric = torch.zeros(n, len(ACCOUNT_NUMERIC_FIELDS), dtype=torch.float32)
        cat_values: list[list[Any]] = [[] for _ in range(n)]
        cat_mask = torch.zeros(n, sum(1 for _ in ACCOUNT_CAT_FIELDS), dtype=torch.long)
        paystate = torch.zeros(n, 60, dtype=torch.long)
        mask = torch.ones(n, dtype=torch.long)

        for i, a in enumerate(items):
            basic = a.get('accountBasic', {}) or {}
            latest = a.get('latestInfo', {}) or {}
            # numeric（合并 basic + latestInfo）+ log1p 压缩大金额
            for j, f in enumerate(ACCOUNT_NUMERIC_FIELDS):
                v = basic.get(f) or latest.get(f)
                fv = _safe_float(v)
                if abs(fv) > 1000:
                    import math
                    fv = math.copysign(math.log1p(abs(fv)), fv)
                numeric[i, j] = fv
            # cat
            for j, (f, _t) in enumerate(ACCOUNT_CAT_FIELDS):
                v = basic.get(f)
                cat_values[i].append(v)
                if v not in (None, ''):
                    cat_mask[i, j] = 1
            # paystate
            paystate[i] = torch.tensor(_parse_paystate_to_60(a), dtype=torch.long)

        result[t] = {
            'numeric': numeric,
            'cat_values': cat_values,
            'cat_mask': cat_mask,
            'paystate': paystate,
            'mask': mask,
        }
    return result


def build_queries(report: dict, ref: datetime.datetime) -> tuple[list[list[float]], list[list[Any]], list[list[int]]]:
    """返回 (每条 numeric, 每条 cat_values, 每条 cat_mask)。days_ago 做 log1p 压缩。"""
    import math
    recs = get_path(report, 'queryRecords', []) or []
    nums: list[list[float]] = []
    cats: list[list[Any]] = []
    cmask: list[list[int]] = []
    for r in recs:
        days_ago = days_since(parse_date(r.get('ph010r01')), ref)
        if np.isnan(days_ago):
            days_ago = 0.0
        # log1p 压缩大数值（避免天数 1-3000 主导损失）
        days_ago = math.log1p(max(0.0, days_ago))
        nums.append([days_ago])
        cat_row = []
        cmask_row = []
        for f, _t in QUERY_CAT_FIELDS:
            v = r.get(f)
            cat_row.append(v)
            cmask_row.append(0 if v in (None, '') else 1)
        cats.append(cat_row)
        cmask.append(cmask_row)
    return nums, cats, cmask


def build_publics(report: dict, ref: datetime.datetime) -> tuple[list[list[float]], list[list[Any]], list[list[int]]]:
    """统一映射为 (type_id, amount, days_ago) 序列。

    注意：publicInfo 下子表字段名各异（PF01J01/PF02AJ01 等），这里用启发式扫描
    'j01'（金额）/ 'r01'（日期）子串。生产环境如果字段名规范，可以改为 fields.py
    里的显式字段映射（见 PUBLIC_TYPES 注释）。
    """
    import math
    pinfo = get_path(report, 'publicInfo', {}) or {}
    nums: list[list[float]] = []
    cats: list[list[Any]] = []
    cmask: list[list[int]] = []
    for xml_node, type_name in PUBLIC_TYPES:
        key = xml_node.split('_')[-1]
        items = pinfo.get(key) or pinfo.get(key.upper())
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for it in items:
            amount = 0.0
            for k, v in it.items():
                if 'j01' in k.lower():
                    amount = _safe_float(v, 0.0)
                    break
            days_ago = 0.0
            for k in it:
                if 'r01' in k.lower():
                    days_ago = days_since(parse_date(it.get(k)), ref)
                    if np.isnan(days_ago):
                        days_ago = 0.0
                    days_ago = math.log1p(max(0.0, days_ago))
                    break
            # amount 也压缩
            if abs(amount) > 1000:
                amount = math.copysign(math.log1p(abs(amount)), amount)
            nums.append([amount, days_ago])
            cats.append([type_name])
            cmask.append([1])
    return nums, cats, cmask


# ============================================================
# 主入口：单份 JSON → sample dict
# ============================================================

def build_sample(report: dict, vocab: dict | None = None) -> dict:
    """把一份 CrisPbc.json 报告解析为 sample dict。"""
    sample: dict = {}
    ref = _report_ref_date(report)

    # user
    numeric, cat_values, cat_mask = build_user(report, ref)
    sample['user_numeric'] = numeric
    sample['user_cat_ids'] = torch.tensor(
        [0] * len(cat_values), dtype=torch.long,
    )  # 占位，vocab 编码后替换
    sample['user_cat_mask'] = torch.tensor(cat_mask, dtype=torch.long)
    sample['_user_cat_values'] = cat_values  # 临时存原始值

    # summary
    s_num, s_cat, s_cmask = build_summary(report)
    sample['summary_numeric'] = torch.tensor(s_num, dtype=torch.float32)
    sample['summary_cat_ids'] = torch.tensor([0] * len(s_cat), dtype=torch.long)
    sample['summary_cat_mask'] = torch.tensor(s_cmask, dtype=torch.long)
    sample['_summary_cat_values'] = s_cat

    # accounts
    accounts = build_accounts(report)
    for t in ACCOUNT_TYPES:
        item = accounts[t]
        sample[f'{t.lower()}_numeric'] = item['numeric']
        sample[f'{t.lower()}_cat_ids'] = torch.zeros(
            item['cat_values'].__len__(),
            len(ACCOUNT_CAT_FIELDS),
            dtype=torch.long,
        )
        sample[f'{t.lower()}_cat_mask'] = item['cat_mask']
        sample[f'{t.lower()}_paystate'] = item['paystate']
        sample[f'{t.lower()}_mask'] = item['mask']
        sample[f'_acc_{t.lower()}_cat_values'] = item['cat_values']

    # queries
    q_num, q_cat, q_cmask = build_queries(report, ref)
    n_q = len(q_num)
    sample['query_numeric'] = torch.tensor(q_num, dtype=torch.float32) if q_num else torch.zeros(0, 1)
    sample['query_cat_ids'] = torch.zeros(n_q, len(QUERY_CAT_FIELDS), dtype=torch.long)
    sample['query_cat_mask'] = torch.tensor(q_cmask, dtype=torch.long) if q_cmask else torch.zeros(0, len(QUERY_CAT_FIELDS), dtype=torch.long)
    sample['query_mask'] = torch.ones(n_q, dtype=torch.long)
    sample['_query_cat_values'] = q_cat

    # publics
    p_num, p_cat, p_cmask = build_publics(report, ref)
    n_p = len(p_num)
    sample['public_numeric'] = torch.tensor(p_num, dtype=torch.float32) if p_num else torch.zeros(0, 2)
    sample['public_cat_ids'] = torch.zeros(n_p, 1, dtype=torch.long)
    sample['public_cat_mask'] = torch.tensor(p_cmask, dtype=torch.long) if p_cmask else torch.zeros(0, 1, dtype=torch.long)
    sample['public_mask'] = torch.ones(n_p, dtype=torch.long)
    sample['_public_cat_values'] = p_cat

    sample['report_id'] = report.get('reportsn', '')

    # 清理 numeric 字段：NaN → 0，inf → 0
    for k, v in list(sample.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            sample[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return sample


def encode_sample(sample: dict, vocab: dict):
    """把 _*_cat_values 临时字段编码为 *_cat_ids（在 sample 构建后调用）。"""
    # user
    sample['user_cat_ids'] = torch.tensor([
        encode_value('user', table, v, vocab)
        for v, (_p, table) in zip(sample['_user_cat_values'], USER_CAT_FIELDS)
    ], dtype=torch.long)
    del sample['_user_cat_values']

    # summary — 对齐 cat 字段（包含 table=None 的位置）
    tables = [t for _n, _l, _nf, cf in SUMMARY_TABLES for _f, t in cf]
    sample['summary_cat_ids'] = torch.tensor([
        encode_value('summary', tables[i], v, vocab) if tables[i] else 0
        for i, v in enumerate(sample['_summary_cat_values'])
    ], dtype=torch.long)
    del sample['_summary_cat_values']

    # accounts
    from .fields import ACCOUNT_TYPES, ACCOUNT_CAT_FIELDS
    acc_tables = [t for _f, t in ACCOUNT_CAT_FIELDS]
    for ty in ACCOUNT_TYPES:
        k = ty.lower()
        vals = sample.get(f'_acc_{k}_cat_values', [])
        if vals:
            sample[f'{k}_cat_ids'] = torch.tensor([
                [encode_value('account', acc_tables[j], row[j], vocab)
                 for j in range(len(row))]
                for row in vals
            ], dtype=torch.long)
        else:
            sample[f'{k}_cat_ids'] = torch.zeros(0, len(ACCOUNT_CAT_FIELDS), dtype=torch.long)
        del sample[f'_acc_{k}_cat_values']

    # queries
    q_tables = [t for _f, t in QUERY_CAT_FIELDS]
    if sample['_query_cat_values']:
        sample['query_cat_ids'] = torch.tensor([
            [encode_value('query', q_tables[j], row[j], vocab)
             for j in range(len(row))]
            for row in sample['_query_cat_values']
        ], dtype=torch.long)
    else:
        sample['query_cat_ids'] = torch.zeros(0, len(QUERY_CAT_FIELDS), dtype=torch.long)
    del sample['_query_cat_values']

    # publics
    if sample['_public_cat_values']:
        sample['public_cat_ids'] = torch.tensor([
            [PUBLIC_TYPE_VOCAB.get(row[0], 0)] for row in sample['_public_cat_values']
        ], dtype=torch.long)
    else:
        sample['public_cat_ids'] = torch.zeros(0, 1, dtype=torch.long)
    del sample['_public_cat_values']


# ============================================================
# CLI（流式写）
# ============================================================

def stream_write_samples(input_jsons: list[str], output_pkl: str, vocab: dict, label: float | None = None):
    """逐条解析 + 写 pkl（流式）。"""
    n = 0
    with open(output_pkl, 'wb') as f:
        for path in input_jsons:
            with open(path, encoding='utf-8') as fin:
                report = json.load(fin)
            sample = build_sample(report, vocab)
            encode_sample(sample, vocab)
            if label is not None:
                sample['target'] = torch.tensor([label], dtype=torch.float32)
            pickle.dump(sample, f)
            n += 1
    print(f'  wrote {n} samples → {output_pkl}')
    return n


# ============================================================
# JSON-friendly 序列化（给 Spark UDF / 数据库存储用）
# ============================================================

def to_jsonable(sample: dict) -> dict:
    """把 tensor sample 转成纯 list/dict，可直接 json.dumps。

    保留所有模态字段（去除 _ 前缀的临时字段）。
    用于：
      - Spark 端 UDF 产出 pbc_struct 列（字符串）
      - 数据库存储
      - 训练端从数据库读后反序列化
    """
    out: dict = {}
    for k, v in sample.items():
        if k.startswith('_'):
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.tolist()
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            out[k] = None
        else:
            # 其他类型（dict/list）递归不强求；保留原值
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def from_jsonable(d: dict) -> dict:
    """反向：把 jsonable dict 转回 tensor sample（供训练端用）。"""
    out: dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, list):
            # 推断 dtype：默认 long；如果是 float 则 float
            is_long = all((isinstance(x, int) and not isinstance(x, bool)) for x in v if x is not None) if v else True
            # 嵌套 list：看第 0 个元素是 list 还是数字
            if v and isinstance(v[0], list):
                # 2D+ tensor
                flat = v[0] if v[0] else [0]
                if isinstance(flat[0] if flat else 0, int):
                    out[k] = torch.tensor(v, dtype=torch.long)
                else:
                    out[k] = torch.tensor(v, dtype=torch.float32)
            elif is_long:
                out[k] = torch.tensor(v, dtype=torch.long)
            else:
                out[k] = torch.tensor(v, dtype=torch.float32)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True,
                        help='CrisPbc.json 路径列表（单文件或多文件）')
    parser.add_argument('--output', required=True, help='输出 pkl 路径')
    parser.add_argument('--vocab', default='data/pbc/processed/cat_vocab.json',
                        help='vocab.json 路径（不存在则构建）')
    parser.add_argument('--label', type=float, default=None)
    args = parser.parse_args()

    if not Path(args.vocab).exists():
        print(f'building vocab → {args.vocab}')
        vocab = build_cat_vocab()
        save_vocab(vocab, args.vocab)
    else:
        vocab = load_vocab(args.vocab)
        print(f'loaded vocab: {args.vocab}')

    stream_write_samples(args.input, args.output, vocab, args.label)


if __name__ == '__main__':
    main()
