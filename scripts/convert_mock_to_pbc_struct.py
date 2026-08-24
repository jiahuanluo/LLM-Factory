"""Mock CrisPbc JSON → pbc_struct 训练格式转换器（对齐 mvp_user_d1.sql 语义）。

把校准后的 mock 报告（data/pbc/cris_json_split_calibrated/*.json）转成
PbcDataset 可读的 JSONL，输出与 scripts/sql/mvp_user_d1.sql + postprocess_pbc_struct.py
相同的扁平结构：

  {"reportsn": "...", "pbc_struct": "<stringified>"}
  pbc_struct 内层：
    user_numeric[18], user_cat_ids[12], user_cat_mask[12]
    {d1,r1,r2,r3,r4,c1}_numeric[[N,10]] / _cat_ids[[N,12]] / _cat_mask[[N,12]]
    / _paystate[[N,60]] / _mask[N]

与 SQL 端的语义对齐（字段表导入 src/pbc_credit/fields.py，单一事实源）：
  - user 32 numeric = 13 基础（v_user 定义，**不含 score 块**——生产覆盖率 4.7% 有域偏移）
    + 19 report 级聚合（账户计数/呆账逾期/金额和/D1、R2 专项/使用率/逾期月数/查询密度；
    公式见 fields.py USER_NUMERIC_FIELDS 注释，生产侧将来按同名公式在 v_user 补列）
  - account 13 numeric 按 v_all_accounts 定义（P1 起：+pd01cj02 已用额度、
    +pd01cj06 当前逾期、+credit_utilization=cj02/aj02 clamp[0,1.5]）；
    paystate 用 latest5yearDetails 按月排序取最近 60 月、PAYSTATE_VOCAB 编码、左 pad 0
  - account 13 cat（P1 起：+pd01cd01 活跃账户状态，码值表=个人借贷账户状态代码表(R2/R3账户)）
  - cat encode：先 --build-vocab 从语料构建 cat_vocab_mock.json
    （0=<UNK>，1..N 按 code_value 字典序，与 SQL 04_build_cat_vocab 约定一致），
    之后 --encode 用同一 vocab 保证确定性

与 SQL 端的**有意差异**（混训生产 dump 前必须同步，否则 mock 与生产量纲不一致）：
  1. 金额列做 log1p：account numeric idx 0/1/2/4/5/9
     （aj01/aj02/aj03/bj01/bj02/special_trades_amount；SQL 直出原始元值）
  2. 年份列 clamp：account_age_years∈[0,80]、years_to_maturity∈[-50,50]、age_years∈[0,100]
  3. 评分列缩放：score_value÷1000、score_rank÷100（0=缺失语义不变；原始 600-950/1-99
     量纲会主导 user 分支 MSE，其余 16 列学不动）

用法：
  # 1) 本地文件模式：构建 vocab + 编码（一次完成；vocab 已存在时仅编码并校验无新值）
  python scripts/convert_mock_to_pbc_struct.py \
      --src data/pbc/cris_json_split_calibrated \
      --out-dir data/pbc/processed \
      --pbc-src src            # 含 src/pbc_credit 的 checkout（默认本仓库根）

  # 2) Spark dump 模式：集群内已用 spark_convert_pbc_struct.py 转完，本地只做
  #    _error 过滤 + 确定性切分 + 统计（不重转换）
  python scripts/convert_mock_to_pbc_struct.py \
      --from-dump dump_prod.jsonl \
      --out-dir data/pbc/processed_prod \
      --vocab-name cat_vocab_prod.json \
      --train-name train_prod.jsonl --val-name val_prod.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from hashlib import md5
from pathlib import Path

# account numeric 列（与 fields.ACCOUNT_NUMERIC_FIELDS 对齐）中做 log1p 的下标
LOG1P_COLS = {0, 1, 2, 4, 5, 6, 7, 12}   # aj01, aj02, aj03, bj01, bj02, cj02, cj06, st_amt
# credit_utilization(idx 8) clamp [0,1.5] 不做 log1p；年份/计数列原值（年份 clamp 见 build_account）

_ACCOUNT_TYPES = ['D1', 'R1', 'R2', 'R3', 'R4', 'C1']


def _load_fields(pbc_src: str):
    sys.path.insert(0, str(Path(pbc_src).resolve()))
    from pbc_credit.fields import (  # noqa: F401
        PAYSTATE_VOCAB, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS,
    )
    return PAYSTATE_VOCAB, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS


# ---------- 基础解析 ----------

def _pdate(v):
    if not v or not isinstance(v, str) or len(v) < 10:
        return None
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


def _pfloat(v, default=0.0):
    s = str(v).strip() if v is not None else ''
    if not s or not s.replace('.', '', 1).replace('-', '', 1).isdigit():
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _years_between(d1: date, d2: date) -> float:
    return round((d1 - d2).days / 365.25, 4)


def _s(v):
    return str(v).strip() if v is not None else ''


# ---------- 报告 → 扁平 sample ----------

def build_user_numeric(r: dict, rt: date) -> list[float]:
    person = r.get('personInfo') or {}
    ident = person.get('identity') or {}
    marriage = person.get('marriage') or {}
    profs = person.get('professionals') or []
    resis = person.get('residences') or []
    mobiles = ident.get('mobiles') or []
    prof0 = profs[0] if profs else {}
    resi0 = resis[0] if resis else {}
    header = (r.get('header') or {})

    cert = _s(r.get('certNo'))
    birth_year = None
    if len(cert) >= 10 and cert[6:10].isdigit():
        birth_year = int(cert[6:10])          # SQL: SUBSTR(cert_no_mask, 7, 4)
    else:
        dob = _pdate(ident.get('pb01ar01'))
        birth_year = dob.year if dob else None

    mob_dates = [_pdate(m.get('pb01br01')) for m in mobiles]
    mob_dates = [d for d in mob_dates if d]

    return [
        float(max(0, min(100, rt.year - birth_year))) if birth_year else 0.0,  # age_years
        float(len(mobiles)),                                                   # num_mobiles
        float(len(resis)),                                                     # num_residences
        float(len(profs)),                                                     # num_professionals
        1.0 if _s(marriage.get('pb020d01')) else 0.0,                          # has_marriage
        _years_between(rt, min(mob_dates)) if mob_dates else 0.0,              # yrs_since_earliest_mobile
        _years_between(rt, max(mob_dates)) if mob_dates else 0.0,              # yrs_since_latest_mobile
        float(rt.year - int(_s(prof0.get('pb040r01'))))                        # yrs_current_employer
        if _s(prof0.get('pb040r01')).isdigit() and len(_s(prof0.get('pb040r01'))) == 4 else 0.0,
        1.0 if _s(ident.get('pb01aq01')) else 0.0,                             # has_email
        float(len(header.get('identityOthers') or [])),                        # num_identity_other_docs
        _years_between(rt, _pdate(prof0.get('pb040r02')))                      # yrs_since_prof_update
        if _pdate(prof0.get('pb040r02')) else 0.0,
        _years_between(rt, _pdate(resi0.get('pb030r01')))                      # yrs_at_address
        if _pdate(resi0.get('pb030r01')) else 0.0,
        1.0 if _s(marriage.get('pb020d01')) else 0.0,                          # marriage_record_count
    ]


def build_user_cat(r: dict, user_fields, vocab) -> tuple[list[int], list[int]]:
    person = r.get('personInfo') or {}
    sources = {
        **{k: person.get('identity') or {} for k in ('pb01ad01', 'pb01ad02', 'pb01ad03', 'pb01ad04', 'pb01ad05')},
        **{k: person.get('marriage') or {} for k in ('pb020d01',)},
        **{k: (person.get('professionals') or [{}])[0] for k in ('pb040d02', 'pb040d03', 'pb040d04', 'pb040d05', 'pb040d06')},
        **{k: (person.get('residences') or [{}])[0] for k in ('pb030d01',)},
    }
    ids, masks = [], []
    for f, table in user_fields:
        v = _s(sources.get(f, {}).get(f))
        masks.append(1 if v else 0)
        ids.append(vocab['user'].get(table, {}).get(v, 0) if v else 0)
    return ids, masks


def build_account(acc: dict, rt: date, acc_fields, vocab, paystate_vocab) -> dict | None:
    basic = acc.get('accountBasic') or {}
    latest = acc.get('latestInfo') or {}
    mps = acc.get('latestMonthPayState') or {}
    trades = acc.get('specialTrades') or []
    atype = _s(basic.get('pd01ad01'))
    if atype not in _ACCOUNT_TYPES:
        return None

    open_d, mat_d = _pdate(basic.get('pd01ar01')), _pdate(basic.get('pd01ar02'))
    age = max(0.0, min(80.0, _years_between(rt, open_d))) if open_d else 0.0
    mat = max(-50.0, min(50.0, _years_between(mat_d, rt))) if mat_d else 0.0
    st_amount = sum(_pfloat(t.get('pd01fj01')) for t in trades)

    cj02 = _pfloat(mps.get('pd01cj02'))       # 已用额度（活跃账户才有）
    cj06 = _pfloat(mps.get('pd01cj06'))       # 当前逾期总额
    limit = _pfloat(basic.get('pd01aj02'))
    util = max(0.0, min(1.5, cj02 / limit)) if limit > 0 and cj02 > 0 else 0.0

    # report 级聚合原料（transform_report 汇总成 19 个 user 聚合特征，公式见 fields.py 注释）
    bd01, cd01 = _s(latest.get('pd01bd01')), _s(mps.get('pd01cd01'))
    active = bool(mps) and not bd01
    bad = bd01 == '4' or cd01 == '5'
    overdue_flag = bd01 == '2' or cd01 in ('2', '3') or cj06 > 0
    overdue_amt = max(_pfloat(latest.get('pd01bj02')), cj06 if cj06 > 0 else 0.0)
    used = cj02 if active else (_pfloat(latest.get('pd01bj01')) if bd01 in ('2', '4') else 0.0)
    rows = ((acc.get('latest5year') or {}).get('latest5yearDetails')) or []
    overdue_months = sum(1 for x in rows if _s(x.get('pd01ed01'))[:1] in '1234567BDG' and _s(x.get('pd01ed01')))
    agg = {
        'atype': atype, 'active': active, 'bad': bad, 'overdue_flag': overdue_flag,
        'balance': _pfloat(latest.get('pd01bj01')), 'overdue_amt': overdue_amt,
        'aj01': _pfloat(basic.get('pd01aj01')), 'aj02': limit, 'used': used,
        'util': util, 'overdue_months': overdue_months,
    }

    numeric = [
        _pfloat(basic.get('pd01aj01')),
        _pfloat(basic.get('pd01aj02')),
        _pfloat(basic.get('pd01aj03')),
        _pfloat(basic.get('pd01as01')),
        _pfloat(latest.get('pd01bj01')),
        _pfloat(latest.get('pd01bj02')),
        cj02, cj06, util,
        age, mat, float(len(trades)), st_amount,
    ]
    numeric = [math.log1p(max(0.0, x)) if i in LOG1P_COLS else x
               for i, x in enumerate(numeric)]

    cat_src = {**basic, **latest, **mps}
    ids, masks = [], []
    for f, table in acc_fields:
        v = _s(cat_src.get(f))
        masks.append(1 if v else 0)
        ids.append(vocab['account'].get(table, {}).get(v, 0) if v else 0)

    # paystate：latest5yearDetails 按月升序取最近 60，PAYSTATE_VOCAB 编码，左 pad 0
    rows = ((acc.get('latest5year') or {}).get('latest5yearDetails')) or []
    by_month = sorted((_s(x.get('pd01er03')), _s(x.get('pd01ed01'))) for x in rows if _s(x.get('pd01er03')))
    encoded = [paystate_vocab.get(ch, 1) if ch else 0 for _, ch in by_month[-60:]]
    paystate = [0] * (60 - len(encoded)) + encoded

    return {'numeric': numeric, 'cat_ids': ids, 'cat_mask': masks, 'paystate': paystate, 'agg': agg}


def _report_aggregate_features(aggs: list, r: dict, rt: date) -> list:
    """19 个 report 级统计衍生特征（公式与 fields.py USER_NUMERIC_FIELDS 注释一一对应）。

    金额与大计数 log1p；比率/小计数原值。空报告全 0。
    """
    lp = math.log1p
    total = len(aggs)
    d1 = [a for a in aggs if a['atype'] == 'D1']
    r2 = [a for a in aggs if a['atype'] == 'R2']
    r2_utils = [a['util'] for a in r2 if a['util'] > 0]
    queries = r.get('queryRecords') or []

    def q_days(q):
        d = _pdate(_s(q.get('ph010r01')))
        return (rt - d).days if d else None

    q_pairs = [(_s(q.get('ph010q03')), q_days(q)) for q in queries]
    days = [d for _reason, d in q_pairs if d is not None and d >= 0]

    return [
        lp(total),
        (sum(1 for a in aggs if a['active']) / total) if total else 0.0,
        lp(sum(1 for a in aggs if a['bad'])),
        lp(sum(1 for a in aggs if a['overdue_flag'])),
        lp(sum(a['balance'] for a in aggs)),
        lp(sum(a['overdue_amt'] for a in aggs)),
        lp(max((a['overdue_amt'] for a in aggs), default=0.0)),
        lp(len(d1)),
        lp(sum(a['aj01'] for a in d1)),
        lp(sum(a['balance'] for a in d1)),
        lp(len(r2)),
        lp(sum(a['aj02'] for a in r2)),
        lp(sum(a['used'] for a in r2)),
        (sum(r2_utils) / len(r2_utils)) if r2_utils else 0.0,
        lp(sum(1 for a in r2 if a['util'] > 1.0)),
        lp(sum(a['overdue_months'] for a in aggs)),
        lp(sum(1 for d in days if d <= 31)),
        lp(sum(1 for d in days if d <= 730)),
        lp(sum(1 for reason, d in q_pairs if d is not None and 0 <= d <= 31 and reason in ('02', '24'))),
    ]


def transform_report(r: dict, user_fields, acc_fields, paystate_vocab, vocab) -> dict:
    rt = _pdate(_s(r.get('reportTime'))[:19]) or _pdate(_s(r.get('tranDate')))
    out = {
        'user_numeric': None, 'user_cat_ids': None, 'user_cat_mask': None,
    }
    out['user_cat_ids'], out['user_cat_mask'] = build_user_cat(r, user_fields, vocab)
    per_type = {t: [] for t in _ACCOUNT_TYPES}
    for acc in r.get('accountInfos') or []:
        built = build_account(acc, rt, acc_fields, vocab, paystate_vocab)
        if built is not None:
            per_type[_s((acc.get('accountBasic') or {}).get('pd01ad01'))].append(built)
    for t, lst in per_type.items():
        tl = t.lower()
        out[f'{tl}_numeric'] = [a['numeric'] for a in lst]
        out[f'{tl}_cat_ids'] = [a['cat_ids'] for a in lst]
        out[f'{tl}_cat_mask'] = [a['cat_mask'] for a in lst]
        out[f'{tl}_paystate'] = [a['paystate'] for a in lst]
        out[f'{tl}_mask'] = [1] * len(lst)
    aggs = [a['agg'] for lst in per_type.values() for a in lst]
    out['user_numeric'] = build_user_numeric(r, rt) + _report_aggregate_features(aggs, r, rt)
    return out


# ---------- vocab 构建 ----------

def vocab_from_sets(vals: dict) -> dict:
    """{'user': {table: set(values)}, 'account': {...}} → vocab（0=<UNK>，1..N 按字典序）。

    id 排序规则的单一来源：本地 collect_vocab_values 与 Spark 端（spark_convert_pbc_struct.py）
    都走这里，保证两端 id 空间一致。
    """
    vocab = {'user': {}, 'account': {}}
    for sec in ('user', 'account'):
        for table, vs in sorted(vals.get(sec, {}).items()):
            vocab[sec][table] = {'<UNK>': 0, **{v: i for i, v in enumerate(sorted(vs), start=1)}}
    return vocab


def collect_vocab_values(reports: list, user_fields, acc_fields) -> dict:
    vals = {'user': defaultdict(set), 'account': defaultdict(set)}
    for r in reports:
        person = r.get('personInfo') or {}
        sources = {
            **{k: person.get('identity') or {} for k in ('pb01ad01', 'pb01ad02', 'pb01ad03', 'pb01ad04', 'pb01ad05')},
            'pb020d01': person.get('marriage') or {},
            **{k: (person.get('professionals') or [{}])[0] for k in ('pb040d02', 'pb040d03', 'pb040d04', 'pb040d05', 'pb040d06')},
            'pb030d01': (person.get('residences') or [{}])[0],
        }
        for f, table in user_fields:
            v = _s(sources[f].get(f))
            if v:
                vals['user'][table].add(v)
        for acc in r.get('accountInfos') or []:
            cat_src = {**(acc.get('accountBasic') or {}), **(acc.get('latestInfo') or {}),
                       **(acc.get('latestMonthPayState') or {})}
            for f, table in acc_fields:
                v = _s(cat_src.get(f))
                if v:
                    vals['account'][table].add(v)
    return vocab_from_sets(vals)


# ---------- main ----------

def run_from_dump(args) -> int:
    """Spark 结果表导出 JSONL → 过滤 _error → md5 确定性切分 → train/val + 统计。

    pbc_struct 已是最终格式（Spark UDF 用同一套转换逻辑产出），本模式**不做任何重转换**；
    vocab 不重建——用 Spark pass1 落盘的同名 vocab 文件（缺失时告警）。
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / args.vocab_name).exists():
        print(f'warn: {out_dir / args.vocab_name} 不存在——训练需要 Spark pass1 落盘的 vocab，'
              f'请拷贝到该路径（id 空间必须与转换时一致）')

    n_ok = n_err = n_train = n_val = 0
    acc_counts = Counter()
    train_f = open(out_dir / args.train_name, 'w', encoding='utf-8')
    val_f = open(out_dir / args.val_name, 'w', encoding='utf-8')
    with open(args.from_dump, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outer = json.loads(line)
            struct = outer.get('pbc_struct')
            if not isinstance(struct, str) or not struct:
                continue
            if struct.lstrip().startswith('{"_error'):
                n_err += 1
                continue
            rid = _s(outer.get('reportsn')) or _s(outer.get('busi_sno'))
            payload = {'reportsn': rid, 'pbc_struct': struct}
            if _s(outer.get('busi_sno')):
                payload['busi_sno'] = _s(outer['busi_sno'])   # 留给标签 join
            f_out = val_f if int(md5(rid.encode()).hexdigest(), 16) % args.val_holdout == 0 else train_f
            f_out.write(json.dumps(payload, ensure_ascii=False) + '\n')
            if f_out is train_f:
                n_train += 1
            else:
                n_val += 1
            n_ok += 1
            sample = json.loads(struct)
            for t in _ACCOUNT_TYPES:
                acc_counts[t] += len(sample.get(f'{t.lower()}_mask') or [])
    train_f.close()
    val_f.close()

    print(f'完成: ok {n_ok}（train {n_train} / val {n_val}，holdout 1/{args.val_holdout}）| _error 过滤 {n_err}')
    print(f'账户分布: {dict(acc_counts)}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='Mock JSON → pbc_struct 转换器（对齐 mvp_user_d1.sql）')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--src', help='CrisPbc JSON 报文目录（本地文件模式，全量转换）')
    g.add_argument('--from-dump', dest='from_dump',
                   help='Spark 结果表导出的 JSONL（每行含 reportsn/pbc_struct[+busi_sno/ds]）：'
                        '只做 _error 过滤 + 确定性切分 + 统计，不重转换')
    ap.add_argument('--out-dir', required=True, help='输出目录（processed/）')
    ap.add_argument('--pbc-src', default=str(Path(__file__).resolve().parents[1] / 'src'),
                    help='含 src/pbc_credit 的 checkout（导入 fields.py 单一事实源）')
    ap.add_argument('--vocab-name', default='cat_vocab_mock.json')
    ap.add_argument('--train-name', default='train_mock.jsonl')
    ap.add_argument('--val-name', default='val_mock.jsonl')
    ap.add_argument('--val-holdout', type=int, default=10, help='md5 %% N == 0 进 val（N=10 → 10%%）')
    args = ap.parse_args()

    if args.from_dump:
        return run_from_dump(args)

    paystate_vocab, user_fields, acc_fields = _load_fields(args.pbc_src)

    paths = sorted(Path(args.src).glob('*.json'))
    reports = []
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f'SKIP {p.name}: JSON 解析失败')
            continue
        if not isinstance(d, dict) or 'accountInfos' not in d or 'personInfo' not in d:
            print(f'SKIP {p.name}: 离群文件')
            continue
        reports.append((p.name, d))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / args.vocab_name
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
        fresh = collect_vocab_values([d for _, d in reports], user_fields, acc_fields)
        unseen = {}
        for sec, tabs in fresh.items():
            for t, table in tabs.items():
                new_vals = {v for v in table if v != '<UNK>'} - set(vocab.get(sec, {}).get(t, {}))
                if new_vals:
                    unseen.setdefault(sec, {})[t] = sorted(new_vals)
        if any(unseen.values()):
            print(f'ERROR 语料出现 vocab 未见过的值（先删 {vocab_path} 重建）: {unseen}')
            return 1
    else:
        vocab = collect_vocab_values([d for _, d in reports], user_fields, acc_fields)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'vocab 已构建: {vocab_path}（user {len(vocab["user"])} 表 / account {len(vocab["account"])} 表）')

    n_train = n_val = 0
    acc_counts = Counter()
    unk_cells = Counter()
    train_f = open(out_dir / args.train_name, 'w', encoding='utf-8')
    val_f = open(out_dir / args.val_name, 'w', encoding='utf-8')
    for name, d in reports:
        sample = transform_report(d, user_fields, acc_fields, paystate_vocab, vocab)
        rid = _s(d.get('reportsn')) or name
        line = json.dumps({'reportsn': rid, 'pbc_struct': json.dumps(sample, ensure_ascii=False)},
                          ensure_ascii=False)
        f = val_f if int(md5(name.encode()).hexdigest(), 16) % args.val_holdout == 0 else train_f
        f.write(line + '\n')
        if f is train_f:
            n_train += 1
        else:
            n_val += 1
        for t in _ACCOUNT_TYPES:
            tl = t.lower()
            acc_counts[t] += len(sample[f'{tl}_mask'])
            for ids, masks in zip(sample[f'{tl}_cat_ids'], sample[f'{tl}_cat_mask']):
                unk_cells[t] += sum(1 for cid, m in zip(ids, masks) if m == 1 and cid == 0)
    train_f.close()
    val_f.close()

    print(f'完成: train {n_train} / val {n_val}（holdout 1/{args.val_holdout}）')
    print(f'账户分布: {dict(acc_counts)}')
    print(f'cat UNK（mask=1 但 id=0）: {dict(unk_cells) if unk_cells else "0（全覆盖）"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
