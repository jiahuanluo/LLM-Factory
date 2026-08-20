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
  - user 18 numeric 按 v_user 定义（certNo 出生年算年龄、手机/居住/职业计数、
    score 5 字段等）
  - account 10 numeric 按 v_all_accounts 定义；paystate 用 latest5yearDetails
    按月排序取最近 60 月、PAYSTATE_VOCAB 编码、左 pad 0
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
  # 1) 构建 vocab + 编码（一次完成；vocab 已存在时仅编码并校验无新值）
  python scripts/convert_mock_to_pbc_struct.py \
      --src data/pbc/cris_json_split_calibrated \
      --out-dir data/pbc/processed \
      --pbc-src src            # 含 src/pbc_credit 的 checkout（默认本仓库根）
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
LOG1P_COLS = {0, 1, 2, 4, 5, 9}   # aj01, aj02, aj03, bj01, bj02, special_trades_amount
AGE_COL, MAT_COL = 6, 7           # account_age_years, years_to_maturity

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
    score = r.get('score') or {}
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
        _pfloat(score.get('pc010q01')) / 1000.0,                               # score_value（÷1000 缩放）
        _pfloat(score.get('pc010q02')) / 100.0,                                # score_rank（÷100 缩放）
        _pfloat(score.get('pc010s01')),                                        # score_query_count
        float(len([x for x in _s(score.get('pc010d01')).split(',') if x])),    # score_num_institutions
        1.0 if score else 0.0,                                                 # score_present
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
    trades = acc.get('specialTrades') or []
    atype = _s(basic.get('pd01ad01'))
    if atype not in _ACCOUNT_TYPES:
        return None

    open_d, mat_d = _pdate(basic.get('pd01ar01')), _pdate(basic.get('pd01ar02'))
    age = max(0.0, min(80.0, _years_between(rt, open_d))) if open_d else 0.0
    mat = max(-50.0, min(50.0, _years_between(mat_d, rt))) if mat_d else 0.0
    st_amount = sum(_pfloat(t.get('pd01fj01')) for t in trades)

    numeric = [
        _pfloat(basic.get('pd01aj01')),
        _pfloat(basic.get('pd01aj02')),
        _pfloat(basic.get('pd01aj03')),
        _pfloat(basic.get('pd01as01')),
        _pfloat(latest.get('pd01bj01')),
        _pfloat(latest.get('pd01bj02')),
        age, mat, float(len(trades)), st_amount,
    ]
    numeric = [math.log1p(max(0.0, x)) if i in LOG1P_COLS else x
               for i, x in enumerate(numeric)]

    cat_src = {**basic, **latest}
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

    return {'numeric': numeric, 'cat_ids': ids, 'cat_mask': masks, 'paystate': paystate}


def transform_report(r: dict, user_fields, acc_fields, paystate_vocab, vocab) -> dict:
    rt = _pdate(_s(r.get('reportTime'))[:19]) or _pdate(_s(r.get('tranDate')))
    out = {
        'user_numeric': build_user_numeric(r, rt),
        'user_cat_ids': None, 'user_cat_mask': None,
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
    return out


# ---------- vocab 构建 ----------

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
            cat_src = {**(acc.get('accountBasic') or {}), **(acc.get('latestInfo') or {})}
            for f, table in acc_fields:
                v = _s(cat_src.get(f))
                if v:
                    vals['account'][table].add(v)
    vocab = {'user': {}, 'account': {}}
    for sec in ('user', 'account'):
        for table, vs in sorted(vals[sec].items()):
            vocab[sec][table] = {'<UNK>': 0, **{v: i for i, v in enumerate(sorted(vs), start=1)}}
    return vocab


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description='Mock JSON → pbc_struct 转换器（对齐 mvp_user_d1.sql）')
    ap.add_argument('--src', required=True, help='校准后 mock 报告目录')
    ap.add_argument('--out-dir', required=True, help='输出目录（processed/）')
    ap.add_argument('--pbc-src', default=str(Path(__file__).resolve().parents[1] / 'src'),
                    help='含 src/pbc_credit 的 checkout（导入 fields.py 单一事实源）')
    ap.add_argument('--vocab-name', default='cat_vocab_mock.json')
    ap.add_argument('--train-name', default='train_mock.jsonl')
    ap.add_argument('--val-name', default='val_mock.jsonl')
    ap.add_argument('--val-holdout', type=int, default=10, help='md5 %% N == 0 进 val（N=10 → 10%%）')
    args = ap.parse_args()

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
