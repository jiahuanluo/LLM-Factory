"""PBC Spark 单文件生产脚本 — raw JSON → pbc_struct 全流程。

**自包含**：所有字段定义、解析逻辑、Spark 管道都在这一个文件里，不依赖
pbc_credit 包或 torch。生产 spark-submit 直接跑。

============================================================
  字段定义来源
============================================================
与 src/pbc_credit/fields.py + sample_builder.py 保持同步。
本地改字段后需同步到此文件（注释里有「SYNC WITH」标记）。

============================================================
  用法
============================================================
1. spark-submit 直接执行（先改顶部配置常量）：
     spark-submit --files /path/to/cat_vocab.json spark_parse_pbc.py

2. PySpark / Notebook 交互（exec 整个脚本）：
     spark  # 已有的 SparkSession
     exec(open('spark_parse_pbc.py').read())

3. 本地 CLI 测试（无 Spark 也能跑）：
     python spark_parse_pbc.py --input report.json --vocab cat_vocab.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path


# ============================================================
# 配置常量（按生产环境修改）
# ============================================================
INPUT_TABLE = 'raw_pbc_reports'      # 含 raw JSON 的源表
INPUT_COLUMN = 'report_json'         # raw JSON 列名
OUTPUT_TABLE = 'pbc_reports'         # 目标表（含 pbc_struct STRING 列）
BIZ_SNO_COLUMN = 'biz_sno'           # 业务流水号列（透传 + 作 join key）
TEXT_COLUMN = 'pbc_text'             # 已有 pbc_text 列（透传）；None=不带
VOCAB_PATH = 'cat_vocab.json'        # 配合 spark-submit --files 分发
APP_NAME = 'pbc-struct-parse'


# ============================================================
# 字段定义（SYNC WITH: src/pbc_credit/fields.py）
# ============================================================

ACCOUNT_TYPES = ['D1', 'R1', 'R2', 'R3', 'R4', 'C1']

PAYSTATE_VOCAB = {
    '<PAD>': 0, '<UNK>': 1,
    '#': 2, '*': 3, 'M': 4,
    '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11,
    'B': 12, 'C': 13, 'G': 14, 'D': 15, 'Z': 16, 'N': 17, 'A': 18, 'E': 19,
}

# User cat: 12 字段（11 原 + pb040d06 职称）
USER_CAT_FIELDS = [
    ('personInfo.identity.pb01ad01', '性别代码表'),
    ('personInfo.identity.pb01ad02', '学历代码表'),
    ('personInfo.identity.pb01ad03', '学位代码表'),
    ('personInfo.identity.pb01ad04', '就业状况代码表'),
    ('personInfo.identity.pb01ad05', '世界各国和地区名称代码'),
    ('personInfo.marriage.pb020d01', '婚姻状况代码表'),
    ('personInfo.professionals.0.pb040d02', '单位性质代码表'),
    ('personInfo.professionals.0.pb040d03', '国民经济行业代码表'),
    ('personInfo.professionals.0.pb040d04', '职业代码表'),
    ('personInfo.professionals.0.pb040d05', '职务代码表'),
    ('personInfo.professionals.0.pb040d06', '职称代码表'),
    ('personInfo.residences.0.pb030d01', '居住状况代码表'),
]

# Account cat: 9 字段（6 原 + pd01ad08/09/10）
ACCOUNT_CAT_FIELDS = [
    ('pd01ad01', '个人借贷账户类型代码表'),
    ('pd01ad02', '个人借贷交易业务种类代码表'),
    ('pd01ad03', '个人借贷交易担保方式代码表'),
    ('pd01ad04', '币种代码表'),
    ('pd01ad06', '个人借贷交易还款频率代码表'),
    ('pd01ad07', '个人借贷交易担保方式代码表'),
    ('pd01ad08', '个人贷款发放形式代码表'),
    ('pd01ad09', '个人借贷交易共同借款标志代码表'),
    ('pd01ad10', '债权转移时的还款状态代码表'),
]

# Account numeric: 15 字段（8 base + 2 specialTrades + 1 age + 4 ratio）
ACCOUNT_NUMERIC_FIELDS = [
    'pd01ad05', 'pd01aj01', 'pd01aj02', 'pd01aj03',
    'pd01aj04', 'pd01as01', 'pd01bj01', 'pd01bj02',
    '__special_trades_count__', '__special_trades_amount__',
    '__account_age_years__',
    '__utilization_ratio__', '__repayment_progress__',
    '__overdue_ratio__', '__tenure_ratio__',
]

SUMMARY_TABLES = [
    ('tradeTips', True,
     ['pc02as01', 'pc02as03'],
     [('pc02ad01', '个人信贷交易提示业务类型代码表'), ('pc02ad02', '业务大类')]),
    ('recoveries', True,
     ['pc02bj01', 'pc02bj02'],
     [('pc02bd01', '个人被追偿汇总信息业务类型代码表')]),
    ('badDebit', False,
     ['pc02cj01'],
     [('pc02cs01', None)]),
    ('overdues', True,
     ['pc02dj01'],
     [('pc02dd01', '个人逾期（透支）汇总信息账户类型代码表'), ('pc02ds04', None)]),
    ('nonrevolvingLoan', False,
     ['pc02ej01', 'pc02ej02', 'pc02ej03'],
     [('pc02es01', None), ('pc02es02', None)]),
    ('revolvingCreditLoan', False,
     ['pc02fj01', 'pc02fj02', 'pc02fj03'],
     [('pc02fs01', None), ('pc02fs02', None)]),
    ('revolvingLoanAccount', False,
     ['pc02gj01', 'pc02gj02', 'pc02gj03'],
     [('pc02gs01', None), ('pc02gs02', None)]),
    ('loanCardAccount', False,
     ['pc02hj01', 'pc02hj02', 'pc02hj03', 'pc02hj04', 'pc02hj05'],
     [('pc02hs01', None), ('pc02hs02', None)]),
    ('standardLoancardAccount', False,
     ['pc02ij01', 'pc02ij02', 'pc02ij03', 'pc02ij04', 'pc02ij05'],
     [('pc02is01', None), ('pc02is02', None)]),
    ('relatedRepayDutys', True,
     ['pc02kj01', 'pc02kj02'],
     [('pc02kd01', '相关还款责任人类型代码表'),
      ('pc02kd02', '个人 借贷交易相关还款责任类型代码表')]),
    ('postpaySummary', False,
     ['pc030j01'],
     [('pc030d01', '后付费业务类型代码表')]),
    ('publics', True,
     ['pc040j01'],
     [('pc040d01', '公共信息类型代码表')]),
    ('querySummary', False,
     ['pc05bs01', 'pc05bs02', 'pc05bs03', 'pc05bs04',
      'pc05bs05', 'pc05bs06', 'pc05bs07', 'pc05bs08'],
     []),
]

QUERY_CAT_FIELDS = [
    ('ph010d01', '机构类型代码'),
    ('ph010q03', '查询原因代码表'),
]
QUERY_NUMERIC_FIELDS = ['ph010r01_days_ago']

PUBLIC_TYPES = [
    ('pco_pf01', 'taxes'), ('pco_pf02', 'judgments'),
    ('pco_pf03', 'enforcement'), ('pco_pf04', 'penalties'),
    ('pco_pf05', 'low_income_relief'), ('pco_pf06', 'interest_arrears'),
    ('pco_pf07', 'professional_qual'), ('pco_pf08', 'awards'),
]
PUBLIC_TYPE_VOCAB = {name: i for i, (_n, name) in enumerate(PUBLIC_TYPES)}

OBLIGATION_TYPES = ['agreement', 'postpay', 'related_repay']
OBLIGATION_TYPE_VOCAB = {name: i for i, name in enumerate(OBLIGATION_TYPES)}
OBLIGATIONS_CAT_FIELDS = [
    ('agreement', 'pd02ad01', '个人借贷交易业务种类代码表'),
    ('agreement', 'pd02ad02', '业务大类'),
    ('postpay', 'pe01ad01', '后付费业务类型代码表'),
    ('postpay', 'pe01ad02', '后付费业务状态代码表'),
    ('related_repay', 'pd03ad01', '相关还款责任人类型代码表'),
    ('related_repay', 'pd03ad02', '个人 借贷交易相关还款责任类型代码表'),
]
OBLIGATION_AMOUNT_FIELD = {
    'agreement': 'pd02aj01', 'postpay': 'pe01aj01', 'related_repay': None,
}
OBLIGATION_DATE_FIELD = {
    'agreement': None, 'postpay': 'pe01ar01', 'related_repay': 'pd03ar01',
}


# ============================================================
# JSON 路径取值（SYNC WITH: fields.get_path）
# ============================================================
def get_path(obj, path: str, default=None):
    if path == '':
        return default
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return default
        if part.isdigit() and isinstance(cur, list):
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else default
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
    return cur if cur is not None and cur != '' else default


# ============================================================
# 日期 / 数值工具（SYNC WITH: sample_builder）
# ============================================================
def parse_date(s):
    if s is None or s == '' or not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m', '%Y%m%d', '%Y%m%d%H%M%S'):
        try:
            return datetime.datetime.strptime(s[:len(fmt)] if 'T' in fmt else s, fmt)
        except ValueError:
            continue
    return None


def years_since(dt, ref=None):
    if dt is None:
        return float('nan')
    if ref is None:
        ref = datetime.datetime.now()
    return (ref - dt).total_seconds() / (365.25 * 86400)


def days_since(dt, ref=None):
    if dt is None:
        return float('nan')
    if ref is None:
        ref = datetime.datetime.now()
    return (ref - dt).total_seconds() / 86400


def _safe_float(v, default=float('nan')):
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _isnan(x):
    return isinstance(x, float) and x != x


def _tame(v):
    """|v|>1000 走 log1p。"""
    if v is None or _isnan(v):
        return 0.0
    if abs(v) > 1000:
        return math.copysign(math.log1p(abs(v)), v)
    return float(v)


def _report_ref_date(report):
    for path in ('tranDate', 'reportTime', 'header.request.tranDate'):
        v = get_path(report, path)
        dt = parse_date(v)
        if dt:
            return dt
    return datetime.datetime.now()


# ============================================================
# Vocab 编码（SYNC WITH: vocab.encode_value）
# ============================================================
def encode_value(branch: str, table: str, value, vocab: dict) -> int:
    if value is None or value == '':
        return 0
    if branch in ('paystate', 'public_type', 'obligation_type'):
        return vocab[branch]['<all>'].get(value, 0)
    table_vocab = vocab.get(branch, {}).get(table, {})
    return table_vocab.get(str(value).strip(), 0)


def load_vocab(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 各分支 builder（SYNC WITH: sample_builder，输出 list 而非 tensor）
# ============================================================
def build_user(report, ref, vocab):
    identity = get_path(report, 'personInfo.identity', {}) or {}
    mobiles = get_path(report, 'personInfo.identity.mobiles', []) or []
    residences = get_path(report, 'personInfo.residences', []) or []
    professionals = get_path(report, 'personInfo.professionals', []) or []
    marriage = get_path(report, 'personInfo.marriage', {}) or {}
    identity_others = get_path(report, 'header.identityOthers', []) or []

    dob = parse_date(identity.get('pb01ar01'))
    age = years_since(dob, ref) if dob else float('nan')

    latest_mobile = earliest_mobile = None
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

    job_end_dt = parse_date(professionals[0].get('pb040r02')) if professionals else None
    years_since_job_end = years_since(job_end_dt, ref) if job_end_dt else 0.0
    if _isnan(years_since_job_end):
        years_since_job_end = 0.0

    addr_dt = parse_date(residences[0].get('pb030r01')) if residences else None
    years_at_addr = years_since(addr_dt, ref) if addr_dt else 0.0
    if _isnan(years_at_addr):
        years_at_addr = 0.0

    marriage_change = _safe_float(marriage.get('pb020d02'), 0.0) if marriage else 0.0

    score = report.get('score') or {}
    score_present = 1.0 if score else 0.0
    score_value = _safe_float(score.get('pc010q01'), 0.0) if score else 0.0
    score_rank = _safe_float(score.get('pc010q02'), 0.0) if score else 0.0
    score_qcount = _safe_float(score.get('pc010s01'), 0.0) if score else 0.0
    pc010d01 = score.get('pc010d01') if score else None
    if pc010d01 and isinstance(pc010d01, str) and pc010d01.strip():
        score_num_inst = float(len([x for x in pc010d01.split(',') if x.strip()]))
    else:
        score_num_inst = 0.0

    numeric = [
        age, float(len(mobiles)), float(len(residences)), float(len(professionals)),
        1.0 if marriage else 0.0,
        years_since(earliest_mobile, ref), years_since(latest_mobile, ref),
        years_since(employer_year, ref),
        1.0 if identity.get('pb01aq01') else 0.0,
        float(len(identity_others)),
        years_since_job_end, years_at_addr, marriage_change,
        score_value, score_rank, score_qcount, score_num_inst, score_present,
    ]
    numeric = [0.0 if _isnan(v) else v for v in numeric]

    cat_ids = []
    cat_mask = []
    for path, _t in USER_CAT_FIELDS:
        v = get_path(report, path)
        cat_ids.append(v)
        cat_mask.append(0 if (v is None or v == '') else 1)

    cat_ids = [encode_value('user', t, v, vocab)
               for v, (_p, t) in zip(cat_ids, USER_CAT_FIELDS)]
    return numeric, cat_ids, cat_mask


def build_summary(report, vocab):
    sinfo = get_path(report, 'summaryInfo', {}) or {}
    nums = []
    cat_vals = []
    cmask = []

    for name, is_list, num_fields, cat_fields in SUMMARY_TABLES:
        node = sinfo.get(name)
        if node is None:
            if is_list:
                nums.append(0.0)
            for _ in num_fields:
                nums.append(0.0)
            for cf, _t in cat_fields:
                cat_vals.append(None)
                cmask.append(0)
            continue

        if is_list:
            items = node if isinstance(node, list) else []
            nums.append(float(len(items)))
            for nf in num_fields:
                total = sum(_safe_float(it.get(nf), 0.0) for it in items)
                nums.append(_tame(total))
            for cf, _t in cat_fields:
                v = items[0].get(cf) if items else None
                cat_vals.append(v)
                cmask.append(1 if items else 0)
        else:
            for nf in num_fields:
                nums.append(_tame(_safe_float(node.get(nf))))
            for cf, _t in cat_fields:
                v = node.get(cf)
                cat_vals.append(v)
                cmask.append(0 if v in (None, '') else 1)

    cat_tables = [t for _n, _l, _nf, cf in SUMMARY_TABLES for _f, t in cf]
    cat_ids = [encode_value('summary', cat_tables[i], v, vocab) if cat_tables[i] else 0
               for i, v in enumerate(cat_vals)]
    return nums, cat_ids, cmask


def _parse_paystate_to_60(account):
    det = get_path(account, 'latest5year.latest5yearDetails')
    states = []
    if det and isinstance(det, list) and len(det) > 0:
        for row in det:
            s = row.get('pd01ed01', '')
            ch = str(s).strip()[:1].upper() if s else ''
            states.append(ch)
        states = states[-60:]
        states = ['<PAD>'] * (60 - len(states)) + states
    else:
        s24 = get_path(account, 'latest24PayState.latest24state')
        if s24 and isinstance(s24, str):
            chars = list(s24.upper())[-24:]
            chars = ['<PAD>'] * (24 - len(chars)) + chars
            states = ['<PAD>'] * (60 - len(chars)) + chars
        else:
            states = ['<PAD>'] * 60

    lms = account.get('latestMonthPayState') if isinstance(account, dict) else None
    if isinstance(lms, dict):
        v = lms.get('pd01cd01')
        if v is not None and str(v).strip():
            states[-1] = str(v).strip()[:1].upper()

    out = []
    for ch in states:
        if ch == '<PAD>':
            out.append(0)
        else:
            out.append(PAYSTATE_VOCAB.get(ch, 1))
    return out


def build_accounts(report, ref, vocab):
    accs = get_path(report, 'accountInfos', []) or []
    by_type = {t: [] for t in ACCOUNT_TYPES}
    for a in accs:
        b = a.get('accountBasic', {}) or {}
        t = b.get('pd01ad01', '').strip()
        if t in by_type:
            by_type[t].append(a)

    result = {}
    for t, items in by_type.items():
        n = len(items)
        numeric = [[0.0] * len(ACCOUNT_NUMERIC_FIELDS) for _ in range(n)]
        cat_ids = [[0] * len(ACCOUNT_CAT_FIELDS) for _ in range(n)]
        cat_mask = [[0] * len(ACCOUNT_CAT_FIELDS) for _ in range(n)]
        paystate = [[0] * 60 for _ in range(n)]

        for i, a in enumerate(items):
            basic = a.get('accountBasic', {}) or {}
            latest = a.get('latestInfo', {}) or {}
            for j, f in enumerate(ACCOUNT_NUMERIC_FIELDS):
                if f == '__special_trades_count__':
                    st = a.get('specialTrades') or []
                    st = st if isinstance(st, list) else []
                    numeric[i][j] = float(len(st))
                elif f == '__special_trades_amount__':
                    st = a.get('specialTrades') or []
                    st = st if isinstance(st, list) else []
                    total = 0.0
                    for it in st:
                        if isinstance(it, dict):
                            for k, v in it.items():
                                if 'j01' in k.lower():
                                    total += _safe_float(v, 0.0)
                                    break
                    numeric[i][j] = _tame(total)
                elif f == '__account_age_years__':
                    open_dt = parse_date(basic.get('pd01ar01'))
                    y = years_since(open_dt, ref) if open_dt else 0.0
                    numeric[i][j] = 0.0 if _isnan(y) else y
                elif f == '__utilization_ratio__':
                    granted = _safe_float(basic.get('pd01aj01'), 0.0)
                    cur = _safe_float(basic.get('pd01bj01') or latest.get('pd01bj01'), 0.0)
                    numeric[i][j] = (cur / granted) if granted > 0 else 0.0
                elif f == '__repayment_progress__':
                    granted = _safe_float(basic.get('pd01aj01'), 0.0)
                    paid = _safe_float(basic.get('pd01aj03'), 0.0)
                    numeric[i][j] = (paid / granted) if granted > 0 else 0.0
                elif f == '__overdue_ratio__':
                    cur = _safe_float(basic.get('pd01bj01') or latest.get('pd01bj01'), 0.0)
                    od = _safe_float(basic.get('pd01bj02') or latest.get('pd01bj02'), 0.0)
                    numeric[i][j] = (od / cur) if cur > 0 else 0.0
                elif f == '__tenure_ratio__':
                    period = _safe_float(basic.get('pd01ad05'), 0.0)
                    used = _safe_float(basic.get('pd01as01'), 0.0)
                    numeric[i][j] = (used / period) if period > 0 else 0.0
                else:
                    v = basic.get(f) or latest.get(f)
                    fv = _safe_float(v)
                    if abs(fv) > 1000 if not _isnan(fv) else False:
                        fv = math.copysign(math.log1p(abs(fv)), fv)
                    if _isnan(fv):
                        fv = 0.0
                    numeric[i][j] = fv
            for j, (f, t_name) in enumerate(ACCOUNT_CAT_FIELDS):
                v = basic.get(f)
                cat_mask[i][j] = 0 if v in (None, '') else 1
                cat_ids[i][j] = encode_value('account', t_name, v, vocab)
            paystate[i] = _parse_paystate_to_60(a)

        mask = [1] * n
        result[t.lower()] = {
            'numeric': numeric, 'cat_ids': cat_ids,
            'cat_mask': cat_mask, 'paystate': paystate, 'mask': mask,
        }
    return result


def build_queries(report, ref, vocab):
    recs = get_path(report, 'queryRecords', []) or []
    nums, cat_ids, cmask = [], [], []
    for r in recs:
        d = parse_date(r.get('ph010r01'))
        da = days_since(d, ref)
        if _isnan(da):
            da = 0.0
        da = math.log1p(max(0.0, da))
        nums.append([da])
        cat_row = []
        cmask_row = []
        for f, _t in QUERY_CAT_FIELDS:
            v = r.get(f)
            cat_row.append(v)
            cmask_row.append(0 if v in (None, '') else 1)
        cat_ids.append([encode_value('query', QUERY_CAT_FIELDS[j][1], cat_row[j], vocab)
                        for j in range(len(cat_row))])
        cmask.append(cmask_row)
    mask = [1] * len(nums)
    return nums, cat_ids, cmask, mask


def build_publics(report, ref, vocab):
    pinfo = get_path(report, 'publicInfo', {}) or {}
    nums, cat_ids, cmask = [], [], []
    type_by_name = {name: name for _x, name in PUBLIC_TYPES}
    key_aliases = {
        'taxarrears': 'taxes', 'civiljudgements': 'judgments',
        'forceexecutions': 'enforcement', 'adminpunishments': 'penalties',
        'salvations': 'low_income_relief', 'accfunds': 'interest_arrears',
        'competences': 'professional_qual', 'adminawards': 'awards',
    }
    candidates = []
    for _xml, type_name in PUBLIC_TYPES:
        key = _xml.split('_')[-1]
        items = pinfo.get(key) or pinfo.get(key.upper()) or pinfo.get(key.lower())
        if items:
            items = [items] if isinstance(items, dict) else items
            candidates.append((items, type_name))
    seen = set()
    for k_top, v in pinfo.items():
        if k_top in seen:
            continue
        if isinstance(v, list) and v:
            alias = key_aliases.get(k_top.lower().replace('-', '').replace('_', ''))
            type_name = type_by_name.get(alias, 'taxes')
            candidates.append((v, type_name))
            seen.add(k_top)

    for items, type_name in candidates:
        for it in items:
            if not isinstance(it, dict):
                continue
            amount = 0.0
            for k, v in it.items():
                if 'j01' in k.lower():
                    amount = _safe_float(v, 0.0)
                    break
            days_ago = 0.0
            for k in it:
                if 'r01' in k.lower():
                    days_ago = days_since(parse_date(it.get(k)), ref)
                    if _isnan(days_ago):
                        days_ago = 0.0
                    days_ago = math.log1p(max(0.0, days_ago))
                    break
            nums.append([_tame(amount), days_ago])
            cat_ids.append([PUBLIC_TYPE_VOCAB.get(type_name, 0)])
            cmask.append([1])
    mask = [1] * len(nums)
    return nums, cat_ids, cmask, mask


def build_obligations(report, ref, vocab):
    sources = [
        ('agreement', report.get('agreementInfos') or []),
        ('postpay', report.get('postpays') or []),
        ('related_repay', report.get('relatedRepayDutyInfos') or []),
    ]
    cat_fields_by_type = {t: [] for t in OBLIGATION_TYPES}
    for t, f, _tb in OBLIGATIONS_CAT_FIELDS:
        cat_fields_by_type[t].append(f)

    nums, cat_ids_list, cmask = [], [], []
    for obl_type, items in sources:
        if not items or not isinstance(items, list):
            continue
        amount_field = OBLIGATION_AMOUNT_FIELD.get(obl_type)
        date_field = OBLIGATION_DATE_FIELD.get(obl_type)
        for raw in items:
            if not isinstance(raw, dict):
                continue
            node = raw.get('agreementBasic', raw) if obl_type == 'agreement' else raw
            amount = 0.0
            if amount_field:
                amount = _safe_float(node.get(amount_field), 0.0)
            if abs(amount) > 1000 if not _isnan(amount) else False:
                amount = math.copysign(math.log1p(abs(amount)), amount)
            if _isnan(amount):
                amount = 0.0
            days_ago = 0.0
            if date_field:
                d = parse_date(node.get(date_field))
                if d is not None:
                    da = days_since(d, ref)
                    if not _isnan(da):
                        days_ago = math.log1p(max(0.0, da))
            nums.append([amount, days_ago])

            cat_row = [obl_type]
            cmask_row = [1]
            for t in OBLIGATION_TYPES:
                fields = cat_fields_by_type[t]
                if t == obl_type:
                    for f in fields:
                        v = node.get(f)
                        cat_row.append(v)
                        cmask_row.append(0 if (v is None or v == '') else 1)
                else:
                    for _f in fields:
                        cat_row.append(None)
                        cmask_row.append(0)

            type_id = OBLIGATION_TYPE_VOCAB.get(cat_row[0], 0)
            field_ids = []
            for idx, (_ot, _f, table) in enumerate(OBLIGATIONS_CAT_FIELDS):
                val = cat_row[1 + idx] if 1 + idx < len(cat_row) else None
                fid = encode_value('obligation', table, val, vocab) if table else 0
                field_ids.append(fid)
            cat_ids_list.append([type_id] + field_ids)
            cmask.append(cmask_row)
    mask = [1] * len(nums)
    return nums, cat_ids_list, cmask, mask


# ============================================================
# 主入口：report dict → pbc_struct dict（list-only，无 tensor）
# ============================================================
def build_pbc_struct(report: dict, vocab: dict) -> dict:
    """把 CrisPbc.json 报告解析为 pbc_struct dict（全部 list/数值，可 json.dumps）。"""
    ref = _report_ref_date(report)
    out = {}

    u_num, u_cat, u_cmask = build_user(report, ref, vocab)
    out['user_numeric'] = u_num
    out['user_cat_ids'] = u_cat
    out['user_cat_mask'] = u_cmask

    s_num, s_cat, s_cmask = build_summary(report, vocab)
    out['summary_numeric'] = s_num
    out['summary_cat_ids'] = s_cat
    out['summary_cat_mask'] = s_cmask

    accs = build_accounts(report, ref, vocab)
    for t in ACCOUNT_TYPES:
        item = accs.get(t.lower(), {
            'numeric': [], 'cat_ids': [], 'cat_mask': [], 'paystate': [], 'mask': [],
        })
        out[f'{t.lower()}_numeric'] = item['numeric']
        out[f'{t.lower()}_cat_ids'] = item['cat_ids']
        out[f'{t.lower()}_cat_mask'] = item['cat_mask']
        out[f'{t.lower()}_paystate'] = item['paystate']
        out[f'{t.lower()}_mask'] = item['mask']

    q_num, q_cat, q_cmask, q_mask = build_queries(report, ref, vocab)
    out['query_numeric'] = q_num
    out['query_cat_ids'] = q_cat
    out['query_cat_mask'] = q_cmask
    out['query_mask'] = q_mask

    p_num, p_cat, p_cmask, p_mask = build_publics(report, ref, vocab)
    out['public_numeric'] = p_num
    out['public_cat_ids'] = p_cat
    out['public_cat_mask'] = p_cmask
    out['public_mask'] = p_mask

    o_num, o_cat, o_cmask, o_mask = build_obligations(report, ref, vocab)
    out['obligation_numeric'] = o_num
    out['obligation_cat_ids'] = o_cat
    out['obligation_cat_mask'] = o_cmask
    out['obligation_mask'] = o_mask

    out['report_id'] = report.get('reportsn', '')
    return out


def parse_report_to_struct_json(report_json_str: str, vocab: dict | None = None) -> str:
    """Spark UDF：JSON 字符串 → pbc_struct JSON 字符串。"""
    if report_json_str is None or report_json_str == '':
        return ''
    if vocab is None:
        vocab = _load_vocab_cached()
    try:
        report = json.loads(report_json_str)
        struct = build_pbc_struct(report, vocab)
        return json.dumps(struct, ensure_ascii=False)
    except Exception:
        return ''


# ============================================================
# Vocab 缓存（executor 端）
# ============================================================
_VOCAB_CACHE: dict | None = None
_VOCAB_BROADCAST = None


def _load_vocab_cached() -> dict:
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        if _VOCAB_BROADCAST is not None:
            _VOCAB_CACHE = _VOCAB_BROADCAST.value
        else:
            _VOCAB_CACHE = load_vocab(VOCAB_PATH)
    return _VOCAB_CACHE


# ============================================================
# Spark 端管道
# ============================================================
def register_udf(spark):
    """注册 UDF + broadcast vocab。返回 udf 对象。"""
    global _VOCAB_BROADCAST
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    vocab = load_vocab(VOCAB_PATH)
    _VOCAB_BROADCAST = spark.sparkContext.broadcast(vocab)
    return udf(parse_report_to_struct_json, StringType())


def run_pipeline(spark,
                 input_table: str = INPUT_TABLE,
                 output_table: str = OUTPUT_TABLE,
                 input_column: str = INPUT_COLUMN,
                 biz_sno_column: str = BIZ_SNO_COLUMN,
                 text_column: str | None = TEXT_COLUMN,
                 mode: str = 'overwrite'):
    """读 raw 表 → withColumn pbc_struct → 写新表。"""
    from pyspark.sql.functions import col
    parse_udf = register_udf(spark)

    df = spark.read.table(input_table)
    cols_to_keep = [biz_sno_column] if biz_sno_column in df.columns else []
    if text_column and text_column in df.columns:
        cols_to_keep.append(text_column)

    df_out = (
        df
        .withColumn('pbc_struct', parse_udf(col(input_column)))
        .filter(col('pbc_struct') != '')
    )
    if cols_to_keep:
        df_out = df_out.select(*cols_to_keep, 'pbc_struct')
    else:
        df_out = df_out.select('pbc_struct')

    df_out.write.mode(mode).saveAsTable(output_table)
    print(f'=== Done. {df_out.count()} rows → {output_table} ===')


# ============================================================
# CLI：本地无 Spark 测试
# ============================================================
def _local_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='CrisPbc.json 文件路径')
    parser.add_argument('--output', help='输出文件；不填则 stdout')
    parser.add_argument('--vocab', default=VOCAB_PATH)
    args = parser.parse_args()

    if not args.input:
        print('本地测试需传 --input <json 文件>', file=sys.stderr)
        sys.exit(1)

    global _VOCAB_CACHE
    _VOCAB_CACHE = load_vocab(args.vocab)

    with open(args.input, encoding='utf-8') as f:
        report_json = f.read()
    struct_str = parse_report_to_struct_json(report_json)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(struct_str)
        print(f'wrote → {args.output}')
    else:
        print(struct_str[:2000])
        if len(struct_str) > 2000:
            print(f'... (total {len(struct_str)} chars)')


# ============================================================
# 入口：spark-submit 或 python
# ============================================================
if __name__ == '__main__':
    if '--input' in sys.argv:
        # 本地测试模式
        _local_cli()
    else:
        # spark-submit 模式
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            print('PySpark not installed. For local test use: '
                  'python spark_parse_pbc.py --input <json>', file=sys.stderr)
            sys.exit(1)

        spark = (
            SparkSession.builder
            .appName(APP_NAME)
            .config('spark.sql.parquet.compression.codec', 'snappy')
            .getOrCreate()
        )
        run_pipeline(spark)
        spark.stop()
