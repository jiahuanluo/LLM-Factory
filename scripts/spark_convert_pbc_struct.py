"""单个脚本完成生产数据转换：原始 CrisPbc 报文大表 → 模型训练数据。

**自包含单文件**：粘贴进 notebook / spark-submit 直接跑，不依赖仓库其它文件
（不 import convert_mock_to_pbc_struct / src）。

数据源：brm_wyd_ods_mask.ods_marm_raw_credit_data_bdcn_v2（content = 完整报文 JSON）
输出表：erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds
  busi_sno | reportsn | pbc_struct | is_val | ds
  - pbc_struct：PbcDataset 最终格式（user 32 维 + 6 类账户 13 numeric/13 cat/60 paystate，
    含 log1p 归一化 + 19 个 report 级聚合），离线**零处理**直接给模型
  - is_val：md5(reportsn)%10==0，离线导出时按它切 train/val，无需本地切分逻辑
  - 失败行保留 {"_error":...}，导出时过滤即可

转换后离线只剩导出 + 训练：
  SELECT reportsn, pbc_struct FROM <表> WHERE ds='<ds>' AND NOT is_val
    AND pbc_struct NOT LIKE '{"_error%'   → train_prod.jsonl
  SELECT ... AND is_val AND ...            → val_prod.jsonl
  # cat_vocab_prod_<ds>.json（本脚本落盘）拷到 processed 目录供 run_pbc_pretrain 用
  python run_pbc_pretrain.py configs/pbc_pretrain.yaml

vocab 两遍扫描：pass1 集群内 distinct 码值 → driver 建 vocab（0=UNK，1..N 字典序，
与离线端约定一致）→ broadcast → pass2 转换，保证 UNK=0；vocab 同时落盘。

⚠️ 本文件内的常量与转换函数是 scripts/convert_mock_to_pbc_struct.py 的镜像
   （语义逐位一致，已交叉验证）——改 fields.py/转换器后必须同步这里并重跑
   spark_convert_pbc_struct.py --local-test。

用法（集群/notebook，spark 已就绪）：改 RUN_DATE 后执行 run_spark()
本地自检（无 pyspark 依赖）：
  python scripts/spark_convert_pbc_struct.py --local-test <某份报文.json>
"""
import json
import math
import sys
from datetime import date
from hashlib import md5

# ============================================================
# 常量（与 src/pbc_credit/fields.py 同步！）
# ============================================================

ACCOUNT_TYPES = ['D1', 'R1', 'R2', 'R3', 'R4', 'C1']

PAYSTATE_VOCAB = {
    '<PAD>': 0, '<UNK>': 1, '#': 2, '*': 3, 'M': 4, '1': 5, '2': 6, '3': 7, '4': 8,
    '5': 9, '6': 10, '7': 11, 'B': 12, 'C': 13, 'G': 14, 'D': 15, 'Z': 16, 'N': 17,
    'A': 18, 'E': 19,
}

USER_CAT_FIELDS = [
    ('pb01ad01', '性别代码表'), ('pb01ad02', '学历代码表'), ('pb01ad03', '学位代码表'),
    ('pb01ad04', '就业状况代码表'), ('pb01ad05', '世界各国和地区名称代码'),
    ('pb020d01', '婚姻状况代码表'), ('pb040d02', '单位性质代码表'),
    ('pb040d03', '国民经济行业代码表'), ('pb040d04', '职业代码表'),
    ('pb040d05', '职务代码表'), ('pb040d06', '职称代码表'), ('pb030d01', '居住状况代码表'),
]
ACCOUNT_CAT_FIELDS = [
    ('pd01ad02', '机构类型代码'), ('pd01ad03', '个人借贷交易业务种类代码表'),
    ('pd01ad04', '币种代码表'), ('pd01ad05', '个人借贷交易还款方式代码表'),
    ('pd01ad06', '个人借贷交易还款频率代码表'), ('pd01ad07', '个人借贷交易担保方式代码表'),
    ('pd01ad08', '个人贷款发放形式代码表'), ('pd01ad09', '个人借贷交易共同借款标志代码表'),
    ('pd01ad10', '债权转移时的还款状态代码表'), ('pd01bd01', '个人借贷账户状态代码表(D1账户)'),
    ('pd01bd03', '五级分类代码表'), ('pd01bd04', '区分不了账户类型是R1还是D1\\R4'),
    ('pd01cd01', '个人借贷账户状态代码表(R2/R3账户)'),
]

# account numeric 13 列中做 log1p 的下标（aj01,aj02,aj03,bj01,bj02,cj02,cj06,st_amt）
LOG1P_COLS = {0, 1, 2, 4, 5, 6, 7, 12}

# ============================================================
# 转换纯函数（镜像 convert_mock_to_pbc_struct.py，语义逐位一致）
# ============================================================

def _pdate(v):
    if not v or not isinstance(v, str) or len(v) < 10:
        return None
    # 手写解析：executor 是老版本 Python（无 date.fromisoformat，3.7 才有）
    s = v[:10]
    if len(s) != 10 or s[4] != '-' or s[7] != '-':
        return None
    try:
        y, m, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return None
        return date(y, m, d)
    except (ValueError, TypeError):
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


def build_user_numeric(r: dict, rt: date) -> list:
    person = r.get('personInfo') or {}
    ident = person.get('identity') or {}
    marriage = person.get('marriage') or {}
    profs = person.get('professionals') or []
    resis = person.get('residences') or []
    mobiles = ident.get('mobiles') or []
    prof0 = profs[0] if profs else {}
    resi0 = resis[0] if resis else {}
    header = r.get('header') or {}

    cert = _s(r.get('certNo'))
    birth_year = None
    if len(cert) >= 10 and cert[6:10].isdigit():
        birth_year = int(cert[6:10])
    else:
        dob = _pdate(ident.get('pb01ar01'))
        birth_year = dob.year if dob else None

    mob_dates = [d for d in (_pdate(m.get('pb01br01')) for m in mobiles) if d]

    return [
        float(max(0, min(100, rt.year - birth_year))) if birth_year else 0.0,
        float(len(mobiles)),
        float(len(resis)),
        float(len(profs)),
        1.0 if _s(marriage.get('pb020d01')) else 0.0,
        _years_between(rt, min(mob_dates)) if mob_dates else 0.0,
        _years_between(rt, max(mob_dates)) if mob_dates else 0.0,
        float(rt.year - int(_s(prof0.get('pb040r01'))))
        if _s(prof0.get('pb040r01')).isdigit() and len(_s(prof0.get('pb040r01'))) == 4 else 0.0,
        1.0 if _s(ident.get('pb01aq01')) else 0.0,
        float(len(header.get('identityOthers') or [])),
        _years_between(rt, _pdate(prof0.get('pb040r02')))
        if _pdate(prof0.get('pb040r02')) else 0.0,
        _years_between(rt, _pdate(resi0.get('pb030r01')))
        if _pdate(resi0.get('pb030r01')) else 0.0,
        1.0 if _s(marriage.get('pb020d01')) else 0.0,
    ]


def build_user_cat(r: dict, vocab: dict) -> tuple:
    person = r.get('personInfo') or {}
    sources = {
        **{k: person.get('identity') or {} for k in ('pb01ad01', 'pb01ad02', 'pb01ad03', 'pb01ad04', 'pb01ad05')},
        'pb020d01': person.get('marriage') or {},
        **{k: (person.get('professionals') or [{}])[0] for k in ('pb040d02', 'pb040d03', 'pb040d04', 'pb040d05', 'pb040d06')},
        'pb030d01': (person.get('residences') or [{}])[0],
    }
    ids, masks = [], []
    for f, table in USER_CAT_FIELDS:
        v = _s(sources.get(f, {}).get(f))
        masks.append(1 if v else 0)
        ids.append(vocab['user'].get(table, {}).get(v, 0) if v else 0)
    return ids, masks


def build_account(acc: dict, rt: date, vocab: dict) -> dict:
    basic = acc.get('accountBasic') or {}
    latest = acc.get('latestInfo') or {}
    mps = acc.get('latestMonthPayState') or {}
    trades = acc.get('specialTrades') or []
    atype = _s(basic.get('pd01ad01'))
    if atype not in ACCOUNT_TYPES:
        return None

    open_d, mat_d = _pdate(basic.get('pd01ar01')), _pdate(basic.get('pd01ar02'))
    age = max(0.0, min(80.0, _years_between(rt, open_d))) if open_d else 0.0
    mat = max(-50.0, min(50.0, _years_between(mat_d, rt))) if mat_d else 0.0
    st_amount = sum(_pfloat(t.get('pd01fj01')) for t in trades)

    cj02 = _pfloat(mps.get('pd01cj02'))
    cj06 = _pfloat(mps.get('pd01cj06'))
    limit = _pfloat(basic.get('pd01aj02'))
    util = max(0.0, min(1.5, cj02 / limit)) if limit > 0 and cj02 > 0 else 0.0

    numeric = [
        _pfloat(basic.get('pd01aj01')), _pfloat(basic.get('pd01aj02')),
        _pfloat(basic.get('pd01aj03')), _pfloat(basic.get('pd01as01')),
        _pfloat(latest.get('pd01bj01')), _pfloat(latest.get('pd01bj02')),
        cj02, cj06, util, age, mat, float(len(trades)), st_amount,
    ]
    numeric = [math.log1p(max(0.0, x)) if i in LOG1P_COLS else x for i, x in enumerate(numeric)]

    cat_src = {**basic, **latest, **mps}
    ids, masks = [], []
    for f, table in ACCOUNT_CAT_FIELDS:
        v = _s(cat_src.get(f))
        masks.append(1 if v else 0)
        ids.append(vocab['account'].get(table, {}).get(v, 0) if v else 0)

    rows = ((acc.get('latest5year') or {}).get('latest5yearDetails')) or []
    by_month = sorted((_s(x.get('pd01er03')), _s(x.get('pd01ed01'))) for x in rows if _s(x.get('pd01er03')))
    encoded = [PAYSTATE_VOCAB.get(ch, 1) if ch else 0 for _, ch in by_month[-60:]]
    paystate = [0] * (60 - len(encoded)) + encoded

    bd01, cd01 = _s(latest.get('pd01bd01')), _s(mps.get('pd01cd01'))
    active = bool(mps) and not bd01
    overdue_amt = max(_pfloat(latest.get('pd01bj02')), cj06 if cj06 > 0 else 0.0)
    used = cj02 if active else (_pfloat(latest.get('pd01bj01')) if bd01 in ('2', '4') else 0.0)
    overdue_months = sum(1 for x in rows if _s(x.get('pd01ed01')) and _s(x.get('pd01ed01'))[:1] in '1234567BDG')
    agg = {
        'atype': atype, 'active': active, 'bad': bd01 == '4' or cd01 == '5',
        'overdue_flag': bd01 == '2' or cd01 in ('2', '3') or cj06 > 0,
        'balance': _pfloat(latest.get('pd01bj01')), 'overdue_amt': overdue_amt,
        'aj01': _pfloat(basic.get('pd01aj01')), 'aj02': limit, 'used': used,
        'util': util, 'overdue_months': overdue_months,
    }
    return {'numeric': numeric, 'cat_ids': ids, 'cat_mask': masks, 'paystate': paystate, 'agg': agg}


def report_aggregate_features(aggs: list, r: dict, rt: date) -> list:
    lp = math.log1p
    total = len(aggs)
    d1 = [a for a in aggs if a['atype'] == 'D1']
    r2 = [a for a in aggs if a['atype'] == 'R2']
    r2_utils = [a['util'] for a in r2 if a['util'] > 0]

    def q_days(q):
        d = _pdate(_s(q.get('ph010r01')))
        return (rt - d).days if d else None

    q_pairs = [(_s(q.get('ph010q03')), q_days(q)) for q in (r.get('queryRecords') or [])]
    days = [d for _reason, d in q_pairs if d is not None and d >= 0]

    return [
        lp(total),
        (sum(1 for a in aggs if a['active']) / total) if total else 0.0,
        lp(sum(1 for a in aggs if a['bad'])),
        lp(sum(1 for a in aggs if a['overdue_flag'])),
        lp(sum(a['balance'] for a in aggs)),
        lp(sum(a['overdue_amt'] for a in aggs)),
        lp(max((a['overdue_amt'] for a in aggs), default=0.0)),
        lp(len(d1)), lp(sum(a['aj01'] for a in d1)), lp(sum(a['balance'] for a in d1)),
        lp(len(r2)), lp(sum(a['aj02'] for a in r2)), lp(sum(a['used'] for a in r2)),
        (sum(r2_utils) / len(r2_utils)) if r2_utils else 0.0,
        lp(sum(1 for a in r2 if a['util'] > 1.0)),
        lp(sum(a['overdue_months'] for a in aggs)),
        lp(sum(1 for d in days if d <= 31)),
        lp(sum(1 for d in days if d <= 730)),
        lp(sum(1 for reason, d in q_pairs if d is not None and 0 <= d <= 31 and reason in ('02', '24'))),
    ]


def transform_report(r: dict, vocab: dict) -> dict:
    rt = _pdate(_s(r.get('reportTime'))[:19]) or _pdate(_s(r.get('tranDate')))
    out = {'user_cat_ids': None, 'user_cat_mask': None}
    out['user_cat_ids'], out['user_cat_mask'] = build_user_cat(r, vocab)
    per_type = {t: [] for t in ACCOUNT_TYPES}
    for acc in r.get('accountInfos') or []:
        built = build_account(acc, rt, vocab)
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
    out['user_numeric'] = build_user_numeric(r, rt) + report_aggregate_features(aggs, r, rt)
    return out


def collect_vocab_values(r: dict) -> list:
    """单条报文 → ['section|table|value', ...]（供 pass1 集群 distinct）。"""
    vals = []
    person = r.get('personInfo') or {}
    sources = {
        **{k: person.get('identity') or {} for k in ('pb01ad01', 'pb01ad02', 'pb01ad03', 'pb01ad04', 'pb01ad05')},
        'pb020d01': person.get('marriage') or {},
        **{k: (person.get('professionals') or [{}])[0] for k in ('pb040d02', 'pb040d03', 'pb040d04', 'pb040d05', 'pb040d06')},
        'pb030d01': (person.get('residences') or [{}])[0],
    }
    for f, table in USER_CAT_FIELDS:
        v = _s(sources[f].get(f))
        if v:
            vals.append(f'user|{table}|{v}')
    for acc in r.get('accountInfos') or []:
        cat_src = {**(acc.get('accountBasic') or {}), **(acc.get('latestInfo') or {}),
                   **(acc.get('latestMonthPayState') or {})}
        for f, table in ACCOUNT_CAT_FIELDS:
            v = _s(cat_src.get(f))
            if v:
                vals.append(f'account|{table}|{v}')
    return vals


def build_vocab_from_pairs(pairs: list) -> dict:
    """['section|table|value', ...] → vocab（0=UNK，1..N 按字典序，与离线端一致）。"""
    sets = {'user': {}, 'account': {}}
    for p in pairs:
        sec, table, v = p.split('|', 2)
        sets.setdefault(sec, {}).setdefault(table, set()).add(v)
    vocab = {'user': {}, 'account': {}}
    for sec, tables in sets.items():
        for table, vs in tables.items():
            vocab[sec][table] = {'<UNK>': 0, **{v: i for i, v in enumerate(sorted(vs), start=1)}}
    return vocab


def parse_report_to_struct_json(report_json_str: str, vocab: dict) -> str:
    """JSON 字符串 → pbc_struct JSON 字符串。单条失败不拖死 job：返回 {"_error", "report_id"}。"""
    rid = ''
    try:
        report = json.loads(report_json_str)
        rid = _s(report.get('reportsn'))
        sample = transform_report(report, vocab)
        return json.dumps(sample, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({'_error': f'{type(e).__name__}: {e}'[:200], 'report_id': rid},
                          ensure_ascii=False)


# ============================================================
# Spark driver（集群上执行；本地自检走 --local-test）
# ============================================================

RUN_DATE = '20260811'            # ← 要跑的分区日期，按需修改
DST_TABLE = 'erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds'
# 首次运行前需建表：
# CREATE TABLE IF NOT EXISTS erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds (
#   busi_sno string, reportsn string, pbc_struct string, is_val boolean
# ) PARTITIONED BY (ds string) STORED AS ORC;


def run_spark():
    from pyspark.sql.functions import col, explode, udf
    from pyspark.sql.types import ArrayType, BooleanType, StringType

    str_sql = f'''select
        busi_sno, content, pbcg2.reportsn, ds
    from (
        select busi_sno, content, ds,
               get_json_object(content, '$.reportsn') as reportsn
        from brm_wyd_ods_mask.ods_marm_raw_credit_data_bdcn_v2
        where ds = '{RUN_DATE}'
          and type in ('PBCG2', 'V2_CRIS_PBC_G2', 'CRIS_PBC_G2_TOUT')
          and get_json_object(content, '$.reportsn') is not null
    ) pbcg2'''

    spark.sql('set spark.executor.memory=50g')
    spark.sql('set hive.exec.dynamic.partition.mode=nostrict')
    df = spark.sql(str_sql).cache()
    n_input = df.count()
    print(f'=== pass0 输入报文: {n_input:,} 条（ds={RUN_DATE}）===')

    # ---- pass 1：集群内 distinct 收码值 → driver 建 vocab ----
    @udf(ArrayType(StringType()))
    def collect_vocab_udf(s):
        try:
            return collect_vocab_values(json.loads(s))
        except Exception:
            return []

    pairs = (df.select(explode(collect_vocab_udf(df['content'])).alias('kv'))
               .distinct().collect())
    vocab = build_vocab_from_pairs([row['kv'] for row in pairs])
    n_values = sum(len(t) - 1 for s in vocab.values() for t in s.values())
    print(f'=== pass1 vocab: user {len(vocab["user"])} 表 + account {len(vocab["account"])} 表'
          f'（{n_values:,} 个码值）===')
    try:  # vocab 落盘（driver 本地）：run_pbc_pretrain 需要 + 混训并集用
        with open(f'cat_vocab_prod_{RUN_DATE}.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f'=== vocab 已落盘: cat_vocab_prod_{RUN_DATE}.json（随 notebook 工作目录）===')
    except OSError as e:
        print(f'warn: vocab 落盘失败（不影响转换）: {e}')

    # ---- pass 2：broadcast vocab → 转换 + is_val 切分列 ----
    vocab_bc = spark.sparkContext.broadcast(vocab)

    @udf(StringType())
    def to_struct_udf(s):
        return parse_report_to_struct_json(s, vocab_bc.value)

    @udf(BooleanType())
    def is_val_udf(rid):
        return bool(rid) and int(md5(_s(rid).encode()).hexdigest(), 16) % 10 == 0

    out = (df.withColumn('pbc_struct', to_struct_udf(df['content']))
             .withColumn('is_val', is_val_udf(df['reportsn']))
             .drop('content'))
    n_err = out.where(col('pbc_struct').like('{"_error%')).count()
    n_val = out.where(col('is_val') & ~col('pbc_struct').like('{"_error%')).count()
    print(f'=== pass2 转换完成: ok {n_input - n_err:,}（val {n_val:,} / train {n_input - n_err - n_val:,}）'
          f' | 失败 {n_err:,}（保留 _error 行便于排查）===')

    out[['busi_sno', 'reportsn', 'pbc_struct', 'is_val', 'ds']].write.mode('overwrite').insertInto(DST_TABLE)
    print(f'=== 已写入 {DST_TABLE}（ds={RUN_DATE}）===')


def local_test(path: str) -> int:
    report = json.loads(open(path, encoding='utf-8').read())
    vocab = build_vocab_from_pairs(collect_vocab_values(report))
    out = json.loads(parse_report_to_struct_json(json.dumps(report, ensure_ascii=False), vocab))
    assert '_error' not in out, out
    n_user = len(out['user_numeric'])
    print(f'user_numeric: {n_user} 维（预期 32）')
    for t in ACCOUNT_TYPES:
        rows = out.get(f'{t.lower()}_numeric') or []
        if rows:
            print(f'{t.lower()}: {len(rows)} 账户 × numeric {len(rows[0])} / cat {len(out[f"{t.lower()}_cat_ids"][0])} / paystate {len(out[f"{t.lower()}_paystate"][0])}')
            assert len(rows[0]) == 13 and len(out[f'{t.lower()}_cat_ids'][0]) == 13
    assert n_user == 32
    print('=== 本地自检通过（单文件自包含，32/13/13 维）===')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--local-test':
        sys.exit(local_test(sys.argv[2]))
    run_spark()  # 集群上直接执行；notebook 里粘贴后手动调 run_spark()
