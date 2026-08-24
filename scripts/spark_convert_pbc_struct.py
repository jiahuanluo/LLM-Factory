"""Spark 薄驱动：把原始 CrisPbc 报文大表转成 pbc_struct 特征表。

转换逻辑**单一实现**在 scripts/convert_mock_to_pbc_struct.py——本文件只做 Spark 编排
（读表 / 两遍扫描 / broadcast / UDF / 写表），不再复制任何转换函数。

数据源：brm_wyd_ods_mask.ods_marm_raw_credit_data_bdcn_v2（content = 完整报文 JSON）
输出：erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds（busi_sno, reportsn, pbc_struct, ds）

部署方式（三选一）：
  1) spark-submit --py-files scripts/convert_mock_to_pbc_struct.py scripts/spark_convert_pbc_struct.py
  2) notebook：先把 convert_mock_to_pbc_struct.py 全文粘到前面的 cell，再粘本文件
  3) 两文件放同目录后直接执行（脚本会自动加同目录到 sys.path）

vocab 两遍扫描：pass1 集群内 distinct 码值 → driver 用 conv.vocab_from_sets 建 vocab
（与离线端同一 id 排序规则）→ broadcast → pass2 转换，保证 UNK=0；vocab 同时落盘，
离线 --from-dump 模式直接消费。

用法（集群/notebook，spark 已就绪）：改 RUN_DATE 后执行 run_spark()
本地自检（无 pyspark 依赖）：
  python scripts/spark_convert_pbc_struct.py --local-test <某份报文.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# --- 单一转换实现：convert_mock_to_pbc_struct（同目录 / --py-files / notebook 已粘贴） ---
try:
    if '__file__' in globals():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_mock_to_pbc_struct as conv
except ImportError:
    conv = sys.modules.get('__main__')
    if conv is None or not hasattr(conv, 'transform_report'):
        raise SystemExit('缺少转换实现：请用 --py-files 附带 scripts/convert_mock_to_pbc_struct.py，'
                         '或先在 notebook 前置 cell 粘贴其全文')

# --- 常量：优先从仓库 fields.py 导入（driver 有仓库时）；否则用 fallback ---
try:
    _repo_src = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(_repo_src))
    from pbc_credit.fields import (  # noqa: F401
        PAYSTATE_VOCAB, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS,
    )
except (ImportError, NameError):
    # ⚠️ fallback：与 src/pbc_credit/fields.py 同步（改 fields 后必须同步这里）
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


# ============================================================
# 单条报文处理（包装 conv 的纯函数，供 UDF 调用）
# ============================================================

def parse_report_to_struct_json(report_json_str: str, vocab: dict) -> str:
    """JSON 字符串 → pbc_struct JSON 字符串。单条失败不拖死 job：返回 {"_error", "report_id"}。"""
    rid = ''
    try:
        report = json.loads(report_json_str)
        rid = conv._s(report.get('reportsn'))
        sample = conv.transform_report(report, USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS,
                                       PAYSTATE_VOCAB, vocab)
        return json.dumps(sample, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({'_error': f'{type(e).__name__}: {e}'[:200], 'report_id': rid},
                          ensure_ascii=False)


def _row_vocab_values(report_json_str: str) -> list:
    """单条报文 → ['section|table|value', ...]（供 pass1 集群 distinct）。"""
    try:
        v = conv.collect_vocab_values([json.loads(report_json_str)],
                                      USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS)
        return [f'{sec}|{table}|{val}'
                for sec in ('user', 'account')
                for table, tab in v.get(sec, {}).items()
                for val in tab if val != '<UNK>']
    except Exception:  # noqa: BLE001
        return []


# ============================================================
# Spark driver
# ============================================================

RUN_DATE = '20260811'            # ← 要跑的分区日期，按需修改
DST_TABLE = 'erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds'
# 首次运行前需建表：
# CREATE TABLE IF NOT EXISTS erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds (
#   busi_sno string, reportsn string, pbc_struct string
# ) PARTITIONED BY (ds string) STORED AS ORC;


def run_spark():
    from pyspark.sql.functions import col, explode, udf
    from pyspark.sql.types import ArrayType, StringType

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

    # ---- pass 1：集群内 distinct 收码值 → driver 建 vocab（conv.vocab_from_sets 单一排序规则） ----
    @udf(ArrayType(StringType()))
    def collect_vocab_udf(s):
        return _row_vocab_values(s)

    pairs = (df.select(explode(collect_vocab_udf(df['content'])).alias('kv'))
               .distinct().collect())
    sets = {'user': {}, 'account': {}}
    for row in pairs:
        sec, table, val = row['kv'].split('|', 2)
        sets.setdefault(sec, {}).setdefault(table, set()).add(val)
    vocab = conv.vocab_from_sets(sets)
    n_values = sum(len(t) - 1 for s in vocab.values() for t in s.values())
    print(f'=== pass1 vocab: user {len(vocab["user"])} 表 + account {len(vocab["account"])} 表'
          f'（{n_values:,} 个码值）===')
    try:  # vocab 落盘（driver 本地）：离线 --from-dump 与混训并集都依赖它
        with open(f'cat_vocab_prod_{RUN_DATE}.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f'=== vocab 已落盘: cat_vocab_prod_{RUN_DATE}.json（随 notebook 工作目录）===')
    except OSError as e:
        print(f'warn: vocab 落盘失败（不影响转换）: {e}')

    # ---- pass 2：broadcast vocab → 转换 ----
    vocab_bc = spark.sparkContext.broadcast(vocab)

    @udf(StringType())
    def to_struct_udf(s):
        return parse_report_to_struct_json(s, vocab_bc.value)

    out = df.withColumn('pbc_struct', to_struct_udf(df['content'])).drop('content')
    n_err = out.where(col('pbc_struct').like('{"_error%')).count()
    print(f'=== pass2 转换完成: ok {n_input - n_err:,} | 失败 {n_err:,}（失败行保留 _error 便于排查）===')

    out[['busi_sno', 'reportsn', 'pbc_struct', 'ds']].write.mode('overwrite').insertInto(DST_TABLE)
    print(f'=== 已写入 {DST_TABLE}（ds={RUN_DATE}）===')


def local_test(path: str) -> int:
    report = json.loads(open(path, encoding='utf-8').read())
    vocab = conv.vocab_from_sets(conv.collect_vocab_values([report], USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS))
    out = json.loads(parse_report_to_struct_json(json.dumps(report, ensure_ascii=False), vocab))
    assert '_error' not in out, out
    n_user = len(out['user_numeric'])
    print(f'user_numeric: {n_user} 维（预期 32）')
    for t in ('d1', 'r1', 'r2', 'r3', 'r4', 'c1'):
        rows = out.get(f'{t}_numeric') or []
        if rows:
            print(f'{t}: {len(rows)} 账户 × numeric {len(rows[0])} / cat {len(out[f"{t}_cat_ids"][0])} / paystate {len(out[f"{t}_paystate"][0])}')
            assert len(rows[0]) == 13 and len(out[f'{t}_cat_ids'][0]) == 13
    assert n_user == 32
    print('=== 本地自检通过（32/13/13 维，转换逻辑来自 convert_mock_to_pbc_struct 单一实现）===')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--local-test':
        sys.exit(local_test(sys.argv[2]))
    run_spark()  # 集群上直接执行；notebook 里粘贴后手动调 run_spark()
