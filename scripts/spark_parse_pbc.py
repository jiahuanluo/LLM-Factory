"""Spark 端 PBC JSON → pbc_struct 列解析（生产参考）。

把这个脚本扔到 Spark 集群，注册 UDF，对原始 JSON 列做解析，输出 pbc_struct 字符串列。

本地测试：
  python scripts/spark_parse_pbc.py --input data/home-credit/个人征信/CrisPbc.json --output /tmp/pbc_struct.json

生产用法（PySpark）：
  from pyspark.sql.functions import udf, col
  from pyspark.sql.types import StringType
  from spark_parse_pbc import parse_report_to_struct_json

  parse_udf = udf(parse_report_to_struct_json, StringType())

  df = spark.read.table("raw_pbc_reports")
  df = df.withColumn("pbc_struct", parse_udf(col("report_json")))
  df.write.insertInto("pbc_reports")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.sample_builder import build_sample, encode_sample, to_jsonable
from pbc_credit.vocab import load_vocab, build_cat_vocab, save_vocab


# vocab 路径生产可以放到 HDFS / 共享存储
DEFAULT_VOCAB_PATH = 'data/pbc/processed/cat_vocab.json'


def parse_report_to_struct_json(report_json_str: str, vocab: dict | None = None) -> str:
    """Spark UDF 主函数：JSON 字符串 → pbc_struct JSON 字符串。

    Args:
        report_json_str: CrisPbc.json 的字符串形式
        vocab: 码值 vocab；生产端 broadcast 一次即可
    Returns:
        pbc_struct 列的字符串值（jsonable dict）
    """
    if vocab is None:
        vocab = _load_vocab_cached()
    report = json.loads(report_json_str)
    sample = build_sample(report, vocab)
    encode_sample(sample, vocab)
    return json.dumps(to_jsonable(sample), ensure_ascii=False)


_VOCAB_CACHE: dict | None = None


def _load_vocab_cached() -> dict:
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = load_vocab(DEFAULT_VOCAB_PATH)
    return _VOCAB_CACHE


def main():
    """本地测试入口：单条 JSON → pbc_struct 字符串。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CrisPbc.json 路径')
    parser.add_argument('--output', help='输出 pbc_struct JSON 路径（不填则 stdout）')
    parser.add_argument('--vocab', default=DEFAULT_VOCAB_PATH)
    args = parser.parse_args()

    if not Path(args.vocab).exists():
        print(f'vocab 不存在，从码值表构建: {args.vocab}')
        vocab = build_cat_vocab()
        save_vocab(vocab, args.vocab)
    else:
        vocab = load_vocab(args.vocab)

    with open(args.input, encoding='utf-8') as f:
        report_json = f.read()
    struct_str = parse_report_to_struct_json(report_json, vocab)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(struct_str)
        print(f'wrote → {args.output}')
    else:
        # stdout：前 2000 字符预览
        print(struct_str[:2000])
        if len(struct_str) > 2000:
            print(f'... (total {len(struct_str)} chars)')


if __name__ == '__main__':
    main()
