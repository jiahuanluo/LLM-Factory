"""Vocab 加载：cat_vocab_prod.json（SQL 端 04_build_cat_vocab.sql 的 dump 聚合版）。

cat_vocab_prod.json 格式：
{
  "user": {"性别代码表": {"<UNK>": 0, "1": 1, ...}, ...},
  "account": {"机构类型代码": {...}, ...}
}

编码（cat → id）已全部在 SQL 端通过 JOIN jiahuanluo_ind.cat_vocab 完成；
Python 端只消费 vocab 大小（构建 model config 的 embedding 尺寸），
不再做 encode。

id 分配（SQL 与 Python dump 聚合一致）：0=<UNK>，1..N 按 code_value 字典序。
"""
from __future__ import annotations

import json
from pathlib import Path

from .fields import USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS


def collect_used_tables() -> dict[str, list[str]]:
    """收集代码中真正用到的码值表名，按分支分组。"""
    used = {'user': [], 'account': []}
    for _field, table in USER_CAT_FIELDS:
        if table:
            used['user'].append(table)
    for _field, table in ACCOUNT_CAT_FIELDS:
        if table:
            used['account'].append(table)
    return used


def build_cat_vocab(dump_path: str | Path) -> dict:
    """从 SQL dump JSONL（04_build_cat_vocab.sql 最后的 SELECT 输出）聚合 vocab。

    每行：{"section": "user", "code_table": "性别代码表", "code_value": "1"}
    """
    from collections import defaultdict

    vocab_data = defaultdict(lambda: defaultdict(set))
    with open(dump_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            section = row.get('section')
            table = row.get('code_table')
            value = row.get('code_value')
            if section and table and value is not None:
                vocab_data[section][table].add(str(value))

    vocab: dict[str, dict[str, dict]] = {}
    for section, tables in vocab_data.items():
        vocab[section] = {}
        for table, values in tables.items():
            table_vocab = {'<UNK>': 0}
            for i, v in enumerate(sorted(values), start=1):
                table_vocab[v] = i
            vocab[section][table] = table_vocab
    return vocab


def save_vocab(vocab: dict, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def get_vocab_size(branch: str, table: str, vocab: dict) -> int:
    """返回某分支某表 vocab 大小（含 UNK）。"""
    return len(vocab.get(branch, {}).get(table, {'<UNK>': 0}))


def encode_value(branch: str, table: str, value, vocab: dict) -> int:
    """把码值编码成 id；空值/未知都返回 0 (<UNK>)。（仅调试/验证用，训练不走这里）"""
    if value is None or value == '':
        return 0
    return vocab.get(branch, {}).get(table, {}).get(str(value).strip(), 0)
