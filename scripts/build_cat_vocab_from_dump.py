"""从 SQL dump 构建 cat_vocab_prod.json

输入：04_build_cat_vocab.sql 最后那个 SELECT 的输出，dump 成 JSONL
      每行：{"section": "user", "code_table": "性别代码表", "code_value": "1"}

输出：cat_vocab_prod.json（供 run_pbc_pretrain.py 构建 model config 的表尺寸）
      格式：
      {
        "user": {"性别代码表": {"<UNK>": 0, "1": 1, "2": 2, ...}, ...},
        "account": {...}
      }

ID 分配规则：
  - 0 = <UNK>（未知值兜底；SQL 端 COALESCE(code_id, 0) 与之对齐）
  - 1, 2, 3, ... = 按 code_value 字典序分配（与 SQL ROW_NUMBER ORDER BY code_value 一致）

注意：训练数据的 encode 全部在 SQL 端完成（JOIN jiahuanluo_ind.cat_vocab），
此 json 只用于确定各码值表的 embedding 尺寸（大小一致即可，id 无需逐一对齐）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.vocab import build_cat_vocab, save_vocab


def main():
    if len(sys.argv) != 3:
        print('Usage: python build_cat_vocab_from_dump.py <dump.jsonl> <output.json>')
        sys.exit(1)
    dump_path, output_path = sys.argv[1:3]

    vocab = build_cat_vocab(dump_path)
    save_vocab(vocab, output_path)

    # 统计
    print(f'Built {output_path}')
    for section, tables in vocab.items():
        print(f'  {section}: {len(tables)} tables')
        for t, v in tables.items():
            print(f'    {t}: {len(v)} values (incl <UNK>)')


if __name__ == '__main__':
    main()
