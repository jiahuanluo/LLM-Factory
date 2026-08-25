"""Spark 物化表 dump → PbcDataset 训练格式 JSONL（pandarallel 并行版）

输入：每行 `{"reportsn": "...", "pbc_struct": "<stringified JSON>"}`
       pbc_struct 内层（P0+P1 后的格式，SQL 已 encode）：
         user_numeric: [18]
         user_cat_ids: [12]  ← SQL 已 encode
         user_cat_mask: [12]
         d1/r1/r2/r3/r4/c1: [{numeric:[10], cat_ids:[12], cat_mask:[12], paystate:[60]}]
                                                                  ↑ SQL 已 pad+encode

输出：每行 `{"reportsn": "...", "pbc_struct": "<stringified JSON>"}`
       pbc_struct 内层（PbcDataset 格式，扁平）：
         user_numeric: [18]
         user_cat_ids: [12], user_cat_mask: [12]
         d1_numeric: [[N,10]], d1_cat_ids: [[N,12]], d1_cat_mask: [[N,12]],
         d1_paystate: [[N,60]], d1_mask: [N]
         (r1/r2/r3/r4/c1 同结构)

用法：
  python scripts/postprocess_pbc_struct.py dump.jsonl ignored train.jsonl
  （第二个参数 cat_vocab.json 不再需要，保留为兼容占位）
"""
import json
import sys

import pandas as pd


ACCOUNT_TYPES = ['d1', 'r1', 'r2', 'r3', 'r4', 'c1']


def flatten_accounts(accounts_list):
    """list of {numeric, cat_ids, cat_mask, paystate} → 5 个并列数组"""
    numeric_list, cat_ids_list, cat_mask_list, paystate_list, mask_list = [], [], [], [], []
    for acc in accounts_list or []:
        numeric_list.append(acc.get('numeric', []))
        cat_ids_list.append(acc.get('cat_ids', []))
        cat_mask_list.append(acc.get('cat_mask', []))
        paystate_list.append(acc.get('paystate', []))
        mask_list.append(1)
    return numeric_list, cat_ids_list, cat_mask_list, paystate_list, mask_list


def transform_sample(sample):
    """SQL 输出 → PbcDataset 扁平格式"""
    out = {
        'user_numeric': sample.get('user_numeric', []),
        'user_cat_ids': sample.get('user_cat_ids', []),
        'user_cat_mask': sample.get('user_cat_mask', []),
    }
    for t in ACCOUNT_TYPES:
        accounts = sample.get(t) or []
        numeric_list, cat_ids_list, cat_mask_list, paystate_list, mask_list = flatten_accounts(accounts)
        out[f'{t}_numeric'] = numeric_list
        out[f'{t}_cat_ids'] = cat_ids_list
        out[f'{t}_cat_mask'] = cat_mask_list
        out[f'{t}_paystate'] = paystate_list
        out[f'{t}_mask'] = mask_list
    return out


def transform_row(pbc_struct_str):
    """单行转换：stringified JSON in → stringified JSON out，失败返回 None"""
    try:
        sample = json.loads(pbc_struct_str)
    except (json.JSONDecodeError, TypeError):
        return None
    return json.dumps(transform_sample(sample), ensure_ascii=False)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    input_path, _ignored_vocab, output_path = sys.argv[1:4]

    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    df = pd.read_json(input_path, lines=True)
    n_in = len(df)

    df['pbc_struct'] = df['pbc_struct'].parallel_apply(transform_row)
    df = df.dropna(subset=['pbc_struct'])
    n_out = len(df)

    df.to_json(output_path, orient='records', lines=True, force_ascii=False)

    print(f'\nDone: {n_in} input → {n_out} output ({n_in - n_out} skipped)')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
