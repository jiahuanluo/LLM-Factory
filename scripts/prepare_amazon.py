"""Amazon Polarity 长文本分类数据准备

通过 hf-mirror.com 抽样 25k 训练 + 25k 测试，匹配 IMDB 规模便于对比。
输出到 data/amazon/：
  - train.csv: sentence,label（CLS 训练）
  - val.csv:   sentence,label（CLS 验证）
  - test.csv:  sentence,label（CLS 测试，复制自 val.csv）
  - pretrain_corpus.txt: 纯文本（MLM 训练，从 train 去 label）
  - val_corpus.txt: 纯文本（MLM 验证，从 val 去 label）
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import csv
import random
import shutil
from pathlib import Path

from datasets import load_dataset

random.seed(42)

OUT = Path('data/amazon')
OUT.mkdir(parents=True, exist_ok=True)


def sample(split: str, n: int):
    print(f'Sampling {n} from {split}...')
    ds = load_dataset('amazon_polarity', split=split, streaming=True)
    items = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        if i % 5000 == 0:
            print(f'  {i}/{n}')
        text = (ex['title'] + '. ' + ex['content']).strip()
        items.append((text, ex['label']))
    random.shuffle(items)
    return items


def write_csv(items, path, with_label=True):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if with_label:
            w.writerow(['sentence', 'label'])
            for text, label in items:
                w.writerow([text, label])
        else:
            w.writerow(['sentence'])
            for text, _ in items:
                w.writerow([text])


def write_txt_from_csv(csv_path, txt_path):
    """从 csv（含 sentence header）转 txt（去 header，每行一句）"""
    with open(csv_path, 'r', encoding='utf-8') as src, open(txt_path, 'w', encoding='utf-8') as dst:
        next(src)  # skip header
        for line in src:
            dst.write(line)


def length_stats(items, name):
    word_lens = [len(t.split()) for t, _ in items[:5000]]
    word_lens.sort()
    print(f'{name} 词长分布 (n={len(word_lens)}): '
          f'p50={word_lens[len(word_lens)//2]}, '
          f'p90={word_lens[int(len(word_lens)*0.9)]}, '
          f'p99={word_lens[int(len(word_lens)*0.99)]}, '
          f'max={word_lens[-1]}')


train = sample('train', 25000)
test = sample('test', 25000)

length_stats(train, 'train')
length_stats(test, 'test')

write_csv(train, OUT / 'train.csv')
write_csv(test, OUT / 'val.csv')
shutil.copy(OUT / 'val.csv', OUT / 'test.csv')

write_txt_from_csv(OUT / 'train.csv', OUT / 'pretrain_corpus.txt')
write_txt_from_csv(OUT / 'val.csv', OUT / 'val_corpus.txt')

print('Done. Output:')
for p in sorted(OUT.iterdir()):
    print(f'  {p.name} ({p.stat().st_size // 1024} KB)')
