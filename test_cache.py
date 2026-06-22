"""测试多进程下 datasets .map() 缓存是否生效。

模拟实际的 main_process_first + .map() 模式，不使用 save_to_disk。
用法:
  torchrun --nproc_per_node 2 test_cache.py
  torchrun --nproc_per_node 4 test_cache.py
"""

import os
import time

import torch
import torch.distributed as dist
from datasets import load_dataset


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 创建较大的 CSV 数据集
    tmpdir = "/tmp/test_cache_datasets"
    train_path = os.path.join(tmpdir, "train.csv")
    val_path = os.path.join(tmpdir, "val.csv")

    if rank == 0:
        os.makedirs(tmpdir, exist_ok=True)
        with open(train_path, "w") as f:
            f.write("sentence,label\n")
            for i in range(500000):
                f.write(f"this is a longer sentence number {i} with more tokens to make tokenize actually take time,{i % 2}\n")
        with open(val_path, "w") as f:
            f.write("sentence,label\n")
            for i in range(50000):
                f.write(f"val sentence number {i} with more tokens for validation,{i % 2}\n")

    dist.barrier()

    # 模拟自定义版本的加载方式
    data_files = {"train": train_path}
    raw_datasets = load_dataset("csv", data_files=data_files)
    val_ds = load_dataset("csv", data_files={"validation": val_path})
    raw_datasets["validation"] = val_ds["validation"]

    # 打印 fingerprint
    for name, ds in raw_datasets.items():
        print(f"[Rank {rank}/{world_size}] split={name}, fingerprint={ds._fingerprint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=64, truncation=True)

    # ===== 测试 1: num_proc=None =====
    print(f"\n[Rank {rank}] === Test 1: num_proc=None ===")
    start = time.time()
    if rank == 0:
        tokenized = raw_datasets.map(tokenize, batched=True, load_from_cache_file=True, desc=f"Rank {rank} [no_mp]")
        print(f"[Rank {rank}] num_proc=None took {time.time() - start:.2f}s")
    dist.barrier()
    if rank != 0:
        start2 = time.time()
        tokenized = raw_datasets.map(tokenize, batched=True, load_from_cache_file=True, desc=f"Rank {rank} [no_mp]")
        print(f"[Rank {rank}] num_proc=None took {time.time() - start2:.2f}s")
    dist.barrier()

    # ===== 测试 2: num_proc=5 =====
    print(f"\n[Rank {rank}] === Test 2: num_proc=5 ===")
    start = time.time()
    if rank == 0:
        tokenized2 = raw_datasets.map(tokenize, batched=True, num_proc=5, load_from_cache_file=True, desc=f"Rank {rank} [mp5]")
        print(f"[Rank {rank}] num_proc=5 took {time.time() - start:.2f}s")
    dist.barrier()
    if rank != 0:
        start2 = time.time()
        tokenized2 = raw_datasets.map(tokenize, batched=True, num_proc=5, load_from_cache_file=True, desc=f"Rank {rank} [mp5]")
        print(f"[Rank {rank}] num_proc=5 took {time.time() - start2:.2f}s")

    dist.barrier()

    if rank == 0:
        print("\n===== RESULT =====")
        print("Rank 1+ time >> Rank 0 time  =>  cache NOT working")
        print("Rank 1+ time ~ 0s            =>  cache working fine")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
