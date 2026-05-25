# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 HuggingFace Transformers 的 LLM 训练工具集，包含两个独立训练脚本：分类微调和掩码语言模型继续预训练。均改编自 Transformers 官方示例。无测试套件，无 CI/CD。

## 训练脚本

- **`run_classification.py`** — 文本分类微调（单标签 / 多标签 / 回归），使用 sklearn 做评估指标
- **`run_mlm.py`** — 掩码语言模型继续预训练（BERT、ALBERT、RoBERTa 等），支持从头训练

## 运行方式

```bash
# YAML 配置文件（推荐，支持 CLI 覆盖）
python run_classification.py configs/sft_cls_auc.yaml
python run_classification.py configs/sft_cls_auc.yaml --learning_rate 1e-4

# 多卡训练
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls_auc.yaml

# DeepSpeed
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls_auc.yaml --deepspeed configs/ds_z2.json

# 命令行参数（兼容原有方式）
python run_classification.py --model_name_or_path <model> --train_file train.csv --do_train --do_eval
```

参数传递：两个脚本均通过 `HfArgumentParser` 解析 `ModelArguments` + `DataTrainingArguments` + `TrainingArguments` 三类 dataclass。支持 YAML、JSON、命令行三种配置方式。

## 配置文件

`configs/` 目录提供常用配置模板：

- `sft_cls_auc.yaml` — 二分类 AUC 评估
- `sft_cls_multi_val.yaml` — 多验证集评估
- `pt_mlm.yaml` — MLM 继续预训练
- `ds_z2.json` / `ds_z3.json` — DeepSpeed ZeRO-2/3 配置

## 依赖

核心：`transformers>=4.57.0`、`datasets`、`torch`、`accelerate`、`scikit-learn`、`pyyaml`。详见各脚本头部 PEP 723 inline metadata 及 `requirements.txt`。`evaluate` 已替换为本地 sklearn 指标，不再依赖。

## 架构要点

### run_classification.py

- 任务类型自动推断：label 列为 float → 回归（MSE），为 list → 多标签（F1 micro），其他 → 单标签（默认 accuracy，支持 auc/ks）
- 多验证集：`--validation_files` 接受逗号分隔路径，单个文件加载为 `"validation"` split，多个加载为 `"validation_0"`, `"validation_1"` ... Trainer 对每个 key 分别评估
- label 发现：从 train + 所有 validation/test split 合并 label list，防止 val 出现 train 未见的标签

### run_mlm.py

- 两种文本处理：`--line_by_line` 逐行 tokenize；默认模式拼接后按 `max_seq_length` 分块（`group_texts`）
- 无 validation split 时自动从 train 按 `--validation_split_percentage`（默认 5%）切分
- 评估输出 perplexity（`exp(eval_loss)`）和 accuracy
- 支持从零训练（`model_name_or_path=None` + `--model_type`）和 streaming 模式

### 共性

- 使用 Transformers `Trainer` API，支持分布式训练、混合精度、checkpoint 恢复
- `processing_class=tokenizer` 传递给 Trainer（非旧版 `tokenizer=` 参数）
- 本地文件支持 CSV 和 JSON 格式；Hub 数据集通过 `dataset_name` 指定
- YAML 配置支持：`read_args()` 函数解析 YAML/JSON/CLI，支持 YAML + CLI 覆盖
- checkpoint 自动恢复：`--resume_from_checkpoint` 不指定时自动从 `output_dir` 最新 checkpoint 恢复
- `datasets` 日志级别固定为 WARNING，避免分布式环境下重复输出
