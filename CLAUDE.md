# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 HuggingFace Transformers 的 LLM 微调工具集，包含两个训练脚本，均改编自 Transformers 官方示例。

## 训练脚本

- **`run_classification.py`** — 文本分类微调（支持单标签、多标签、回归任务）
- **`run_mlm.py`** — 掩码语言模型（MLM）微调（适用于 BERT、ALBERT、RoBERTa 等）

两个脚本均使用 `HfArgumentParser` 解析三类参数：`ModelArguments`、`DataTrainingArguments`、`TrainingArguments`。支持通过命令行参数或 JSON 配置文件传参。

## 运行方式

```bash
# 文本分类
python run_classification.py --model_name_or_path <model> --dataset_name <dataset> --do_train --do_eval

# 掩码语言模型
python run_mlm.py --model_name_or_path <model> --dataset_name <dataset> --do_train --do_eval

# 使用 JSON 配置文件
python run_classification.py config.json

# 多验证集评估
python run_classification.py --model_name_or_path <model> --train_file train.csv \
  --validation_files val1.csv,val2.csv,val3.csv --do_eval
```

## 依赖

核心依赖：`transformers`（开发版，需 4.57.0+）、`datasets`、`torch`、`evaluate`、`accelerate`、`scikit-learn`。具体版本见各脚本头部的 PEP 723 inline metadata。

## 架构要点

- 数据加载支持 HuggingFace Hub 数据集和本地 CSV/JSON/TXT 文件
- `run_classification.py` 自动推断任务类型（回归/单标签/多标签），默认评估指标分别为 MSE/accuracy/F1
- `run_mlm.py` 支持两种文本处理模式：逐行 tokenize（`--line_by_line`）和拼接后分块（默认）
- 使用 Transformers `Trainer` API 进行训练，支持分布式训练、混合精度、checkpoint 恢复
