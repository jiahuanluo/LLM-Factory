# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 HuggingFace Transformers 的 LLM 训练工具集，包含两个独立训练脚本：分类微调和掩码语言模型继续预训练。均改编自 Transformers 官方示例。

**新增**：`src/pbc_credit/` 模块 — 度小满 Model 1 多模态架构在央行二代征信（CrisPbc.json）上的实现。5 模态（User/Summary/Accounts×5/Queries/Publics）+ 3 交互对，无 tokenizer 依赖，可直接搬到生产处理真实 JSON 报告。详见 `data/home-credit/个人征信/CLAUDE.md` 和 `.omc/autopilot/spec.md`。

## 模块清单

| 模块 | 路径 | 用途 |
|---|---|---|
| 文本分类微调 | `run_classification.py` | 单标签/多标签/回归，sklearn 评估 |
| MLM 继续预训练 | `run_mlm.py` | BERT/RoBERTa/ALBERT 等 |
| Home Credit Model 1 | `src/home_credit/` | 多模态风控（CSV 输入） |
| PBC 征信 Model 1 | `src/pbc_credit/` | 多模态风控（JSON 报告输入，5 模态） |

## 功能概览

本仓库提供以下核心功能：

- **文本分类微调** (`run_classification.py`) — 支持单标签 / 多标签 / 回归三类任务，内置 accuracy、auc、ks、f1、mse 五种评估指标
- **MLM 继续预训练** (`run_mlm.py`) — 支持 BERT、RoBERTa、ALBERT 等掩码语言模型继续预训练，也可从零训练
- **YAML 配置系统** (`args_parser.py`) — 统一的 YAML / JSON / CLI 三模式参数解析，支持 YAML + CLI 覆盖组合
- **DeepSpeed 分布式训练** — 内置 ZeRO-2 / ZeRO-3 配置模板，支持多卡训练
- **Checkpoint 自动恢复** — 训练中断后重新运行自动从最新 checkpoint 恢复
- **UNK Token 检查** — 训练前自动检测 tokenizer 与数据语言匹配度
- **Claude Code 离线部署** (`setup_claude.py`) — 离线环境一键安装 Claude Code 及插件

## 快速开始

### 1. 文本分类微调

```bash
# 使用 YAML 配置（推荐）
python run_classification.py configs/sft_cls_auc.yaml

# YAML + CLI 覆盖参数
python run_classification.py configs/sft_cls_auc.yaml --learning_rate 1e-4 --num_train_epochs 5
```

### 2. MLM 继续预训练

```bash
python run_mlm.py configs/pt_mlm.yaml
```

### 3. 多卡训练

```bash
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls_auc.yaml
```

### 4. DeepSpeed 训练

```bash
# ZeRO-2（推荐大多数场景）
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls_auc.yaml --deepspeed configs/ds_z2.json

# ZeRO-3（超大模型）
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls_auc.yaml --deepspeed configs/ds_z3.json
```

### 5. 命令行参数（兼容 HuggingFace 原生方式）

```bash
python run_classification.py --model_name_or_path <model> --train_file train.csv --do_train --do_eval
```

参数传递：两个脚本均通过 `HfArgumentParser` 解析 `ModelArguments` + `DataTrainingArguments` + `TrainingArguments` 三类 dataclass。支持 YAML、JSON、命令行三种配置方式。

## 安装与依赖

### 核心依赖

- `transformers>=4.57.0`
- `datasets`
- `torch`
- `accelerate`
- `scikit-learn`
- `pyyaml`

详见各脚本头部 PEP 723 inline metadata 及 `requirements.txt`。`evaluate` 已替换为本地 sklearn 指标，不再依赖。

### 安装方式

```bash
# 直接运行（PEP 723 自动拉取依赖）
python run_classification.py configs/sft_cls_auc.yaml

# 或手动安装
pip install -r requirements.txt
```

### 离线环境

在 MLSS v2 平台或类似离线环境，通过 `setup_claude.py` 一键部署 Claude Code 工具链：

```bash
python setup_claude.py --dry-run  # 预览
python setup_claude.py            # 执行
```

### HuggingFace 模型下载

当前环境无法直连 `huggingface.co`，必须走 [hf-mirror.com](https://hf-mirror.com) 镜像：

```bash
# 方式 1：设环境变量后用 huggingface_hub / from_pretrained 自动走镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方式 2：直接 curl 下载单文件（可跳过权重，只取 config + tokenizer + modeling 代码）
curl -sSL -o models/<model>/config.json https://hf-mirror.com/<repo>/resolve/main/config.json
```

例如 `models/neobert/` 就是从 `chandar-lab/NeoBERT` 用方式 2 下载的（跳过了 `model.safetensors` 权重，只跑从头训练）。

## 训练脚本

- **`run_classification.py`** — 文本分类微调（单标签 / 多标签 / 回归），使用 sklearn 做评估指标
- **`run_mlm.py`** — 掩码语言模型继续预训练（BERT、ALBERT、RoBERTa 等），支持从头训练

## 示例用例

### 分类任务示例

```bash
# 二分类 AUC 评估（风控场景）
python run_classification.py configs/sft_cls_auc.yaml

# 多验证集评估（OOT 时间外样本）
python run_classification.py configs/sft_cls_multi_val.yaml

# 预测
python run_classification.py configs/predict_cls.yaml
```

### MLM 预训练示例

```bash
# 逐行模式（每行一句完整句子）
python run_mlm.py configs/pt_mlm.yaml --line_by_line true

# 拼接模式（默认，所有文本拼接后按 max_seq_length 分块）
python run_mlm.py configs/pt_mlm.yaml

# 从零训练（不加载预训练权重）
python run_mlm.py configs/pt_mlm.yaml --model_name_or_path null --model_type bert
```

### 数据格式示例

**分类 CSV：**
```csv
sentence,label
"This movie is great.",1
"Terrible film.",0
```

**多标签 JSON：**
```json
{"sentence": "Great movie.", "label": ["positive", "love"]}
```

**MLM 纯文本：**
```
This is a sentence for pretraining.
Another sentence for the model to learn from.
```

## 数据格式

### 分类任务

支持 CSV 和 JSON（JSONL）格式：

| label 类型 | 任务 | 默认指标 |
|---|---|---|
| 整数（0, 1, 2...） | 单标签分类 | accuracy |
| 浮点数（0.0, 1.0） | 回归 | mse |
| 列表（["a", "b"]） | 多标签分类 | f1 |

文本列通过 `text_column_names` 指定（多列逗号分隔）。label 发现：从 train + 所有 validation/test split 合并 label list，防止 val 出现 train 未见的标签。

### MLM 任务

支持纯文本（`.txt`，推荐）、CSV、JSON 格式。CSV/JSON 需指定 `text_column_name`（默认 `text`）。

- **逐行模式**（`line_by_line: true`）：每行独立 tokenize，适合每行是完整句子
- **拼接模式**（默认）：所有文本拼接后按 `max_seq_length` 分块（`group_texts`），适合长文本

### 数据加载

- 本地文件支持 CSV 和 JSON 格式；Hub 数据集通过 `dataset_name` 指定
- 路径支持 `~` 展开、环境变量 `$VAR`、相对路径

## 配置文件

`configs/` 目录提供常用配置模板：

| 文件 | 用途 |
|---|---|
| `sft_cls_auc.yaml` | 二分类 AUC 评估（风控场景） |
| `sft_cls_multi_val.yaml` | 多验证集评估 |
| `pt_mlm.yaml` | MLM 继续预训练 |
| `ds_z2.json` | DeepSpeed ZeRO-2 配置 |
| `ds_z3.json` | DeepSpeed ZeRO-3 配置 |

### YAML 配置特性

- `read_args()` 函数解析 YAML/JSON/CLI，支持 YAML + CLI 覆盖
- CLI 参数优先级高于 YAML
- 支持路径展开：`~` / `$ENV_VAR` / 相对路径

### 关键配置项

```yaml
# 任务控制
do_train: true          # 训练
do_eval: true           # 评估
do_predict: false       # 预测

# 评估策略
eval_strategy: epoch
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_auc
greater_is_better: true

# 混合精度
fp16: false
bf16: false  # A100/H100 推荐
```

## 架构要点

### run_classification.py

- 任务类型自动推断：label 列为 float → 回归（MSE），为 list → 多标签（F1 micro），其他 → 单标签（默认 accuracy，支持 auc/ks）
- 多验证集：`--validation_files` 接受逗号分隔路径，单个文件加载为 `"validation"` split，多个加载为 `"validation_0"`, `"validation_1"` ... Trainer 对每个 key 分别评估
- 多测试集：`--test_file` 同样支持逗号分隔，单个输出 `predict_results.csv`（CSV 格式，逗号分隔，含逗号字段自动加引号转义），多个输出 `predict_results_test_0.csv`, `predict_results_test_1.csv` ...
- label 发现：从 train + 所有 validation/test split 合并 label list，防止 val 出现 train 未见的标签
- predict-only 模式：`do_predict=true` 且 `do_train=false` 且 `do_eval=false` 且未设 `train_file` 时自动启用，跳过 train/validation 加载，label 直接从 `model.config.label2id` 读取，test_file 可不带 label 列
- 预测输出保留原始数据所有非文本列，附加 `prediction` 列（二分类为正类概率，多分类为标签名，多标签为各标签 sigmoid 概率，回归为预测值）

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

## 训练日志与监控

- 训练日志自动保存到 `output_dir/train.log`，包含完整训练过程和异常 traceback
- TensorBoard：`tensorboard --logdir output/my_model/runs`
- 评估指标：accuracy / auc / ks（随 auc 自动）/ f1 / mse

## 常见问题（FAQ）

### 显存不够

```yaml
# 方法 1：减小 batch size
per_device_train_batch_size: 8

# 方法 2：开启梯度累积（等效大 batch）
gradient_accumulation_steps: 4

# 方法 3：开启混合精度
fp16: true  # 或 bf16: true（A100/H100）

# 方法 4：使用 DeepSpeed ZeRO-2
# torchrun ... --deepspeed configs/ds_z2.json

# 方法 5：减小序列长度
max_seq_length: 64
```

### 训练太慢

```yaml
# 多卡训练：torchrun --nproc_per_node 8 ...
fp16: true
per_device_train_batch_size: 64
use_fast_tokenizer: true
```

### 模型加载报错

```yaml
# 分类头维度不匹配时忽略
ignore_mismatched_sizes: true

# 使用私有模型
token: "your-hf-token"
```

### UNK Token 占比超过 10%

训练前自动检查 tokenizer 与数据匹配度。报错说明 tokenizer 与数据语言不匹配（例如用英文 tokenizer 处理中文数据）。请更换匹配语言的 tokenizer。

### 预测时是否需要 train_file / validation_files？

**不需要**（predict-only 模式）。当满足以下条件时自动进入 predict-only 路径：

- `do_predict: true` 且 `do_train: false` 且 `do_eval: false`
- 未设置 `train_file`

此时：
- 跳过 train/validation 文件加载（避免不必要的 IO）
- label 信息直接从 `model.config.label2id` 读取（训练时已写入 ckpt 的 `config.json`）
- `test_file` 可不带 label 列

参考最小配置：`configs/predict_only.yaml`。如果使用旧版 `configs/predict_cls.yaml`（含 `train_file`），仍按兼容路径加载，但会多读两个文件。

## 故障排查

| 症状 | 排查方向 |
|---|---|
| 训练中断 | 重新运行会自动从 `output_dir` 最新 checkpoint 恢复，无需手动指定 `--resume_from_checkpoint` |
| UNK token 报错 | 检查 tokenizer 语言与数据语言是否匹配；参见 `setup_claude.py` 的采样统计逻辑 |
| 多卡训练无日志 | `datasets` 日志级别固定为 WARNING，避免分布式环境下重复输出，属正常现象 |
| 旧版 Trainer API 警告 | 本仓库使用 `processing_class=tokenizer`（新版），不再使用 `tokenizer=`（旧版） |
| YAML 参数未生效 | CLI 参数优先级高于 YAML；检查是否有同名 CLI 覆盖 |
| 路径无法解析 | 检查是否使用 `~` / `$ENV_VAR` / 相对路径；`read_args()` 支持自动展开 |
| label 未见报错 | label 列表从 train + 所有 val/test split 合并发现，防止 val 出现 train 未见标签 |
| 多测试集输出位置 | 单测试集 → `predict_results.csv`（CSV 逗号分隔）；多测试集 → `predict_results_test_0.csv`、`predict_results_test_1.csv` ... |

## 常见错误

- **`UNK token 占比超过 10%`** — tokenizer 与数据语言不匹配。解决：更换 tokenizer 或检查数据。
- **`ignore_mismatched_sizes`** — 修改了分类头维度时需手动开启：`ignore_mismatched_sizes: true`。
- **`processing_class` vs `tokenizer`** — 本仓库使用新版 Trainer API（`processing_class=tokenizer`）。如遇 transformers 版本兼容问题，检查 `transformers>=4.57.0`。
- **DeepSpeed ZeRO 选择** — 大多数场景用 ZeRO-2（`ds_z2.json`）；超大模型用 ZeRO-3（`ds_z3.json`）。
- **`metric_for_best_model` 格式** — 多验证集场景必须指定 split，如 `eval_validation_0_auc`，否则无法对齐评估 key。

## 依赖版本

- `transformers>=4.57.0`（强制要求，用于新版 `processing_class` API）
- `torch`（支持 CPU / CUDA / 昇腾 NPU）
- `accelerate`（分布式训练启动）
- `datasets`（数据加载，日志级别固定 WARNING）
- `scikit-learn`（评估指标，替代 `evaluate`）
- `pyyaml`（YAML 配置解析）
