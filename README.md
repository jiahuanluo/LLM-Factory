# LLM-Factory

基于 HuggingFace Transformers 的文本分类微调工具集，支持二分类 / 多分类 / 多标签 / 回归任务，以及 MLM 继续预训练。

## 实际使用流程

```
模型广场下载 checkpoint → 数据库拉取 token 文本 + label → 导入 MLSS 微调 → 输出预测结果
```

1. **下载预训练模型**：从模型广场获取对应的预训练 checkpoint，放到 `cache_dir`（默认 `~/workspace/cache`）或直接指定本地路径
2. **准备数据**：从数据库对应的 token 串表获取文本和 label，导出为 CSV/JSON 格式
3. **上传到 MLSS 平台进行微调**

## 功能概览

- **文本分类微调** (`run_classification.py`) — 二分类 / 多分类 / 多标签 / 回归，支持 AUC、KS、Accuracy 等指标
- **MLM 继续预训练** (`run_mlm.py`) — BERT、RoBERTa、ALBERT 等掩码语言模型继续预训练
- **Claude Code 离线部署** (`setup_claude.py`) — 离线环境一键安装 Claude Code + 插件

---

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：

| 包 | 用途 |
|---|---|
| `transformers>=4.57.0` | 模型加载、训练框架 |
| `datasets>=2.14.0` | 数据加载 |
| `torch>=1.3` | 深度学习框架 |
| `accelerate>=0.12.0` | 分布式训练 |
| `scikit-learn` | 评估指标（AUC、KS、F1 等） |
| `scipy` | softmax、sigmoid 数值稳定计算 |
| `pyyaml>=6.0` | YAML 配置解析 |

---

## 快速开始

### 1. 文本分类微调

```bash
# 准备数据：从数据库导出 CSV，包含文本列和 label 列
# data/train.csv 示例：
# sentence,label
# "This movie is great.",1
# "Terrible film.",0

# 训练 + 评估
python run_classification.py configs/sft_cls.yaml

# CLI 覆盖参数
python run_classification.py configs/sft_cls.yaml --learning_rate 1e-4 --num_train_epochs 5
```

### 2. 预测

```bash
# 修改 model_name_or_path 指向微调后的 checkpoint
python run_classification.py configs/predict_cls.yaml
```

### 3. MLM 继续预训练

```bash
# 准备数据：每行一句纯文本
python run_mlm.py configs/pt_mlm.yaml
```

### 4. 多卡训练

```bash
# 2 卡
torchrun --nproc_per_node 2 run_classification.py configs/sft_cls.yaml

# 8 卡
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls.yaml
```

### 5. DeepSpeed 训练

```bash
# ZeRO-2（推荐大多数场景）
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls.yaml --deepspeed configs/ds_z2.json

# ZeRO-3（超大模型）
torchrun --nproc_per_node 8 run_classification.py configs/sft_cls.yaml --deepspeed configs/ds_z3.json
```

---

## 文本分类详解 (`run_classification.py`)

### 数据格式

支持 CSV 和 JSON（JSONL）格式：

**CSV 格式：**
```csv
sentence,label
"This movie is great.",1
"Terrible film.",0
```

**JSON 格式（多标签分类）：**
```json
{"sentence": "Great movie.", "label": ["positive", "love"]}
{"sentence": "Bad film.", "label": ["negative", "boring"]}
```

### 任务类型自动推断

脚本根据 `label` 列的类型自动判断任务：

| label 类型 | 任务 | 默认指标 |
|---|---|---|
| 整数（0, 1, 2...） | 单标签分类 | accuracy |
| 浮点数（0.0, 1.0） | 回归 | mse |
| 列表（["a", "b"]） | 多标签分类 | f1 |

通过 `metric_name` 可手动指定指标：`accuracy`、`auc`、`ks`、`f1`、`mse`。

> 使用 `auc` 时会自动同时计算 KS 指标。

### 多验证集

支持同时在多个验证集上评估，适合时间外（OOT）验证：

```yaml
validation_files: data/val_oot1.csv,data/val_oot2.csv,data/val_oot3.csv
```

评估结果会分别输出为 `eval_validation_0_auc`、`eval_validation_1_auc` 等。

`metric_for_best_model` 格式：`eval_validation_0_auc`（指定用哪个验证集的指标选最优模型）。

### 预测输出

```yaml
do_predict: true
test_file: data/test.csv  # 多个用逗号分隔: test1.csv,test2.csv
```

预测结果保留原始数据中所有非文本列，附加 `prediction` 列：

| 任务 | prediction 格式 |
|---|---|
| 二分类 | 正类概率（0~1） |
| 多分类 | 预测标签名 |
| 多标签 | 各标签 sigmoid 概率 |
| 回归 | 预测值 |

自动排除：文本列（`text_column_names`）、`label` 列、tokenizer 列。可通过 `remove_columns` 额外排除指定列。

单测试集输出 `predict_results.txt`，多测试集输出 `predict_results_test_0.txt`、`predict_results_test_1.txt` 等。

### 完整配置参考

```yaml
# === 必填 ===
model_name_or_path: bert-base-uncased    # 模型路径或 HuggingFace ID
train_file: data/train.csv               # 训练集
validation_files: data/val.csv           # 验证集（多个逗号分隔）
text_column_names: sentence              # 文本列名（多列逗号分隔）
output_dir: output/my_model              # 输出目录

# === 模型缓存 ===
cache_dir: ~/workspace/cache             # 模型缓存目录，留空或 null 用 HuggingFace 默认路径
trust_remote_code: true                  # 信任远程代码

# === 常用 ===
learning_rate: 2e-5                      # 学习率
num_train_epochs: 3                      # 训练轮次
per_device_train_batch_size: 32          # 每 GPU batch size
max_seq_length: 128                      # 最大序列长度
metric_name: auc                         # 评估指标

# === 任务控制 ===
do_train: true                           # 训练
do_eval: true                            # 评估
do_predict: false                        # 预测
do_regression: false                     # 回归任务

# === 评估和保存 ===
eval_strategy: epoch                     # 评估策略
save_strategy: epoch                     # 保存策略
load_best_model_at_end: true             # 训练结束加载最优模型
metric_for_best_model: eval_auc          # 最优模型指标
greater_is_better: true                  # 指标方向

# === 混合精度 ===
fp16: false                              # FP16
bf16: false                              # BF16（A100/H100 推荐）

# === 调试 ===
max_train_samples: 100                   # 限制训练样本数
max_eval_samples: 50                     # 限制验证样本数
logging_steps: 10                        # 日志间隔
report_to: tensorboard                   # 日志工具

# === 分布式 ===
deepspeed: null                          # DeepSpeed 配置路径
gradient_accumulation_steps: 1           # 梯度累积
```

---

## MLM 继续预训练详解 (`run_mlm.py`)

### 数据格式

支持纯文本（`.txt`）、CSV、JSON 格式：

**纯文本（推荐）：**
```
This is a sentence for pretraining.
Another sentence for the model to learn from.
```

**CSV/JSON：** 需指定 `text_column_name`（默认 `text`）。

### 文本处理模式

| 模式 | 配置 | 说明 |
|---|---|---|
| 逐行模式 | `line_by_line: true` | 每行独立 tokenize，适合每行是完整句子 |
| 拼接模式 | `line_by_line: false`（默认） | 所有文本拼接后按 `max_seq_length` 分块，适合长文本 |

### 验证集

```yaml
# 方式 1：指定验证集文件
validation_file: data/val.txt

# 方式 2：从训练集自动切分
validation_split_percentage: 10  # 从训练集切 10% 作为验证集

# 方式 3：不使用验证集（默认）
validation_split_percentage: 0
do_eval: false
```

> `validation_split_percentage` 默认为 0，即不自动切分。设为大于 0 时使用 `train_test_split` 在内存中切分，不会重复读盘。

### 完整配置参考

```yaml
# === 必填 ===
model_name_or_path: bert-base-uncased    # 模型路径
train_file: data/train.txt               # 训练数据
output_dir: output/mlm                   # 输出目录

# === 模型缓存 ===
cache_dir: ~/workspace/cache             # 模型缓存目录，留空或 null 用 HuggingFace 默认路径
trust_remote_code: true                  # 信任远程代码

# === 常用 ===
learning_rate: 5e-5                      # 学习率（MLM 通常比微调大）
num_train_epochs: 10                     # 训练轮次
per_device_train_batch_size: 16          # 每 GPU batch size
max_seq_length: 512                      # 最大序列长度

# === MLM 配置 ===
mlm_probability: 0.15                    # 掩码概率（BERT 原始设置）
line_by_line: true                       # 逐行模式

# === 验证集 ===
validation_file: null                    # 验证集文件
validation_split_percentage: 0           # 自动切分比例（0=不切分）

# === 调试 ===
max_train_samples: null                  # 限制样本数
logging_steps: 50                        # 日志间隔
```

---

## 配置系统

### YAML 配置（推荐）

所有参数都可以通过 YAML 配置文件设置，支持三种方式组合使用：

```bash
# 1. 纯 YAML
python run_classification.py configs/sft_cls.yaml

# 2. YAML + CLI 覆盖（CLI 优先）
python run_classification.py configs/sft_cls.yaml --learning_rate 1e-4 --num_train_epochs 5

# 3. 纯 CLI（兼容 HuggingFace 原生方式）
python run_classification.py --model_name_or_path bert-base-uncased --train_file data/train.csv --do_train
```

### 路径展开

配置文件中的路径支持：
- `~` 展开：`~/data/train.csv` → `/home/user/data/train.csv`
- 环境变量：`$DATA_DIR/train.csv` → 实际路径
- 相对路径：`./data/train.csv`

### 预置配置模板

| 文件 | 用途 |
|---|---|
| `configs/sft_cls.yaml` | 分类微调（accuracy） |
| `configs/sft_cls_auc.yaml` | 二分类 AUC 评估（风控场景） |
| `configs/sft_cls_multi_val.yaml` | 多验证集评估 |
| `configs/predict_cls.yaml` | 分类预测 |
| `configs/example_cls.yaml` | 分类完整配置示例（带详细注释） |
| `configs/pt_mlm.yaml` | MLM 继续预训练 |
| `configs/example_mlm.yaml` | MLM 完整配置示例（带详细注释） |
| `configs/ds_z2.json` | DeepSpeed ZeRO-2 配置 |
| `configs/ds_z3.json` | DeepSpeed ZeRO-3 配置 |

---

## 训练特性

### Checkpoint 自动恢复

训练中断后重新运行，会自动从 `output_dir` 最新 checkpoint 恢复：

```bash
# 自动恢复
python run_classification.py configs/sft_cls.yaml

# 指定 checkpoint
python run_classification.py configs/sft_cls.yaml --resume_from_checkpoint output/my_model/checkpoint-500
```

### UNK Token 检查

训练前自动检查 tokenizer 与数据的匹配度：随机采样 100 条数据，统计 UNK token 占比。超过 10% 会报错停止，提示检查 tokenizer 是否与数据语言匹配。

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir output/my_model/runs

# 查看训练曲线：loss、learning_rate、eval 指标等
```

### 评估指标

| 指标 | 说明 | 适用任务 |
|---|---|---|
| `accuracy` | 准确率 | 单标签分类 |
| `auc` | AUC-ROC | 二分类 / 多分类 |
| `ks` | KS 统计量 | 二分类（随 auc 自动计算） |
| `f1` | F1-micro | 多标签分类 |
| `mse` | 均方误差 | 回归 |

---

## Claude Code 离线部署 (`setup_claude.py`)

用于离线环境一键安装 Claude Code 及插件。

### 使用方法

```bash
# 1. 编辑配置文件
vim configs/claude_config.yaml

# 2. 预览（不实际执行）
python setup_claude.py --dry-run

# 3. 安装
python setup_claude.py

# 4. 指定其他配置
python setup_claude.py other_config.yaml

# 5. 只装 Claude Code，不装插件
python setup_claude.py --skip-plugins
```

### 配置文件

```yaml
# configs/claude_config.yaml
api_base_url: "https://your-api-proxy/anthropic"  # API 代理地址
api_key: "your-api-key"                            # API Key
model: "claude-sonnet-4-6"                         # 默认模型
api_timeout_ms: 3000000                            # 超时时间（毫秒）
plugins_enabled: true                              # 是否安装插件
plugins_dir: "deploy/plugins"                      # 插件源目录
```

### 离线插件

`deploy/plugins/` 目录包含预打包的插件快照：

- **superpowers** — 增强技能系统
- **code-review** — 代码审查
- **code-simplifier** — 代码简化
- **ralph-loop** — 持续执行循环
- **frontend-design** — 前端设计

安装后插件配置自动写入 `~/.claude/settings.json`，Claude Code 启动时自动加载。

---

## 项目结构

```
LLM-Factory/
├── run_classification.py          # 文本分类微调脚本
├── run_mlm.py                     # MLM 继续预训练脚本
├── args_parser.py                 # 共享参数解析（YAML/JSON/CLI）
├── setup_claude.py                # Claude Code 离线部署脚本
├── requirements.txt               # Python 依赖
├── configs/                       # 配置文件模板
│   ├── sft_cls.yaml              # 分类微调
│   ├── predict_cls.yaml          # 分类预测
│   ├── sft_cls_auc.yaml          # 二分类 AUC
│   ├── sft_cls_multi_val.yaml    # 多验证集
│   ├── example_cls.yaml          # 分类完整示例
│   ├── pt_mlm.yaml               # MLM 预训练
│   ├── example_mlm.yaml          # MLM 完整示例
│   ├── ds_z2.json                # DeepSpeed ZeRO-2
│   ├── ds_z3.json                # DeepSpeed ZeRO-3
│   └── claude_config.yaml        # Claude Code 配置
├── data/                          # 示例数据
│   ├── train.csv                  # 分类训练集
│   └── val.csv                    # 分类验证集
└── deploy/                        # 离线部署资源
    └── plugins/                   # 插件快照
```

---

## 常见问题

### 显存不够

```yaml
# 方法 1：减小 batch size
per_device_train_batch_size: 8

# 方法 2：开启梯度累积（等效大 batch）
gradient_accumulation_steps: 4  # 等效 batch_size = 8 * 4 = 32

# 方法 3：开启混合精度
fp16: true  # 或 bf16: true（A100/H100）

# 方法 4：使用 DeepSpeed ZeRO-2
# torchrun ... --deepspeed configs/ds_z2.json

# 方法 5：减小序列长度
max_seq_length: 64
```

### 训练太慢

```yaml
# 方法 1：多卡训练
# torchrun --nproc_per_node 8 ...

# 方法 2：开启混合精度
fp16: true

# 方法 3：增大 batch size（在显存允许范围内）
per_device_train_batch_size: 64

# 方法 4：使用 fast tokenizer
use_fast_tokenizer: true
```

### 模型加载报错

```yaml
# 如果修改了分类头维度，忽略维度不匹配
ignore_mismatched_sizes: true

# 使用私有模型
token: "your-hf-token"
```

### 验证集指标异常

训练前会自动检查 UNK token 占比。如果报错 `UNK token 占比超过 10%`，说明 tokenizer 与数据语言不匹配，例如用英文 tokenizer 处理中文数据。
