# 多验证集支持设计

## 背景

当前 `run_classification.py` 只支持单个验证集文件。需要支持多个验证集，分别评估并输出各自的结果。

## 方案

扩展 `validation_file` 参数为 `validation_files`，接受逗号分隔的多个文件路径，兼容单文件。

## 改动范围

仅修改 `run_classification.py`，`run_mlm.py` 不涉及。

## 详细设计

### 1. 参数定义

`DataTrainingArguments` 中：

- `validation_file: str | None` → `validation_files: str | None`
- 语义：逗号分隔的多个文件路径
- 示例：`--validation_files val1.csv,val2.csv,val3.csv` 或 `--validation_files val1.csv`

`__post_init__` 校验：

- `split(",")` 拆分后校验每个文件扩展名一致（都是 csv 或都是 json）
- 当 `dataset_name` 为 None 且 `do_train` 时，仍要求提供 `train_file`

### 2. 数据加载

- `validation_files` 用 `split(",")` 拆分为列表
- 每个验证集文件独立加载，取 `["validation"]` split 后存入 dict：
  ```python
  eval_dict = {}
  for i, val_file in enumerate(validation_files_list):
      ds = load_dataset(extension, data_files={"validation": val_file}, ...)
      eval_dict[f"validation_{i}"] = ds["validation"]
  ```
- train 文件加载逻辑不变

### 3. 评估

- `eval_dict` 直接传给 `Trainer(eval_dataset=eval_dict)`
- Trainer 自动对每个 key 分别评估，指标前缀为 key 名（如 `eval_validation_0_accuracy`）
- 结果文件由 Trainer 自动管理

### 4. 不变的部分

- train 加载逻辑
- predict 逻辑
- `run_mlm.py`

## 用法示例

```bash
# 单个验证集（兼容原有用法）
python run_classification.py \
  --model_name_or_path bert-base-uncased \
  --train_file train.csv \
  --validation_files val.csv \
  --do_train --do_eval

# 多个验证集
python run_classification.py \
  --model_name_or_path bert-base-uncased \
  --train_file train.csv \
  --validation_files val1.csv,val2.csv,val3.csv \
  --do_train --do_eval
```
