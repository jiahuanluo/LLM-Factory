---
name: classification-finetune
description: 文本分类微调全流程操作手册（run_classification.py）：配置编写、训练启动、监控、指标解读、预测、断点恢复。只要用户提到微调、训练、finetune、跑分类/风控模型、AUC/KS/F1、出预测、predict、恢复训练、显存不够、训练太慢，即使没有点名"分类"二字也应使用本 skill。不覆盖 run_mlm.py 的掩码语言模型预训练。
---

# 分类微调全流程（run_classification.py）

按顺序执行下列环节。跳步是训练任务翻车的主要来源——配置错误往往要等到
第一个 epoch 结束才爆，浪费的是小时级时间。

## 1. 先确认三件事（缺一就问，不要猜）

1. **目标**：训练新模型 / 只出预测 / 只评估已有 checkpoint？
2. **数据**：路径 + label 形态。label 是整数 → 单标签分类；浮点 → 回归；
   列表 → 多标签。这决定 problem_type 与指标，猜错整个任务就错了。
3. **模型**：基座路径。models/ 下各目录语义差异很大（官方权重 vs 自研架构
   vs 提速变体），选择依据见 `references/config-guide.md` §模型选择。

## 2. 编写配置

- **从最接近的模板复制改**，不要从零写：
  - `configs/sft_cls_full.yaml` — 训练完整版（含 8 卡生产参数注释，推荐起点）
  - `configs/example_cls.yaml` — 训练精简示例
  - `configs/predict_only.yaml` — 纯预测最小配置（另有 `predict_cls_full.yaml` 全字段版）
  - 多验证集：任一训练模板 + `validation_files` 写逗号分隔路径即可
- CLI 参数优先级高于 YAML：临时实验参数走 CLI，验证有效后再沉淀进 YAML。
- 按模式核对必填项：

| 模式 | 必填 | 可省 |
|---|---|---|
| 训练 | `train_file` + `validation_files`（无豁免，缺了直接报错） | `test_file` |
| 纯预测 | `do_predict: true` 且 `do_train`/`do_eval: false` 且**不写** `train_file` | test 可不带 label 列，label 从 ckpt 的 config.label2id 读 |

- `metric_for_best_model` 必须与 `metric_name` 匹配：单验证集 `eval_auc`；
  多验证集必须带 split 前缀 `eval_validation_0_auc`。格式不符起跑后才报错。
- 回归任务配 `load_best_model_at_end` 时记得 `greater_is_better: false`（mse 越小越好）。

## 3. 起跑前 smoke（必做）

用小规模先跑通全链路再放真任务：

```bash
python run_classification.py my_config.yaml --max_steps 2 --per_device_train_batch_size 8
```

理由：tokenize → 训练 → 评估 → 保存任何一环的配置错误，都能在两分钟内暴露，
而不是烧完一个 epoch。smoke 通过后原命令去掉 `--max_steps` 正式跑。

注意 UNK 检查的行为：脚本自动对 raw_datasets 里**所有 split** 各采样 100 条，
UNK 占比超 10% 直接终止并标注 split 名。这几乎总是 tokenizer 与数据语言不匹配
（如英文 tokenizer 配中文数据）——正确动作是换 tokenizer，不是调阈值绕过。

## 4. 启动

```bash
# 单卡
python run_classification.py my_config.yaml
# 多卡
torchrun --nproc_per_node 8 run_classification.py my_config.yaml
```

- 训练日志自动写入 `output_dir/train.log`（含异常完整 traceback），无需手动重定向。
- 不要用管道/缓冲包装 stdout：tqdm 进度条是判断训练是否卡死的主要信号。

## 5. 监控与恢复

- 看进度：`tail -f <output_dir>/train.log` 或 TensorBoard
  `tensorboard --logdir <output_dir>/runs`。
- **中断后直接重跑同命令**：自动从 output_dir 最新 checkpoint 恢复，
  不需要手动指定 `--resume_from_checkpoint`。
- 多卡无 INFO 日志是正常的（非 rank0 只输出 ERROR）。

## 6. 指标解读（重要语义）

- **AUC/KS 自 2026-11 起在 softmax 概率上计算**（更早版本在 raw logits 上算）。
  与历史实验数字不可直接比较——对比旧实验必须同版本重跑，不要混表。
- 二分类 `metric_name: auc` 自动附带 ks；ks 仅支持二分类。
- 多标签的 auc/ks 按标签逐个输出：`auc_{label}`、`ks_{label}`。
- 多验证集指标形如 `eval_validation_0_auc`，逐 split 独立，别只看汇总。

## 7. 预测输出

- 输出 `output_dir/predict_results.csv`：保留原始非文本列 + `prediction` 列
  （二分类 = 正类概率；多分类 = 标签名；多标签 = 各标签 sigmoid 概率；回归 = 预测值）。
- 多测试集（`test_file` 逗号分隔）→ `predict_results_test_0.csv`、`predict_results_test_1.csv` ...

## 8. 调优与排障

显存不够、太慢、padding 与注意力后端（`pad_to_max_length` × 模型侧
`unpad_inputs`/`use_memory_efficient_attention`）如何搭配 → 读
`references/config-guide.md`。

数据格式细节（CSV/JSON 列要求、多标签写法）与既有排错表 → 仓库根
`CLAUDE.md` 的「数据格式 / 常见问题 / 故障排查」章节，不要凭记忆写数据文件。

## 9. 边界

- **多分类**（整数 label 且类别 > 2）优先用专用脚本 `run_multiclass_classification.py`
  （eval 同时输出 accuracy + AUC，无需 metric_name 二选一），配置参考
  `configs/mc_cls_example.yaml`；本 skill 只覆盖 run_classification.py。
- MLM 继续预训练（run_mlm.py）不在本 skill 范围。
- `run_classification_official.py` 是 HF 原版参考副本，不用于日常训练。
- 修改训练脚本本身（加功能/改逻辑）不是本 skill 的用途——那是代码任务。
