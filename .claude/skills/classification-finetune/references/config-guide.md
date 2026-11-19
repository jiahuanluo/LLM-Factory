# 配置决策参考（run_classification.py）

## 模型选择（models/ 目录）

| 目录 | 语义 | 适用 |
|---|---|---|
| `gte-large-en-v1.5` | 官方权重 + 官方配置（unpad/meff 关） | 需要"官方权重直接微调"的基线 |
| `gte-large-eval-fixed` | 官方权重 + unpad 提速（unpad/meff 开） | 同上但更快；须 CUDA + fp16/bf16 |
| `gte-new` | 深度改造架构（GQA/silu/RMSNorm/yarn），从头训 | 复现自研架构实验 |
| `gte-new-fa2` | 双后端 attention（flash-attn 优先，SDPA 回落） | 装了 flash-attn 的机器 |
| `gte-new-orig-npu` | 昇腾 NPU fusion_attention 适配 | NPU 集群 |

选择原则：做"官方权重 vs 从头训"对比实验时，基线用 `gte-large-en-v1.5`
（或提速版 `gte-large-eval-fixed`），自研侧用 `gte-new`。混用其他变体会引入
无关变量。

## padding × 注意力后端矩阵

脚本侧 `pad_to_max_length` 决定进模型前 batch 里的 padding 量；模型侧
`unpad_inputs` 决定模型内是否剥掉 padding；`use_memory_efficient_attention`
（下称 meff）决定注意力后端是否换 xformers varlen。

| pad_to_max_length | unpad | meff | 效果 / 约束 |
|---|---|---|---|
| true | false | false | 全 padding 浪费 + 标准注意力，最慢（官方 v1.5 默认即此） |
| true | true | true | 静态 pad 但模型内剥离，计算量 ≈ 动态；形状稳定（eval-fixed 默认） |
| false | true | true | 同上且 batch 占用更小，可再上调 batch size |
| false | false | false | 仅脚本侧动态 padding（官方 v1.5 的最优解） |

配套规则：

- unpad 已经把 padding 浪费剥掉了，`group_by_length: true` 与它收益重叠，
  同时开没有增益——分桶只在"模型侧无 unpad 且动态 padding"时有用
  （即官方 v1.5 + `pad_to_max_length: false` 的组合）。
- meff 要求 CUDA + fp16/bf16（xformers 半精度限制），CPU 不可用；
  且依赖 xformers 已安装——modeling.py 里有硬断言，未装直接报错退出。
- meff 路径不支持 `output_attentions`，`attn_implementation` 会被强制覆盖回 eager。
- 换模型目录后记得换 `output_dir`：自动恢复机制会捡到旧模型的 checkpoint。
- 修改模型目录下 config.json 里这两个参数即可切换，权重文件不用动。

## 任务类型与指标

| label 形态 | 任务 | 默认指标 | 可选 |
|---|---|---|---|
| 整数（0,1,2...） | 单标签 | accuracy | auc（二分类自动附带 ks）、ks（仅二分类） |
| 浮点（0.0, 1.0） | 回归 | mse | — |
| 列表（["a","b"]） | 多标签 | f1 | auc / ks（逐标签输出 `auc_{label}`） |

## metric_for_best_model 合法格式

```
eval_{metric}                      # 单验证集
eval_validation_{N}_{metric}       # 多验证集（必须带 split 前缀）
eval_{metric}_{label}              # per-label 指标
```

## 显存不够（按优先级尝试）

1. `per_device_train_batch_size` 减半
2. `gradient_accumulation_steps` 相应上调（等效大 batch）
3. `fp16: true`（V100）/ `bf16: true`（A100/H100）
4. DeepSpeed ZeRO-2：`--deepspeed configs/ds_z2.json`
5. `gradient_checkpointing: true` + `use_reentrant: false`（激活显存大降，速度慢 20-30%，OOM 前的最后手段）
6. `max_seq_length` 下调（会截断长样本、改变任务语义，放靠后）
7. 超大模型再上 ZeRO-3（`ds_z3.json`）

## 太慢（按优先级尝试）

1. `fp16` / `bf16`
2. 增大 batch size（显存允许时）
3. `pad_to_max_length: false`（模型侧无 unpad 时，配 `group_by_length: true`）
4. `per_device_eval_batch_size` 上调（eval/predict 无梯度无反传，可以远大于 train batch）
5. 多卡 `torchrun`
6. 换 `gte-large-eval-fixed`（unpad + meff，前提 fp16 + CUDA）
7. `dataloader_num_workers: 4`（collate 阶段 CPU 并行）、`use_fast_tokenizer: true`

## 混合精度

| 硬件 | 选择 |
|---|---|
| A100 / H100 | bf16 |
| V100 | fp16 |
| 昇腾 NPU | 参考 `gte-new-orig-npu` 目录内实践 |

## 多文件输入输出

| 输入 | 写法 | 输出 |
|---|---|---|
| 多验证集 | `validation_files: a.csv,b.csv` | 指标 split：`validation_0`、`validation_1` ... |
| 多测试集 | `test_file: a.csv,b.csv` | `predict_results_test_0.csv`、`predict_results_test_1.csv` ... |
| 单测试集 | `test_file: a.csv` | `predict_results.csv` |
