# Home Credit Model 1 设计文档

**Date**: 2026-07-22
**Author**: Claude + jiahuanluo
**Status**: Approved, ready for implementation plan
**Data**: `data/home-credit/` (Home Credit Default Risk, 已下载)
**Reference**: 度小满《AI 在度小满征信解读中的应用》Model 1

## 1. 背景与目标

### 目标

实现度小满 Model 1 架构（多模态 attention 风控模型）的改造版，在 Home Credit Default Risk 数据集上完成：
1. 多模态联合 mask 预训练
2. 监督微调
3. 在固定 val split 上报告 AUC + KS

### 验证标准

val AUC > 0.72 即视为"合理"。不追求打榜（Kaggle 公开 LB 强基线 0.78~0.80）。如未达标，分析原因并写入结论，不阻塞流程。

### 数据映射

Home Credit 数据与度小满 Model 1 原设计（央行征信报告）的 5 路输入映射如下：

| Model 1 原输入 | Home Credit 映射 | 状态 |
|---|---|---|
| Loan List | bureau.csv (1.7M) + previous_application.csv (1.67M) | **拆分为两路**（外部 vs HC 自身） |
| Credit List | credit_card_balance.csv (3.8M 月度余额) | 直接映射 |
| User Profile | application_train 的 ~50 个非序列字段 | 直接映射 |
| Query List | AMT_REQ_CREDIT_BUREAU_{HOUR/DAY/WEEK/MON/QRT/YEAR} | 退化为 6 长度伪序列 |
| Occup/Addr/Hfr 文本 | Home Credit 无此字段 | **用 7 个枚举字段拼接为伪文本** |

最终为 **6 路输入**。

### 与原 Model 1 的偏离

| 偏离点 | 原因 | 影响 |
|---|---|---|
| Loan 拆分为 Bureau + PrevApp 两路 | 数据天然分两个表，语义差异大 | Interactive Module 配对从 1 对变 3 对 |
| Text 用枚举字段拼接代替真实文本 | Home Credit 无真实文本字段 | Text 分支信号弱，但仍走完整 encoder |
| Query 用 6 个聚合数字伪造成序列 | 只有聚合数字可用 | MHA 在长度 6 的序列上信息量有限 |
| 多模态联合 mask 预训练（含连续值 MSE + text CE） | 复现度小满 pretrain-then-finetune 范式 | 显著增加实现复杂度 |

## 2. 样本格式

每个 SK_ID_CURR 对应一个样本，结构：

```python
{
  'sk_id_curr': int,

  'user_profile': {
    'numeric':     Tensor[F_num],          # 标准化数值，约 65 维
    'categorical': {
      'ids':       Tensor[F_cat],          # encode 后的 id，约 16 维
      'mask':      Tensor[F_cat],          # 0 = 该字段缺失
    },
  },

  'query_list': {
    'features': Tensor[6, 1],              # 6 个 AMT_REQ_CREDIT_BUREAU_* 值
    'mask':     Tensor[6],                 # 1 = 有效，0 = NaN
  },

  'bureau_features': Tensor[N_bureau, F_bureau],  # 见下表，numeric + categorical embeddings 已 concat
  'bureau_mask':     Tensor[N_bureau],

  'prev_features':   Tensor[N_prev, F_prev],
  'prev_mask':       Tensor[N_prev],

  'card_features':   Tensor[N_card, F_card],
  'card_mask':       Tensor[N_card],

  'text_input_ids':      Tensor[L_text],   # 枚举字段拼接后 tokenize
  'text_attention_mask': Tensor[L_text],

  'target': Tensor[1] or absent,           # 仅 train 有
}
```

### 伪文本构造

```python
def row_to_text(row):
    return " ".join([
        f"income_type_{row.NAME_INCOME_TYPE}",
        f"education_{row.NAME_EDUCATION_TYPE}",
        f"occupation_{row.OCCUPATION_TYPE or 'unknown'}",
        f"organization_{row.ORGANIZATION_TYPE}",
        f"family_{row.NAME_FAMILY_STATUS}",
        f"housing_{row.NAME_HOUSING_TYPE}",
        f"contract_{row.NAME_CONTRACT_TYPE}",
    ])
```

### 字段筛选与预处理

**User Profile - 数值字段（65 维）**：
- AMT_*（10）: 收入、信贷、年金、商品价、6 个查询计数
- DAYS_*（5）: 出生、入职、登记、换证件、换手机（**DAYS_EMPLOYED=365243 替换为 NaN**）
- EXT_SOURCE_*（3）
- CNT_*（2）: 子女、家庭人数
- REGION_*（3）: 区域评分
- 建筑信息归一化字段（精选 10 个缺失率 <50% 的）
- 其他 SOCIAL_CIRCLE / PHONE 相关数值（22）

**User Profile - 类别字段（16 维）**：
- CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY
- NAME_CONTRACT_TYPE, NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE
- OCCUPATION_TYPE, ORGANIZATION_TYPE, WEEKDAY_APPR_PROCESS_START
- HOUSETYPE_MODE, WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE, FONDKAPREMONT_MODE
- CODE_GENDER 中的 XNA / ORGANIZATION_TYPE 中的 XNA → 当作单独类别

**标准化**：数值字段用 train split 的 mean/std 做 z-score（统计量在 sample_builder 阶段计算并保存）。

### Sub-table 特征编码

三个子表都有混合 numeric + categorical 字段，统一处理：每行的 categorical 字段先经过 per-table 学习的 Embedding，再与 numeric 字段拼接，作为 encoder 的输入。

| 表 | 总列数 | ID 列 | numeric | categorical | 编码后行维度 F |
|---|---|---|---|---|---|
| bureau.csv | 17 | SK_ID_CURR, SK_ID_BUREAU | 12 | 3 (CREDIT_ACTIVE, CREDIT_CURRENCY, CREDIT_TYPE) | 12 + 3×4 = 24 |
| previous_application.csv | 37 | SK_ID_PREV, SK_ID_CURR | 19 | 16 (NAME_*, CODE_*, etc.) | 19 + 16×4 = 83 |
| credit_card_balance.csv | 23 | SK_ID_PREV, SK_ID_CURR | 20 | 1 (NAME_CONTRACT_STATUS) | 20 + 1×4 = 24 |

- Embedding 维度统一 4，per-table per-field 独立 vocab
- Vocab 在 sample_builder 阶段从 train+test 数据学习并保存到 `cat_vocab.json`
- NaN categorical → 单独的 `<UNK>` id
- Numeric NaN → 用 train 集该字段均值填充（不引入额外 sentinel）

实际 encoder 接受的 F_bureau=24, F_prev=83, F_card=24（sample 里的 `*_features` 已经是编码后的维度）。

### 变长统计与 PAD 上限

| 字段 | 平均长度 | P99 | PAD 上限 |
|---|---|---|---|
| bureau_list | 5.6 | 30 | 30 |
| prev_list | 4.9 | 26 | 30 |
| card_list | 38 个月 | 96 | 96 |
| text tokens | ~50 | ~70 | 64 |

超长截断，超短 PAD。

## 3. 模型架构

### 整体拓扑

```
                                ┌─────────────────────────┐
                                │      User Profile       │
                                └────────────┬─────────────┘
                                             │  UserEncoder
                                             ▼
                                          h_user [D]
                                              │
                                ┌─────────────────────────┐
                                │      Query List (6)     │
                                └────────────┬─────────────┘
                                             │  QueryEncoder
                                             ▼
                                          h_query [D]
                                              │
   Bureau Loan List ────► BureauEncoder ────► h_bureau [D] ─────┐
                                                                 │
   PrevApp Loan List ──► PrevAppEncoder ────► h_prev [D] ───────┤
                                                                 ├──► Interactive Module
   Credit Card List ───► CardEncoder ─────────► h_card [D] ────┘    (3 pairs × 1 = 3 features)
                                                                 │
   Text (枚举拼)  ─────► TextEncoder ─────────► h_text [D]        │
                                                                 ▼
                                          CONCAT [h_user, h_query, h_bureau, h_prev, h_card,
                                                  h_text, h_int_bp, h_int_bc, h_int_pc]  (9D)
                                                                 │
                                                                 ▼
                                                          Dense / ReLU / Drop
                                                                 │
                                                                 ▼
                                                              Dense → logit
```

### 各 Encoder 细节

| Encoder | 输入处理 | 内部结构 | 输出 |
|---|---|---|---|
| **UserEncoder** | numeric(F_num) + categorical_emb(F_cat → D/2) concat | Linear → LayerNorm → ReLU → Dropout | h_user [D] |
| **QueryEncoder** | 6 × 1 数值 → 每个投影到 D | Linear → 2-layer MHA(over 6 tokens) → AvgPool | h_query [D] |
| **BureauEncoder** | N × F_bureau(24) → Linear 投影到 D | 2-layer MHA + AvgPool（带 mask） | h_bureau [D] |
| **PrevAppEncoder** | N × F_prev(83) → Linear 投影到 D | 2-layer MHA + AvgPool | h_prev [D] |
| **CardEncoder** | N × F_card(24) → Linear 投影到 D | 2-layer MHA + AvgPool | h_card [D] |
| **TextEncoder** | L_text × vocab → Embedding | 2-layer MHA + AvgPool（带 attention_mask） | h_text [D] |

所有 MHA 共享相同的 head 配置（heads=4），但参数独立。

### Interactive Module

3 个交易序列两两配对，单向 cross-attention：

| 配对 | Query | Key/Value | 输出 |
|---|---|---|---|
| Bureau × PrevApp | bureau tokens | prev tokens | h_int_bp [D] |
| Bureau × Card | bureau tokens | card tokens | h_int_bc [D] |
| PrevApp × Card | prev tokens | card tokens | h_int_pc [D] |

每对：`MHA(q=A_tokens, kv=B_tokens) → AvgPool → [D]`。

**注**：Model 1 原文为双向（Loan×Credit×Credit 和 Credit×Loan×Loan），本项目为控制参数量只做单向 3 对。如果效果不理想可加反向。

### 顶层

```python
concat = torch.cat([h_user, h_query, h_bureau, h_prev, h_card, h_text,
                    h_int_bp, h_int_bc, h_int_pc], dim=-1)  # [B, 9D]
hidden = Dropout(ReLU(LayerNorm(Linear(concat))))            # [B, hidden]
logit  = Linear(hidden)                                       # [B, 1]
```

Pretrain 模式下不计算 logit，改为从被 mask 的位置还原原值。

### 关键超参

| 参数 | 值 | 备注 |
|---|---|---|
| D（hidden dim） | 128 | 9D 拼接后 1152 维 |
| MHA heads | 4 | head_dim=32 |
| Encoder layers | 2 | 度小满经验：深了反而差 |
| Dropout | 0.1 | |
| Mask ratio | 0.15 | BERT 默认 |
| Text vocab | 用 bert-base-uncased 的 tokenizer | 复用 HF |
| Batch size | 64 | 单卡 24G |
| Pretrain lr | 1e-4 | AdamW, weight_decay=0.01 |
| Finetune lr | 5e-5 | AdamW |
| Pretrain epochs | 2 | 355k 样本 × 2 ≈ 70k steps |
| Finetune epochs | 5 | 307k 样本，early stop on val AUC |

## 4. Collator

```python
class HomeCreditCollator:
    def __init__(self, tokenizer, pretrain_mode=False):
        self.tokenizer = tokenizer
        self.pretrain_mode = pretrain_mode

    def __call__(self, samples: list[dict]) -> dict:
        # 1. 固定维度直接 stack
        batch = {
            'user_numeric':    torch.stack([s['user_profile']['numeric']           for s in samples]),
            'user_cat_ids':    torch.stack([s['user_profile']['categorical']['ids']  for s in samples]),
            'user_cat_mask':   torch.stack([s['user_profile']['categorical']['mask'] for s in samples]),
            'query_features':  torch.stack([s['query_list']['features'] for s in samples]),
            'query_mask':      torch.stack([s['query_list']['mask']     for s in samples]),
        }

        # 2. 变长序列 pad
        for branch, feat_dim in [('bureau', 24), ('prev', 83), ('card', 24)]:
            feats = [s[f'{branch}_features'] for s in samples]
            masks = [s[f'{branch}_mask']     for s in samples]
            padded_feats, padded_masks = pad_2d(feats, masks, pad_value=0.0)
            batch[f'{branch}_features'] = padded_feats
            batch[f'{branch}_mask']     = padded_masks

        # 3. Text 用 tokenizer.pad
        text_batch = self.tokenizer.pad(
            {'input_ids': [s['text_input_ids'] for s in samples]},
            return_tensors='pt',
        )
        batch['text_input_ids']      = text_batch['input_ids']
        batch['text_attention_mask'] = text_batch['attention_mask']

        # 4. Target
        if 'target' in samples[0]:
            batch['target'] = torch.stack([s['target'] for s in samples])

        # 5. Pretrain mask 策略
        if self.pretrain_mode:
            batch.update(self._generate_masks(batch))

        return batch
```

### pad_2d 实现

```python
def pad_2d(feats: list[Tensor], masks: list[Tensor], pad_value: float = 0.0):
    # feats: list of [N_i, F], masks: list of [N_i]
    B = len(feats)
    N_max = max(f.shape[0] for f in feats)
    F = feats[0].shape[1]
    padded_feats = torch.full((B, N_max, F), pad_value, dtype=feats[0].dtype)
    padded_masks = torch.zeros((B, N_max), dtype=masks[0].dtype)
    for i, (f, m) in enumerate(zip(feats, masks)):
        n = f.shape[0]
        padded_feats[i, :n] = f
        padded_masks[i, :n] = m
    return padded_feats, padded_masks
```

### Mask 生成策略（仅 pretrain）

```python
def _generate_masks(self, batch):
    masks = {}
    for branch in ['bureau', 'prev', 'card']:
        # 整行 mask：以 0.15 概率 mask 一整行（避免局部字段相关捷径）
        token_mask = batch[f'{branch}_mask']                  # [B, N_max], 1=valid
        rand = torch.rand_like(token_mask, dtype=torch.float)
        masked_pos = (rand < 0.15) & (token_mask == 1)        # [B, N_max]
        # 把被 mask 位置的特征替换为 [MASK] embedding（学习的 sentinel）
        masks[f'{branch}_masked_pos'] = masked_pos
    # Text: 用 whole word mask (tokenizer 自带工具或手动)
    masks['text_masked_pos'] = self._text_wwm(batch['text_input_ids'], batch['text_attention_mask'])
    return masks
```

**[MASK] sentinel**：每个序列分支引入一个可学习的 `[MASK]` 行向量（shape `[1, F]`），mask 时替换原始行。

## 5. 预训练阶段

### 目标

多模态联合 mask：让 encoder 学会从其他模态 + 同模态未 mask 部分还原被 mask 的内容。

### 数据

- 355k 样本（application_train 307k + application_test 48k）
- 不使用 TARGET

### 损失

四个分量等权相加：

```python
loss =
    + mse(bureau_pred[bureau_masked_pos], bureau_orig[bureau_masked_pos])
    + mse(prev_pred[prev_masked_pos],     prev_orig[prev_masked_pos])
    + mse(card_pred[card_masked_pos],     card_orig[card_masked_pos])
    + ce(text_logits[text_masked_pos],    text_orig_ids[text_masked_pos])
```

每个分支单独的还原 head：
- Bureau / Prev / Card: Linear(D, F_branch) → 输出 F_branch 维（连续值）
- Text: Linear(D, vocab_size) → 输出 vocab 维 logits

### 度小满陷阱的规避

度小满发现纯 BERT 式 mask 会学到局部相关捷径（如"贷款额度→月供"）。本项目的规避策略：

1. **整行 mask 而非字段级 mask**：让模型必须从 *其他行* + *其他模态* 推断，而不是从同行其他字段猜
2. **不在 User Profile / Query 上做 mask**：固定维度字段 mask 意义不大且容易学到平凡关系
3. **如果效果仍不理想，备用方案**：度小满的"组合离散化 + hierarchical softmax"，作为后续优化项不在 MVP 范围

### 输出

`output/hc_pretrained/encoder_state.pt` — 保存所有 encoder + interactive module 的 state_dict。

## 6. 微调阶段

### 数据

- 307k labeled train samples
- 切分：固定 seed 42，80/20 → 246k train / 61k val
- val split 在 sample_builder 阶段一次性确定，保存到独立文件，避免后续随机性

### 损失

BCEWithLogitsLoss，pos_weight = (282686 / 24825) ≈ 11.4（应对类不平衡）。

### 评估指标

每 epoch 末在 val 上计算：
- AUC
- KS
- Accuracy @ threshold 0.5

`metric_for_best_model = eval_auc`, `greater_is_better = true`, `load_best_model_at_end = true`。

### 与 run_classification.py 的关系

不复用 run_classification.py 的 Trainer 包装（那个是单输入文本分类）。但参考其：
- YAML 配置风格
- read_args() 解析
- 多验证集评估输出格式
- tqdm 进度条可见性

## 7. 文件布局

```
src/home_credit/
  __init__.py
  config.py             # HomeCreditModelConfig dataclass
  sample_builder.py     # 从 raw CSV 构造 samples
  dataset.py            # HomeCreditDataset
  collator.py           # HomeCreditCollator + pad_2d
  model.py              # HomeCreditModel: 6 Encoders + Interactive + 顶层
  masking.py            # 行级 mask + text WWM
  losses.py             # 多模态 mask loss

scripts/
  prepare_home_credit_samples.py

run_home_credit_pretrain.py
run_home_credit_finetune.py

configs/
  hc_pretrain.yaml
  hc_finetune.yaml

data/home-credit/processed/   # sample_builder 输出
  samples_train.pkl           # 307k labeled
  samples_test_unlabeled.pkl  # 48k (仅 pretrain 用)
  samples_val.pkl             # 61k (固定 val split)
  normalizer.json             # 数值字段的 mean/std
  cat_vocab.json              # 类别字段的 vocab
  split.json                  # train/val SK_ID_CURR 列表

output/
  hc_pretrained/
    encoder_state.pt
  hc_finetuned/
    best_model/
    eval_results.json
    val_predictions.csv
```

## 8. 实现顺序（ultragoal 目标分解）

按依赖关系排序，每个目标对应 ultragoal 的一个 story：

1. **G001 数据预处理脚本** — sample_builder + prepare_home_credit_samples.py，输出 samples + normalizer + cat_vocab + split
2. **G002 Dataset + Collator** — HomeCreditDataset + HomeCreditCollator + pad_2d
3. **G003 Model 架构** — 6 Encoders + Interactive Module + 顶层 forward
4. **G004 Masking 模块** — 行级 mask + text WWM
5. **G005 预训练脚本** — run_home_credit_pretrain.py + losses.py
6. **G006 微调脚本** — run_home_credit_finetune.py + eval 指标
7. **G007 端到端跑通** — 实际训练（预训练 2 epoch + 微调 5 epoch）+ val AUC 报告

## 9. 已知风险与待观察项

| 风险 | 影响 | 缓解 |
|---|---|---|
| 预训练学不到有用表示（局部相关陷阱） | finetune 起点差，val AUC 低于 0.72 | 先跑通流程；如不达标，备用方案是度小满的组合离散化 |
| Text 分支信号弱（枚举字段拼接） | 整体 AUC 损失约 0.5~1 个百分点 | 接受现状，记录在结论里 |
| 显存不够（N_card P99=96 很长） | OOM | 减小 batch size 或减小 N_card pad 上限 |
| bureau + previous_application 字段类型差异大（编码后 24 vs 83 维） | encoder 输入维度不一致 | 各自独立 Linear 投影到 D |
| 6 路 input 的 forward 复杂度高 | 训练慢 | 接受；可后续优化 dataloader workers |

## 10. 验收清单

- [ ] `samples_train.pkl` / `samples_val.pkl` 生成，大小符合预期（约 3GB / 600MB）
- [ ] Pretrain loss 在 2 epoch 内单调下降
- [ ] Finetune val AUC > 0.72（弱基线）且记录到 `eval_results.json`
- [ ] 输出 `val_predictions.csv`（含 SK_ID_CURR + target + prediction 三列）
- [ ] `data/home-credit/CLAUDE.md` 更新一节"模型实验结果"，记录 val AUC + KS + 是否达标
