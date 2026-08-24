# PBC 生产数据接入 Runbook（路径 B：原始 CrisPbc JSON 报文 → 模型）

适用：生产可导出原始 CrisPbc JSON 报文文件。转换器与真实报文同构（校准 mock 按
同一 schema 生成），P1 特征（cd01/已用额度/当前逾期/使用率）、19 个 report 级聚合、
log1p 归一化全部就绪；输出仅含 reportsn + 特征，不带 PII。

## 0. 前置

- 环境：conda alphalab（`/home/daxigua/package/miniconda3/envs/alphalab/bin/python`）
- 代码：feat 分支已合并 `worktree-mock-calibration`（含 `scripts/convert_mock_to_pbc_struct.py`）
- 生产侧：导出原始报文 `*.json` 到一个目录（无脚本要跑）

## 0.5 生产内转换（原始报文量大时，推荐）：Spark UDF 版

`scripts/spark_convert_pbc_struct.py`（自包含，executor 不依赖 src 包；纯函数与离线转换器
逐位一致——已交叉验证 numeric/paystate/cat_mask 完全相同）：

1. **首次先建表**（脚本头部有 DDL）：
   ```sql
   CREATE TABLE IF NOT EXISTS erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds
     (busi_sno string, reportsn string, pbc_struct string)
   PARTITIONED BY (ds string) STORED AS ORC;
   ```
2. 集群/notebook 粘贴整份脚本，改 `RUN_DATE`，执行 `run_spark()`：
   - pass0 打印输入报文数（`ods_marm_raw_credit_data_bdcn_v2`，ds 分区，type 三类过滤）
   - pass1 集群内 distinct 码值 → 建 vocab（保证 UNK=0）→ 同时落盘 `cat_vocab_prod_<ds>.json`
   - pass2 broadcast vocab → UDF 转换 → 写结果表；**失败行保留 `{"_error":...}` 不拖死 job**
3. **预期输出**：结果表行数 = 输入报文数，ok 率应 >99.9%；`pbc_struct` 列即 PbcDataset
   扁平格式（user 32 + 6 类账户 13/13/60），无需 postprocess
4. **离线取数 → 训练**：
   ```sql
   SELECT busi_sno, reportsn, pbc_struct FROM erm_mx_data_work.marm_pbcg2_pbcstruct_v1_ds
   WHERE ds='<RUN_DATE>' AND pbc_struct NOT LIKE '{"_error%'
   ```
   导出为 `train_prod.jsonl` 后按 md5(reportsn)%10==0 切出 val，直接给 `PbcDataset`
5. **混训注意**：Spark 版 vocab 是本次语料的 id 空间，与 `cat_vocab_mock.json` 不同；
   混训前取两者并集、两侧重编码（pass1 落盘的 json 就是为这一步准备的）

## 1. 转换（一条命令，小体量可直接导出文件时）

```bash
python scripts/convert_mock_to_pbc_struct.py \
    --src <真实报文目录> \
    --out-dir data/pbc/processed_prod \
    --vocab-name cat_vocab_prod.json \
    --train-name train_prod.jsonl --val-name val_prod.jsonl \
    --pbc-src src
```

- vocab 首跑自动从真实语料构建（0=UNK，1..N 字典序）；重跑时校验无 vocab 外新值，有则报错拒绝（安全）
- 离群报文（缺 `accountInfos`/`personInfo` 的 stub/错误信封）自动跳过并记日志
- 确定性：同输入同输出；1/10 holdout 切分（`--val-holdout` 可调）

## 2. 训练前快速验收

转换器尾部打印：行数、各类型账户分布、cat UNK 率。检查：

- [ ] 行数 ≈ 导出报文数 − 跳过数
- [ ] **UNK = 0**；不为 0 说明真实语料有码值表外新值（先人工确认再重建 vocab）
- [ ] 抽一条看维度：`user_numeric` 32、每账户 numeric 13 / cat 13 / paystate 60
- [ ] `max 账户数`：>500 的大报告训练时显存吃紧，先小 batch 试跑（top-K 截断是待办）
- [ ] 金额口径确认为"元"（转换器按元做 log1p；若生产导出为万元需先换算）

## 3. 预训练

```bash
# 改 configs/pbc_pretrain.yaml 三处路径 → processed_prod 三件，然后：
python run_pbc_pretrain.py configs/pbc_pretrain.yaml
torchrun --nproc_per_node=N run_pbc_pretrain.py configs/pbc_pretrain.yaml   # 多卡
```

产出 `output/pbc_pretrained/encoder_state.pt`（已 strip mask heads，可直接接微调）。

## 4. 微调（需要标签）

```bash
python scripts/join_label.py ...        # 生产标签 → train_prod_labeled.jsonl（外层加 label 字段）
python run_pbc_finetune.py configs/pbc_finetune.yaml   # init_from_pretrain 指向上面 ckpt
```

## 已知边界

| 事项 | 说明 |
|---|---|
| mock+生产混训 | 两份 vocab（cat_vocab_mock / cat_vocab_prod）取**并集**后分别重编码，否则 id 空间不一致 |
| query 3 个聚合特征 | 依赖报文含 `queryRecords`；缺失自动置 0 |
| 超大报告 | 生产 max 1653 账户/份，SeqEncoder O(N²)；建议后续按余额/新近度 top-K 截断 + 截断计数特征（P2 待办） |
| score 块 | 已不用于特征（生产覆盖率 4.7%）；报文里有也无害 |
| Spark 拆分表路径 | 若将来走 SQL 管道（路径 A），需先把 `postprocess_pbc_struct.py` 升级到与本转换器同构（归一化 + 聚合） |
