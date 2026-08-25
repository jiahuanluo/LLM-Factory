-- ============================================================
-- 02 数据质量分析：表大小 + 填充率 + 码值分布
-- 目的：决定哪些字段值得加入模型（生产真有数据 vs 字典定义但全空）
-- ============================================================

-- ============================================================
-- Step 1: 每张相关表的行数 + 唯一 reportsn 数
--   关注点：
--   - 同表 reportsn 数应一致；如果某张表 reportsn 远少于 person_identity，
--     说明该表覆盖稀疏（如 score 只有 4.7% 报告有）
-- ============================================================

SELECT '=== person_identity ===' AS tbl,
       COUNT(*) AS n_rows,
       COUNT(DISTINCT reportsn) AS n_reports
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_bibf
UNION ALL
SELECT 'person_marriage', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_person_marriage_bibf
UNION ALL
SELECT 'person_mobile', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_person_mobile_bibf
UNION ALL
SELECT 'person_professional', COUNT(*), COUNT(DISTINCT reportsn)
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_bibf
UNION ALL
SELECT 'person_residence', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_person_residence_bibf
UNION ALL
SELECT 'header_identity_other', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_header_identity_other_bibf
UNION ALL
SELECT 'score', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_score_bibf
UNION ALL
SELECT 'account_basic', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf
UNION ALL
SELECT 'account_latest_info', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf
UNION ALL
SELECT 'account_latest_month_pay_state', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_bibf
UNION ALL
SELECT 'account_latest_5_year_detail', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_5_year_detail_bibf
UNION ALL
SELECT 'account_special_trade', COUNT(*), COUNT(DISTINCT reportsn)
FROM brm_wyd_ods_mask.cris_pbcg2_account_special_trade_bibf;


-- ============================================================
-- Step 2: account_basic 各账户类型分布（pd01ad01）
--   看生产 D1/R1/R2/R3/R4/C1 实际占比
-- ============================================================
SELECT
  pd01ad01 AS account_type,
  COUNT(*) AS n_accounts,
  COUNT(DISTINCT reportsn) AS n_reports
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf
GROUP BY pd01ad01
ORDER BY n_accounts DESC;
-- 期望看到 D1/R1/R2/R3/R4/C1；如果出现别的（D2/R5/...），spark 脚本要扩展


-- ============================================================
-- Step 3: ⭐ 金矿字段填充率（重点验证）
--   这些字段在字典里有，但 mock 数据稀疏。生产真实填充率决定模型能否用
-- ============================================================

-- 3.1 五级分类 PD01BD03 + 账户状态 PD01BD01（在 account_latest_info）
SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN pd01bd01 IS NOT NULL AND pd01bd01 <> '' THEN 1 ELSE 0 END) AS filled_bd01,
  SUM(CASE WHEN pd01bd03 IS NOT NULL AND pd01bd03 <> '' THEN 1 ELSE 0 END) AS filled_bd03,
  SUM(CASE WHEN pd01bd04 IS NOT NULL AND pd01bd04 <> '' THEN 1 ELSE 0 END) AS filled_bd04,
  SUM(CASE WHEN pd01bj01 IS NOT NULL AND pd01bj01 <> '' THEN 1 ELSE 0 END) AS filled_bj01,
  SUM(CASE WHEN pd01bj02 IS NOT NULL AND pd01bj02 <> '' THEN 1 ELSE 0 END) AS filled_bj02,
  ROUND(100.0 * SUM(CASE WHEN pd01bd03 IS NOT NULL AND pd01bd03 <> '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_bd03
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf;
-- 期望：bd03 五级分类 填充率 > 80%（监管要求强制字段）

-- 3.2 月度表现 CJ* 系列金矿字段（在 account_latest_month_pay_state）
SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN pd01cj04 IS NOT NULL AND pd01cj04 <> '' THEN 1 ELSE 0 END) AS filled_cj04,
  SUM(CASE WHEN pd01cj05 IS NOT NULL AND pd01cj05 <> '' THEN 1 ELSE 0 END) AS filled_cj05,
  SUM(CASE WHEN pd01cj06 IS NOT NULL AND pd01cj06 <> '' THEN 1 ELSE 0 END) AS filled_cj06,
  SUM(CASE WHEN pd01cj07 IS NOT NULL AND pd01cj07 <> '' THEN 1 ELSE 0 END) AS filled_cj07,
  SUM(CASE WHEN pd01cj08 IS NOT NULL AND pd01cj08 <> '' THEN 1 ELSE 0 END) AS filled_cj08,
  SUM(CASE WHEN pd01cj09 IS NOT NULL AND pd01cj09 <> '' THEN 1 ELSE 0 END) AS filled_cj09,
  SUM(CASE WHEN pd01cj10 IS NOT NULL AND pd01cj10 <> '' THEN 1 ELSE 0 END) AS filled_cj10,
  SUM(CASE WHEN pd01cj11 IS NOT NULL AND pd01cj11 <> '' THEN 1 ELSE 0 END) AS filled_cj11,
  SUM(CASE WHEN pd01cj12 IS NOT NULL AND pd01cj12 <> '' THEN 1 ELSE 0 END) AS filled_cj12,
  SUM(CASE WHEN pd01cj13 IS NOT NULL AND pd01cj13 <> '' THEN 1 ELSE 0 END) AS filled_cj13,
  SUM(CASE WHEN pd01cj14 IS NOT NULL AND pd01cj14 <> '' THEN 1 ELSE 0 END) AS filled_cj14,
  SUM(CASE WHEN pd01cj15 IS NOT NULL AND pd01cj15 <> '' THEN 1 ELSE 0 END) AS filled_cj15
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_bibf;
-- 期望：cj04/cj05/cj06 填充率高（应还/实还/逾期总额），cj07-cj11 低（分段逾期，只有发生时填）

-- 3.3 月度表现状态字段（CD01/CD02，R2/R3 账户专用）
SELECT
  pd01ad01 AS account_type,
  COUNT(*) AS total,
  SUM(CASE WHEN lm.pd01cd01 IS NOT NULL AND lm.pd01cd01 <> '' THEN 1 ELSE 0 END) AS filled_cd01,
  SUM(CASE WHEN lm.pd01cd02 IS NOT NULL AND lm.pd01cd02 <> '' THEN 1 ELSE 0 END) AS filled_cd02
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_bibf lm
JOIN brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf b
  ON lm.reportsn = b.reportsn AND lm.seq = b.seq
GROUP BY b.pd01ad01;
-- 看不同账户类型（D1/R1/R2/...）哪些字段有值


-- ============================================================
-- Step 4: 码值分布（验证码值表覆盖度）
--   关注点：是否出现字典里没有的新码值
-- ============================================================

-- 4.1 五级分类 PD01BD03 码值分布
SELECT
  pd01bd03,
  COUNT(*) AS n
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf
WHERE pd01bd03 IS NOT NULL AND pd01bd03 <> ''
GROUP BY pd01bd03
ORDER BY n DESC;
-- 期望：1/2/3/4/5/9（正常/关注/次级/可疑/损失/未分类）

-- 4.2 账户类型 PD01AD01 码值分布
SELECT
  pd01ad01,
  COUNT(*) AS n
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf
GROUP BY pd01ad01
ORDER BY n DESC;
-- 期望：D1/R1/R2/R3/R4/C1；其它码值说明 spark_parse_pbc.py 要扩展类型

-- 4.3 业务种类 PD01AD02（验证码值表完整性）
SELECT
  pd01ad02,
  COUNT(*) AS n
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf
WHERE pd01ad02 IS NOT NULL AND pd01ad02 <> ''
GROUP BY pd01ad02
ORDER BY n DESC
LIMIT 30;


-- ============================================================
-- Step 5: paystate 5 年明细长度分布
--   关注点：60 月是不是真的最长？有没有 >60？有没有全是空？
-- ============================================================
SELECT
  MIN(n) AS min_months,
  MAX(n) AS max_months,
  AVG(n) AS avg_months,
  PERCENTILE(CAST(n AS DOUBLE), 0.5) AS p50,
  PERCENTILE(CAST(n AS DOUBLE), 0.95) AS p95
FROM (
  SELECT reportsn, seq, COUNT(*) AS n
  FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_5_year_detail_bibf
  GROUP BY reportsn, seq
) t;
-- 期望 max_months <= 60；如果 >60 要在 SQL 里 SLICE 取最后 60 月


-- ============================================================
-- Step 6: tran_date 格式确认
--   是 'YYYYMMDD' 字符串还是 'YYYY-MM-DD' 还是 DATE 类型？
-- ============================================================
SELECT DISTINCT
  tran_date,
  LENGTH(tran_date) AS len
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_bibf
LIMIT 5;
-- 长度 8 = 'YYYYMMDD' 字符串（我假设的）
-- 长度 10 = 'YYYY-MM-DD'
-- 其它 = 需要确认


-- ============================================================
-- Step 7: score 表覆盖率（人行评分）
--   字典说 4.7% 报告有 score；生产验证
-- ============================================================
SELECT
  (SELECT COUNT(DISTINCT reportsn) FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf) AS total_reports,
  (SELECT COUNT(DISTINCT reportsn) FROM brm_wyd_ods_mask.cris_pbcg2_score_bibf) AS reports_with_score,
  ROUND(100.0 * (SELECT COUNT(DISTINCT reportsn) FROM brm_wyd_ods_mask.cris_pbcg2_score_bibf)
        / NULLIF((SELECT COUNT(DISTINCT reportsn) FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf), 0), 2) AS pct;
-- 期望：4-5%。如果远高于，说明生产入库比 mock 全


-- ============================================================
-- 跑完反馈重点
--   1. 哪些金矿字段填充率 > 80%（必加入模型）
--   2. 哪些字段填充率 < 5%（暂缓加入）
--   3. 码值表是否完整（有无字典外新值）
--   4. tran_date 格式
--   5. 60 月 paystate 是否真最长 60
-- ============================================================
