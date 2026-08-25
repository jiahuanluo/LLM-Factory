-- ============================================================
-- 01 schema 摸底：列出可用表 + DESCRIBE + 抽样 1 行
-- 目的：确认实际 schema 跟字典是否一致，发现未在字典里的 ETL 字段
-- ============================================================

-- ============================================================
-- Step 1: 列出 brm_wyd_ods_mask 下所有 cris_pbcg2_*_bibf 表
-- ============================================================
SHOW TABLES IN brm_wyd_ods_mask LIKE 'cris_pbcg2_*_bibf';
-- 期望 56 张；如果不到 56，说明有些表生产没建或改名


-- ============================================================
-- Step 2: 每张相关表 DESCRIBE（看实际字段类型 + 是否有 ETL 列）
--   关注点：
--   - 字段类型（VARCHAR vs INT vs DECIMAL vs DATE）
--   - 是否有 tran_date / create_time / etl_batch_id 等 ETL 列
--   - 字段名大小写
-- ============================================================

-- === User 分支相关 ===
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_person_identity_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_person_marriage_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_person_mobile_bibf;
DESCRIBE erm_tm_ods_mask.cris_pbcg2_person_professional_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_person_residence_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_header_identity_other_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_score_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_report_bibf;       -- 含 cert_no_mask（A 格式，前 14 位明文）

-- === cert_no_mask 格式确认（决定能否提取地区码 + 出生年份）===
-- 中国身份证 18 位：1-6 地区码 / 7-14 出生日期 / 15-17 序列 / 18 校验
-- 看实际 mask 保留了哪几位明文
SELECT
  cert_no_mask,
  LENGTH(cert_no_mask) AS len,
  SUBSTR(cert_no_mask, 1, 6)  AS first_6,    -- 地区码
  SUBSTR(cert_no_mask, 7, 4)  AS birth_year, -- 出生年份（7-10 位）
  SUBSTR(cert_no_mask, 11, 4) AS birth_md,   -- 出生月日（11-14 位）
  SUBSTR(cert_no_mask, 15, 4) AS last_4      -- 序列+校验
FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf
WHERE ds = '20260811' AND cert_no_mask IS NOT NULL AND cert_no_mask <> ''
LIMIT 10;

-- === D1 账户相关 ===
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf;
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf;          -- ⭐ 含五级分类
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_bibf; -- ⭐ 含 CJ06/CJ12 金矿
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_account_latest_5_year_detail_bibf;  -- 60 月 paystate 来源
DESCRIBE brm_wyd_ods_mask.cris_pbcg2_account_special_trade_bibf;


-- ============================================================
-- Step 3: 抽样 1 行看真实数据
--   关注点：
--   - 字段实际值格式（如 tran_date 是 '20250124' 还是 '2025-01-24'）
--   - 空值是 NULL 还是 ''
--   - 码值是字符串还是数字（pd01ad01='D1' 还是 1）
-- ============================================================

-- === User 分支 ===
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_person_marriage_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_person_mobile_bibf LIMIT 1;
SELECT * FROM erm_tm_ods_mask.cris_pbcg2_person_professional_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_person_residence_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_header_identity_other_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_score_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf LIMIT 1;

-- === D1 账户 ===
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_5_year_detail_bibf LIMIT 1;
SELECT * FROM brm_wyd_ods_mask.cris_pbcg2_account_special_trade_bibf LIMIT 1;


-- ============================================================
-- Step 4: 同 reportsn 多版本情况确认
--   看 person_identity 同一 reportsn 是否真有多行
--   影响最新版本过滤的逻辑（rn=1）
-- ============================================================
SELECT
  reportsn,
  COUNT(*) AS n_versions,
  COUNT(DISTINCT tran_date) AS n_tran_dates,
  MIN(tran_date) AS min_tran,
  MAX(tran_date) AS max_tran
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_bibf
GROUP BY reportsn
HAVING COUNT(*) > 1
LIMIT 10;
-- 如果有输出：同一 reportsn 多版本，必须用 ROW_NUMBER 取最新
-- 如果无输出：每 reportsn 唯一，不用 ROW_NUMBER（pipeline 简化）


-- ============================================================
-- Step 5: reportsn 跟账户 seq 的唯一性
--   account_basic 主键应该是 (reportsn, seq)
-- ============================================================
SELECT
  reportsn,
  seq,
  COUNT(*) AS n_rows
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_bibf
GROUP BY reportsn, seq
HAVING COUNT(*) > 1
LIMIT 10;
-- 期望无输出。如果有：可能 (reportsn, seq) 还要加 tran_date 才唯一


-- ============================================================
-- 跑完反馈
--   - 56 张表都 SHOW 出来了吗？
--   - 字段类型跟我假设的一致吗？（tran_date 是 STRING 还是 DATE？）
--   - 空值是 NULL 还是 ''？
--   - 多版本情况？
-- 我根据反馈改 mvp_user_d1.sql
-- ============================================================
