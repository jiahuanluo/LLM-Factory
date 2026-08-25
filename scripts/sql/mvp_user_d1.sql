-- ============================================================
-- PBC pbc_struct MVP SQL：user + 6 类账户（按 DB 字典重写）
-- 字段语义以《个人征信DB表结构字典.xlsx》为准
-- ============================================================
-- 设计说明：
-- 1. cat 字段在 SQL 里 JOIN cat_vocab 直接 encode 到 int（code_id）
--    输出 cat_ids + cat_mask 数组（cat_mask: 原值非空时=1）
-- 2. reportsn 取 ds='20260811' 分区内最新 tran_date 版本
-- 3. PII 字段（手机/邮箱/地址/出生日期/证件号）走 _mask 或 _hash 后缀
-- 4. paystate 在 SQL 里 SORT_ARRAY + SLICE(-60) + TRANSFORM(CASE encode)
--    + CONCAT 左 pad，输出 60-int 数组（0=PAD）
-- 5. 库：brm_wyd_ods_mask（普通表）+ erm_tm_ods_mask（敏感表 person_professional）
-- 6. 输出 pbc_struct 字段：
--    user_numeric[13]（基础，不含 score 块；+19 个 report 级聚合由 Python 端补齐后共 32）,
--    user_cat_ids[12], user_cat_mask[12]
--    d1/r1/r2/r3/r4/c1: [{numeric[13], cat_ids[13], cat_mask[13], paystate[60]}]
--    Python 端只需 flatten STRUCT list → per-type flat arrays
-- ============================================================


-- ============================================================
-- Step 1: reportsn 主表
--   _dcb 下同 reportsn 可能有多个 tran_date 版本，用 ROW_NUMBER 取最新
--   cert_no_mask：A 格式 18 位，前 14 位明文（地区码 + 出生日期）
-- ============================================================
CREATE OR REPLACE TEMP VIEW v_latest_report AS
SELECT reportsn, latest_tran_date, cert_no_mask
FROM (
  SELECT reportsn,
         tran_date AS latest_tran_date,
         cert_no_mask,
         ROW_NUMBER() OVER (PARTITION BY reportsn ORDER BY tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_report_dcb
  WHERE ds = '20260811'
) t
WHERE rn = 1;


-- ============================================================
-- Step 2: User 分支
-- ============================================================

-- 2.1 person_identity 最新版（同 reportsn 多版本取最新）
CREATE OR REPLACE TEMP VIEW v_identity AS
SELECT i.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn ORDER BY t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb t
  WHERE t.ds = '20260811'
) i
WHERE i.rn = 1;

-- 2.2 person_marriage 最新版
CREATE OR REPLACE TEMP VIEW v_marriage AS
SELECT m.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn ORDER BY t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_person_marriage_dcb t
  WHERE t.ds = '20260811'
) m
WHERE m.rn = 1;

-- 2.3 person_professional 最新版（取 seq ASC，第一条为当前在职）
CREATE OR REPLACE TEMP VIEW v_professional AS
SELECT p.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn ORDER BY t.seq ASC, t.tran_date DESC) AS rn
  FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb t
  WHERE t.ds = '20260811'
) p
WHERE p.rn = 1;

-- 2.4 person_residence 最新版（取 seq ASC）
CREATE OR REPLACE TEMP VIEW v_residence AS
SELECT r.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn ORDER BY t.seq ASC, t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_person_residence_dcb t
  WHERE t.ds = '20260811'
) r
WHERE r.rn = 1;

-- 2.5 score 最新版（4.7% 报告有 score）
CREATE OR REPLACE TEMP VIEW v_score AS
SELECT s.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn ORDER BY t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_score_dcb t
  WHERE t.ds = '20260811'
) s
WHERE s.rn = 1;

-- 2.6 手机号最早/最近更新日期（聚合后 JOIN，避免相关子查询）
CREATE OR REPLACE TEMP VIEW v_mobile_stats AS
SELECT reportsn,
       MIN(pb01br01) AS earliest_mobile_update,
       MAX(pb01br01) AS latest_mobile_update
FROM brm_wyd_ods_mask.cris_pbcg2_person_mobile_dcb
WHERE ds = '20260811'
GROUP BY reportsn;


-- ============================================================
-- Step 2.7: User 最终视图
--   user_numeric 18 + user_cat 12
--   字段语义全部按字典
-- ============================================================
CREATE OR REPLACE TEMP VIEW v_user AS
SELECT
  lr.reportsn,

  -- === user_numeric (18) ===
  -- 1. age_years：从 cert_no_mask 第 7-10 位（出生年份）算
  CASE WHEN LENGTH(lr.cert_no_mask) >= 10
            AND REGEXP_LIKE(SUBSTR(lr.cert_no_mask, 7, 4), '^[0-9]{4}$')
       THEN CAST(SUBSTR(lr.latest_tran_date, 1, 4) AS INT)
            - CAST(SUBSTR(lr.cert_no_mask, 7, 4) AS INT)
       ELSE 0.0 END AS age_years,
  -- 2. num_mobiles：person_mobile 记录数
  (SELECT COUNT(*) FROM brm_wyd_ods_mask.cris_pbcg2_person_mobile_dcb
   WHERE reportsn = lr.reportsn AND ds = '20260811') AS num_mobiles,
  -- 3. num_residences：person_residence 记录数
  (SELECT COUNT(*) FROM brm_wyd_ods_mask.cris_pbcg2_person_residence_dcb
   WHERE reportsn = lr.reportsn AND ds = '20260811') AS num_residences,
  -- 4. num_professionals：person_professional 记录数
  (SELECT COUNT(*) FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
   WHERE reportsn = lr.reportsn AND ds = '20260811') AS num_professionals,
  -- 5. has_marriage：person_marriage 是否存在记录
  CASE WHEN m.reportsn IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_marriage,
  -- 6. years_since_earliest_mobile_update（日期格式 yyyy-MM-dd）
  CASE WHEN mob.earliest_mobile_update IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(lr.latest_tran_date, 'yyyyMMdd'),
                           to_date(mob.earliest_mobile_update, 'yyyy-MM-dd')) / 365.25, 4)
       ELSE 0.0 END AS years_since_earliest_mobile,
  -- 7. years_since_latest_mobile_update
  CASE WHEN mob.latest_mobile_update IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(lr.latest_tran_date, 'yyyyMMdd'),
                           to_date(mob.latest_mobile_update, 'yyyy-MM-dd')) / 365.25, 4)
       ELSE 0.0 END AS years_since_latest_mobile,
  -- 8. years_current_employer：进入本单位年份（4 位字符串）距今
  CASE WHEN p.pb040r01 IS NOT NULL AND LENGTH(p.pb040r01) = 4
            AND REGEXP_LIKE(p.pb040r01, '^[0-9]{4}$')
       THEN CAST(SUBSTR(lr.latest_tran_date, 1, 4) AS INT) - CAST(p.pb040r01 AS INT)
       ELSE 0.0 END AS years_current_employer,
  -- 9. has_email：邮箱（_mask 后缀）是否存在
  CASE WHEN i.pb01aq01_mask IS NOT NULL AND i.pb01aq01_mask <> '' THEN 1.0 ELSE 0.0 END AS has_email,
  -- 10. num_identity_other_docs：header_identity_other 记录数
  (SELECT COUNT(*) FROM brm_wyd_ods_mask.cris_pbcg2_header_identity_other_dcb
   WHERE reportsn = lr.reportsn AND ds = '20260811') AS num_identity_other_docs,
  -- 11. years_since_professional_update：职业信息更新日期距今
  CASE WHEN p.pb040r02 IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(lr.latest_tran_date, 'yyyyMMdd'),
                           to_date(p.pb040r02, 'yyyy-MM-dd')) / 365.25, 4)
       ELSE 0.0 END AS years_since_professional_update,
  -- 12. years_at_current_address：居住信息更新日期距今
  CASE WHEN rs.pb030r01 IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(lr.latest_tran_date, 'yyyyMMdd'),
                           to_date(rs.pb030r01, 'yyyy-MM-dd')) / 365.25, 4)
       ELSE 0.0 END AS years_at_current_address,
  -- 13. marriage_record_count：person_marriage 记录条数（>1 表示有变更）
  (SELECT COUNT(*) FROM brm_wyd_ods_mask.cris_pbcg2_person_marriage_dcb
   WHERE reportsn = lr.reportsn AND ds = '20260811') AS marriage_record_count,
  -- 14-18. score 5 字段
  CASE WHEN s.pc010q01 IS NOT NULL AND s.pc010q01 <> ''
       THEN CAST(s.pc010q01 AS DOUBLE) ELSE 0.0 END AS score_value,
  CASE WHEN s.pc010q02 IS NOT NULL AND s.pc010q02 <> ''
       THEN CAST(s.pc010q02 AS DOUBLE) ELSE 0.0 END AS score_rank,
  CASE WHEN s.pc010s01 IS NOT NULL AND s.pc010s01 <> ''
       THEN CAST(s.pc010s01 AS DOUBLE) ELSE 0.0 END AS score_query_count,
  CASE WHEN s.pc010d01 IS NOT NULL AND s.pc010d01 <> ''
       THEN SIZE(SPLIT(s.pc010d01, ',')) ELSE 0.0 END AS score_num_institutions,
  CASE WHEN s.reportsn IS NOT NULL THEN 1.0 ELSE 0.0 END AS score_present,

  -- === user_cat (12 字段 → encoded int id + mask) ===
  --   cat_id: JOIN cat_vocab，0=UNK；cat_mask: 原值非空时=1
  COALESCE(v01.code_id, 0) AS pb01ad01_id,
  COALESCE(v02.code_id, 0) AS pb01ad02_id,
  COALESCE(v03.code_id, 0) AS pb01ad03_id,
  COALESCE(v04.code_id, 0) AS pb01ad04_id,
  COALESCE(v05.code_id, 0) AS pb01ad05_id,
  COALESCE(v06.code_id, 0) AS pb020d01_id,
  COALESCE(v07.code_id, 0) AS pb040d02_id,
  COALESCE(v08.code_id, 0) AS pb040d03_id,
  COALESCE(v09.code_id, 0) AS pb040d04_id,
  COALESCE(v10.code_id, 0) AS pb040d05_id,
  COALESCE(v11.code_id, 0) AS pb040d06_id,
  COALESCE(v12.code_id, 0) AS pb030d01_id,

  CASE WHEN i.pb01ad01  IS NOT NULL AND i.pb01ad01  <> '' THEN 1 ELSE 0 END AS pb01ad01_mask,
  CASE WHEN i.pb01ad02  IS NOT NULL AND i.pb01ad02  <> '' THEN 1 ELSE 0 END AS pb01ad02_mask,
  CASE WHEN i.pb01ad03  IS NOT NULL AND i.pb01ad03  <> '' THEN 1 ELSE 0 END AS pb01ad03_mask,
  CASE WHEN i.pb01ad04  IS NOT NULL AND i.pb01ad04  <> '' THEN 1 ELSE 0 END AS pb01ad04_mask,
  CASE WHEN i.pb01ad05  IS NOT NULL AND i.pb01ad05  <> '' THEN 1 ELSE 0 END AS pb01ad05_mask,
  CASE WHEN m.pb020d01  IS NOT NULL AND m.pb020d01  <> '' THEN 1 ELSE 0 END AS pb020d01_mask,
  CASE WHEN p.pb040d02  IS NOT NULL AND p.pb040d02  <> '' THEN 1 ELSE 0 END AS pb040d02_mask,
  CASE WHEN p.pb040d03  IS NOT NULL AND p.pb040d03  <> '' THEN 1 ELSE 0 END AS pb040d03_mask,
  CASE WHEN p.pb040d04  IS NOT NULL AND p.pb040d04  <> '' THEN 1 ELSE 0 END AS pb040d04_mask,
  CASE WHEN p.pb040d05  IS NOT NULL AND p.pb040d05  <> '' THEN 1 ELSE 0 END AS pb040d05_mask,
  CASE WHEN p.pb040d06  IS NOT NULL AND p.pb040d06  <> '' THEN 1 ELSE 0 END AS pb040d06_mask,
  CASE WHEN rs.pb030d01 IS NOT NULL AND rs.pb030d01 <> '' THEN 1 ELSE 0 END AS pb030d01_mask
FROM v_latest_report lr
LEFT JOIN v_identity i        ON lr.reportsn = i.reportsn
LEFT JOIN v_marriage m        ON lr.reportsn = m.reportsn
LEFT JOIN v_professional p    ON lr.reportsn = p.reportsn
LEFT JOIN v_residence rs      ON lr.reportsn = rs.reportsn
LEFT JOIN v_score s           ON lr.reportsn = s.reportsn
LEFT JOIN v_mobile_stats mob  ON lr.reportsn = mob.reportsn
-- 12 个 cat_vocab JOIN（user 段）
LEFT JOIN jiahuanluo_ind.cat_vocab v01
  ON v01.section='user' AND v01.code_table='性别代码表' AND v01.code_value=CAST(i.pb01ad01 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v02
  ON v02.section='user' AND v02.code_table='学历代码表' AND v02.code_value=CAST(i.pb01ad02 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v03
  ON v03.section='user' AND v03.code_table='学位代码表' AND v03.code_value=CAST(i.pb01ad03 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v04
  ON v04.section='user' AND v04.code_table='就业状况代码表' AND v04.code_value=CAST(i.pb01ad04 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v05
  ON v05.section='user' AND v05.code_table='世界各国和地区名称代码' AND v05.code_value=CAST(i.pb01ad05 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v06
  ON v06.section='user' AND v06.code_table='婚姻状况代码表' AND v06.code_value=CAST(m.pb020d01 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v07
  ON v07.section='user' AND v07.code_table='单位性质代码表' AND v07.code_value=CAST(p.pb040d02 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v08
  ON v08.section='user' AND v08.code_table='国民经济行业代码表' AND v08.code_value=CAST(p.pb040d03 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v09
  ON v09.section='user' AND v09.code_table='职业代码表' AND v09.code_value=CAST(p.pb040d04 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v10
  ON v10.section='user' AND v10.code_table='职务代码表' AND v10.code_value=CAST(p.pb040d05 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v11
  ON v11.section='user' AND v11.code_table='职称代码表' AND v11.code_value=CAST(p.pb040d06 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab v12
  ON v12.section='user' AND v12.code_table='居住状况代码表' AND v12.code_value=CAST(rs.pb030d01 AS STRING);


-- ============================================================
-- Step 3: 全部 6 类账户分支（D1/R1/R2/R3/R4/C1）
--   字段语义按字典：
--     account_basic: PD01AJ01 借款金额 / AJ02 账户授信额度 / AJ03 共享授信额度 / AS01 还款期数
--     account_latest_info: BJ01 余额 / BJ02 最近一次还款金额 / BD01 账户状态 / BD03 五级分类
--     account_special_trade: FJ01 特殊交易发生金额
--     account_latest_5_year_detail: ER03 月份 / ED01 还款状态 / EJ01 逾期总额
-- ============================================================

-- 3.1 latest_info（按 reportsn+seq 取最新版本，全类型共用）
CREATE OR REPLACE TEMP VIEW v_accounts_latest_info AS
SELECT li.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn, t.seq ORDER BY t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_dcb t
  WHERE t.ds = '20260811'
) li
WHERE li.rn = 1;

-- 3.2 special_trade 聚合（每 reportsn+seq 的笔数 + 总金额）
CREATE OR REPLACE TEMP VIEW v_accounts_special_trade AS
SELECT reportsn, seq,
       COUNT(*) AS cnt,
       SUM(CASE WHEN pd01fj01 IS NOT NULL AND pd01fj01 <> ''
                THEN CAST(pd01fj01 AS DOUBLE) ELSE 0.0 END) AS total_amount
FROM brm_wyd_ods_mask.cris_pbcg2_account_special_trade_dcb
WHERE ds = '20260811'
GROUP BY reportsn, seq;

-- 3.3 paystate 60 月明细 → 60-int 数组（左 pad 0=PAD，state char encode 到 paystate vocab id）
--   Spark：COLLECT_LIST → SORT_ARRAY(by dt asc) → SLICE 最后 60 → TRANSFORM encode → CONCAT 左 pad
CREATE OR REPLACE TEMP VIEW v_accounts_paystate AS
SELECT reportsn, seq,
       CONCAT(
         ARRAY_REPEAT(0, 60 - SIZE(encoded)),
         encoded
       ) AS paystate
FROM (
  SELECT reportsn, seq,
         TRANSFORM(
           SLICE(
             SORT_ARRAY(COLLECT_LIST(STRUCT(
               CAST(pd01er03 AS STRING) AS dt,
               COALESCE(pd01ed01, '') AS state
             ))),
             -60, 60
           ),
           s -> CASE
                  WHEN s.state IS NULL OR s.state = '' THEN 0   -- PAD
                  WHEN s.state = '#' THEN 2                     -- 未知
                  WHEN s.state = '*' THEN 3                     -- 当月未出账
                  WHEN s.state = 'M' THEN 4
                  WHEN s.state = '1' THEN 5                     -- 逾期 1-30 天
                  WHEN s.state = '2' THEN 6
                  WHEN s.state = '3' THEN 7
                  WHEN s.state = '4' THEN 8
                  WHEN s.state = '5' THEN 9
                  WHEN s.state = '6' THEN 10
                  WHEN s.state = '7' THEN 11                    -- 逾期 180+ 天
                  WHEN s.state = 'B' THEN 12                    -- 呆账
                  WHEN s.state = 'C' THEN 13                    -- 结清
                  WHEN s.state = 'G' THEN 14
                  WHEN s.state = 'D' THEN 15                    -- 担保人代还
                  WHEN s.state = 'Z' THEN 16                    -- 以资抵债
                  WHEN s.state = 'N' THEN 17                    -- 正常
                  WHEN s.state = 'A' THEN 18                    -- 账单日调整
                  WHEN s.state = 'E' THEN 19                    -- 特殊事件
                  ELSE 1                                        -- UNK
                END
         ) AS encoded
  FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_5_year_detail_dcb
  WHERE ds = '20260811'
  GROUP BY reportsn, seq
) t;


-- 3.3b latest_month_pay_state（活跃账户月度状态：pd01cd01 账户状态 / pd01cj02 已用额度 / pd01cj06 当前逾期）
--   P1 起：cd01 状态与已用/逾期进入 account cat/numeric；码值表=个人借贷账户状态代码表(R2/R3账户)
CREATE OR REPLACE TEMP VIEW v_accounts_month_pay_state AS
SELECT m.* FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn, t.seq ORDER BY t.tran_date DESC) rn
  FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_month_pay_state_dcb t
  WHERE t.ds = '20260811'
) m
WHERE m.rn = 1;


-- 3.4 所有 6 类账户统一视图（按 pd01ad01 区分类型）
--   每行 = 1 个账户；同 reportsn 可有多行多类型
CREATE OR REPLACE TEMP VIEW v_all_accounts AS
SELECT
  b.reportsn,
  b.seq AS account_seq,
  b.pd01ad01 AS account_type,   -- D1/R1/R2/R3/R4/C1

  -- === cat (12 字段 → encoded int id + mask) ===
  COALESCE(va01.code_id, 0) AS pd01ad02_id,
  COALESCE(va02.code_id, 0) AS pd01ad03_id,
  COALESCE(va03.code_id, 0) AS pd01ad04_id,
  COALESCE(va04.code_id, 0) AS pd01ad05_id,
  COALESCE(va05.code_id, 0) AS pd01ad06_id,
  COALESCE(va06.code_id, 0) AS pd01ad07_id,
  COALESCE(va07.code_id, 0) AS pd01ad08_id,
  COALESCE(va08.code_id, 0) AS pd01ad09_id,
  COALESCE(va09.code_id, 0) AS pd01ad10_id,
  COALESCE(va10.code_id, 0) AS pd01bd01_id,
  COALESCE(va11.code_id, 0) AS pd01bd03_id,
  COALESCE(va12.code_id, 0) AS pd01bd04_id,
  COALESCE(va13.code_id, 0) AS pd01cd01_id,

  CASE WHEN b.pd01ad02  IS NOT NULL AND b.pd01ad02  <> '' THEN 1 ELSE 0 END AS pd01ad02_mask,
  CASE WHEN b.pd01ad03  IS NOT NULL AND b.pd01ad03  <> '' THEN 1 ELSE 0 END AS pd01ad03_mask,
  CASE WHEN b.pd01ad04  IS NOT NULL AND b.pd01ad04  <> '' THEN 1 ELSE 0 END AS pd01ad04_mask,
  CASE WHEN b.pd01ad05  IS NOT NULL AND b.pd01ad05  <> '' THEN 1 ELSE 0 END AS pd01ad05_mask,
  CASE WHEN b.pd01ad06  IS NOT NULL AND b.pd01ad06  <> '' THEN 1 ELSE 0 END AS pd01ad06_mask,
  CASE WHEN b.pd01ad07  IS NOT NULL AND b.pd01ad07  <> '' THEN 1 ELSE 0 END AS pd01ad07_mask,
  CASE WHEN b.pd01ad08  IS NOT NULL AND b.pd01ad08  <> '' THEN 1 ELSE 0 END AS pd01ad08_mask,
  CASE WHEN b.pd01ad09  IS NOT NULL AND b.pd01ad09  <> '' THEN 1 ELSE 0 END AS pd01ad09_mask,
  CASE WHEN b.pd01ad10  IS NOT NULL AND b.pd01ad10  <> '' THEN 1 ELSE 0 END AS pd01ad10_mask,
  CASE WHEN li.pd01bd01 IS NOT NULL AND li.pd01bd01 <> '' THEN 1 ELSE 0 END AS pd01bd01_mask,
  CASE WHEN li.pd01bd03 IS NOT NULL AND li.pd01bd03 <> '' THEN 1 ELSE 0 END AS pd01bd03_mask,
  CASE WHEN li.pd01bd04 IS NOT NULL AND li.pd01bd04 <> '' THEN 1 ELSE 0 END AS pd01bd04_mask,
  CASE WHEN mps.pd01cd01 IS NOT NULL AND mps.pd01cd01 <> '' THEN 1 ELSE 0 END AS pd01cd01_mask,

  -- === numeric (10) ===
  CASE WHEN b.pd01aj01 IS NOT NULL AND b.pd01aj01 <> '' THEN CAST(b.pd01aj01 AS DOUBLE) ELSE 0.0 END AS pd01aj01,
  CASE WHEN b.pd01aj02 IS NOT NULL AND b.pd01aj02 <> '' THEN CAST(b.pd01aj02 AS DOUBLE) ELSE 0.0 END AS pd01aj02,
  CASE WHEN b.pd01aj03 IS NOT NULL AND b.pd01aj03 <> '' THEN CAST(b.pd01aj03 AS DOUBLE) ELSE 0.0 END AS pd01aj03,
  CASE WHEN b.pd01as01 IS NOT NULL AND b.pd01as01 <> '' THEN CAST(b.pd01as01 AS DOUBLE) ELSE 0.0 END AS pd01as01,
  CASE WHEN li.pd01bj01 IS NOT NULL AND li.pd01bj01 <> '' THEN CAST(li.pd01bj01 AS DOUBLE) ELSE 0.0 END AS pd01bj01,
  CASE WHEN li.pd01bj02 IS NOT NULL AND li.pd01bj02 <> '' THEN CAST(li.pd01bj02 AS DOUBLE) ELSE 0.0 END AS pd01bj02,
  CASE WHEN mps.pd01cj02 IS NOT NULL AND mps.pd01cj02 <> '' THEN CAST(mps.pd01cj02 AS DOUBLE) ELSE 0.0 END AS pd01cj02,
  CASE WHEN mps.pd01cj06 IS NOT NULL AND mps.pd01cj06 <> '' THEN CAST(mps.pd01cj06 AS DOUBLE) ELSE 0.0 END AS pd01cj06,
  CASE WHEN mps.pd01cj02 IS NOT NULL AND mps.pd01cj02 <> ''
            AND b.pd01aj02 IS NOT NULL AND b.pd01aj02 <> ''
            AND CAST(mps.pd01cj02 AS DOUBLE) > 0 AND CAST(b.pd01aj02 AS DOUBLE) > 0
       THEN LEAST(CAST(mps.pd01cj02 AS DOUBLE) / CAST(b.pd01aj02 AS DOUBLE), 1.5)
       ELSE 0.0 END AS credit_utilization,
  CASE WHEN b.pd01ar01 IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(lr.latest_tran_date, 'yyyyMMdd'),
                           to_date(b.pd01ar01, 'yyyy-MM-dd')) / 365.25, 4)
       ELSE 0.0 END AS account_age_years,
  CASE WHEN b.pd01ar02 IS NOT NULL
       THEN ROUND(DATEDIFF(to_date(b.pd01ar02, 'yyyy-MM-dd'),
                           to_date(lr.latest_tran_date, 'yyyyMMdd')) / 365.25, 4)
       ELSE 0.0 END AS years_to_maturity,
  COALESCE(st.cnt, 0) AS special_trades_count,
  COALESCE(st.total_amount, 0.0) AS special_trades_amount,

  COALESCE(pay.paystate, ARRAY_REPEAT(0, 60)) AS paystate   -- 无 5 年明细的账户 → 全 PAD
FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY t.reportsn, t.seq ORDER BY t.tran_date DESC) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb t
  WHERE t.ds = '20260811'
    AND t.pd01ad01 IN ('D1', 'R1', 'R2', 'R3', 'R4', 'C1')   -- 6 类账户
) b
JOIN v_latest_report lr ON b.reportsn = lr.reportsn
LEFT JOIN v_accounts_latest_info li ON b.reportsn = li.reportsn AND b.seq = li.seq
LEFT JOIN v_accounts_special_trade st ON b.reportsn = st.reportsn AND b.seq = st.seq
LEFT JOIN v_accounts_paystate pay ON b.reportsn = pay.reportsn AND b.seq = pay.seq
LEFT JOIN v_accounts_month_pay_state mps ON b.reportsn = mps.reportsn AND b.seq = mps.seq
-- 12 个 cat_vocab JOIN（account 段）
LEFT JOIN jiahuanluo_ind.cat_vocab va01
  ON va01.section='account' AND va01.code_table='机构类型代码' AND va01.code_value=CAST(b.pd01ad02 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va02
  ON va02.section='account' AND va02.code_table='个人借贷交易业务种类代码表' AND va02.code_value=CAST(b.pd01ad03 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va03
  ON va03.section='account' AND va03.code_table='币种代码表' AND va03.code_value=CAST(b.pd01ad04 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va04
  ON va04.section='account' AND va04.code_table='个人借贷交易还款方式代码表' AND va04.code_value=CAST(b.pd01ad05 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va05
  ON va05.section='account' AND va05.code_table='个人借贷交易还款频率代码表' AND va05.code_value=CAST(b.pd01ad06 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va06
  ON va06.section='account' AND va06.code_table='个人借贷交易担保方式代码表' AND va06.code_value=CAST(b.pd01ad07 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va07
  ON va07.section='account' AND va07.code_table='个人贷款发放形式代码表' AND va07.code_value=CAST(b.pd01ad08 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va08
  ON va08.section='account' AND va08.code_table='个人借贷交易共同借款标志代码表' AND va08.code_value=CAST(b.pd01ad09 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va09
  ON va09.section='account' AND va09.code_table='债权转移时的还款状态代码表' AND va09.code_value=CAST(b.pd01ad10 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va10
  ON va10.section='account' AND va10.code_table='个人借贷账户状态代码表(D1账户)' AND va10.code_value=CAST(li.pd01bd01 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va11
  ON va11.section='account' AND va11.code_table='五级分类代码表' AND va11.code_value=CAST(li.pd01bd03 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va12
  ON va12.section='account' AND va12.code_table='区分不了账户类型是R1还是D1\R4' AND va12.code_value=CAST(li.pd01bd04 AS STRING)
LEFT JOIN jiahuanluo_ind.cat_vocab va13
  ON va13.section='account' AND va13.code_table='个人借贷账户状态代码表(R2/R3账户)' AND va13.code_value=CAST(mps.pd01cd01 AS STRING)
WHERE b.rn = 1;

-- 3.5 按类型分视图（兼容下游：v_d1_accounts / v_r1_accounts / ...）
CREATE OR REPLACE TEMP VIEW v_d1_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'D1';
CREATE OR REPLACE TEMP VIEW v_r1_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'R1';
CREATE OR REPLACE TEMP VIEW v_r2_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'R2';
CREATE OR REPLACE TEMP VIEW v_r3_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'R3';
CREATE OR REPLACE TEMP VIEW v_r4_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'R4';
CREATE OR REPLACE TEMP VIEW v_c1_accounts AS SELECT * FROM v_all_accounts WHERE account_type = 'C1';


-- ============================================================
-- Step 4.6: 预聚合 6 类账户 → 每类 1 行/reportsn
--   避免 Step 5 里 scalar subquery + COLLECT_LIST 被 Spark 优化成 JOIN 导致行膨胀
--   每个视图输出：reportsn + accounts_list (COLLECT_LIST of STRUCT)
-- ============================================================

CREATE OR REPLACE TEMP VIEW v_d1_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_d1_accounts
GROUP BY reportsn;

CREATE OR REPLACE TEMP VIEW v_r1_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_r1_accounts
GROUP BY reportsn;

CREATE OR REPLACE TEMP VIEW v_r2_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_r2_accounts
GROUP BY reportsn;

CREATE OR REPLACE TEMP VIEW v_r3_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_r3_accounts
GROUP BY reportsn;

CREATE OR REPLACE TEMP VIEW v_r4_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_r4_accounts
GROUP BY reportsn;

CREATE OR REPLACE TEMP VIEW v_c1_collected AS
SELECT reportsn,
       COLLECT_LIST(NAMED_STRUCT(
         'numeric', ARRAY(
           pd01aj01, pd01aj02, pd01aj03, pd01as01,
           pd01bj01, pd01bj02,
           pd01cj02, pd01cj06, credit_utilization,
           account_age_years, years_to_maturity,
           special_trades_count, special_trades_amount
         ),
         'cat_ids', ARRAY(
           pd01ad02_id, pd01ad03_id, pd01ad04_id, pd01ad05_id, pd01ad06_id,
           pd01ad07_id, pd01ad08_id, pd01ad09_id, pd01ad10_id,
           pd01bd01_id, pd01bd03_id, pd01bd04_id, pd01cd01_id
         ),
         'cat_mask', ARRAY(
           pd01ad02_mask, pd01ad03_mask, pd01ad04_mask, pd01ad05_mask, pd01ad06_mask,
           pd01ad07_mask, pd01ad08_mask, pd01ad09_mask, pd01ad10_mask,
           pd01bd01_mask, pd01bd03_mask, pd01bd04_mask, pd01cd01_mask
         ),
         'paystate', paystate
       )) AS accounts_list
FROM v_c1_accounts
GROUP BY reportsn;


-- ============================================================
-- Step 5: 物化到 jiahuanluo_ind.pbc_struct_materialized_mvp
--   关键改动：用 LEFT JOIN 预聚合视图替代 scalar subquery，避免行膨胀
-- ============================================================

DROP TABLE IF EXISTS jiahuanluo_ind.pbc_struct_materialized_mvp;
CREATE TABLE jiahuanluo_ind.pbc_struct_materialized_mvp AS
SELECT
  lr.reportsn,
  TO_JSON(
    NAMED_STRUCT(
      'user_numeric', ARRAY(
        COALESCE(u.age_years, 0.0),
        COALESCE(u.num_mobiles, 0.0),
        COALESCE(u.num_residences, 0.0),
        COALESCE(u.num_professionals, 0.0),
        COALESCE(u.has_marriage, 0.0),
        COALESCE(u.years_since_earliest_mobile, 0.0),
        COALESCE(u.years_since_latest_mobile, 0.0),
        COALESCE(u.years_current_employer, 0.0),
        COALESCE(u.has_email, 0.0),
        COALESCE(u.num_identity_other_docs, 0.0),
        COALESCE(u.years_since_professional_update, 0.0),
        COALESCE(u.years_at_current_address, 0.0),
        COALESCE(u.marriage_record_count, 0.0)
      ),
      'user_cat_ids', ARRAY(
        u.pb01ad01_id, u.pb01ad02_id, u.pb01ad03_id, u.pb01ad04_id, u.pb01ad05_id,
        u.pb020d01_id, u.pb040d02_id, u.pb040d03_id, u.pb040d04_id, u.pb040d05_id,
        u.pb040d06_id, u.pb030d01_id
      ),
      'user_cat_mask', ARRAY(
        u.pb01ad01_mask, u.pb01ad02_mask, u.pb01ad03_mask, u.pb01ad04_mask, u.pb01ad05_mask,
        u.pb020d01_mask, u.pb040d02_mask, u.pb040d03_mask, u.pb040d04_mask, u.pb040d05_mask,
        u.pb040d06_mask, u.pb030d01_mask
      ),
      'd1', d1.accounts_list,
      'r1', r1.accounts_list,
      'r2', r2.accounts_list,
      'r3', r3.accounts_list,
      'r4', r4.accounts_list,
      'c1', c1.accounts_list
    )
  ) AS pbc_struct
FROM v_latest_report lr
LEFT JOIN v_user u           ON lr.reportsn = u.reportsn
LEFT JOIN v_d1_collected d1  ON lr.reportsn = d1.reportsn
LEFT JOIN v_r1_collected r1  ON lr.reportsn = r1.reportsn
LEFT JOIN v_r2_collected r2  ON lr.reportsn = r2.reportsn
LEFT JOIN v_r3_collected r3  ON lr.reportsn = r3.reportsn
LEFT JOIN v_r4_collected r4  ON lr.reportsn = r4.reportsn
LEFT JOIN v_c1_collected c1  ON lr.reportsn = c1.reportsn;


-- ============================================================
-- Step 6: 抽样验证物化表
-- ============================================================

-- 6.1 看 5 行：每行的 reportsn + JSON 长度
-- SELECT reportsn, LENGTH(pbc_struct) AS json_size FROM jiahuanluo_ind.pbc_struct_materialized_mvp LIMIT 5;

-- 6.2 行数（应该跟 v_latest_report 一致 ~79337）
-- SELECT COUNT(*) AS total_rows FROM jiahuanluo_ind.pbc_struct_materialized_mvp;

-- 6.3 抽一份看 JSON 结构（手动 parse 看是否合理）
-- SELECT reportsn, pbc_struct FROM jiahuanluo_ind.pbc_struct_materialized_mvp LIMIT 1;

