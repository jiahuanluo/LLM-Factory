-- ============================================================
-- 04 从生产表扫描所有 cat 字段的 distinct 码值
--   输出：长表 (section, code_table, code_value)
--   用途：构建生产版 cat_vocab.json，替代基于 mock 的版本
-- ============================================================
-- 用法：
--   1. 在 Spark 跑整个脚本，结果是一张长表
--   2. 导出为 TSV/JSON，用 Python 聚合成 cat_vocab_prod.json
--   3. 在 postprocess_pbc_struct.py 用新 vocab 替代旧 cat_vocab.json
-- ============================================================


-- ============================================================
-- 12 个 user cat 字段 → (section='user', code_table, code_value)
-- ============================================================
CREATE OR REPLACE TEMP VIEW v_user_cat_values AS
SELECT 'user' AS section, '性别代码表' AS code_table, pb01ad01 AS code_value
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb
WHERE ds = '20260811' AND pb01ad01 IS NOT NULL AND pb01ad01 <> ''
GROUP BY pb01ad01

UNION ALL
SELECT 'user', '学历代码表', pb01ad02
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb
WHERE ds = '20260811' AND pb01ad02 IS NOT NULL AND pb01ad02 <> ''
GROUP BY pb01ad02

UNION ALL
SELECT 'user', '学位代码表', pb01ad03
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb
WHERE ds = '20260811' AND pb01ad03 IS NOT NULL AND pb01ad03 <> ''
GROUP BY pb01ad03

UNION ALL
SELECT 'user', '就业状况代码表', pb01ad04
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb
WHERE ds = '20260811' AND pb01ad04 IS NOT NULL AND pb01ad04 <> ''
GROUP BY pb01ad04

UNION ALL
SELECT 'user', '世界各国和地区名称代码', pb01ad05
FROM brm_wyd_ods_mask.cris_pbcg2_person_identity_dcb
WHERE ds = '20260811' AND pb01ad05 IS NOT NULL AND pb01ad05 <> ''
GROUP BY pb01ad05

UNION ALL
SELECT 'user', '婚姻状况代码表', pb020d01
FROM brm_wyd_ods_mask.cris_pbcg2_person_marriage_dcb
WHERE ds = '20260811' AND pb020d01 IS NOT NULL AND pb020d01 <> ''
GROUP BY pb020d01

UNION ALL
SELECT 'user', '单位性质代码表', pb040d02
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
WHERE ds = '20260811' AND pb040d02 IS NOT NULL AND pb040d02 <> ''
GROUP BY pb040d02

UNION ALL
SELECT 'user', '国民经济行业代码表', pb040d03
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
WHERE ds = '20260811' AND pb040d03 IS NOT NULL AND pb040d03 <> ''
GROUP BY pb040d03

UNION ALL
SELECT 'user', '职业代码表', pb040d04
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
WHERE ds = '20260811' AND pb040d04 IS NOT NULL AND pb040d04 <> ''
GROUP BY pb040d04

UNION ALL
SELECT 'user', '职务代码表', pb040d05
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
WHERE ds = '20260811' AND pb040d05 IS NOT NULL AND pb040d05 <> ''
GROUP BY pb040d05

UNION ALL
SELECT 'user', '职称代码表', pb040d06
FROM erm_tm_ods_mask.cris_pbcg2_person_professional_dcb
WHERE ds = '20260811' AND pb040d06 IS NOT NULL AND pb040d06 <> ''
GROUP BY pb040d06

UNION ALL
SELECT 'user', '居住状况代码表', pb030d01
FROM brm_wyd_ods_mask.cris_pbcg2_person_residence_dcb
WHERE ds = '20260811' AND pb030d01 IS NOT NULL AND pb030d01 <> ''
GROUP BY pb030d01;


-- ============================================================
-- 12 个 account cat 字段 → (section='account', code_table, code_value)
--   注意 account_basic 单条记录多账户类型（D1/R1/R2/...），
--   扫描时不加 pd01ad01 过滤，覆盖所有类型的码值
-- ============================================================
CREATE OR REPLACE TEMP VIEW v_account_cat_values AS
SELECT 'account' AS section, '机构类型代码' AS code_table, pd01ad02 AS code_value
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad02 IS NOT NULL AND pd01ad02 <> ''
GROUP BY pd01ad02

UNION ALL
SELECT 'account', '个人借贷交易业务种类代码表', pd01ad03
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad03 IS NOT NULL AND pd01ad03 <> ''
GROUP BY pd01ad03

UNION ALL
SELECT 'account', '币种代码表', pd01ad04
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad04 IS NOT NULL AND pd01ad04 <> ''
GROUP BY pd01ad04

UNION ALL
SELECT 'account', '个人借贷交易还款方式代码表', pd01ad05
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad05 IS NOT NULL AND pd01ad05 <> ''
GROUP BY pd01ad05

UNION ALL
SELECT 'account', '个人借贷交易还款频率代码表', pd01ad06
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad06 IS NOT NULL AND pd01ad06 <> ''
GROUP BY pd01ad06

UNION ALL
SELECT 'account', '个人借贷交易担保方式代码表', pd01ad07
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad07 IS NOT NULL AND pd01ad07 <> ''
GROUP BY pd01ad07

UNION ALL
SELECT 'account', '个人贷款发放形式代码表', pd01ad08
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad08 IS NOT NULL AND pd01ad08 <> ''
GROUP BY pd01ad08

UNION ALL
SELECT 'account', '个人借贷交易共同借款标志代码表', pd01ad09
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad09 IS NOT NULL AND pd01ad09 <> ''
GROUP BY pd01ad09

UNION ALL
SELECT 'account', '债权转移时的还款状态代码表', pd01ad10
FROM brm_wyd_ods_mask.cris_pbcg2_account_basic_dcb
WHERE ds = '20260811' AND pd01ad10 IS NOT NULL AND pd01ad10 <> ''
GROUP BY pd01ad10

UNION ALL
SELECT 'account', '个人借贷账户状态代码表(D1账户)', pd01bd01
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_dcb
WHERE ds = '20260811' AND pd01bd01 IS NOT NULL AND pd01bd01 <> ''
GROUP BY pd01bd01

UNION ALL
SELECT 'account', '五级分类代码表', pd01bd03
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_dcb
WHERE ds = '20260811' AND pd01bd03 IS NOT NULL AND pd01bd03 <> ''
GROUP BY pd01bd03

UNION ALL
SELECT 'account', '区分不了账户类型是R1还是D1\R4', pd01bd04
FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_dcb
WHERE ds = '20260811' AND pd01bd04 IS NOT NULL AND pd01bd04 <> ''
GROUP BY pd01bd04;


-- ============================================================
-- 物化到 jiahuanluo_ind.cat_vocab_distinct
-- ============================================================
DROP TABLE IF EXISTS jiahuanluo_ind.cat_vocab_distinct;
CREATE TABLE jiahuanluo_ind.cat_vocab_distinct AS
SELECT section, code_table, code_value
FROM (
  SELECT * FROM v_user_cat_values
  UNION ALL
  SELECT * FROM v_account_cat_values
) t;


-- ============================================================
-- 构建带 code_id 的 vocab 表（用于 mvp_user_d1.sql 的 JOIN encode）
--   code_id 从 1 开始，0 留给 UNK（vocab 中没有的值）
-- ============================================================
DROP TABLE IF EXISTS jiahuanluo_ind.cat_vocab;
CREATE TABLE jiahuanluo_ind.cat_vocab AS
SELECT section,
       code_table,
       code_value,
       ROW_NUMBER() OVER (PARTITION BY section, code_table ORDER BY code_value) AS code_id
FROM jiahuanluo_ind.cat_vocab_distinct;


-- ============================================================
-- 验证：每张码值表的 distinct 值数量 + code_id 范围
-- ============================================================
-- SELECT section, code_table, COUNT(*) AS n_distinct, MIN(code_id) AS min_id, MAX(code_id) AS max_id
-- FROM jiahuanluo_ind.cat_vocab
-- GROUP BY section, code_table
-- ORDER BY section, code_table;


-- ============================================================
-- 导出为 JSONL（在 Spark 端用 .write.json，或 collect 后写文件）
-- ============================================================
-- 每行：{"section":"user", "code_table":"性别代码表", "code_value":"1"}
SELECT section, code_table, code_value
FROM jiahuanluo_ind.cat_vocab_distinct
ORDER BY section, code_table, code_value;
