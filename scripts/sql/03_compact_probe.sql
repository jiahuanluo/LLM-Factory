-- ============================================================
-- 单条精简探针：4 件事一次出，输出 ~5 行
-- ============================================================

-- (1) cert_no_mask 格式 + tran_date 类型 + 同 reportsn 多版本
SELECT
  '=== cert_no_mask 格式 ===' AS section,
  MIN(LENGTH(cert_no_mask)) AS min_len,
  MAX(LENGTH(cert_no_mask)) AS max_len,
  COUNT(DISTINCT cert_no_mask) AS n_distinct
FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf
WHERE ds = '20260811' AND cert_no_mask IS NOT NULL AND cert_no_mask <> ''

UNION ALL

SELECT
  '=== cert_no_mask 样本（前 5） ===' AS section,
  CAST(SUBSTR(MIN(cert_no_mask), 1, 6) AS BIGINT) AS min_len,
  CAST(SUBSTR(MAX(cert_no_mask), 1, 6) AS BIGINT) AS max_len,
  NULL AS n_distinct
FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf
WHERE ds = '20260811'

UNION ALL

-- (2) report 表行数 + 同 reportsn 多版本统计
SELECT
  '=== report 表多版本 ===' AS section,
  COUNT(*) AS min_len,
  COUNT(DISTINCT reportsn) AS max_len,
  SUM(CASE WHEN rn > 1 THEN 1 ELSE 0 END) AS n_distinct
FROM (
  SELECT reportsn, COUNT(*) AS rn
  FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf
  WHERE ds = '20260811'
  GROUP BY reportsn
) t


-- ============================================================
-- 单独跑这个，看 cert_no_mask 实际样本（5 行）
-- ============================================================
-- SELECT cert_no_mask
-- FROM brm_wyd_ods_mask.cris_pbcg2_report_bibf
-- WHERE ds = '20260811' AND cert_no_mask IS NOT NULL
-- LIMIT 5;


-- ============================================================
-- 单独跑这个，看金矿字段填充率（1 行）
-- ============================================================
-- SELECT
--   COUNT(*) AS total,
--   SUM(CASE WHEN pd01bd03 IS NOT NULL AND pd01bd03 <> '' THEN 1 ELSE 0 END) AS filled_bd03,
--   SUM(CASE WHEN pd01cj06 IS NOT NULL AND pd01cj06 <> '' THEN 1 ELSE 0 END) AS filled_cj06,
--   SUM(CASE WHEN pd01cj12 IS NOT NULL AND pd01cj12 <> '' THEN 1 ELSE 0 END) AS filled_cj12
-- FROM brm_wyd_ods_mask.cris_pbcg2_account_latest_info_bibf
-- WHERE ds = '20260811';
