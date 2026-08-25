-- ============================================================
-- SQL 方言探针：测 4 种日期解析写法，看哪种能跑通
--   跑完后告诉我哪个 SELECT 不报错
-- ============================================================

-- 方言 A：Spark 3.x 原生（to_date + datediff）
SELECT
  to_date('20250124', 'yyyyMMdd') AS d1,
  to_date('19850101', 'yyyyMMdd') AS d2,
  datediff(to_date('20250124', 'yyyyMMdd'), to_date('19850101', 'yyyyMMdd')) AS days_diff,
  round(datediff(to_date('20250124', 'yyyyMMdd'), to_date('19850101', 'yyyyMMdd')) / 365.25, 4) AS years_diff;
-- 期望：days_diff ≈ 14675，years_diff ≈ 40.18


-- 方言 B：Spark 老版本（from_unixtime + unix_timestamp）
SELECT
  datediff(
    from_unixtime(unix_timestamp('20250124', 'yyyyMMdd'), 'yyyy-MM-dd'),
    from_unixtime(unix_timestamp('19850101', 'yyyyMMdd'), 'yyyy-MM-dd')
  ) AS days_diff;


-- 方言 C：HiveQL（date_format + to_date）
SELECT
  datediff(
    to_date('2025-01-24'),
    to_date('1985-01-01')
  ) AS days_diff;


-- 方言 D：Presto/Trino（date_parse）
SELECT
  date_diff('day',
            date_parse('19850101', 'yyyyMMdd'),
            date_parse('20250124', 'yyyyMMdd')) AS days_diff;


-- ============================================================
-- 此外：你的 Spark 版本
-- ============================================================
-- 跑这个看版本：
-- SELECT version();
