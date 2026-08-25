"""PBC 二代征信字段定义（SQL 管道版，对齐 scripts/sql/mvp_user_d1.sql）。

数据来源：jiahuanluo_ind.pbc_struct_materialized_mvp（Spark 物化表 dump）
所有 encode（cat → int code_id、paystate → vocab id + 60 月左 pad）已在 SQL 完成；
本文件只定义维度和字段顺序，供 dataset / collator / model 对齐。

字段顺序必须与 SQL SELECT 输出一致，改这里必须同步改 SQL。
"""
from __future__ import annotations

# ============================================================
# 账户类型（pd01ad01）— 6 类（含 C1 催收账户）
# ============================================================

ACCOUNT_TYPES = ['D1', 'R1', 'R2', 'R3', 'R4', 'C1']

# ============================================================
# 还款状态字母表（account_latest_5_year_detail.pd01ed01）
# 与 mvp_user_d1.sql v_accounts_paystate 的 CASE encode 完全一致：
#   '' → 0（PAD）；vocab 外的值 → 1（UNK）；其余按下表
# ============================================================

PAYSTATE_VOCAB = {
    '<PAD>': 0,
    '<UNK>': 1,
    '#': 2,   # 未知
    '*': 3,   # 当月未出账/未使用
    'M': 4,   # 约定还款日后月底前还
    '1': 5,   # 逾期 1-30 天
    '2': 6,
    '3': 7,
    '4': 8,
    '5': 9,
    '6': 10,
    '7': 11,  # 逾期 180+ 天
    'B': 12,  # 呆账
    'C': 13,  # 结清/销户
    'G': 14,  # 结束
    'D': 15,  # 担保人代还
    'Z': 16,  # 以资抵债
    'N': 17,  # 正常还款
    'A': 18,  # 账单日调整（R2 专用）
    'E': 19,  # 特殊事件（预留）
}
PAYSTATE_VOCAB_SIZE = max(PAYSTATE_VOCAB.values()) + 1  # 20
PAYSTATE_LEN = 60  # 最近 5 年 = 60 月

# ============================================================
# User 分支（固定维度）
# ============================================================

# categorical 14 字段：(db_field, code_table_name)
# 顺序 = SQL v_user 的 *_id 输出顺序（pb01ad01 → pb030d01）+ cert 地区 2 项（转换器注入）
USER_CAT_FIELDS = [
    ('pb01ad01', '性别代码表'),            # person_identity
    ('pb01ad02', '学历代码表'),
    ('pb01ad03', '学位代码表'),
    ('pb01ad04', '就业状况代码表'),
    ('pb01ad05', '世界各国和地区名称代码'),  # 国籍
    ('pb020d01', '婚姻状况代码表'),         # person_marriage
    ('pb040d02', '单位性质代码表'),         # person_professional
    ('pb040d03', '国民经济行业代码表'),
    ('pb040d04', '职业代码表'),
    ('pb040d05', '职务代码表'),
    ('pb040d06', '职称代码表'),
    ('pb030d01', '居住状况代码表'),         # person_residence
    # 生产报文内出生日期/地址是哈希值，户籍地区从 cert_no_mask 提取
    # （A 格式 18 位、前 14 位明文；口径同 mvp_user_d1.sql 的 v_latest_report）
    ('cert_prov', '行政区划代码表(省级)'),    # cert_no_mask 第 1-2 位
    ('cert_city', '行政区划代码表(地市级)'),  # cert_no_mask 第 1-4 位
]

# numeric 32 字段 = 13 基础 + 19 report 级聚合（顺序 = SQL v_user 的 user_numeric 数组顺序）
# score 块不用于特征：生产覆盖率仅 4.7%，与 mock 100% 覆盖存在域偏移；如需恢复按 v_user 补列
USER_NUMERIC_FIELDS = [
    'age_years',                     # cert_no_mask 第 7-14 位出生日期推算（报文内为哈希）
    'num_mobiles',                   # person_mobile 记录数
    'num_residences',                # person_residence 记录数
    'num_professionals',             # person_professional 记录数
    'has_marriage',                  # person_marriage 是否存在
    'years_since_earliest_mobile',   # 最早手机更新距报告年份
    'years_since_latest_mobile',     # 最近手机更新距报告年份
    'years_current_employer',        # pb040r01 进入本单位年份距今
    'has_email',                     # pb01aq01_mask 非空
    'num_identity_other_docs',       # header_identity_other 记录数
    'years_since_professional_update',  # pb040r02 职业信息更新距今
    'years_at_current_address',      # pb030r01 居住信息更新距今
    'marriage_record_count',         # person_marriage 记录条数
    # ---- 以下 19 项为 report 级统计衍生特征（P2，转换器 Python 端计算；生产侧在 v_user
    #      按同名公式补列即可对齐，金额与大计数 log1p，比率/小计数原值）----
    'acc_total_count',               # log1p(len(accountInfos))
    'acc_active_ratio',              # 活跃账户(latestMonthPayState 存在) / 总数
    'acc_bad_count',                 # log1p(# 呆账: bd01='4' 或 cd01='5')
    'acc_overdue_count',             # log1p(# 逾期: bd01='2' 或 cd01 in {2,3} 或 cj06>0)
    'acc_total_balance',             # log1p(Σ pd01bj01)
    'acc_total_overdue',             # log1p(Σ max(pd01bj02, pd01cj06>0 ? cj06 : 0))
    'acc_max_overdue',               # log1p(max 单账户逾期金额)
    'd1_count',                      # log1p(# D1)
    'd1_total_amount',               # log1p(Σ D1 pd01aj01)
    'd1_total_balance',              # log1p(Σ D1 pd01bj01)
    'r2_count',                      # log1p(# R2)
    'r2_total_limit',                # log1p(Σ R2 pd01aj02)
    'r2_total_used',                 # log1p(Σ R2 已用：活跃 pd01cj02，呆账/逾期关闭 pd01bj01)
    'r2_mean_util',                  # mean(R2 使用率 cj02/aj02 clamp[0,1.5]，仅额度>0)
    'r2_overlimit_count',            # log1p(# R2 使用率>1.0)
    'overdue_months_total',          # log1p(Σ 每账户 5 年明细中逾期字符[1-7BDG]计数)
    'query_count_1m',                # log1p(# 查询 距报告日 ≤31 天)
    'query_count_2y',                # log1p(# 查询 距报告日 ≤730 天)
    'query_loan_approval_1m',        # log1p(# 查询 原因 in {02,24} 且 ≤31 天)
]

USER_NUMERIC_DIM = len(USER_NUMERIC_FIELDS)        # 32（13 基础 + 19 聚合，不含 score）
USER_CAT_DIM = len(USER_CAT_FIELDS)                # 14（12 报文字段 + cert 地区 2 项）

# ============================================================
# Account 分支（6 类账户共享字段定义，变长 N）
# ============================================================

# categorical 13 字段：顺序 = SQL v_all_accounts 的 *_id 输出顺序
#   pd01ad02-ad10 来自 account_basic；pd01bd01/bd03/bd04 来自 account_latest_info
ACCOUNT_CAT_FIELDS = [
    ('pd01ad02', '机构类型代码'),
    ('pd01ad03', '个人借贷交易业务种类代码表'),
    ('pd01ad04', '币种代码表'),
    ('pd01ad05', '个人借贷交易还款方式代码表'),
    ('pd01ad06', '个人借贷交易还款频率代码表'),
    ('pd01ad07', '个人借贷交易担保方式代码表'),
    ('pd01ad08', '个人贷款发放形式代码表'),
    ('pd01ad09', '个人借贷交易共同借款标志代码表'),
    ('pd01ad10', '债权转移时的还款状态代码表'),
    ('pd01bd01', '个人借贷账户状态代码表(D1账户)'),
    ('pd01bd03', '五级分类代码表'),
    ('pd01bd04', '区分不了账户类型是R1还是D1\\R4'),
    ('pd01cd01', '个人借贷账户状态代码表(R2/R3账户)'),   # 活跃账户状态（latest_month_pay_state）
]

# numeric 13 字段：顺序 = SQL v_all_accounts 的 numeric 数组顺序
ACCOUNT_NUMERIC_FIELDS = [
    'pd01aj01',               # 借款金额（元）
    'pd01aj02',               # 账户授信额度（元）
    'pd01aj03',               # 共享授信额度（元）
    'pd01as01',               # 还款期数
    'pd01bj01',               # 余额（latest_info，元）
    'pd01bj02',               # 最近一次还款金额/逾期金额（latest_info，元，语义以生产确认为准）
    'pd01cj02',               # 已用额度（latest_month_pay_state，元）
    'pd01cj06',               # 当前逾期总额（latest_month_pay_state，元）
    'credit_utilization',     # cj02/aj02 使用率 clamp [0,1.5]，贷款类=0（管道端派生）
    'account_age_years',      # pd01ar01 发放日期距报告日期
    'years_to_maturity',      # pd01ar02 到期日期距报告日期（可 >0 未到期）
    'special_trades_count',   # 特殊交易笔数（聚合）
    'special_trades_amount',  # 特殊交易总金额（聚合，元）
]

ACCOUNT_NUMERIC_DIM = len(ACCOUNT_NUMERIC_FIELDS)  # 13
ACCOUNT_CAT_DIM = len(ACCOUNT_CAT_FIELDS)          # 13
