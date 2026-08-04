"""PBC 二代征信字段定义。

所有字段路径采用点号分隔的 JSON 路径，如 'personInfo.identity.pb01ad01'。
字段名一律小写（与 CrisPbc.json 保持一致；DB 字典 xlsx 里大写，映射时 lower()）。
"""
from __future__ import annotations

# ============================================================
# 账户类型（pd01ad01）— 6 类（含 C1 催收账户，生产数据中常见）
# ============================================================

ACCOUNT_TYPES = ['D1', 'R1', 'R2', 'R3', 'R4', 'C1']

# ============================================================
# 还款状态字母表（latest24PayState / latest5year）
# 来自码值表"个人借贷账户还款状态代码表"
# id 0 = <PAD>; id 1 = <UNK>（实际数据中未出现的符号归 1）
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

# ============================================================
# User 分支字段（固定维度）
# ============================================================

# categorical 字段：(json_path, code_table_name)
# 维度 = 12：原 11 + 职称（pb040d06）
USER_CAT_FIELDS = [
    ('personInfo.identity.pb01ad01', '性别代码表'),          # 性别
    ('personInfo.identity.pb01ad02', '学历代码表'),          # 学历
    ('personInfo.identity.pb01ad03', '学位代码表'),          # 学位
    ('personInfo.identity.pb01ad04', '就业状况代码表'),      # 就业状况
    ('personInfo.identity.pb01ad05', '世界各国和地区名称代码'),  # 国籍
    ('personInfo.marriage.pb020d01', '婚姻状况代码表'),      # 婚姻
    ('personInfo.professionals.0.pb040d02', '单位性质代码表'),  # 单位性质（取最新一条）
    ('personInfo.professionals.0.pb040d03', '国民经济行业代码表'),  # 行业
    ('personInfo.professionals.0.pb040d04', '职业代码表'),   # 职业
    ('personInfo.professionals.0.pb040d05', '职务代码表'),   # 职务
    ('personInfo.professionals.0.pb040d06', '职称代码表'),   # 职称（0/1/2/3/9）
    ('personInfo.residences.0.pb030d01', '居住状况代码表'),  # 居住状况（取最新）
]

# numeric 字段：(json_path_or_callable_name, description)
# 用空字符串占位 path；实际计算在 sample_builder 里做（聚合/日期差等）
# 维度 = 18：基础 13（含 3 个稳定性衍生）+ score 5
USER_NUMERIC_SPECS = [
    ('age_years', '年龄（由 pb01ar01 出生日期计算）'),
    ('num_mobiles', '手机号个数'),
    ('num_residences', '居住地址个数'),
    ('num_professionals', '职业记录个数'),
    ('num_marriage_updates', '婚姻记录条数（>1 表示变更过）'),
    ('years_since_oldest_mobile_update', '最早手机更新距今年份'),
    ('years_since_latest_mobile_update', '最近手机更新距今年份'),
    ('years_current_employer', '进入本单位年份距今'),
    ('has_email', '是否填了邮箱（0/1）'),
    ('num_identity_other_docs', '其他证件数'),
    # 稳定性衍生（生产高价值字段）
    ('years_since_latest_job_end', '最近一份工作结束距今年份（0=在职）'),
    ('years_at_current_address', '当前居住地址居住年限'),
    ('marriage_change_count', '婚姻状态变更次数 pb020d02（数值化；缺失=0）'),
    # score 字段（4.7% 报告含；缺失则 5 项全 0，score_present=0）
    ('score_value', '人行评分 pc010q01（数值化；缺失=0）'),
    ('score_rank', '排名位次 pc010q02（数值化；缺失=0）'),
    ('score_query_count', '累计查询次数 pc010s01（数值化；缺失=0）'),
    ('score_num_institutions', '评分机构数 pc010d01 逗号串计数（缺失=0）'),
    ('score_present', 'score 块是否存在（0/1）'),
]

# ============================================================
# Summary 分支（summaryInfo 下 13 张子表的聚合）
# ============================================================

# (子表名, 是否 list, 字段列表)
# list=True 表示 summaryInfo[子表] 是 list；list=False 表示 dict
# 对 list 子表，做聚合：count + 每个数值字段 sum/max
SUMMARY_TABLES = [
    # (path, is_list, numeric_fields, cat_field_with_code_table)
    ('tradeTips', True,
     ['pc02as01', 'pc02as03'],                       # 账户数合计, 每业务类型账户数
     [('pc02ad01', '个人信贷交易提示业务类型代码表'),
      ('pc02ad02', '业务大类')]),
    ('recoveries', True,
     ['pc02bj01', 'pc02bj02'],                       # 余额合计
     [('pc02bd01', '个人被追偿汇总信息业务类型代码表')]),
    ('badDebit', False,
     ['pc02cj01'],                                   # 呆账余额
     [('pc02cs01', None)]),                          # 仅账户数
    ('overdues', True,
     ['pc02dj01'],                                   # 单月最高逾期总额
     [('pc02dd01', '个人逾期（透支）汇总信息账户类型代码表'),
      ('pc02ds04', None)]),                          # 最长逾期月数
    ('nonrevolvingLoan', False,
     ['pc02ej01', 'pc02ej02', 'pc02ej03'],
     [('pc02es01', None), ('pc02es02', None)]),
    ('revolvingCreditLoan', False,
     ['pc02fj01', 'pc02fj02', 'pc02fj03'],
     [('pc02fs01', None), ('pc02fs02', None)]),
    ('revolvingLoanAccount', False,
     ['pc02gj01', 'pc02gj02', 'pc02gj03'],
     [('pc02gs01', None), ('pc02gs02', None)]),
    ('loanCardAccount', False,
     ['pc02hj01', 'pc02hj02', 'pc02hj03', 'pc02hj04', 'pc02hj05'],
     [('pc02hs01', None), ('pc02hs02', None)]),
    ('standardLoancardAccount', False,
     ['pc02ij01', 'pc02ij02', 'pc02ij03', 'pc02ij04', 'pc02ij05'],
     [('pc02is01', None), ('pc02is02', None)]),
    ('relatedRepayDutys', True,
     ['pc02kj01', 'pc02kj02'],
     [('pc02kd01', '相关还款责任人类型代码表'),
      ('pc02kd02', '个人 借贷交易相关还款责任类型代码表')]),
    ('postpaySummary', False,
     ['pc030j01'],                                   # 欠费金额
     [('pc030d01', '后付费业务类型代码表')]),
    ('publics', True,
     ['pc040j01'],                                   # 涉及金额
     [('pc040d01', '公共信息类型代码表')]),
    ('querySummary', False,
     ['pc05bs01', 'pc05bs02', 'pc05bs03', 'pc05bs04',
      'pc05bs05', 'pc05bs06', 'pc05bs07', 'pc05bs08'],
     []),                                            # 纯数值
]

# ============================================================
# Account 分支（accountInfos，按 pd01ad01 分桶到 5 类）
# ============================================================

ACCOUNT_CAT_FIELDS = [
    ('pd01ad01', '个人借贷账户类型代码表'),            # 账户类型（已用于分桶，仍保留作 feature）
    ('pd01ad02', '个人借贷交易业务种类代码表'),         # 业务种类
    ('pd01ad03', '个人借贷交易担保方式代码表'),         # 担保方式
    ('pd01ad04', '币种代码表'),                        # 币种
    ('pd01ad06', '个人借贷交易还款频率代码表'),         # 还款频率
    ('pd01ad07', '个人借贷交易担保方式代码表'),         # 第二担保（复用担保码表，简化）
    # 新增 3 个生产高价值 cat 字段
    ('pd01ad08', '个人贷款发放形式代码表'),             # 贷款发放形式（1-新增/5-其他机构转入）
    ('pd01ad09', '个人借贷交易共同借款标志代码表'),     # 共同借款标志（0-无/1-主/2-从）
    ('pd01ad10', '债权转移时的还款状态代码表'),         # 债权转移还款状态（0-7/9）
]

# 这些字段在 accountBasic 中作 numeric（含还款期数 pd01ad05）
# 维度 = 15：原 8 + specialTrades_count + specialTrades_amount + account_age_years
#          + 4 cross-field ratio（使用率/还款进度/逾期比/账龄进度）
ACCOUNT_NUMERIC_FIELDS = [
    'pd01ad05',  # 还款期数（从 cat 移过来）
    'pd01aj01',  # 发放金额
    'pd01aj02',  # 贷款余额
    'pd01aj03',  # 已还款额
    'pd01aj04',  # 剩余还款期数
    'pd01as01',  # 已使用期数
    'pd01bj01',  # 最新余额
    'pd01bj02',  # 最新逾期金额
    # specialTrades 衍生
    '__special_trades_count__',
    '__special_trades_amount__',
    # 账户账龄
    '__account_age_years__',
    # cross-field 衍生 ratio（clamped 0-1，提升风控表征力）
    '__utilization_ratio__',     # pd01bj01 / pd01aj01 (当前余额 / 授信额度)
    '__repayment_progress__',    # pd01aj03 / pd01aj01 (已还 / 发放)
    '__overdue_ratio__',         # pd01bj02 / pd01bj01 (逾期 / 当前余额；bj01=0 时为 0)
    '__tenure_ratio__',          # pd01as01 / pd01ad05 (已用期数 / 总期数)
]

# （旧定义已移到上方）

# 还款状态从这两个字段之一取 60 月
# 优先用 latest5year.latest5yearDetails（60 月）；否则用 latest24PayState.latest24state（24 月，左填充 pad）
PAYSTATE_SOURCE_5Y = 'latest5year.latest5yearDetails'  # list of {pd01er03, pd01ed01, pd01ej01}
PAYSTATE_SOURCE_24 = 'latest24PayState.latest24state'  # 24 字符 string

# ============================================================
# Query 分支（queryRecords）
# ============================================================

QUERY_CAT_FIELDS = [
    ('ph010d01', '机构类型代码'),     # 机构类型
    ('ph010q03', '查询原因代码表'),   # 查询原因
]
QUERY_NUMERIC_FIELDS = [
    'ph010r01_days_ago',  # 查询日期距今（天）
]

# ============================================================
# Public 分支（publicInfo 下 9 子表扁平化）
# ============================================================

# 统一映射成 (type_code, amount, days_ago) 三元组序列
PUBLIC_TYPES = [
    ('pco_pf01', 'taxes'),              # 欠税 PF01
    ('pco_pf02', 'judgments'),          # 民事判决
    ('pco_pf03', 'enforcement'),        # 强制执行
    ('pco_pf04', 'penalties'),          # 行政处罚
    ('pco_pf05', 'low_income_relief'),  # 低保救助
    ('pco_pf06', 'interest_arrears'),   # 欠息
    ('pco_pf07', 'professional_qual'),  # 职业资格
    ('pco_pf08', 'awards'),             # 奖励
]

PUBLIC_TYPE_VOCAB = {name: i for i, (node, name) in enumerate(PUBLIC_TYPES)}
PUBLIC_TYPE_VOCAB_SIZE = len(PUBLIC_TYPE_VOCAB)

# ============================================================
# Obligations 分支：agreementInfos + postpays + relatedRepayDutyInfos 合并
# ============================================================

# 三类 obligation 类型 ID（cat embed 用）
OBLIGATION_TYPES = ['agreement', 'postpay', 'related_repay']
OBLIGATION_TYPE_VOCAB = {name: i for i, name in enumerate(OBLIGATION_TYPES)}
OBLIGATION_TYPE_VOCAB_SIZE = len(OBLIGATION_TYPE_VOCAB)

# 每类 obligation 的 cat 字段（统一 2 列 + type 1 列 = 3 列 cat）
# 字段名用生产实际路径中的字段（pd02*/pe01*/pd03*）
OBLIGATIONS_CAT_FIELDS = [
    # (obligation_type, field_name, code_table_name)
    ('agreement', 'pd02ad01', '个人借贷交易业务种类代码表'),    # 协议业务类型
    ('agreement', 'pd02ad02', '业务大类'),                     # 业务大类
    ('postpay', 'pe01ad01', '后付费业务类型代码表'),           # 后付费业务类型
    ('postpay', 'pe01ad02', '后付费业务状态代码表'),           # 后付费业务状态
    ('related_repay', 'pd03ad01', '相关还款责任人类型代码表'),  # 责任人类型
    ('related_repay', 'pd03ad02', '个人 借贷交易相关还款责任类型代码表'),  # 责任类型
]

# 每类 obligation 的金额字段（numeric 第 0 列）和日期字段（numeric 第 1 列 days_ago）
OBLIGATION_AMOUNT_FIELD = {
    'agreement': 'pd02aj01',
    'postpay': 'pe01aj01',
    'related_repay': None,  # related_repay 通常无金额
}
OBLIGATION_DATE_FIELD = {
    'agreement': None,      # agreement 通常无明确单日期
    'postpay': 'pe01ar01',
    'related_repay': 'pd03ar01',
}

# ============================================================
# JSON 路径取值辅助
# ============================================================

def get_path(obj, path: str, default=None):
    """JSON 路径取值：'personInfo.identity.pb01ad01' 或 'list.0.field'"""
    if path == '':
        return default
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return default
        if part.isdigit() and isinstance(cur, list):
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else default
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
    return cur if cur is not None and cur != '' else default
