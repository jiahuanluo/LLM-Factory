"""配置模块：定义排除字段、special tokens等"""

# 排除字段清单（敏感信息）
EXCLUDED_FIELDS = {
    # 个人身份信息
    "name", "pb01bq01",  # 姓名
    "certNo", "pa01bi01",  # 证件号码
    "pb01aq01",  # 电子邮箱
    "pb01aq02",  # 通讯地址
    "pb01aq03",  # 户籍地址
    "pb01bq01",  # 手机号码

    # 配偶信息
    "pb020q01",  # 配偶姓名
    "pb020i01",  # 配偶证件号码
    "pb020q02",  # 配偶工作单位
    "pb020q03",  # 配偶联系电话

    # 居住地址详情
    "pb030q01",  # 居住地址
    "pb030q02",  # 住宅电话

    # 职业地址详情
    "pb040q01",  # 工作单位（名称）
    "pb040q02",  # 单位地址
    "pb040q03",  # 单位电话

    # 账户关联方信息
    "pd03aq01",  # 担保人名称
    "pd03aq02",  # 担保人证件号

    # 公共信息中的机构名称
    "pf01aq01",  # 税务机关名称
    "pf02aq01",  # 法院名称
    "pf03aq01",  # 法院名称
    "pf04aq01",  # 处罚机关名称
}

# Special Tokens
SPECIAL_TOKENS = {
    "CLS": "[CLS]",
    "SEP": "[SEP]",
    "PAD": "[PAD]",
    "HDR": "[HDR]",
    "PERS": "[PERS]",
    "ACCT": "[ACCT]",
    "SUMM": "[SUMM]",
    "PUB": "[PUB]",
    "QUERY": "[QUERY]",
}

# 报告模块映射
SECTION_MAPPING = {
    "header": "HDR",
    "personInfo": "PERS",
    "summaryInfo": "SUMM",
    "accountInfos": "ACCT",
    "publicInfo": "PUB",
    "queryRecords": "QUERY",
}
