"""Mock 征信报告校准器：把模板化 mock 校准到真实生产取值范围。

输入：data/pbc/cris_json_split/*.json（598 份近退化 mock，3 个手写模板的副本）
输出：--dst 目录（默认 cris_json_split_calibrated/），离群文件（空 stub / 异构信封 /
      残缺报告）跳过并记日志。

规则唯一来源：data/pbc/MOCK校正说明文档.md（下称 spec），逐节实现：
  §3.1 报告信封     §3.2 personInfo     §3.3 score
  §3.4 accountInfos §3.5 queryRecords   §3.6 publicInfo
  §3.7 summaryInfo（从生成数据重算）    §3.8 obligation 数量重采样  §3.9 otherMarks

设计要点：
  - 确定性：rng = random.Random(文件名)，同输入同输出（spec 3.0.4）。
  - 账户全字段按骨架模板重建（不依赖原文件节点组合），稀疏节点
    （specialTrades/specialEvents/largeSpecialInstalments）从原文件池低概率继承。
  - 写盘前逐报告自检 16 项业务不变量（调用 verify_mock_calibration.check_report），
    有违反即报错退出，保证 spec §4 全部满足。

用法：
  python scripts/calibrate_mock_reports.py --src <mock目录> --dst <输出目录> [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_mock_calibration import BAD_ACTIVE, check_report  # noqa: E402

# ============ 常量（spec §3 实测频率 / 分位锚点） ============

LOAN_TYPES, CARD_TYPES = ('D1', 'R4', 'C1'), ('R1', 'R2', 'R3')

TYPE_FREQ = {'D1': 487, 'R2': 229, 'R1': 146, 'R4': 134, 'R3': 15, 'C1': 10}
SUBTYPE_FREQ = {'11': 564924, '24': 64003, '51': 40896, '21': 13825, '23': 4846,
                '12': 4538, '22': 2361, '15': 2310, '16': 1082}
BUSINESS_FREQ = {'91': 237419, '41': 216317, '81': 152873, '99': 43683, '21': 10235,
                 '11': 9946, '51': 9415, '82': 7350, '52': 4362, '92': 2578, '12': 1575,
                 '13': 555, '53': 375, 'B1': 232, '62': 68, '42': 61, '32': 43, 'A1': 16}
CURRENCIES = {'CNY': 9900, 'USD': 25, 'EUR': 25, 'JPY': 25, 'HKD': 25}

AMOUNT_KNOTS = {
    'D1': [(0.05, 1086), (0.25, 10000), (0.50, 40000), (0.75, 173000), (0.95, 1000000), (0.99, 3000000), (1.0, 39000000)],
    'R4': [(0.05, 3200), (0.25, 25000), (0.50, 100000), (0.75, 300000), (0.95, 1200000), (0.99, 4000000), (1.0, 30000000)],
    'C1': [(0.05, 453), (0.25, 3428), (0.50, 19738), (0.75, 70477), (0.95, 233227), (0.99, 405795), (1.0, 2271505)],
    'R1': [(0.05, 100), (0.25, 8000), (0.50, 39239), (0.75, 161000), (0.95, 894500), (0.99, 2600000), (1.0, 25000000)],
    'R2': [(0.05, 1964), (0.25, 8000), (0.50, 20000), (0.75, 48000), (0.95, 102379), (0.99, 231080), (1.0, 17500000)],
    'R3': [(0.05, 50), (0.25, 2000), (0.50, 5000), (0.75, 5000), (0.95, 20000), (0.99, 50000), (1.0, 200000)],
}
BALANCE_KNOTS = {
    'D1': [(0.25, 15000), (0.50, 100000), (0.95, 2331851), (1.0, 39000000)],
    'R4': [(0.25, 55000), (0.50, 200000), (0.95, 2200000), (1.0, 16000000)],
    'C1': [(0.25, 3324), (0.50, 19117), (0.95, 240715), (1.0, 2271505)],
    'R1': [(0.25, 0), (0.50, 200), (0.95, 500000), (1.0, 19107000)],
    'R2': [(0.25, 0), (0.50, 0), (0.95, 64252), (1.0, 1836233)],
    'R3': [(0.25, 0), (0.50, 0), (0.95, 20000), (1.0, 100000)],
}
OVERDUE_KNOTS = [(0.25, 3978), (0.50, 15498), (0.95, 308935), (0.99, 1315394), (1.0, 17901989)]

ACTIVE_RATE = 0.23
ACTIVE_CD01_FREQ = {'1': 70, '5': 8, '2': 6, '3': 5, '4': 4, '6': 4, '31': 2, '8': 1}
CLOSED_BD01_FREQ = {'3': 40, '4': 20, '1': 12, '2': 10, '5': 7, '6': 5, '7': 4, '8': 2}
FIVE_LEVEL_FREQ = {'3': 2, '4': 2, '5': 1}
AGE_MEDIAN_MONTHS = {'D1': 77, 'R2': 139, 'R1': 21, 'R4': 6, 'C1': 36, 'R3': 214}

ACCOUNT_COUNT_KNOTS = [(0.0, 0), (0.003, 0), (0.25, 22), (0.50, 42), (0.75, 75),
                       (0.95, 172), (0.99, 304), (1.0, 1653)]
QUERY_COUNT_KNOTS = [(0.0, 0), (0.25, 9), (0.50, 18), (0.75, 34), (0.95, 88), (1.0, 416)]
ACCOUNT_CAP, QUERY_CAP = 200, 300

GENDER_FREQ = {'1': 9781, '2': 1922}
EDU_FREQ = {'30': 3351, '20': 2978, '60': 1748, '90': 1016, '40': 963, '91': 807, '10': 404, '99': 390}
DEGREE_FREQ = {'9': 6663, '5': 3809, '4': 966, '3': 124, '0': 50, '2': 21, '1': 6}
EMPLOY_FREQ = {'91': 5622, '21': 1595, '17': 1494, '90': 943, '54': 767, '13': 412, '11': 230}
OCCUP_FREQ = {'91': 3000, '24': 1500, '21': 1200, '17': 1000, '29': 900, '13': 800, '11': 600, '54': 500}
MARRIAGE_FREQ = {'20': 18921, '10': 18977, '40': 699, '91': 150}
RESIDENCE_FREQ = {'9': 5293, '1': 2819, '7': 1722, '2': 769, '11': 332, '3': 276, '5': 381}
SCORE_FACTORS = ['00', '01', '03', '05', '06', '15', '16']

QUERY_REASON_FREQ = {'02': 211430, '08': 102496, '03': 12409, '20': 2103, '24': 2068}
QUERY_ORG_FREQ = {'11': 193527, '53': 47154, '51': 40412, '24': 36267, '22': 4619,
                  '41': 2101, '15': 2007, '12': 1701, '23': 1389}

REPORT_MONTH_RANGE = ((2020, 2), (2026, 8))   # 真实跨度
RECENT_BOOST = 30.0                            # 2026-07/08 假定计数（真实 97.6% 偏态），取 sqrt 弱化

PUBLIC_FILL_RATE = {'accFunds': 0.030, 'civilJudgements': 0.0001, 'forceExecutions': 0.0018,
                    'taxarrears': 0.0005, 'adminPunishments': 0.0005, 'salvations': 0.005,
                    'competences': 0.005, 'adminAwards': 0.005}

SURNAMES = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严金魏陶姜'
GIVEN_CHARS = '伟芳娜秀英敏静丽强磊军洋勇艳杰娟涛明超霞平刚桂华建文军鑫宇欣怡宁萌泽'
AREA_CODES = ['110101', '310104', '440305', '330106', '510107', '420106', '610113', '320105']
COMPANIES = ['三星电子北京分公司', '中科软科技股份有限公司', '中国平安人寿保险股份公司',
             '东方国信科技有限公司', '华润万家超市连锁', '顺丰速运集团', '新东方教育科技集团',
             '国家电网北京市电力公司']
EMAIL_DOMAINS = ['qq.com', '163.com', '126.com', 'hotmail.com', 'gmail.com']
CITIES_ADDRS = ['北京市朝阳区春晓园北区7号楼C{room}室', '北京市西城区金融大街35号国际企业大厦A座{room}室',
                '上海市浦东新区张江路625号{room}室', '广州市天河区体育西路57号{room}室',
                '深圳市南山区科技园科苑路15号{room}室', '杭州市西湖区文三路90号{room}室']

# 字段骨架（键集来自真实 mock 模板，值全部重写）
BASIC_KEYS = ['pd01ai01', 'pd01ad01', 'pd01ad02', 'pd01ai02', 'pd01ai03', 'pd01ai04',
              'pd01ad03', 'pd01ar01', 'pd01ad04', 'pd01aj01', 'pd01aj02', 'pd01aj03',
              'pd01ar02', 'pd01ad05', 'pd01ad06', 'pd01as01', 'pd01ad07', 'pd01ad08',
              'pd01ad09', 'pd01ad10']
LATEST_KEYS = ['pd01bd01', 'pd01br01', 'pd01br04', 'pd01bj01', 'pd01br02', 'pd01bj02',
               'pd01bd03', 'pd01bd04', 'pd01br03']
MPS_KEYS = ['pd01cr01', 'pd01cd01', 'pd01cj01', 'pd01cj02', 'pd01cj03', 'pd01cd02',
            'pd01cs01', 'pd01cr02', 'pd01cj04', 'pd01cj05', 'pd01cr03', 'pd01cs02',
            'pd01cj06', 'pd01cj07', 'pd01cj08', 'pd01cj09', 'pd01cj10', 'pd01cj11',
            'pd01cj12', 'pd01cj13', 'pd01cj14', 'pd01cj15', 'pd01cr04']

DIGIT_CHARS = '1112223334455667'   # 逾期字符池（低段概率高）


def M(**fields) -> dict:
    """带冗余元字段的节点骨架（值由 refresh_meta 统一刷新）。"""
    return {'tranDate': '', 'createTime': '', 'seq': '', 'reportsn': '', **fields}


# ============ 工具函数 ============

def sample_freq(rng: random.Random, freq: dict) -> str:
    ks, ws = list(freq.keys()), list(freq.values())
    return ks[rng.choices(range(len(ks)), weights=ws, k=1)[0]]


def inv_cdf(rng: random.Random, knots: list, tail_prob: float = 0.01) -> int:
    """分位锚点逆 CDF：1% 落入最后一段 (p次末, max] 均匀展开保长尾，其余相邻锚点线性插值。"""
    if rng.random() < tail_prob:
        u = 1.0 - rng.random() * (1.0 - knots[-2][0])
    else:
        u = rng.random()
    if u <= knots[0][0]:
        return int(knots[0][1])
    for (q0, v0), (q1, v1) in zip(knots, knots[1:]):
        if u <= q1:
            if q1 <= q0:
                return int(v0)
            return int(round(v0 + (v1 - v0) * (u - q0) / (q1 - q0)))
    return int(knots[-1][1])


def round_amount(x: int) -> int:
    """金额精度：>10万取整千，>1千取整百（spec 3.4）。"""
    if x > 100_000:
        return max(1000, round(x / 1000) * 1000)
    if x > 1000:
        return max(100, round(x / 100) * 100)
    return max(1, x)


def ym_add(ym_t: tuple, k: int) -> tuple:
    idx = ym_t[0] * 12 + ym_t[1] - 1 + k
    return idx // 12, idx % 12 + 1


def ym_str(ym_t: tuple) -> str:
    return f'{ym_t[0]:04d}-{ym_t[1]:02d}'


def rand_before(rng: random.Random, ref: date, max_days: int) -> date:
    return ref - timedelta(days=rng.randint(1, max(1, max_days)))


def make_cert_no(rng: random.Random, dob: date, male: bool) -> str:
    """18 位身份证：6 地区 + 8 生日 + 3 顺序（第 17 位奇男偶女）+ 校验位（spec 3.2）。"""
    seq_digit = rng.randrange(1, 10)
    third = rng.randrange(0, 10)
    gender_digit = rng.choice([1, 3, 5, 7, 9]) if male else rng.choice([0, 2, 4, 6, 8])
    body = (rng.choice(AREA_CODES) + dob.strftime('%Y%m%d')
            + f'{seq_digit}{third}{gender_digit}')
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    s = sum(int(c) * w for c, w in zip(body, weights))
    return body + '10X98765432'[s % 11]


def rand_name(rng: random.Random, prefix_mock: bool = True) -> str:
    n = rng.choice(SURNAMES) + ''.join(rng.choice(GIVEN_CHARS) for _ in range(rng.randint(1, 2)))
    return ('测' + n) if prefix_mock else n


def rand_addr(rng: random.Random) -> str:
    room = f"{rng.randint(1, 30)}{rng.choice('ABCDE')}{rng.randint(101, 3200)}"
    return rng.choice(CITIES_ADDRS).format(room=room) + str(rng.randint(0, 99))


def jitter_amount(rng: random.Random, v, lo: float = 0.5, hi: float = 1.5) -> str:
    x = str(v).strip() if v is not None else ''
    if not x.lstrip('-').isdigit():
        return str(v) if v is not None else ''
    return str(max(1, round(int(x) * rng.uniform(lo, hi))))


def jitter_node_amounts(rng: random.Random, node: dict) -> None:
    """递归抖动节点内全部金额字段（j 结尾的 q/f 字段：pd0XjYY / pd01fj01 等）。"""
    for k, v in node.items():
        if isinstance(v, str) and v.isdigit() and ('aj' in k or 'bj' in k or 'cj' in k
                                                   or 'j0' in k or k.endswith('j01') or k.endswith('j02')):
            node[k] = jitter_amount(rng, v)
        elif isinstance(v, dict):
            jitter_node_amounts(rng, v)


# ============ §3.1 报告信封 ============

def sample_report_time(rng: random.Random) -> date:
    (y0, m0), (y1, m1) = REPORT_MONTH_RANGE
    months, weights = [], []
    ym_t = (y0, m0)
    while ym_t <= (y1, m1):
        months.append(ym_t)
        weights.append(RECENT_BOOST ** 0.5 if ym_t >= (2026, 7) else 1.0)
        ym_t = ym_add(ym_t, 1)
    y, m = rng.choices(months, weights=weights, k=1)[0]
    return date(y, m, rng.randint(1, 28))


# ============ §3.2 personInfo ============

def calibrate_person(rng: random.Random, report: dict, rt: date) -> None:
    person = report.get('personInfo') or {}
    ident = person.setdefault('identity', {})
    gender = sample_freq(rng, GENDER_FREQ)
    male = gender == '1'
    age_years = rng.randint(25, 65)
    dob = date(rt.year - age_years, rt.month, min(rt.day, 28)) - timedelta(days=rng.randint(0, 340))
    if (rt - dob).days // 365 > 65:
        dob = date(rt.year - 65, rt.month, min(rt.day, 28))

    ident.update({
        'pb01ad01': gender, 'pb01ar01': dob.isoformat(),
        'pb01ad02': sample_freq(rng, EDU_FREQ), 'pb01ad03': sample_freq(rng, DEGREE_FREQ),
        'pb01ad04': sample_freq(rng, EMPLOY_FREQ), 'pb01ad05': 'CHN',
        'pb01aq01': f'user{rng.randint(1000, 999999)}@{rng.choice(EMAIL_DOMAINS)}',
        'pb01aq02': rand_addr(rng), 'pb01aq03': rand_addr(rng),
    })
    # 手机号 1-5 条（13x + 8 位，共 11 位）
    ident['mobiles'] = [M(
        pb01bs01=str(rng.randint(1, 6)),
        pb01bq01=f"1{rng.choice('3578')}{rng.randint(0, 10**8 - 1):08d}",
        pb01br01=rand_before(rng, rt, 3600).isoformat(),
    ) for _ in range(rng.randint(1, 5))]
    person['residences'] = [M(
        pb030d01=sample_freq(rng, RESIDENCE_FREQ), pb030q01=rand_addr(rng),
        pb030q02=f"010-{rng.randint(60000000, 89999999)}",
        pb030r01=rand_before(rng, rt, 3600).isoformat(),
    ) for _ in range(rng.randint(1, 5))]
    person['professionals'] = [M(
        pb040d01=sample_freq(rng, OCCUP_FREQ), pb040q01=rng.choice(COMPANIES),
        pb040d02=rng.choice(['10', '20', '30', '60']), pb040d03=rng.choice('JKLMNOPQ'),
        pb040q02=rand_addr(rng), pb040q03=f"010-{rng.randint(60000000, 89999999)}",
        pb040d04=str(rng.randint(1, 4)), pb040d05=str(rng.randint(1, 3)),
        pb040d06=str(rng.randint(1, 3)), pb040r01=str(rng.randint(1990, rt.year)),
        pb040r02=rand_before(rng, rt, 3600).isoformat(),
    ) for _ in range(rng.randint(1, 5))]

    marriage = person.setdefault('marriage', {})
    marriage['pb020d01'] = sample_freq(rng, MARRIAGE_FREQ)
    if marriage['pb020d01'] == '10':   # 未婚 → 配偶置空
        marriage.update({'pb020q01': '', 'pb020d02': '', 'pb020i01': '', 'pb020q02': '', 'pb020q03': ''})
    else:
        marriage.update({
            'pb020q01': rand_name(rng, prefix_mock=False),
            'pb020d02': rng.choice(['1', '2', '8', '10']),
            'pb020i01': f'{rng.randint(10**14, 10**15 - 1)}',
            'pb020q02': rng.choice(COMPANIES),
            'pb020q03': f"1{rng.choice('3578')}{rng.randint(0, 10**8 - 1):08d}",
        })

    # 证件：顶层 + header.request 同步，出生年/性别与 identity 一致
    cert = make_cert_no(rng, dob, male)
    report['certNo'] = cert
    report['name'] = rand_name(rng)
    request = ((report.get('header') or {}).get('request') or {})
    request['pa01bi01'] = cert
    request['pa01bq01'] = report['name']


# ============ §3.3 score ============

def calibrate_score(rng: random.Random, report: dict) -> None:
    factors = rng.sample(SCORE_FACTORS, 2)
    report['score'] = M(
        pc010q01=str(rng.randint(600, 950)),
        pc010q02=str(rng.randint(1, 99)),
        pc010s01='2',
        pc010d01=','.join(sorted(factors)),
    )


# ============ §3.4 accountInfos ============

def paystate_char(rng: random.Random, lifecycle: str, last: bool = False) -> str:
    """按生命周期采还款状态字符（spec 3.4 字符频率语义）。"""
    if lifecycle == 'normal':
        table = [('N', 80), ('*', 15), ('#', 3), ('1', 2)]
    elif lifecycle == 'bad':
        table = [('C', 15), ('N', 15), ('#', 15), ('B', 12), ('*', 10), ('/', 10),
                 ('G', 8), ('D', 5), ('1', 5), ('2', 2), ('3', 1)]
    elif lifecycle == 'settled':
        if last:
            return 'C'
        table = [('N', 75), ('C', 15), ('*', 10)]
    else:   # overdue
        table = [('N', 40), ('1', 15), ('*', 15), ('2', 10), ('#', 10), ('3', 5),
                 ('4', 3), ('5', 2)]
    chars, ws = zip(*table)
    return rng.choices(chars, weights=ws, k=1)[0]


def make_latest24(rng: random.Random, lifecycle: str, rt_ym: tuple, age_months: int) -> dict | None:
    if age_months < 24 or rng.random() > 0.6:
        return None
    end_back = rng.randint(0, min(12, age_months - 24))
    end_ym = ym_add(rt_ym, -end_back)
    start_ym = ym_add(end_ym, -23)
    chars = ''.join(paystate_char(rng, lifecycle, last=(i == 23)) for i in range(24))
    return M(pd01dr01=ym_str(start_ym), pd01dr02=ym_str(end_ym), latest24state=chars)


def make_latest5year(rng: random.Random, lifecycle: str, rt_ym: tuple, age_months: int) -> dict | None:
    if rng.random() > 0.8:
        return None
    med = {'normal': 59, 'settled': 1, 'bad': 26}.get(lifecycle, 57)
    rows = int(round(med * rng.uniform(0.3, 1.4)))
    rows = max(1, min(rows, 60, age_months + 1))
    newest_back = rng.randint(0, min(3, age_months))
    details = []
    for i in range(rows):
        ch = paystate_char(rng, lifecycle, last=(i == 0))
        overdue = '' if ch in 'N*#/CG' else str(round_amount(inv_cdf(rng, OVERDUE_KNOTS)))
        details.append(M(seq1='', pd01er03=ym_str(ym_add(rt_ym, -(newest_back + i))),
                         pd01ed01=ch, pd01ej01=overdue if overdue else '0'))
    return M(pd01er01=details[-1]['pd01er03'], pd01er02=details[0]['pd01er03'],
             pd01es01=str(rows), latest5yearDetails=details)


def calibrate_accounts(rng: random.Random, report: dict, rt: date, rare_pool: list) -> list:
    n = inv_cdf(rng, ACCOUNT_COUNT_KNOTS)
    n = min(n, ACCOUNT_CAP)
    rt_ym = (rt.year, rt.month)
    accounts = []
    for _ in range(n):
        atype = sample_freq(rng, TYPE_FREQ)
        is_loan = atype in LOAN_TYPES
        active = rng.random() < ACTIVE_RATE

        basic = M(**dict.fromkeys(BASIC_KEYS, ''))
        basic.update({
            'pd01ai01': f'{rng.randint(10000, 99999)}',
            'pd01ad01': atype,
            'pd01ad02': sample_freq(rng, SUBTYPE_FREQ),
            'pd01ai02': f'{rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}{rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}',
            'pd01ad03': sample_freq(rng, BUSINESS_FREQ),
            'pd01ad04': sample_freq(rng, CURRENCIES),
        })
        age_months = max(1, min(480, round(AGE_MEDIAN_MONTHS[atype] * rng.uniform(0.15, 1.85))))
        open_d = ym_add((rt.year, rt.month), -age_months)
        basic['pd01ar01'] = f"{open_d[0]:04d}-{open_d[1]:02d}-{rng.randint(1, 28):02d}"

        if is_loan:
            basic['pd01aj01'] = str(round_amount(inv_cdf(rng, AMOUNT_KNOTS[atype])))
            basic['pd01ad06'] = rng.choice(['01', '02', '03'])
            basic['pd01ad07'] = rng.choice(['1', '2', '4'])
            basic['pd01ad08'] = rng.choice(['1', '2'])
            basic['pd01ad09'] = rng.choice(['1', '2'])
        else:
            basic['pd01aj02'] = str(round_amount(inv_cdf(rng, AMOUNT_KNOTS[atype])))
            basic['pd01ad07'] = rng.choice(['1', '2', '4'])

        latest = M(**dict.fromkeys(LATEST_KEYS, ''))
        mps = M(**dict.fromkeys(MPS_KEYS, '')) if active else None
        if active:
            cd01 = sample_freq(rng, ACTIVE_CD01_FREQ)
            mps['pd01cd01'] = cd01
            mps['pd01cr01'] = ym_str(rt_ym)
            br03 = rand_before(rng, rt, 60)
            latest['pd01br03'] = latest['pd01br02'] = br03.isoformat()
            mps['pd01cr02'] = br03.isoformat()
            mps['pd01cr03'] = (br03 - timedelta(days=rng.randint(0, 5))).isoformat()
            mps['pd01cr04'] = br03.isoformat()
            bad = cd01 in BAD_ACTIVE
            if not is_loan:  # 活跃卡：余额=已用 ≤ 额度×1.1（超限值在 [0.85,1.1] 展开，避免质量点）
                limit = int(basic['pd01aj02'])
                used = round_amount(inv_cdf(rng, BALANCE_KNOTS[atype]))
                if used > limit * 1.1:
                    used = round(limit * rng.uniform(0.85, 1.1))
                latest['pd01bj01'] = mps['pd01cj01'] = mps['pd01cj02'] = str(used)
                mps['pd01cj03'] = str(max(0, limit - used))
                mps['pd01cj12'] = basic['pd01aj02']
                mps['pd01cj14'] = str(used)
                mps['pd01cs01'] = ''
            else:
                bal = round_amount(inv_cdf(rng, BALANCE_KNOTS[atype]))
                latest['pd01bj01'] = mps['pd01cj01'] = str(bal)
                remain = max(1, round(age_months * rng.uniform(0.2, 1.5)))
                mps['pd01cs01'] = str(remain)
                basic['pd01as01'] = str(age_months + remain)
                end_ym = ym_add(rt_ym, remain)
                basic['pd01ar02'] = f"{end_ym[0]:04d}-{end_ym[1]:02d}-{rng.randint(1, 28):02d}"
            if bad:
                od = str(round_amount(inv_cdf(rng, OVERDUE_KNOTS)))
                latest['pd01bj02'] = od
                mps['pd01cj06'] = od
                mps['pd01cs02'] = str(rng.randint(1, 6))
                latest['pd01bd03'] = sample_freq(rng, FIVE_LEVEL_FREQ) if cd01 == '5' else ''
            else:
                mps['pd01cj06'] = '0'
            lifecycle = 'bad' if cd01 == '5' else ('overdue' if bad else 'normal')
        else:
            bd01 = sample_freq(rng, CLOSED_BD01_FREQ)
            if atype in ('R2', 'R3') and bd01 == '3':
                bd01 = '4'   # 信用卡无结清态（spec 4.6）
            latest['pd01bd01'] = bd01
            close_back = rng.randint(0, age_months)
            close_ym = ym_add(rt_ym, -close_back)
            close_d = f"{close_ym[0]:04d}-{close_ym[1]:02d}-{rng.randint(1, 28):02d}"
            if close_d < basic['pd01ar01']:
                close_d = basic['pd01ar01']
            if close_d > rt.isoformat():   # close_back=0 时可能同月晚于报告日
                close_d = rt.isoformat()
            latest['pd01br01'] = close_d
            latest['pd01br03'] = latest['pd01br02'] = close_d
            bad = bd01 in ('2', '4')
            if is_loan:
                basic['pd01ar02'] = close_d
                span = (int(close_d[:4]) * 12 + int(close_d[5:7])) - (int(basic['pd01ar01'][:4]) * 12 + int(basic['pd01ar01'][5:7]))
                basic['pd01as01'] = str(max(1, span + rng.randint(0, 48)))
            if bad:
                latest['pd01bj01'] = str(round_amount(inv_cdf(rng, BALANCE_KNOTS[atype])))
                latest['pd01bj02'] = str(round_amount(inv_cdf(rng, OVERDUE_KNOTS)))
                if bd01 == '4':
                    latest['pd01bd03'] = sample_freq(rng, FIVE_LEVEL_FREQ)
            else:
                latest['pd01bj01'] = '0'
                latest['pd01bj02'] = ''
            lifecycle = {'3': 'settled', '4': 'bad', '2': 'overdue'}.get(bd01, 'normal')

        acc = {'seq': '', 'reportsn': '', 'accountBasic': basic, 'latestInfo': latest}
        if mps is not None:
            acc['latestMonthPayState'] = mps
        p24 = make_latest24(rng, lifecycle, rt_ym, age_months)
        if p24:
            acc['latest24PayState'] = p24
        p5y = make_latest5year(rng, lifecycle, rt_ym, age_months)
        if p5y:
            acc['latest5year'] = p5y
        if rng.random() < 0.05:
            acc['specialTrades'] = [M(
                pd01fs01='1', pd01fd01=rng.choice(['1', '2', '3']),
                pd01fr01=rand_before(rng, rt, 1800).isoformat(), pd01fs02='0',
                pd01fj01=jitter_amount(rng, rng.choice(['1000', '5000', '10000', '50000'])),
                pd01fq01='该贷款由XX公司代偿10,000元',
            )]
        for tmpl in rare_pool:   # 稀疏节点从原文件模板低概率继承
            if rng.random() < 0.02:
                node = copy.deepcopy(rng.choice(tmpl[1]))
                jitter_node_amounts(rng, node[0] if isinstance(node, list) else node)
                acc[tmpl[0]] = node
        accounts.append(acc)
    return accounts


# ============ §3.5 queryRecords ============

def calibrate_queries(rng: random.Random, rt: date) -> list:
    n = min(inv_cdf(rng, QUERY_COUNT_KNOTS), QUERY_CAP)
    return [M(
        ph010r01=rand_before(rng, rt, 730).isoformat(),
        ph010d01=sample_freq(rng, QUERY_ORG_FREQ),
        ph010q02=f'{rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}{rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}',
        ph010q03=sample_freq(rng, QUERY_REASON_FREQ),
    ) for _ in range(n)]


# ============ §3.6 publicInfo / §3.9 otherMarks ============

def calibrate_public(rng: random.Random, report: dict, rt: date) -> None:
    pub = report.get('publicInfo') or {}
    for key, rate in PUBLIC_FILL_RATE.items():
        entries = pub.get(key) or []
        if entries and rng.random() < rate:
            for e in entries:
                jitter_node_amounts(rng, e)
                for dk in list(e):
                    if dk.endswith('r01') or dk.endswith('r02') or dk.endswith('r03'):
                        e[dk] = rand_before(rng, rt, 3600).isoformat()
            pub[key] = entries
        else:
            pub[key] = []
    report['publicInfo'] = pub
    report['otherMarks'] = []


# ============ §3.8 agreementInfos / postpays / relatedRepayDutyInfos ============

def resample_list(rng: random.Random, template: list, n: int, rt: date) -> list:
    if not template or n <= 0:
        return []
    out = []
    for i in range(n):
        e = copy.deepcopy(template[i % len(template)])
        jitter_node_amounts(rng, e)
        for dk in list(e):
            if dk.endswith(('r01', 'r02', 'r03')):
                e[dk] = rand_before(rng, rt, 3600).isoformat()
        out.append(e)
    return out


def calibrate_obligations(rng: random.Random, report: dict, rt: date) -> None:
    report['agreementInfos'] = resample_list(rng, report.get('agreementInfos') or [],
                                             min(30, max(0, round(9 * rng.uniform(0.3, 1.7)))), rt)
    report['postpays'] = resample_list(rng, report.get('postpays') or [],
                                       0 if rng.random() < 0.5 else rng.randint(1, 3), rt)
    report['relatedRepayDutyInfos'] = resample_list(rng, report.get('relatedRepayDutyInfos') or [],
                                                    min(15, max(0, round(4 * rng.uniform(0.2, 1.8)))), rt)


# ============ §3.7 summaryInfo 重算 ============

def recompute_summary(report: dict, rt: date) -> None:
    accounts = report.get('accountInfos') or []
    queries = report.get('queryRecords') or []
    by_type = defaultdict(list)
    for a in accounts:
        by_type[str((a.get('accountBasic') or {}).get('pd01ad01') or '')].append(a)

    def org(a):
        return str((a.get('accountBasic') or {}).get('pd01ai02') or '')

    def to_int(a, path, default=0):
        node = a
        for k in path:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        v = str(node).strip() if node is not None else ''
        return int(v) if v.isdigit() else default

    def loan_seg(prefix, accs):
        seg = {f'pc02{prefix}{s}': '0' for s in ('s01', 's02', 'j01', 'j02', 'j03')}
        if not accs:
            return {**M(), **seg}
        j01 = sum(to_int(a, ('accountBasic', 'pd01aj01')) for a in accs)
        j02 = sum(to_int(a, ('latestInfo', 'pd01bj01')) for a in accs)
        seg.update({
            f'pc02{prefix}s01': str(len({org(a) for a in accs})),
            f'pc02{prefix}s02': str(len(accs)),
            f'pc02{prefix}j01': str(j01), f'pc02{prefix}j02': str(j02),
            f'pc02{prefix}j03': str(round(j02 * 0.05)),
        })
        return {**M(), **seg}

    def card_seg(prefix, accs):
        base = {f'pc02{prefix}{s}': '0' for s in ('s01', 's02', 'j01', 'j02', 'j03', 'j04', 'j05')}
        if not accs:
            return base
        per_org = defaultdict(int)
        for a in accs:
            per_org[org(a)] += to_int(a, ('accountBasic', 'pd01aj02'))
        used = 0
        for a in accs:
            u = to_int(a, ('latestMonthPayState', 'pd01cj02'))
            if not u:
                u = to_int(a, ('latestInfo', 'pd01bj01')) if str((a.get('latestInfo') or {}).get('pd01bd01') or '') in ('2', '4') else 0
            used += u
        base.update({
            f'pc02{prefix}s01': str(len(per_org)), f'pc02{prefix}s02': str(len(accs)),
            f'pc02{prefix}j01': str(sum(per_org.values())),
            f'pc02{prefix}j02': str(max(per_org.values())),
            f'pc02{prefix}j03': str(min(per_org.values())),
            f'pc02{prefix}j04': str(used), f'pc02{prefix}j05': str(round(used * 0.5)),
        })
        return {**M(), **base}

    s = report.get('summaryInfo') or {}
    s['nonrevolvingLoan'] = loan_seg('e', by_type.get('D1', []))
    s['revolvingCreditLoan'] = loan_seg('f', by_type.get('R1', []))
    s['revolvingLoanAccount'] = loan_seg('g', by_type.get('R4', []))
    s['loanCardAccount'] = card_seg('h', by_type.get('R2', []))
    s['standardLoancardAccount'] = card_seg('i', by_type.get('R3', []))

    # 逾期/呆账账户按业务种类分组
    def is_overdue_acc(a):
        bd01 = str((a.get('latestInfo') or {}).get('pd01bd01') or '')
        cd01 = str((a.get('latestMonthPayState') or {}).get('pd01cd01') or '')
        return bd01 in ('2', '4') or cd01 in BAD_ACTIVE

    groups = defaultdict(list)
    for a in accounts:
        if is_overdue_acc(a):
            groups[str((a.get('accountBasic') or {}).get('pd01ad03') or '99')].append(a)

    def max_consecutive_digits(chars: str) -> int:
        best = cur = 0
        for c in chars:
            cur = cur + 1 if c in '1234567' else 0
            best = max(best, cur)
        return best

    s['overdues'] = []
    for bus, accs in sorted(groups.items()):
        months, longest, peak = set(), 0, 0
        for a in accs:
            five = ((a.get('latest5year') or {}).get('latest5yearDetails')) or []
            for row in five:
                ch, m = str(row.get('pd01ed01') or ''), str(row.get('pd01er03') or '')
                amt = str(row.get('pd01ej01') or '0')
                if ch in '1234567BDG' or (amt.isdigit() and int(amt) > 0):
                    months.add(m)
                    peak = max(peak, int(amt) if amt.isdigit() else 0)
            p24 = str(((a.get('latest24PayState') or {}).get('latest24state')) or '')
            longest = max(longest, max_consecutive_digits(p24))
            bj02 = to_int(a, ('latestInfo', 'pd01bj02'))
            peak = max(peak, bj02)
        s['overdues'].append(M(
            pc02ds01=bus, pc02dd01='1', pc02ds02=str(len(accs)),
            pc02ds03=str(min(60, len(months))), pc02dj01=str(peak),
            pc02ds04=str(max(1, longest)),
        ))

    # 查询汇总：取最近一条 + 时间窗统计
    def within_days(q, days):
        d = str(q.get('ph010r01') or '')
        try:
            return (rt - date.fromisoformat(d)).days < days
        except ValueError:
            return False

    if queries:
        latest_q = max(queries, key=lambda q: str(q.get('ph010r01') or ''))
        s['querySummary'] = M(
            pc05ar01=str(latest_q.get('ph010r01') or ''), pc05ad01=str(latest_q.get('ph010d01') or ''),
            pc05ai01=str(latest_q.get('ph010q02') or ''), pc05aq01=str(latest_q.get('ph010q03') or ''),
            pc05bs01=str(len({str(q.get('ph010q02')) for q in queries if q.get('ph010q03') == '02' and within_days(q, 31)})),
            pc05bs02=str(len({str(q.get('ph010q02')) for q in queries if q.get('ph010q03') == '03' and within_days(q, 31)})),
            pc05bs03=str(sum(1 for q in queries if q.get('ph010q03') == '02' and within_days(q, 31))),
            pc05bs04=str(sum(1 for q in queries if q.get('ph010q03') == '03' and within_days(q, 31))),
            pc05bs05='0', pc05bs06='0',
            pc05bs07=str(sum(1 for q in queries if q.get('ph010q03') == '08' and within_days(q, 731))),
            pc05bs08=str(sum(1 for q in queries if q.get('ph010q03') == '20' and within_days(q, 731))),
        )
    else:
        s['querySummary'] = M(**{k: ('' if k in ('pc05ar01', 'pc05ad01', 'pc05ai01', 'pc05aq01') else '0')
                                 for k in ('pc05ar01', 'pc05ad01', 'pc05ai01', 'pc05aq01',
                                           'pc05bs01', 'pc05bs02', 'pc05bs03', 'pc05bs04',
                                           'pc05bs05', 'pc05bs06', 'pc05bs07', 'pc05bs08')})

    bad_accs = [a for a in accounts
                if str((a.get('latestInfo') or {}).get('pd01bd01') or '') == '4'
                or str((a.get('latestMonthPayState') or {}).get('pd01cd01') or '') == '5']
    s['badDebit'] = M(
        pc02cs01=str(len(bad_accs)),
        pc02cj01=str(sum(to_int(a, ('latestInfo', 'pd01bj02')) for a in bad_accs)),
    )

    bus_groups = defaultdict(list)
    for a in accounts:
        bus_groups[str((a.get('accountBasic') or {}).get('pd01ad03') or '99')].append(a)
    tips = []
    for bus, accs in sorted(bus_groups.items(), key=lambda kv: -len(kv[1]))[:5]:
        opens = [str((a.get('accountBasic') or {}).get('pd01ar01') or '9999') for a in accs]
        tips.append(M(
            pc02as01=str(max(1, (rt.year - int(min(opens)[:4])) * 12)),
            pc02as02=str(len(accs)), pc02ad01=bus, pc02ad02='1',
            pc02as03=str(sum(1 for a in accs if is_overdue_acc(a))),
            pc02ar01=min(opens)[:7],
        ))
    s['tradeTips'] = tips
    s['recoveries'] = []
    s['relatedRepayDutys'] = []
    s['publics'] = []
    s['postpaySummary'] = M(pc030s01='0', pc030d01='', pc030s02='0', pc030j01='0')
    report['summaryInfo'] = s


# ============ 元字段刷新 + 主流程 ============

def refresh_meta(node, td: str, rsn: str, seq: str | None = None, parent_seq: str | None = None) -> None:
    """递归刷新冗余元字段；列表位置重编 seq。

    latest5yearDetails 行同时带 seq（账户号，来自外层账户）与 seq1（行号，列表位置）。
    """
    if isinstance(node, dict):
        if 'tranDate' in node:
            node['tranDate'] = td
            node['createTime'] = td
        if 'reportsn' in node:
            node['reportsn'] = rsn
        if seq is not None:
            if 'seq1' in node:
                node['seq1'] = seq
                if parent_seq is not None and 'seq' in node:
                    node['seq'] = parent_seq
            elif 'seq' in node:
                node['seq'] = seq
        for v in node.values():
            if isinstance(v, dict):
                refresh_meta(v, td, rsn, seq, parent_seq)
            elif isinstance(v, list):
                my_seq = seq if 'seq' in node else parent_seq
                for i, item in enumerate(v):
                    refresh_meta(item, td, rsn, str(i + 1), parent_seq=my_seq)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            refresh_meta(item, td, rsn, str(i + 1), parent_seq=parent_seq)


def calibrate_one(report: dict, stem: str) -> tuple[dict, int]:
    rng = random.Random(stem)
    rt = sample_report_time(rng)
    rare_pool = [(k, a[k]) for a in (report.get('accountInfos') or [])
                 for k in ('specialEvents', 'largeSpecialInstalments') if k in a]

    report['reportTime'] = f"{rt.isoformat()}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
    report['tranDate'] = report['createTime'] = rt.strftime('%Y%m%d')
    report['reportsn'] = f"mock_{rt.strftime('%Y%m%d%H%M%S')}{rng.randint(100000, 999999)}"

    calibrate_person(rng, report, rt)
    calibrate_score(rng, report)
    report['accountInfos'] = calibrate_accounts(rng, report, rt, rare_pool)
    report['queryRecords'] = calibrate_queries(rng, rt)
    calibrate_public(rng, report, rt)
    calibrate_obligations(rng, report, rt)
    recompute_summary(report, rt)
    refresh_meta(report, report['tranDate'], report['reportsn'])
    return report, len(report['accountInfos'])


def main() -> int:
    ap = argparse.ArgumentParser(description='Mock 征信报告校准器（按 MOCK校正说明文档.md）')
    ap.add_argument('--src', required=True, help='原 mock 目录')
    ap.add_argument('--dst', required=True, help='校准输出目录')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 份（试跑）')
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob('*.json'))
    if args.limit:
        paths = paths[:args.limit]

    n_ok = n_skip = n_capped = 0
    for p in paths:
        try:
            report = json.loads(p.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f'SKIP {p.name}: JSON 解析失败 {e}')
            n_skip += 1
            continue
        if not isinstance(report, dict) or 'accountInfos' not in report or 'personInfo' not in report:
            print(f'SKIP {p.name}: 离群文件（空 stub / 异构信封 / 残缺报告）')
            n_skip += 1
            continue
        out, n_acc = calibrate_one(report, p.stem)
        if n_acc >= ACCOUNT_CAP:
            n_capped += 1
        violations, _ = check_report(out)   # 写盘前自检 16 项不变量
        if violations:
            print(f'ERROR {p.name}: 自检发现 {len(violations)} 处不变量违反，前 5 处：')
            for line in violations[:5]:
                print(f'  {line}')
            return 1
        (dst / p.name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        n_ok += 1

    print(f'完成：校准 {n_ok} | 跳过 {n_skip} | 账户数封顶({ACCOUNT_CAP})报告 {n_capped} | 输出 {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
