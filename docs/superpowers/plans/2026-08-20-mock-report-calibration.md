# Mock 征信报告校准 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `data/pbc/cris_json_split/`（598 份近退化 mock）校准为字段取值范围与 11,730 份真实生产报告一致的合成数据，输出 `data/pbc/cris_json_split_calibrated/`（596 份，跳过 2 个离群文件）。

**Architecture:** 两个独立可运行的纯标准库脚本。`scripts/calibrate_mock_reports.py` 以"每份文件自带账户为结构骨架 deepcopy + 按 spec 逐字段重写"的方式合成（保留 specialEvents/largeSpecialInstalments 等稀疏节点多样性），summaryInfo 从生成后的账户/查询重算；`scripts/verify_mock_calibration.py` 实现 16 项业务不变量 + 分布验收门，先对原 mock 跑出红（TDD），校准后必须全绿。参数唯一来源：`data/pbc/MOCK校正说明文档.md`（下称 spec），验收对照：`data/pbc/mock_calib_verify.txt`。

**Tech Stack:** Python 3 标准库（json / random / argparse / datetime / copy / pathlib / statistics），无第三方依赖。

**关键事实（探索已确认，与 spec §2 的出入以真实文件为准）：**
- 账户节点名：`accountBasic` / `latestInfo` / `latestMonthPayState` / `latest24PayState` / `latest5year`（内含 `latest5yearDetails[]`）/ `specialTrades`，稀疏：`specialEvents` / `largeSpecialInstalments`
- publicInfo 子键为 `taxarrears`（全小写）
- 所有数值字段是**字符串**（如 `"360000"`），空值是 `""`，不丢 `'0'` 语义
- 每个节点带冗余元字段 `tranDate`/`createTime`/`seq`/`reportsn`，重写时必须同步
- 离群文件：`json_4.json`（无 accountInfos/personInfo 的 stub）、`json_352.json`（顶层为 `data`/`stype` 司法信封）→ 跳过不复制
- 输入 81MB 未被 git 跟踪；输出目录预计数百 MB，写入用户 checkout（不进 git）

---

### 常量表（校准器唯一参数源，直接嵌入 `calibrate_mock_reports.py`）

```python
# 账户类型频率（spec 3.4，R3/C1 已提额保证覆盖）
TYPE_FREQ   = {'D1': 487, 'R2': 229, 'R1': 146, 'R4': 134, 'R3': 15, 'C1': 10}
LOAN_TYPES, CARD_TYPES = ('D1', 'R4', 'C1'), ('R1', 'R2', 'R3')

# 金额逆 CDF 分位锚点 [(q, 值), ...]（spec 3.4）
AMOUNT_KNOTS = {  # 贷款类填 pd01aj01，信用卡类填 pd01aj02
    'D1': [(0.05, 1086), (0.25, 10000), (0.50, 40000), (0.75, 173000), (0.95, 1000000), (0.99, 3000000), (1.0, 39000000)],
    'R4': [(0.05, 3200), (0.25, 25000), (0.50, 100000), (0.75, 300000), (0.95, 1200000), (0.99, 4000000), (1.0, 30000000)],
    'C1': [(0.05, 453), (0.25, 3428), (0.50, 19738), (0.75, 70477), (0.95, 233227), (0.99, 405795), (1.0, 2271505)],
    'R1': [(0.05, 100), (0.25, 8000), (0.50, 39239), (0.75, 161000), (0.95, 894500), (0.99, 2600000), (1.0, 25000000)],
    'R2': [(0.05, 1964), (0.25, 8000), (0.50, 20000), (0.75, 48000), (0.95, 102379), (0.99, 231080), (1.0, 17500000)],
    'R3': [(0.05, 50), (0.25, 2000), (0.50, 5000), (0.75, 5000), (0.95, 20000), (0.99, 50000), (1.0, 200000)],
}
# 余额/已用额度锚点（贷款类=余额 pd01bj01；信用卡类=已用 pd01cj02，余额=已用）
BALANCE_KNOTS = {
    'D1': [(0.25, 15000), (0.50, 100000), (0.95, 2331851), (1.0, 39000000)],
    'R4': [(0.25, 55000), (0.50, 200000), (0.95, 2200000), (1.0, 16000000)],
    'C1': [(0.25, 3324), (0.50, 19117), (0.95, 240715), (1.0, 2271505)],
    'R1': [(0.25, 0), (0.50, 200), (0.95, 500000), (1.0, 19107000)],
    'R2': [(0.25, 0), (0.50, 0), (0.95, 64252), (1.0, 1836233)],
    'R3': [(0.25, 0), (0.50, 0), (0.95, 20000), (1.0, 100000)],  # spec 未给 R3，按额度量级缩放
}
# 正逾期金额锚点（pd01bj02 / pd01ej01 / pd01cj06）
OVERDUE_KNOTS = [(0.25, 3978), (0.50, 15498), (0.95, 308935), (0.99, 1315394), (1.0, 17901989)]

ACTIVE_RATE = 0.23
ACTIVE_CD01_FREQ  = {'1': 70, '5': 8, '2': 6, '3': 5, '4': 4, '6': 4, '31': 2, '8': 1}   # 去偏扁平
CLOSED_BD01_FREQ  = {'3': 40, '4': 20, '1': 12, '2': 10, '5': 7, '6': 5, '7': 4, '8': 2}  # 去偏扁平
BAD_CD_SET = {'2', '3', '4', '5', '8'}          # 活跃逾期/呆账态集合
FIVE_LEVEL_FREQ = {'3': 2, '4': 2, '5': 1}      # 五级分类 pd01bd03（仅呆账有）

# 账龄中位月数（spec 3.4）：采样 months = clamp(round(med*(0.15+1.7*u)), 1, 480)
AGE_MEDIAN_MONTHS = {'D1': 77, 'R2': 139, 'R1': 21, 'R4': 6, 'C1': 36, 'R3': 214}

# 每报告计数锚点（spec 3.4/3.5）
ACCOUNT_COUNT_KNOTS = [(0.0, 0), (0.003, 0), (0.25, 22), (0.50, 42), (0.75, 75), (0.95, 172), (0.99, 304), (1.0, 1653)]  # 封顶 200
QUERY_COUNT_KNOTS   = [(0.0, 0), (0.25, 9), (0.50, 18), (0.75, 34), (0.95, 88), (1.0, 416)]                              # 封顶 300
ACCOUNT_CAP, QUERY_CAP = 200, 300

SUBTYPE_FREQ  = {'11': 564924, '24': 64003, '51': 40896, '21': 13825, '23': 4846, '12': 4538, '22': 2361, '15': 2310, '16': 1082}
BUSINESS_FREQ = {'91': 237419, '41': 216317, '81': 152873, '99': 43683, '21': 10235, '11': 9946, '51': 9415, '82': 7350,
                 '52': 4362, '92': 2578, '12': 1575, '13': 555, '53': 375, 'B1': 232, '62': 68, '42': 61, '32': 43, 'A1': 16}
GENDER_FREQ   = {'1': 9781, '2': 1922}
EDU_FREQ      = {'30': 3351, '20': 2978, '60': 1748, '90': 1016, '40': 963, '91': 807, '10': 404, '99': 390}
DEGREE_FREQ   = {'9': 6663, '5': 3809, '4': 966, '3': 124, '0': 50, '2': 21, '1': 6}
EMPLOY_FREQ   = {'91': 5622, '21': 1595, '17': 1494, '90': 943, '54': 767, '13': 412, '11': 230}
MARRIAGE_FREQ = {'20': 18921, '10': 18977, '40': 699, '91': 150}
RESIDENCE_FREQ= {'9': 5293, '1': 2819, '7': 1722, '2': 769, '11': 332, '3': 276, '5': 381}
QUERY_REASON_FREQ = {'02': 211430, '08': 102496, '03': 12409, '20': 2103, '24': 2068}
QUERY_ORG_FREQ    = {'11': 193527, '53': 47154, '51': 40412, '24': 36267, '22': 4619, '41': 2101, '15': 2007, '12': 1701, '23': 1389}
SCORE_FACTORS = ['00', '01', '03', '05', '06', '15', '16']
CURRENCIES = [('CNY', 0.99)] + [(c, 0.0025) for c in ('USD', 'EUR', 'JPY', 'HKD')]

# reportTime 月份权重：2020-02..2026-08，一般月假定计数 1，2026-07/08 假定计数 30（对应真实 97.6% 偏态），取 sqrt 弱化
REPORT_MONTH_RANGE = (2020, 2), (2026, 8)
RECENT_BOOST = 30.0

# publicInfo 子段填充率（真实）×0.5 再稀疏化（spec 3.6）
PUBLIC_FILL_RATE = {'accFunds': 0.030, 'civilJudgements': 0.0001, 'forceExecutions': 0.0018,
                    'taxarrears': 0.0005, 'adminPunishments': 0.0005, 'salvations': 0.005,
                    'competences': 0.005, 'adminAwards': 0.005}
```

### 关键算法

**逆 CDF 采样**（spec 3.4）：`u~U(0,1)`；1% 概率直取 `knots[-1][1]`（保长尾极值）；否则在相邻锚点线性插值。金额精度：`>100000 → 整千`，`>1000 → 整百`，否则整数。

```python
def inv_cdf(rng, knots, tail_prob=0.01):
    if rng.random() < tail_prob:
        return knots[-1][1]
    u = rng.random()
    for (q0, v0), (q1, v1) in zip(knots, knots[1:]):
        if u <= q1:
            return round(v0 + (v1 - v0) * (u - q0) / (q1 - q0)) if q1 > q0 else v0
    return knots[-1][1]
```

**身份证校验位**（spec 3.2）：17 位本体（6 位地区码 + 8 位生日 + 3 位顺序，第 17 位奇=男/偶=女）→ 权重 `[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]` 加权和 mod 11 → 查表 `'10X98765432'`。

**月份运算**：`ym = y*12+m` 整数比较；`str_to_ym('2018-10')→(2018,10)`；`add_months(ym, k)`；锚定"结束月 ≤ 报告月"全部用整数比较实现。

**latest24state 生成**：24 字符按生命周期逐月采字符（旧→新）：
- normal（活跃'1'/关闭'1'）：`N 0.80 / * 0.15 / # 0.03 / 数字1-3 0.02`
- bad（关闭'4'/活跃'5'）：`B 0.35 / D 0.10 / G 0.10 / 数字1-7 0.15 / N 0.15 / * 0.10 / C 0.05`
- settled（关闭'3'）：`N 0.75 / * 0.10 / C 0.15`（最后一月强制 `C`）
- overdue（关闭'2'/活跃∈{2,3}）：`N 0.40 / 数字1-7 0.35 / * 0.15 / # 0.10`
结束月 = 报告月 − `U(0, min(12, age−24))`；起始月 = 结束月 − 23；仅 `age ≥ 24` 且 `rng < 0.6` 的账户有该节点。

**latest5yearDetails**：行数 = `min(按生命周期中位采样, age+1, 60)`（normal med 59 / settled med 1 / bad med 26 / other med 57）；月份从报告月往前逐行；状态字符按生命周期频率（normal: `N/*/#`，bad: `/ C N #`，settled: `C N *`，overdue 同上）；字符是数字或 `B/D/G` 时 `pd01ej01` 取正逾期采样否则 `'0'`。`pd01er01/er02` = 最旧/最新行月份，`pd01es01` = 行数。

**summaryInfo 重算**（spec 3.7，全部从生成后的账户/查询计算）：
- 贷款段（D1→pc02e*，R1→pc02f*，R4→pc02g*）：`s01`=该类型去重机构数(pd01ai02)、`s02`=账户数、`j01`=Σpd01aj01、`j02`=Σpd01bj01、`j03`≈`round(j02*0.05)`
- 信用卡段（R2→pc02h*，R3→pc02i*）：`j01`=Σ授信、`j02`=单家机构最高授信、`j03`=单家最低（>0 家时）、`j04`=Σ已用、`j05`≈`round(j04*0.5)`、`s01/s02` 同上
- overdues[]：对"逾期/呆账账户"按 pd01ad03 分组 → `pc02ds01`=业务种类、`pc02dd01`='1'、`pc02ds02`=账户数、`pc02ds03`=逾期月份数（组内 5y 明细逾期行月去重）、`pc02dj01`=max(pd01ej01)、`pc02ds04`=最长连续逾期月数
- querySummary：`pc05ar01/ad01/ai01/aq01` 取 ph010r01 最近一条；`bs01/bs02`=1 月内原因 02/03 的去重机构数、`bs03/bs04`=1 月内原因 02/03 计数、`bs05`=原因'01'1 月内计数（生成端不产 → 0）、`bs06`=原因'01'2 年内计数（0）、`bs07`=原因'08'2 年内计数、`bs08`=原因'20'2 年内计数
- badDebit：`pc02cs01`=呆账账户数、`pc02cj01`=Σ呆账逾期金额；无则 '0'
- tradeTips：按 pd01ad03 分组各一条（`pc02as01`=组内最大账龄月、`pc02as02`=账户数、`pc02ad01`=业务种类、`pc02ad02`='1'、`pc02as03`=逾期账户数、`pc02ar01`=最早开立年月）
- recoveries / relatedRepayDutys / publics / postpaySummary：置空/零（生成端无对应明细时不虚造）

**元字段刷新**：写完所有业务字段后，递归遍历整份报告，把每个 dict 节点的 `tranDate`/`createTime` 覆盖为新报告日 `YYYYMMDD`、`reportsn` 覆盖为新 reportsn；列表节点的 `seq` 按新序号 1..N 重编（`latest5yearDetails` 的 `seq1` 行号 1..M）。

---

### Task 1: 校验器（TDD 红）

**Files:**
- Create: `scripts/verify_mock_calibration.py`

- [ ] **Step 1: 实现不变量检查**。CLI：`python scripts/verify_mock_calibration.py <目录> [--json out.json]`。对目录内每份合法报告逐账户检查 spec §4 的 16 项：
  1. `latestInfo.pd01bd01` 与 `latestMonthPayState.pd01cd01` 互斥（同账户两者不同时有值）
  2. 关闭账户无 `latestMonthPayState` 节点
  3. 活跃账户有 `latestMonthPayState` 且 `pd01bd01 == ''`
  4. 贷款类（D1/R4/C1）有 `pd01aj01` 无 `pd01aj02`
  5. 信用卡类（R1/R2/R3）有 `pd01aj02` 无 `pd01aj01`
  6. 活跃信用卡 `pd01bj01 == pd01cj02`
  7. `pd01cj02 ≤ pd01aj02 × 1.1`（信用卡）
  8. `pd01br01 ≥ pd01ar01` 且 `pd01ar01 ≤ reportTime`
  9. 关闭贷款 `pd01as01 ≥ 存续月数`
  10. `latest24PayState.pd01dr02 ≤ 报告月` 且 `pd01dr01 = pd01dr02 − 23 个月`
  11. 5y 明细所有 `pd01er03 ≤ 报告月`
  12. `pd01bd01 == '2'` → `pd01bj02 > 0`
  13. `pd01cj06 > 0` → `pd01cd01 ∈ {2,3,4,5,8}`
  14. `pd01bd03` 非空 → `pd01bd01 == '4'` 或 `pd01cd01 == '5'`
  15. R2/R3 关闭态 `pd01bd01 != '3'`
  16. `summaryInfo.loanCardAccount.pc02hs02 == 报告 R2 账户数` 且 `nonrevolvingLoan.pc02es02 == D1 账户数`
- [ ] **Step 2: 实现分布统计与验收门**（spec §5 表）：每报告账户数 p50∈[30,80]、查询数 p50∈[15,25]、D1 借款 max≥3e7、R2 额度 p50∈[15000,30000]、逾期 max≥1.7e7、pd01bd01 枚举 ⊇ {1..8}（不含 31）、pd01cd01 覆盖 ≥6 种、latest24state 去重 >100、reportTime 覆盖 2020 与 2026 年份、评分∈[600,950]、年龄∈[25,65]、男女都有、R2 利用率 ≤1.1、零账户比例 ∈[0,0.01]、不变量违反率 <0.1%。输出人类可读报告 + 逐门 PASS/FAIL，任一 FAIL → exit 1。
- [ ] **Step 3: 对原 mock 跑红**。`python scripts/verify_mock_calibration.py /home/daxigua/workspace/LLM-Factory/data/pbc/cris_json_split; echo "exit=$?"`。预期：账户数 p50=8、类型分布失衡、D1 max=520000 等大量 FAIL，exit=1。
- [ ] **Step 4: Commit** `feat(pbc): mock 校准验证器（16 不变量 + 分布验收门）`

### Task 2: 校准器 — 信封 / personInfo / score（spec §3.1-3.3）

**Files:**
- Create: `scripts/calibrate_mock_reports.py`

- [ ] **Step 1: 骨架**：`main(src_dir, dst_dir)` 遍历 `*.json`；`rng = random.Random(filename_stem)`；非 dict / 缺 `accountInfos`/`personInfo` → 记日志跳过（json_4、json_352 自然命中）。常量表 + `inv_cdf` + 月份/身份证工具函数。
- [ ] **Step 2: 信封**：reportTime 月份按 sqrt 权重采样、日 U(1,28)、时:分:秒随机；`tranDate`/`createTime` = 同日 `YYYYMMDD`（修复原 mock 二者不一致）；`reportsn = f'mock_{YYYYMMDD}{HHMMSS}{6位随机}'`；`name = '测' + 姓 + 名`（内置姓/名字池）。
- [ ] **Step 3: personInfo**：identity 五枚举按频率重采、DOB=报告日−U(25,65) 岁、certNo 18 位合成并同步 `header.request.pa01bi01`/`pa01bq01`；mobiles 1-5 条（`13x+8位`）；residences/professionals 1-5 条（枚举重采 + 日期 ≤ 报告日）；marriage 按频率，'10' 未婚清空配偶字段否则填随机配偶。
- [ ] **Step 4: score 补齐**：`pc010q01=U(600,950)`、`pc010q02=U(1,99)`、`pc010s01='2'`、`pc010d01`=SCORE_FACTORS 随机 2 个去重逗号拼接。
- [ ] **Step 5: Commit** `feat(pbc): 校准器信封/personInfo/score`

### Task 3: 校准器 — accountInfos 合成（spec §3.4 + §4）

**Files:**
- Modify: `scripts/calibrate_mock_reports.py`

- [ ] **Step 1: 账户数重采样**：ACCOUNT_COUNT_KNOTS 逆 CDF，封顶 200（命中记日志）；骨架池 = 原文件账户按"有无 latestMonthPayState"分两组 deepcopy 备用。
- [ ] **Step 2: 逐账户 accountBasic**：类型/子类/业务种类/币种按频率；`pd01ai01`=5 位数字、`pd01ai02`=2 字母；账龄按 AGE_MEDIAN_MONTHS 采样 → `pd01ar01 = 报告日 − 账龄`；金额按类型走 AMOUNT_KNOTS（贷款填 aj01 清 aj02，信用卡填 aj02 清 aj01）。
- [ ] **Step 3: 生命周期**：活跃 23% → `pd01bd01=''`、保留/植入 latestMonthPayState（pd01cd01 按 ACTIVE_CD01_FREQ）；关闭 → 删 latestMonthPayState、pd01bd01 按 CLOSED_BD01_FREQ（R2/R3 采到 '3' 改 '4'）、`pd01br01 = 报告日 − U(0, 账龄)`、`pd01ar02 = pd01br01`（关闭贷款）、`pd01as01 ≥ 存续月数`；活跃贷款 `pd01ar02 = 报告日 + 剩余期`、`pd01as01 = 账龄 + U(6,120)`；`pd01br03 = 报告日 − U(1,60) 天`。
- [ ] **Step 4: 余额/已用/逾期**：信用卡已用 = BALANCE_KNOTS 采样 clamp 到 ≤额度×1.1，活跃卡 `pd01bj01 = pd01cj02 = 已用`，`pd01cj01 = 已用`；贷款余额 = BALANCE_KNOTS（活跃/呆账/逾期），结清/正常关闭 = '0'；呆账/逾期账户 `pd01bj02` 与活跃 `pd01cj06` = OVERDUE_KNOTS 采样，正常 = '0'/''；五级分类 pd01bd03 仅呆账有（FIVE_LEVEL_FREQ）。
- [ ] **Step 5: 还款历史**：按"关键算法"生成 latest24state 与 latest5yearDetails（含 pd01ej01 联动）；specialTrades 5% 保留（金额 ±50% 抖动）；specialEvents/largeSpecialInstalments 随骨架带入（金额抖动）。
- [ ] **Step 6: 单元自检**：对生成账户直接调用 Task 1 的不变量函数（`from verify_mock_calibration import check_report_invariants`），试跑 3 份文件断言 0 违反，迭代修复。
- [ ] **Step 7: Commit** `feat(pbc): 校准器账户合成核心`

### Task 4: 校准器 — 查询 / 公共信息 / summaryInfo 重算（spec §3.5-3.9）

**Files:**
- Modify: `scripts/calibrate_mock_reports.py`

- [ ] **Step 1: queryRecords**：QUERY_COUNT_KNOTS 重采样封顶 300；`ph010r01 = 报告日 − U(1,730) 天`；原因/机构类型按频率；`ph010q02` 2 字母。
- [ ] **Step 2: publicInfo**：每子段按 PUBLIC_FILL_RATE 伯努利保留（保留则金额 ±50% 抖动、日期 ≤ 报告日），否则 `[]`；otherMarks 置 `[]`。
- [ ] **Step 3: agreementInfos/postpays/relatedRepayDutyInfos**：数量按中位（9/0/4）lognormal-ish 重采，复制模板条目金额 ±50% 抖动、日期 ≤ 报告日。
- [ ] **Step 4: summaryInfo 重算**：按"关键算法"重算五段 + overdues + querySummary + badDebit + tradeTips，其余置空。
- [ ] **Step 5: 元字段刷新 + 落盘**：递归刷 tranDate/createTime/reportsn、重编 seq；`json.dump(..., ensure_ascii=False, indent=1)`。
- [ ] **Step 6: Commit** `feat(pbc): 校准器查询/公共信息/汇总重算`

### Task 5: 小样本 → 全量 → 三方对比 → 交付

- [ ] **Step 1: 5 份试跑**：`python scripts/calibrate_mock_reports.py --src <用户checkout>/cris_json_split --dst <用户checkout>/cris_json_split_calibrated --limit 5`，跑 verify 全绿；失败则回 Task 2-4 迭代。
- [ ] **Step 2: 全量**：去掉 `--limit` 跑 598 份（预期 596 输出 + 2 跳过日志），后台运行。
- [ ] **Step 3: 验收**：verify 全绿（exit 0）；统计三方对比（真实 | 原 mock | 校准后）逐项对照 `mock_calib_verify.txt` 的数字量级：账户 p50≈46/p95≈154/max 200、D1 借款 p50≈4.3万/p95≈127万/max=39000000、逾期 max≈17902000、latest24state 数百种、状态枚举 1-8 全覆盖。把对比报告写到 `<用户checkout>/data/pbc/mock_calib_verify_new.txt`。
- [ ] **Step 4: 收尾**：data/pbc/CLAUDE.md 属于用户分支（本 worktree 基于 main 无此文件，不新建以免冲突）；最终 Commit + push worktree 分支；向用户报告输出路径与分支名。

**验收命令汇总：**
```bash
python scripts/verify_mock_calibration.py <dst>                 # exit 0 = 全绿
python scripts/calibrate_mock_reports.py --src <src> --dst <dst>  # 确定性：同输入同输出
```
