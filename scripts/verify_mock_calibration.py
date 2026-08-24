"""Mock 征信报告校准验证器：16 项业务不变量 + 分布验收门。

用法：
  python scripts/verify_mock_calibration.py <报告目录> [--out report.txt]
  exit 0 = 所有验收门通过；exit 1 = 存在失败项。

不变量与验收门来自 data/pbc/MOCK校正说明文档.md §4/§5，
对照基线为 data/pbc/mock_calib_verify.txt（真实 11,730 份生产报告画像）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

LOAN_TYPES = ('D1', 'R4', 'C1')   # 贷款类：有借款金额 pd01aj01
CARD_TYPES = ('R1', 'R2', 'R3')   # 信用卡类：有授信额度 pd01aj02
BAD_ACTIVE = {'2', '3', '4', '5', '8'}  # 活跃态里的逾期/呆账集合


# ---------- 解析工具 ----------

def pint(v):
    """宽松转 int：'' / '#' / None / 非数字 → None（不把空值当 0）。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or not s.lstrip('-').isdigit():
        return None
    return int(s)


def pdate(v):
    if not v or not isinstance(v, str):
        return None
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


def ym(v):
    """'YYYY-MM' / 'YYYY-MM-DD' → y*12+m 整数，便于比较。"""
    if not v or not isinstance(v, str):
        return None
    try:
        y, m = int(v[:4]), int(v[5:7])
        return y * 12 + m
    except (ValueError, IndexError):
        return None


def months_between(d1: date, d2: date) -> int:
    """d1→d2 经过的月数（向上取整，用于存续期下界）。"""
    return max(0, (d2.year - d1.year) * 12 + d2.month - d1.month)


def quantile(xs: list, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]


# ---------- 16 项业务不变量（spec §4） ----------

def check_report(report: dict) -> tuple[list, int]:
    """返回 (violation 描述列表, 检查过的账户数)。"""
    v = []
    rt = None
    if isinstance(report.get('reportTime'), str):
        rt = pdate(report['reportTime'][:19])
    rt_ym = ym(report.get('reportTime', '')[:7]) if report.get('reportTime') else None

    accounts = report.get('accountInfos') or []
    for acc in accounts:
        basic = acc.get('accountBasic') or {}
        latest = acc.get('latestInfo') or {}
        mps = acc.get('latestMonthPayState') or {}
        atype = str(basic.get('pd01ad01') or '')
        label = f"账号{basic.get('pd01ai01')}/{atype}"
        bd01 = str(latest.get('pd01bd01') or '')
        cd01 = str(mps.get('pd01cd01') or '')

        # 4.1 生命周期状态选源
        if bd01 and cd01:
            v.append(f"[{label}] pd01bd01={bd01} 与 pd01cd01={cd01} 同时有值")
        if bd01 and 'latestMonthPayState' in acc:
            v.append(f"[{label}] 关闭态账户仍带 latestMonthPayState 节点")
        if not bd01 and 'latestMonthPayState' not in acc:
            v.append(f"[{label}] 活跃账户缺 latestMonthPayState 节点")

        # 4.2 类型 ↔ 金额字段
        aj01, aj02 = basic.get('pd01aj01'), basic.get('pd01aj02')
        if atype in LOAN_TYPES:
            if pint(aj01) is None:
                v.append(f"[{label}] 贷款类缺借款金额 pd01aj01")
            if pint(aj02) is not None:
                v.append(f"[{label}] 贷款类不应有授信额度 pd01aj02={aj02}")
        elif atype in CARD_TYPES:
            if pint(aj02) is None:
                v.append(f"[{label}] 信用卡类缺授信额度 pd01aj02")
            if pint(aj01) is not None:
                v.append(f"[{label}] 信用卡类不应有借款金额 pd01aj01={aj01}")

        # 4.3 信用卡额度一致性
        if atype in CARD_TYPES and not bd01:  # 活跃
            bj01, cj02 = pint(latest.get('pd01bj01')), pint(mps.get('pd01cj02'))
            if bj01 is not None and cj02 is not None and bj01 != cj02:
                v.append(f"[{label}] 活跃卡余额 pd01bj01={bj01} != 已用 pd01cj02={cj02}")
            limit = pint(aj02)
            if cj02 is not None and limit is not None and cj02 > limit * 1.1 + 0.5:
                v.append(f"[{label}] 已用 {cj02} 超授信 {limit}×1.1")

        # 4.4 日期一致性
        ar01, br01 = pdate(basic.get('pd01ar01')), pdate(latest.get('pd01br01'))
        if ar01 and br01 and br01 < ar01:
            v.append(f"[{label}] 关闭日期 {br01} 早于开立日期 {ar01}")
        if ar01 and rt and ar01 > rt:
            v.append(f"[{label}] 开立日期 {ar01} 晚于报告日 {rt}")
        if br01 and rt and br01 > rt:
            v.append(f"[{label}] 关闭日期 {br01} 晚于报告日 {rt}")
        if bd01 and br01 and ar01:
            span = months_between(ar01, br01)
            terms = pint(basic.get('pd01as01'))
            if terms is not None and terms < span:
                v.append(f"[{label}] 期数 {terms} < 存续月数 {span}")

        p24 = acc.get('latest24PayState') or {}
        dr02, dr01 = ym(p24.get('pd01dr02')), ym(p24.get('pd01dr01'))
        if dr02 is not None and rt_ym is not None and dr02 > rt_ym:
            v.append(f"[{label}] latest24 结束月 {p24.get('pd01dr02')} 晚于报告月")
        if dr01 is not None and dr02 is not None and dr02 - dr01 != 23:
            v.append(f"[{label}] latest24 起止跨 {dr02 - dr01} 月 != 23")
        five = acc.get('latest5year') or {}
        for row in five.get('latest5yearDetails') or []:
            if (m := ym(row.get('pd01er03'))) is not None and rt_ym is not None and m > rt_ym:
                v.append(f"[{label}] 5y 明细月份 {row.get('pd01er03')} 晚于报告月")
                break

        # 4.5 逾期一致性
        bj02 = pint(latest.get('pd01bj02'))
        if bd01 == '2' and not (bj02 and bj02 > 0):
            v.append(f"[{label}] 关闭态逾期(2)但 pd01bj02={latest.get('pd01bj02')}")
        cj06 = pint(mps.get('pd01cj06'))
        if cj06 is not None and cj06 > 0 and cd01 and cd01 not in BAD_ACTIVE:
            v.append(f"[{label}] 当前逾期 {cj06} 但活跃态 {cd01} 非逾期/呆账")
        if str(latest.get('pd01bd03') or '') and not (bd01 == '4' or cd01 == '5'):
            v.append(f"[{label}] 非呆账账户带五级分类 pd01bd03={latest.get('pd01bd03')}")

        # 4.6 R2/R3 信用卡不结清
        if atype in ('R2', 'R3') and bd01 == '3':
            v.append(f"[{label}] 信用卡被标结清(3)")

    # 4.7 summaryInfo 重算正确
    n = {t: 0 for t in LOAN_TYPES + CARD_TYPES}
    for acc in accounts:
        t = str((acc.get('accountBasic') or {}).get('pd01ad01') or '')
        if t in n:
            n[t] += 1
    s = report.get('summaryInfo') or {}
    if s.get('loanCardAccount'):
        hs02 = pint((s['loanCardAccount'] or {}).get('pc02hs02'))
        if hs02 is not None and hs02 != n['R2']:
            v.append(f"[summary] pc02hs02={hs02} != 实际 R2 账户数 {n['R2']}")
    if s.get('nonrevolvingLoan'):
        es02 = pint((s['nonrevolvingLoan'] or {}).get('pc02es02'))
        if es02 is not None and es02 != n['D1']:
            v.append(f"[summary] pc02es02={es02} != 实际 D1 账户数 {n['D1']}")

    return v, len(accounts)


# ---------- 分布统计（spec §5 三方对比口径） ----------

def collect_stats(reports: list) -> dict:
    st = {
        'files': len(reports),
        'accounts_per_report': [], 'queries_per_report': [],
        'types': Counter(), 'bd01': Counter(), 'cd01': Counter(),
        'd1_amount': [], 'r2_limit': [], 'overdue': [],
        'latest24': set(), 'query_reasons': Counter(),
        'gender': Counter(), 'edu': Counter(), 'marriage': Counter(),
        'scores': [], 'ages': [], 'report_times': [],
        'r2_util_abuse': 0, 'zero_account': 0,
    }
    for r in reports:
        accounts = r.get('accountInfos') or []
        st['accounts_per_report'].append(len(accounts))
        if not accounts:
            st['zero_account'] += 1
        st['queries_per_report'].append(len(r.get('queryRecords') or []))
        rt = pdate(str(r.get('reportTime') or '')[:19])
        if rt:
            st['report_times'].append(rt)
        idn = ((r.get('personInfo') or {}).get('identity') or {})
        if idn.get('pb01ad01'):
            st['gender'][str(idn['pb01ad01'])] += 1
        if idn.get('pb01ad02'):
            st['edu'][str(idn['pb01ad02'])] += 1
        mar = ((r.get('personInfo') or {}).get('marriage') or {})
        if mar.get('pb020d01'):
            st['marriage'][str(mar['pb020d01'])] += 1
        dob = pdate(idn.get('pb01ar01'))
        if dob and rt:
            st['ages'].append((rt - dob).days // 365)
        sc = pint((r.get('score') or {}).get('pc010q01'))
        if sc is not None:
            st['scores'].append(sc)
        for acc in accounts:
            basic, latest, mps = acc.get('accountBasic') or {}, acc.get('latestInfo') or {}, acc.get('latestMonthPayState') or {}
            t = str(basic.get('pd01ad01') or '')
            st['types'][t] += 1
            if str(latest.get('pd01bd01') or ''):
                st['bd01'][str(latest['pd01bd01'])] += 1
            if str(mps.get('pd01cd01') or ''):
                st['cd01'][str(mps['pd01cd01'])] += 1
            if t == 'D1' and pint(basic.get('pd01aj01')) is not None:
                st['d1_amount'].append(pint(basic['pd01aj01']))
            if t == 'R2' and pint(basic.get('pd01aj02')) is not None:
                st['r2_limit'].append(pint(basic['pd01aj02']))
                used, limit = pint(mps.get('pd01cj02')), pint(basic['pd01aj02'])
                if used is not None and limit and used > limit * 1.1 + 0.5:
                    st['r2_util_abuse'] += 1
            for k in ('pd01bj02',):
                x = pint(latest.get(k))
                if x and x > 0:
                    st['overdue'].append(x)
            x = pint(mps.get('pd01cj06'))
            if x and x > 0:
                st['overdue'].append(x)
            p24 = (acc.get('latest24PayState') or {}).get('latest24state')
            if p24:
                st['latest24'].add(p24)
        for q in r.get('queryRecords') or []:
            if q.get('ph010q03'):
                st['query_reasons'][str(q['ph010q03'])] += 1
    return st


def evaluate_gates(st: dict, violations: int, accounts_total: int) -> list:
    """返回 [(门名, PASS/FAIL, 实测值)]。验收标准：spec §5。"""
    g = []

    def gate(name, ok, detail):
        g.append((name, 'PASS' if ok else 'FAIL', detail))

    acc = st['accounts_per_report']
    p50 = quantile(acc, 0.5)
    gate('每报告账户数 p50 ∈ [30,80]', p50 is not None and 30 <= p50 <= 80,
         f"p50={p50} p95={quantile(acc, 0.95)} max={max(acc) if acc else None}")
    q = st['queries_per_report']
    qp50 = quantile(q, 0.5)
    gate('每报告查询数 p50 ∈ [15,25]', qp50 is not None and 15 <= qp50 <= 25,
         f"p50={qp50} p95={quantile(q, 0.95)}")
    d1max = max(st['d1_amount'], default=None)
    gate('D1 借款金额 max ≥ 3千万(接近3900万)', d1max is not None and d1max >= 30_000_000,
         f"p50={quantile(st['d1_amount'], .5)} p95={quantile(st['d1_amount'], .95)} max={d1max}")
    r2p50 = quantile(st['r2_limit'], 0.5)
    gate('R2 授信额度 p50 ∈ [1.5万,3万]', r2p50 is not None and 15_000 <= r2p50 <= 30_000,
         f"p50={r2p50} p95={quantile(st['r2_limit'], .95)} max={max(st['r2_limit'], default=None)}")
    omax = max(st['overdue'], default=None)
    gate('逾期金额 max ≥ 1700万(接近1790万)', omax is not None and omax >= 17_000_000,
         f"p50={quantile(st['overdue'], .5)} max={omax}")
    need = {str(i) for i in range(1, 9)}
    gate('pd01bd01 枚举 1-8 全覆盖', need <= set(st['bd01']), dict(sorted(st['bd01'].items())))
    gate('pd01cd01 枚举 ≥6 种', len(st['cd01']) >= 6, dict(sorted(st['cd01'].items())))
    gate('latest24state 去重 > 100', len(st['latest24']) > 100, f"{len(st['latest24'])} 种")
    years = {d.year for d in st['report_times']}
    gate('reportTime 跨 2020-2026', 2020 in years and 2026 in years,
         f"{min(st['report_times'])} ~ {max(st['report_times'])}" if st['report_times'] else '无')
    gate('评分 ∈ [600,950]', st['scores'] and 600 <= min(st['scores']) and max(st['scores']) <= 950,
         f"min={min(st['scores'])} max={max(st['scores'])}" if st['scores'] else '无评分')
    gate('年龄 ∈ [25,65]', st['ages'] and 25 <= min(st['ages']) and max(st['ages']) <= 65,
         f"min={min(st['ages'])} max={max(st['ages'])}" if st['ages'] else '无DOB')
    gate('性别男女都有', set(st['gender']) >= {'1', '2'}, dict(st['gender']))
    gate('学历枚举 ≥6 种', len(st['edu']) >= 6, dict(st['edu']))
    gate('婚姻枚举 ≥3 种', len(st['marriage']) >= 3, dict(st['marriage']))
    gate('R2 利用率 ≤1.1 (无超限)', st['r2_util_abuse'] == 0, f"违反 {st['r2_util_abuse']} 例")
    zero_ratio = st['zero_account'] / st['files'] if st['files'] else 0
    gate('零账户报告比例 ≤1%', zero_ratio <= 0.01, f"{st['zero_account']}/{st['files']}={zero_ratio:.3%}")
    viol_rate = violations / accounts_total if accounts_total else 0
    gate('业务不变量违反率 <0.1%', viol_rate < 0.001, f"{violations}/{accounts_total}={viol_rate:.4%}")
    return g


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description='校准 mock 征信报告验证器')
    ap.add_argument('dir', help='报告目录（*.json）')
    ap.add_argument('--out', default=None, help='报告输出路径（默认只打印）')
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob('*.json'))
    reports, skipped, violations = [], [], 0
    acc_total = 0
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped.append(p.name)
            continue
        if not isinstance(d, dict) or 'accountInfos' not in d or 'personInfo' not in d:
            skipped.append(p.name)
            continue
        v, n = check_report(d)
        violations += len(v)
        acc_total += n
        reports.append(d)
        if len(v) <= 3:
            for line in v:
                print(f"  VIOLATION {p.name}: {line}")
        else:
            print(f"  VIOLATION {p.name}: {len(v)} 处（省略明细）")

    st = collect_stats(reports)
    gates = evaluate_gates(st, violations, acc_total)

    lines = []
    lines.append(f'验证目录: {args.dir}')
    lines.append(f'文件: 总 {len(paths)} | 有效 {len(reports)} | 跳过 {len(skipped)} {skipped}')
    lines.append(f'账户总数: {acc_total} | 不变量违反: {violations}')
    lines.append('')
    for name, status, detail in gates:
        lines.append(f'[{status}] {name} -> {detail}')
    n_pass = sum(1 for _, s, _ in gates if s == 'PASS')
    lines.append('')
    lines.append(f'===== {n_pass}/{len(gates)} 门通过 =====')

    text = '\n'.join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')
        print(f'\n报告已写入 {args.out}')
    return 0 if n_pass == len(gates) else 1


if __name__ == '__main__':
    sys.exit(main())
