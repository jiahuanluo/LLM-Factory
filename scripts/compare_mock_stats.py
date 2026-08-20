"""三方取值范围对比：真实生产画像 | 原 mock | 校准后 mock。

真实画像数字来自 data/pbc/mock_calib_verify.txt（11,730 份真实报告实测）。
用法：
  python scripts/compare_mock_stats.py --orig <原mock目录> --calib <校准后目录> [--out report.txt]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_mock_calibration import collect_stats, quantile  # noqa: E402

REAL = {  # 真实语料画像（画像分位）
    'reportTime': "['2020-02-15', '2026-08-04']",
    'accounts': 'min=0 p50=42 p95=172 max=1653',
    'types': 'D1:340912 R2:160223 R1:102372 R4:93711 R3:1982 C1:246',
    'd1': 'min=1 p50=40000 p95=1000000 max=39000000',
    'r2': 'min=1 p50=20000 p95=102379 max=17500000',
    'overdue': 'min=1 p50=15498 p95=308935 max=17901989',
    'queries': 'min=0 p50=18 p95=88 max=416',
    'reasons': "02:211430 08:102496 03:12409 20:2103 24:2068",
    'score': 'min=744 p50=878 p95=933 max=962（校准目标 600-950）',
    'gender': "男 83.6% / 女 16.4%",
}


def qline(xs):
    if not xs:
        return '无'
    return f"min={min(xs)} p50={quantile(xs, .5)} p95={quantile(xs, .95)} max={max(xs)}"


def main() -> int:
    ap = argparse.ArgumentParser(description='三方取值范围对比')
    ap.add_argument('--orig', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    def load(d):
        reports = []
        for p in sorted(Path(d).glob('*.json')):
            try:
                r = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            if isinstance(r, dict) and 'accountInfos' in r and 'personInfo' in r:
                reports.append(r)
        return collect_stats(reports)

    import json
    o, c = load(args.orig), load(args.calib)
    L = []
    L.append('**mock 校准三方对比：真实语料(画像) | 原 mock | 本次校准后 mock**')
    L.append('')
    L.append(f"文件数：原 mock={o['files']}  校准后={c['files']}（离群文件 json_4/json_351/json_352 跳过）")
    L.append('')

    def sec(title, real, orig, calib):
        L.append(f'**[{title}]**')
        L.append(f'真实(画像): {real}')
        L.append(f'原mock: {orig}')
        L.append(f'校准后: {calib}')
        L.append('')

    def types_line(st):
        return ' '.join(f"{k}:{v}" for k, v in sorted(st['types'].items(), key=lambda kv: -kv[1]))

    def reasons_line(st):
        return ' '.join(f"{k}:{v}" for k, v in sorted(st['query_reasons'].items(), key=lambda kv: -kv[1]))

    def rt_line(st):
        if not st['report_times']:
            return '无'
        return f"{min(st['report_times'])} ~ {max(st['report_times'])}"

    def cnt_line(st, key):
        xs = st[key]
        return f"min={min(xs)} p50={quantile(xs, .5)} p95={quantile(xs, .95)} max={max(xs)}" if xs else '无'

    def enum_line(st, key):
        return ' '.join(f"{k}:{v}" for k, v in sorted(st[key].items(), key=lambda kv: -kv[1])) or '无'

    def score_line(st):
        xs = st['scores']
        return f"min={min(xs)} p50={quantile(xs, .5)} p95={quantile(xs, .95)} max={max(xs)}" if xs else '无评分'

    def age_line(st):
        xs = st['ages']
        return f"min={min(xs)} p50={quantile(xs, .5)} p95={quantile(xs, .95)} max={max(xs)}" if xs else '无DOB'

    sec('报告日期 reportTime 跨度', REAL['reportTime'], rt_line(o), rt_line(c))
    sec('每份报告账户数', REAL['accounts'], cnt_line(o, 'accounts_per_report'), cnt_line(c, 'accounts_per_report'))
    sec('账户类型覆盖 pd01ad01', REAL['types'], types_line(o), types_line(c))
    sec('D1 借款金额 pd01aj01', REAL['d1'], qline(o['d1_amount']), qline(c['d1_amount']))
    sec('R2 授信额度 pd01aj02', REAL['r2'], qline(o['r2_limit']), qline(c['r2_limit']))
    sec('逾期金额 (>0)', REAL['overdue'], qline(o['overdue']), qline(c['overdue']))
    sec('账户状态 pd01bd01 (关闭态)', '1-8 全覆盖（校准目标）', enum_line(o, 'bd01'), enum_line(c, 'bd01'))
    sec('活跃态 pd01cd01', '≥6 种（校准目标）', enum_line(o, 'cd01'), enum_line(c, 'cd01'))
    sec('latest24state 去重字符串数', '数百种', f"{len(o['latest24'])} 种", f"{len(c['latest24'])} 种")
    sec('每份报告查询数', REAL['queries'], cnt_line(o, 'queries_per_report'), cnt_line(c, 'queries_per_report'))
    sec('查询原因 ph010q03', REAL['reasons'], reasons_line(o), reasons_line(c))
    sec('性别', REAL['gender'], enum_line(o, 'gender'), enum_line(c, 'gender'))
    sec('学历 pb01ad02', '8 种枚举', enum_line(o, 'edu'), enum_line(c, 'edu'))
    sec('婚姻 pb020d01', '4 种枚举', enum_line(o, 'marriage'), enum_line(c, 'marriage'))
    sec('评分 pc010q01', REAL['score'], score_line(o), score_line(c))
    sec('年龄 (reportTime - DOB)', '25-65（校准目标）', age_line(o), age_line(c))
    sec('零账户报告比例', '~0.3%',
        f"{o['zero_account']}/{o['files']}", f"{c['zero_account']}/{c['files']}")

    text = '\n'.join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')
        print(f'\n报告已写入 {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
