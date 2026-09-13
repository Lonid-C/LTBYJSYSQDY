# -*- coding: utf-8 -*-
"""交付前验收：文档中的关键数字必须与 results4/ 的结果文件逐一对得上，且不得残留旧口径措辞。

复核意见 §13 的可执行版本。分三组检查：
  A 数字一致性——样本外表、下界三口径、CVaR、bootstrap 区间、逐次结算、下单计价配对、选参适应度；
  B 措辞与术语——旧数字、旧术语、审查清单式表达、被作废的比例；
  C 交付结构——插图引用齐全、HTML 由 Markdown 生成且同步、Excel 费用列与逐日账本一致。

退出码非 0 即表示本包不可定稿。
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
PAPER = ROOT / '论文交付/2_论文部分/第四问_论文正文.md'
HTML = ROOT / '论文交付/2_论文部分/第四问_论文正文.html'
BASE = ROOT / '论文交付/数据底稿_所有数字出处.md'
SOLU = ROOT / '论文交付/4_第四问解答部分/第四问_完整解答.md'
ABST = ROOT / '论文交付/1_摘要部分/摘要.md'
DOCS = [PAPER, BASE, SOLU, ABST]

fails: list[str] = []
checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


def money(x):
    return f'{x:,.2f}'


# ---------------------------------------------------------------- A 数字
def check_numbers():
    oos = pd.read_csv(R / 'q4_out_of_sample.csv')
    paper = PAPER.read_text(encoding='utf8')
    base = BASE.read_text(encoding='utf8')
    for _, r in oos.iterrows():
        for col in ('total_cost', 'adjusted_cost'):
            v = money(r[col])
            ok(v in paper, f'[A1] 论文正文缺少 {r["problem"]} {str(r["method"]).split()[0]} 的 {col}={v}')
            ok(v in base, f'[A1] 数据底稿缺少 {r["problem"]} {str(r["method"]).split()[0]} 的 {col}={v}')

    lb = pd.read_csv(R / 'q4_full_horizon_bound.csv')
    for _, r in lb.iterrows():
        for col, name in (('adjusted_cost', '库存校正口径下界'),
                          ('cash_cost', '该解的现金分量'),
                          ('cash_lower_bound', '现金口径下界')):
            ok(money(r[col]) in paper, f'[A2] 论文正文缺少 {r["problem"]} 的{name} {money(r[col])}')
    # 三个量必须互不相等，且论文明确区分
    ok('不是现金最小值' in paper, '[A2] 论文未声明"该解的现金分量不是现金最小值"')

    cv2 = oos[(oos.problem == 'Q4-2') & oos.method.str.startswith('M2')].cvar95.iloc[0]
    cv3 = oos[(oos.problem == 'Q4-3') & oos.method.str.startswith('H3')].cvar95.iloc[0]
    for v in (cv2, cv3):
        ok(money(v) in paper, f'[A3] 论文正文缺少 CVaR95 = {money(v)}')

    boot = pd.read_csv(R / 'q4_cvar_bootstrap.csv')
    sel = boot[(boot.problem == 'Q4-3') & boot.variant.str.startswith('H3')]
    ok(len(sel) == 3, '[A4] Q4-3 bootstrap 行数不是 3（块长 7/14/28）')
    for _, r in sel.iterrows():
        for x in (r.ci_lo, r.ci_hi):
            s = f'{round(x):,}'.replace('-', '−')
            ok(s in paper.replace('+', ''), f'[A4] 论文正文缺少 bootstrap 区间端点 {s}（块长 {int(r.block_len)}）')
        ok(r.ci_lo <= 0 <= r.ci_hi, f'[A4] Q4-3 块长 {int(r.block_len)} 区间未跨零，正文结论需改写')
    ok('all valid starts' in str(boot.method.iloc[0]) and 'ceil and truncate' in str(boot.method.iloc[0]),
       '[A4] bootstrap 结果文件不是修复后的抽样实现')
    ok(int(boot.bootstrap_sample_size.iloc[0]) == 334, '[A4] bootstrap 重采样长度不是 334')

    chk = json.loads((R / 'q4_repair_checks.json').read_text(encoding='utf8'))
    for k in ('detail_invariance_stepwise_0', 'detail_invariance_stepwise_1e+12',
              'detail_invariance_net_refund_0', 'detail_invariance_net_refund_1e+12'):
        ok(float(chk[k]) == 0.0, f'[A5] detail 不变性检查未通过：{k}={chk[k]}')
    ok(chk['stepwise_ge_net_for_same_plan'] is True, '[A5] 同一计划下逐次≥净额的断言未通过')
    st = chk['rerun_stepwise_summary']
    for v in (st['total_cost'], st['adjusted_cost'],
              chk['fixed_main_plan_stepwise_cash'], chk['fixed_main_plan_stepwise_adjusted']):
        ok(money(v) in paper, f'[A5] 论文正文缺少逐次结算数值 {money(v)}')
    main_adj = float(oos[(oos.problem == 'Q4-3') & oos.method.str.startswith('H3')].adjusted_cost.iloc[0])
    ok(money(st['adjusted_cost'] - main_adj) in paper,
       f'[A5] 论文正文缺少逐次相对主口径的差额 {money(st["adjusted_cost"] - main_adj)}')

    rp = json.loads((R / 'q4_release_pricing.json').read_text(encoding='utf8'))
    ok(f'{rp["total_pairs"]:,}' in paper, f'[A6] 论文正文缺少有效配对数 {rp["total_pairs"]:,}')
    ok(f'{rp["total_hits"]:,}' in paper, f'[A6] 论文正文缺少满足判据的配对数 {rp["total_hits"]:,}')
    ok(f'{rp["share_percent"]:.2f}%' in paper, f'[A6] 论文正文缺少配对占比 {rp["share_percent"]:.2f}%')
    for clock, d in rp['per_release'].items():
        ok(f'{d["share_percent"]:.2f}%' in paper, f'[A6] 论文正文缺少 {clock} 的占比 {d["share_percent"]:.2f}%')

    sel_pol = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    for key, label in (('Q4-2_quantile', 'Q4-2 分位法'), ('Q4-3', 'Q4-3 HYBRID')):
        v = money(sel_pol[key]['fitness'])
        ok(v in paper, f'[A7] 论文正文缺少{label}的 1 月训练适应度 {v}')
    ok(money(sel_pol['Q4-3_unconstrained']['fitness']) in paper,
       '[A7] 论文正文缺少 Q4-3 超范围候选解的训练适应度')


# ---------------------------------------------------------------- B 措辞
FORBIDDEN = [
    ('479,390.29', '已作废的训练适应度'),
    ('479,320', '已作废的旧训练适应度'),
    ('13,460,722', '已作废的逐次结算错误值'),
    ('57.6%', '已作废的未配对比例'),
    ('0.6648', '已作废的单点比较值'),
    ('分段线性凸', '术语错误，应为分段线性凹'),
    ('凸性成立', '术语错误，应为凹性成立'),
    ('迭加', '错别字，应为叠加'),
    ('開始', '繁体字，应为开始'),
    ('文化基因', 'memetic 的不当译名，应为模因算法'),
    ('NP 困难的 MILP', '把一般类别结论套在具体实例上'),
    ('诚实性条款第', '审查清单式表达'),
    ('7 位有效数字', '不严谨表述'),
    ('本意相反', '题目未说明条款设计目的'),
    ('不存在需要权衡取舍', '把有限搜索现象写成理论结论'),
    ('没有 recourse。', '与 4.6.5 第 3 条矛盾'),
]


# 写作纪律段落会**引用**被禁用的措辞来说明"不要这么写"，这些行豁免
EXEMPT = ('不要', '不得', '作废', '旧稿', '已删除', '已整体作废', '不可再', '不能写成')


def check_wording():
    for f in DOCS:
        for i, line in enumerate(f.read_text(encoding='utf8').split('\n'), 1):
            if any(k in line for k in EXEMPT):
                continue
            for bad, why in FORBIDDEN:
                ok(bad not in line, f'[B1] {f.name}:{i} 残留「{bad}」——{why}')
    paper = PAPER.read_text(encoding='utf8')
    for must, why in [
        ('4.6.5 模型解释边界与稳健性说明', '4.6.5 小节标题未改'),
        ('分段线性凹', '未改用凹函数术语'),
        ('因此 Q4-2 仍包含 recourse', '未明确 Q4-2 仍有 recourse'),
        ('不构成严格独立盲测', '未按口径 B 声明盲测边界'),
        ('局部平稳近似', 'bootstrap 未说明季节性/局部平稳前提'),
        ('并非严格意义上的 VSS', '未声明 VSS 命名边界'),
        ('不构成严格定义的 EVPI', '未声明 EVPI 命名边界'),
        ('未观察到稳定且可辨识的成本—CVaR 权衡', 'NSGA-II 结论未收紧'),
        ('各自分支的 B0', '1.42%/0.46% 未写明基线'),
    ]:
        ok(must in paper, f'[B2] 论文正文缺少必需表述：{why}')
    # 口径 A 与口径 B 不得并存
    ok('在看到评价期表现之前就已冻结' not in paper and '不构成对评价期数据的窥探' not in paper,
       '[B3] 论文正文残留口径 A 表述，与口径 B 冲突')


# ---------------------------------------------------------------- C 结构
def check_structure():
    paper = PAPER.read_text(encoding='utf8')
    figdir = ROOT / '论文交付/2_论文部分/figures'
    refs = re.findall(r'\[\[FIG:([^|]+)\|([^\]]+)\]\]', paper)
    for src, cap in refs:
        ok((figdir / src.strip()).exists(), f'[C1] 插图缺失：{src}')
    nums = [re.match(r'^图 4-(\d+)', c.strip()) for _, c in refs]
    ok(all(nums), '[C1] 有图题未按「图 4-N」编号')
    seq = [int(m.group(1)) for m in nums if m]
    ok(seq == list(range(1, len(seq) + 1)), f'[C1] 图号不连续：{seq}')

    # HTML 必须是由当前 Markdown 生成的（重跑生成器后内容不变）
    before = HTML.read_bytes()
    subprocess.run([sys.executable, str(Path(__file__).with_name('q4_build_paper_html.py'))],
                   check=True, capture_output=True)
    ok(HTML.read_bytes() == before, '[C2] HTML 与 Markdown 不同步（已重新生成，请重新打包）')

    for f, tag, sheets in (('result4-2.xlsx', 'q4_2', [('计划购电量', 'total_cost')]),
                           ('result4-3.xlsx', 'q4_3', [('计划购电量', 'plan_cost'),
                                                       ('调整购电量', 'total_cost')])):
        wb = openpyxl.load_workbook(ROOT / f, read_only=True)
        daily = pd.read_csv(R / f'{tag}_daily.csv')
        for sheet, col in sheets:
            rows = [x for x in wb[sheet].iter_rows(values_only=True)]
            s = float(np.array([x[-1] for x in rows[1:]], float).sum())
            ok(abs(s - float(daily[col].sum())) < 1e-4,
               f'[C3] {f} 的「{sheet}」末列合计 {money(s)} 与 {tag}_daily.{col} 不符')
        summ = {r[0]: r[1] for r in wb['费用汇总与版本'].iter_rows(values_only=True)}
        ok('各表末列口径' in summ, f'[C3] {f} 的费用汇总页缺少「各表末列口径」说明')
        for item, col in (('计划购电费', 'plan_cost'), ('增购费', 'up_cost'),
                          ('减购净额', 'down_net_cost'), ('紧急购电费', 'emergency_cost')):
            ok(abs(float(summ[item]) - float(daily[col].sum())) < 1e-4,
               f'[C3] {f} 费用汇总页「{item}」与 {tag}_daily.{col} 不符')

    man = json.loads((R / 'fingerprint_manifest.json').read_text(encoding='utf8'))
    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    for group, d in (('code', man['code_sha256_16']), ('data', man['data_sha256_16'])):
        for n, h in d.items():
            ok(sha(ROOT / group / n) == h, f'[C4] {group}/{n} 指纹与清单不符（需重跑 q4_freeze.py）')
    for n, h in man.get('config_sha256_16', {}).items():
        ok(sha(ROOT / n) == h, f'[C4] {n} 指纹与清单不符（需重跑 q4_freeze.py）')
    figdir2 = ROOT / '论文交付/2_论文部分/figures'
    for n, h in man.get('figure_sha256_16', {}).items():
        ok(sha(figdir2 / n) == h, f'[C4] 插图 {n} 指纹与清单不符（需重跑 q4_freeze.py）')
    for n, h in man['result_sha256_16'].items():
        ok(sha(R / n) == h, f'[C4] results4/{n} 指纹与清单不符（需重跑 q4_freeze.py）')
    for n, h in man['deliverable_sha256_16'].items():
        ok(sha(ROOT / n) == h, f'[C4] {n} 指纹与清单不符（需重跑 q4_freeze.py）')
    for f in (HTML, BASE, ROOT / 'README_Q4.md'):
        ok(man['variant_id'] in f.read_text(encoding='utf8'), f'[C4] {f.name} 未标注 variant_id')


def main():
    check_numbers(); check_wording(); check_structure()
    print(f'共执行 {checks} 项检查')
    if fails:
        print(f'\n✗ 未通过 {len(fails)} 项：')
        for m in fails:
            print('  -', m)
        sys.exit(1)
    print('✓ 全部通过：文档数字、术语口径、交付结构与指纹均一致')


if __name__ == '__main__':
    main()
