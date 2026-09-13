# -*- coding: utf-8 -*-
"""复现性抽查：用包内唯一配置入口重跑若干天，与冻结的逐日账本逐项比对。

第四问交付包不含第三问的 results/ 目录，继承参数改由 config/q3_inherited_parameters.json
提供（见 q4_config.inherited_parameters）。本脚本用来证明该配置确实复现了冻结结果，
而不是"看起来像"的一组数。默认抽查评价期前 10 天；--days 可加大抽查规模。
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import q4_policy as Q4P
import run_q4_experiment as E

R = E.R
COLS = ['plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost',
        'total_cost', 'inventory_adjusted_cost', 'plan_kwh', 'emergency_kwh', 'soc_end']


def run_branch(g, sel, branch, days):
    """与 q4_finalize.go() 完全相同的调用口径，只把 end 截短到抽查天数。"""
    end = 31 + days
    if branch == 'Q4-3':
        pol, mask, tag, mode = E.policy_from(sel['Q4-3']['x']), (6, 12, 18), 'q4_3', 'stoch'
    else:
        pol = Q4P.Q4Policy(n_scen=E.N_SCEN, alpha=float(sel['Q4-2_quantile']['alpha']))
        mask, tag, mode = (), 'q4_2', 'det'
    df, *_ = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'], pol,
                        E.G['shape'], E.G['edges'], g['vE_ref'], mask=mask, start=31, end=end,
                        initial=E.G[f'init_{tag}'], mode=mode)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=10)
    a = ap.parse_args()
    g = E.prepare()
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    bad, report = [], []
    for branch, frozen in (('Q4-2', 'q4_2_daily.csv'), ('Q4-3', 'q4_3_daily.csv')):
        got = run_branch(g, sel, branch, a.days)
        want = pd.read_csv(R / frozen).head(a.days)
        assert got.date.tolist() == want.date.tolist(), f'{branch} 日期不对齐'
        err = {c: float(np.abs(got[c].to_numpy() - want[c].to_numpy()).max()) for c in COLS}
        worst = max(err, key=err.get)
        report.append(f'  {branch}：{a.days} 天，最大逐列绝对误差 {err[worst]:.3e}（{worst}）')
        if err[worst] > 1e-4:
            bad.append(dict(branch=branch, errors=err))
    print(f'用 config/q3_inherited_parameters.json 复跑评价期前 {a.days} 天：')
    print('\n'.join(report))
    if bad:
        print('\n✗ 复现失败：', json.dumps(bad, ensure_ascii=False, default=float))
        sys.exit(1)
    print('✓ 复现通过：包内配置入口与冻结账本一致')


if __name__ == '__main__':
    main()
