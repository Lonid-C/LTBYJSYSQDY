"""边界自动延展（沿用改进版 P0-1 的做法）：若选中参数落在搜索边界上，扩大边界重搜，
直到最优解位于已测试区域内部；随后只重跑受影响的样本外分支并更新账本。

这是防止"最优解被搜索框截断"的标准步骤，不是事后调参：延展只用 1 月的同一适应度，
评价期仍然完全留出。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time, multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import hybrid_search as HSch
import q4_policy as Q4P
import run_q4_experiment as E

R = E.R
WIDE = {'Q4-2': [(0.0, 2.0), (0.005, 1.60), (0.002, 2.00), (0.0, 3.0)],
        'Q4-3': [(0.0, 2.0), (0.20, 1.40), (0.05, 1.80), (0.0, 2.5), (0.0, 700.0), (0.60, 2.60)]}
# 命令行给出的问题名会被**强制**重搜一轮（即使判据认为最优点在内部），用于确认更宽的框里没有更好的点。
FORCE = set(sys.argv[1:])
KEYS = {'Q4-2': Q4P.Q4_KEYS_2, 'Q4-3': Q4P.Q4_KEYS}
BASE = {'Q4-2': Q4P.Q4_BOUNDS_2, 'Q4-3': Q4P.Q4_BOUNDS}
TOL = 0.01          # 相对量程 1%——原来的 1e-3 且带 max(1.0, ·) 兜底过严，会把贴边的解判成内部

# 有**结构性下界**的参数：再往下没有意义，贴到该端点不算"被搜索框截断"，不需要也不能再延展。
#   temper=0  情景等权（权重指数为 0）
#   shrink_*=0 认为残差为零，即完全不确定性对冲
#   kappa=0   不计终端库存价值
#   threshold=0 不设接受门槛
STRUCTURAL_LOW = {'temper': 0.0, 'shrink_net': 0.0, 'shrink_price': 0.0,
                  'kappa': 0.0, 'threshold': 0.0}


def on_edge(x, bounds, keys, tol=TOL):
    """距任一端点不足量程 tol 即视为贴边；命中结构性下界的不计入（另行标注）。"""
    hits, structural = [], []
    for k, (v, (lo, hi)) in zip(keys, zip(x, bounds)):
        span = hi - lo
        low = (v - lo) <= tol * span
        high = (hi - v) <= tol * span
        if low and k in STRUCTURAL_LOW and abs(lo - STRUCTURAL_LOW[k]) < 1e-9:
            structural.append(k)
            low = False
        if low or high:
            hits.append(k)
    return hits, structural


def main():
    E.prepare()
    if E.N_WORKERS > 1:
        mp.set_start_method('fork', force=True)
        E._POOL = mp.Pool(E.N_WORKERS, initializer=E.boot)
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    log = []
    changed = {}
    for label, fitfn, mask, tag, seedv in (('Q4-2', E._fit2, (), 'q4_2', 20260931),
                                           ('Q4-3', E._fit3, (6, 12, 18), 'q4_3', 20260932)):
        x = np.array(sel[label]['x'], float)
        keys = KEYS[label]
        edge, struct = on_edge(x, BASE[label], keys)
        forced = label in FORCE
        log.append(dict(problem=label, step=0, bounds=str(BASE[label]), fitness=sel[label]['fitness'],
                        at_edge=','.join(edge), structural_corner=','.join(struct),
                        params=json.dumps(sel[label]['params'], ensure_ascii=False)))
        print(f'{label} 初始最优在边界的参数: {edge or "无"}'
              + (f'；命中结构性下界（不可延展）: {struct}' if struct else '')
              + ('（强制再延展一轮确认）' if forced and not edge else ''), flush=True)
        if not edge and not forced:
            continue
        fb = lambda X: E.batch(fitfn, X)
        b = WIDE[label]
        t0 = time.time()
        xg, fg, hg, ng = HSch.ga_optimize(fb, b, pop=16, gens=8, seed=seedv, log=True)
        xn, fn, hn, nn = HSch.nelder_mead(fb, xg, b, maxiter=25, log=True)
        bx, bf = (xn, fn) if fn <= fg else (xg, fg)
        edge2, struct2 = on_edge(bx, b, keys)
        log.append(dict(problem=label, step=1, bounds=str(b), fitness=bf, at_edge=','.join(edge2),
                        structural_corner=','.join(struct2),
                        params=json.dumps(dict(zip(keys, map(float, bx))), ensure_ascii=False)))
        print(f'{label} 延展后 best={bf:,.1f} x={np.round(bx,4)} 仍在边界: {edge2 or "无"}'
              + (f'；结构性下界: {struct2}' if struct2 else '') + f' ({time.time()-t0:.0f}s)', flush=True)
        if bf < sel[label]['fitness']:
            sel[label] = dict(x=list(map(float, bx)), fitness=float(bf),
                              evaluations=int(ng + nn), seconds=float(time.time() - t0),
                              keys=list(keys), params=dict(zip(keys, map(float, bx))),
                              bounds_extended=True, bounds=b,
                              **({'inert_fixed': dict(Q4P.INERT_Q4_2)} if label == 'Q4-2' else {}))
            changed[label] = (mask, tag)
    pd.DataFrame(log).to_csv(R / 'q4_boundary_extension.csv', index=False, encoding='utf-8-sig')
    (R / 'q4_selected_policy.json').write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding='utf8')
    if not changed:
        print('无需更新样本外结果', flush=True)
        return
    print('选参已更新，请运行 q4_finalize.py 重建全部下游账本', flush=True)
    return

    g = E.boot()
    res = pd.read_csv(R / 'q4_out_of_sample.csv')
    for label, (mask, tag) in changed.items():
        pref = 'H2' if label == 'Q4-2' else 'H3'
        upref = 'U2' if label == 'Q4-2' else 'U3'
        for prefix, mode in ((pref, 'forecast'), (upref, 'perfect')):
            pol = E.policy_from(sel[label]['x'], mode)
            t0 = time.time()
            df, iv, dc, tx = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'], pol,
                                    E.G['shape'], E.G['edges'], g['vE_ref'], mask=mask, start=31, end=365,
                                    initial=E.G[f'init_{tag}'], mode='stoch', detail=(prefix == pref))
            s = Q4P.summarize(df, method='', problem=label, seconds=time.time() - t0)
            m = (res.problem == label) & (res.method.str.startswith(prefix))
            for k, v in s.items():
                if k in res.columns and k not in ('method', 'problem', 'note'):
                    res.loc[m, k] = v
            print(f'  重跑 {label} {prefix}: adj={s["adjusted_cost"]:,.1f}', flush=True)
            if prefix == pref:
                df.to_csv(R / f'{tag}_daily.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
                df.to_csv(R / f'q4_daily_{label}_{prefix}.csv', index=False, encoding='utf-8-sig',
                          float_format='%.6f')
                iv.to_csv(R / f'{tag}_intervals.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
                if len(dc):
                    dc.to_csv(R / f'{tag}_decisions.csv', index=False, encoding='utf-8-sig',
                              float_format='%.6f')
                base = pd.read_csv(R / f'q4_daily_{label}_B0.csv')
                a, b2 = base.copy(), df.copy()
                for d in (a, b2):
                    d['month'] = pd.to_datetime(d.date).dt.month
                mm = pd.DataFrame(dict(month=sorted(a.month.unique())))
                mm['baseline'] = a.groupby('month').inventory_adjusted_cost.sum().values
                mm['hybrid'] = b2.groupby('month').inventory_adjusted_cost.sum().values
                mm['saving'] = mm.baseline - mm.hybrid
                mm['saving_percent'] = 100 * mm.saving / mm.baseline
                mm.to_csv(R / f'q4_monthly_{label}.csv', index=False, encoding='utf-8-sig',
                          float_format='%.6f')
            else:
                df.to_csv(R / f'q4_daily_{label}_{prefix}.csv', index=False, encoding='utf-8-sig',
                          float_format='%.6f')
    for prob in ('Q4-2', 'Q4-3'):
        m = res.problem == prob
        b0 = float(res[m & res.method.str.startswith('B0')].adjusted_cost.iloc[0])
        res.loc[m, 'gap_vs_baseline_percent'] = 100 * (res.loc[m, 'adjusted_cost'] - b0) / b0
    res.to_csv(R / 'q4_out_of_sample.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    summ = json.loads((R / 'q4_experiment_summary.json').read_text(encoding='utf8'))
    summ['selected'] = sel
    summ['out_of_sample'] = res.to_dict('records')
    summ['boundary_extension'] = log
    (R / 'q4_experiment_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float),
                                                  encoding='utf8')
    print('完成', flush=True)


if __name__ == '__main__':
    main()
