"""按 results4/q4_selected_policy.json 的当前选参，**重跑全部样本外分支并重建全部下游账本**。

寻优（可能分几轮：初次搜索、基础框加宽、边界延展）结束后跑这一个脚本，
避免"只更新了一部分 CSV"造成的账本不自洽。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q4_policy as Q4P
import q4_config as CFG
import run_q4_experiment as E

R = E.R


def main():
    t0all = time.time()
    g = E.prepare()
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    rows, store = [], {}
    detail_keep = {}

    def go(pol, mask, tag, mode, name, problem, note='', detail=False):
        t0 = time.time()
        df, iv, dc, tx = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'], pol,
                                E.G['shape'], E.G['edges'], g['vE_ref'], mask=mask, start=31, end=365,
                                initial=E.G[f'init_{tag}'], mode=mode, detail=detail)
        s = Q4P.summarize(df, method=name, problem=problem, seconds=time.time() - t0, note=note)
        rows.append(s); store[f'{problem}|{name}'] = df
        if detail:
            detail_keep[problem] = (df, iv, dc, tx, tag)
        print(f'  {name} [{problem}]: adj={s["adjusted_cost"]:,.1f} emg={s["emergency_kwh"]:,.0f} '
              f'{time.time()-t0:.0f}s', flush=True)
        return df

    p3 = E.policy_from(sel['Q4-3']['x'])
    p2h = E.policy_from(sel['Q4-2']['x'])
    a2 = float(sel['Q4-2_quantile']['alpha'])
    a3q = float(sel['Q4-3_quantile']['alpha'])
    Q = lambda a, **kw: Q4P.Q4Policy(n_scen=E.N_SCEN, alpha=a, **kw)
    print('=== 重跑样本外全部分支 ===', flush=True)

    # ---------------- Q4-3：主策略为 HYBRID ----------------
    go(Q(0.60), (6, 12, 18), 'q4_3', 'det', 'B0 分位法 α=0.60（沿用第三问标定）', 'Q4-3')
    if abs(a3q - 0.60) > 1e-9:
        go(Q(a3q), (6, 12, 18), 'q4_3', 'det', f'M3 分位法 α={a3q:.3f}（1 月重标定·基准）', 'Q4-3')
    go(Q(a3q, price_mode='tou'), (6, 12, 18), 'q4_3', 'det',
       'B1 分位法+当作固定分时电价', 'Q4-3', note='误以为电价不波动')
    df3 = go(p3, (6, 12, 18), 'q4_3', 'stoch', 'H3 HYBRID-4L（联合情景随机 LP·主策略）', 'Q4-3', detail=True)
    go(E.policy_from(sel['Q4-3']['x'], 'perfect'), (6, 12, 18), 'q4_3', 'stoch',
       'U3 主策略+完美电价', 'Q4-3', note='冻结策略的完美价格替换对照；非理论上界')
    go(E.policy_from(sel['Q4-3']['x'], 'perfect', 'perfect', n_scen=1), (6, 12, 18), 'q4_3', 'stoch',
       'W3 滚动视野完美信息对照', 'Q4-3',
       note='完美电价+完美净负荷，但仍受 24h 滚动视野与终端代理约束——不是下界')
    go(p3, (), 'q4_3', 'stoch', 'A3 HYBRID 不做日内更新', 'Q4-3')
    if 'Q4-3_unconstrained' in sel:
        go(E.policy_from(sel['Q4-3_unconstrained']['x']), (6, 12, 18), 'q4_3', 'stoch',
           'X3 1月无约束最优（shrink 超出建模含义范围·未采纳）', 'Q4-3',
           note='1 月适应度更低但 shrink_net=0.215；并列报告以验证不采纳的决定')

    # ---------------- Q4-2：主策略为分位法（HYBRID 未能胜出，如实并列） ----------------
    go(Q(0.60), (), 'q4_2', 'det', 'B0 分位法 α=0.60（沿用第三问标定）', 'Q4-2')
    df2 = go(Q(a2), (), 'q4_2', 'det', f'M2 分位法 α={a2:.3f}（1 月重标定·主策略）', 'Q4-2', detail=True)
    go(Q(a2, price_mode='tou'), (), 'q4_2', 'det',
       'B1 分位法+当作固定分时电价', 'Q4-2', note='误以为电价不波动')
    go(p2h, (), 'q4_2', 'stoch', 'H2 HYBRID-4L（联合情景随机 LP）', 'Q4-2',
       note='shrink 按结构性论证固定为 1；该口径下未能胜过分位法，如实报告')
    go(Q(a2, price_mode='perfect'), (), 'q4_2', 'det', 'U2 主策略+完美电价', 'Q4-2',
       note='冻结策略的完美价格替换对照；非理论上界')
    go(E.policy_from(sel['Q4-2']['x'], 'perfect', 'perfect', n_scen=1), (), 'q4_2', 'stoch',
       'W2 滚动视野完美信息对照', 'Q4-2',
       note='完美电价+完美净负荷，但仍受 24h 滚动视野与终端代理约束——不是下界')

    for prob, (df, iv, dc, tx, tag) in detail_keep.items():
        df.to_csv(R / f'{tag}_daily.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        iv.to_csv(R / f'{tag}_intervals.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        if len(dc):
            dc.to_csv(R / f'{tag}_decisions.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        tx.to_csv(R / f'{tag}_transactions.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f'  {tag} 逐笔交易 {len(tx)} 条', flush=True)
    for key, d in store.items():
        prob, nm = key.split('|', 1)
        d.to_csv(R / f'q4_daily_{prob}_{nm.split()[0]}.csv', index=False, encoding='utf-8-sig',
                 float_format='%.6f')

    res = pd.DataFrame(rows)
    # 参照点统一取「1 月重标定过的分位法」——两个问题都拿公平校准过的确定性基线作分母
    ref = {}
    for prob in ('Q4-2', 'Q4-3'):
        m = res.problem == prob
        pref = 'M' + prob[-1]
        gref = res[m & res.method.str.startswith(pref)]
        if not len(gref):
            gref = res[m & res.method.str.startswith('B0')]
        base = float(gref.adjusted_cost.iloc[0])
        ref[prob] = base
        res.loc[m, 'gap_vs_baseline_percent'] = 100 * (res.loc[m, 'adjusted_cost'] - base) / base
    res.to_csv(R / 'q4_out_of_sample.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

    # 逐月分解：一律以 B0（沿用第三问 α=0.60 的分位法）为参照，回答"这一轮工作逐月买到了什么"。
    # 注意不能用 refname（Q4-2 的参照策略本身就是主策略，会变成自己跟自己比）。
    for prob, dfh in (('Q4-2', df2), ('Q4-3', df3)):
        a = store[next(k for k in store if k.startswith(prob + '|B0'))].copy()
        b = dfh.copy()
        for d in (a, b):
            d['month'] = pd.to_datetime(d.date).dt.month
        m = pd.DataFrame(dict(month=sorted(a.month.unique())))
        m['baseline'] = a.groupby('month').inventory_adjusted_cost.sum().values
        m['main'] = b.groupby('month').inventory_adjusted_cost.sum().values
        m['saving'] = m.baseline - m.main
        m['saving_percent'] = 100 * m.saving / m.baseline
        m.to_csv(R / f'q4_monthly_{prob}.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

    # All downstream risk statistics use the same exact empirical-CVaR implementation.
    from q4_refresh_statistics import refresh
    info, bt = refresh()
    res = pd.read_csv(R / 'q4_out_of_sample.csv')

    cfg = CFG.write(g['p'], g['tou'], g['hp'], E.N_SCEN, g['vE_ref'], selected=sel)
    print(f"  variant_id = {cfg['variant_id']}", flush=True)
    summ = json.loads((R / 'q4_experiment_summary.json').read_text(encoding='utf8'))
    summ['variant_id'] = cfg['variant_id']
    summ.update(selected=sel, out_of_sample=res.to_dict('records'), information_value=info,
                cvar_bootstrap=bt, finalized_minutes=(time.time() - t0all) / 60)
    (R / 'q4_experiment_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float),
                                                  encoding='utf8')
    print('=== 重建完成，用时 %.1f 分钟 ===' % ((time.time() - t0all) / 60), flush=True)


if __name__ == '__main__':
    import multiprocessing as mp
    E.prepare()
    main()
