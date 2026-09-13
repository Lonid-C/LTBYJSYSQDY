"""第四问主实验：波动电价下的 Q4-2 与 Q4-3，沿用第三问的 HYBRID-4L 分层融合。

  L0  负荷预报（沿用第三问）+ **对数价格比预报**（新增，同一估计量，按预测精度做滚动原点 CV 选参）
  L1  **联合情景**：从同一历史日成对取 (净负荷残差, 价格对数残差)，k-medoids 带权削减
  L2  **电价随情景变化的两阶段情景随机 LP**：第一阶段按期望电价签合同，第二阶段按情景电价补救
  L2′ 终端库存价值 = 当日价格水平 × 归一化分段线性形状（一次 Bellman 回代拟合）
  L3  GA + Nelder-Mead（memetic），**只在 1 月三个滚动验证块上**评价，Q4-2 与 Q4-3 分别选参
  对照 分位数 + 点预测电价的确定性滚动 LP（把第三问改进版直接搬过来），以及
      「以为电价不波动」（附件 1 固定分时电价）与「完美价格预见」两条价格信息口径
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time, multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q3
from q3 import Params, replace, Bank, DATES
import hybrid_core as H
import hybrid_search as HSch
import q4_core as Q4C
import q4_policy as Q4P
import q4_config as CFG

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
FOLDS = ((7, 15), (15, 23), (23, 31))
N_SCEN = CFG.runtime_settings()['n_scen']
N_WORKERS = max(1, min(2, os.cpu_count() or 2))
G = {}


def boot():
    if G:
        return G
    L, P, F, tou, seed, V = Q4C.load4(R / 'price_data_audit.json')
    pars = CFG.inherited_parameters()
    sel = pars['selected']
    p = replace(Params(), **{k: sel[k] for k in ('alpha0', 'alpha_update', 'window', 'decay_days',
                                                 'bias_window', 'bias_decay', 'slope', 'weekday', 'epsilon')})
    p = CFG.apply_runtime(p)
    bank = Bank(L, P, F, seed, p)
    rb = H.RiskBank(bank, p.window)
    hpf = R / 'price_hp.json'
    if hpf.exists():
        hp = json.loads(hpf.read_text(encoding='utf8'))
    else:
        hp, _, _ = Q4C.select_price_hp(V, tou, FOLDS, out=R / 'price_forecast_cv.csv',
                                       out_bias=R / 'price_bias_cv.csv')
        hpf.write_text(json.dumps(hp, ensure_ascii=False, indent=2), encoding='utf8')
    pbank = Q4C.PriceBank(V, tou, hp)
    # 终端库存的**评价参考价**：必须是事前已知、且对所有方法相同的常数。
    # 旧实现取「2—12 月真实电价逐日最低值的均值」——那是评价期的真实数据，而这个值又进入
    # 1 月训练适应度的库存校正项，等于把评价期信息泄漏进了选参判据（逐日校正项平均 620 元，
    # 占日均费用约 1.37%，不是可忽略的量级）。改用附件 1 给定分时电价的最低值，
    # 它是题目直接给出的参考价，决策前已知，与附件 4 的任何实现值无关。
    vE_ref = float(tou.min() / p.eta)
    G.update(L=L, P=P, F=F, tou=tou, seed=seed, V=V, p=p, bank=bank, rb=rb, pbank=pbank,
             hp=hp, vE_ref=vE_ref)
    tvf = R / 'q4_terminal_shape.npz'
    if tvf.exists():
        z = np.load(tvf)
        G['shape'], G['edges'] = z['shape'], z['edges']
    return G


def prepare():
    g = boot()
    if 'shape' not in G:
        print('=== L2′ 终端价值形状（一次 Bellman 回代）===', flush=True)
        shape, edges, info = Q4C.fit_terminal_shape(g['bank'], g['pbank'], g['p'], list(range(7, 31)))
        np.savez(R / 'q4_terminal_shape.npz', shape=shape, edges=edges)
        pd.DataFrame(dict(soc_kwh=info['grid'], normalized_cost=info['normalized_cost'])).to_csv(
            R / 'q4_terminal_curve.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(dict(seg_lo=edges[:-1], seg_hi=edges[1:], shape=shape)).to_csv(
            R / 'q4_terminal_segments.csv', index=False, encoding='utf-8-sig')
        G['shape'], G['edges'] = shape, edges
        print('  shape', np.round(shape, 4), flush=True)
    # 1 月预热（用确定性对照口径，两个问题各一条，保证同一问题内各方法起点一致）
    for tag, mask in (('q4_2', ()), ('q4_3', (6, 12, 18))):
        f = R / f'warmup_{tag}.csv'
        if f.exists():
            G[f'warm_{tag}'] = pd.read_csv(f).soc_start.to_numpy(float)
            continue
        df, _, _, _ = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'],
                              Q4P.Q4Policy(n_scen=N_SCEN), G['shape'], G['edges'], g['vE_ref'],
                              mask=mask, start=0, end=31, initial=6000., mode='det')
        df.to_csv(f, index=False, encoding='utf-8-sig', float_format='%.6f')
        G[f'warm_{tag}'] = df.soc_start.to_numpy(float)
        print(f'  预热 {tag}: 2/1 初始库存 = {df.soc_end.iloc[-1]:.2f} kWh', flush=True)
    for tag in ('q4_2', 'q4_3'):
        G[f'init_{tag}'] = float(pd.read_csv(R / f'warmup_{tag}.csv').soc_end.iloc[-1])
    return G


def policy_from(x, price_mode='forecast', net_mode='forecast', n_scen=None):
    """4 维向量（Q4-2）按惰性参数的中性值补齐；6 维向量（Q4-3）原样使用。"""
    x = list(map(float, x))
    if len(x) == 2:          # Q4-2：temper, kappa（shrink 固定为 1，threshold/adj_premium 惰性）
        x = [x[0], Q4P.FIXED_Q4_2['shrink_net'], Q4P.FIXED_Q4_2['shrink_price'], x[1]]
    if len(x) == 4:
        x = x + [Q4P.INERT_Q4_2['threshold'], Q4P.INERT_Q4_2['adj_premium']]
    return Q4P.Q4Policy(n_scen=n_scen or N_SCEN, temper=x[0], shrink_net=x[1], shrink_price=x[2],
                        kappa=x[3], threshold=x[4], adj_premium=x[5],
                        price_mode=price_mode, net_mode=net_mode)


def fold_eval(pol, mask, tag, mode='stoch'):
    g = boot()
    warm = G[f'warm_{tag}']
    costs = []
    for s, e in FOLDS:
        df, _, _, _ = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'], pol,
                              G['shape'], G['edges'], g['vE_ref'], mask=mask, start=s, end=e,
                              initial=float(warm[s]), mode=mode)
        costs.append(float(df.inventory_adjusted_cost.sum()))
    return float(np.mean(costs)), costs


def _fit3(x):
    try:
        return fold_eval(policy_from(x), (6, 12, 18), 'q4_3')[0]
    except Exception:
        return 1e12


def _fit2(x):
    """Q4-2 只搜 4 维；惰性的 threshold / adj_premium 由 policy_from 补中性值。"""
    try:
        return fold_eval(policy_from(x), (), 'q4_2')[0]
    except Exception:
        return 1e12


_POOL = None


def batch(fn, X):
    X = np.atleast_2d(X)
    return [fn(x) for x in X] if _POOL is None else _POOL.map(fn, list(X))


def search(fitfn, label, seed, pop=16, gens=8, nm_iter=30, bounds=None):
    bounds = bounds or Q4P.Q4_BOUNDS
    fb = lambda X: batch(fitfn, X)
    hist = []
    t0 = time.time()
    x_ga, f_ga, h_ga, n_ga = HSch.ga_optimize(fb, bounds, pop=pop, gens=gens, seed=seed, log=True)
    hist += [dict(problem=label, **h) for h in h_ga]
    print(f'  [{label}] GA best={f_ga:,.1f} x={np.round(x_ga,4)} ({time.time()-t0:.0f}s)', flush=True)
    t1 = time.time()
    x_nm, f_nm, h_nm, n_nm = HSch.nelder_mead(fb, x_ga, bounds, maxiter=nm_iter, log=True)
    hist += [dict(problem=label, **h) for h in h_nm]
    print(f'  [{label}] NM best={f_nm:,.1f} x={np.round(x_nm,4)} ({time.time()-t1:.0f}s)', flush=True)
    best_x, best_f = (x_nm, f_nm) if f_nm <= f_ga else (x_ga, f_ga)
    return best_x, best_f, hist, n_ga + n_nm, time.time() - t0


def main():
    t_all = time.time()
    g = prepare()
    out = dict(price_hp=g['hp'], vE_ref=g['vE_ref'], n_scen=N_SCEN,
               terminal_shape=list(map(float, G['shape'])), terminal_edges=list(map(float, G['edges'])),
               init_q4_2=G['init_q4_2'], init_q4_3=G['init_q4_3'])
    print('=== 情景数（SAA）收敛扫描 ===', flush=True)
    conv = []
    for S in (6, 8, 10, 12, 16, 20):
        for label, mask, tag in (('Q4-3', (6, 12, 18), 'q4_3'), ('Q4-2', (), 'q4_2')):
            t0 = time.time()
            pol = policy_from([0.7, 0.7, 0.5, 1.1, 300.0, 1.15] if label == 'Q4-3' else [0.7, 0.7, 0.5, 1.1],
                              n_scen=S)
            m, blocks = fold_eval(pol, mask, tag)
            conv.append(dict(problem=label, n_scen=S, train_fitness=m, worst_block=max(blocks),
                             seconds=time.time() - t0))
            print(f'  {label} S={S:2d} 训练适应度={m:,.1f} ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(conv).to_csv(R / 'q4_saa_convergence.csv', index=False, encoding='utf-8-sig',
                              float_format='%.6f')
    out['saa_convergence'] = conv

    print('=== L3 参数寻优（严格因果：仅 1 月三个验证块）===', flush=True)
    hists, sel = [], {}
    base3 = fold_eval(Q4P.Q4Policy(n_scen=N_SCEN), (6, 12, 18), 'q4_3', mode='det')
    base2 = fold_eval(Q4P.Q4Policy(n_scen=N_SCEN), (), 'q4_2', mode='det')
    print(f'  对照基线训练适应度 Q4-3 {base3[0]:,.1f} / Q4-2 {base2[0]:,.1f}', flush=True)
    out['baseline_training_fitness'] = dict(q4_3=base3[0], q4_3_folds=base3[1],
                                            q4_2=base2[0], q4_2_folds=base2[1])
    for label, fitfn, seedv, bnds, keys in (
            ('Q4-3', _fit3, 20260921, Q4P.Q4_BOUNDS, Q4P.Q4_KEYS),
            ('Q4-2', _fit2, 20260922, Q4P.Q4_BOUNDS_2, Q4P.Q4_KEYS_2)):
        x, f, h, nev, sec = search(fitfn, label, seedv, bounds=bnds)
        hists += h
        sel[label] = dict(x=list(map(float, x)), fitness=f, evaluations=int(nev), seconds=sec,
                          keys=list(keys), params=dict(zip(keys, map(float, x))))
        if label == 'Q4-2':
            sel[label]['inert_fixed'] = dict(Q4P.INERT_Q4_2)
        print(f'  选定 {label}: {sel[label]["params"]}  训练适应度 {f:,.1f}', flush=True)
    pd.DataFrame(hists).to_csv(R / 'q4_search_convergence.csv', index=False, encoding='utf-8-sig')
    out['selected'] = sel
    (R / 'q4_selected_policy.json').write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding='utf8')

    print('=== 样本外全年回测 2025-02-01 — 12-31 ===', flush=True)
    rows, store = [], {}

    def add(name, problem, df, sec, note=''):
        s = Q4P.summarize(df, method=name, problem=problem, seconds=sec, note=note)
        rows.append(s); store[f'{problem}|{name}'] = df
        print(f'  {name}: adj={s["adjusted_cost"]:,.1f} emg={s["emergency_kwh"]:,.0f} {sec:.0f}s', flush=True)

    def go(pol, mask, tag, mode, name, problem, note='', detail=False):
        t0 = time.time()
        df, iv, dc, tx = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], g['p'], pol,
                                G['shape'], G['edges'], g['vE_ref'], mask=mask, start=31, end=365,
                                initial=G[f'init_{tag}'], mode=mode, detail=detail)
        add(name, problem, df, time.time() - t0, note)
        return df, iv, dc

    p3 = policy_from(sel['Q4-3']['x'])
    p2 = policy_from(sel['Q4-2']['x'])
    go(Q4P.Q4Policy(n_scen=N_SCEN), (6, 12, 18), 'q4_3', 'det', 'B0 分位法+点预测电价（基线）', 'Q4-3')
    go(Q4P.Q4Policy(n_scen=N_SCEN, price_mode='tou'), (6, 12, 18), 'q4_3', 'det',
       'B1 分位法+当作固定分时电价', 'Q4-3', note='误以为电价不波动')
    df3, iv3, dc3 = go(p3, (6, 12, 18), 'q4_3', 'stoch', 'H3 HYBRID-4L（联合情景随机 LP）', 'Q4-3', detail=True)
    go(policy_from(sel['Q4-3']['x'], 'perfect'), (6, 12, 18), 'q4_3', 'stoch',
       'U3 完美价格预见（上界）', 'Q4-3', note='仅作价格信息价值上界，不可实现')
    go(p3, (), 'q4_3', 'stoch', 'A3 HYBRID 不做日内更新', 'Q4-3')
    go(Q4P.Q4Policy(n_scen=N_SCEN), (), 'q4_2', 'det', 'B0 分位法+点预测电价（基线）', 'Q4-2')
    go(Q4P.Q4Policy(n_scen=N_SCEN, price_mode='tou'), (), 'q4_2', 'det',
       'B1 分位法+当作固定分时电价', 'Q4-2', note='误以为电价不波动')
    df2, iv2, dc2 = go(p2, (), 'q4_2', 'stoch', 'H2 HYBRID-4L（联合情景随机 LP）', 'Q4-2', detail=True)
    go(policy_from(sel['Q4-2']['x'], 'perfect'), (), 'q4_2', 'stoch',
       'U2 完美价格预见（上界）', 'Q4-2', note='仅作价格信息价值上界，不可实现')
    go(policy_from(sel['Q4-2']['x'], 'perfect', 'perfect', n_scen=1), (), 'q4_2', 'stoch',
       'W2 完美价格+完美净负荷（全知下界）', 'Q4-2', note='wait-and-see 事后最优，理论下界')
    go(policy_from(sel['Q4-3']['x'], 'perfect', 'perfect', n_scen=1), (6, 12, 18), 'q4_3', 'stoch',
       'W3 完美价格+完美净负荷（全知下界）', 'Q4-3', note='wait-and-see 事后最优，理论下界')

    if len(dc3):
        pass
    iv3.to_csv(R / 'q4_3_intervals.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    iv2.to_csv(R / 'q4_2_intervals.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    dc3.to_csv(R / 'q4_3_decisions.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    df3.to_csv(R / 'q4_3_daily.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    df2.to_csv(R / 'q4_2_daily.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    for key, d in store.items():
        prob, nm = key.split('|', 1)
        d.to_csv(R / f'q4_daily_{prob}_{nm.split()[0]}.csv', index=False, encoding='utf-8-sig',
                 float_format='%.6f')
    res = pd.DataFrame(rows)
    for prob in ('Q4-2', 'Q4-3'):
        m = res.problem == prob
        base = float(res[m & res.method.str.startswith('B0')].adjusted_cost.iloc[0])
        res.loc[m, 'gap_vs_baseline_percent'] = 100 * (res.loc[m, 'adjusted_cost'] - base) / base
    res.to_csv(R / 'q4_out_of_sample.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    out['out_of_sample'] = res.to_dict('records')

    # 逐月分解（各问题：HYBRID 相对同问题基线）
    for prob, dfb, dfh in (('Q4-2', store['Q4-2|B0 分位法+点预测电价（基线）'], df2),
                           ('Q4-3', store['Q4-3|B0 分位法+点预测电价（基线）'], df3)):
        a, b = dfb.copy(), dfh.copy()
        for d in (a, b):
            d['month'] = pd.to_datetime(d.date).dt.month
        m = pd.DataFrame(dict(month=sorted(a.month.unique())))
        m['baseline'] = a.groupby('month').inventory_adjusted_cost.sum().values
        m['hybrid'] = b.groupby('month').inventory_adjusted_cost.sum().values
        m['saving'] = m.baseline - m.hybrid
        m['saving_percent'] = 100 * m.saving / m.baseline
        m.to_csv(R / f'q4_monthly_{prob}.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f'--- {prob} 逐月 ---\n' + m.to_string(index=False), flush=True)
    # ---- 信息价值拆分：VSS（随机规划价值）/ EVPI（价格信息）/ 剩余（净负荷信息） ----
    info = []
    for prob, hp_ in (('Q4-2', 'H2'), ('Q4-3', 'H3')):
        def pick(pref):
            g2 = res[(res.problem == prob) & (res.method.str.startswith(pref))]
            return float(g2.adjusted_cost.iloc[0]) if len(g2) else float('nan')
        b0, hh, uu, ww, tt = pick('B0'), pick(hp_), pick('U' + prob[-1]), pick('W' + prob[-1]), pick('B1')
        info.append(dict(problem=prob, baseline_deterministic=b0, hybrid_stochastic=hh,
                         perfect_price=uu, perfect_everything=ww, price_assumed_fixed=tt,
                         VSS_stochastic_value=b0 - hh, EVPI_price=hh - uu,
                         value_of_net_load_info=uu - ww, total_gap_to_full_information=hh - ww,
                         VSS_percent=100 * (b0 - hh) / b0, EVPI_percent=100 * (hh - uu) / hh,
                         full_gap_percent=100 * (hh - ww) / hh))
    pd.DataFrame(info).to_csv(R / 'q4_information_value.csv', index=False, encoding='utf-8-sig',
                              float_format='%.6f')
    out['information_value'] = info
    print(pd.DataFrame(info)[['problem', 'VSS_stochastic_value', 'EVPI_price',
                              'value_of_net_load_info']].to_string(index=False), flush=True)

    # ---- 尾部风险：CVaR95 的配对区块 bootstrap ----
    rng = np.random.default_rng(20260941)
    boot_rows = []
    for prob, base_df, hyb_df in (('Q4-2', store['Q4-2|B0 分位法+点预测电价（基线）'], df2),
                                  ('Q4-3', store['Q4-3|B0 分位法+点预测电价（基线）'], df3)):
        a = base_df.total_cost.to_numpy(); b = hyb_df.total_cost.to_numpy()
        cv = lambda x, q=.95: x[x >= np.quantile(x, q)].mean()
        obs = cv(b) - cv(a)
        n, Lb = len(a), 7
        nb = n // Lb
        dd = []
        for _ in range(4000):
            st = rng.integers(0, n - Lb, nb)
            idx = np.concatenate([np.arange(t0, t0 + Lb) for t0 in st])
            dd.append(cv(b[idx]) - cv(a[idx]))
        dd = np.array(dd)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        boot_rows.append(dict(problem=prob, cvar95_baseline=cv(a), cvar95_hybrid=cv(b),
                              diff=obs, diff_percent=100 * obs / cv(a), ci_lo=lo, ci_hi=hi,
                              p_worse=float((dd > 0).mean()), block_len=Lb, resamples=4000))
        print(f'  {prob} CVaR95 差 {obs:+.1f} 元，95% CI [{lo:+.0f}, {hi:+.0f}]，'
              f'P(更差)={float((dd > 0).mean()):.3f}', flush=True)
    pd.DataFrame(boot_rows).to_csv(R / 'q4_cvar_bootstrap.csv', index=False, encoding='utf-8-sig',
                                   float_format='%.6f')
    out['cvar_bootstrap'] = boot_rows

    out['elapsed_minutes'] = (time.time() - t_all) / 60
    (R / 'q4_experiment_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                                  encoding='utf8')
    print('=== 完成，用时 %.1f 分钟 ===' % ((time.time() - t_all) / 60), flush=True)


if __name__ == '__main__':
    prepare()
    if N_WORKERS > 1:
        mp.set_start_method('fork', force=True)
        _POOL = mp.Pool(N_WORKERS, initializer=boot)
        globals()['_POOL'] = _POOL
    main()
