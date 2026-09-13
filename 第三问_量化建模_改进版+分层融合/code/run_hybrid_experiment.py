"""第三问：分层融合算法（HYBRID-4L）主实验。

流程
  0  载入主模型已选定的参数与预报库（L0 预测层完全沿用，不重新调）
  1  L2 终端价值层：一次 Bellman 回代拟合分段线性凸终端库存价值 V(E)
  2  情景数收敛研究（SAA 收敛）：n_scen 作为数值离散参数，用目标收敛而非账单来定
  3  L3 参数寻优：GA / DE / SA 同预算对照 + Nelder-Mead 局部精修（memetic GA）
     —— 适应度只用评价期之前的 1 月三个验证块（严格因果）
  4  样本外全年回测（2/1—12/31）：HYBRID 与基线、消融分支并列
  5  L4 NSGA-II：成本—CVaR95 Pareto 前沿，前沿解再做样本外评估
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from pathlib import Path
import sys, json, time, pickle, multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q3
from q3 import Params, replace, load, Bank, DATES
import hybrid_core as H
import hybrid_policy as HP
import hybrid_search as HSch
import method_comparison as mc

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
N_SCEN = 12                    # 由 SAA 收敛研究确定（见 hybrid_saa_convergence.csv）
FOLDS = ((7, 15), (15, 23), (23, 31))
SEARCH_KEYS = ('temper', 'shrink', 'kappa', 'threshold', 'adj_premium')
SEARCH_BOUNDS = [(0.0, 2.0), (0.60, 1.30), (0.0, 2.0), (0.0, 500.0), (0.80, 2.50)]
N_WORKERS = max(1, min(2, (os.cpu_count() or 2)))

G = {}


def boot():
    if G:
        return G
    L, P, F, price, seed = load()
    pars = json.loads((R / 'parameters.json').read_text(encoding='utf8'))
    sel = pars['selected']
    p = replace(Params(), **{k: sel[k] for k in ('alpha0', 'alpha_update', 'window', 'decay_days',
                                                 'bias_window', 'bias_decay', 'slope', 'weekday', 'epsilon')})
    bank = Bank(L, P, F, seed, p)
    rb = H.RiskBank(bank, p.window)
    warm = pd.read_csv(R / 'warmup_january_selected.csv')
    G.update(L=L, P=P, F=F, price=price, seed=seed, p=p, bank=bank, rb=rb,
             initial_feb1=float(pars['initial_feb1']),
             soc_start=warm.soc_start.to_numpy(float),
             vE=float(price.min() / p.eta))
    tvp = R / 'hybrid_terminal_value.pkl'
    if tvp.exists():
        G['tv'] = pickle.loads(tvp.read_bytes())
    return G


def policy_from(x, n_scen=N_SCEN, cvar_beta=0.0, cvar_alpha=0.90):
    return HP.Policy(n_scen=n_scen, temper=float(x[0]), shrink=float(x[1]), tail_relax=1.0,
                     cvar_beta=cvar_beta, cvar_alpha=cvar_alpha, kappa=float(x[2]),
                     threshold=float(x[3]), adj_premium=float(x[4]))


def fold_eval(pol, folds=FOLDS, n_scen=None):
    g = boot()
    if n_scen is not None:
        pol = HP.Policy(**{**pol.__dict__, 'n_scen': n_scen})
    costs, emg, cv = [], [], []
    for s, e in folds:
        df, _, _ = HP.run_stoch2(g['L'], g['P'], g['price'], g['bank'], g['rb'], g['p'], pol, g['tv'],
                                 mask=(6, 12, 18), start=s, end=e, initial=float(g['soc_start'][s]))
        costs.append(float(df.inventory_adjusted_cost.sum()))
        emg.append(float(df.emergency_kwh.sum()))
        cv.append(float(df.total_cost[df.total_cost >= df.total_cost.quantile(.90)].mean()))
    return dict(mean=float(np.mean(costs)), worst=float(np.max(costs)), blocks=costs,
                emergency=float(np.sum(emg)), cvar=float(np.mean(cv)))


def _fit_one(x):
    try:
        return fold_eval(policy_from(x))['mean']
    except Exception:
        return 1e12


def _mo_one(x):
    try:
        r = fold_eval(policy_from(x[:5], cvar_beta=float(x[5]), cvar_alpha=float(x[6])))
        return [r['mean'], r['cvar']]
    except Exception:
        return [1e12, 1e12]


_POOL = None


def batch(fn, X):
    X = np.atleast_2d(X)
    global _POOL
    if _POOL is None:
        return [fn(x) for x in X]
    return _POOL.map(fn, list(X))


def prepare_tv():
    """L2 终端价值层：必须在创建进程池之前完成，子进程 fork 时才能继承。"""
    g = boot()
    print('=== L2 终端库存价值函数（一次 Bellman 回代）===', flush=True)
    t0 = time.time()
    tv, info = H.fit_terminal_value(g['bank'], g['price'], g['p'], days=list(range(7, 31)), K=8, ngrid=25)
    (R / 'hybrid_terminal_value.pkl').write_bytes(pickle.dumps(tv))
    G['tv'] = tv
    pd.DataFrame(dict(soc_kwh=info['grid'], future_cost=info['cost_curve'])).to_csv(
        R / 'hybrid_terminal_value_curve.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(dict(seg_lo=tv.edges[:-1], seg_hi=tv.edges[1:], marginal_value=tv.marginals)).to_csv(
        R / 'hybrid_terminal_value_segments.csv', index=False, encoding='utf-8-sig')
    print(f'  v_E(linear)={g["vE"]:.4f}  分段边际={np.round(tv.marginals, 4)}  {time.time() - t0:.1f}s', flush=True)
    return tv


def main():
    t_all = time.time()
    g = boot()
    p, price = g['p'], g['price']
    out = {}

    tv = G['tv']
    out['terminal_value'] = dict(vE_linear=g['vE'], edges=list(tv.edges), marginals=list(tv.marginals))

    # ---------------- 2  SAA 情景数收敛 ----------------
    print('=== 情景数（SAA）收敛研究 ===', flush=True)
    conv = []
    for S in (3, 5, 8, 12, 16, 20, 27):
        t0 = time.time()
        r = fold_eval(HP.Policy(n_scen=S), folds=(FOLDS[0],))
        conv.append(dict(n_scen=S, fold1_adjusted_cost=r['mean'], emergency_kwh=r['emergency'],
                         seconds=time.time() - t0))
        print(f'  S={S:2d} cost={r["mean"]:,.1f} {time.time()-t0:.1f}s', flush=True)
    pd.DataFrame(conv).to_csv(R / 'hybrid_saa_convergence.csv', index=False, encoding='utf-8-sig')
    out['saa_convergence'] = conv

    # ---------------- 3  L3 参数寻优（算法组合） ----------------
    print(f'=== L3 参数寻优（GA / DE / SA + Nelder-Mead 精修），n_scen={N_SCEN} ===', flush=True)
    fb = lambda X: batch(_fit_one, X)
    runs, hists = {}, []
    t0 = time.time(); x_ga, f_ga, h_ga, n_ga = HSch.ga_optimize(fb, SEARCH_BOUNDS, pop=16, gens=10, seed=20260912, log=True)
    runs['GA'] = dict(x=x_ga.tolist(), fitness=f_ga, evaluations=n_ga, seconds=time.time() - t0); hists += h_ga
    print(f'  GA best={f_ga:,.1f} x={np.round(x_ga,4)}', flush=True)
    t0 = time.time(); x_de, f_de, h_de, n_de = HSch.de_optimize(fb, SEARCH_BOUNDS, pop=16, gens=10, seed=20260913, log=True)
    runs['DE'] = dict(x=x_de.tolist(), fitness=f_de, evaluations=n_de, seconds=time.time() - t0); hists += h_de
    print(f'  DE best={f_de:,.1f} x={np.round(x_de,4)}', flush=True)
    t0 = time.time(); x_sa, f_sa, h_sa, n_sa = HSch.sa_optimize(fb, SEARCH_BOUNDS, steps=12, chains=8, seed=20260914, log=True)
    runs['SA'] = dict(x=x_sa.tolist(), fitness=f_sa, evaluations=n_sa, seconds=time.time() - t0); hists += h_sa
    print(f'  SA best={f_sa:,.1f} x={np.round(x_sa,4)}', flush=True)
    seed_x = min([(f_ga, x_ga), (f_de, x_de), (f_sa, x_sa)], key=lambda t: t[0])[1]
    t0 = time.time(); x_nm, f_nm, h_nm, n_nm = HSch.nelder_mead(fb, seed_x, SEARCH_BOUNDS, maxiter=35, log=True)
    runs['MEMETIC(GA/DE/SA best + NM)'] = dict(x=x_nm.tolist(), fitness=f_nm, evaluations=n_nm,
                                               seconds=time.time() - t0); hists += h_nm
    print(f'  NM best={f_nm:,.1f} x={np.round(x_nm,4)}', flush=True)
    pd.DataFrame(hists).to_csv(R / 'hybrid_search_convergence.csv', index=False, encoding='utf-8-sig')
    best_x = min([(f_ga, x_ga), (f_de, x_de), (f_sa, x_sa), (f_nm, x_nm)], key=lambda t: t[0])[1]
    best_f = min(f_ga, f_de, f_sa, f_nm)
    pol_star = policy_from(best_x)
    out['search'] = runs
    out['selected_policy'] = {**pol_star.__dict__, 'training_fitness': best_f,
                              'keys': list(SEARCH_KEYS), 'bounds': SEARCH_BOUNDS}
    detail = fold_eval(pol_star)
    out['selected_policy']['fold_costs'] = detail['blocks']
    pd.DataFrame([dict(optimizer=k, **{kk: v[kk] for kk in ('fitness', 'evaluations', 'seconds')},
                       **dict(zip(SEARCH_KEYS, v['x']))) for k, v in runs.items()]).to_csv(
        R / 'hybrid_optimizer_portfolio.csv', index=False, encoding='utf-8-sig')
    print('SELECTED', pol_star, flush=True)
    (R / 'hybrid_selected_policy.json').write_text(json.dumps(out['selected_policy'], ensure_ascii=False,
                                                              indent=2, default=float), encoding='utf8')

    # ---------------- 4  样本外全年回测 ----------------
    print('=== 样本外全年回测 2025-02-01 — 12-31 ===', flush=True)
    init = g['initial_feb1']
    tv_const = H.TerminalValue.constant(g['vE'])
    rows, daily_store = [], {}

    def add(name, df, sec, note=''):
        s = H.summarize(df, method=name, seconds=sec, note=note)
        rows.append(s); daily_store[name] = df
        print(f'  {name}: adj={s["adjusted_cost"]:,.1f} emg={s["emergency_kwh"]:,.0f} {sec:.1f}s', flush=True)

    t0 = time.time(); df, _ = mc.run_generic(g['L'], g['P'], price, g['bank'], p, q3.solve,
                                             mask=(6, 12, 18), start=31, end=365, initial=init)
    add('A0 LP-MPC（改进版基线·分位法）', df, time.time() - t0)
    t0 = time.time(); df, _, _ = H.run_hybrid(g['L'], g['P'], price, g['rb'], p, H.Theta(), tv,
                                              mask=(6, 12, 18), start=31, end=365, initial=init)
    add('A1 分位法 + 分段线性终端价值（只加 L2）', df, time.time() - t0)
    pol_const = HP.Policy(**{**pol_star.__dict__, 'kappa': 1.0})
    t0 = time.time(); df, _, _ = HP.run_stoch2(g['L'], g['P'], price, g['bank'], g['rb'], p, pol_const, tv_const,
                                               mask=(6, 12, 18), start=31, end=365, initial=init)
    add('A2 情景随机 LP + 常数终端价值（只加 L1）', df, time.time() - t0)
    t0 = time.time(); dfh, ih, dh = HP.run_stoch2(g['L'], g['P'], price, g['bank'], g['rb'], p, pol_star, tv,
                                                  mask=(6, 12, 18), start=31, end=365, initial=init, detail=True)
    add('A3 HYBRID-4L（完整融合）', dfh, time.time() - t0)
    ih.to_csv(R / 'hybrid_intervals.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    dh.to_csv(R / 'hybrid_decisions.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    t0 = time.time(); df, _, _ = HP.run_stoch2(g['L'], g['P'], price, g['bank'], g['rb'], p, pol_star, tv,
                                               mask=(), start=31, end=365, initial=init)
    add('A4 HYBRID 不做日内更新（仅 0:00）', df, time.time() - t0)
    for name, d in daily_store.items():
        tag = name.split()[0]
        d.to_csv(R / f'hybrid_daily_{tag}.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    base = rows[0]['adjusted_cost']
    for r in rows:
        r['gap_vs_baseline_percent'] = 100 * (r['adjusted_cost'] - base) / base
    pd.DataFrame(rows).to_csv(R / 'hybrid_out_of_sample.csv', index=False, encoding='utf-8-sig',
                              float_format='%.6f')
    out['out_of_sample'] = rows

    # 逐月分解
    b = daily_store['A0 LP-MPC（改进版基线·分位法）'].copy()
    h = daily_store['A3 HYBRID-4L（完整融合）'].copy()
    for d in (b, h):
        d['month'] = pd.to_datetime(d.date).dt.month
    m = pd.DataFrame(dict(month=sorted(b.month.unique())))
    m['baseline'] = b.groupby('month').inventory_adjusted_cost.sum().values
    m['hybrid'] = h.groupby('month').inventory_adjusted_cost.sum().values
    m['saving'] = m.baseline - m.hybrid
    m['saving_percent'] = 100 * m.saving / m.baseline
    m.to_csv(R / 'hybrid_monthly.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    print(m.to_string(index=False), flush=True)

    # ---------------- 5  L4 NSGA-II 多目标 ----------------
    print('=== L4 NSGA-II 成本—CVaR Pareto ===', flush=True)
    mo_bounds = SEARCH_BOUNDS + [(0.0, 3.0), (0.80, 0.97)]
    ob = lambda X: batch(_mo_one, X)
    t0 = time.time()
    Xp, Fp, fronts, n_mo = HSch.nsga2(ob, mo_bounds, pop=14, gens=8, seed=20260915, log=True)
    t_mo = time.time() - t0
    f0 = fronts[0]
    front = pd.DataFrame([dict(train_cost=float(Fp[i][0]), train_cvar90=float(Fp[i][1]),
                               **dict(zip(list(SEARCH_KEYS) + ['cvar_beta', 'cvar_alpha'], Xp[i])))
                          for i in f0]).sort_values('train_cost').reset_index(drop=True)
    front.to_csv(R / 'hybrid_pareto_front_train.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    print(f'  前沿 {len(front)} 解，用时 {t_mo:.1f}s；样本外评估其中最多 5 个代表解', flush=True)
    pick = front.iloc[np.unique(np.linspace(0, len(front) - 1, min(5, len(front))).astype(int))]
    oos = []
    for _, r in pick.iterrows():
        pol = policy_from([r[k] for k in SEARCH_KEYS], cvar_beta=float(r.cvar_beta), cvar_alpha=float(r.cvar_alpha))
        t0 = time.time()
        d, _, _ = HP.run_stoch2(g['L'], g['P'], price, g['bank'], g['rb'], p, pol, tv,
                                mask=(6, 12, 18), start=31, end=365, initial=init)
        s = H.summarize(d, method='PARETO', seconds=time.time() - t0, cvar_beta=float(r.cvar_beta),
                        cvar_alpha=float(r.cvar_alpha), train_cost=float(r.train_cost))
        oos.append(s)
        print(f'  beta={r.cvar_beta:.2f} -> OOS adj={s["adjusted_cost"]:,.1f} CVaR95={s["cvar95"]:,.1f}', flush=True)
    pd.DataFrame(oos).to_csv(R / 'hybrid_pareto_out_of_sample.csv', index=False, encoding='utf-8-sig',
                             float_format='%.6f')
    out['pareto'] = dict(front_size=int(len(front)), seconds=t_mo, evaluations=int(n_mo), oos=oos)

    out['elapsed_seconds'] = time.time() - t_all
    out['n_scen'] = N_SCEN
    out['workers'] = N_WORKERS
    (R / 'hybrid_experiment_summary.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding='utf8')
    print('=== 全部完成，用时 %.1f 分钟 ===' % ((time.time() - t_all) / 60), flush=True)


if __name__ == '__main__':
    boot()
    prepare_tv()
    if N_WORKERS > 1:
        mp.set_start_method('fork', force=True)
        _POOL = mp.Pool(N_WORKERS, initializer=boot)
        globals()['_POOL'] = _POOL
    main()
