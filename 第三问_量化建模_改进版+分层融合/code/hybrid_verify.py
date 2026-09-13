"""HYBRID-4L 的独立复核：嵌套还原、账本独立重算、物理约束、严格因果性。"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, pickle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q3
from q3 import Params, replace, load, Bank, DATES, DT
import hybrid_core as H
import hybrid_policy as HP
import method_comparison as mc

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'


def main():
    L, P, F, price, seed = load()
    pars = json.loads((R / 'parameters.json').read_text(encoding='utf8'))
    sel = pars['selected']
    p = replace(Params(), **{k: sel[k] for k in ('alpha0', 'alpha_update', 'window', 'decay_days',
                                                 'bias_window', 'bias_decay', 'slope', 'weekday', 'epsilon')})
    bank = Bank(L, P, F, seed, p)
    rb = H.RiskBank(bank, p.window)
    init = float(pars['initial_feb1'])
    vE = float(price.min() / p.eta)
    tv = pickle.loads((R / 'hybrid_terminal_value.pkl').read_bytes())
    polj = json.loads((R / 'hybrid_selected_policy.json').read_text(encoding='utf8'))
    pol = HP.Policy(**{k: polj[k] for k in HP.POLICY_KEYS})
    out = {}

    # ---- 1 嵌套还原：常数分位 + 常数终端价值时，HYBRID 框架必须逐位复现主模型 ----
    tvc = H.TerminalValue.constant(vE)
    days = (31, 61)
    a, _ = mc.run_generic(L, P, price, bank, p, q3.solve, mask=(6, 12, 18), start=days[0], end=days[1], initial=init)
    b, _, _ = H.run_hybrid(L, P, price, rb, p, H.Theta(), tvc, mask=(6, 12, 18), start=days[0], end=days[1], initial=init)
    d = float(np.abs(a.total_cost.to_numpy() - b.total_cost.to_numpy()).max())
    out['nesting_max_daily_cost_diff_yuan'] = d
    out['nesting_pass'] = bool(d < 1e-6)
    print('nesting max daily diff = %.3e' % d, flush=True)

    # ---- 2 账本独立重算 ----
    iv = pd.read_csv(R / 'hybrid_intervals.csv')
    dl = pd.read_csv(R / 'hybrid_daily_A3.csv')
    errs = dict(cost=0.0, balance=0.0, soc=0.0, power=0.0)
    idx = {str(x.date()): i for i, x in enumerate(DATES)}
    prev_soc = init
    for date, g in iv.groupby('date', sort=True):
        g = g.sort_values('t')
        di = idx[date]
        plan = g.plan.to_numpy(); adj = g.adjusted.to_numpy()
        c = g.charge.to_numpy(); dis = g.discharge.to_numpy()
        z = g.emergency.to_numpy(); sp = g.spill.to_numpy(); soc = g.soc_end.to_numpy()
        delta = adj - plan
        pc = float(price @ plan)
        uc = float(price @ (p.up * np.maximum(delta, 0)))
        dc = float(price @ (-p.down * np.maximum(-delta, 0)))
        ec = float(p.emergency * (price @ z))
        row = dl[dl.date == date].iloc[0]
        errs['cost'] = max(errs['cost'], abs(pc + uc + dc + ec - row.total_cost))
        net = (L[di] - P[di]) * DT
        errs['balance'] = max(errs['balance'], float(np.abs(adj + z + dis - c - sp - net).max()))
        st = np.r_[prev_soc, soc]
        errs['soc'] = max(errs['soc'], float(np.abs(np.diff(st) - p.eta * c + dis / p.eta).max()))
        errs['power'] = max(errs['power'], float(max(c.max(), dis.max()) - p.power * DT))
        assert soc.min() >= 0.1 * p.capacity - 1e-4 and soc.max() <= 0.9 * p.capacity + 1e-4
        prev_soc = soc[-1]
    out['independent_recompute'] = {k: float(v) for k, v in errs.items()}
    out['recompute_pass'] = bool(errs['cost'] < 1e-3 and errs['balance'] < 1e-3 and errs['soc'] < 1e-3
                                 and errs['power'] < 1e-6)
    print('independent recompute', out['independent_recompute'], flush=True)

    # ---- 3 严格因果性：扰动决策时点之后的真实数据，当日 0:00 计划必须不变 ----
    day = 200
    nets, w = None, None
    resid = bank.residual[max(1, day - p.window):day, 0, :]
    import hybrid_stoch as HS
    nets, w = HS.make_scenarios(bank.mean[day, 0], resid, pol.n_scen, seed=97 * day,
                                temper=pol.temper, shrink=pol.shrink)
    q_ref, _, _ = HS.solve_stoch2(nets, w, price, init, p, tv.scaled(pol.kappa), None, 0, HP.EXEC_STEPS,
                                  emergency_scale=HP.emergency_scale(pol), adj_premium=pol.adj_premium)
    L2, P2, F2 = L.copy(), P.copy(), F.copy()
    rng = np.random.default_rng(0)
    L2[day:] *= 1 + 0.3 * rng.random(L2[day:].shape)
    P2[day:] *= 1 + 0.3 * rng.random(P2[day:].shape)
    F2[day + 1:] *= 1.4
    bank2 = Bank(L2, P2, F2, seed, p)
    resid2 = bank2.residual[max(1, day - p.window):day, 0, :]
    nets2, w2 = HS.make_scenarios(bank2.mean[day, 0], resid2, pol.n_scen, seed=97 * day,
                                  temper=pol.temper, shrink=pol.shrink)
    q_prb, _, _ = HS.solve_stoch2(nets2, w2, price, init, p, tv.scaled(pol.kappa), None, 0, HP.EXEC_STEPS,
                                  emergency_scale=HP.emergency_scale(pol), adj_premium=pol.adj_premium)
    dd = float(np.abs(np.asarray(q_ref) - np.asarray(q_prb)).max())
    out['causality_max_plan_diff_kwh'] = dd
    out['causality_pass'] = bool(dd < 1e-6)
    print('causality max plan diff = %.3e kWh' % dd, flush=True)

    # ---- 4 汇总断言 ----
    out['daily_max_balance_error'] = float(dl.balance_error.max())
    out['daily_max_dynamics_error'] = float(dl.dynamics_error.max())
    out['soc_bounds_ok'] = bool(dl.soc_min.min() >= 0.1 * p.capacity - 1e-4 and
                                dl.soc_max.max() <= 0.9 * p.capacity + 1e-4)
    out['days'] = int(len(dl))
    out['all_pass'] = bool(out['nesting_pass'] and out['recompute_pass'] and out['causality_pass']
                           and out['soc_bounds_ok'])
    (R / 'hybrid_verification.json').write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding='utf8')
    (R / 'hybrid_nesting_check.json').write_text(json.dumps(
        {'window': f'{DATES[days[0]].date()}/{DATES[days[1]-1].date()}',
         'max_daily_total_cost_diff_yuan': d,
         'note': 'HYBRID 框架取 Theta(全部分位=0.60) 与常数终端价值时应逐位复现改进版主模型'},
        ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
