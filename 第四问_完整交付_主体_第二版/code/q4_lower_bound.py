"""全期完美信息下界：把 2—12 月 334 天 × 144 段**一次性**作为单个 LP 求解。

为什么需要它：原先用的 W2/W3 是"完美电价 + 完美净负荷"，但仍然跑在**滚动 24 小时视野 + 终端
价值代理**的框架里。滚动视野与终端代理都是**额外的约束**，因此那个数是"带约束的事后最优"，
**不构成任何可实施策略的下界**，不能叫"理论下界"。真正的下界必须解除这两条约束。

下界的合法性：任何因果策略在本题结算口径下的相同终端估值目标都不低于本 LP 的最优值（纯现金下界另以 vE_ref=0 求解）。
  · 净额结算下，取用 q 的实际支付为 p·q + 0.5p·u + 0.5p·v ≥ p·q（增购与减购罚金都非负）；
  · 紧急购电 5p ≥ p。
因此"以交付时段电价买下全部实际取用电量、且完美预见负荷/光伏/电价"的最优值在 vE_ref=0 时是现金成本下界；正终端估值时是库存校正目标下界。

规模：48,096 段 × 6 变量 ≈ 28.9 万变量，9.6 万条等式约束，稀疏 LP，HiGHS 直接求解。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import q4_core as Q4C
import q4_policy as Q4P
import q4_config as CFG
from q3 import Params, replace, DT, DATES

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
LO, HI = 0.1, 0.9


def solve_full_horizon(net, price, initial, p, vE_ref, eps=0.0):
    """单个大 LP：x = [q, c, dis, spill, E, z]，每块长 n。"""
    n = len(net)
    idx = np.arange(n)
    rr, cc, vv = [], [], []

    def put(r, c, v):
        r = np.asarray(r); c = np.asarray(c)
        rr.extend(r.tolist()); cc.extend(c.tolist()); vv.extend(np.broadcast_to(v, len(r)).tolist())

    # 平衡：q + z - c + dis - spill = net
    put(idx, idx, 1.0); put(idx, 5 * n + idx, 1.0); put(idx, n + idx, -1.0)
    put(idx, 2 * n + idx, 1.0); put(idx, 3 * n + idx, -1.0)
    # 动态：E_t - eta*c_t + dis_t/eta - E_{t-1} = 0
    put(n + idx, 4 * n + idx, 1.0); put(n + idx, n + idx, -p.eta); put(n + idx, 2 * n + idx, 1 / p.eta)
    put(n + idx[1:], 4 * n + idx[:-1], -1.0)
    b = np.r_[net, initial, np.zeros(n - 1)]

    cost = np.zeros(6 * n)
    cost[:n] = price
    cost[n:3 * n] = eps
    cost[5 * n:] = p.emergency * price
    cost[5 * n - 1] = -vE_ref                      # 期末库存按统一参考价折回
    bounds = [(0.0, None)] * (6 * n)
    pmax = p.power * DT
    for i in range(n):
        bounds[n + i] = (0.0, pmax)
        bounds[2 * n + i] = (0.0, pmax)
        bounds[4 * n + i] = (LO * p.capacity, HI * p.capacity)
    A = coo_matrix((vv, (rr, cc)), shape=(len(b), 6 * n)).tocsr()
    t0 = time.time()
    fit = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method='highs')
    if not fit.success:
        raise RuntimeError('full-horizon LP infeasible: ' + str(fit.message))
    x = fit.x
    q, c, dis, spill, E, z = (x[k * n:(k + 1) * n] for k in range(6))
    assert np.max(np.abs(A @ x - b)) < 1e-4
    cash = float(price @ q + p.emergency * (price @ z))
    return dict(seconds=time.time() - t0, cash_cost=cash,
                adjusted_cost=cash + vE_ref * (initial - float(E[-1])),
                purchase_kwh=float(q.sum()), emergency_kwh=float(z.sum()),
                spill_kwh=float(spill.sum()), charge_kwh=float(c.sum()),
                discharge_kwh=float(dis.sum()), final_soc=float(E[-1]),
                equivalent_cycles=float(dis.sum() / (2 * p.capacity)),
                soc_min=float(E.min()), soc_max=float(E.max()),
                balance_error=float(np.max(np.abs(A @ x - b))))


def main():
    L, P, F, tou, seed, V = Q4C.load4()
    pars = CFG.inherited_parameters()   # 唯一配置入口，见 q4_config.inherited_parameters
    sel0 = pars['selected']
    p = replace(Params(), **{k: sel0[k] for k in ('eta', 'capacity', 'power', 'emergency', 'up', 'down',
                                                  'epsilon')})
    p = CFG.apply_runtime(p)
    vE_ref = float(tou.min() / p.eta)
    start, end = 31, 365
    net = np.concatenate([(L[d] - P[d]) * DT for d in range(start, end)])
    price = np.concatenate([V[d] for d in range(start, end)])
    print(f'全期 LP 规模：{len(net):,} 段 × 6 变量 = {6*len(net):,} 变量', flush=True)
    rows = []
    for tag in ('q4_2', 'q4_3'):
        init = float(pd.read_csv(R / f'warmup_{tag}.csv').soc_end.iloc[-1])
        r = solve_full_horizon(net, price, init, p, vE_ref)
        cash_only = solve_full_horizon(net, price, init, p, 0.0)
        r['cash_lower_bound'] = cash_only['cash_cost']
        r['cash_optimal_final_soc'] = cash_only['final_soc']
        r.update(problem='Q4-2' if tag == 'q4_2' else 'Q4-3', initial_soc=init, vE_ref=vE_ref,
                 intervals=len(net), days=end - start)
        rows.append(r)
        print(f"  {r['problem']}（起始库存 {init:,.1f} kWh）现金下界 = {r['cash_cost']:,.2f} 元，"
              f"库存校正 = {r['adjusted_cost']:,.2f} 元，紧急 {r['emergency_kwh']:,.1f} kWh，"
              f"{r['seconds']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / 'q4_full_horizon_bound.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    print(df[['problem', 'cash_cost', 'adjusted_cost', 'emergency_kwh', 'spill_kwh']].to_string(index=False),
          flush=True)


if __name__ == '__main__':
    main()
