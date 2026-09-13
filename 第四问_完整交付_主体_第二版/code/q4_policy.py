"""第四问策略层与滚动回测：波动电价下的 Q4-2（每天 0:00 一次）与 Q4-3（0/6/12/18 四次）。

物理执行、储能动态、结算口径与第三问逐位一致，**只把电价换成附件 4 的实际逐日逐时电价**：
  · 计划购电费  = Σ_t V[d,t] q0_t
  · 增购        = 1.5 × V[d,t] × (最终-计划)^+
  · 减购净额    = −0.5 × V[d,t] × (计划-最终)^+
  · 紧急购电    = 5 × V[d,t] × z_t
决策端只能用决策时点之前的信息（负荷/光伏/电价预报），账单端一律用**实际电价**。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q3
from q4_statistics import empirical_cvar
from q3 import DT, ISSUES, DATES, execute
import hybrid_core as H
import q4_core as Q4C
import q4_stoch as Q4S

EXEC_STEPS = 36
SOC_LO_FRAC, SOC_HI_FRAC = 0.1, 0.9


@dataclass(frozen=True)
class Q4Policy:
    n_scen: int = 12
    temper: float = 0.435
    shrink_net: float = 0.741
    shrink_price: float = 1.0
    kappa: float = 1.0
    threshold: float = 295.4
    adj_premium: float = 1.18
    alpha: float = 0.60            # 仅确定性对照口径使用
    price_mode: str = 'forecast'   # forecast（主口径）| tou（当作附件1固定电价）| perfect（完美预见上界）
    net_mode: str = 'forecast'     # forecast（主口径）| perfect（完美净负荷，仅用于全知下界）


Q4_KEYS = ('temper', 'shrink_net', 'shrink_price', 'kappa', 'threshold', 'adj_premium')
Q4_BOUNDS = [(0.0, 2.0), (0.55, 1.30), (0.30, 1.60), (0.0, 2.0), (0.0, 500.0), (0.80, 2.20)]

# Q4-2 的日内普通增减购电锁定，threshold 和 adj_premium 因而惰性。
# 充放电、库存与紧急购电仍为情景相关的补救决策，仍有情景内前视近似。
# 固定 shrink=1 是短训练窗口下的降维正则化选择，不是结构性定理。
Q4_KEYS_2 = ('temper', 'kappa')
FIXED_Q4_2 = dict(shrink_net=1.0, shrink_price=1.0)
Q4_KEYS_2_FULL = ('temper', 'shrink_net', 'shrink_price', 'kappa')
# Q4-2 的基础搜索框比 Q4-3 宽：上一轮的边界延展已经证明 Q4-3 的框（shrink_net≥0.55、
# shrink_price≥0.30）对 Q4-2 太窄——没有日内调整通道时最优的对冲力度明显更低，
# 最优点会落到框外。既然这一事实是在**同一个 1 月窗口**上得到的，直接把它写进基础框，
# 而不是每次都靠延展去补救。
Q4_BOUNDS_2 = [(0.0, 2.0), (0.0, 2.5)]
Q4_BOUNDS_2_FULL = [(0.0, 2.0), (0.02, 1.40), (0.005, 1.80), (0.0, 2.5)]
INERT_Q4_2 = dict(threshold=0.0, adj_premium=1.0)


def emergency_scale(n=144, lock=EXEC_STEPS, tail_relax=1.0):
    s = np.full(n, float(tail_relax)); s[:lock] = 1.0
    return s


def price_paths(pbank, pol, d, k):
    """按口径给出该发布时刻的电价情景输入的**中心路径**。"""
    if pol.price_mode == 'tou':
        t = ISSUES[k] * 6
        return np.r_[pbank.tou, pbank.tou][t:t + 144]
    if pol.price_mode == 'perfect':
        a = pbank.actual_horizon(d, k)
        return np.where(np.isfinite(a), a, pbank.mean[d, k])
    return pbank.mean[d, k]


def run_q4(L, P, V, bank, rbank, pbank, p, pol: Q4Policy, tv_shape, tv_edges, vE_ref,
           mask=(6, 12, 18), start=31, end=365, initial=6000., mode='stoch',
           detail=False, seed=0, window=28):
    """滚动回测。mode='stoch' 为 HYBRID 联合情景随机 LP；mode='det' 为分位数 + 点预测电价的确定性对照。"""
    dcoef = q3.settle_coef(p)
    lock = EXEC_STEPS if mask else 144
    esc = emergency_scale(144, lock)
    tv_shape = np.asarray(tv_shape, float)
    soc = initial
    daily, intervals, decisions, transactions = [], [], [], []

    step_up = step_down = 0.0

    def actual_net(day, k):
        t = ISSUES[k] * 6
        a = np.r_[(L[day] - P[day]) * DT,
                  (L[day + 1] - P[day + 1]) * DT if day + 1 < L.shape[0] else np.zeros(144)]
        return a[t:t + 144]

    def scen(day, k):
        center = price_paths(pbank, pol, day, k)
        nres = bank.residual[max(1, day - window):day, k, :]
        nmean = actual_net(day, k) if pol.net_mode == 'perfect' else bank.mean[day, k]
        sp = 0.0 if pol.price_mode == 'perfect' else pol.shrink_price
        sn = 0.0 if pol.net_mode == 'perfect' else pol.shrink_net
        plog = np.zeros_like(nres) if pol.price_mode == 'perfect' else pbank.resid_window(day, k, window)
        nets, prices, w = Q4C.joint_scenarios(nmean, nres, center, plog, pol.n_scen,
                                              seed=seed + 97 * day + k, temper=pol.temper,
                                              shrink_net=sn, shrink_price=sp)
        if pol.net_mode == 'perfect' and pol.price_mode == 'perfect':
            nets, prices, w = nets[:1], prices[:1], np.array([1.0])
        return nets, prices, w, center

    for day in range(start, end):
        begin = soc
        step_up = step_down = 0.0
        nets, prices, w, center = scen(day, 0)
        vE_day = float(center.min() / p.eta)
        tvm = pol.kappa * tv_shape * vE_day
        if mode == 'det':
            netq = rbank.risk(day, 0, np.full(144, pol.alpha))
            out = Q4S.solve_q4_deterministic(netq, center, soc, p, tvm, tv_edges)
        else:
            out = Q4S.solve_q4(nets, prices, w, soc, p, tvm, tv_edges, None, 0, lock,
                               emergency_scale=esc, adj_premium=pol.adj_premium)
        if out is None:
            raise RuntimeError(f'day-ahead infeasible at day {day}')
        q0, refday, _ = out
        final = np.asarray(q0, float).copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        states = [soc]
        for k, h in enumerate(ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                tail = n1
                netsk, pricesk, wk, ck = scen(day, k)
                base = final[t:].copy() if p.settlements == 'stepwise' else q0[t:]
                vE_k = float(ck.min() / p.eta)
                tvk = pol.kappa * tv_shape * vE_k
                if mode == 'det':
                    netq = rbank.risk(day, k, np.full(144, pol.alpha))
                    cand = Q4S.solve_q4_deterministic(netq, ck, soc, p, tvk, tv_edges, base, tail)
                    mid = netq
                    midp = ck
                else:
                    cand = Q4S.solve_q4(netsk, pricesk, wk, soc, p, tvk, tv_edges, base, tail, lock,
                                        emergency_scale=esc, adj_premium=pol.adj_premium)
                    mid = netsk.T @ wk
                    midp = pricesk.T @ wk
                if cand is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = cand

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(mid):
                        _, _, ee, _, ss = execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = midp[:tail] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = midp[tail:] @ q[tail:]
                    return adj + future + p.emergency * (midp @ zz) - vE_k * (ss - soc)

                old = np.r_[final[t:], mid[tail:]]
                gain = expected(old, exec_ref) - expected(qn, rn)
                accept = (pol.threshold <= 0) or (gain > pol.threshold)
                if accept:
                    prev = final[t:].copy()          # 本次调整前正在执行的计划
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                    # 计费永远执行，明细导出开关不影响账单。
                    change = final[t:] - prev
                    step_up += float(V[day, t:] @ (p.up * np.maximum(change, 0)))
                    step_down += float(V[day, t:] @ (dcoef * np.maximum(-change, 0)))
                    if detail:
                        # 逐笔交易账本：本次相对**上一次有效计划**的增减，按交付时段实际电价计价。
                        # 注意主口径是净额结算（日末相对 0:00 基准计划一次结清），
                        # 因此这里的 fee 是**归因**，不是账单本身；账单见 daily.csv。
                        # settlements='stepwise' 时这些 fee 之和才等于当日调整费。
                        dd = final[t:] - prev
                        nz = np.nonzero(np.abs(dd) > 1e-9)[0]
                        for j in nz:
                            tt = t + int(j)
                            delta = float(dd[j])
                            fee = float(V[day, tt] * (p.up * delta if delta > 0 else dcoef * (-delta)))
                            transactions.append(dict(
                                date=str(DATES[day].date()), issue=h, t=tt,
                                q_prev=float(prev[j]), q_new=float(final[t + int(j)]), delta=delta,
                                q_baseline_0000=float(q0[tt]),
                                delivery_price=float(V[day, tt]), issue_price=float(V[day, t - 1] if t else V[day, 0]),
                                fee_attributed=fee))
                if detail:
                    decisions.append(dict(date=str(DATES[day].date()), issue=h,
                                          estimated_gain=float(gain), accepted=bool(accept),
                                          up_kwh=float(np.maximum(qn[:tail] - base, 0).sum()),
                                          down_kwh=float(np.maximum(base - qn[:tail], 0).sum())))
            for j in range(EXEC_STEPS):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = execute(final[tt], (L[day, tt] - P[day, tt]) * DT,
                                                               soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        pr = V[day]
        delta = final - q0
        pc = float(pr @ q0)
        if p.settlements == 'stepwise':
            # 逐次调整分别计费：当日各笔交易的归因费之和
            uc, dc = step_up, step_down
        else:
            uc = float(pr @ (p.up * np.maximum(delta, 0)))
            dc = float(pr @ (dcoef * np.maximum(-delta, 0)))
        ec = float(p.emergency * (pr @ z))
        cash = pc + uc + dc + ec
        residual = final + z + dis - c - spill - (L[day] - P[day]) * DT
        dyn = np.diff(states) - p.eta * c + dis / p.eta
        assert abs(residual).max() < 1e-4 and abs(dyn).max() < 1e-4
        assert states.min() >= SOC_LO_FRAC * p.capacity - 1e-4 and states.max() <= SOC_HI_FRAC * p.capacity + 1e-4
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-4
        assert max(c.max(), dis.max()) <= p.power * DT + 1e-4
        daily.append(dict(date=str(DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc,
                          emergency_cost=ec, total_cost=cash,
                          inventory_adjusted_cost=cash + vE_ref * (begin - soc),
                          plan_kwh=q0.sum(), final_kwh=final.sum(), emergency_kwh=z.sum(),
                          up_kwh=float(np.maximum(delta, 0).sum()), down_kwh=float(np.maximum(-delta, 0).sum()),
                          charge_kwh=c.sum(), discharge_kwh=dis.sum(), spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc, mean_price=float(pr.mean()),
                          balance_error=abs(residual).max(), dynamics_error=abs(dyn).max(),
                          soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, price=float(pr[tt]),
                                      plan=q0[tt], adjusted=final[tt], charge=c[tt], discharge=dis[tt],
                                      emergency=z[tt], spill=spill[tt], soc_end=states[tt + 1]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions), pd.DataFrame(transactions)


def summarize(df, **kw):
    return dict(**kw, total_cost=float(df.total_cost.sum()),
                adjusted_cost=float(df.inventory_adjusted_cost.sum()),
                plan_cost=float(df.plan_cost.sum()), up_cost=float(df.up_cost.sum()),
                down_net_cost=float(df.down_net_cost.sum()), emergency_cost=float(df.emergency_cost.sum()),
                emergency_kwh=float(df.emergency_kwh.sum()), up_kwh=float(df.up_kwh.sum()),
                down_kwh=float(df.down_kwh.sum()), charge_kwh=float(df.charge_kwh.sum()),
                discharge_kwh=float(df.discharge_kwh.sum()), spill_kwh=float(df.spill_kwh.sum()),
                plan_kwh=float(df.plan_kwh.sum()),
                equivalent_cycles=float(df.discharge_kwh.sum() / (2 * 12000.0)),
                cvar95=empirical_cvar(df.total_cost.to_numpy(), .95),
                top17_mean=float(np.sort(df.total_cost.to_numpy())[-17:].mean()),
                max_daily=float(df.total_cost.max()), daily_std=float(df.total_cost.std()),
                final_soc=float(df.soc_end.iloc[-1]))
