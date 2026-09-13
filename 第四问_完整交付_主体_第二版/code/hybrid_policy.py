"""HYBRID-4L：把各层拼成一个策略，并提供统一的滚动回测入口。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q3
from q3 import DT, ISSUES, DATES, execute
import hybrid_core as H
from hybrid_core import RiskBank, TerminalValue, SOC_LO_FRAC, SOC_HI_FRAC
import hybrid_stoch as HS

EXEC_STEPS = 36            # 每个发布时刻真正执行的段数（6 小时）


@dataclass(frozen=True)
class Policy:
    """L3 需要搜索的策略参数（全部在评价期之前的窗口上选定）。"""
    n_scen: int = 8            # 情景数（k-medoids 削减后）
    temper: float = 1.0        # 情景权重温度：w ∝ cnt^temper
    shrink: float = 1.0        # 残差整体缩放（<1 表示认为历史残差偏保守）
    tail_relax: float = 1.0    # 视野内「会被下次发布重新优化」的尾部时段，其紧急成本折减系数
    cvar_beta: float = 0.0     # CVaR 权重
    cvar_alpha: float = 0.90   # CVaR 置信水平
    kappa: float = 1.0         # 终端价值函数缩放
    threshold: float = 0.0     # 日内调整接受门槛（元）
    adj_premium: float = 1.0   # 第二阶段再调整的「期权行权溢价」，修正两阶段完美预见乐观偏差

    def vector(self):
        return np.array([self.n_scen, self.temper, self.shrink, self.tail_relax,
                         self.cvar_beta, self.cvar_alpha, self.kappa, self.threshold,
                         self.adj_premium], float)

    @staticmethod
    def from_vector(x):
        return Policy(n_scen=int(round(x[0])), temper=float(x[1]), shrink=float(x[2]),
                      tail_relax=float(x[3]), cvar_beta=float(x[4]), cvar_alpha=float(x[5]),
                      kappa=float(x[6]), threshold=float(x[7]))


POLICY_KEYS = ('n_scen', 'temper', 'shrink', 'tail_relax', 'cvar_beta', 'cvar_alpha',
               'kappa', 'threshold', 'adj_premium')
POLICY_BOUNDS = [(3, 14), (0.0, 2.0), (0.6, 1.4), (0.05, 1.0),
                 (0.0, 3.0), (0.80, 0.98), (0.0, 2.0), (0.0, 500.0), (1.0, 4.0)]


def emergency_scale(pol, n=144):
    """前 6 小时锁定执行 → 紧急成本按全额；之后一定会被下次发布重新优化 → 按 tail_relax 折减。"""
    s = np.full(n, float(pol.tail_relax))
    s[:EXEC_STEPS] = 1.0
    return s


def run_stoch(L, P, price, bank, rbank, p, pol: Policy, tv: TerminalValue,
              mask=(6, 12, 18), start=31, end=365, initial=6000., detail=False, seed=0):
    """两阶段情景随机 LP 驱动的滚动回测；物理执行与结算逻辑与主模型逐位一致。"""
    vE_lin = float(price.min() / p.eta)
    tv_use = tv.scaled(pol.kappa)
    dcoef = q3.settle_coef(p)
    esc = emergency_scale(pol)
    soc = initial
    daily, intervals, decisions = [], [], []

    def scen(day, k):
        resid = bank.residual[max(1, day - p.window):day, k, :]
        return HS.make_scenarios(bank.mean[day, k], resid, pol.n_scen,
                                 seed=seed + 97 * day + k, temper=pol.temper, shrink=pol.shrink)

    for day in range(start, end):
        begin = soc
        nets, w = scen(day, 0)
        pred = HS.solve_stochastic(nets, w, price, soc, p, tv_use, emergency_scale=esc,
                                   cvar_beta=pol.cvar_beta, cvar_alpha=pol.cvar_alpha)
        if pred is None:
            raise RuntimeError('day-ahead infeasible')
        q0, refday, _ = pred
        final = np.asarray(q0, float).copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        states = [soc]
        for k, h in enumerate(ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                tail = n1
                netsk, wk = scen(day, k)
                priceH = np.r_[price[t:], price[:t]]
                base = q0[t:]
                cand = HS.solve_stochastic(netsk, wk, priceH, soc, p, tv_use, base, tail,
                                           emergency_scale=esc, cvar_beta=pol.cvar_beta,
                                           cvar_alpha=pol.cvar_alpha)
                if cand is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = cand
                mid = netsk.T @ wk                       # 加权平均净负荷路径，用于评估接受与否

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(mid):
                        _, _, ee, _, ss = execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = price[t:] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = priceH[tail:] @ q[tail:]
                    return adj + future + p.emergency * (priceH @ zz) - tv_use.value_at(ss) + tv_use.value_at(soc)

                old = np.r_[final[t:], mid[tail:]]
                gain = expected(old, exec_ref) - expected(qn, rn)
                accept = (pol.threshold <= 0) or (gain > pol.threshold)
                if accept:
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                if detail:
                    decisions.append(dict(date=str(DATES[day].date()), issue=h,
                                          estimated_gain=float(gain), accepted=bool(accept)))
            for j in range(EXEC_STEPS):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = execute(final[tt], (L[day, tt] - P[day, tt]) * DT,
                                                               soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        delta = final - q0
        pc = float(price @ q0)
        uc = float(price @ (p.up * np.maximum(delta, 0)))
        dc = float(price @ (dcoef * np.maximum(-delta, 0)))
        ec = float(p.emergency * (price @ z))
        cash = pc + uc + dc + ec
        residual = final + z + dis - c - spill - (L[day] - P[day]) * DT
        dyn = np.diff(states) - p.eta * c + dis / p.eta
        assert abs(residual).max() < 1e-4 and abs(dyn).max() < 1e-4
        assert states.min() >= SOC_LO_FRAC * p.capacity - 1e-4 and states.max() <= SOC_HI_FRAC * p.capacity + 1e-4
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-4
        assert max(c.max(), dis.max()) <= p.power * DT + 1e-4
        daily.append(dict(date=str(DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc,
                          emergency_cost=ec, total_cost=cash,
                          inventory_adjusted_cost=cash + vE_lin * (begin - soc),
                          plan_kwh=q0.sum(), final_kwh=final.sum(), emergency_kwh=z.sum(),
                          up_kwh=float(np.maximum(delta, 0).sum()), down_kwh=float(np.maximum(-delta, 0).sum()),
                          charge_kwh=c.sum(), discharge_kwh=dis.sum(), spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc, balance_error=abs(residual).max(),
                          dynamics_error=abs(dyn).max(), soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, plan=q0[tt], adjusted=final[tt],
                                      charge=c[tt], discharge=dis[tt], emergency=z[tt], spill=spill[tt],
                                      soc_end=states[tt + 1]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions)


def run_stoch2(L, P, price, bank, rbank, p, pol: Policy, tv: TerminalValue,
               mask=(6, 12, 18), start=31, end=365, initial=6000., detail=False, seed=0):
    """HYBRID 主回测：内层为「带调整期权的两阶段情景 LP」。物理执行与结算与主模型逐位一致。"""
    vE_lin = float(price.min() / p.eta)
    tv_use = tv.scaled(pol.kappa)
    dcoef = q3.settle_coef(p)
    esc = emergency_scale(pol)
    soc = initial
    daily, intervals, decisions = [], [], []

    def scen(day, k):
        resid = bank.residual[max(1, day - p.window):day, k, :]
        return HS.make_scenarios(bank.mean[day, k], resid, pol.n_scen,
                                 seed=seed + 97 * day + k, temper=pol.temper, shrink=pol.shrink)

    for day in range(start, end):
        begin = soc
        nets, w = scen(day, 0)
        pred = HS.solve_stoch2(nets, w, price, soc, p, tv_use, None, 0, EXEC_STEPS,
                               emergency_scale=esc, cvar_beta=pol.cvar_beta, cvar_alpha=pol.cvar_alpha,
                               adj_premium=pol.adj_premium)
        if pred is None:
            raise RuntimeError('day-ahead infeasible')
        q0, refday, _ = pred
        final = np.asarray(q0, float).copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        states = [soc]
        for k, h in enumerate(ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                tail = n1
                netsk, wk = scen(day, k)
                priceH = np.r_[price[t:], price[:t]]
                base = q0[t:]
                cand = HS.solve_stoch2(netsk, wk, priceH, soc, p, tv_use, base, tail, EXEC_STEPS,
                                       emergency_scale=esc, cvar_beta=pol.cvar_beta, cvar_alpha=pol.cvar_alpha,
                                       adj_premium=pol.adj_premium)
                if cand is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = cand
                mid = netsk.T @ wk

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(mid):
                        _, _, ee, _, ss = execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = price[t:] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = priceH[tail:] @ q[tail:]
                    return adj + future + p.emergency * (priceH @ zz) - tv_use.value_at(ss) + tv_use.value_at(soc)

                old = np.r_[final[t:], mid[tail:]]
                gain = expected(old, exec_ref) - expected(qn, rn)
                accept = (pol.threshold <= 0) or (gain > pol.threshold)
                if accept:
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                if detail:
                    decisions.append(dict(date=str(DATES[day].date()), issue=h,
                                          estimated_gain=float(gain), accepted=bool(accept)))
            for j in range(EXEC_STEPS):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = execute(final[tt], (L[day, tt] - P[day, tt]) * DT,
                                                               soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        delta = final - q0
        pc = float(price @ q0)
        uc = float(price @ (p.up * np.maximum(delta, 0)))
        dc = float(price @ (dcoef * np.maximum(-delta, 0)))
        ec = float(p.emergency * (price @ z))
        cash = pc + uc + dc + ec
        residual = final + z + dis - c - spill - (L[day] - P[day]) * DT
        dyn = np.diff(states) - p.eta * c + dis / p.eta
        assert abs(residual).max() < 1e-4 and abs(dyn).max() < 1e-4
        assert states.min() >= SOC_LO_FRAC * p.capacity - 1e-4 and states.max() <= SOC_HI_FRAC * p.capacity + 1e-4
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-4
        assert max(c.max(), dis.max()) <= p.power * DT + 1e-4
        daily.append(dict(date=str(DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc,
                          emergency_cost=ec, total_cost=cash,
                          inventory_adjusted_cost=cash + vE_lin * (begin - soc),
                          plan_kwh=q0.sum(), final_kwh=final.sum(), emergency_kwh=z.sum(),
                          up_kwh=float(np.maximum(delta, 0).sum()), down_kwh=float(np.maximum(-delta, 0).sum()),
                          charge_kwh=c.sum(), discharge_kwh=dis.sum(), spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc, balance_error=abs(residual).max(),
                          dynamics_error=abs(dyn).max(), soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, plan=q0[tt], adjusted=final[tt],
                                      charge=c[tt], discharge=dis[tt], emergency=z[tt], spill=spill[tt],
                                      soc_end=states[tt + 1]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions)
