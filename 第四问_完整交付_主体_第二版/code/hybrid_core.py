"""第三问：分层融合求解器（HYBRID-4L）核心实现。

设计原则：**把每种算法放到它真正有优势的那一层**，而不是让某一种算法包办全部。

  L0 预测层      岭正则局部线性趋势 + 星期效应（按预测精度选超参，沿用主模型，不改动）
  L1 风险整形层  把「单一风险分位」推广为**提前量相关的风险分位曲线** alpha(tau)；
                 依据：每个发布时刻只有最前面 6 小时会被真正执行，其余时段一定会被下一次
                 发布重新优化，因此尾部时段不需要按同样的分位去「买保险」。
  L2 终端价值层  用**一次 Bellman 回代（拟合值迭代 / 参数化 LP）**把「常数边际库存价值
                 v_E = p_min/eta」升级为**分段线性凸终端价值函数** V(E)，直接嵌入 LP，
                 LP 仍然是精确解（凸性保证分段按边际价值降序自动填充）。
  L3 参数寻优层  memetic GA（实数编码 GA + Nelder-Mead 局部精修），并与 DE、SA 同预算对照；
                 只在**评价期之前的 1 月**滚动验证折上评价适应度。
  L4 多目标层    NSGA-II 在同一参数化上给出 成本—CVaR95 / 成本—紧急电量 的 Pareto 前沿。

内层始终是 LP（HiGHS）：题目的单阶段问题是线性的，LP 全局最优且最快，不应该用启发式替代。
启发式只负责 LP **写不进去**的部分：非凸的策略参数与多目标权衡。

时间口径：全部预测与残差分位只使用决策时点之前已经可获得的信息（严格因果）。
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import sys, json, time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import q3
from q3 import DT, ISSUES, DATES, Params, replace, execute

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
FIG = ROOT / 'figures'

SOC_LO_FRAC, SOC_HI_FRAC = 0.1, 0.9
LEAD_HOURS = np.arange(1, 145) / 6.0        # 视野内第 j 段的提前量（小时）


# ================================================================ L1 风险整形层
@dataclass(frozen=True)
class Theta:
    """HYBRID 策略参数（全部由 L3 在评价期之前的窗口上选定）。"""
    a0_near: float = 0.60      # 日前发布：提前量 -> 0 时的风险分位
    a0_far: float = 0.60       # 日前发布：提前量 -> 24h 时的风险分位
    a0_tau: float = 6.0        # 日前发布：分位曲线的衰减时间常数（小时）
    ah_near: float = 0.60      # 日内发布：近端分位
    ah_far: float = 0.60       # 日内发布：远端分位
    ah_tau: float = 6.0        # 日内发布：衰减时间常数
    kappa: float = 1.0         # 终端价值函数整体缩放
    threshold: float = 0.0     # 日内调整的接受门槛（元）

    def profile(self, intraday: bool) -> np.ndarray:
        near, far, tau = (self.ah_near, self.ah_far, self.ah_tau) if intraday else (self.a0_near, self.a0_far, self.a0_tau)
        a = far + (near - far) * np.exp(-LEAD_HOURS / max(tau, 1e-3))
        return np.clip(a, 0.02, 0.98)


THETA_KEYS = tuple(Theta().__dataclass_fields__.keys())
# 搜索边界：分位在 [0.35,0.97]，时间常数在 [0.5,24]h，终端价值缩放在 [0.0,2.0]，门槛在 [0,400] 元
THETA_BOUNDS = [(0.35, 0.97), (0.35, 0.97), (0.5, 24.0),
                (0.35, 0.97), (0.35, 0.97), (0.5, 24.0),
                (0.0, 2.0), (0.0, 400.0)]


class RiskBank:
    """在原 Bank 之上提供「逐段分位向量」的风险预算，向量化实现，支持 GA 的高频调用。"""

    def __init__(self, bank, window=28):
        self.bank = bank
        self.window = window
        self._sorted = {}

    def _sorted_resid(self, d, k):
        key = (d, k)
        if key not in self._sorted:
            err = self.bank.residual[max(1, d - self.window):d, k, :]
            if err.shape[0] == 0:
                self._sorted[key] = (None, None)
            else:
                cnt = np.isfinite(err).sum(axis=0)
                S = np.sort(np.where(np.isfinite(err), err, np.inf), axis=0)
                self._sorted[key] = (S, cnt)
        return self._sorted[key]

    def risk(self, d, k, alpha_vec):
        """mean + 逐段经验分位（每段用自己的 alpha），与 q3.Bank.risk 在常数 alpha 下数值一致。"""
        if d < 7:
            return self.bank.mean[d, k].copy()
        S, cnt = self._sorted_resid(d, k)
        if S is None:
            return self.bank.mean[d, k].copy()
        n = S.shape[1]
        cols = np.arange(n)
        valid = np.maximum(cnt, 1)
        pos = np.clip(np.asarray(alpha_vec, float), 0.0, 1.0) * (valid - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, valid - 1)
        w = pos - lo
        adj = S[lo, cols] * (1 - w) + S[hi, cols] * w
        adj = np.where(cnt > 0, adj, 0.0)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        return self.bank.mean[d, k] + adj


# ================================================================ L2 终端价值层
@dataclass(frozen=True)
class TerminalValue:
    """分段线性凸终端价值：V(E) = sum_k v_k * e_k，e_k 为第 k 段填充量，v_k 非增。"""
    edges: tuple            # 长度 K+1 的分段端点（kWh）
    marginals: tuple        # 长度 K 的段边际价值（元/kWh），非增

    @staticmethod
    def constant(vE, capacity=12000.0, K=8):
        edges = np.linspace(SOC_LO_FRAC * capacity, SOC_HI_FRAC * capacity, K + 1)
        return TerminalValue(tuple(edges.tolist()), tuple([float(vE)] * K))

    def scaled(self, kappa):
        return TerminalValue(self.edges, tuple(float(kappa) * m for m in self.marginals))

    def widths(self):
        e = np.asarray(self.edges)
        return np.diff(e)

    def value_at(self, E):
        e = np.asarray(self.edges); m = np.asarray(self.marginals)
        fill = np.clip(E - e[:-1], 0, np.diff(e))
        return float(fill @ m)


def fit_terminal_value(bank, price, p, days, K=8, capacity=12000.0, ngrid=25):
    """一次 Bellman 回代：用参数化 LP 求「以库存 E 起步的未来 24 小时最优成本」C(E)，
    取 v(E) = -dC/dE 作为终端库存的边际价值，再压成 K 段非增分段常数（凸性保证 LP 精确）。

    基准终端条件仍是线性的 v_E = p_min/eta（第 0 次值函数），因此这是标准的拟合值迭代
    第一步；得到的 V(E) 比常数边际更贴近真实的「明天再用掉这些电」的价值。
    """
    lo, hi = SOC_LO_FRAC * capacity, SOC_HI_FRAC * capacity
    grid = np.linspace(lo, hi, ngrid)
    vE = float(price.min() / p.eta)
    curves = []
    for d in days:
        net = bank.risk(d, 0, p.alpha0)
        c = []
        for E in grid:
            out = q3.solve(net, price, float(E), p, vE)
            if out is None:
                c.append(np.nan)
            else:
                c.append(out[2])
        curves.append(c)
    C = np.nanmean(np.array(curves, float), axis=0)
    # 数值凸化（保证边际价值非增）
    marg = -np.diff(C) / np.diff(grid)
    marg = np.maximum(marg, 0.0)
    marg = np.minimum.accumulate(marg)                      # 非增
    edges = np.linspace(lo, hi, K + 1)
    seg = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (grid[:-1] >= a - 1e-9) & (grid[:-1] < b - 1e-9)
        seg.append(float(marg[m].mean()) if m.any() else float(marg[-1]))
    seg = np.minimum.accumulate(np.array(seg))
    return TerminalValue(tuple(edges.tolist()), tuple(seg.tolist())), dict(
        grid=grid.tolist(), cost_curve=C.tolist(), marginal_raw=marg.tolist(),
        vE_linear=vE, days=[str(DATES[d].date()) for d in days])


# ================================================================ 内层 LP（含分段线性终端价值）
def solve_pwl(net, price, initial, p, tv: TerminalValue, original=None, tail_from=None):
    """滚动视野 LP：x = [q, c, dis, spill, E, u, v, e_1..e_K]。

    与 q3.solve 完全相同的物理与结算约束，只把终端库存价值由 -vE*E_D 换成分段线性
    -sum_k v_k e_k（v_k 非增 => 该分段函数是凹价值 / 凸成本 => LP 自动按边际价值降序填充，
    无需整数变量，解仍是该 LP 的全局最优）。
    """
    n = len(net)
    K = len(tv.marginals)
    w = tv.widths()
    E0 = float(tv.edges[0])
    N = 7 * n + K
    idx = np.arange(n)
    rr, cc, vv = [], [], []

    def put(r, c, v):
        rr.extend(np.asarray(r).tolist())
        cc.extend(np.asarray(c).tolist())
        vv.extend(np.broadcast_to(v, len(r)).tolist())

    put(idx, idx, 1); put(idx, n + idx, -1); put(idx, 2 * n + idx, 1); put(idx, 3 * n + idx, -1)
    put(n + idx, 4 * n + idx, 1); put(n + idx, n + idx, -p.eta); put(n + idx, 2 * n + idx, 1 / p.eta)
    put(n + idx[1:], 4 * n + idx[:-1], -1)
    rhs = np.r_[net, initial, np.zeros(n - 1)]

    cost = np.zeros(N)
    cost[n:3 * n] = p.epsilon
    bounds = [(0, None)] * (4 * n) + [(SOC_LO_FRAC * p.capacity, SOC_HI_FRAC * p.capacity)] * n \
             + [(0, 0)] * (2 * n) + [(0, float(wk)) for wk in w]
    for i in range(n):
        bounds[n + i] = bounds[2 * n + i] = (0, p.power * DT)

    if original is None:
        cost[:n] = price
    else:
        m = n if tail_from is None else tail_from
        jj = np.arange(m)
        put(2 * n + jj, jj, 1); put(2 * n + jj, 5 * n + jj, -1); put(2 * n + jj, 6 * n + jj, 1)
        rhs = np.r_[rhs, original]
        cost[5 * n:5 * n + m] = p.up * price[:m]
        coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
        cost[6 * n:6 * n + m] = coef * price[:m]
        cost[m:n] = price[m:]
        bounds[5 * n:5 * n + m] = [(0, None)] * m
        bounds[6 * n:6 * n + m] = [(0, None)] * m

    # 终端分段：sum_k e_k - E_{n-1} = -E0
    row = len(rhs)
    put(np.full(K, row), 7 * n + np.arange(K), 1.0)
    put([row], [5 * n - 1], -1.0)
    rhs = np.r_[rhs, -E0]
    cost[7 * n:] = -np.asarray(tv.marginals, float)

    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), N)).tocsr()
    fit = linprog(cost, A_eq=A, b_eq=rhs, bounds=bounds, method='highs')
    if not fit.success:
        return None
    assert np.max(np.abs(A @ fit.x - rhs)) < 1e-4
    x = fit.x
    q = x[:n]
    E = x[4 * n:5 * n]
    return q, np.r_[initial, E], float(fit.fun)


# ================================================================ HYBRID 滚动回测
def run_hybrid(L, P, price, rbank, p, theta: Theta, tv: TerminalValue,
               mask=(6, 12, 18), start=31, end=365, initial=6000., detail=False):
    """与 q3.run / method_comparison.run_generic 完全相同的物理执行与结算逻辑；
    差异只在三处：① 风险分位改为提前量曲线；② 终端库存价值改为分段线性；③ 调整接受门槛。"""
    vE_lin = float(price.min() / p.eta)
    tv_use = tv.scaled(theta.kappa)
    dcoef = q3.settle_coef(p)
    stepwise = (p.settlements == 'stepwise')
    prof0 = theta.profile(False)
    profh = theta.profile(True)
    soc = initial
    daily, intervals, decisions = [], [], []
    for day in range(start, end):
        begin = soc
        pred = solve_pwl(rbank.risk(day, 0, prof0), price, soc, p, tv_use)
        if pred is None:
            raise RuntimeError('day-ahead infeasible')
        q0, refday, _ = pred
        final = np.asarray(q0, float).copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        step_up_cost = step_dn_cost = step_up_kwh = step_dn_kwh = 0.0
        states = [soc]
        for k, h in enumerate(ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                tail = n1
                risk_k = rbank.risk(day, k, profh)
                priceH = np.r_[price[t:], price[:t]]
                base = final[t:] if stepwise else q0[t:]
                cand = solve_pwl(risk_k, priceH, soc, p, tv_use, base, tail)
                if cand is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = cand

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(risk_k):
                        _, _, ee, _, ss = execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = price[t:] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = priceH[tail:] @ q[tail:]
                    return adj + future + p.emergency * (priceH @ zz) - tv_use.value_at(ss) + tv_use.value_at(soc)

                old = np.r_[final[t:], risk_k[tail:]]
                gain = expected(old, exec_ref) - expected(qn[:len(risk_k)], rn)
                accept = (theta.threshold <= 0) or (gain > theta.threshold)
                if accept:
                    if stepwise:
                        dd = qn[:tail] - final[t:]
                        step_up_cost += float(price[t:] @ (p.up * np.maximum(dd, 0)))
                        step_dn_cost += float(price[t:] @ (dcoef * np.maximum(-dd, 0)))
                        step_up_kwh += float(np.maximum(dd, 0).sum())
                        step_dn_kwh += float(np.maximum(-dd, 0).sum())
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                if detail:
                    decisions.append(dict(date=str(DATES[day].date()), issue=h, estimated_gain=float(gain),
                                          accepted=bool(accept)))
            for j in range(36):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = execute(final[tt], (L[day, tt] - P[day, tt]) * DT,
                                                               soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        delta = final - q0
        pc = float(price @ q0)
        if stepwise:
            uc, dc, up_kwh, down_kwh = step_up_cost, step_dn_cost, step_up_kwh, step_dn_kwh
        else:
            uc = float(price @ (p.up * np.maximum(delta, 0)))
            dc = float(price @ (dcoef * np.maximum(-delta, 0)))
            up_kwh = float(np.maximum(delta, 0).sum())
            down_kwh = float(np.maximum(-delta, 0).sum())
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
                          up_kwh=up_kwh, down_kwh=down_kwh, charge_kwh=c.sum(), discharge_kwh=dis.sum(),
                          spill_kwh=spill.sum(), soc_start=begin, soc_end=soc,
                          balance_error=abs(residual).max(), dynamics_error=abs(dyn).max(),
                          soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, plan=q0[tt], adjusted=final[tt],
                                      charge=c[tt], discharge=dis[tt], emergency=z[tt], spill=spill[tt],
                                      soc_end=states[tt + 1]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions)


def summarize(df, **kw):
    return dict(**kw, total_cost=float(df.total_cost.sum()),
                adjusted_cost=float(df.inventory_adjusted_cost.sum()),
                plan_cost=float(df.plan_cost.sum()), up_cost=float(df.up_cost.sum()),
                down_net_cost=float(df.down_net_cost.sum()), emergency_cost=float(df.emergency_cost.sum()),
                emergency_kwh=float(df.emergency_kwh.sum()), up_kwh=float(df.up_kwh.sum()),
                down_kwh=float(df.down_kwh.sum()), charge_kwh=float(df.charge_kwh.sum()),
                discharge_kwh=float(df.discharge_kwh.sum()), spill_kwh=float(df.spill_kwh.sum()),
                equivalent_cycles=float(df.discharge_kwh.sum() / (2 * 12000.0)),
                cvar95=float(df.total_cost[df.total_cost >= df.total_cost.quantile(.95)].mean()),
                max_daily=float(df.total_cost.max()), daily_std=float(df.total_cost.std()),
                final_soc=float(df.soc_end.iloc[-1]))
