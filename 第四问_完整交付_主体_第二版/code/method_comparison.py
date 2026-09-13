"""第三问：多种求解方法的分层对比实验。

把「合适的方法放到题目的不同部分」：本模块实现四类求解器，并让它们在**同一物理模型、
同一结算口径、同一评价期、同一初始库存**下回测，从而可直接比较优劣。

方法清单
  LP   —— 连续线性规划（HiGHS），当前主模型，全局最优基准
  DP   —— 动态规划，以 SOC 为离散状态、以时段为阶段，单阶段精确（离散化误差内）
  GA   —— 遗传算法：① 直接编码 144 维购电计划（说明为何不适用）
                        ② 优化规则型策略参数（说明其真正的落点）
  RULE —— 阈值型反馈规则（只用电价与 SOC，可解释、无需优化）
  NSGA-II —— 多目标进化，给出「购电成本 × 电池循环寿命」的 Pareto 前沿

分层
  L1 日前计划（单阶段最优性）：LP / DP / GA-直接 / RULE
  L2 全年滚动策略（真实经济效果）：LP-MPC / DP-MPC / RULE(手工) / RULE(GA调参) / 无日内更新
  L3 多目标：成本—风险（分位谱）与成本—寿命（NSGA-II）两条 Pareto 前沿
"""
from pathlib import Path
import sys, json, time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

import q3
from q3 import DT, ISSUES, DATES, Params, replace, load, Bank, execute, load_forecast

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
FIG = ROOT / 'figures'
NG_DEFAULT = 160                       # SOC 离散网格数（DP 主口径）
THETA_KEYS = ('qlo', 'qhi', 'soc_hi', 'soc_lo', 'c_rate', 'd_rate')
THETA_MANUAL = dict(qlo=0.30, qhi=0.70, soc_hi=0.85, soc_lo=0.15, c_rate=0.6, d_rate=0.6)


def setup_font():
    for font in ['/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc']:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})


def dump(name, obj):
    (R / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                          encoding='utf8')


def csv(name, frame):
    pd.DataFrame(frame).to_csv(R / name, index=False, encoding='utf-8-sig', float_format='%.10f')


# ================================================================ DP：动态规划求解器
def dp_solve_clean(net, price, initial, p, vE=0.0, original=None, tail_from=None, ngrid=NG_DEFAULT):
    """DP 求解器（含起点状态网格化，返回可执行计划与 SOC 轨迹）。"""
    n = len(net)
    m = n if tail_from is None else tail_from
    base = None if original is None else np.asarray(original, float)
    lo, hi = 0.1 * p.capacity, 0.9 * p.capacity
    grid = np.linspace(lo, hi, ngrid)
    eta, pmax = p.eta, p.power * DT
    coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down

    def stage_cost(q, t):
        if original is None or t >= m:
            return price[t] * q
        return p.up * price[t] * np.maximum(q - base[t], 0.0) + coef * price[t] * np.maximum(base[t] - q, 0.0)

    def move(s_cur, s_nxt, net_t):
        """由「当前 SOC → 下一 SOC」反推该区间购电量；返回 (q, 是否可行，充/放电量)。"""
        d = s_nxt - s_cur
        charge = np.where(d > 0, d / eta, 0.0)
        dis = np.where(d < 0, -d * eta, 0.0)
        ok = (charge <= pmax + 1e-9) & (dis <= pmax + 1e-9) & (~((dis > 1e-9) & (dis > net_t + 1e-9)))
        q = np.maximum(net_t + charge - dis, 0.0)
        return q, ok, charge, dis

    # ---- 反向：在 SOC 网格上递推值函数 V_t(s) = min [ 阶段成本 + V_{t+1}(s') ] ----
    V = np.empty((n + 1, ngrid))
    V[n] = -vE * grid
    for t in range(n - 1, -1, -1):
        q, ok, _, _ = move(grid[:, None], grid[None, :], net[t])
        ct = stage_cost(q, t)
        ct = np.where(ok, ct, np.inf)
        V[t] = (ct + V[t + 1][None, :]).min(axis=1)

    # ---- 正向：从**真实**初始 SOC 出发（可能不在网格上），逐段选最优下一步并线性插值值函数 ----
    q = np.empty(n)
    states = np.empty(n + 1)
    states[0] = initial
    s = initial
    value = 0.0
    for t in range(n):
        cand = np.r_[grid, s]                       # 候选下一状态：网格点 + 原地不动
        qc, ok, _, _ = move(s, cand, net[t])
        ct = np.where(ok, stage_cost(qc, t), np.inf)
        vn = np.interp(cand, grid, V[t + 1])
        tot = ct + vn
        k = int(np.argmin(tot))
        if not np.isfinite(tot[k]):
            raise RuntimeError('DP infeasible at stage %d' % t)
        q[t] = qc[k]
        s = float(cand[k])
        states[t + 1] = s
        value += float(ct[k])
    ref = np.r_[initial, states[1:]]
    return q, ref, float(value)


# ================================================================ RULE：阈值型反馈规则
def simulate_ref(q, net, initial, p):
    ref = np.empty(len(q) + 1)
    ref[0] = initial
    s = initial
    for j in range(len(q)):
        _, _, _, _, s = execute(q[j], net[j], s, 0.1 * p.capacity, p)
        ref[j + 1] = s
    return ref


def rule_plan(net, price, initial, p, theta):
    """阈值型规则：按电价分位决定「多买充电」还是「少买放电」，只用当前 SOC 与电价。

    注意：预测净负荷可能为负（光伏盈余时段），因此最后必须无条件把计划截断到 q>=0；
    否则会出现「负购电」这种物理上不存在的指令。
    """
    q = np.asarray(net, float).copy()
    plo = float(np.quantile(price, theta['qlo']))
    phi = float(np.quantile(price, theta['qhi']))
    cap = p.capacity
    if initial <= theta['soc_hi'] * cap:
        q = q + theta['c_rate'] * p.power * DT * (price <= plo)
    if initial >= theta['soc_lo'] * cap:
        q = q - theta['d_rate'] * p.power * DT * (price >= phi)
    return np.maximum(q, 0.0)


def make_rule_solver(theta):
    def rule_solve(net, price, initial, p, vE=0.0, original=None, tail_from=None):
        q = rule_plan(net, price, initial, p, theta)
        return q, simulate_ref(q, net, initial, p), 0.0
    return rule_solve


# ================================================================ 通用滚动回测
def run_generic(L, P, price, bank, p, solver, mask=(6, 12, 18), start=31, end=365, initial=6000.,
                detail=False):
    """与 q3.run 完全相同的物理执行与结算逻辑，只把内层求解器替换为 solver。"""
    vE = float(price.min() / p.eta)
    dcoef = q3.settle_coef(p)
    stepwise = (p.settlements == 'stepwise')
    soc = initial
    daily, intervals = [], []
    for day in range(start, end):
        begin = soc
        pred = solver(bank.risk(day, 0, p.alpha0), price, soc, p, vE)
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
                risk_k = bank.risk(day, k, p.alpha_update)
                priceH = np.r_[price[t:], price[:t]]
                base = final[t:] if stepwise else q0[t:]
                cand = solver(risk_k, priceH, soc, p, vE, base, tail)
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
                    return adj + future + p.emergency * (priceH @ zz) - vE * (ss - soc)

                old = np.r_[final[t:], risk_k[tail:]]
                gain = expected(old, exec_ref) - expected(qn[:len(risk_k)], rn)
                if p.threshold <= 0 or gain > p.threshold:
                    if stepwise:
                        dd = qn[:tail] - final[t:]
                        step_up_cost += float(price[t:] @ (p.up * np.maximum(dd, 0)))
                        step_dn_cost += float(price[t:] @ (dcoef * np.maximum(-dd, 0)))
                        step_up_kwh += float(np.maximum(dd, 0).sum())
                        step_dn_kwh += float(np.maximum(-dd, 0).sum())
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
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
        assert abs(residual).max() < 1e-5 and abs(dyn).max() < 1e-5
        assert states.min() >= .1 * p.capacity - 1e-5 and states.max() <= .9 * p.capacity + 1e-5
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-5, (
            f'day={day} c_min={c.min():.3e} dis_min={dis.min():.3e} z_min={z.min():.3e} '
            f'spill_min={spill.min():.3e} final_min={final.min():.3e} soc_start={begin:.3f}')
        assert max(c.max(), dis.max()) <= p.power * DT + 1e-5
        daily.append(dict(date=str(DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc,
                          emergency_cost=ec, total_cost=cash,
                          inventory_adjusted_cost=cash + vE * (begin - soc),
                          plan_kwh=q0.sum(), emergency_kwh=z.sum(), up_kwh=up_kwh, down_kwh=down_kwh,
                          charge_kwh=c.sum(), discharge_kwh=dis.sum(), spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc, balance_error=abs(residual).max(),
                          dynamics_error=abs(dyn).max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, plan=q0[tt], adjusted=final[tt],
                                      charge=c[tt], discharge=dis[tt]))
    return pd.DataFrame(daily), pd.DataFrame(intervals)


def summarize(df, **kw):
    return dict(**kw, total_cost=df.total_cost.sum(),
                adjusted_cost=df.inventory_adjusted_cost.sum(),
                plan_cost=df.plan_cost.sum(), up_cost=df.up_cost.sum(),
                down_net_cost=df.down_net_cost.sum(), emergency_cost=df.emergency_cost.sum(),
                emergency_kwh=df.emergency_kwh.sum(), up_kwh=df.up_kwh.sum(), down_kwh=df.down_kwh.sum(),
                charge_kwh=df.charge_kwh.sum(), discharge_kwh=df.discharge_kwh.sum(),
                spill_kwh=df.spill_kwh.sum(),
                equivalent_cycles=float(df.discharge_kwh.sum() / (2 * 12000.0)),
                cvar95=float(df.total_cost[df.total_cost >= df.total_cost.quantile(.95)].mean()),
                final_soc=df.soc_end.iloc[-1])


# ================================================================ 遗传算法（实编码）
def ga_optimize(fitness, bounds, pop=40, gens=40, seed=7, pm=0.20, pc=0.85, elite=2, verbose=True):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    ndim = len(bounds)

    def repair(X):
        X = np.clip(X, lo, hi)
        a = X[:, 0].copy(); b = X[:, 1].copy()
        X[:, 0] = np.minimum(a, b); X[:, 1] = np.maximum(a, b)
        a = X[:, 2].copy(); b = X[:, 3].copy()
        X[:, 2] = np.maximum(a, b); X[:, 3] = np.minimum(a, b)
        return X

    P = repair(lo + rng.random((pop, ndim)) * (hi - lo))
    fit = np.array([fitness(x) for x in P])
    hist = []
    for g in range(gens):
        order = np.argsort(fit)
        P, fit = P[order], fit[order]
        hist.append(dict(generation=g, best=float(fit[0]), mean=float(fit.mean())))
        if verbose and g % 10 == 0:
            print(f'  GA gen={g} best={fit[0]:,.1f} mean={fit.mean():,.1f}', flush=True)
        newP = [P[i].copy() for i in range(elite)]
        while len(newP) < pop:
            i, j = rng.integers(0, pop, 2)
            p1 = P[i] if fit[i] <= fit[j] else P[j]
            i, j = rng.integers(0, pop, 2)
            p2 = P[i] if fit[i] <= fit[j] else P[j]
            if rng.random() < pc:
                u = rng.random(ndim)
                alpha = 0.4
                c1 = p1 + alpha * (u * (p2 - p1)) + (1 - u) * 0.0
                c2 = p2 - alpha * (u * (p2 - p1))
                # BLX-alpha
                gmin = np.minimum(p1, p2); gmax = np.maximum(p1, p2)
                rng_span = gmax - gmin
                c1 = rng.uniform(gmin - alpha * rng_span, gmax + alpha * rng_span)
                c2 = rng.uniform(gmin - alpha * rng_span, gmax + alpha * rng_span)
                ch = c1 if rng.random() < .5 else c2
            else:
                ch = p1.copy()
            m = rng.random(ndim) < pm
            ch = ch + m * rng.normal(0, 0.12, ndim) * (hi - lo)
            newP.append(ch)
        P = repair(np.array(newP[:pop]))
        fit = np.array([fitness(x) for x in P])
    order = np.argsort(fit)
    return P[order[0]], float(fit[order[0]]), hist


def nsga2(objectives, bounds, pop=20, gens=15, seed=11, pm=0.2, pc=0.9, verbose=True):
    """紧凑版 NSGA-II：快速非支配排序 + 拥挤距离。objectives(x) -> (f1, f2)，均最小化。"""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    ndim = len(bounds)

    def repair(X):
        X = np.clip(X, lo, hi)
        X[:, 0], X[:, 1] = np.minimum(X[:, 0], X[:, 1]), np.maximum(X[:, 0], X[:, 1])
        X[:, 2], X[:, 3] = np.maximum(X[:, 2], X[:, 3]), np.minimum(X[:, 2], X[:, 3])
        return X

    def dominates(a, b):
        return (a <= b).all() and (a < b).any()

    def nondominated_front(F):
        n = len(F)
        fronts, S = [], [[] for _ in range(n)]
        nd = np.zeros(n, int)
        first = []
        for p in range(n):
            for q_ in range(n):
                if p == q_:
                    continue
                if dominates(F[p], F[q_]):
                    S[p].append(q_)
                elif dominates(F[q_], F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                first.append(p)
        fronts.append(first)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q_ in S[p]:
                    nd[q_] -= 1
                    if nd[q_] == 0:
                        nxt.append(q_)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    def crowding(F, idx):
        l = len(idx)
        d = np.zeros(l)
        if l == 0:
            return d
        for m in range(F.shape[1]):
            order = np.argsort(F[idx, m])
            d[order[0]] = d[order[-1]] = np.inf
            rng_m = F[idx, m].max() - F[idx, m].min()
            if rng_m > 0:
                d[order[1:-1]] += (F[idx, m][order[2:]] - F[idx, m][order[:-2]]) / rng_m
        return d

    P = repair(lo + rng.random((pop, ndim)) * (hi - lo))
    F = np.array([objectives(x) for x in P])
    for g in range(gens):
        fronts = nondominated_front(F)
        rank = np.empty(pop, int)
        crowd = np.empty(pop)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding(F, np.array(fr))
        order = np.lexsort((-crowd, rank))
        P, F, rank, crowd = P[order], F[order], rank[order], crowd[order]
        if verbose and g % 5 == 0:
            print(f'  NSGA-II gen={g} front0={int((rank == 0).sum())} best_cost={F[:, 0].min():,.0f}', flush=True)
        newP = [P[i].copy() for i in range(2)]
        while len(newP) < pop:
            i, j = rng.integers(0, pop, 2)
            p1 = P[i] if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else P[j]
            i, j = rng.integers(0, pop, 2)
            p2 = P[i] if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else P[j]
            if rng.random() < pc:
                gmin = np.minimum(p1, p2); gmax = np.maximum(p1, p2)
                span = gmax - gmin
                ch = rng.uniform(gmin - 0.4 * span, gmax + 0.4 * span)
            else:
                ch = p1.copy()
            ch = ch + (rng.random(ndim) < pm) * rng.normal(0, 0.12, ndim) * (hi - lo)
            newP.append(ch)
        P = repair(np.array(newP[:pop]))
        F = np.array([objectives(x) for x in P])
    fronts = nondominated_front(F)
    return P, F, fronts


# ================================================================ L1：日前计划单阶段对比
def evaluate_plan(q, net, price, initial, p, vE):
    """按与 LP 相同的目标函数评价一个购电计划，并给出物理执行结果。"""
    q = np.asarray(q, float)
    s = initial
    zs = []
    for j in range(len(q)):
        _, _, zz, _, s = execute(q[j], net[j], s, 0.1 * p.capacity, p)
        zs.append(zz)
    zs = np.array(zs)
    return dict(purchase_cost=float(price @ q),
                terminal_value=float(vE * s),
                emergency_kwh=float(zs.sum()),
                emergency_cost=float(p.emergency * (price @ zs)),
                objective=float(price @ q - vE * s),
                realizable_cost=float(price @ q + p.emergency * (price @ zs) - vE * s),
                final_soc=float(s))


def level1(bank, price, p, vE, days, initial, ga_budget=(60, 120), ngrid_sweep=(40, 80, 160)):
    rows, conv = [], []
    for day in days:
        net = bank.risk(day, 0, p.alpha0)
        t0 = time.time()
        q_lp, ref_lp, _ = q3.solve(net, price, initial, p, vE)
        t_lp = time.time() - t0
        lp = evaluate_plan(q_lp, net, price, initial, p, vE)
        for ng in ngrid_sweep:
            t0 = time.time()
            q_dp, ref_dp, _ = dp_solve_clean(net, price, initial, p, vE, ngrid=ng)
            el = time.time() - t0
            dp = evaluate_plan(q_dp, net, price, initial, p, vE)
            rows.append(dict(level='L1', day=str(DATES[day].date()), method=f'DP(ngrid={ng})',
                             objective=dp['objective'], purchase_cost=dp['purchase_cost'],
                             emergency_kwh=dp['emergency_kwh'],
                             gap_vs_lp_percent=100 * (dp['objective'] - lp['objective']) / abs(lp['objective']),
                             seconds=el, ngrid=ng))
        q_rp = rule_plan(net, price, initial, p, THETA_MANUAL)
        rp = evaluate_plan(q_rp, net, price, initial, p, vE)
        rows.append(dict(level='L1', day=str(DATES[day].date()), method='RULE(手工阈值)',
                         objective=rp['objective'], purchase_cost=rp['purchase_cost'],
                         emergency_kwh=rp['emergency_kwh'],
                         gap_vs_lp_percent=100 * (rp['objective'] - lp['objective']) / abs(lp['objective']),
                         seconds=0.0, ngrid=np.nan))
        # GA 直接编码 144 维购电计划
        hi = float(max(net.max(), 0) + p.power * DT)

        def fitness(x):
            return evaluate_plan(x, net, price, initial, p, vE)['realizable_cost']

        t0 = time.time()
        best, bestfit, hist = ga_optimize(fitness, [(0.0, hi)] * 144, pop=ga_budget[0], gens=ga_budget[1],
                                          seed=1000 + day, verbose=False)
        el = time.time() - t0
        ga = evaluate_plan(best, net, price, initial, p, vE)
        rows.append(dict(level='L1', day=str(DATES[day].date()), method='GA(直接编码144维)',
                         objective=ga['objective'], purchase_cost=ga['purchase_cost'],
                         emergency_kwh=ga['emergency_kwh'],
                         gap_vs_lp_percent=100 * (ga['objective'] - lp['objective']) / abs(lp['objective']),
                         seconds=el, ngrid=np.nan))
        rows.append(dict(level='L1', day=str(DATES[day].date()), method='LP(基准)',
                         objective=lp['objective'], purchase_cost=lp['purchase_cost'],
                         emergency_kwh=lp['emergency_kwh'], gap_vs_lp_percent=0.0, seconds=t_lp, ngrid=np.nan))
        for h in hist:
            conv.append(dict(day=str(DATES[day].date()), **h))
        print(f'L1 day {DATES[day].date()} done', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(conv)


# ================================================================ L2：全年滚动策略对比
def level2(L, P, price, bank, p, initial, theta_ga):
    out = []
    solvers = [
        ('LP-MPC（基准·精确）', q3.solve),
        ('DP-MPC（SOC离散）', dp_solve_clean),
        ('RULE（手工阈值）', make_rule_solver(THETA_MANUAL)),
        ('RULE（GA调参）', make_rule_solver(theta_ga)),
    ]
    daily_store = {}
    tag_map = {'LP-MPC（基准·精确）': 'LP-MPC', 'DP-MPC（SOC离散）': 'DP-MPC',
               'RULE（手工阈值）': 'RULE-manual', 'RULE（GA调参）': 'RULE-ga'}
    for name, solver in solvers:
        t0 = time.time()
        df, _ = run_generic(L, P, price, bank, p, solver, mask=(6, 12, 18), start=31, end=365,
                            initial=initial)
        el = time.time() - t0
        csv(f'method_daily_{tag_map[name]}.csv', df)
        s = summarize(df, method=name, seconds=el, intraday_update=True)
        out.append(s)
        daily_store[name] = df
        print(f'L2 {name}: total={s["total_cost"]:,.0f} elapsed={el:.1f}s', flush=True)
    # 无日内更新（只用 0:00 计划）
    for name, solver in [('LP（不更新，仅0:00计划）', q3.solve), ('RULE（不更新，仅0:00计划）', make_rule_solver(theta_ga))]:
        t0 = time.time()
        df, _ = run_generic(L, P, price, bank, p, solver, mask=(), start=31, end=365, initial=initial)
        el = time.time() - t0
        out.append(summarize(df, method=name, seconds=el, intraday_update=False))
        daily_store[name] = df
        print(f'L2 {name}: total={df.total_cost.sum():,.0f} elapsed={el:.1f}s', flush=True)
    base = next(r for r in out if r['method'].startswith('LP-MPC'))
    base_adj = next(r for r in out if r['method'].startswith('LP-MPC'))['adjusted_cost']
    for r in out:
        r['gap_vs_lp_percent'] = 100 * (r['adjusted_cost'] - base_adj) / base_adj
    return pd.DataFrame(out), daily_store


def main():
    setup_font()
    R.exists() or R.mkdir(parents=True)
    clock = time.time()
    L, P, F, price, seed = load()
    pars = json.loads((R / 'parameters.json').read_text(encoding='utf8'))
    sel = pars['selected']
    p = replace(Params(), alpha0=sel['alpha0'], alpha_update=sel['alpha_update'],
                slope=sel['slope'], weekday=sel['weekday'],
                window=sel['window'], decay_days=sel['decay_days'],
                bias_window=sel['bias_window'], bias_decay=sel['bias_decay'] if 'bias_decay' in sel else Params().bias_decay,
                epsilon=sel['epsilon'])
    initial = float(pars['initial_feb1'])
    vE = float(price.min() / p.eta)
    bank = Bank(L, P, F, seed, p)
    dump('method_comparison_config.json',
         {'selected_params': sel, 'initial_feb1': initial, 'vE': vE, 'ngrid_default': NG_DEFAULT,
          'evaluation_period': '2025-02-01/2025-12-31', 'calibration_window_for_rule': '2025-01-08/2025-01-31',
          'setttlement': 'net_refund (same for all methods)'})

    warm = pd.read_csv(R / 'warmup_january_selected.csv')
    init_jan8 = float(warm.soc_start.iloc[7])

    # ---------------- GA 调参规则策略（只在评价期之前的数据上选参） ----------------
    print('=== GA 调参规则型策略（校准窗口 1/8—1/31，早于评价期）===', flush=True)
    trace = []

    def rule_fitness(x):
        th = dict(zip(THETA_KEYS, x))
        sol = make_rule_solver(th)
        try:
            df, _ = run_generic(L, P, price, bank, p, sol, mask=(6, 12, 18), start=7, end=31,
                                initial=init_jan8)
            return float(df.inventory_adjusted_cost.sum())
        except Exception:
            return 1e12                      # 不可行个体给极大罚值，避免中断整个 GA

    bounds = [(0.05, 0.45), (0.55, 0.95), (0.30, 0.90), (0.10, 0.60), (0.10, 1.00), (0.10, 1.00)]
    t0 = time.time()
    best_x, best_f, hist_rule = ga_optimize(rule_fitness, bounds, pop=40, gens=40, seed=20260912)
    t_ga_rule = time.time() - t0
    theta_ga = dict(zip(THETA_KEYS, [float(v) for v in best_x]))
    csv('method_ga_rule_convergence.csv', hist_rule)
    dump('method_ga_rule_result.json', {'theta': theta_ga, 'calibration_adjusted_cost': best_f,
                                        'seconds': t_ga_rule, 'bounds': bounds,
                                        'population': 40, 'generations': 40,
                                        'evaluations': 40 * 40,
                                        'note': 'GA 只优化 6 个规则参数，适应度为评价期之前的 1/8—1/31 库存校正费用'})
    print(f'GA 规则调参完成 theta={theta_ga} cost={best_f:,.1f} 用时 {t_ga_rule:.1f}s', flush=True)

    # ---------------- L1 ----------------
    print('=== L1 日前计划单阶段最优性对比 ===', flush=True)
    l1_days = list(range(40, 400, 36))[:10]
    l1, l1conv = level1(bank, price, p, vE, l1_days, initial)
    csv('method_level1.csv', l1.to_dict('records'))
    csv('method_ga_direct_convergence.csv', l1conv.to_dict('records'))

    # ---------------- L2 ----------------
    print('=== L2 全年滚动策略对比 ===', flush=True)
    l2, daily_store = level2(L, P, price, bank, p, initial, theta_ga)
    csv('method_level2.csv', l2.to_dict('records'))

    # ---------------- L3a：成本—风险 Pareto（分位谱，LP 规划器） ----------------
    print('=== L3a 成本—风险 Pareto（风险分位谱）===', flush=True)
    par = []
    for a0 in (0.45, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95):
        pp = replace(p, alpha0=a0)
        df, _ = run_generic(L, P, price, bank, pp, q3.solve, mask=(6, 12, 18), start=31, end=365,
                            initial=initial)
        s = summarize(df, method='LP', alpha0=a0)
        par.append(dict(alpha0=a0, adjusted_cost=s['adjusted_cost'], total_cost=s['total_cost'],
                        cvar95=s['cvar95'], emergency_kwh=s['emergency_kwh'],
                        daily_std=float(df.total_cost.std()), max_daily=float(df.total_cost.max())))
        print(f'  alpha0={a0} cost={s["adjusted_cost"]:,.0f} CVaR95={s["cvar95"]:,.1f}', flush=True)
    ppareto = pd.DataFrame(par)
    csv('method_pareto_cost_risk.csv', ppareto.to_dict('records'))

    # ---------------- L3b：成本—寿命 Pareto（NSGA-II 调规则参数） ----------------
    print('=== L3b 成本—寿命 Pareto（NSGA-II）===', flush=True)

    def mo(x):
        th = dict(zip(THETA_KEYS, x))
        sol = make_rule_solver(th)
        try:
            df, _ = run_generic(L, P, price, bank, p, sol, mask=(6, 12, 18), start=7, end=31,
                                initial=init_jan8)
            return np.array([df.inventory_adjusted_cost.sum(), df.discharge_kwh.sum() / (2 * 12000.0)])
        except Exception:
            return np.array([1e12, 1e6])

    t0 = time.time()
    Pn, Fn, fronts = nsga2(mo, bounds, pop=20, gens=15, seed=20260913)
    t_nsga = time.time() - t0
    front0 = np.array([Fn[i] for i in fronts[0]])
    csv('method_pareto_cost_life.csv', [dict(cost=float(a), equivalent_cycles=float(b))
                                        for a, b in front0])
    dump('method_nsga2_result.json', {'front_size': int(len(front0)), 'seconds': t_nsga,
                                      'population': 20, 'generations': 15, 'evaluations': 300,
                                      'objectives': ['adjusted_cost (元)', 'equivalent_cycles (次)'],
                                      'window': '2025-01-08/2025-01-31'})
    print(f'NSGA-II 完成 front={len(front0)} 用时 {t_nsga:.1f}s', flush=True)

    # ---------------- 汇总 ----------------
    summary_json = {'level1': json.loads(l1.groupby('method').gap_vs_lp_percent.mean().to_json()),
                    'level2': l2.to_dict('records'),
                    'ga_rule_theta': theta_ga,
                    'elapsed_seconds': time.time() - clock}
    dump('method_comparison_summary.json', summary_json)
    print('=== 全部完成，用时 %.1f 秒 ===' % (time.time() - clock), flush=True)


if __name__ == '__main__':
    main()
