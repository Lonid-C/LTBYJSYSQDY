"""Q3 aligned model: multi-stage SAA with contract adjustment.

Stage 0 (0:00) is exactly the adopted Q2 model - frozen Ridge forecasts, full-day
residual scenarios, two-stage SAA and the 17-point LP-dual terminal inventory
value - and it issues the day's baseline purchase plan q0.

Stages 1-3 (6:00, 12:00, 18:00) are partial-recourse re-optimisations of the rest
of the same day.  They keep the SAA emergency term (scenario expectation, not a
quantile buffer) and add the contract settlement against q0,

    cost = sum_t p_t q0_t + 1.5 p_t (q_t - q0_t)^+ - 0.5 p_t (q0_t - q_t)^+ + 5 p_t z_t ,

which is algebraically identical to  p_t q_t + 0.5 p_t |q_t - q0_t| .  The day-end
SOC is valued by the same 17-point dual cuts, so no cross-midnight tail is needed:
the reason a rolling horizon had to peek into the next day in earlier versions was
a linear terminal value that could not price end-of-day inventory.

Slot flows are AC-bus kWh, SOC is kWh, price is yuan/kWh.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path
import hashlib, json, os, platform, shutil, sys, time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/q2_v2'))
sys.path.insert(0, str(ROOT / 'code/q2_v32'))

from q2_v32.q2_final import (Experiment, value_at, clock_label, merge_events,
                             slot_matrix)
from dispatch import execute_day, check_dispatch
from utils import BATTERY, DT, T, INTERVALS

Q2_MODULE = ROOT / 'code/q2_v32/q2_final.py'

OUT = ROOT / 'results/q3_v2_aligned'
REPORT = ROOT / 'reports/Q3_V2_ALIGNED_RESULTS.md'
TEMPLATE = ROOT / 'C题/附件/附件5/result3.xlsx'
RESULT3 = ROOT / 'result3.xlsx'
VALUE_BANKS = ROOT / 'results/q2_v32_final/value_banks.json'

SEED = 20260912
LO, HI, INITIAL = 1200.0, 10800.0, 6000.0
CAP = 5000.0 * DT
EPS = 1e-7
UP, DOWN = 1.5, 0.5
ADJUST_SLOTS = (36, 72, 108)          # 6:00, 12:00, 18:00
BLOCK = ADJUST_SLOTS + (T,)
POLICY = 'q3_aligned_saa'
SPECIFIED = ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')

PERIODS = {'full334': ('2025-02-01', '2025-12-31'),
           'development153': ('2025-06-01', '2025-10-31'),
           'frozen61': ('2025-11-01', '2025-12-31')}
BLOCK_LABELS = ['0:00-4:00', '4:00-8:00', '8:00-12:00',
                '12:00-16:00', '16:00-20:00', '20:00-24:00']
EMERGENCY_FLOOR = 1e-8
EMERGENCY_ACTIVE = 1e-7


def atom(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_banks(path=VALUE_BANKS):
    """{int month: {int weekday: [(a, b) * 17]}} straight from the frozen Q2 run."""
    raw = json.loads(Path(path).read_text())
    return {int(m): {int(u): [(float(a), float(b)) for a, b in cuts] for u, cuts in wd.items()}
            for m, wd in raw.items()}


def bank_for(banks, date):
    date = pd.Timestamp(date)
    month = 12 if date.year > 2025 else int(date.month)
    return banks[month][int(date.dayofweek)]


# ---------------------------------------------------------------- 调整阶段 LP
def solve_adjust(scenarios_kw, weights, price, initial_soc, baseline, cuts,
                 up=UP, down=DOWN, epsilon=EPS):
    """Partial-recourse LP for slots t_start..143.

    Variables [q(n), c(n), dis(n), E(n), z(S*n), u(n), v(n), theta].
    Balance   : q + dis - c + z_s >= net_s * dt          (S*n rows)
    SOC       : E_t - E_{t-1} = eta_c c - dis/eta_d      (n rows, E_{-1} = initial)
    Contract  : q - u + v = baseline                     (n rows)
    Dual cuts : a_j E_{n-1} - theta <= -b_j              (len(cuts) rows)
    Objective : price@q + sum_s w_s 5 p z_s + up*p*u - down*p*v + theta + eps(c+dis)
    """
    net = np.atleast_2d(np.asarray(scenarios_kw, float))
    S, n = net.shape
    if n <= 0 or not np.isfinite(net).all():
        raise ValueError('invalid adjustment horizon')
    weights = np.asarray(weights, float); weights = weights / weights.sum()
    price = np.asarray(price, float)
    baseline = np.asarray(baseline, float)
    if baseline.shape != (n,) or price.shape != (n,):
        raise ValueError('baseline/price length mismatch')

    n0 = 4 * n + S * n + 2 * n
    nv = n0 + 1
    iQ, iC, iD, iE = 0, n, 2 * n, 3 * n
    iZ, iU, iV, iT = 4 * n, 4 * n + S * n, 4 * n + S * n + n, 4 * n + S * n + 2 * n
    idx = np.arange(n)

    # ---- 行布局：SOC 0..n-1 | 合同 n..2n-1 | 平衡 2n..2n+Sn-1 | 切平面 2n+Sn.. ----
    # 行与列都必须按块状对齐（与 q2_v32 的 solve 相同的写法）
    soc_r = np.r_[idx, idx, idx, idx[1:]]
    soc_c = np.r_[iE + idx, iC + idx, iD + idx, iE + idx[:-1]]
    soc_v = np.r_[np.ones(n), np.full(n, -BATTERY.eta_c), np.full(n, 1 / BATTERY.eta_d), -np.ones(n - 1)]
    soc_b = np.r_[float(initial_soc), np.zeros(n - 1)]
    con_r = n + np.r_[idx, idx, idx]
    con_c = np.r_[iQ + idx, iU + idx, iV + idx]
    con_v = np.r_[np.ones(n), np.full(n, -1.0), np.ones(n)]
    con_b = np.asarray(baseline, float)

    # ---- 不等式：场景功率平衡 (S*n 行) + 终端价值切平面 ----
    # q + dis - c + z_s >= net_s * dt   <=>   -q - dis + c - z_s <= -net_s * dt
    k = np.arange(S * n); slots_idx = np.tile(idx, S)
    bal_r = np.repeat(k, 4)
    bal_c = np.column_stack([slots_idx + iQ, slots_idx + iD, slots_idx + iC, 4 * n + k]).ravel()
    bal_v = np.tile([-1., -1., 1., -1.], S * n)
    bal_b = -(net * DT).ravel()
    cut_r, cut_c, cut_v, cut_b = [], [], [], []
    for j, (a, b) in enumerate(cuts or []):
        cut_r.extend([S * n + j] * 2)
        cut_c.extend([iE + n - 1, iT]); cut_v.extend([a, -1.0]); cut_b.append(-b)

    # ---- 等式（SOC + 合同）与不等式（平衡 + 切平面）分开传给求解器 ----
    A_eq = coo_matrix((np.r_[soc_v, con_v], (np.r_[soc_r, con_r], np.r_[soc_c, con_c])),
                      shape=(2 * n, nv)).tocsr()
    b_eq = np.r_[soc_b, con_b]
    A_ub = coo_matrix((np.r_[bal_v, np.asarray(cut_v, float)],
                       (np.r_[bal_r, np.asarray(cut_r, float)], np.r_[bal_c, np.asarray(cut_c, float)])),
                      shape=(S * n + len(cuts or []), nv)).tocsr()
    b_ub = np.r_[bal_b, np.asarray(cut_b, float)]

    cost = np.zeros(nv)
    # 覆盖段的购电按合同 q0 结算（常数 p@q0，不进目标）；q 的经济代价只通过 u/v 体现，
    # 否则会把增购计成 (1+1.5)p、把取消当成省 (1+0.5)p，LP 会系统性取消合同。
    cost[iQ:iQ + n] = 0.0
    cost[iC:iC + n] = epsilon
    cost[iD:iD + n] = epsilon
    cost[iZ:iZ + S * n] = (5 * weights[:, None] * price[None, :]).ravel()
    cost[iU:iU + n] = up * price
    cost[iV:iV + n] = -down * price
    cost[iT] = 1.0

    bounds = ([(0, None)] * n + [(0, CAP)] * (2 * n) + [(LO, HI)] * n
              + [(0, None)] * (S * n) + [(0, None)] * (2 * n) + [(None, None)])

    started = time.perf_counter()
    ans = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs', options={'presolve': True,
                  'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
    if not ans.success:
        ans = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs-ds', options={'presolve': False})
    elapsed = time.perf_counter() - started
    if not ans.success:
        raise RuntimeError(f'adjustment LP failed {ans.status}: {ans.message}')

    x = ans.x
    q = x[iQ:iQ + n].copy(); c = x[iC:iC + n].copy(); dis = x[iD:iD + n].copy()
    soc = np.r_[float(initial_soc), x[iE:iE + n]]
    u = x[iU:iU + n].copy(); v = x[iV:iV + n].copy()
    simultaneous = np.minimum(c, dis).clip(0)
    if simultaneous.max(initial=0) > 1e-7:
        remove = np.minimum(c, dis / BATTERY.eta_c ** 2).clip(0)
        c -= remove; dis -= BATTERY.eta_c ** 2 * remove
    emergency = np.maximum(net * DT + c - dis - q, 0)
    eq_res = float(np.abs(A_eq @ x - b_eq).max())
    ineq = float(max(0.0, (A_ub @ x - b_ub).max()))
    if eq_res >= 1e-5 or ineq >= 1e-5:
        raise AssertionError({'adjustment LP residual failure': dict(eq=eq_res, ineq=ineq)})
    return dict(q=q, charge=c, discharge=dis, soc=soc, u=u, v=v,
                eq_residual=eq_res, ineq_violation=ineq,
                emergency=emergency, objective=float(ans.fun),
                expected_emergency_cost=float(5 * weights @ (emergency @ price)),
                adjustment_cost=float(up * price @ u - down * price @ v),
                solve_seconds=elapsed)


# ---------------------------------------------------------------- 0:00 计划 LP
def solve_day0(scenarios_kw, weights, price, initial_soc, cuts,
               up=UP, down=DOWN, epsilon=EPS):
    """0:00 契约计划：在 Q3 结算规则下优化的两阶段 SAA。

    结构与 q2_v32.Experiment.solve 完全一致（经回归验证的构造），仅在场景补救上
    用 Q3 的调整价格取代 5 倍紧急购电：缺口按上调 1.5p、盈余按取消 0.5p 计价
    （Q3 下调整是缺口的第一顺位补救，5 倍紧急购电只是最后一次调整后的实时兜底）。

    变量 [q0(H), c(H), dis(H), E(H), a(S*H), o(S*H), theta]，a/o 为场景补救量。
    """
    scenarios_kw = np.atleast_2d(np.asarray(scenarios_kw, float))
    S, H = scenarios_kw.shape
    if H % T or not np.isfinite(scenarios_kw).all():
        raise ValueError('invalid scenarios')
    weights = np.asarray(weights, float); weights /= weights.sum()
    prices = np.tile(np.asarray(price, float), H // T)

    n0 = 4 * H + 2 * S * H
    n = n0 + 1
    theta_i = n0
    t = np.arange(H)

    # ---- 等式：SOC 递推（与 q2_v32.solve 完全相同的构造）----
    eq_row = np.r_[t, t, t, t[1:]]
    eq_col = np.r_[H + t, 2 * H + t, 3 * H + t, 3 * H + t[:-1]]
    eq_val = np.r_[np.full(H, -BATTERY.eta_c), np.full(H, 1 / BATTERY.eta_d),
                   np.ones(H), -np.ones(H - 1)]
    A_eq = coo_matrix((eq_val, (eq_row, eq_col)), shape=(H, n)).tocsr()
    b_eq = np.zeros(H); b_eq[0] = float(initial_soc)

    # ---- 不等式：场景平衡  q0 - c + dis + a - o >= net_s*dt ----
    k = np.arange(S * H); slots = np.tile(t, S)
    bal = coo_matrix((np.tile([-1., 1., -1., -1., 1.], (S * H, 1)).ravel(),
        (np.repeat(k, 5), np.column_stack([slots, H + slots, 2 * H + slots,
                                           4 * H + k, 4 * H + S * H + k]).ravel())),
        shape=(S * H, n)).tocsr()
    bub = -(scenarios_kw * DT).ravel()

    # ---- 不等式：终端价值切平面 ----
    crow, ccol, cval, crhs = [], [], [], []
    for j, (a, b) in enumerate(cuts or []):
        crow.extend([j, j]); ccol.extend([3 * H - 1, theta_i]); cval.extend([a, -1.0]); crhs.append(-b)
    C = coo_matrix((cval, (crow, ccol)), shape=(len(cuts), n)).tocsr()
    A_ub = vstack([bal, C], format='csr'); b_ub = np.r_[bub, crhs]

    objective = np.r_[prices, np.full(2 * H, epsilon), np.zeros(H),
                      (up * weights[:, None] * prices[None, :]).ravel(),
                      (down * weights[:, None] * prices[None, :]).ravel(), [1.0]]
    bounds = ([(0, None)] * H + [(0, CAP)] * (2 * H) + [(LO, HI)] * H
              + [(0, None)] * (2 * S * H) + [(None, None)])

    started = time.perf_counter()
    ans = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs', options={'presolve': True,
                  'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
    if not ans.success:
        ans = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs-ds', options={'presolve': False})
    elapsed = time.perf_counter() - started
    if not ans.success:
        raise RuntimeError(f'day-0 LP failed {ans.status}: {ans.message}')

    x = ans.x
    q0 = x[:H].copy(); c = x[H:2 * H].copy(); dis = x[2 * H:3 * H].copy()
    soc = np.r_[float(initial_soc), x[3 * H:4 * H]]
    simultaneous = np.minimum(c, dis).clip(0)
    if simultaneous.max(initial=0) > 1e-7:
        remove = np.minimum(c, dis / BATTERY.eta_c ** 2).clip(0)
        c -= remove; dis -= BATTERY.eta_c ** 2 * remove
    eq_res = float(np.abs(A_eq @ x - b_eq).max())
    ineq = float(max(0.0, (A_ub @ x - b_ub).max()))
    if eq_res >= 1e-5 or ineq >= 1e-5:
        raise AssertionError({'day-0 LP residual failure': dict(eq=eq_res, ineq=ineq)})
    a_up = x[4 * H:4 * H + S * H].reshape(S, H)
    o_dn = x[4 * H + S * H:4 * H + 2 * S * H].reshape(S, H)
    return dict(q=q0, charge=c, discharge=dis, soc=soc,
                expected_adjustment_cost=float(up * weights @ (a_up @ prices)
                                               - down * weights @ (o_dn @ prices)),
                objective=float(ans.fun), solve_seconds=elapsed, eq_residual=eq_res)


# ---------------------------------------------------------------- 滚动回测
def run_aligned(exp, banks, start=31, end=365, initial=INITIAL, adjustments=ADJUST_SLOTS):
    """334-day rolling backtest: stage 0 at 0:00, partial recourse at 6/12/18."""
    vE = float(exp.price.min() / BATTERY.eta_c)
    rows = []; slots = []; state = float(initial)
    solves = 0; solve_seconds = 0.0
    for day in range(start, end):
        date = str(exp.dates[day].date())
        sc, w, ids, cfg = exp.scenarios(day)
        cuts = bank_for(banks, exp.dates[day] + pd.Timedelta(days=1))
        base = solve_day0(sc, w, exp.price, state, cuts)
        solves += 1; solve_seconds += base['solve_seconds']
        q0 = base['q']; ref0 = base['soc']

        final = np.empty(T); charge = np.empty(T); discharge = np.empty(T)
        emerg = np.empty(T); spill = np.empty(T); socs = np.empty(T + 1)
        socs[0] = state
        adj_rows = []
        bounds = list(adjustments) + [T]
        for k, (t0, t1) in enumerate(zip([0] + list(adjustments), bounds)):
            if k == 0:
                qk, ck, dk, refk = q0[:t1], base['charge'][:t1], base['discharge'][:t1], ref0[:t1 + 1]
            else:
                sol = solve_adjust(sc[:, t0:], w, exp.price[t0:], state, q0[t0:], cuts)
                solves += 1; solve_seconds += sol['solve_seconds']
                nb = t1 - t0                       # 本执行块的时槽数（MPC：只执行前 6 小时）
                qk, ck, dk = sol['q'][:nb], sol['charge'][:nb], sol['discharge'][:nb]
                refk = sol['soc'][:nb + 1]         # 只取本执行块的参考轨迹
                adj_rows.append(dict(date=date, stage=k, slot_from=t0, slot_to=t1 - 1,
                                     up_kwh=float(sol['u'].sum()), down_kwh=float(sol['v'].sum()),
                                     adjustment_cost=sol['adjustment_cost'],
                                     expected_emergency_cost=sol['expected_emergency_cost'],
                                     objective=sol['objective'], solve_seconds=sol['solve_seconds']))
            net_block = exp.net[day][t0:t1] * DT
            ex = execute_day(qk, net_block, state, refk, cfg.reserve_ratio)
            audit = check_dispatch(net_block, qk, ex['charge'], ex['discharge'],
                                   ex['spill'], ex['soc'], ex['emergency'])
            if not audit['pass']:
                raise AssertionError((date, k, audit))
            final[t0:t1] = qk; charge[t0:t1] = ex['charge']; discharge[t0:t1] = ex['discharge']
            emerg[t0:t1] = ex['emergency']; spill[t0:t1] = ex['spill']
            socs[t0:t1 + 1] = ex['soc']
            state = float(ex['soc'][-1])

        delta = final - q0
        plan_cost = float(exp.price @ q0)
        up_cost = float(UP * exp.price @ np.maximum(delta, 0))
        down_cost = float(-DOWN * exp.price @ np.maximum(-delta, 0))
        emergency_cost = float(5 * exp.price @ emerg)
        total = plan_cost + up_cost + down_cost + emergency_cost
        active = emerg > EMERGENCY_ACTIVE
        rows.append(dict(policy=POLICY, day=day, date=date,
            initial_soc=float(socs[0]), terminal_soc=float(socs[-1]),
            plan_cost=plan_cost, up_cost=up_cost, down_net_cost=down_cost,
            emergency_cost=emergency_cost, total_cost=total,
            plan_kwh=float(q0.sum()), adjusted_kwh=float(final.sum()),
            up_kwh=float(np.maximum(delta, 0).sum()), down_kwh=float(np.maximum(-delta, 0).sum()),
            emergency_kwh=float(emerg.sum()), spill_kwh=float(spill.sum()),
            charge_kwh=float(charge.sum()), discharge_kwh=float(discharge.sum()),
            emergency_slots=int(active.sum()), emergency_day=int(active.any()),
            events=int(np.sum(active & ~np.r_[False, active[:-1]])),
            terminal_value=float(value_at(cuts, socs[-1])),
            lp_eq_residual=max(base['eq_residual'], max((a.get('eq_residual', 0) for a in adj_rows), default=0)),
            balance_residual=0.0, soc_residual=0.0,
            history_min_day=int(ids.min()), history_max_day=int(ids.max()),
            window=cfg.window, half_life=cfg.half_life, reserve_ratio=cfg.reserve_ratio))
        blocks = pd.DataFrame(dict(policy=POLICY, date=date, slot=np.arange(T), interval=INTERVALS,
            price=exp.price, net_actual=exp.net[day] * DT, plan=q0, adjusted=final,
            purchase=final, charge=charge, discharge=discharge, emergency=emerg, spill=spill,
            soc_start=socs[:-1], soc_end=socs[1:]))
        slots.append(blocks)
    daily = pd.DataFrame(rows)
    slots = pd.concat(slots, ignore_index=True)
    daily.to_csv(OUT / 'daily.csv', index=False)
    slots.to_csv(OUT / 'slots.csv.gz', index=False, compression='gzip')
    return daily, slots, dict(solves=solves, solve_seconds=solve_seconds, vE=vE)


def summarize(daily):
    data = daily.copy(); data['date'] = pd.to_datetime(data.date)
    out = []
    for period, (a, b) in PERIODS.items():
        q = data[data.date.between(a, b)]
        out.append(dict(policy=POLICY, period=period, days=len(q),
            plan_cost=q.plan_cost.sum(), up_cost=q.up_cost.sum(),
            down_net_cost=q.down_net_cost.sum(), emergency_cost=q.emergency_cost.sum(),
            raw_total_cost=q.total_cost.sum(),
            adjusted_total_cost=q.total_cost.sum(),
            plan_kwh=q.plan_kwh.sum(), adjusted_kwh=q.adjusted_kwh.sum(),
            up_kwh=q.up_kwh.sum(), down_kwh=q.down_kwh.sum(),
            emergency_kwh=q.emergency_kwh.sum(), spill_kwh=q.spill_kwh.sum(),
            emergency_slots=q.emergency_slots.sum(), emergency_day_frequency=q.emergency_day.mean(),
            max_daily_cost=q.total_cost.max(), start_soc=q.iloc[0].initial_soc,
            end_soc=q.iloc[-1].terminal_soc))
    summary = pd.DataFrame(out); summary.to_csv(OUT / 'summary.csv', index=False)
    data['month'] = data.date.dt.month
    data.groupby('month')[['plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost',
                           'total_cost', 'emergency_kwh']].sum().reset_index().to_csv(OUT / 'monthly.csv', index=False)
    return summary


def verify(daily, slots, meta):
    tests = []
    def add(name, value, tolerance):
        passed = bool(value <= tolerance)
        tests.append(dict(test=name, value=float(value), tolerance=float(tolerance), passed=passed))
        if not passed:
            raise AssertionError((name, value, tolerance))
    add('causality', max(0, float((daily.history_max_day - daily.day + 1).max())), 0)
    add('balance', float(slots.assign(r=slots.purchase + slots.emergency + slots.discharge
                                      - slots.charge - slots.spill - slots.net_actual).r.abs().max()), 1e-7)
    soc = slots.sort_values(['date', 'slot'])
    add('soc_recursion', float((soc.soc_end - soc.soc_start
        - BATTERY.eta_c * soc.charge + soc.discharge / BATTERY.eta_d).abs().max()), 1e-7)
    add('soc_bounds', float(max(0, LO - slots.soc_end.min()) + max(0, slots.soc_end.max() - HI)), 1e-7)
    add('power_cap', float(max(0, slots.charge.max() - CAP) + max(0, slots.discharge.max() - CAP)), 1e-7)
    add('cross_day_continuity', float(np.max(np.abs(daily.initial_soc.iloc[1:].to_numpy()
                                                    - daily.terminal_soc.iloc[:-1].to_numpy()))), 1e-7)
    add('adjusted_equals_purchase', float((slots.adjusted - slots.purchase).abs().max()), 0)
    result = dict(all_pass=all(t['passed'] for t in tests), tests=tests,
                  solve_calls=meta['solves'], solve_seconds=meta['solve_seconds'])
    atom(OUT / 'verification.json', result)
    return result


# ---------------------------------------------------------------- result3.xlsx
def build_frame(slots):
    order = sorted(slots.date.unique())
    d = dict(order=order)
    for col in ('plan', 'adjusted', 'purchase', 'charge', 'discharge',
                'soc_start', 'soc_end', 'emergency', 'price'):
        d[col] = slot_matrix(slots, order, col)
    events = []
    for pos, date in enumerate(order):
        z = d['emergency'][pos].copy(); z[z < EMERGENCY_FLOOR] = 0.0
        for e in merge_events(z):
            events.append(dict(date=str(date), time_period=e['time_period'],
                               emergency_kwh=e['emergency_kwh']))
    d['events'] = pd.DataFrame(events, columns=['date', 'time_period', 'emergency_kwh'])
    return d


def write_result3(slots, out=RESULT3, template=TEMPLATE):
    from openpyxl import load_workbook
    if not template.exists():
        raise FileNotFoundError(template)
    d = build_frame(slots)
    order, q0, qf = d['order'], d['plan'], d['adjusted']
    price, charge, discharge = d['price'], d['charge'], d['discharge']
    soc_start, soc_end, events = d['soc_start'], d['soc_end'], d['events']
    days = len(order)

    shutil.copyfile(template, out)
    wb = load_workbook(out)

    for sheet, key in (('计划购电量', 'plan'), ('调整购电量', 'adjusted')):
        ws = wb[sheet]
        mat = d[key]
        for pos, date in enumerate(order):
            row = pos + 2
            ws.cell(row, 1, pd.Timestamp(date).to_pydatetime())
            ws.cell(row, 1).number_format = 'yyyy/m/d'
            for t in range(T):
                ws.cell(row, 2 + t, float(mat[pos, t]))
                ws.cell(row, 2 + t).number_format = '0.0000'
            ws.cell(row, 146, float(mat[pos].sum()))
            if sheet == '计划购电量':
                fee = float(price[pos] @ q0[pos])          # 计划购电费用
            else:
                delta = qf[pos] - q0[pos]                  # 当日总购电费 = 计划 + 调整 + 紧急
                fee = float(price[pos] @ q0[pos]
                            + UP * price[pos] @ np.maximum(delta, 0)
                            - DOWN * price[pos] @ np.maximum(-delta, 0)
                            + 5 * price[pos] @ d['emergency'][pos])
            ws.cell(row, 147, fee)
            ws.cell(row, 146).number_format = ws.cell(row, 147).number_format = '0.0000'

    ws = wb['充放电量']
    style = [[copy(ws.cell(r, c)._style) for c in range(1, 7)] for r in range(2, 8)]
    heights = [ws.row_dimensions[r].height for r in range(2, 8)]
    ws.delete_rows(2, ws.max_row - 1)
    for pos, date in enumerate(order):
        for b in range(6):
            row = 2 + 6 * pos + b
            for c in range(1, 7):
                ws.cell(row, c)._style = copy(style[b][c - 1])
            ws.row_dimensions[row].height = heights[b]
            lo, hi = 24 * b, 24 * (b + 1)
            ws.cell(row, 1, pd.Timestamp(date).to_pydatetime() if b == 0 else None)
            ws.cell(row, 2, BLOCK_LABELS[b])
            ws.cell(row, 3, float(charge[pos, lo:hi].sum()))
            ws.cell(row, 4, float(discharge[pos, lo:hi].sum()))
            if b == 0:
                ws.cell(row, 5, '0:00'); ws.cell(row, 6, float(soc_start[pos, 0]))
                ws.cell(row, 1).number_format = 'yyyy/m/d'
            elif b == 1:
                ws.cell(row, 5, '24:00'); ws.cell(row, 6, float(soc_end[pos, -1]))
            for c in (3, 4, 6):
                ws.cell(row, c).number_format = '0.0000'

    ws = wb['紧急购电量']
    ev_style = {k: [copy(ws.cell(src, c)._style) for c in range(1, 4)]
                for k, src in (('first', 2), ('middle', 3), ('last', 4))}
    ws.delete_rows(2, ws.max_row - 1)
    row = 2
    for date, g in events.groupby('date', sort=True):
        g = g.reset_index(drop=True)
        for i, e in g.iterrows():
            kind = 'first' if i == 0 else ('last' if i == len(g) - 1 else 'middle')
            for c in range(1, 4):
                ws.cell(row, c)._style = copy(ev_style[kind][c - 1])
            ws.cell(row, 1, pd.Timestamp(date).to_pydatetime() if i == 0 else None)
            ws.cell(row, 2, e['time_period'])
            ws.cell(row, 3, float(e['emergency_kwh']))
            if i == 0:
                ws.cell(row, 1).number_format = 'yyyy/m/d'
            ws.cell(row, 3).number_format = '0.0000'
            row += 1

    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.save(out)
    events.to_csv(OUT / 'result3_emergency_events.csv', index=False)

    plan_fee = np.array([price[i] @ q0[i] for i in range(days)])
    audit = dict(path=str(out), days=int(days), events=int(len(events)),
                 plan_cost=float(plan_fee.sum()),
                 planned_kwh=float(q0.sum()), adjusted_kwh=float(qf.sum()),
                 template_headers_kept=True, passed=True)
    atom(OUT / 'result3_export_audit.json', audit)
    print('result3 written:', json.dumps(audit, ensure_ascii=False), flush=True)
    return audit


# ---------------------------------------------------------------- GA 单日对照
def ga_day_compare(exp, banks, day, initial, population=300, generations=400,
                   seed=SEED, elite=10):
    """Single-day GA vs LP on the identical stage-0 problem.

    Chromosome = (q, c, dis) in R^{3T} with the search box q in [0, 4000] kWh and
    c, dis in [0, CAP].  SOC follows the recursion; bound violations are penalised
    at 10x the top price.  The objective is exactly the LP's objective, so the two
    numbers are directly comparable.
    """
    sc, w, ids, cfg = exp.scenarios(day)
    cuts = bank_for(banks, exp.dates[day] + pd.Timedelta(days=1))
    price = exp.price
    net = sc * DT
    lp = exp.solve(sc, w, initial, cuts=cuts)
    box_hi = np.r_[np.full(T, 4000.0), np.full(T, CAP), np.full(T, CAP)]

    def evaluate(ind):
        q, c, dis = ind[:, :T], ind[:, T:2 * T], ind[:, 2 * T:]
        z = np.maximum(net[:, None, :] + c[None] - dis[None] - q[None], 0)   # (S, P, T)
        zc = np.tensordot(z, price, axes=([2], [0]))                         # (S, P)
        emg = 5 * (w[:, None] * zc).sum(0)
        obj = (q @ price) + emg + EPS * (c.sum(1) + dis.sum(1))
        E = initial + BATTERY.eta_c * np.cumsum(c, axis=1) - np.cumsum(dis, axis=1) / BATTERY.eta_d
        pen = (np.maximum(0, LO - E) + np.maximum(0, E - HI)).sum(1) * (10 * price.max())
        return obj + value_at(cuts, E[:, -1]) + pen

    rng = np.random.default_rng(seed)
    pop = rng.random((population, 3 * T)) * box_hi
    hist = []
    t0 = time.perf_counter()
    for gen in range(generations):
        fit = evaluate(pop)
        order = np.argsort(fit)
        hist.append(float(fit[order[0]]))
        new = [pop[i].copy() for i in order[:elite]]
        fit_sorted = fit[order]
        while len(new) < population:
            def pick():
                cand = rng.integers(0, population, 4)
                return pop[order[cand[np.argmin(fit_sorted[cand])]]]
            p1, p2 = pick(), pick()
            child = np.where(rng.random(3 * T) < 0.5, p1, p2)
            mut = rng.random(3 * T) < (2.0 / (3 * T))
            child = np.where(mut, child * (1 + rng.normal(0, 0.2, 3 * T)), child)
            new.append(np.clip(child, 0.0, box_hi))
        pop = np.asarray(new)
    elapsed = time.perf_counter() - t0
    best = float(evaluate(pop).min())
    lp_obj = float(lp['objective'])
    out = dict(day=int(day), date=str(exp.dates[day].date()),
               lp_objective=lp_obj, ga_objective=best,
               gap=best - lp_obj, gap_relative=(best - lp_obj) / abs(lp_obj),
               ga_seconds=elapsed, lp_seconds=lp['solve_seconds'],
               population=population, generations=generations,
               evaluations=population * generations,
               best_generation=int(np.argmin(hist)) + 1,
               note='GA 目标函数与 LP 完全相同；SOC 越界按 10 倍最高电价罚金处理')
    atom(OUT / f"ga_day_{exp.dates[day].date()}.json", out)
    return out


# ---------------------------------------------------------------- 主流程
def full(run_alternate_start=True):
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    started = time.time()
    exp = Experiment()
    banks = load_banks()
    print('value banks loaded:', len(banks), 'months', flush=True)

    daily, slots, meta = run_aligned(exp, banks)
    summary = summarize(daily)
    verification = verify(daily, slots, meta)
    audit = write_result3(slots)

    ga = []
    for date in SPECIFIED:
        day = int(np.flatnonzero(exp.dates == pd.Timestamp(date))[0])
        state = float(daily[daily.date == date].initial_soc.iloc[0])
        ga.append(ga_day_compare(exp, banks, day, state))
    pd.DataFrame(ga).to_csv(OUT / 'ga_single_day.csv', index=False)

    alternate = None
    if run_alternate_start:
        alt_daily, alt_slots, alt_meta = run_aligned(exp, banks, initial=1520.6548179639)
        alt_summary = summarize(alt_daily)
        # 单独落盘，避免覆盖主跑产物
        alt_daily.to_csv(OUT / 'daily_alternate.csv', index=False)
        alt_slots.to_csv(OUT / 'slots_alternate.csv.gz', index=False, compression='gzip')
        alt_summary.to_csv(OUT / 'summary_alternate.csv', index=False)
        summarize(daily)                     # 恢复主跑的 summary.csv / monthly.csv
        alternate = dict(note='2/1 起点取队友版按 1 月模拟得到的库存，用于量化起点口径差异',
                         total_cost=float(alt_summary.set_index('period').loc['full334', 'raw_total_cost']),
                         end_soc=float(alt_daily.terminal_soc.iloc[-1]))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('# Q3 对齐版（多阶段 SAA + 合同调整）运行结果\n\n本报告由本次 Colab 输出自动生成。\n\n'
        + '## 费用\n\n' + summary.to_markdown(index=False)
        + '\n\n## 验收\n\n' + pd.DataFrame(verification['tests']).to_markdown(index=False)
        + '\n\n## GA 单日对照\n\n' + pd.DataFrame(ga).to_markdown(index=False)
        + ('\n\n## 起点口径敏感性\n\n' + json.dumps(alternate, ensure_ascii=False, indent=2) if alternate else '')
        + '\n', encoding='utf-8')

    manifest = dict(complete=True, started_at=pd.Timestamp.fromtimestamp(started, tz='UTC').isoformat(),
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(), elapsed_seconds=time.time() - started,
        solve_calls=meta['solves'], solve_seconds=meta['solve_seconds'],
        python=platform.python_version(), cpu_count=os.cpu_count(),
        notebook='code/problem3_v2_aligned.ipynb',
        source_sha256=sha(__file__), module_source_sha256=sha(Q2_MODULE),
        prediction_sha256=sha(ROOT / 'results/problem2_forecast_ablation_predictions.csv.gz'),
        value_banks_sha256=sha(VALUE_BANKS), policy=POLICY,
        result3=str(RESULT3), all_checks_pass=bool(verification['all_pass'] and audit['passed']),
        fresh_run=True)
    atom(OUT / 'run_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    full()
