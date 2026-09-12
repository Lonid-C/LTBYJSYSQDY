"""Q3 融合版：队友滚动 MPC 框架 + Q2 冻结 Ridge 负荷预测 + Q2 17 点对偶终端价值。

相对 q3.py 只改两处，其余机制（发布时刻、接受机制、净额结算、执行规则、
校准协议、消融与敏感性套件）全部原样保留：

  1. 负荷剖面（FusionBank）：0:00 之后的滚动预报基底换成 Q2 口径的冻结 Ridge
     预测（Study.fc['Ridge']，前 90 天用审计过的因果冷启动 legacy 剖面，与
     Q2 生产版完全同源）。队友的日内偏差修正（过去 1 小时中位数 + 6 小时衰减）
     与附件 3 光伏更新原样保留。
  2. 终端库存价值（solve_f / run_f）：标量 vE = p_min/eta 换成 Q2 学到的
     17 点对偶切平面，按"视野末端所属次日"的月份/星期取切平面（θ 变量 +
     支撑行，仍是 LP）。0:00 发布的 24 小时视野恰好终止于次日 0:00，与 Q2
     的终端定义严格一致；6/12/18 视野终止于次日 h:00，沿用次日切平面是
     显式声明的近似（只多计次日已过去 h 小时的库存价值，方向为略偏保守）。

报告口径：daily 的 inventory_adjusted_cost 仍按 q3 的标量 vE 记账，保证
两版 summary 列完全同构、可直接对比；LP 内部终端用切平面。
"""
from pathlib import Path
import json, sys, time

import numpy as np
import pandas as pd

import q3

# ---------------------------------------------------------------- Q2 资产定位
def _project_root(start):
    """按仓库根独有的 C题/附件/附件2.xlsx 向上查找（不依赖本文件深度）。"""
    for c in (start, *start.parents):
        if (c / 'C题/附件/附件2.xlsx').exists():
            return c
    raise FileNotFoundError('未找到含 C题/附件/附件2.xlsx 的项目根目录')


Q2_ROOT = _project_root(Path(__file__).resolve().parent)
ROOT = q3.ROOT                       # 队友目录（data/、results/ 所在）
sys.path.insert(0, str(Q2_ROOT / 'code'))
sys.path.insert(0, str(Q2_ROOT / 'code/q2_v2'))
from q2_planning import Study        # noqa: E402

VALUE_BANKS = Q2_ROOT / 'results/q2_v32_final/value_banks.json'
_DATES = q3.DATES


def load_banks(path=VALUE_BANKS):
    """{int month: {int weekday: [(a, b) ...]}}，来自冻结的 Q2 v3.2 运行。"""
    raw = json.loads(Path(path).read_text())
    return {int(m): {int(u): [(float(a), float(b)) for a, b in cuts] for u, cuts in wd.items()}
            for m, wd in raw.items()}


def bank_for(date):
    date = pd.Timestamp(date)
    month = 12 if date.year > 2025 else min(max(int(date.month), 2), 12)
    return BANKS[month][int(date.dayofweek)]


BANKS = load_banks()


def value_at(cuts, energy):
    e = np.asarray(energy, float)
    return np.max(np.stack([a * e + b for a, b in cuts]), axis=0)


_STUDY = None


def load_profiles():
    """Q2 冻结 Ridge 负荷剖面 (365, 144) kW：前 90 天为审计过的因果冷启动。"""
    global _STUDY
    if _STUDY is None:
        _STUDY = Study(Q2_ROOT)
    fc = _STUDY.fc['Ridge'].copy()
    fc[:, :90] = _STUDY.legacy[:, :90]          # 与 q2_v32.Experiment 相同的冷启动恢复
    return np.maximum(fc[0], 0.0)


_LF = None


def _profiles():
    global _LF
    if _LF is None:
        _LF = load_profiles()
    return _LF


# ---------------------------------------------------------------- 融合预测库
class FusionBank(q3.Bank):
    """self.load 换成冻结 Ridge 剖面 + 队友的日内偏差修正；其余逻辑继承。"""

    def __init__(self, L, P, F, seed, p):
        self.p = p
        self.L = L
        self.P = P
        self.mean = np.full((365, 4, 144), np.nan)
        self.pv = self.mean.copy()
        self.load = self.mean.copy()
        lf = _profiles()
        for d in range(365):
            if d < 7:                            # Study.legacy 前 7 天为 NaN：回退队友因果预报
                pair = q3.load_forecast(L, d, seed, p.window, p.decay_days, 1, p.slope, p.weekday)
                b0, b1 = np.maximum(pair[0], 0.), np.maximum(pair[1], 0.)
            else:
                b0 = lf[d]
                if d + 1 < 365:
                    b1 = lf[d + 1]
                else:                            # 12/31 的跨午夜段：队友的趋势外推
                    b1 = q3.load_forecast(L, 364, seed, p.window, p.decay_days, 1,
                                          p.slope, p.weekday)[1]
            for k, h in enumerate(q3.ISSUES):
                t = h * 6
                n1 = 144 - t
                anchor = P[d, t - 1] if t else (P[d - 1, -1] if d else 0.)
                hrs = h + q3.HOURS
                if p.interpolate == 'linear':
                    pv = np.interp(hrs, h + np.arange(25), np.r_[anchor, F[d, k]])
                else:
                    off = np.minimum(np.ceil(hrs - h).astype(int) - 1, 23)
                    pv = F[d, k, off]
                pv = np.maximum(pv * p.forecast_scale, 0)
                bias = float(np.median(L[d, max(0, t - p.bias_window):t]
                                       - b0[max(0, t - p.bias_window):t])) if t else 0.
                decay = np.exp(-(hrs - h) / p.bias_decay)
                l = np.empty(144)
                l[:n1] = b0[t:] + p.correction * bias * decay[:n1]
                l[n1:] = b1[:t] + p.correction * bias * decay[n1:]
                self.pv[d, k] = pv
                self.load[d, k] = np.maximum(l, 0)
        self.refresh()


# ---------------------------------------------------------------- 融合 LP
def solve_f(net, price, initial, p, cuts=None, vE=0.0, original=None, tail_from=None):
    """q3.solve 的融合版：终端库存价值用对偶切平面（θ 变量 + 支撑行，仍是 LP）。

    x = [q, c, dis, spill, E, u, v] (+θ)。切平面行：θ ≤ a·E_T + b（对每条支撑）。
    p.terminal='fixed_target' 的对照口径维持原实现（界约束，不用切平面）。
    """
    n = len(net)
    idx = np.arange(n)
    use_cuts = cuts is not None and p.terminal == 'inventory_value'
    nv = 7 * n + (1 if use_cuts else 0)
    theta_i = 7 * n
    rr, cc, vv = [], [], []

    def put(r, c, v):
        rr.extend(np.asarray(r).tolist())
        cc.extend(np.asarray(c).tolist())
        vv.extend(np.broadcast_to(v, len(r)).tolist())

    put(idx, idx, 1); put(idx, n + idx, -1); put(idx, 2 * n + idx, 1); put(idx, 3 * n + idx, -1)
    put(n + idx, 4 * n + idx, 1); put(n + idx, n + idx, -p.eta); put(n + idx, 2 * n + idx, 1 / p.eta)
    put(n + idx[1:], 4 * n + idx[:-1], -1)
    rhs = np.r_[net, initial, np.zeros(n - 1)]

    cost = np.zeros(nv)
    cost[n:3 * n] = p.epsilon
    if use_cuts:
        cost[theta_i] = 1.0                  # θ = 视野末库存的成本函量溢价（斜率<0 → 高库存=信用）
    bounds = [(0, None)] * (4 * n) + [(.1 * p.capacity, .9 * p.capacity)] * n + [(0, 0)] * (2 * n)
    for i in range(n):
        bounds[n + i] = bounds[2 * n + i] = (0, p.power * q3.DT)

    if p.terminal == 'fixed_target':
        tgt = p.target * p.capacity
        bounds[5 * n - 1] = (tgt, tgt)
    elif not use_cuts:
        cost[5 * n - 1] = -vE

    A_ub = None; b_ub = None
    if use_cuts:
        crow, ccol, cval, crhs = [], [], [], []
        for j, (a, b) in enumerate(cuts):
            crow.extend([j, j]); ccol.extend([5 * n - 1, theta_i])
            cval.extend([a, -1.0]); crhs.append(-b)
        from scipy.sparse import coo_matrix as _coo
        A_ub = _coo((cval, (crow, ccol)), shape=(len(cuts), nv)).tocsr()
        b_ub = np.asarray(crhs, float)

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
    if use_cuts:
        bounds.append((None, None))          # θ 自由
    from scipy.sparse import coo_matrix as _coo
    A = _coo((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    kw = dict(A_eq=A, b_eq=rhs, bounds=bounds, method='highs')
    if use_cuts:
        kw.update(A_ub=A_ub, b_ub=b_ub)
    fit = q3.linprog(cost, **kw)
    if not fit.success:
        return None
    assert np.max(np.abs(A @ fit.x - rhs)) < 1e-5
    if use_cuts:
        assert float(np.max(A_ub @ fit.x - b_ub)) < 1e-5
    q, c, dis, spill, E, u, v = fit.x[:7 * n].reshape(7, n)
    return q, np.r_[initial, E], float(fit.fun)


def run_f(L, P, price, bank, p, mask=(6, 12, 18), start=31, end=365, initial=6000., detail=False):
    """q3.run 的融合版：solve→solve_f，expected() 的终端项用切平面价值差。"""
    dcoef = q3.settle_coef(p)
    stepwise = (p.settlements == 'stepwise')
    soc = initial
    daily, intervals, decisions = [], [], []
    for day in range(start, end):
        begin = soc
        cuts = bank_for(_DATES[day] + pd.Timedelta(days=1))
        pred = solve_f(bank.risk(day, 0, p.alpha0), price, soc, p, cuts)
        if pred is None:
            raise RuntimeError('day-ahead infeasible')
        q0, refday, _ = pred
        final = q0.copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        step_up_cost = 0.0; step_dn_cost = 0.0; step_up_kwh = 0.0; step_dn_kwh = 0.0
        states = [soc]
        for k, h in enumerate(q3.ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                if p.horizon == 'day':
                    tail = n1
                    risk_k = bank.risk(day, k, p.alpha_update)[:n1]
                    priceH = price[t:]
                    exec_ref = refday[t:]
                else:
                    tail = n1
                    risk_k = bank.risk(day, k, p.alpha_update)
                    priceH = np.r_[price[t:], price[:t]]
                    exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
                base = final[t:] if stepwise else q0[t:]
                candidate = solve_f(risk_k, priceH, soc, p, cuts, original=base, tail_from=tail)
                if candidate is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = candidate

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(risk_k):
                        _, _, ee, _, ss = q3.execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = price[t:] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = priceH[tail:] @ q[tail:]
                    term = value_at(cuts, np.array([ss, soc]))
                    return adj + future + p.emergency * (priceH @ zz) + float(term[0] - term[1])
                old = np.r_[final[t:], risk_k[tail:]]
                gain = expected(old, exec_ref) - expected(qn, rn)
                accept = (p.threshold <= 0 or gain > p.threshold)
                if accept:
                    if stepwise:
                        d = qn[:tail] - final[t:]
                        step_up_cost += float(price[t:] @ (p.up * np.maximum(d, 0)))
                        step_dn_cost += float(price[t:] @ (dcoef * np.maximum(-d, 0)))
                        step_up_kwh += float(np.maximum(d, 0).sum())
                        step_dn_kwh += float(np.maximum(-d, 0).sum())
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                if detail:
                    decisions.append(dict(date=str(q3.DATES[day].date()), issue=h, estimated_gain=gain, accepted=accept,
                                          history_end=str(q3.DATES[day - 1].date()), observed_end=f'{h:02d}:00',
                                          up_kwh=float(np.maximum(qn[:tail] - base, 0).sum()),
                                          down_kwh=float(np.maximum(base - qn[:tail], 0).sum())))
            for j in range(36):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = q3.execute(final[tt], (L[day, tt] - P[day, tt]) * q3.DT, soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        delta = final - q0
        pc = float(price @ q0)
        if stepwise:
            uc, dc = step_up_cost, step_dn_cost
            up_kwh, down_kwh = step_up_kwh, step_dn_kwh
        else:
            uc = float(price @ (p.up * np.maximum(delta, 0)))
            dc = float(price @ (dcoef * np.maximum(-delta, 0)))
            up_kwh = float(np.maximum(delta, 0).sum())
            down_kwh = float(np.maximum(-delta, 0).sum())
        ec = float(p.emergency * (price @ z))
        cash = pc + uc + dc + ec
        vE = float(price.min() / p.eta)          # 记账口径保持与 q3 相同
        residual = final + z + dis - c - spill - (L[day] - P[day]) * q3.DT
        dyn = np.diff(states) - p.eta * c + dis / p.eta
        assert abs(residual).max() < 1e-5 and abs(dyn).max() < 1e-5
        assert states.min() >= .1 * p.capacity - 1e-5 and states.max() <= .9 * p.capacity + 1e-5
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-5
        assert max(c.max(), dis.max()) <= p.power * q3.DT + 1e-5 and np.minimum(c, dis).max() < 1e-5
        daily.append(dict(date=str(q3.DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc, emergency_cost=ec,
                          total_cost=cash, inventory_adjusted_cost=cash + vE * (begin - soc),
                          plan_kwh=q0.sum(), final_kwh=final.sum(), emergency_kwh=z.sum(),
                          up_kwh=up_kwh, down_kwh=down_kwh, spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc,
                          balance_error=abs(residual).max(), dynamics_error=abs(dyn).max(),
                          soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(q3.DATES[day].date()), t=tt, price=price[tt], load_kw=L[day, tt], pv_kw=P[day, tt],
                                      plan=q0[tt], adjusted=final[tt], charge=c[tt], discharge=dis[tt], emergency=z[tt],
                                      spill=spill[tt], soc_start=states[tt], soc_end=states[tt + 1],
                                      plan_cost=price[tt] * q0[tt], up_cost=price[tt] * p.up * max(delta[tt], 0),
                                      down_net_cost=price[tt] * dcoef * max(-delta[tt], 0),
                                      emergency_cost=price[tt] * p.emergency * z[tt]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions)


# ---------------------------------------------------------------- 融合版主流程
def main_f():
    """q3.main 的融合镜像：同一套校准/验证/消融/敏感性协议，仅预测源与终端价值不同。"""
    out = ROOT / 'results_fusion'
    q3.OUT = out                          # csv()/dump() 全部写入融合目录
    out.exists() or out.mkdir(parents=True)
    clock = time.time()
    _profiles()                           # 预加载 Study（48MB 预测文件）
    L, P, F, price, seed = q3.load()
    p = q3.Params()
    p_min = float(price.min())
    vE = p_min / p.eta

    q3.run = run_f                        # 校准/验证/稳定性复用融合 run
    slope, weekday, _ = q3.select_ridge(L, seed, p)
    p = q3.replace(p, slope=slope, weekday=weekday)
    bank = FusionBank(L, P, F, seed, p)

    warm, _, _ = run_f(L, P, price, bank, p, mask=(), start=0, end=31)
    q3.csv('warmup_january.csv', warm)

    records, best_sw, calib_log, grid0, gridh = q3.calibrate(L, P, price, bank, p, warm.soc_start.iloc[14])
    q3.csv('calibration.csv', records)
    q3.dump('calibration_grid_extension.json', calib_log)

    _, best_rv = q3.rolling_validation(L, P, price, bank, p, warm, grid0, gridh)
    p_sw = q3.replace(p, alpha0=best_sw['alpha0'], alpha_update=best_sw['alpha_update'])
    p_rv = q3.replace(p, alpha0=float(best_rv.alpha0), alpha_update=float(best_rv.alpha_update))
    p = p_rv
    warm_sel, _, _ = run_f(L, P, price, bank, p, mask=(), start=0, end=31)
    q3.csv('warmup_january_selected.csv', warm_sel)
    initial = warm_sel.soc_end.iloc[-1]
    p_theory = q3.replace(p, alpha0=q3.THEORY_ALPHA0, alpha_update=q3.THEORY_ALPHA_UPDATE)
    print('CALIBRATED', q3.asdict(p), 'elapsed', time.time() - clock, flush=True)

    srows = q3.stability_map(L, P, price, bank, p, initial, records)
    beat = float(np.mean([r['beats_none'] for r in srows]))
    costs = np.array([r['full_adjusted_cost'] for r in srows])
    stable_frac = float(np.mean(costs <= costs.min() * 1.002))
    interior = (0 < grid0.index(p.alpha0) < len(grid0) - 1) and (0 < gridh.index(p.alpha_update) < len(gridh) - 1)

    q3.dump('parameters.json', {
        'variant': 'fusion: frozen-Ridge load forecast (Q2 study.fc[Ridge], legacy cold-start <=3/31) '
                   '+ 17-point dual-terminal cuts (Q2 v3.2 value_banks) on the teammate rolling-MPC framework',
        'selected': q3.asdict(p),
        'theory_base': {'alpha0_base': q3.THEORY_ALPHA0, 'alpha_update_base': q3.THEORY_ALPHA_UPDATE},
        'selection': {'primary_alpha0': float(best_rv.alpha0), 'primary_alpha_update': float(best_rv.alpha_update),
                      'single_window_alpha0': best_sw['alpha0'], 'single_window_alpha_update': best_sw['alpha_update'],
                      'grid_extension_log': calib_log, 'alpha0_grid': grid0, 'alpha_update_grid': gridh,
                      'optimum_interior': bool(interior)},
        'ridge_hyperparameters': {'slope': slope, 'weekday': weekday,
                                  'note': '仅影响 12/31 跨午夜外推回退；评价期预测来自冻结 Ridge 文件'},
        'terminal_inventory_value_vE': vE, 'price_min': p_min, 'eta': p.eta,
        'value_banks_source': 'results/q2_v32_final/value_banks.json',
        'initial_feb1': initial, 'parameter_candidates': len(records),
        'stability': {'fraction_full_beats_none': beat, 'fraction_within_0.2pct_of_min': stable_frac,
                      'n_candidates': len(srows)},
        'optimization_horizon': 'rolling 24 hours from each issue; only the next 6 hours are executed',
        'settlement': 'net_refund（主口径，与队友版一致）',
        'qualification': 'chronological historical simulation; every calibrated parameter uses pre-evaluation data only'})

    runs, comparisons = {}, []
    for bits in product_([False, True], repeat=3):
        mask = tuple(h for h, b in zip((6, 12, 18), bits) if b)
        label = 'none' if not mask else '+'.join(map(str, mask))
        df, iv, dec = run_f(L, P, price, bank, p, mask=mask, initial=initial, detail=all(bits))
        runs[label] = df
        comparisons.append(q3.summary(df, strategy=label))
        q3.csv(f'daily_{label}.csv', df)
        if all(bits):
            q3.csv('intervals.csv', iv)
            q3.csv('decisions.csv', dec)
        print('ABLATION', label, df.total_cost.sum(), flush=True)
    q3.csv('strategy_comparison.csv', comparisons)
    full = runs['6+12+18']

    variants = [('main_net_refund', p, True),
                ('legacy_gross_settlement', q3.replace(p, settlements='legacy_gross'), False),
                ('stepwise_settlement', q3.replace(p, settlements='stepwise'), False),
                ('fixed_target_soc', q3.replace(p, terminal='fixed_target'), False),
                ('day_horizon_only', q3.replace(p, horizon='day'), True),
                ('theory_quantiles', p_theory, False),
                ('single_window_params', p_sw, False),
                ('no_intraday_bias_correction', q3.replace(p, correction=0.), True)]
    vrows = []
    for label, pp, rebuild in variants:
        bb = FusionBank(L, P, F, seed, pp) if rebuild else bank
        df, _, _ = run_f(L, P, price, bb, pp, initial=initial)
        q3.csv(f'variant_daily_{label}.csv', df)
        vrows.append(q3.summary(df, variant=label, alpha0=pp.alpha0, alpha_update=pp.alpha_update,
                                settlement=pp.settlements, terminal=pp.terminal))
    q3.csv('variant_comparison.csv', vrows)
    legacy = next(r for r in vrows if r['variant'] == 'legacy_gross_settlement')
    stepwise_row = next(r for r in vrows if r['variant'] == 'stepwise_settlement')

    stale = FusionBank(L, P, F, seed, p)
    stale.stale_pv_issues(range(4))
    stale_df, _, _ = run_f(L, P, price, stale, p, initial=initial)
    q3.csv('daily_stale_pv.csv', stale_df)
    q3.csv('pure_pv_update_value.csv', [dict(stale_pv_cost=stale_df.total_cost.sum(),
                                             fresh_pv_cost=full.total_cost.sum(),
                                             cash_saving=stale_df.total_cost.sum() - full.total_cost.sum(),
                                             inventory_adjusted_saving=stale_df.inventory_adjusted_cost.sum() - full.inventory_adjusted_cost.sum())])

    sens = []
    variants_sens = (
        [('alpha0', x) for x in grid0] +
        [('alpha_update', x) for x in gridh] +
        [('eta', x) for x in (.85, float(np.sqrt(.9)), .95)] +
        [('capacity', x) for x in (9600., 14400.)] +
        [('power', x) for x in (4000., 6000.)] +
        [('window', x) for x in (14, 56)] +
        [('decay_days', x) for x in (7., 28.)] +
        [('bias_window', x) for x in (3, 12)] +
        [('bias_decay', x) for x in (3., 12.)] +
        [('slope', x) for x in (0.2, 3.2)] +
        [('weekday', x) for x in (0.0125, 0.1)] +
        [('reserve', x) for x in (0., .25, .5)] +
        [('up', x) for x in (1.2, 1.8)] +
        [('emergency', x) for x in (3., 7.)] +
        [('forecast_scale', x) for x in (.9, 1.1)] +
        [('correction', x) for x in (0.,)] +
        [('threshold', x) for x in (20., 100.)] +
        [('interpolate', x) for x in ('hold',)] +
        [('epsilon', x) for x in (1e-8, 1e-6)] +
        [('settlements', x) for x in ('legacy_gross', 'stepwise')] +
        [('terminal', x) for x in ('fixed_target',)] +
        [('horizon', x) for x in ('day',)]
    )
    bank_affecting = {'window', 'decay_days', 'bias_window', 'bias_decay', 'forecast_scale', 'correction',
                      'interpolate', 'slope', 'weekday'}
    for name, value in variants_sens:
        if getattr(p, name) == value:
            continue
        pp = q3.replace(p, **{name: value})
        bb = FusionBank(L, P, F, seed, pp) if name in bank_affecting else bank
        ini = initial * pp.capacity / p.capacity
        df, _, _ = run_f(L, P, price, bb, pp, initial=ini)
        label = f'{name}={value}'
        q3.csv(f'sensitivity_daily_{label}.csv', df)
        sens.append(q3.summary(df, parameter=name, value=value, baseline=getattr(p, name)))
        print('SENSITIVITY', label, df.total_cost.sum(), flush=True)
    q3.csv('sensitivity.csv', sens)

    rng = np.random.default_rng(20260911)
    bootstrap = []
    base = runs['none'].inventory_adjusted_cost.to_numpy()
    for label, df in runs.items():
        if label == 'none':
            continue
        delta = base - df.inventory_adjusted_cost.to_numpy()
        n = len(delta)
        for block in (7, 14, 28):
            starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
            ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
            samples = delta[ix].sum(axis=1)
            bootstrap.append(dict(strategy=label, block_days=block, annual_saving=delta.sum(),
                                  ci_low=np.quantile(samples, .025), ci_high=np.quantile(samples, .975),
                                  fraction_positive=float((samples > 0).mean()),
                                  daily_win_rate=float((delta > 0).mean())))
    q3.csv('bootstrap.csv', bootstrap)

    metrics = []
    for k, h in enumerate(q3.ISSUES):
        t = h * 6
        tgt = np.array([bank.actual_horizon(P, d, k) for d in range(31, 365)])
        for lo, hi, label in [(0, 36, 'executed_next_6h'), (0, 144, 'full_24h_horizon')]:
            err = bank.pv[31:, k, lo:hi] - tgt[:, lo:hi]
            metrics.append(dict(issue=h, target=label, samples=int(err.size), mae_kw=float(np.nanmean(abs(err))),
                                rmse_kw=float(np.sqrt(np.nanmean(err ** 2)))))
    q3.csv('forecast_accuracy_pv.csv', metrics)

    day, k, t = 78, 2, 72
    LL, PP, FF = L.copy(), P.copy(), F.copy()
    LL[day, t:] += 12345; LL[day + 1:] += 23456
    PP[day, t:] += 3333; PP[day + 1:] += 4444
    FF[day, k + 1:] += 5555; FF[day + 1:] += 6666
    altered = FusionBank(LL, PP, FF, seed, p)
    causal_diff = float(np.max(abs(bank.risk(day, k, p.alpha_update) - altered.risk(day, k, p.alpha_update))))
    assert causal_diff == 0
    assert legacy['down_kwh'] < 1e-6, '旧解释下减购应被支配为零'
    check = {'causality_future_perturbation_max_diff': causal_diff,
             'balance_max_error': float(full.balance_error.max()),
             'soc_dynamics_max_error': float(full.dynamics_error.max()),
             'soc_min': float(full.soc_min.min()), 'soc_max': float(full.soc_max.max()),
             'cross_day_soc_max_gap': float(abs(full.soc_start.to_numpy()[1:] - full.soc_end.to_numpy()[:-1]).max()),
             'main_down_kwh': float(full.down_kwh.sum()), 'main_up_kwh': float(full.up_kwh.sum()),
             'legacy_down_kwh': float(legacy['down_kwh']),
             'stepwise_cost_minus_net': float(stepwise_row['total_cost'] - full.total_cost.sum()),
             'dates': len(full),
             'alpha0_interior': bool(0 < grid0.index(p.alpha0) < len(grid0) - 1),
             'alpha_update_interior': bool(0 < gridh.index(p.alpha_update) < len(gridh) - 1),
             'fraction_full_beats_none': beat, 'all_assertions_passed': True,
             'elapsed_seconds': time.time() - clock}
    q3.dump('checks.json', check)
    print('FUSION-COMPLETE', check, flush=True)


def product_(*args, **kw):
    from itertools import product
    return product(*args, **kw)
