"""Q2 V3.1: causal terminal-condition ablation with genuine LP-dual cuts.

The frozen Ridge forecasts are reused.  Every policy uses the same residual
scenarios and the same H1 causal executor; only the terminal treatment changes.
All slot-flow variables are AC-bus kWh, SOC is kWh, and price is yuan/kWh.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib, json, os, platform, shutil, sys, time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

def project_root(start):
    """按只在仓库根存在的文件向上查找项目根。

    使本模块不依赖自身在 code/ 下的深度，归档移动后仍可运行。
    """
    for candidate in (start, *start.parents):
        if (candidate / 'C题/附件/附件2.xlsx').exists():
            return candidate
    raise FileNotFoundError('未找到含 C题/附件/附件2.xlsx 的项目根目录')


HERE = Path(__file__).resolve().parent
ROOT = project_root(HERE)
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/q2_v2'))

from q2_planning import Study, weights_for
from dispatch import execute_day, check_dispatch
from utils import BATTERY, DT, T, INTERVALS

OUT = ROOT / 'results/q2_v31_terminal_ablation'
FIG = ROOT / 'figures/q2_v31_terminal_ablation'
REPORT = ROOT / 'reports/Q2_V31_TERMINAL_ABLATION_RESULTS.md'
SEED = 20260912
LO, HI, INITIAL = 1200.0, 10800.0, 6000.0
CAP = 5000.0 * DT
EPS = 1e-8
BLOCK, BOOT_DRAWS = 7, 2000
VALUE_HISTORY_DAYS = 84       # 12 complete weeks
VALUE_HISTORY_HALF = 28       # four-week recency half-life
VALUE_TOL = 1e-6              # yuan; numerical convergence only
VALUE_MAX_ITER = 50           # numerical safeguard, not a fitted parameter
GRID_SIZES = (9, 17)
POLICIES = ('hard_6000', 'cyclic_initial', 'free_terminal',
            'dual_terminal_g9', 'dual_terminal_g17')


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


def block_ci(x, seed=SEED, ci=.95):
    x = np.asarray(x, float); n = len(x); b = min(BLOCK, n)
    rng = np.random.default_rng(seed); blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(BOOT_DRAWS, blocks))
    ids = ((starts[:, :, None] + np.arange(b)) % n).reshape(BOOT_DRAWS, -1)[:, :n]
    samples = x[ids].mean(1); alpha = (1 - ci) / 2
    return float(x.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1-alpha))


def value_at(cuts, energy):
    e = np.asarray(energy, float)
    if not cuts:
        return np.zeros_like(e, dtype=float)
    result = np.max(np.stack([a * e + b for a, b in cuts]), axis=0)
    return float(result) if result.ndim == 0 else result


def dual_supporting_cuts(grid, objectives, duals, reference=INITIAL):
    """Build exact supporting hyperplanes Q(e_j)+pi_j(e-e_j).

    For a minimization LP parameterized by the RHS initial energy, the equality
    marginal pi_j is a subgradient of the convex optimal-value function.  Hence
    each affine function is a global lower support on the physical SOC domain.
    """
    grid = np.asarray(grid, float); objectives = np.asarray(objectives, float)
    duals = np.asarray(duals, float)
    raw = [(float(pi), float(q - pi * e)) for e, q, pi in zip(grid, objectives, duals)]
    support = np.column_stack([a * grid + b for a, b in raw]).max(axis=1)
    violation = float(np.max(support - objectives))
    own_error = float(np.max(np.abs(np.array([a*e+b for e,(a,b) in zip(grid, raw)]) - objectives)))
    if violation > 1e-5 or own_error > 1e-5:
        raise AssertionError({'dual_support_violation': violation, 'own_cut_error': own_error})
    shift = float(value_at(raw, reference))
    cuts = [(a, b-shift) for a, b in raw]
    return cuts, violation, own_error


@dataclass(frozen=True)
class ScenarioConfig:
    window: int
    half_life: int
    reserve_ratio: float = .5


EARLY = ScenarioConfig(21, 14, .5)
DEPLOYED = ScenarioConfig(14, 7, .5)


class Experiment:
    def __init__(self):
        self.study = Study(ROOT)
        self.price = self.study.price
        self.dates = self.study.dates
        self.net = self.study.net
        self.fc = self.study.fc['Ridge'].copy()
        # Restore the audited cold-start forecast; never substitute actual day values.
        self.fc[:, :90] = self.study.legacy[:, :90]
        self.netfc = self.fc[0] - self.fc[1]
        self.solve_calls = 0
        self.solve_seconds = 0.0

    def config(self, day):
        # V3 rolling validation accepted 14-day/7-day on June 1 and froze it later.
        return EARLY if self.dates[day] < pd.Timestamp('2025-06-01') else DEPLOYED

    def scenarios(self, day):
        cfg = self.config(day)
        ids = np.arange(max(7, day-cfg.window), day, dtype=int)
        weights = weights_for(len(ids), cfg.half_life)
        residual = self.net[ids] - self.netfc[ids]
        scenarios = self.netfc[day][None, :] + residual  # kW
        if ids.max() >= day or not np.isfinite(scenarios).all():
            raise AssertionError('non-causal or non-finite scenario')
        return scenarios, weights, ids, cfg

    def weekday_distribution(self, origin, weekday):
        ids = np.arange(max(7, origin-VALUE_HISTORY_DAYS), origin-1, dtype=int)
        ids = ids[self.dates[ids].dayofweek == weekday]
        if len(ids) < 3:
            raise AssertionError('insufficient causal weekday history')
        weights = weights_for(len(ids), VALUE_HISTORY_HALF)
        return self.net[ids], weights, ids

    def solve(self, scenarios_kw, weights, initial_soc, cuts=None, terminal=None):
        """Solve the risk-neutral finite-scenario LP in kWh units."""
        scenarios_kw = np.atleast_2d(np.asarray(scenarios_kw, float))
        S, H = scenarios_kw.shape
        if H % T or not np.isfinite(scenarios_kw).all():
            raise ValueError('invalid scenarios')
        weights = np.asarray(weights, float); weights /= weights.sum()
        prices = np.tile(self.price, H // T)
        use_value = cuts is not None
        n0 = 4*H + S*H; n = n0 + int(use_value); theta = n0 if use_value else None

        t = np.arange(H)
        eq_row = np.r_[t, t, t, t[1:]]
        eq_col = np.r_[H+t, 2*H+t, 3*H+t, 3*H+t[:-1]]
        eq_val = np.r_[np.full(H, -BATTERY.eta_c), np.full(H, 1/BATTERY.eta_d),
                       np.ones(H), -np.ones(H-1)]
        Aeq = coo_matrix((eq_val, (eq_row, eq_col)), shape=(H, n)).tocsr()
        beq = np.zeros(H); beq[0] = float(initial_soc)

        k = np.arange(S*H); slots = np.tile(t, S)
        # q + discharge - charge + emergency >= net_power * DT
        Aub = coo_matrix((np.tile([-1., 1., -1., -1.], (S*H, 1)).ravel(),
            (np.repeat(k, 4), np.column_stack([slots, H+slots, 2*H+slots, 4*H+k]).ravel())),
            shape=(S*H, n)).tocsr()
        bub = -(scenarios_kw * DT).ravel()
        if use_value:
            rows=[]; cols=[]; vals=[]; rhs=[]
            for j, (a, b) in enumerate(cuts):
                rows.extend([j, j]); cols.extend([4*H-1, theta]); vals.extend([a, -1.]); rhs.append(-b)
            C = coo_matrix((vals, (rows, cols)), shape=(len(cuts), n)).tocsr()
            Aub = vstack([Aub, C], format='csr'); bub = np.r_[bub, rhs]

        objective = np.r_[prices, np.full(2*H, EPS), np.zeros(H),
                          (5*weights[:, None]*prices).ravel(), [1.] if use_value else []]
        energy_bounds = [(LO, HI)] * H
        if terminal is not None:
            energy_bounds[-1] = (float(terminal), float(terminal))
        bounds = ([(0, None)]*H + [(0, CAP)]*(2*H) + energy_bounds
                  + [(0, None)]*(S*H) + ([(None, None)] if use_value else []))
        started = time.perf_counter()
        ans = linprog(objective, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                      bounds=bounds, method='highs', options={'presolve': True,
                      'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
        if not ans.success:
            ans = linprog(objective, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                          bounds=bounds, method='highs-ds', options={'presolve': False})
        elapsed = time.perf_counter() - started
        self.solve_calls += 1; self.solve_seconds += elapsed
        if not ans.success:
            raise RuntimeError(f'LP failed {ans.status}: {ans.message}')

        x = ans.x; q = x[:H].copy(); charge = x[H:2*H].copy(); discharge = x[2*H:3*H].copy()
        soc = np.r_[initial_soc, x[3*H:4*H]]
        simultaneous = np.minimum(charge, discharge).clip(0)
        if simultaneous.max(initial=0) > 1e-7:
            remove = np.minimum(charge, discharge/BATTERY.eta_c**2).clip(0)
            charge -= remove; discharge -= BATTERY.eta_c**2*remove
        emergency = np.maximum(scenarios_kw*DT + charge-discharge-q, 0)
        physical = float(prices@q + weights@(emergency@(5*prices)))
        terminal_value = float(value_at(cuts, soc[-1])) if use_value else 0.0
        eq_residual = float(np.abs(Aeq@x-beq).max())
        ineq_violation = float(max(0, np.max(Aub@x-bub)))
        if eq_residual >= 1e-5 or ineq_violation >= 1e-5:
            raise AssertionError('LP residual failure')
        return dict(q=q, charge=charge, discharge=discharge, soc=soc,
            expected_cost=physical, terminal_value=terminal_value,
            objective=physical+terminal_value, solver_objective=float(ans.fun),
            initial_dual=float(ans.eqlin.marginals[0]),
            solve_seconds=elapsed, eq_residual=eq_residual,
            ineq_violation=ineq_violation, simultaneous_kwh=float(simultaneous.sum()))


def learn_value_bank(exp, origin, grid_size):
    grid = np.linspace(LO, HI, grid_size)
    banks = {u: [(0.0, 0.0)] for u in range(7)}
    old_values = {u: np.zeros(grid_size) for u in range(7)}
    history=[]; dual_rows=[]; check_rows=[]
    for iteration in range(1, VALUE_MAX_ITER+1):
        updated={}; new_values={}
        for weekday in range(6, -1, -1):
            next_weekday = (weekday+1) % 7
            next_cuts = updated.get(next_weekday, banks[next_weekday])
            scenarios, weights, ids = exp.weekday_distribution(origin, weekday)
            objectives=[]; duals=[]
            for energy in grid:
                solution = exp.solve(scenarios, weights, float(energy), cuts=next_cuts)
                # The dual marginal and function value must refer to exactly the
                # same perturbed solver objective; physical cost remains the
                # reporting metric outside value-function construction.
                objectives.append(solution['solver_objective']); duals.append(solution['initial_dual'])
            cuts, violation, own_error = dual_supporting_cuts(grid, objectives, duals)
            updated[weekday] = cuts; new_values[weekday] = value_at(cuts, grid)
            check_rows.append(dict(origin=str(exp.dates[origin].date()), grid_size=grid_size,
                iteration=iteration, weekday=weekday, support_violation=violation,
                own_cut_error=own_error, dual_max=max(duals), dual_min=min(duals),
                history_min_day=int(ids.min()), history_max_day=int(ids.max())))
            dual_rows.extend(dict(origin=str(exp.dates[origin].date()), grid_size=grid_size,
                iteration=iteration, weekday=weekday, initial_soc=float(e), objective=float(q),
                initial_soc_dual=float(pi), cut_slope=float(pi), cut_intercept=float(q-pi*e))
                for e,q,pi in zip(grid,objectives,duals))
        gap = max(float(np.max(np.abs(new_values[u]-old_values[u]))) for u in range(7))
        history.append(dict(origin=str(exp.dates[origin].date()), grid_size=grid_size,
                            iteration=iteration, max_value_change=gap, converged=gap<=VALUE_TOL))
        banks = updated; old_values = new_values
        if gap <= VALUE_TOL:
            break
    if history[-1]['converged'] is not True:
        raise AssertionError('relative value iteration did not converge')
    return banks, pd.DataFrame(history), pd.DataFrame(dual_rows), pd.DataFrame(check_rows)


def build_value_banks(exp):
    all_banks={}; histories=[]; duals=[]; checks=[]; curves=[]
    dense = np.linspace(LO, HI, 65)
    for grid_size in GRID_SIZES:
        all_banks[grid_size] = {}
        for month in range(2, 13):
            origin = int(np.flatnonzero(exp.dates == pd.Timestamp(2025, month, 1))[0])
            bank,h,d,c = learn_value_bank(exp, origin, grid_size)
            all_banks[grid_size][month] = bank
            histories.append(h); duals.append(d); checks.append(c)
            for weekday,cuts in bank.items():
                curves.extend(dict(grid_size=grid_size, month=month, weekday=weekday,
                                   soc=float(e), relative_future_cost=float(value_at(cuts,e))) for e in dense)
            print('value-bank', grid_size, month, 'iterations', len(h), 'gap', h.iloc[-1].max_value_change, flush=True)
    atom(OUT/'value_banks.json', {g:{m:{u:c for u,c in b.items()} for m,b in months.items()}
                                  for g,months in all_banks.items()})
    pd.concat(histories, ignore_index=True).to_csv(OUT/'value_iteration.csv', index=False)
    pd.concat(duals, ignore_index=True).to_csv(OUT/'terminal_duals.csv', index=False)
    pd.concat(checks, ignore_index=True).to_csv(OUT/'dual_cut_checks.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'terminal_value_curves.csv', index=False)
    return all_banks


def bank_for_date(exp, banks, grid_size, date):
    date = pd.Timestamp(date)
    month = 12 if date.year > 2025 else int(date.month)
    return banks[grid_size][month][int(date.dayofweek)]


def actual_row(exp, day, policy, state, solution, ids, cfg):
    executed = execute_day(solution['q'], exp.net[day]*DT, state, solution['soc'], cfg.reserve_ratio)
    audit = check_dispatch(exp.net[day]*DT, solution['q'], executed['charge'], executed['discharge'],
                           executed['spill'], executed['soc'], executed['emergency'])
    if not audit['pass']:
        raise AssertionError(audit)
    planned = float(exp.price@solution['q']); emergency = float(5*exp.price@executed['emergency'])
    z = executed['emergency']; active = z > 1e-7
    row = dict(policy=policy, day=day, date=str(exp.dates[day].date()),
        initial_soc=float(state), terminal_soc=float(executed['soc'][-1]),
        planned_cost=planned, emergency_cost=emergency, total_cost=planned+emergency,
        planned_kwh=float(solution['q'].sum()), emergency_kwh=float(z.sum()),
        spill_kwh=float(executed['spill'].sum()), charge_kwh=float(executed['charge'].sum()),
        discharge_kwh=float(executed['discharge'].sum()), emergency_slots=int(active.sum()),
        emergency_day=int(active.any()), events=int(np.sum(active & ~np.r_[False,active[:-1]])),
        reference_terminal_soc=float(solution['soc'][-1]), terminal_value=float(solution['terminal_value']),
        lp_objective=float(solution['objective']), solve_seconds=float(solution['solve_seconds']),
        lp_eq_residual=solution['eq_residual'], lp_ineq_violation=solution['ineq_violation'],
        balance_residual=audit['balance_max_abs'], soc_residual=audit['soc_equation_max_abs'],
        history_min_day=int(ids.min()), history_max_day=int(ids.max()), **asdict(cfg))
    slots = pd.DataFrame(dict(policy=policy, date=row['date'], slot=np.arange(T), interval=INTERVALS,
        price=exp.price, net_actual=exp.net[day]*DT, purchase=solution['q'],
        charge=executed['charge'], discharge=executed['discharge'], emergency=z,
        spill=executed['spill'], soc_start=executed['soc'][:-1], soc_end=executed['soc'][1:],
        reference_soc=solution['soc'][1:]))
    return row, slots, float(executed['soc'][-1])


def run_policies(exp, banks):
    rows=[]; slots=[]
    for policy in POLICIES:
        state = INITIAL
        for day in range(31, 365):
            scenarios, weights, ids, cfg = exp.scenarios(day)
            terminal=None; cuts=None
            if policy == 'hard_6000': terminal = INITIAL
            elif policy == 'cyclic_initial': terminal = state
            elif policy == 'dual_terminal_g9':
                cuts = bank_for_date(exp, banks, 9, exp.dates[day]+pd.Timedelta(days=1))
            elif policy == 'dual_terminal_g17':
                cuts = bank_for_date(exp, banks, 17, exp.dates[day]+pd.Timedelta(days=1))
            solution = exp.solve(scenarios, weights, state, cuts=cuts, terminal=terminal)
            row,slot,state = actual_row(exp,day,policy,state,solution,ids,cfg)
            rows.append(row); slots.append(slot)
        print('policy', policy, 'completed', flush=True)
    daily = pd.DataFrame(rows); daily.to_csv(OUT/'daily.csv', index=False)
    pd.concat(slots, ignore_index=True).to_csv(OUT/'slots.csv.gz', index=False, compression='gzip')
    return daily


PERIODS = {'full334':('2025-02-01','2025-12-31'),
           'development153':('2025-06-01','2025-10-31'),
           'frozen61':('2025-11-01','2025-12-31')}


def inventory_boundary(exp, banks17, first_date, first_soc, last_date, last_soc):
    start_cuts = bank_for_date(exp, {17:banks17}, 17, first_date)
    next_date = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    end_cuts = bank_for_date(exp, {17:banks17}, 17, next_date)
    start_value = float(value_at(start_cuts, first_soc)); end_value = float(value_at(end_cuts, last_soc))
    return start_value, end_value, end_value-start_value


def summarize(exp, daily, banks):
    data=daily.copy(); data['date']=pd.to_datetime(data.date)
    summaries=[]
    for policy,g in data.groupby('policy'):
        g=g.sort_values('date')
        for period,(a,b) in PERIODS.items():
            q=g[g.date.between(a,b)]
            start_value,end_value,adjustment = inventory_boundary(exp,banks[17],a,q.iloc[0].initial_soc,
                                                                   b,q.iloc[-1].terminal_soc)
            summaries.append(dict(policy=policy,period=period,days=len(q),
                planned_cost=q.planned_cost.sum(),emergency_cost=q.emergency_cost.sum(),
                raw_total_cost=q.total_cost.sum(),start_inventory_value=start_value,
                end_inventory_value=end_value,inventory_adjustment=adjustment,
                adjusted_total_cost=q.total_cost.sum()+adjustment,planned_kwh=q.planned_kwh.sum(),
                emergency_kwh=q.emergency_kwh.sum(),spill_kwh=q.spill_kwh.sum(),
                emergency_slots=q.emergency_slots.sum(),emergency_day_frequency=q.emergency_day.mean(),
                max_daily_cost=q.total_cost.max(),start_soc=q.iloc[0].initial_soc,
                end_soc=q.iloc[-1].terminal_soc))
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'summary.csv',index=False)

    reference='dual_terminal_g17'; boots=[]
    main=data[data.policy==reference][['date','total_cost']]
    summary_idx=summary.set_index(['policy','period'])
    for policy in [p for p in POLICIES if p!=reference]:
        pair=main.merge(data[data.policy==policy][['date','total_cost']],on='date',suffixes=('_dual17','_other'))
        for period,(a,b) in PERIODS.items():
            q=pair[pair.date.between(a,b)]
            mean,lo,hi=block_ci(q.total_cost_other-q.total_cost_dual17,SEED+len(q)+POLICIES.index(policy))
            other_adj=summary_idx.loc[(policy,period),'inventory_adjustment']
            dual_adj=summary_idx.loc[(reference,period),'inventory_adjustment']
            shift=float((other_adj-dual_adj)/len(q))
            boots.append(dict(reference=policy,challenger=reference,period=period,days=len(q),
                raw_daily_saving_mean=mean,raw_ci_low95=lo,raw_ci_high95=hi,
                boundary_shift_per_day=shift,adjusted_daily_saving_mean=mean+shift,
                adjusted_ci_low95=lo+shift,adjusted_ci_high95=hi+shift))
    pd.DataFrame(boots).to_csv(OUT/'bootstrap.csv',index=False)
    data['month']=data.date.dt.month
    data.groupby(['policy','month'])[['planned_cost','emergency_cost','total_cost','emergency_kwh','spill_kwh']].sum().reset_index().to_csv(OUT/'monthly.csv',index=False)
    return summary


def grid_convergence(exp,banks,daily,summary):
    dense=np.linspace(LO,HI,257); rows=[]
    for month in range(2,13):
        for weekday in range(7):
            v9=value_at(banks[9][month][weekday],dense);v17=value_at(banks[17][month][weekday],dense)
            rows.append(dict(month=month,weekday=weekday,max_abs_value_difference=float(np.max(np.abs(v9-v17))),
                mean_abs_value_difference=float(np.mean(np.abs(v9-v17))),
                value_scale=float(max(1,np.max(np.abs(v17)))),relative_max_difference=float(np.max(np.abs(v9-v17))/max(1,np.max(np.abs(v17))))))
    frame=pd.DataFrame(rows)
    s=summary.set_index(['policy','period'])
    policy_rows=[]
    for period in PERIODS:
        a=s.loc[('dual_terminal_g9',period)];b=s.loc[('dual_terminal_g17',period)]
        policy_rows.append(dict(period=period,g9_adjusted_cost=a.adjusted_total_cost,
            g17_adjusted_cost=b.adjusted_total_cost,cost_difference_g9_minus_g17=a.adjusted_total_cost-b.adjusted_total_cost,
            relative_cost_difference=abs(a.adjusted_total_cost-b.adjusted_total_cost)/b.adjusted_total_cost,
            end_soc_difference=a.end_soc-b.end_soc))
    frame.to_csv(OUT/'grid_curve_convergence.csv',index=False)
    pd.DataFrame(policy_rows).to_csv(OUT/'grid_policy_convergence.csv',index=False)


def verify(exp,daily):
    dual_checks=pd.read_csv(OUT/'dual_cut_checks.csv');vi=pd.read_csv(OUT/'value_iteration.csv')
    tests=[]
    def add(name,value,tolerance):
        passed=bool(value<=tolerance);tests.append(dict(test=name,value=float(value),tolerance=float(tolerance),passed=passed))
        if not passed:raise AssertionError((name,value,tolerance))
    add('causality',max(0,float((daily.history_max_day-daily.day+1).max())),0)
    add('balance',float(daily.balance_residual.max()),1e-7)
    add('soc_recursion',float(daily.soc_residual.max()),1e-7)
    add('lp_eq',float(daily.lp_eq_residual.max()),1e-5)
    add('lp_ineq',float(daily.lp_ineq_violation.max()),1e-5)
    add('dual_support',max(0,float(dual_checks.support_violation.max())),1e-5)
    add('dual_own_cut',float(dual_checks.own_cut_error.max()),1e-5)
    add('value_iteration_convergence',float((~vi.groupby(['origin','grid_size']).tail(1).converged).sum()),0)
    for policy,g in daily.groupby('policy'):
        add('soc_continuity_'+policy,float(np.max(np.abs(g.initial_soc.iloc[1:].to_numpy()-g.terminal_soc.iloc[:-1].to_numpy()))),1e-7)
    result={'all_pass':all(x['passed'] for x in tests),'tests':tests,
            'solve_calls':exp.solve_calls,'solve_seconds':exp.solve_seconds}
    atom(OUT/'verification.json',result);return result


def configure_chinese_font():
    import matplotlib.font_manager as fm
    candidates=[]
    for root in ['/usr/share/fonts','/usr/local/share/fonts',str(Path.home()/'.fonts')]:
        p=Path(root)
        if p.exists():candidates.extend(p.rglob('*NotoSansCJK*Regular*.ttc'))
    if candidates:
        fm.fontManager.addfont(str(candidates[0]))
        return fm.FontProperties(fname=str(candidates[0])).get_name(),str(candidates[0])
    for name in ['Arial Unicode MS','PingFang SC','SimHei']:
        try:
            fm.findfont(name,fallback_to_default=False);return name,name
        except Exception:pass
    return 'DejaVu Sans','fallback'


def figures_and_report(summary):
    import matplotlib.pyplot as plt
    font_name,font_path=configure_chinese_font();plt.rcParams['font.sans-serif']=[font_name]
    plt.rcParams['axes.unicode_minus']=False
    FIG.mkdir(parents=True,exist_ok=True)
    labels={'hard_6000':'终端6000','cyclic_initial':'日内能量闭合','free_terminal':'无终端价值',
            'dual_terminal_g9':'对偶价值-9点','dual_terminal_g17':'对偶价值-17点'}
    full=summary[summary.period=='full334'].set_index('policy').loc[list(POLICIES)]
    best=full.adjusted_total_cost.min();delta=(full.adjusted_total_cost-best)/1000
    fig,ax=plt.subplots(figsize=(8.4,4.8));ax.bar([labels[x] for x in delta.index],delta,color='#4472C4')
    ax.set_ylabel('相对最低方案的调整后费用/千元');ax.tick_params(axis='x',rotation=12)
    fig.tight_layout();fig.savefig(FIG/'01_adjusted_cost_delta.pdf');fig.savefig(FIG/'01_adjusted_cost_delta.png',dpi=200);plt.close(fig)

    monthly=pd.read_csv(OUT/'monthly.csv');p=monthly.pivot(index='month',columns='policy',values='total_cost')
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for policy in POLICIES:
        ax.plot(p.index,(p[policy]-p['hard_6000'])/1000,marker='o',ms=3,label=labels[policy])
    ax.axhline(0,color='black',lw=.8);ax.set_xlabel('月份');ax.set_ylabel('相对终端6000的月费用/千元');ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(FIG/'02_monthly_cost_delta.pdf');fig.savefig(FIG/'02_monthly_cost_delta.png',dpi=200);plt.close(fig)

    curves=pd.read_csv(OUT/'terminal_value_curves.csv');g=curves[(curves.month==10)&(curves.weekday==0)]
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for n,z in g.groupby('grid_size'):ax.plot(z.soc,z.relative_future_cost,label=f'{int(n)}点网格',lw=2)
    ax.axvline(INITIAL,color='black',ls='--',lw=.8);ax.set_xlabel('荷电状态/kWh');ax.set_ylabel('相对未来运行成本/元');ax.legend()
    fig.tight_layout();fig.savefig(FIG/'03_grid_value_comparison.pdf');fig.savefig(FIG/'03_grid_value_comparison.png',dpi=200);plt.close(fig)

    daily=pd.read_csv(OUT/'daily.csv');box=[daily[daily.policy==x].terminal_soc for x in POLICIES]
    fig,ax=plt.subplots(figsize=(8.4,4.8));ax.boxplot(box,tick_labels=[labels[x] for x in POLICIES],showfliers=False)
    ax.axhline(INITIAL,color='black',ls='--',lw=.8);ax.set_ylabel('每日实际期末SOC/kWh');ax.tick_params(axis='x',rotation=12)
    fig.tight_layout();fig.savefig(FIG/'04_soc_distribution.pdf');fig.savefig(FIG/'04_soc_distribution.png',dpi=200);plt.close(fig)
    atom(FIG/'figure_manifest.json',{'font_name':font_name,'font_path':font_path,'files':[x.name for x in sorted(FIG.iterdir())]})

    boot=pd.read_csv(OUT/'bootstrap.csv');grid=pd.read_csv(OUT/'grid_policy_convergence.csv')
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('# Q2 V3.1 终端条件消融结果\n\n本报告由本次 Colab 输出自动生成。\n\n'
        +'## 全年与分期费用（双边库存价值修正）\n\n'+summary.to_markdown(index=False)
        +'\n\n## 配对7日块Bootstrap\n\n'+boot.to_markdown(index=False)
        +'\n\n## 9点—17点网格策略收敛\n\n'+grid.to_markdown(index=False)
        +'\n\n## 图表\n\n'+'\n'.join(f'- `{x.name}`' for x in sorted(FIG.glob('*.pdf')))+'\n',encoding='utf-8')


def full():
    if OUT.exists():shutil.rmtree(OUT)
    if FIG.exists():shutil.rmtree(FIG)
    OUT.mkdir(parents=True);FIG.mkdir(parents=True)
    started=time.time();exp=Experiment();banks=build_value_banks(exp);daily=run_policies(exp,banks)
    summary=summarize(exp,daily,banks);grid_convergence(exp,banks,daily,summary)
    verification=verify(exp,daily);figures_and_report(summary)
    manifest=dict(complete=True,started_at=pd.Timestamp.fromtimestamp(started,tz='UTC').isoformat(),
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.time()-started,
        solve_calls=exp.solve_calls,solve_seconds=exp.solve_seconds,python=platform.python_version(),
        cpu_count=os.cpu_count(),notebook='code/problem2_v31_terminal_ablation.ipynb',
        source_sha256=sha(__file__),prediction_sha256=sha(ROOT/'results/problem2_forecast_ablation_predictions.csv.gz'),
        grid_sizes=list(GRID_SIZES),all_checks_pass=verification['all_pass'],fresh_run=True)
    atom(OUT/'run_manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    full()
