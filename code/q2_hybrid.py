"""Q2 融合规划实验：日前购电计划 + 日内因果储能执行。

本模块不覆盖正式 result2.xlsx，也不修改原 q2_planning.py。功率输入为 kW，
进入优化与执行前统一乘 DT 转换为每个 10 分钟时槽的 kWh。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from q2_planning import (CAP, DT, ETA, HI, INITIAL, LO, SEED, T, Study,
                         atom_json, bootstrap, cvar, file_hash, weights_for)


ALPHA_COARSE = np.round(np.arange(.50, .951, .05), 2)
RHO_GRID = np.array([0., .25, .5, .75, 1.])
WINDOW_GRID = np.array([14, 21, 28, 42, 56])
SMOOTH_GRID = np.array([0, 3, 5])
K_GRID = np.array([14, 21, 28, 42])
FEATURE_GRID = ("season", "weekday", "level")
DEFAULT = {"alpha": .8, "rho": .5, "window": 28, "smooth": 3}
TOL = 1e-6


def _ridge(x, y, weights, penalties):
    """队友方案中的多输出加权岭回归原式。"""
    y = np.asarray(y, float)
    flat = y.ndim == 1
    if flat:
        y = y[:, None]
    gram = x.T @ (weights[:, None] * x) + np.diag(penalties)
    beta = np.linalg.solve(gram, x.T @ (weights[:, None] * y))
    return beta[:, 0] if flat else beta


def _softmax_weights(errors, decay):
    e = np.asarray(errors, float)
    e = e / max(float(e.mean()), 1e-9)
    w = np.exp(-decay * e)
    return w / w.sum()


def _team_load_weekly(hist):
    if len(hist) < 7:
        return hist.mean(axis=0)
    out = hist[-7].copy()
    if len(hist) >= 14:
        ratio = hist[-7:].mean() / max(hist[-14:-7].mean(), 1e-9)
        out *= np.clip(ratio, .85, 1.15)
    return out


def _team_load_trend(hist, dates, target, window=28):
    m = min(len(dates), window)
    dd = dates[-m:]
    lag = np.array([(d - target).days for d in dd], float)
    weekday = np.array([d.weekday() for d in dd])
    x = np.column_stack([np.ones(m), lag / 7,
                         *[(weekday == k).astype(float) for k in range(1, 7)]])
    target_x = np.array([1., 0., *[float(target.weekday() == k) for k in range(1, 7)]])
    beta = _ridge(x, hist[-m:], np.exp(lag / 14),
                  np.array([1e-8, .20, *([.05] * 6)]))
    return target_x @ beta


def _team_pv_physics(hist, shape_days=14, level_window=7,
                     shape_quantile=.9, threshold=.1, shrink=.5):
    recent = hist[-min(len(hist), shape_days):]
    ref = np.quantile(recent, shape_quantile, axis=0)
    peak = float(ref.max())
    if peak <= 1e-9:
        return hist[-min(len(hist), level_window):].mean(axis=0)
    mask = ref > threshold * peak
    if mask.sum() < 6:
        return hist[-min(len(hist), level_window):].mean(axis=0)
    k = np.array([day[mask].sum() / ref[mask].sum() for day in recent])
    level = min(len(k), max(4, 2 * level_window))
    kk = k[-level:]
    lag = np.arange(level) - (level - 1)
    tau = max(2., level / 3)
    x = np.column_stack([np.ones(level), lag])
    beta = _ridge(x, kk, np.exp(lag / tau), np.array([1e-8, .30]))
    estimate = (1 - shrink) * float(beta[0] + beta[1]) + shrink * float(
        np.average(kk, weights=np.exp(lag / tau)))
    out = np.clip(estimate, .05, 1.25) * ref
    out[~mask] = 0
    out[np.max(recent, axis=0) <= 0] = 0
    return np.minimum(np.maximum(out, 0), recent.max(axis=0) * 1.05)


def build_team_a_forecast(load, pv, dates):
    """忠实移植队友 OptimizedBank 的点预测层；所有切片严格止于 d-1。"""
    n = len(dates)
    load_members = np.full((2, n, T), np.nan)
    pv_members = np.full((3, n, T), np.nan)
    for d in range(1, n):
        hl, hp = load[:d], pv[:d]
        load_members[0, d] = _team_load_weekly(hl)
        load_members[1, d] = _team_load_trend(hl, dates[:d], dates[d], 28)
        pv_members[0, d] = hp[-min(5, d):].mean(axis=0)
        m = min(d, 28)
        lag = np.arange(m)[::-1]
        w = np.exp(-lag / 3.)
        pv_members[1, d] = (hp[-m:] * w[:, None]).sum(axis=0) / w.sum()
        pv_members[2, d] = _team_pv_physics(hp)
    wl = np.full((n, 2), .5)
    wp = np.full((n, 3), 1 / 3)
    for d in range(3, n):
        lo = max(1, d - 10)
        if d - lo < 2:
            continue
        el = np.array([np.mean(np.abs(load[lo:d] - load_members[k, lo:d])) for k in range(2)])
        ep = np.array([np.mean(np.abs(pv[lo:d] - pv_members[k, lo:d])) for k in range(3)])
        wl[d], wp[d] = _softmax_weights(el, 8.), _softmax_weights(ep, 8.)
    load_hat = np.einsum("knt,nk->nt", load_members, wl)
    pv_hat = np.einsum("knt,nk->nt", pv_members, wp)
    load_hat = np.maximum(gaussian_filter1d(load_hat, 1., axis=1, mode="nearest"), 0)
    pv_hat = np.maximum(gaussian_filter1d(pv_hat, 1., axis=1, mode="nearest"), 0)
    for d in range(1, n):
        pv_hat[d, np.max(pv[max(0, d - 14):d], axis=0) <= 0] = 0
    raw_net = (load_hat - pv_hat) * DT
    actual_net = (load - pv) * DT
    residual = actual_net - raw_net
    phi = np.zeros(n)
    for d in range(2, n):
        lo = max(1, d - 21)
        if d - lo < 3:
            continue
        prev, curr = residual[lo - 1:d - 1].ravel(), residual[lo:d].ravel()
        den = float(prev @ prev)
        if den > 1e-12:
            phi[d] = np.clip(float(prev @ curr / den), 0, .6)
    net = raw_net.copy()
    net[1:] += phi[1:, None] * np.nan_to_num(residual[:-1])
    # 将 AR 修正归入负荷端，仅为保持统一 load/pv/net 明细结构。
    load_hat += (net - raw_net) / DT
    return np.stack([np.maximum(load_hat, 0), np.maximum(pv_hat, 0)]), {
        "load_weights": wl, "pv_weights": wp, "phi": phi}


def smooth_margin(margin, width):
    if width == 0:
        return np.asarray(margin, float)
    kernel = np.ones(int(width), float) / int(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    return np.convolve(np.pad(margin, (pad_left, pad_right), mode="edge"), kernel, mode="valid")


def execute_causal(q, actual_net, initial_soc, reference_soc, rho):
    """逐槽执行器。函数签名中没有未来实际值，因而天然满足非前视约束。"""
    q, actual_net = np.asarray(q, float), np.asarray(actual_net, float)
    c = np.zeros(T); v = np.zeros(T); z = np.zeros(T); spill = np.zeros(T)
    soc = np.empty(T + 1); soc[0] = float(initial_soc)
    for t in range(T):
        surplus = q[t] - actual_net[t]
        if surplus >= 0:
            c[t] = min(surplus, CAP, max((HI - soc[t]) / ETA, 0))
            spill[t] = surplus - c[t]
        else:
            reserve = LO + rho * (reference_soc[t + 1] - LO)
            v[t] = min(-surplus, CAP, ETA * max(soc[t] - reserve, 0))
            z[t] = -surplus - v[t]
        soc[t + 1] = soc[t] + ETA * c[t] - v[t] / ETA
    return {"c": c, "v": v, "z": z, "spill": spill, "soc": soc}


def execution_checks(q, net, run):
    c, v, z, spill, soc = (run[k] for k in ["c", "v", "z", "spill", "soc"])
    balance = q + z + v - c - spill - net
    dynamics = np.diff(soc) - ETA * c + v / ETA
    checks = {
        "balance_residual": float(np.max(np.abs(balance))),
        "soc_recursion_residual": float(np.max(np.abs(dynamics))),
        "soc_min": float(soc.min()), "soc_max": float(soc.max()),
        "charge_max": float(c.max()), "discharge_max": float(v.max()),
        "negative_min": float(min(x.min() for x in [q, c, v, z, spill])),
        "simultaneous_max": float(np.minimum(c, v).max()),
    }
    checks["pass"] = bool(
        max(checks["balance_residual"], checks["soc_recursion_residual"]) <= TOL
        and checks["soc_min"] >= LO - TOL and checks["soc_max"] <= HI + TOL
        and max(checks["charge_max"], checks["discharge_max"]) <= CAP + TOL
        and checks["negative_min"] >= -TOL and checks["simultaneous_max"] <= TOL)
    return checks


@dataclass(frozen=True)
class PolicyConfig:
    alpha: float = .8
    rho: float = .5
    window: int = 28
    smooth: int = 3
    features: str = "level"
    k: int = 21


class HybridStudy:
    def __init__(self, root):
        self.root = Path(root)
        self.base = Study(self.root)
        self.price = self.base.price
        self.dates = self.base.dates
        self.load, self.pv, self.net = self.base.load, self.base.pv, self.base.net
        self.net_energy = self.net * DT
        # 修复原规划 Study 为兼容表临时写入真实值的冷启动：融合实验始终使用因果预测。
        ridge = self.base.fc["Ridge"].copy(); ridge[:, :90] = self.base.legacy[:, :90]
        team, self.team_diagnostics = build_team_a_forecast(self.load, self.pv, self.dates)
        self.fc = {"ManualWeekly": self.base.legacy.copy(), "Ridge": ridge, "TeamA": team}
        self.out = self.root / "results/q2_hybrid_v1"
        self.fig = self.root / "figures/q2_hybrid_v1"
        self.out.mkdir(parents=True, exist_ok=True); self.fig.mkdir(parents=True, exist_ok=True)
        source = file_hash(self.root / "code/q2_hybrid.py")
        self.fingerprint = hashlib.sha256((source + self.base.fingerprint).encode()).hexdigest()
        self.cache = self.out / "checkpoints" / self.fingerprint[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        self._plan_cache = {}
        self.progress("initialized")

    def progress(self, phase, **kwargs):
        payload = {"phase": phase, "utc": pd.Timestamp.now(tz="UTC").isoformat(),
                   "fingerprint": self.fingerprint, **kwargs}
        atom_json(self.out / "progress.json", payload)
        print(phase, kwargs, flush=True)

    def forecast_net(self, method, d):
        return (self.fc[method][0, d] - self.fc[method][1, d]) * DT

    def residual_pool(self, method, d, window):
        start = max(7, d - int(window))
        fc_hist = (self.fc[method][0, start:d] - self.fc[method][1, start:d]) * DT
        pool = self.net_energy[start:d] - fc_hist
        if len(pool) == 0 or not np.isfinite(pool).all():
            raise ValueError(f"残差池非法: method={method}, d={d}, W={window}")
        return pool, np.arange(start, d)

    def conditional_pool(self, method, d, features, k):
        hist = np.arange(max(7, d - 120), d)
        dayofyear = self.dates.dayofyear.to_numpy()
        weekday = self.dates.dayofweek.to_numpy()
        fc = self.fc[method]
        x = np.column_stack([
            np.sin(2 * np.pi * dayofyear / 365), np.cos(2 * np.pi * dayofyear / 365),
            np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
            fc[0].sum(axis=1) * DT, fc[1].sum(axis=1) * DT])
        dims = {"season": 2, "weekday": 4, "level": 6}[features]
        scale = x[hist, :dims].std(axis=0); scale[scale < 1e-9] = 1
        dist = np.sum(((x[hist, :dims] - x[d, :dims]) / scale) ** 2, axis=1)
        chosen = np.sort(hist[np.argsort(dist, kind="stable")[:min(int(k), len(hist))]])
        fc_hist = (fc[0, chosen] - fc[1, chosen]) * DT
        return self.net_energy[chosen] - fc_hist, chosen

    def risk_net(self, method, d, cfg, conditional=False):
        if conditional:
            pool, hist = self.conditional_pool(method, d, cfg.features, cfg.k)
        else:
            pool, hist = self.residual_pool(method, d, cfg.window)
        margin = smooth_margin(np.quantile(pool, cfg.alpha, axis=0), cfg.smooth)
        return self.forecast_net(method, d) + margin, hist

    def plan(self, method, d, family, cfg, initial, fixed_initial=False):
        plan_initial = INITIAL if fixed_initial else float(initial)
        cache_key = (method, int(d), family, cfg, round(plan_initial, 6))
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]
        if family == "point":
            scen, w, hist = self.forecast_net(method, d)[None, :], np.ones(1), np.array([], int)
        elif family == "quantile":
            risk, hist = self.risk_net(method, d, cfg)
            scen, w = risk[None, :], np.ones(1)
        elif family == "conditional":
            risk, hist = self.risk_net(method, d, cfg, conditional=True)
            scen, w = risk[None, :], np.ones(1)
        elif family == "saa":
            pool, hist = self.residual_pool(method, d, 21)
            scen = self.forecast_net(method, d)[None, :] + pool
            w = weights_for(len(pool), 14)
        else:
            raise KeyError(family)
        sol = self.base.solve(scen / DT, w, initial=plan_initial, terminal=INITIAL)
        result = {**sol, "history": hist, "plan_initial": plan_initial}
        self._plan_cache[cache_key] = result
        return result

    def _day_row(self, method, policy, d, sol, initial, fixed=False, rho=.5, keep_slots=False):
        if fixed:
            realized = self.base.realized(sol, self.net[d])
            run = {"c": sol["c"], "v": sol["v"],
                   "z": np.maximum(self.net_energy[d] + sol["c"] - sol["v"] - sol["q"], 0),
                   "spill": np.maximum(sol["q"] + sol["v"] - self.net_energy[d] - sol["c"], 0),
                   "soc": sol["soc"]}
            checks = execution_checks(sol["q"], self.net_energy[d], run)
            terminal = float(sol["soc"][-1])
        else:
            run = execute_causal(sol["q"], self.net_energy[d], initial, sol["soc"], rho)
            checks = execution_checks(sol["q"], self.net_energy[d], run)
            if not checks["pass"]:
                raise AssertionError(checks)
            z, spill = run["z"], run["spill"]
            realized = {
                "planned_cost": float(self.price @ sol["q"]),
                "emergency_cost": float(5 * self.price @ z),
                "total_cost": float(self.price @ sol["q"] + 5 * self.price @ z),
                "planned_kwh": float(sol["q"].sum()), "emergency_kwh": float(z.sum()),
                "spill_kwh": float(spill.sum()), "charge_kwh": float(run["c"].sum()),
                "discharge_kwh": float(run["v"].sum()),
                "emergency_slots": int((z > TOL).sum()),
                "emergency_day": int((z > TOL).any()),
                "events": int(((z > TOL) & ~np.r_[False, z[:-1] > TOL]).sum()),
                "peak_price_emergency_kwh": float(z[self.price >= np.quantile(self.price, .75)].sum()),
            }
            terminal = float(run["soc"][-1])
        row = {"method": method, "policy": policy, "d": int(d),
               "date": str(self.dates[d].date()), "initial_soc": float(initial),
               "terminal_soc": terminal, "rho": float(rho),
               "max_history_day": int(sol["history"].max()) if len(sol["history"]) else d - 1,
               "solve_seconds": float(sol["seconds"]), **realized, **checks}
        slots = None
        if keep_slots:
            slots = pd.DataFrame({"method": method, "policy": policy, "date": row["date"],
                "slot": np.arange(T), "price": self.price, "net_actual_kwh": self.net_energy[d],
                "q": sol["q"], "c": run["c"], "v": run["v"], "z": run["z"],
                "spill": run["spill"], "soc_start": run["soc"][:-1],
                "soc_end": run["soc"][1:], "reference_soc_end": sol["soc"][1:]})
        return row, slots

    def simulate(self, method, policy, start, stop, config_for_day=None,
                 constant=None, keep_slots=False, initial=INITIAL):
        rows, slots = [], []
        soc = float(initial)
        for d in range(int(start), int(stop)):
            cfg = constant if constant is not None else config_for_day(d)
            if cfg is None:
                cfg = PolicyConfig(**DEFAULT)
            if policy == "H0_fixed_SAA":
                sol = self.plan(method, d, "saa", cfg, INITIAL, fixed_initial=True)
                row, detail = self._day_row(method, policy, d, sol, INITIAL, fixed=True)
                soc = INITIAL
            elif policy == "H1_fixedQ_causal":
                sol = self.plan(method, d, "saa", cfg, INITIAL, fixed_initial=True)
                row, detail = self._day_row(method, policy, d, sol, soc, rho=cfg.rho,
                                            keep_slots=keep_slots)
                soc = row["terminal_soc"]
            else:
                family = {"H1_replan_SAA": "saa", "H2_point": "point",
                          "H3_quantile": "quantile", "H4_conditional": "conditional"}[policy]
                sol = self.plan(method, d, family, cfg, soc)
                row, detail = self._day_row(method, policy, d, sol, soc, rho=cfg.rho,
                                            keep_slots=keep_slots)
                soc = row["terminal_soc"]
            row.update({"alpha": cfg.alpha, "window": cfg.window, "smooth": cfg.smooth,
                        "features": cfg.features, "k": cfg.k})
            rows.append(row)
            if detail is not None:
                slots.append(detail)
        return pd.DataFrame(rows), (pd.concat(slots, ignore_index=True) if slots else None)


def inventory_adjusted_cost(frame, price_min):
    return float(frame.total_cost.sum() - price_min / ETA *
                 (frame.iloc[-1].terminal_soc - frame.iloc[0].initial_soc))


def choose_near(scores, keys):
    best = scores.score.min()
    near = scores[scores.score <= best * 1.001].copy()
    near["tie"] = list(zip(*[near[k] for k in keys]))
    return near.sort_values(["tie", "score"]).iloc[0]


def _score_configs(s, method, policy, origin, configs, stage):
    records = []
    for j, cfg in enumerate(configs):
        frame, _ = s.simulate(method, policy, origin - 56, origin, constant=cfg)
        records.append({"method": method, "month": s.dates[origin].month,
                        "origin": str(s.dates[origin].date()), "stage": stage,
                        **cfg.__dict__, "score": inventory_adjusted_cost(frame, s.price.min()),
                        "raw_cost": float(frame.total_cost.sum()),
                        "terminal_soc": float(frame.iloc[-1].terminal_soc)})
        if (j + 1) % 20 == 0:
            s.progress("selection_progress", method=method, month=s.dates[origin].month,
                       stage=stage, done=j + 1, total=len(configs))
    return pd.DataFrame(records)


def select_monthly(s, method, policy, months=range(6, 11)):
    """每月仅用月初之前 56 个完整日；H3 采用预先写定的分阶段网格。"""
    selected, all_scores = [], []
    for month in months:
        marker = s.cache / f"selection_{method}_{policy}_{month}.json"
        score_path = s.cache / f"selection_{method}_{policy}_{month}.csv"
        if marker.exists() and score_path.exists():
            selected.append(json.loads(marker.read_text())); all_scores.append(pd.read_csv(score_path)); continue
        origin = int(np.flatnonzero(s.dates == pd.Timestamp(2025, month, 1))[0])
        if policy == "H2_point":
            configs = [PolicyConfig(rho=float(r)) for r in RHO_GRID]
            scores = _score_configs(s, method, policy, origin, configs, "rho")
            winner = choose_near(scores, ["rho"])
        elif policy == "H3_quantile":
            coarse = [PolicyConfig(alpha=float(a), rho=.5, window=int(w), smooth=3)
                      for a in ALPHA_COARSE for w in WINDOW_GRID]
            a = _score_configs(s, method, policy, origin, coarse, "alpha_window")
            w1 = choose_near(a.assign(alpha_distance=np.abs(a.alpha - .8)),
                             ["alpha_distance", "window"])
            interaction = []
            fine = np.unique(np.round(np.clip(np.arange(w1.alpha - .04, w1.alpha + .041, .01), .5, .95), 2))
            for alpha in fine:
                for rho in RHO_GRID:
                    for smooth in SMOOTH_GRID:
                        interaction.append(PolicyConfig(float(alpha), float(rho), int(w1.window), int(smooth)))
            b = _score_configs(s, method, policy, origin, interaction, "local_interaction")
            b = b.assign(alpha_distance=np.abs(b.alpha - .8), rho_distance=np.abs(b.rho - .5))
            winner = choose_near(b, ["alpha_distance", "rho_distance", "smooth"])
            scores = pd.concat([a, b], ignore_index=True)
        elif policy == "H4_conditional":
            base = DEFAULT
            configs = [PolicyConfig(base["alpha"], float(r), 28, int(sm), f, int(k))
                       for f in FEATURE_GRID for k in K_GRID for r in RHO_GRID for sm in SMOOTH_GRID]
            scores = _score_configs(s, method, policy, origin, configs, "conditional")
            scores = scores.assign(rho_distance=np.abs(scores.rho - .5),
                                   feature_rank=scores.features.map({"season": 0, "weekday": 1, "level": 2}))
            winner = choose_near(scores, ["feature_rank", "k", "rho_distance", "smooth"])
        else:
            raise KeyError(policy)
        cfg = {k: (int(winner[k]) if k in ["window", "smooth", "k"] else
                   float(winner[k]) if k in ["alpha", "rho"] else winner[k])
               for k in PolicyConfig.__dataclass_fields__}
        record = {"method": method, "policy": policy, "month": month,
                  "origin": str(s.dates[origin].date()), "history_start": str(s.dates[origin - 56].date()),
                  "history_end": str(s.dates[origin - 1].date()), "score": float(winner.score), **cfg}
        atom_json(marker, record); scores.to_csv(score_path, index=False)
        selected.append(record); all_scores.append(scores)
        s.progress("selection_checkpoint", method=method, policy=policy, month=month, config=cfg)
    return pd.DataFrame(selected), pd.concat(all_scores, ignore_index=True)


def config_lookup(selection, default=DEFAULT):
    table = selection.set_index("month")
    def get(d):
        month = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d))
        if month.month < 6:
            return PolicyConfig(**default)
        m = min(month.month, 10)
        return PolicyConfig(**{k: table.loc[m, k] for k in PolicyConfig.__dataclass_fields__})
    return get


def run_pilot(s):
    rows = []
    for method in s.fc:
        for policy in ["H0_fixed_SAA", "H1_fixedQ_causal", "H1_replan_SAA", "H2_point", "H3_quantile"]:
            frame, _ = s.simulate(method, policy, 151, 154, constant=PolicyConfig())
            rows.append({"method": method, "policy": policy, "cost": frame.total_cost.sum(),
                         "seconds": frame.solve_seconds.sum(), "checks_pass": bool(frame["pass"].all())})
    # 因果前缀测试：改变未来观测不应改变本槽 execute_step 的输出。
    q = np.full(T, 500.); ref = np.linspace(INITIAL, INITIAL, T + 1)
    actual = s.net_energy[151].copy(); changed = actual.copy(); changed[72:] += 1e6
    a = execute_causal(q, actual, INITIAL, ref, .5)
    b = execute_causal(q, changed, INITIAL, ref, .5)
    prefix_gap = max(np.max(np.abs(a[k][:72] - b[k][:72])) for k in ["c", "v", "z", "spill"])
    tests = {"all_checks_pass": all(r["checks_pass"] for r in rows),
             "future_perturbation_prefix_gap": float(prefix_gap),
             "information_boundary_pass": bool(prefix_gap < 1e-10)}
    if not tests["all_checks_pass"] or not tests["information_boundary_pass"]:
        raise AssertionError(tests)
    pd.DataFrame(rows).to_csv(s.out / "pilot.csv", index=False)
    atom_json(s.out / "pilot_tests.json", tests)
    s.progress("pilot_passed", tests=tests,
               estimated_seconds_per_policy_day=float(pd.DataFrame(rows).seconds.sum() / 45))
    return pd.DataFrame(rows), tests


def summarize_daily(s, daily):
    daily = daily.copy(); daily["date"] = pd.to_datetime(daily.date)
    periods = {"full334": ("2025-02-01", "2025-12-31"),
               "development153": ("2025-06-01", "2025-10-31"),
               "later61": ("2025-11-01", "2025-12-31")}
    rows = []
    for (method, policy), g in daily.groupby(["method", "policy"]):
        for period, (start, end) in periods.items():
            q = g[g.date.between(start, end)].sort_values("date")
            if len(q) != len(pd.date_range(start, end)):
                continue
            row = {"method": method, "policy": policy, "period": period, "days": len(q)}
            for col in ["planned_cost", "emergency_cost", "total_cost", "planned_kwh",
                        "emergency_kwh", "spill_kwh", "charge_kwh", "discharge_kwh",
                        "emergency_slots", "events", "solve_seconds"]:
                row[col] = float(q[col].sum())
            row.update({"start_soc": float(q.iloc[0].initial_soc),
                        "end_soc": float(q.iloc[-1].terminal_soc),
                        "inventory_adjusted_cost": inventory_adjusted_cost(q, s.price.min()),
                        "cvar90": cvar(q.total_cost, beta=.9), "cvar95": cvar(q.total_cost, beta=.95),
                        "max_daily_cost": float(q.total_cost.max()),
                        "emergency_day_frequency": float(q.emergency_day.mean()),
                        "emergency_slot_frequency": float(q.emergency_slots.sum() / (T * len(q)))})
            rows.append(row)
    return pd.DataFrame(rows)


def matched_oracles(s, summary):
    rows = []
    for _, r in summary[summary.period == "full334"].iterrows():
        terminal = float(r.end_soc)
        sol = s.base.solve(s.net[31:].reshape(1, -1), initial=INITIAL, terminal=terminal)
        oracle = s.base.realized(sol, s.net[31:].ravel(), take=334 * T)
        rows.append({"method": r.method, "policy": r.policy, "terminal_soc": terminal,
                     "actual_cost": r.total_cost, "oracle_cost": oracle["total_cost"],
                     "oracle_gap": r.total_cost - oracle["total_cost"],
                     "oracle_regret_rate": (r.total_cost - oracle["total_cost"]) / oracle["total_cost"],
                     "oracle_dual_gap_relative": sol["checks"]["dual_gap_relative"]})
    return pd.DataFrame(rows)


def bootstrap_against_h3(daily):
    rows = []
    daily = daily.copy(); daily["date"] = pd.to_datetime(daily.date)
    for method, g in daily.groupby("method"):
        base = g[g.policy == "H3_quantile"][["date", "total_cost"]].rename(columns={"total_cost": "base"})
        if base.empty:
            continue
        for policy, q in g.groupby("policy"):
            if policy == "H3_quantile":
                continue
            pair = q.merge(base, on="date")
            for period, start, end in [("development153", "2025-06-01", "2025-10-31"),
                                       ("later61", "2025-11-01", "2025-12-31")]:
                z = pair[pair.date.between(start, end)]
                if len(z) >= 7:
                    mean, lo, hi = bootstrap(z.base - z.total_cost, 7, 2000)
                    rows.append({"method": method, "policy": policy, "period": period,
                                 "saving_vs_H3_daily_mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def figures(s, summary, monthly, selection, daily, slots):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 10})
    manifest = []
    def emit(fig, name, meaning):
        for ext in ["pdf", "png"]:
            fig.savefig(s.fig / f"{name}.{ext}", bbox_inches="tight", dpi=180)
        plt.close(fig); manifest.append({"figure": name, "meaning": meaning})
    q = summary[summary.period == "development153"].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = q.pivot(index="method", columns="policy", values="total_cost") / 1e6
    pivot.plot.bar(ax=ax); ax.set_ylabel("Cost (million yuan)"); ax.set_xlabel("Forecast method")
    emit(fig, "hybrid_cost_comparison", "同一开发期比较固定执行、因果执行、点预测和分位数策略的总费用。")
    q = selection[selection.policy == "H3_quantile"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, g in q.groupby("method"):
        ax.plot(g.month, g.alpha, marker="o", label=method)
    ax.axhline(.8, color="black", linestyle="--", label="economic anchor 0.8")
    ax.set(xlabel="Decision month", ylabel="Selected alpha", ylim=(.48, .97)); ax.legend()
    emit(fig, "selected_alpha_path", "逐月因果选择的分位数与0.8经济学锚点；变化反映误差分布和储能耦合。")
    fig, ax = plt.subplots(figsize=(11, 5))
    m = monthly.pivot_table(index="month", columns="policy", values="total_cost", aggfunc="sum") / 1e6
    m.plot(ax=ax, marker="o"); ax.set_ylabel("Monthly cost (million yuan)")
    emit(fig, "monthly_stability", "各策略月度总费用，检查结论是否由少数月份驱动。")
    if slots is not None and len(slots):
        example = slots[(slots.method == slots.method.iloc[0]) & (slots.policy == "H3_quantile")].iloc[:T]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(example.slot, example.soc_start, label="actual SOC")
        ax.plot(example.slot, example.reference_soc_end, label="reference SOC", linestyle="--")
        ax.set(xlabel="10-minute slot", ylabel="SOC (kWh)"); ax.legend()
        emit(fig, "soc_causal_example", "实际SOC只由已发生误差更新，参考SOC仅用于形成保留线。")
    atom_json(s.fig / "figure_manifest.json", manifest)
    return pd.DataFrame(manifest)


def run_full(root):
    started = time.perf_counter(); s = HybridStudy(root)
    pilot, tests = run_pilot(s)
    selections, score_tables = [], []
    for method in s.fc:
        for policy in ["H2_point", "H3_quantile"]:
            selected, scores = select_monthly(s, method, policy)
            selections.append(selected); score_tables.append(scores)
    selection = pd.concat(selections, ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    selection.to_csv(s.out / "selection.csv", index=False)
    scores.to_csv(s.out / "selection_scores.csv.gz", index=False, compression="gzip")
    daily_parts, slot_parts = [], []
    for method in s.fc:
        for policy in ["H0_fixed_SAA", "H1_fixedQ_causal", "H1_replan_SAA"]:
            frame, detail = s.simulate(method, policy, 31, 365, constant=PolicyConfig(), keep_slots=(policy == "H1_fixedQ_causal"))
            daily_parts.append(frame)
            if detail is not None: slot_parts.append(detail)
            s.progress("policy_checkpoint", method=method, policy=policy)
        for policy in ["H2_point", "H3_quantile"]:
            lookup = config_lookup(selection[(selection.method == method) & (selection.policy == policy)])
            frame, detail = s.simulate(method, policy, 31, 365, config_for_day=lookup, keep_slots=(policy == "H3_quantile"))
            daily_parts.append(frame)
            if detail is not None: slot_parts.append(detail)
            s.progress("policy_checkpoint", method=method, policy=policy)
    daily = pd.concat(daily_parts, ignore_index=True)
    # 只对开发期 H3 成本最低的预测模型运行条件策略，避免无价值的同质堆砌。
    provisional = summarize_daily(s, daily)
    best_method = provisional[(provisional.policy == "H3_quantile") &
                              (provisional.period == "development153")].sort_values("inventory_adjusted_cost").iloc[0].method
    cond_sel, cond_scores = select_monthly(s, best_method, "H4_conditional")
    selection = pd.concat([selection, cond_sel], ignore_index=True)
    scores = pd.concat([scores, cond_scores], ignore_index=True)
    lookup = config_lookup(cond_sel)
    frame, detail = s.simulate(best_method, "H4_conditional", 31, 365, config_for_day=lookup, keep_slots=True)
    daily = pd.concat([daily, frame], ignore_index=True); slot_parts.append(detail)
    slots = pd.concat(slot_parts, ignore_index=True)
    daily.to_csv(s.out / "daily.csv.gz", index=False, compression="gzip")
    slots.to_csv(s.out / "slots.csv.gz", index=False, compression="gzip")
    selection.to_csv(s.out / "selection.csv", index=False)
    scores.to_csv(s.out / "selection_scores.csv.gz", index=False, compression="gzip")
    summary = summarize_daily(s, daily); summary.to_csv(s.out / "summary.csv", index=False)
    monthly_source = daily.copy(); monthly_source["month"] = pd.to_datetime(monthly_source.date).dt.month
    monthly = monthly_source.groupby(["method", "policy", "month"], as_index=False)[
        ["planned_cost", "emergency_cost", "total_cost", "emergency_kwh", "spill_kwh"]].sum()
    monthly.to_csv(s.out / "monthly.csv", index=False)
    oracle = matched_oracles(s, summary); oracle.to_csv(s.out / "matched_oracles.csv", index=False)
    boot = bootstrap_against_h3(daily); boot.to_csv(s.out / "bootstrap.csv", index=False)
    checks = daily.groupby(["method", "policy"])[["balance_residual", "soc_recursion_residual",
        "simultaneous_max", "negative_min", "soc_min", "soc_max", "charge_max", "discharge_max"]].agg(
        {"balance_residual": "max", "soc_recursion_residual": "max", "simultaneous_max": "max",
         "negative_min": "min", "soc_min": "min", "soc_max": "max", "charge_max": "max", "discharge_max": "max"})
    checks.to_csv(s.out / "checks.csv")
    # 两个必须保留的回归证据：我方旧SAA精确复现；队友A移植结果只在实际跑完后判定。
    manual_h0 = summary[(summary.method == "ManualWeekly") & (summary.policy == "H0_fixed_SAA") &
                        (summary.period == "full334")].iloc[0].total_cost
    if abs(manual_h0 - 14539240.537931435) >= 1e-3:
        raise AssertionError((manual_h0, 14539240.537931435))
    team_fixed, _ = s.simulate("TeamA", "H3_quantile", 31, 365, constant=PolicyConfig(.8, .5, 28, 3))
    team_cost = float(team_fixed.total_cost.sum())
    regressions = {"legacy_saa_actual": manual_h0, "legacy_saa_reference": 14539240.537931435,
                   "team_a_port_actual": team_cost, "team_a_reference": 13820986.576531284,
                   "team_a_error": team_cost - 13820986.576531284}
    atom_json(s.out / "regressions.json", regressions)
    figs = figures(s, summary, monthly, selection, daily, slots)
    manifest = {"complete": True, "fingerprint": s.fingerprint, "best_h3_method": best_method,
                "rows": len(daily), "slots": len(slots), "runtime_seconds": time.perf_counter() - started,
                "checks_all_pass": bool(daily["pass"].all()), "pilot": tests,
                "official_result2_overwritten": False, "regressions": regressions,
                "parameter_protocol": "monthly causal 56-day calibration; Nov-Dec use Oct rule",
                "figure_count": len(figs)}
    atom_json(s.out / "manifest.json", manifest)
    s.progress("completed", runtime_seconds=manifest["runtime_seconds"], best_h3_method=best_method)
    return s, summary, monthly, selection, oracle, boot, figs, manifest


def render_results(s, summary, monthly, selection, oracle, boot, figs, manifest):
    """Notebook 最后一格调用；所有结论都从已运行结果生成，不预写优胜者。"""
    from IPython.display import Markdown, display
    display(Markdown("### 实验运行清单")); display(pd.DataFrame([manifest]))
    display(Markdown("### H0—H5费用与风险指标")); display(summary)
    display(Markdown("### 逐月因果选参")); display(selection)
    display(Markdown("### 匹配实际终态的Oracle下界")); display(oracle)
    display(Markdown("### 7日移动块Bootstrap")); display(boot)
    display(Markdown("### 图表含义登记")); display(figs)
    dev = summary[summary.period == "development153"].sort_values("inventory_adjusted_cost")
    later = summary[summary.period == "later61"].set_index(["method", "policy"])
    winner = dev.iloc[0]
    text = (f"开发期库存修正成本最低的是 **{winner.method} / {winner.policy}**，"
            f"费用为 {winner.inventory_adjusted_cost:,.2f} 元。其11—12月库存修正成本为 "
            f"{later.loc[(winner.method, winner.policy), 'inventory_adjusted_cost']:,.2f} 元。"
            "是否替换仍需结合Bootstrap区间、月度稳定性和结构复杂度判断；预测误差最低不自动等于成本最低。")
    display(Markdown(text))


if __name__ == "__main__":
    root = Path(os.environ.get("CUMCM_ROOT", Path(__file__).resolve().parents[1]))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run_full(root)
