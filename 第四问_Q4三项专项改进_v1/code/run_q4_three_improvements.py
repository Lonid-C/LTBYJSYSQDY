"""Focused Q4 run: CVaR candidate, wider causality audit, generalization tables.

The original six Q4-3 parameters are loaded from the frozen second-version
JSON and are never optimized here.  Long loops expose tqdm progress and write
monthly checkpoints for resumption.
"""
from __future__ import annotations

import os

# Process-level parallelism is used outside HiGHS.  Prevent nested oversubscription.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import math
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
PROJECT_ROOT = EXP_ROOT.parent
BASE_ROOT = PROJECT_ROOT / "第四问_完整交付_主体_第二版"
BASE_CODE = BASE_ROOT / "code"
BASE_RESULTS = BASE_ROOT / "results4"
RESULTS = EXP_ROOT / "results"
CHECKPOINTS = EXP_ROOT / "checkpoints"
LOGS = EXP_ROOT / "logs"

for _p in (RESULTS, CHECKPOINTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE_CODE))
sys.path.insert(0, str(HERE))


FROZEN_KEYS = (
    "temper", "shrink_net", "shrink_price", "kappa", "threshold", "adj_premium"
)
EXPECTED_FROZEN = np.array([
    0.4732077236080887,
    0.55,
    0.7620649739917784,
    1.0583818144215367,
    237.0896789891042,
    1.2342410281042373,
])
MONTH_RANGES = (
    (31, 59, "02"), (59, 90, "03"), (90, 120, "04"),
    (120, 151, "05"), (151, 181, "06"), (181, 212, "07"),
    (212, 243, "08"), (243, 273, "09"), (273, 304, "10"),
    (304, 334, "11"), (334, 365, "12"),
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", float_format="%.10f")
    temp.replace(path)


def _atomic_json(value, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    temp.replace(path)


def _load_modules():
    import run_q4_experiment as experiment
    import q4_policy as policy
    import q4_risk_stoch as risk_solver
    return experiment, policy, risk_solver


def _frozen_policy(experiment, policy):
    selected = json.loads(
        (BASE_RESULTS / "q4_selected_policy.json").read_text(encoding="utf-8")
    )["Q4-3"]
    values = np.asarray(selected["x"], float)
    if values.shape != (6,) or not np.allclose(values, EXPECTED_FROZEN, atol=1e-12, rtol=0):
        raise RuntimeError(f"frozen policy mismatch: {values}")
    result = experiment.policy_from(values)
    actual = np.array([getattr(result, key) for key in FROZEN_KEYS])
    if not np.allclose(actual, EXPECTED_FROZEN, atol=1e-12, rtol=0):
        raise RuntimeError("policy construction changed a frozen parameter")
    return result


def _candidate_name(risk_weight: float, seed: int) -> str:
    return f"rho{risk_weight:.2f}_seed{seed}"


def run_candidate(task):
    risk_weight, seed, position = task
    experiment, policy, risk_solver = _load_modules()
    context = experiment.prepare()
    frozen = _frozen_policy(experiment, policy)
    risk_solver.configure(risk_weight=risk_weight, beta=0.95)
    policy.Q4S.solve_q4 = risk_solver.solve_q4

    name = _candidate_name(risk_weight, seed)
    final_path = RESULTS / f"daily_{name}.csv"
    if final_path.exists():
        old = pd.read_csv(final_path, encoding="utf-8-sig")
        if len(old) == 334:
            summary = policy.summarize(old, candidate=name, risk_weight=risk_weight, seed=seed)
            summary["seconds"] = 0.0
            summary["resumed"] = True
            return summary

    start_time = time.perf_counter()
    chunks = []
    soc = float(context["init_q4_3"])
    progress = tqdm(
        MONTH_RANGES,
        desc=f"{name} monthly replay",
        unit="month",
        position=position,
        leave=True,
        dynamic_ncols=True,
    )
    for start, end, month in progress:
        checkpoint = CHECKPOINTS / f"daily_{name}_{month}.csv"
        if checkpoint.exists():
            frame = pd.read_csv(checkpoint, encoding="utf-8-sig")
            expected = end - start
            if len(frame) != expected:
                raise RuntimeError(f"bad checkpoint length: {checkpoint}")
        else:
            frame, _, _, _ = policy.run_q4(
                context["L"], context["P"], context["V"], context["bank"],
                context["rb"], context["pbank"], context["p"], frozen,
                context["shape"], context["edges"], context["vE_ref"],
                mask=(6, 12, 18), start=start, end=end, initial=soc,
                mode="stoch", detail=False, seed=seed,
            )
            _atomic_csv(frame, checkpoint)
        chunks.append(frame)
        soc = float(frame.soc_end.iloc[-1])
        progress.set_postfix(soc=f"{soc:.0f}", days=sum(len(x) for x in chunks))

    daily = pd.concat(chunks, ignore_index=True)
    if len(daily) != 334:
        raise RuntimeError(f"expected 334 days, got {len(daily)}")
    _atomic_csv(daily, final_path)
    summary = policy.summarize(
        daily, candidate=name, risk_weight=risk_weight, seed=seed
    )
    summary["seconds"] = time.perf_counter() - start_time
    summary["resumed"] = False
    return summary


def exact_bootstrap(reference, variant, block_len, resamples, seed):
    from q4_statistics import empirical_cvar, moving_block_indices

    reference = np.asarray(reference, float)
    variant = np.asarray(variant, float)
    rng = np.random.default_rng(seed)
    cvar_diff = np.empty(resamples)
    mean_diff = np.empty(resamples)
    for i in tqdm(
        range(resamples), desc=f"bootstrap L={block_len}", unit="sample",
        leave=False, dynamic_ncols=True,
    ):
        index = moving_block_indices(len(reference), block_len, rng)
        cvar_diff[i] = empirical_cvar(variant[index], 0.95) - empirical_cvar(reference[index], 0.95)
        mean_diff[i] = np.mean(variant[index] - reference[index])
    ref_cvar = empirical_cvar(reference, 0.95)
    var_cvar = empirical_cvar(variant, 0.95)
    ci_lo, ci_hi = np.quantile(cvar_diff, [0.025, 0.975])
    mean_lo, mean_hi = np.quantile(mean_diff, [0.025, 0.975])
    return {
        "block_len": block_len,
        "cvar95_reference": ref_cvar,
        "cvar95_variant": var_cvar,
        "diff": var_cvar - ref_cvar,
        "diff_percent": 100.0 * (var_cvar - ref_cvar) / ref_cvar,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_worse": np.mean(cvar_diff > 0),
        "mean_diff": np.mean(variant - reference),
        "mean_ci_lo": mean_lo,
        "mean_ci_hi": mean_hi,
        "resamples": resamples,
        "seed": seed,
    }


def build_diagnostics(summaries, resamples):
    from q4_statistics import empirical_cvar

    b0 = pd.read_csv(BASE_RESULTS / "q4_daily_Q4-3_B0.csv", encoding="utf-8-sig")
    h3 = pd.read_csv(BASE_RESULTS / "q4_daily_Q4-3_H3.csv", encoding="utf-8-sig")
    tail = b0[[
        "date", "total_cost", "inventory_adjusted_cost", "emergency_cost",
        "emergency_kwh", "soc_start", "soc_end", "soc_min", "mean_price",
    ]].merge(
        h3[[
            "date", "total_cost", "inventory_adjusted_cost", "emergency_cost",
            "emergency_kwh", "soc_start", "soc_end", "soc_min",
        ]], on="date", suffixes=("_B0", "_H3"),
    )
    tail["adjusted_diff_H3_minus_B0"] = (
        tail.inventory_adjusted_cost_H3 - tail.inventory_adjusted_cost_B0
    )
    b0_cut = np.sort(b0.total_cost.to_numpy())[-17]
    h3_cut = np.sort(h3.total_cost.to_numpy())[-17]
    tail["B0_top17"] = tail.total_cost_B0 >= b0_cut
    tail["H3_top17"] = tail.total_cost_H3 >= h3_cut
    _atomic_csv(tail, RESULTS / "tail_day_diagnostics.csv")

    summary_frame = pd.DataFrame(summaries).sort_values(["risk_weight", "seed"])
    _atomic_csv(summary_frame, RESULTS / "cvar_candidate_summary.csv")

    bootstrap_rows = []
    for risk_weight in (0.15, 0.25):
        candidate = pd.read_csv(
            RESULTS / f"daily_{_candidate_name(risk_weight, 0)}.csv",
            encoding="utf-8-sig",
        )
        for block_len in tqdm((7, 14, 28), desc=f"rho={risk_weight:.2f} intervals", unit="block"):
            row = exact_bootstrap(
                b0.total_cost.to_numpy(), candidate.total_cost.to_numpy(),
                block_len, resamples, 20260941,
            )
            row["candidate"] = f"rho={risk_weight:.2f}"
            bootstrap_rows.append(row)
    _atomic_csv(pd.DataFrame(bootstrap_rows), RESULTS / "cvar_bootstrap_7_14_28.csv")

    periods = ((2, 4, "Feb-Apr"), (5, 7, "May-Jul"),
               (8, 10, "Aug-Oct"), (11, 12, "Nov-Dec"))
    temporal = []
    for risk_weight in (0.0, 0.15, 0.25):
        path = (BASE_RESULTS / "q4_daily_Q4-3_H3.csv") if risk_weight == 0.0 else (
            RESULTS / f"daily_{_candidate_name(risk_weight, 0)}.csv"
        )
        candidate = pd.read_csv(path, encoding="utf-8-sig")
        candidate["month"] = pd.to_datetime(candidate.date).dt.month
        base = b0.copy()
        base["month"] = pd.to_datetime(base.date).dt.month
        for lo, hi, label in periods:
            a = base[base.month.between(lo, hi)].total_cost.to_numpy()
            b = candidate[candidate.month.between(lo, hi)].total_cost.to_numpy()
            temporal.append({
                "candidate": f"rho={risk_weight:.2f}", "period": label,
                "days": len(a), "saving": float(a.sum() - b.sum()),
                "mean_daily_saving": float(np.mean(a - b)),
                "cvar95_B0": empirical_cvar(a, 0.95),
                "cvar95_candidate": empirical_cvar(b, 0.95),
                "positive_days": int(np.sum(b < a)),
            })
    _atomic_csv(pd.DataFrame(temporal), RESULTS / "temporal_generalization.csv")

    seeds = []
    for risk_weight in (0.0, 0.15):
        for seed in (0, 1, 2):
            if risk_weight == 0.0 and seed == 0:
                frame = h3
            else:
                frame = pd.read_csv(
                    RESULTS / f"daily_{_candidate_name(risk_weight, seed)}.csv",
                    encoding="utf-8-sig",
                )
            seeds.append({
                "risk_weight": risk_weight, "seed": seed,
                "adjusted_cost": float(frame.inventory_adjusted_cost.sum()),
                "cvar95": empirical_cvar(frame.total_cost.to_numpy(), 0.95),
                "saving_vs_B0": float(
                    b0.inventory_adjusted_cost.sum() - frame.inventory_adjusted_cost.sum()
                ),
            })
    _atomic_csv(pd.DataFrame(seeds), RESULTS / "seed_stability.csv")


def smoke_test():
    experiment, policy, risk_solver = _load_modules()
    context = experiment.prepare()
    frozen = _frozen_policy(experiment, policy)
    start, end = 181, 183
    risk_solver.configure(0.0, 0.95)
    baseline, _, _, _ = policy.run_q4(
        context["L"], context["P"], context["V"], context["bank"], context["rb"],
        context["pbank"], context["p"], frozen, context["shape"], context["edges"],
        context["vE_ref"], start=start, end=end, initial=6000.0, seed=0,
    )
    policy.Q4S.solve_q4 = risk_solver.solve_q4
    replay, _, _, _ = policy.run_q4(
        context["L"], context["P"], context["V"], context["bank"], context["rb"],
        context["pbank"], context["p"], frozen, context["shape"], context["edges"],
        context["vE_ref"], start=start, end=end, initial=6000.0, seed=0,
    )
    numeric = [c for c in baseline.columns if c != "date"]
    error = float(np.max(np.abs(baseline[numeric].to_numpy() - replay[numeric].to_numpy())))
    if error > 1e-8:
        raise RuntimeError(f"risk_weight=0 regression failed: {error}")
    _atomic_json({"days": 2, "max_numeric_diff": error, "passed": True},
                 RESULTS / "smoke_test.json")
    print(f"smoke test passed; max diff={error:.3e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("Q4_WORKERS", "6")))
    args = parser.parse_args()

    manifest = {
        "started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
        "frozen_keys": list(FROZEN_KEYS),
        "frozen_values": EXPECTED_FROZEN.tolist(),
        "risk_weights": [0.0, 0.15, 0.25],
        "seeds": [0, 1, 2],
    }
    _atomic_json(manifest, CHECKPOINTS / "run_manifest.json")
    smoke_test()
    if args.smoke_only:
        return

    # Six predeclared runs: baseline seed stability, one main risk candidate
    # across three seeds, and one seed-0 sensitivity candidate.
    tasks = [
        (0.0, 1, 0), (0.0, 2, 1),
        (0.15, 0, 2), (0.15, 1, 3), (0.15, 2, 4),
        (0.25, 0, 5),
    ]
    summaries = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        future_map = {pool.submit(run_candidate, task): task for task in tasks}
        for future in tqdm(
            as_completed(future_map), total=len(future_map),
            desc="parallel annual replays", unit="run", dynamic_ncols=True,
        ):
            task = future_map[future]
            try:
                summaries.append(future.result())
            except Exception as exc:
                _atomic_json({"task": task, "error": repr(exc)},
                             LOGS / f"failed_{task[0]:.2f}_{task[1]}.json")
                raise

    # Add the existing seed-0 H3 summary without rerunning it.
    experiment, policy, _ = _load_modules()
    h3 = pd.read_csv(BASE_RESULTS / "q4_daily_Q4-3_H3.csv", encoding="utf-8-sig")
    summaries.append(policy.summarize(
        h3, candidate="rho0.00_seed0", risk_weight=0.0, seed=0,
        seconds=0.0, resumed=True,
    ))
    build_diagnostics(summaries, args.resamples)
    manifest["finished_at"] = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest["completed"] = True
    _atomic_json(manifest, CHECKPOINTS / "run_manifest.json")
    print(f"numerical experiments complete in {manifest['wall_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
