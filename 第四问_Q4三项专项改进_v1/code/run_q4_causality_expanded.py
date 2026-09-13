"""Expanded future-information perturbation audit for the frozen H3 policy."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
BASE_ROOT = EXP_ROOT.parent / "第四问_完整交付_主体_第二版"
BASE_CODE = BASE_ROOT / "code"
BASE_RESULTS = BASE_ROOT / "results4"
RESULTS = EXP_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(HERE), str(BASE_CODE)]

STATE = {}


def init_worker():
    import run_q4_experiment as experiment
    import q4_policy as policy
    import q4_core as core
    import q4_stoch as solver
    from q3 import Bank, DATES

    context = experiment.prepare()
    selected = json.loads(
        (BASE_RESULTS / "q4_selected_policy.json").read_text(encoding="utf-8")
    )["Q4-3"]
    frozen = experiment.policy_from(selected["x"])
    STATE.update(
        experiment=experiment, policy=policy, core=core, solver=solver,
        Bank=Bank, DATES=DATES, context=context, frozen=frozen,
    )


def decide(bank, price_bank, day, issue_index, soc, base):
    policy = STATE["policy"]
    core = STATE["core"]
    solver = STATE["solver"]
    context = STATE["context"]
    frozen = STATE["frozen"]
    t = policy.ISSUES[issue_index] * 6
    center = price_bank.mean[day, issue_index]
    nets, prices, weights = core.joint_scenarios(
        bank.mean[day, issue_index],
        bank.residual[max(1, day - context["p"].window):day, issue_index, :],
        center,
        price_bank.resid_window(day, issue_index, context["p"].window),
        frozen.n_scen,
        seed=97 * day + issue_index,
        temper=frozen.temper,
        shrink_net=frozen.shrink_net,
        shrink_price=frozen.shrink_price,
    )
    terminal = frozen.kappa * context["shape"] * float(center.min() / context["p"].eta)
    tail = 0 if base is None else 144 - t
    answer = solver.solve_q4(
        nets, prices, weights, soc, context["p"], terminal, context["edges"],
        base, tail, policy.EXEC_STEPS,
        emergency_scale=policy.emergency_scale(), adj_premium=frozen.adj_premium,
    )
    if answer is None:
        raise RuntimeError(f"infeasible decision day={day} issue={issue_index}")
    return np.asarray(answer[0], float)


def check_day(day):
    context = STATE["context"]
    policy = STATE["policy"]
    core = STATE["core"]
    Bank = STATE["Bank"]
    records = []
    soc = 5000.0

    q0 = decide(context["bank"], context["pbank"], day, 0, soc, None)
    for issue_index, issue_hour in enumerate(policy.ISSUES):
        t = issue_hour * 6
        base = None if issue_index == 0 else q0[t:]
        reference = decide(context["bank"], context["pbank"], day, issue_index, soc, base)

        load = context["L"].copy()
        pv = context["P"].copy()
        forecast = context["F"].copy()
        price = context["V"].copy()
        rng = np.random.default_rng(20260950 + 17 * day + issue_index)

        # Change only values unavailable at the issue time.
        if t < 144:
            load[day, t:] *= 1.0 + 0.35 * rng.random(144 - t)
            pv[day, t:] *= 1.0 + 0.35 * rng.random(144 - t)
            price[day, t:] *= 1.0 + 0.50 * rng.random(144 - t)
        if day + 1 < len(load):
            load[day + 1:] *= 1.0 + 0.20 * rng.random(load[day + 1:].shape)
            pv[day + 1:] *= 1.0 + 0.20 * rng.random(pv[day + 1:].shape)
            price[day + 1:] *= 1.0 + 0.30 * rng.random(price[day + 1:].shape)
        forecast[day, issue_index + 1:] *= 1.60
        if day + 1 < len(forecast):
            forecast[day + 1:] *= 1.60

        perturbed_bank = Bank(load, pv, forecast, context["seed"], context["p"])
        perturbed_price_bank = core.PriceBank(price, context["tou"], context["hp"])
        perturbed_q0 = decide(perturbed_bank, perturbed_price_bank, day, 0, soc, None)
        perturbed_base = None if issue_index == 0 else perturbed_q0[t:]
        perturbed = decide(
            perturbed_bank, perturbed_price_bank, day, issue_index, soc, perturbed_base
        )
        difference = float(np.max(np.abs(reference - perturbed)))
        records.append({
            "date": str(STATE["DATES"][day].date()),
            "day_index": day,
            "issue": int(issue_hour),
            "perturbation": "future_actuals_and_unreleased_forecasts",
            "max_plan_diff_kwh": difference,
            "passed": difference < 1e-6,
        })
    return records


def selected_days(count):
    tail_dates = {
        "2025-05-08", "2025-05-09", "2025-05-17",
        "2025-07-01", "2025-07-02", "2025-07-04",
        "2025-07-06", "2025-07-07", "2025-12-04",
    }
    import q3
    mapping = {str(date.date()): i for i, date in enumerate(q3.DATES)}
    evenly_spaced = np.linspace(31, 364, count, dtype=int).tolist()
    return sorted(set(evenly_spaced + [mapping[x] for x in tail_dates]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", default="48", help="integer sample size or 'all'")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("Q4_WORKERS", "6")))
    args = parser.parse_args()
    days = list(range(31, 365)) if args.days == "all" else selected_days(int(args.days))
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(days)), initializer=init_worker
    ) as pool:
        futures = {pool.submit(check_day, day): day for day in days}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="causality days",
            unit="day", dynamic_ncols=True,
        ):
            rows.extend(future.result())
    frame = pd.DataFrame(rows).sort_values(["day_index", "issue"])
    path = RESULTS / "q4_causality_expanded.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary = {
        "days": len(days), "decision_states": len(frame),
        "workers": args.workers, "max_plan_diff_kwh": float(frame.max_plan_diff_kwh.max()),
        "all_pass": bool(frame.passed.all()), "wall_seconds": time.perf_counter() - start,
    }
    (RESULTS / "q4_causality_expanded.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
