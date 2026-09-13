"""Risk-aware drop-in replacement for q4_stoch.solve_q4.

The original six-dimensional Q4 policy remains frozen.  This module adds only
an optional convex combination of expected scenario cost and CVaR95.  Setting
``risk_weight=0`` dispatches to the original solver byte-for-byte.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import q4_stoch as _base


BASE_SOLVE = _base.solve_q4
RISK_WEIGHT = 0.0
RISK_BETA = 0.95


def configure(risk_weight: float = 0.0, beta: float = 0.95) -> None:
    global RISK_WEIGHT, RISK_BETA
    if not 0.0 <= risk_weight < 1.0:
        raise ValueError("risk_weight must lie in [0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    RISK_WEIGHT = float(risk_weight)
    RISK_BETA = float(beta)


def solve_q4(nets, prices, weights, initial, p, tv_marginals, tv_edges,
             base=None, m=0, lock=36, emergency_scale=None,
             adj_premium=1.0, first_stage_price=None, ret_scen=False):
    """Solve the Q4 stochastic LP with an optional linear CVaR term."""
    rho = RISK_WEIGHT
    if rho <= 0.0:
        return BASE_SOLVE(
            nets, prices, weights, initial, p, tv_marginals, tv_edges,
            base, m, lock, emergency_scale, adj_premium,
            first_stage_price, ret_scen,
        )

    n = nets.shape[1]
    scenarios = len(nets)
    segments = len(tv_marginals)
    edges = np.asarray(tv_edges, float)
    segment_widths = np.diff(edges)
    energy_floor = float(edges[0])
    emergency_multiplier = (
        np.ones(n) if emergency_scale is None
        else np.asarray(emergency_scale, float)
    )
    use_base = base is not None and m > 0
    weights = np.asarray(weights, float)
    expected_price = (
        first_stage_price if first_stage_price is not None
        else (weights[:, None] * prices).sum(0)
    )

    per_scenario = 8
    off_plan, off_up0, off_down0 = 0, n, 2 * n
    off_scenario = 3 * n
    off_segments = off_scenario + scenarios * per_scenario * n
    base_variables = off_segments + scenarios * segments
    off_zeta = base_variables
    off_excess = off_zeta + 1
    n_variables = off_excess + scenarios

    def block(s, which):
        return off_scenario + s * per_scenario * n + which * n

    eq_rows, eq_cols, eq_values, rhs_parts = [], [], [], []
    row = 0
    idx = np.arange(n)

    def put(rows, cols, values):
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        eq_rows.extend(rows.tolist())
        eq_cols.extend(cols.tolist())
        eq_values.extend(np.broadcast_to(values, len(rows)).tolist())

    if use_base:
        first = np.arange(m)
        put(row + first, off_plan + first, 1.0)
        put(row + first, off_up0 + first, -1.0)
        put(row + first, off_down0 + first, 1.0)
        rhs_parts.append(np.asarray(base, float))
        row += m

    for s in range(scenarios):
        put(row + idx, block(s, 0) + idx, 1.0)
        put(row + idx, off_plan + idx, -1.0)
        put(row + idx, block(s, 1) + idx, -1.0)
        put(row + idx, block(s, 2) + idx, 1.0)
        rhs_parts.append(np.zeros(n))
        row += n

        put(row + idx, block(s, 0) + idx, 1.0)
        put(row + idx, block(s, 7) + idx, 1.0)
        put(row + idx, block(s, 3) + idx, -1.0)
        put(row + idx, block(s, 4) + idx, 1.0)
        put(row + idx, block(s, 5) + idx, -1.0)
        rhs_parts.append(np.asarray(nets[s], float))
        row += n

        put(row + idx, block(s, 6) + idx, 1.0)
        put(row + idx, block(s, 3) + idx, -p.eta)
        put(row + idx, block(s, 4) + idx, 1.0 / p.eta)
        put(row + idx[1:], block(s, 6) + idx[:-1], -1.0)
        rhs_parts.append(np.r_[initial, np.zeros(n - 1)])
        row += n

        put(np.full(segments, row),
            off_segments + s * segments + np.arange(segments), 1.0)
        put([row], [block(s, 6) + n - 1], -1.0)
        rhs_parts.append(np.array([-energy_floor]))
        row += 1

    base_cost = np.zeros(base_variables)
    down_coefficient = (
        -p.down if p.settlements in ("net_refund", "stepwise") else p.down
    )
    if use_base:
        base_cost[off_up0:off_up0 + m] = p.up * expected_price[:m]
        base_cost[off_down0:off_down0 + m] = down_coefficient * expected_price[:m]
        base_cost[off_plan + m:off_plan + n] = expected_price[m:]
    else:
        base_cost[off_plan:off_plan + n] = expected_price

    marginals = np.asarray(tv_marginals, float)
    scenario_vectors = []
    for s in range(scenarios):
        ws = float(weights[s])
        scenario_price = np.asarray(prices[s], float)
        scenario_cost = np.zeros(base_variables)

        # First-stage cost under this price scenario.  Its weighted expectation
        # exactly recovers the original expected-price objective.
        if use_base:
            scenario_cost[off_up0:off_up0 + m] = p.up * scenario_price[:m]
            scenario_cost[off_down0:off_down0 + m] = (
                down_coefficient * scenario_price[:m]
            )
            scenario_cost[off_plan + m:off_plan + n] = scenario_price[m:]
        else:
            scenario_cost[off_plan:off_plan + n] = scenario_price

        scenario_cost[block(s, 1):block(s, 1) + n] = (
            adj_premium * p.up * scenario_price
        )
        scenario_cost[block(s, 2):block(s, 2) + n] = (
            down_coefficient * scenario_price / max(adj_premium, 1e-6)
        )
        scenario_cost[block(s, 7):block(s, 7) + n] = (
            p.emergency * scenario_price * emergency_multiplier
        )
        scenario_cost[block(s, 3):block(s, 3) + n] = p.epsilon
        scenario_cost[block(s, 4):block(s, 4) + n] = p.epsilon
        scenario_cost[
            off_segments + s * segments:off_segments + (s + 1) * segments
        ] = -marginals
        scenario_vectors.append(scenario_cost)

        # Populate the expected-cost objective exactly as in the original LP.
        base_cost[block(s, 1):block(s, 1) + n] = (
            ws * adj_premium * p.up * scenario_price
        )
        base_cost[block(s, 2):block(s, 2) + n] = (
            ws * down_coefficient * scenario_price / max(adj_premium, 1e-6)
        )
        base_cost[block(s, 7):block(s, 7) + n] = (
            ws * p.emergency * scenario_price * emergency_multiplier
        )
        base_cost[block(s, 3):block(s, 3) + n] = ws * p.epsilon
        base_cost[block(s, 4):block(s, 4) + n] = ws * p.epsilon
        base_cost[
            off_segments + s * segments:off_segments + (s + 1) * segments
        ] = -ws * marginals

    objective = np.zeros(n_variables)
    objective[:base_variables] = (1.0 - rho) * base_cost
    objective[off_zeta] = rho
    objective[off_excess:off_excess + scenarios] = (
        rho * weights / (1.0 - RISK_BETA)
    )

    bounds = [(0.0, None)] * base_variables + [(None, None)] + [(0.0, None)] * scenarios
    if use_base:
        for i in range(m, n):
            bounds[off_up0 + i] = (0.0, 0.0)
            bounds[off_down0 + i] = (0.0, 0.0)
    else:
        for i in range(n):
            bounds[off_up0 + i] = (0.0, 0.0)
            bounds[off_down0 + i] = (0.0, 0.0)

    for s in range(scenarios):
        for i in range(n):
            if i < lock:
                bounds[block(s, 1) + i] = (0.0, 0.0)
                bounds[block(s, 2) + i] = (0.0, 0.0)
            bounds[block(s, 3) + i] = (0.0, p.power * _base.DT)
            bounds[block(s, 4) + i] = (0.0, p.power * _base.DT)
            bounds[block(s, 6) + i] = (
                _base.SOC_LO_FRAC * p.capacity,
                _base.SOC_HI_FRAC * p.capacity,
            )
        for k in range(segments):
            bounds[off_segments + s * segments + k] = (0.0, float(segment_widths[k]))

    rhs = np.concatenate(rhs_parts)
    a_eq = coo_matrix(
        (eq_values, (eq_rows, eq_cols)), shape=(len(rhs), n_variables)
    ).tocsr()

    # scenario_cost - zeta - excess_s <= 0
    ub_rows, ub_cols, ub_values = [], [], []
    for s, scenario_cost in enumerate(scenario_vectors):
        nz = np.flatnonzero(scenario_cost)
        ub_rows.extend(np.full(len(nz), s).tolist())
        ub_cols.extend(nz.tolist())
        ub_values.extend(scenario_cost[nz].tolist())
        ub_rows.extend([s, s])
        ub_cols.extend([off_zeta, off_excess + s])
        ub_values.extend([-1.0, -1.0])
    a_ub = coo_matrix(
        (ub_values, (ub_rows, ub_cols)), shape=(scenarios, n_variables)
    ).tocsr()

    fit = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.zeros(scenarios),
        A_eq=a_eq,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )
    if not fit.success:
        return None

    x = fit.x
    plan = x[off_plan:off_plan + n]
    expected_soc = np.zeros(n)
    for s in range(scenarios):
        expected_soc += weights[s] * x[block(s, 6):block(s, 6) + n]
    out = (plan, np.r_[initial, expected_soc], float(fit.fun))
    if ret_scen:
        scenario_plans = np.array([
            x[block(s, 0):block(s, 0) + n] for s in range(scenarios)
        ])
        return out + (scenario_plans,)
    return out
