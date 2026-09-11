"""日前线性规划与逐时因果执行。内部变量全部为 kWh。"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from utils import BATTERY, TOL, Battery


@dataclass
class Schedule:
    purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    spill: np.ndarray
    soc: np.ndarray  # N+1 个边界值
    objective: float
    dual_objective: float
    solver_iterations: int


def optimize_day(net: np.ndarray, price: np.ndarray, initial: float,
                 terminal: float = 6000.0, battery: Battery = BATTERY,
                 method: str = "highs") -> Schedule:
    """最小化 sum(price * purchase)，终态给定，允许弃电。

    x 的顺序为 [q, c, d, spill, E_1,...,E_N]，各长度 N。
    LP 后消除同时充放电，保持购电和 SOC 不变，保证物理可执行。
    """
    net, price = np.asarray(net, float), np.asarray(price, float)
    n = len(net)
    if net.shape != price.shape or n == 0 or not np.isfinite(net).all():
        raise ValueError("net/price 必须为等长有限向量")
    if not np.isfinite(price).all() or np.any(price <= 0):
        raise ValueError("price 必须严格为正")
    if not battery.minimum - TOL <= initial <= battery.maximum + TOL:
        raise ValueError(f"初态越界：{initial}")
    if not battery.minimum <= terminal <= battery.maximum:
        raise ValueError(f"终态越界：{terminal}")
    initial = float(np.clip(initial, battery.minimum, battery.maximum))
    row, col, value = [], [], []

    def add(r, c, v):
        row.append(r)
        col.append(c)
        value.append(v)

    for t in range(n):
        for block, coefficient in ((0, 1), (1, -1), (2, 1), (3, -1)):
            add(t, block * n + t, coefficient)
        add(n + t, 4 * n + t, 1)
        add(n + t, n + t, -battery.eta_c)
        add(n + t, 2 * n + t, 1 / battery.eta_d)
        if t:
            add(n + t, 4 * n + t - 1, -1)
    add(2 * n, 5 * n - 1, 1)
    A = coo_matrix((value, (row, col)), shape=(2 * n + 1, 5 * n)).tocsr()
    rhs = np.r_[net, np.zeros(n), terminal]
    rhs[n] = initial
    objective = np.r_[price, np.zeros(4 * n)]
    bounds = ([(0, None)] * n + [(0, battery.limit)] * (2 * n)
              + [(0, None)] * n + [(battery.minimum, battery.maximum)] * n)
    fit = linprog(objective, A_eq=A, b_eq=rhs, bounds=bounds,
                  method=method, options={"presolve": True})
    if not fit.success:
        raise RuntimeError(f"LP 失败：status={fit.status}; {fit.message}")
    q, c, d, spill, soc_end = np.split(fit.x, 5)
    # 最优解消环：少充 a、少放 eta_c*eta_d*a，SOC 不变。
    a = np.minimum(c, d / (battery.eta_c * battery.eta_d))
    c = c - a
    d = d - battery.eta_c * battery.eta_d * a
    spill = spill + (1 - battery.eta_c * battery.eta_d) * a
    soc = np.r_[initial, soc_end]
    lower = np.array([b[0] for b in bounds], float)
    upper = np.array([0 if b[1] is None else b[1] for b in bounds], float)
    dual = (rhs @ fit.eqlin.marginals + lower @ fit.lower.marginals
            + upper @ fit.upper.marginals)
    audit = check_dispatch(net, q, c, d, spill, soc, np.zeros(n), battery)
    if not audit["pass"] or abs(soc[-1] - terminal) > TOL:
        raise AssertionError(f"LP 回代不通过：{audit}")
    return Schedule(q, c, d, spill, soc, float(price @ q), float(dual), int(fit.nit))


def execute_step(purchase: float, actual_net: float, soc: float,
                 reference_soc_next: float, reserve_ratio: float,
                 battery: Battery = BATTERY) -> tuple:
    """只接收当期实际值；不可能访问未来负荷。返回 c,d,z,s,E_next。"""
    surplus = purchase - actual_net
    if surplus >= 0:
        charge = min(surplus, battery.limit,
                     max((battery.maximum - soc) / battery.eta_c, 0.0))
        discharge = emergency = 0.0
        spill = surplus - charge
    else:
        reserve = battery.minimum + reserve_ratio * (reference_soc_next - battery.minimum)
        discharge = min(-surplus, battery.limit,
                        battery.eta_d * max(soc - reserve, 0.0))
        charge = spill = 0.0
        emergency = -surplus - discharge
    next_soc = soc + battery.eta_c * charge - discharge / battery.eta_d
    return charge, discharge, emergency, spill, next_soc


def execute_day(purchase: np.ndarray, actual_net: np.ndarray, initial: float,
                reference_soc: np.ndarray, reserve_ratio: float,
                battery: Battery = BATTERY) -> dict:
    n = len(purchase)
    if len(actual_net) != n or len(reference_soc) != n + 1:
        raise ValueError("执行序列长度不一致")
    if not 0 <= reserve_ratio <= 1:
        raise ValueError("reserve_ratio 必须在 [0,1]")
    c, d, z, s = np.zeros((4, n))
    soc = np.empty(n + 1)
    soc[0] = initial
    for t in range(n):
        c[t], d[t], z[t], s[t], soc[t + 1] = execute_step(
            purchase[t], actual_net[t], soc[t], reference_soc[t + 1],
            reserve_ratio, battery)
    return dict(charge=c, discharge=d, emergency=z, spill=s, soc=soc)


def check_dispatch(net, q, c, d, spill, soc, emergency, battery=BATTERY) -> dict:
    residual = q + emergency + d - c - spill - net
    dynamics = np.diff(soc) - battery.eta_c * c + d / battery.eta_d
    metrics = dict(balance_max_abs=float(np.max(np.abs(residual))),
                   soc_equation_max_abs=float(np.max(np.abs(dynamics))),
                   soc_min=float(np.min(soc)), soc_max=float(np.max(soc)),
                   charge_max=float(np.max(c)), discharge_max=float(np.max(d)),
                   negative_min=float(min(np.min(x) for x in (q, c, d, spill, emergency))),
                   simultaneous_max=float(np.max(np.minimum(c, d))))
    metrics["pass"] = bool(
        metrics["balance_max_abs"] <= TOL and metrics["soc_equation_max_abs"] <= TOL
        and metrics["soc_min"] >= battery.minimum - TOL
        and metrics["soc_max"] <= battery.maximum + TOL
        and max(metrics["charge_max"], metrics["discharge_max"]) <= battery.limit + TOL
        and metrics["negative_min"] >= -TOL and metrics["simultaneous_max"] <= TOL)
    return metrics
