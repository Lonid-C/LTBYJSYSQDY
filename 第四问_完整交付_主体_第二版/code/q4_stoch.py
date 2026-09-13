"""第四问 L2 规划层：**电价也随情景变化**的两阶段情景随机 LP。

相对第三问的 `hybrid_stoch.solve_stoch2`，唯一但关键的推广是：每条情景 s 带自己的电价路径
$p^{(s)}$。于是

  第一阶段（此刻签下的合同）q_t：按**期望电价** $\\bar p_t=\\sum_s w_s p^{(s)}_t$ 计价——
      因为合同在交付时段按当时的实际电价结算，而那时的电价此刻还不知道；
  第二阶段（观测到情景后）：再调整 $u^{(s)},v^{(s)}$、充放电、弃电、紧急购电 $z^{(s)}$，
      全部按该情景自己的电价 $p^{(s)}$ 计价。

这一步是必须的：储能套利的价值来自**价差**，而价差在波动电价下本身是随机的；
用一条点预测电路径去规划会系统性高估套利收益、低估紧急购电风险。
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from q3 import DT

SOC_LO_FRAC, SOC_HI_FRAC = 0.1, 0.9


def solve_q4(nets, prices, weights, initial, p, tv_marginals, tv_edges, base=None, m=0, lock=36,
             emergency_scale=None, adj_premium=1.0, first_stage_price=None, ret_scen=False):
    """两阶段情景随机 LP（净负荷与电价联合随机）。

    变量：qp(n) | U0(n) V0(n) | 每情景 [qf,u,v,c,dis,sp,E,z](8n) | 每情景终端分段(K)
    """
    n = nets.shape[1]
    S = len(nets)
    K = len(tv_marginals)
    edges = np.asarray(tv_edges, float)
    wseg = np.diff(edges)
    E0 = float(edges[0])
    esc = np.ones(n) if emergency_scale is None else np.asarray(emergency_scale, float)
    use_base = base is not None and m > 0
    w = np.asarray(weights, float)
    Ep = first_stage_price if first_stage_price is not None else (w[:, None] * prices).sum(0)

    per = 8
    off_qp, off_U0, off_V0 = 0, n, 2 * n
    off_s = 3 * n
    off_seg = off_s + S * per * n
    N = off_seg + S * K

    def blk(s, which):
        return off_s + s * per * n + which * n

    rr, cc, vv, rhs = [], [], [], []
    row = 0
    idx = np.arange(n)

    def put(r, c, v):
        r = np.asarray(r); c = np.asarray(c)
        rr.extend(r.tolist()); cc.extend(c.tolist()); vv.extend(np.broadcast_to(v, len(r)).tolist())

    if use_base:
        jj = np.arange(m)
        put(row + jj, off_qp + jj, 1.0); put(row + jj, off_U0 + jj, -1.0); put(row + jj, off_V0 + jj, 1.0)
        rhs.append(np.asarray(base, float)); row += m
    for s in range(S):
        put(row + idx, blk(s, 0) + idx, 1.0); put(row + idx, off_qp + idx, -1.0)
        put(row + idx, blk(s, 1) + idx, -1.0); put(row + idx, blk(s, 2) + idx, 1.0)
        rhs.append(np.zeros(n)); row += n
        put(row + idx, blk(s, 0) + idx, 1.0); put(row + idx, blk(s, 7) + idx, 1.0)
        put(row + idx, blk(s, 3) + idx, -1.0); put(row + idx, blk(s, 4) + idx, 1.0)
        put(row + idx, blk(s, 5) + idx, -1.0)
        rhs.append(np.asarray(nets[s], float)); row += n
        put(row + idx, blk(s, 6) + idx, 1.0); put(row + idx, blk(s, 3) + idx, -p.eta)
        put(row + idx, blk(s, 4) + idx, 1.0 / p.eta)
        put(row + idx[1:], blk(s, 6) + idx[:-1], -1.0)
        rhs.append(np.r_[initial, np.zeros(n - 1)]); row += n
        put(np.full(K, row), off_seg + s * K + np.arange(K), 1.0)
        put([row], [blk(s, 6) + n - 1], -1.0)
        rhs.append(np.array([-E0])); row += 1

    cost = np.zeros(N)
    coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
    if use_base:
        cost[off_U0:off_U0 + m] = p.up * Ep[:m]
        cost[off_V0:off_V0 + m] = coef * Ep[:m]
        cost[off_qp + m:off_qp + n] = Ep[m:]
    else:
        cost[off_qp:off_qp + n] = Ep
    marg = np.asarray(tv_marginals, float)
    for s in range(S):
        ws = float(w[s]); ps = np.asarray(prices[s], float)
        cost[blk(s, 1):blk(s, 1) + n] = ws * adj_premium * p.up * ps
        cost[blk(s, 2):blk(s, 2) + n] = ws * coef * ps / max(adj_premium, 1e-6)
        cost[blk(s, 7):blk(s, 7) + n] = ws * p.emergency * ps * esc
        cost[blk(s, 3):blk(s, 3) + n] = ws * p.epsilon
        cost[blk(s, 4):blk(s, 4) + n] = ws * p.epsilon
        cost[off_seg + s * K:off_seg + (s + 1) * K] = -ws * marg

    bounds = [(0.0, None)] * N
    if use_base:
        for i in range(m, n):
            bounds[off_U0 + i] = (0.0, 0.0); bounds[off_V0 + i] = (0.0, 0.0)
    else:
        for i in range(n):
            bounds[off_U0 + i] = (0.0, 0.0); bounds[off_V0 + i] = (0.0, 0.0)
    for s in range(S):
        for i in range(n):
            if i < lock:
                bounds[blk(s, 1) + i] = (0.0, 0.0); bounds[blk(s, 2) + i] = (0.0, 0.0)
            bounds[blk(s, 3) + i] = (0.0, p.power * DT)
            bounds[blk(s, 4) + i] = (0.0, p.power * DT)
            bounds[blk(s, 6) + i] = (SOC_LO_FRAC * p.capacity, SOC_HI_FRAC * p.capacity)
        for k in range(K):
            bounds[off_seg + s * K + k] = (0.0, float(wseg[k]))

    b = np.concatenate(rhs)
    A = coo_matrix((vv, (rr, cc)), shape=(len(b), N)).tocsr()
    fit = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method='highs')
    if not fit.success:
        return None
    x = fit.x
    qp = x[off_qp:off_qp + n]
    Emean = np.zeros(n)
    for s in range(S):
        Emean += float(w[s]) * x[blk(s, 6):blk(s, 6) + n]
    out = (qp, np.r_[initial, Emean], float(fit.fun))
    if ret_scen:
        return out + (np.array([x[blk(s, 0):blk(s, 0) + n] for s in range(S)]),)
    return out


def solve_q4_deterministic(net, price, initial, p, tv_marginals, tv_edges, base=None, m=0):
    """对照基线：把随机性用**分位净负荷路径 + 点预测电价**确定化后的滚动 LP（第三问改进版的做法）。"""
    return solve_q4(net[None, :], price[None, :], np.array([1.0]), initial, p,
                    tv_marginals, tv_edges, base, m, lock=10 ** 6)
