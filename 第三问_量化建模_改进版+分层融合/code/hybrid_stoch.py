"""L1/L2：情景生成（k-medoids 场景削减）+ 两阶段情景随机 LP（含紧急购电 recourse 与 CVaR）。

动机：改进版主模型用「单一风险分位路径」把随机问题确定性化，本质是一个启发式——
分位数 alpha 是被网格搜索出来的，而不是从成本结构里解出来的。真正对应题目结构的写法是：

  第一阶段（此刻必须签下的合同）：计划购电量 q_t，t 覆盖整个视野；
  第二阶段（观测到真实净负荷后的补救）：充放电 c/dis、弃电 spill、紧急购电 z，逐情景取值。

紧急购电 5 倍电价第一次**显式进入规划目标**，于是「买多少才划算」由 LP 自己解出来，
不再需要人工挑分位。情景来自历史残差路径（保留时段间相关性），用 k-medoids 削减到 S 条
代表路径并带权，权重即簇内样本数。
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from q3 import DT

SOC_LO_FRAC, SOC_HI_FRAC = 0.1, 0.9


# ---------------------------------------------------------------- 情景削减（k-medoids）
def kmedoids(X, k, seed=0, iters=40):
    """欧氏 k-medoids（PAM 的快速变体）：返回代表样本下标与各簇样本数。"""
    n = len(X)
    if k >= n:
        return np.arange(n), np.ones(n, int)
    rng = np.random.default_rng(seed)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    # k-means++ 式初始化
    med = [int(rng.integers(n))]
    for _ in range(k - 1):
        d = D[med].min(axis=0) ** 2
        s = d.sum()
        med.append(int(rng.choice(n, p=d / s)) if s > 0 else int(rng.integers(n)))
    med = np.array(med)
    for _ in range(iters):
        lab = np.argmin(D[med], axis=0)
        new = med.copy()
        for j in range(k):
            idx = np.where(lab == j)[0]
            if len(idx) == 0:
                continue
            new[j] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
        if np.array_equal(np.sort(new), np.sort(med)):
            med = new
            break
        med = new
    lab = np.argmin(D[med], axis=0)
    cnt = np.array([max(int((lab == j).sum()), 1) for j in range(len(med))])
    return med, cnt


def make_scenarios(mean_path, resid, n_scen, seed=0, temper=1.0, shrink=1.0):
    """由历史残差路径构造带权情景：net^(s) = mean + shrink * resid^(s)。

    temper：权重温度，w ∝ cnt**temper（temper=0 等权，1 为经验频率）。
    """
    err = resid[np.isfinite(resid).all(axis=1)]
    if len(err) == 0:
        return mean_path[None, :].copy(), np.array([1.0])
    if len(err) <= n_scen:
        med = np.arange(len(err)); cnt = np.ones(len(err), int)
    else:
        med, cnt = kmedoids(err, n_scen, seed=seed)
    w = cnt.astype(float) ** float(temper)
    w = w / w.sum()
    S = mean_path[None, :] + float(shrink) * err[med]
    return S, w


# ---------------------------------------------------------------- 两阶段情景随机 LP
def solve_stochastic(nets, weights, price, initial, p, tv, original=None, tail_from=None,
                     emergency_scale=None, cvar_beta=0.0, cvar_alpha=0.9):
    """两阶段情景随机 LP。

    变量布局
      q            n                      第一阶段计划购电量（若 original 给出，则为调整后的计划）
      u, v         n, n                   相对基准计划的增购 / 减购（仅 original 非空时启用）
      per scenario s: c_s, dis_s, sp_s, E_s, z_s  各 n
      CVaR         eta (1) + xi_s (S)      仅 cvar_beta>0 时启用

    目标
      sum_t base_cost(q_t) + sum_s w_s [ sum_t kappa_t * 5 p_t z_{s,t} - V(E_{s,n-1}) ]
      + beta * ( eta + 1/(1-alpha) sum_s w_s xi_s )
    """
    n = len(price)
    S = len(nets)
    K = len(tv.marginals)
    wseg = tv.widths()
    E0 = float(tv.edges[0])
    use_adj = original is not None
    m = (n if tail_from is None else tail_from) if use_adj else 0
    escale = np.ones(n) if emergency_scale is None else np.asarray(emergency_scale, float)
    use_cvar = cvar_beta > 0

    nq = n
    nuv = 2 * n if use_adj else 0
    per = 5 * n                                   # c, dis, sp, E, z
    nseg = K * S
    ncv = (1 + S) if use_cvar else 0
    N = nq + nuv + S * per + nseg + ncv

    off_uv = nq
    off_s = nq + nuv
    off_seg = off_s + S * per
    off_cv = off_seg + nseg

    def sblk(s, which):
        return off_s + s * per + which * n

    rr, cc, vv = [], [], []
    rhs = []
    row = 0
    idx = np.arange(n)

    def put(r, c, v):
        r = np.asarray(r); c = np.asarray(c)
        rr.extend(r.tolist()); cc.extend(c.tolist())
        vv.extend(np.broadcast_to(v, len(r)).tolist())

    # 每个情景：功率平衡  q + z - c + dis - sp = net_s
    for s in range(S):
        r = row + idx
        put(r, idx, 1.0)
        put(r, sblk(s, 4) + idx, 1.0)
        put(r, sblk(s, 0) + idx, -1.0)
        put(r, sblk(s, 1) + idx, 1.0)
        put(r, sblk(s, 2) + idx, -1.0)
        rhs.append(np.asarray(nets[s], float))
        row += n
    # 每个情景：储能动态  E_t - eta*c_t + dis_t/eta - E_{t-1} = 0（t=0 时 E_{-1}=initial）
    for s in range(S):
        r = row + idx
        put(r, sblk(s, 3) + idx, 1.0)
        put(r, sblk(s, 0) + idx, -p.eta)
        put(r, sblk(s, 1) + idx, 1.0 / p.eta)
        put(row + idx[1:], sblk(s, 3) + idx[:-1], -1.0)
        rhs.append(np.r_[initial, np.zeros(n - 1)])
        row += n
    # 终端分段： sum_k e_{s,k} - E_{s,n-1} = -E0
    for s in range(S):
        put(np.full(K, row), off_seg + s * K + np.arange(K), 1.0)
        put([row], [sblk(s, 3) + n - 1], -1.0)
        rhs.append(np.array([-E0]))
        row += 1
    # 调整净额： q_t - u_t + v_t = q^0_t （只对已签合同的前 m 段）
    if use_adj and m > 0:
        jj = np.arange(m)
        put(row + jj, jj, 1.0)
        put(row + jj, off_uv + jj, -1.0)
        put(row + jj, off_uv + n + jj, 1.0)
        rhs.append(np.asarray(original, float))
        row += m

    cost = np.zeros(N)
    if use_adj:
        cost[off_uv:off_uv + m] = p.up * price[:m]
        coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
        cost[off_uv + n:off_uv + n + m] = coef * price[:m]
        cost[m:n] = price[m:]
    else:
        cost[:n] = price

    marg = np.asarray(tv.marginals, float)
    for s in range(S):
        w = float(weights[s])
        cost[sblk(s, 4):sblk(s, 4) + n] = w * p.emergency * price * escale
        cost[sblk(s, 0):sblk(s, 0) + n] += w * p.epsilon
        cost[sblk(s, 1):sblk(s, 1) + n] += w * p.epsilon
        cost[off_seg + s * K: off_seg + (s + 1) * K] = -w * marg

    bounds = [(0.0, None)] * N
    for i in range(n):
        if use_adj:
            bounds[off_uv + i] = (0.0, None) if i < m else (0.0, 0.0)
            bounds[off_uv + n + i] = (0.0, None) if i < m else (0.0, 0.0)
    for s in range(S):
        for i in range(n):
            bounds[sblk(s, 0) + i] = (0.0, p.power * DT)
            bounds[sblk(s, 1) + i] = (0.0, p.power * DT)
            bounds[sblk(s, 3) + i] = (SOC_LO_FRAC * p.capacity, SOC_HI_FRAC * p.capacity)
        for k in range(K):
            bounds[off_seg + s * K + k] = (0.0, float(wseg[k]))

    # ---- CVaR：以各情景的「补救成本」为损失变量 ----
    if use_cvar:
        eta_i = off_cv
        bounds[eta_i] = (None, None)
        extra_r, extra_c, extra_v, extra_rhs = [], [], [], []
        for s in range(S):
            xi = off_cv + 1 + s
            bounds[xi] = (0.0, None)
            # xi_s >= loss_s - eta ;  loss_s = sum_t 5 p_t escale_t z_{s,t}
            extra_r.extend([row] * (n + 2))
            extra_c.extend(list(range(sblk(s, 4), sblk(s, 4) + n)) + [eta_i, xi])
            extra_v.extend((p.emergency * price * escale).tolist() + [-1.0, -1.0])
            extra_rhs.append(0.0)
            row += 1
        rr.extend(extra_r); cc.extend(extra_c); vv.extend(extra_v)
        n_ub = len(extra_rhs)
        cost[eta_i] = cvar_beta
        for s in range(S):
            cost[off_cv + 1 + s] = cvar_beta * float(weights[s]) / (1 - cvar_alpha)
    else:
        n_ub = 0

    b = np.concatenate(rhs) if rhs else np.zeros(0)
    n_eq = len(b)
    A = coo_matrix((vv, (rr, cc)), shape=(n_eq + n_ub, N)).tocsr()
    A_eq = A[:n_eq]
    A_ub = A[n_eq:] if n_ub else None
    b_ub = np.zeros(n_ub) if n_ub else None
    fit = linprog(cost, A_eq=A_eq, b_eq=b, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not fit.success:
        return None
    x = fit.x
    q = x[:n]
    Emean = np.zeros(n)
    for s in range(S):
        Emean += float(weights[s]) * x[sblk(s, 3):sblk(s, 3) + n]
    return q, np.r_[initial, Emean], float(fit.fun)


# ---------------------------------------------------------------- 两阶段（带调整期权）情景 LP
def solve_stoch2(nets, weights, price, initial, p, tv, base=None, m=0, lock=36,
                 emergency_scale=None, cvar_beta=0.0, cvar_alpha=0.9, ret_scen=False,
                 adj_premium=1.0):
    """把「日内还能再调整」这件事**写进规划模型**的两阶段情景 LP。

    信息结构（与题目完全对应）：每个发布时刻做出的计划里，只有紧随其后的 6 小时会被真正执行，
    其余时段一定会在下一个发布时刻重新决策。于是

      第一阶段（此刻锁定）：计划 qp_t（t 覆盖整个视野，作为合同基准）；前 lock 段不可再调整；
      第二阶段（观测到情景 s 后）：t >= lock 的再调整 u_s/v_s（1.5 倍 / −0.5 倍），
                                   以及充放电、弃电与紧急购电 z_s（5 倍）。

    与「单一分位路径」相比，模型自己解出「买多少」，并且**同时**权衡三种代价：
    计划电价、调整期权的行权代价、紧急购电，而不是由外部网格搜索一个分位数代替。
    """
    n = len(price)
    S = len(nets)
    K = len(tv.marginals)
    wseg = tv.widths()
    E0 = float(tv.edges[0])
    esc = np.ones(n) if emergency_scale is None else np.asarray(emergency_scale, float)
    use_base = base is not None and m > 0
    use_cvar = cvar_beta > 0

    per = 8                                     # qf, u, v, c, dis, sp, E, z
    off_qp = 0
    off_U0 = n
    off_V0 = 2 * n
    off_s = 3 * n
    off_seg = off_s + S * per * n
    off_cv = off_seg + S * K
    N = off_cv + ((1 + S) if use_cvar else 0)

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
        put(row + jj, off_qp + jj, 1.0)
        put(row + jj, off_U0 + jj, -1.0)
        put(row + jj, off_V0 + jj, 1.0)
        rhs.append(np.asarray(base, float)); row += m
    for s in range(S):
        # qf - qp - u + v = 0
        put(row + idx, blk(s, 0) + idx, 1.0)
        put(row + idx, off_qp + idx, -1.0)
        put(row + idx, blk(s, 1) + idx, -1.0)
        put(row + idx, blk(s, 2) + idx, 1.0)
        rhs.append(np.zeros(n)); row += n
        # qf + z - c + dis - sp = net
        put(row + idx, blk(s, 0) + idx, 1.0)
        put(row + idx, blk(s, 7) + idx, 1.0)
        put(row + idx, blk(s, 3) + idx, -1.0)
        put(row + idx, blk(s, 4) + idx, 1.0)
        put(row + idx, blk(s, 5) + idx, -1.0)
        rhs.append(np.asarray(nets[s], float)); row += n
        # E_t - eta c + dis/eta - E_{t-1} = 0
        put(row + idx, blk(s, 6) + idx, 1.0)
        put(row + idx, blk(s, 3) + idx, -p.eta)
        put(row + idx, blk(s, 4) + idx, 1.0 / p.eta)
        put(row + idx[1:], blk(s, 6) + idx[:-1], -1.0)
        rhs.append(np.r_[initial, np.zeros(n - 1)]); row += n
        # 终端分段
        put(np.full(K, row), off_seg + s * K + np.arange(K), 1.0)
        put([row], [blk(s, 6) + n - 1], -1.0)
        rhs.append(np.array([-E0])); row += 1

    cost = np.zeros(N)
    if use_base:
        coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
        cost[off_U0:off_U0 + m] = p.up * price[:m]
        cost[off_V0:off_V0 + m] = coef * price[:m]
        cost[off_qp + m:off_qp + n] = price[m:]
    else:
        cost[off_qp:off_qp + n] = price
    coef2 = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
    marg = np.asarray(tv.marginals, float)
    for s in range(S):
        w = float(weights[s])
        # 第二阶段 recourse 具有「完美预见」乐观偏差：同一情景内可以对整段视野随意改计划。
        # adj_premium (gamma>=1) 是对这种乐观的显式惩罚——把再调整的行权价抬高，
        # 迫使第一阶段多承担承诺，等价于给期权价值打折。gamma 由 L3 在历史窗口上选定。
        cost[blk(s, 1):blk(s, 1) + n] = w * adj_premium * p.up * price
        cost[blk(s, 2):blk(s, 2) + n] = w * coef2 * price / max(adj_premium, 1e-6)
        cost[blk(s, 7):blk(s, 7) + n] = w * p.emergency * price * esc
        cost[blk(s, 3):blk(s, 3) + n] = w * p.epsilon
        cost[blk(s, 4):blk(s, 4) + n] = w * p.epsilon
        cost[off_seg + s * K:off_seg + (s + 1) * K] = -w * marg

    bounds = [(0.0, None)] * N
    if use_base:
        for i in range(m, n):
            bounds[off_U0 + i] = (0.0, 0.0)
            bounds[off_V0 + i] = (0.0, 0.0)
    else:
        for i in range(n):
            bounds[off_U0 + i] = (0.0, 0.0)
            bounds[off_V0 + i] = (0.0, 0.0)
    for s in range(S):
        for i in range(n):
            if i < lock:
                bounds[blk(s, 1) + i] = (0.0, 0.0)
                bounds[blk(s, 2) + i] = (0.0, 0.0)
            bounds[blk(s, 3) + i] = (0.0, p.power * DT)
            bounds[blk(s, 4) + i] = (0.0, p.power * DT)
            bounds[blk(s, 6) + i] = (SOC_LO_FRAC * p.capacity, SOC_HI_FRAC * p.capacity)
        for k in range(K):
            bounds[off_seg + s * K + k] = (0.0, float(wseg[k]))

    n_ub = 0
    if use_cvar:
        eta_i = off_cv
        bounds[eta_i] = (None, None)
        er, ec_, ev = [], [], []
        for s in range(S):
            xi = off_cv + 1 + s
            bounds[xi] = (0.0, None)
            er.extend([row] * (n + 2))
            ec_.extend(list(range(blk(s, 7), blk(s, 7) + n)) + [eta_i, xi])
            ev.extend((p.emergency * price * esc).tolist() + [-1.0, -1.0])
            row += 1; n_ub += 1
        rr.extend(er); cc.extend(ec_); vv.extend(ev)
        cost[eta_i] = cvar_beta
        for s in range(S):
            cost[off_cv + 1 + s] = cvar_beta * float(weights[s]) / (1 - cvar_alpha)

    b = np.concatenate(rhs)
    n_eq = len(b)
    A = coo_matrix((vv, (rr, cc)), shape=(n_eq + n_ub, N)).tocsr()
    fit = linprog(cost, A_eq=A[:n_eq], b_eq=b,
                  A_ub=(A[n_eq:] if n_ub else None), b_ub=(np.zeros(n_ub) if n_ub else None),
                  bounds=bounds, method='highs')
    if not fit.success:
        return None
    x = fit.x
    qp = x[off_qp:off_qp + n]
    Emean = np.zeros(n)
    for s in range(S):
        Emean += float(weights[s]) * x[blk(s, 6):blk(s, 6) + n]
    out = (qp, np.r_[initial, Emean], float(fit.fun))
    if ret_scen:
        qf = np.array([x[blk(s, 0):blk(s, 0) + n] for s in range(S)])
        zz = np.array([x[blk(s, 7):blk(s, 7) + n] for s in range(S)])
        return out + (qf, zz)
    return out
