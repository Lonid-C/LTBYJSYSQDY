"""Pure functions copied unchanged from the supplied q2_hybrid.py for regression comparison."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
DT=1/6
T=144

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
    # 3 槽沿用队友方案的三角核；5 槽使用其自然二项式扩展。
    # 两者均对称、权重和为 1，不会引入相位移动或改变日总残差均值。
    kernel = {3: np.array([1., 2., 1.]) / 4,
              5: np.array([1., 4., 6., 4., 1.]) / 16}[int(width)]
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    return np.convolve(np.pad(margin, (pad_left, pad_right), mode="edge"), kernel, mode="valid")
