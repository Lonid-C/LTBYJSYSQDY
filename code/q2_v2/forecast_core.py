"""从已审计 A 提取的因果预测模块；算法保持不变。"""
import numpy as np
from utils import DT,T
LOAD_MEMBERS=("weekly","trend")
PV_MEMBERS=("mean","expw","physics")

def _ridge(x, y, weights, penalties):
    """多输出加权岭回归；y 为一维时返回一维系数。"""
    y = np.asarray(y, float)
    flat = y.ndim == 1
    if flat:
        y = y[:, None]
    gram = x.T @ (weights[:, None] * x) + np.diag(penalties)
    beta = np.linalg.solve(gram, x.T @ (weights[:, None] * y))
    return beta[:, 0] if flat else beta

def _smooth(x, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, mode="nearest")

def _softmax_weights(errors, decay):
    e = np.asarray(errors, float)
    e = e / max(float(e.mean()), 1e-9)
    w = np.exp(-decay * e)
    return w / w.sum()

def load_forecast_weekly(hist_load):
    """上周同日基准 + 最近两周水平变化。"""
    n = len(hist_load)
    if n < 7:
        return hist_load.mean(axis=0)
    out = hist_load[-7].copy()
    if n >= 14:
        ratio = hist_load[-7:].mean() / max(hist_load[-14:-7].mean(), 1e-9)
        out = out * np.clip(ratio, 0.85, 1.15)
    return out

def load_forecast_trend(hist_load, hist_dates, target_date, window=28):
    """星期虚拟变量 + 线性趋势的加权岭回归。"""
    m = min(len(hist_dates), window)
    days = hist_dates[-m:]
    lag = np.array([(d - target_date).days for d in days], float)
    weekday = np.array([d.weekday() for d in days])
    x = np.column_stack([np.ones(m), lag / 7.0,
                         *[(weekday == k).astype(float) for k in range(1, 7)]])
    target = np.array([1.0, 0.0, *[float(target_date.weekday() == k) for k in range(1, 7)]])
    beta = _ridge(x, hist_load[-m:], np.exp(lag / 14.0),
                  np.array([1e-8, 0.20, *([0.05] * 6)]))
    return target @ beta

def pv_forecast_mean(hist_pv, window=5):
    m = min(len(hist_pv), window)
    return hist_pv[-m:].mean(axis=0)

def pv_forecast_expw(hist_pv, tau=3.0, window=28):
    """指数加权均值：近期日权重更高，窗口外截断。"""
    m = min(len(hist_pv), window)
    h = hist_pv[-m:]
    lag = np.arange(m)[::-1]           # 最近一天 lag = 0
    w = np.exp(-lag / tau)
    return (h * w[:, None]).sum(axis=0) / w.sum()

def pv_forecast_physics(hist_pv, *, shape_days=14, level_window=7,
                        shape_quantile=0.90, threshold=0.10, shrink=0.5):
    """光伏物理分解：晴空参考曲线 I_t × 日清晰度水平 k_d。

    I_t 取历史同期的 shape_quantile 分位数（近似晴天包络）；
    k_d = Σ_t PV_{d,t} / Σ_t I_t（仅有效出力时段，能量口径，比逐点中位数稳健）；
    k̂ 由加权局部线性外推并与近期加权均值收缩，最终 PV_hat = k̂ · I_t，
    并截断到历史同期逐段上包络的 1.05 倍以内。

    实测评注：本数据上物理分解单独使用劣于指数加权均值（见 reports），
    因此它只作为组合预测的一个成员，由滚动误差权重自动决定其占比。
    """
    m = min(len(hist_pv), shape_days)
    recent = hist_pv[-m:]
    ref = np.quantile(recent, shape_quantile, axis=0)
    peak = float(ref.max())
    if peak <= 1e-9:
        return pv_forecast_mean(hist_pv, level_window)
    mask = ref > threshold * peak
    if mask.sum() < 6:
        return pv_forecast_mean(hist_pv, level_window)
    k = np.array([day[mask].sum() / ref[mask].sum() for day in recent], float)
    level = min(len(k), max(4, 2 * level_window))
    kk = k[-level:]
    lag = np.arange(level) - (level - 1)
    tau = max(2.0, level / 3.0)
    x = np.column_stack([np.ones(level), lag])
    beta = _ridge(x, kk, np.exp(lag / tau), np.array([1e-8, 0.30]))
    k_trend = float(beta[0] + beta[1])
    k_mean = float(np.average(kk, weights=np.exp(lag / tau)))
    k_hat = float(np.clip((1.0 - shrink) * k_trend + shrink * k_mean, 0.05, 1.25))
    out = k_hat * ref
    out[~mask] = 0.0
    out[np.max(recent, axis=0) <= 0] = 0.0
    return np.minimum(np.maximum(out, 0.0), recent.max(axis=0) * 1.05)

class OptimizedBank:
    """多成员组合预测 + 残差 AR(1) 修正的因果预测库。

    每一天的预测只用该日之前的完整日数据生成，与原实现的信息边界一致。
    组合权重由该日之前已经产生的滚动误差决定，属于"用过去评估过去"，不读未来。
    """

    def __init__(self, data, *, load_window=28, pv_window=5, pv_tau=3.0,
                 pv_shape_days=14, pv_level_window=7, pv_shape_quantile=0.90,
                 pv_threshold=0.10, combine=True, combine_decay=8.0,
                 combine_memory=10, ar_correction=True, ar_window=21, ar_cap=0.6):
        self.data = data
        self.cfg = dict(load_window=load_window, pv_window=pv_window, pv_tau=pv_tau,
                        pv_shape_days=pv_shape_days, pv_level_window=pv_level_window,
                        pv_shape_quantile=pv_shape_quantile, pv_threshold=pv_threshold,
                        combine=combine, combine_decay=combine_decay,
                        combine_memory=combine_memory, ar_correction=ar_correction,
                        ar_window=ar_window, ar_cap=ar_cap)
        n = len(data["dates"])
        dates, load, pv = data["dates"], data["load"], data["pv"]

        fc_load = {k: np.full((n, T), np.nan) for k in LOAD_MEMBERS}
        fc_pv = {k: np.full((n, T), np.nan) for k in PV_MEMBERS}
        for day in range(1, n):
            hl, hp, hd, target = load[:day], pv[:day], dates[:day], dates[day]
            fc_load["weekly"][day] = load_forecast_weekly(hl)
            fc_load["trend"][day] = load_forecast_trend(hl, hd, target, load_window)
            fc_pv["mean"][day] = pv_forecast_mean(hp, pv_window)
            fc_pv["expw"][day] = pv_forecast_expw(hp, pv_tau)
            fc_pv["physics"][day] = pv_forecast_physics(
                hp, shape_days=pv_shape_days, level_window=pv_level_window,
                shape_quantile=pv_shape_quantile, threshold=pv_threshold)

        self.raw_load, self.raw_pv = fc_load, fc_pv
        actual_net = (load - pv) * DT
        K_load, K_pv = len(LOAD_MEMBERS), len(PV_MEMBERS)

        w_load = np.full((n, K_load), 1.0 / K_load)
        w_pv = np.full((n, K_pv), 1.0 / K_pv)
        if combine:
            for day in range(3, n):
                lo = max(1, day - combine_memory)
                if day - lo < 2:
                    continue
                e_load = np.array([np.mean(np.abs(load[lo:day] - fc_load[k][lo:day]))
                                   for k in LOAD_MEMBERS])
                e_pv = np.array([np.mean(np.abs(pv[lo:day] - fc_pv[k][lo:day]))
                                 for k in PV_MEMBERS])
                w_load[day] = _softmax_weights(e_load, combine_decay)
                w_pv[day] = _softmax_weights(e_pv, combine_decay)
        self.weight_load, self.weight_pv = w_load, w_pv

        stack_load = np.stack([fc_load[k] for k in LOAD_MEMBERS], axis=-1)
        stack_pv = np.stack([fc_pv[k] for k in PV_MEMBERS], axis=-1)
        load_hat = (np.einsum("ntk,nk->nt", stack_load, w_load) if combine
                    else fc_load["trend"])
        pv_hat = (np.einsum("ntk,nk->nt", stack_pv, w_pv) if combine
                  else fc_pv["expw"])
        load_hat = np.maximum(_smooth(np.maximum(load_hat, 0.0)), 0.0)
        pv_hat = np.maximum(_smooth(np.maximum(pv_hat, 0.0)), 0.0)
        # 因果掩码：仅当"该日之前 14 天"同时段从未出现光伏时才强制为零。
        # 不能用全年或 1 月的静态掩码——夏季日照时长不同，会把真实出力清零。
        for day in range(1, n):
            lo = max(0, day - 14)
            if lo < day:
                pv_hat[day][np.max(pv[lo:day], axis=0) <= 0] = 0.0
        self.load_hat, self.pv_hat = load_hat, pv_hat

        raw_net = (load_hat - pv_hat) * DT
        raw_resid = actual_net - raw_net
        phi = np.zeros(n)
        if ar_correction:
            for day in range(2, n):
                lo = max(1, day - ar_window)
                if day - lo < 3:
                    continue
                prev = raw_resid[lo - 1:day - 1].ravel()
                curr = raw_resid[lo:day].ravel()
                denom = float(prev @ prev)
                if denom > 1e-12:
                    phi[day] = float(np.clip(float(prev @ curr / denom), 0.0, ar_cap))
        self.phi = phi
        net = raw_net.copy()
        if ar_correction:
            # 首日预测未定义，用 nan_to_num 避免 0 × NaN 污染整条链路
            lagged = np.nan_to_num(raw_resid[:-1], nan=0.0, posinf=0.0, neginf=0.0)
            net[1:] = raw_net[1:] + phi[1:, None] * lagged
        self.net, self.actual_net = net, actual_net
        self.residual = actual_net - net
        self.raw_residual = raw_resid

    def residual_window(self, day, window=28):
        """严格不含 day 的历史残差整日向量；剔除首日未定义等非有限行。"""
        start = max(1, day - window)
        pool = self.residual[start:day]
        if pool.shape[0]:
            pool = pool[np.isfinite(pool).all(axis=1)]
        return pool

    def scenarios(self, day, n_scenarios, rng, window=28):
        """整日残差 block bootstrap，保留日内与跨日相关结构。"""
        pool = self.residual_window(day, window)
        if pool.shape[0] == 0:
            raise ValueError("残差窗口为空")
        return self.net[day][None, :] + pool[rng.integers(0, pool.shape[0], n_scenarios)]

    def quantile_net(self, day, alpha, window=28):
        """基准口径：逐段边际分位数 + 三点平滑。"""
        pool = self.residual_window(day, window)
        margin = np.quantile(pool, alpha, axis=0)
        margin = np.convolve(np.pad(margin, (1, 1), mode="edge"), [.25, .5, .25], "valid")
        return self.net[day] + margin
