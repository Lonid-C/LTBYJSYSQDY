"""第四问：波动电价下的数据层、价格预报层与联合情景层。

与第三问的差别只有一处，但影响贯穿全模型：**电价不再是每天相同的分时电价，而是逐日逐时波动
（附件 4），且决策时点看不到未来的电价。** 于是随机性从"净负荷"一维变成"净负荷 × 电价"二维。

本模块提供
  load4()        读取附件 1—4，并做一致性断言与数据指纹
  PriceBank      严格因果的对数价格比预报（与负荷预报同一估计量）+ 逐提前量残差库
  joint_scenarios()  联合情景：从**同一历史日**同时取净负荷残差与价格对数残差，
                     因此两者的相关结构无需另行建模，天然保留；再用 k-medoids 带权削减
"""
from __future__ import annotations
from pathlib import Path
import sys, json, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import openpyxl

import q3
from q3 import DT, ISSUES, DATES, HOURS, rows as xlrows
import hybrid_stoch as HS

ROOT = Path(__file__).resolve().parents[1]
# 价格预报超参数（四级参数，按**预测精度**选，不按账单选）。
# 这里的默认值必须与 results4/price_hp.json 中实际选出的一组保持一致——模块默认与上报口径
# 不一致是复现性隐患，即使当前所有脚本都显式传参。选参见 select_price_hp()，
# 网格明细见 results4/price_forecast_cv.csv（第一段）与 price_bias_cv.csv（第二段）。
PRICE_HP = dict(window=28, decay=14.0, slope=1e6, weekday=0.025,
                correction=0.5, bias_window=6, bias_decay=3.0)


def load4(out=None):
    """读取附件 1—4。返回 L, P, F, tou, seed, V（365x144 实际电价）。"""
    L, P, F, tou, seed = q3.load()
    a = xlrows(ROOT / 'data/附件4.xlsx')
    V = np.array([x[1:] for x in a[1:]], float)
    assert V.shape == (365, 144), V.shape
    assert np.isfinite(V).all() and (V > 0).all()
    for i, x in enumerate(a[1:]):
        assert pd.Timestamp(x[0]) == DATES[i]
    if out is not None:
        audit = dict(shape=list(V.shape), min=float(V.min()), max=float(V.max()),
                     mean=float(V.mean()), std=float(V.std()),
                     daily_mean_min=float(V.mean(1).min()), daily_mean_max=float(V.mean(1).max()),
                     ratio_to_tou_std=float((V / tou[None, :]).std()),
                     autocorr_daily_multiplier_lag1=float(np.corrcoef((V / tou).mean(1)[:-1], (V / tou).mean(1)[1:])[0, 1]),
                     autocorr_daily_multiplier_lag7=float(np.corrcoef((V / tou).mean(1)[:-7], (V / tou).mean(1)[7:])[0, 1]),
                     sha256=hashlib.sha256((ROOT / 'data/附件4.xlsx').read_bytes()).hexdigest())
        Path(out).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    return L, P, F, tou, seed, V


# ---------------------------------------------------------------- 价格预报
def ratio_forecast(Y, d, window, decay, slope, weekday, offsets=(0,), eps=1e-8):
    """对 Y=log(实际电价 / 附件1 分时形状) 做局部线性趋势 + 星期效应 + 指数时间权重的因果回归。

    与第三问负荷预报是同一个估计量（同一族岭正则），区别只在**不做非负截断**——
    Y 是对数比值，可以为负。offsets=(0,1) 时同时给出当日与次日的预测，供跨午夜视野使用。
    """
    if d <= 0:
        return [np.zeros(Y.shape[1]) for _ in offsets]
    ids = np.arange(max(0, d - window), d)
    lag = (ids - d) / 7
    wd = np.array([DATES[i].weekday() for i in ids])
    X = np.column_stack([np.ones(len(ids)), lag, *[(wd == j) * 1. for j in range(1, 7)]])
    w = np.exp((ids - d) / decay)
    pen = np.array([eps, slope, *([weekday] * 6)])
    beta = np.linalg.solve(X.T @ (w[:, None] * X) + np.diag(pen), X.T @ (w[:, None] * Y[ids]))
    out = []
    for off in offsets:
        twd = (DATES[0].weekday() + d + off) % 7
        x = np.array([1, off / 7, *[float(twd == j) for j in range(1, 7)]])
        out.append(x @ beta)
    return out


def select_price_hp(V, tou, folds=((7, 15), (15, 23), (23, 31)), out=None, out_bias=None):
    """**两段式**滚动原点时间序列交叉验证选择价格预报超参数，判据只用预测精度（RMSE 主、MAE 次）。

    第一段（日前形状）：window / decay / slope / weekday —— 只评价 offset=0 的全天预报，
        这一段与日内偏差校正无关。
    第二段（日内偏差校正）：correction / bias_window / bias_decay —— 固定第一段的选择，
        在 6:00 / 12:00 / 18:00 三个发布时刻的 24 小时视野上评价。
        这三个参数原先是写死的 (1.0, 6, 6.0)，从未参与选参；实测它们在 12:00 会让预报变差，
        因此必须和其余超参放在同一个 1 月网格里一起选。

    两段都只使用 1 月的验证块，评价期（2/1 起）完全留出。
    """
    Y = np.log(V / tou[None, :])
    table = []
    for window in (21, 28, 42, 56):
        for decay in (7., 14., 28.):
            for slope in (0.05, 0.2, 1.0, 1e6):
                for wdp in (0.00625, 0.025, 0.1, 0.4):
                    errs = []
                    for s0, e0 in folds:
                        for d in range(s0, e0):
                            pred = np.exp(ratio_forecast(Y, d, window, decay, slope, wdp)[0]) * tou
                            errs.append(pred - V[d])
                    err = np.concatenate(errs)
                    table.append(dict(window=window, decay=decay, slope=slope, weekday=wdp,
                                      samples=int(err.size), mae=float(np.abs(err).mean()),
                                      rmse=float(np.sqrt((err ** 2).mean()))))
    df = pd.DataFrame(table).sort_values(['rmse', 'mae']).reset_index(drop=True)
    if out is not None:
        df.to_csv(out, index=False, encoding='utf-8-sig', float_format='%.8f')
    b = df.iloc[0]
    hp = dict(window=int(b.window), decay=float(b.decay), slope=float(b.slope), weekday=float(b.weekday))

    # ---- 第二段：日内偏差校正 ----
    days = [d for s0, e0 in folds for d in range(s0, e0)]
    rows = []
    seen = set()
    for corr in (0.0, 0.25, 0.5, 0.75, 1.0):
        for bw in (3, 6, 12):
            for bd in (3.0, 6.0, 12.0):
                key = (0.0, 0, 0.0) if corr == 0.0 else (corr, bw, bd)
                if key in seen:
                    continue
                seen.add(key)
                pb = PriceBank(V, tou, hp, correction=corr, bias_window=bw, bias_decay=bd)
                errs, per = [], {}
                for k in (1, 2, 3):
                    ek = []
                    for d in days:
                        a = pb.actual_horizon(d, k)
                        m = np.isfinite(a)
                        ek.append(pb.mean[d, k][m] - a[m])
                    ek = np.concatenate(ek)
                    per[f'mae_issue{ISSUES[k]}'] = float(np.abs(ek).mean())
                    errs.append(ek)
                err = np.concatenate(errs)
                rows.append(dict(correction=key[0], bias_window=key[1], bias_decay=key[2],
                                 samples=int(err.size), mae=float(np.abs(err).mean()),
                                 rmse=float(np.sqrt((err ** 2).mean())), **per))
    dfb = pd.DataFrame(rows).sort_values(['rmse', 'mae']).reset_index(drop=True)
    if out_bias is not None:
        dfb.to_csv(out_bias, index=False, encoding='utf-8-sig', float_format='%.8f')
    bb = dfb.iloc[0]
    hp.update(correction=float(bb.correction), bias_window=int(bb.bias_window),
              bias_decay=float(bb.bias_decay) if bb.bias_decay > 0 else 6.0)
    return hp, df, dfb


class PriceBank:
    """逐 (日 d, 发布时刻 k) 给出覆盖 24 小时视野的**期望电价路径**与历史对数残差库。

    严格因果：第 d 天第 k 次发布的预报只用 d 之前的完整历史，加上当天 h:00 之前已观测到的电价
    （用于日内偏差校正）。
    """

    def __init__(self, V, tou, hp=None, correction=None, bias_window=None, bias_decay=None):
        hp = dict(hp or PRICE_HP)
        correction = hp.get('correction', 1.0) if correction is None else correction
        bias_window = hp.get('bias_window', 6) if bias_window is None else bias_window
        bias_decay = hp.get('bias_decay', 6.0) if bias_decay is None else bias_decay
        self.V = V
        self.tou = tou
        Y = np.log(V / tou[None, :])
        self.mean = np.zeros((365, 4, 144))
        self.logresid = np.full((365, 4, 144), np.nan)
        shape = np.r_[tou, tou]                      # 跨午夜视野用的两日分时形状
        for d in range(365):
            b0, b1 = ratio_forecast(Y, d, hp['window'], hp['decay'], hp['slope'], hp['weekday'], (0, 1))
            base = np.r_[b0, b1]                     # 288 维：当日 + 次日的对数比值预测
            actual = np.r_[Y[d], Y[d + 1] if d < 364 else np.full(144, np.nan)]
            for k, h in enumerate(ISSUES):
                t = h * 6
                seg = base[t:t + 144].copy()
                if t and correction:
                    obs = slice(max(0, t - bias_window), t)
                    bias = float(np.median(actual[obs] - base[obs]))
                    seg = seg + correction * bias * np.exp(-HOURS / bias_decay)
                self.mean[d, k] = np.exp(seg) * shape[t:t + 144]
                truth = np.r_[V[d], V[d + 1] if d < 364 else np.full(144, np.nan)][t:t + 144]
                self.logresid[d, k] = np.log(truth) - np.log(self.mean[d, k])
        assert np.isfinite(self.mean).all() and (self.mean > 0).all()

    def horizon(self, d, k):
        return self.mean[d, k]

    def actual_horizon(self, d, k):
        t = ISSUES[k] * 6
        a = np.r_[self.V[d], self.V[d + 1] if d < 364 else np.full(144, np.nan)]
        return a[t:t + 144]

    def resid_window(self, d, k, window=28):
        return self.logresid[max(1, d - window):d, k, :]


# ---------------------------------------------------------------- 联合情景
def joint_scenarios(net_mean, net_resid, price_mean, price_logresid, n_scen, seed=0,
                    temper=1.0, shrink_net=1.0, shrink_price=1.0):
    """从**同一批历史日**成对抽取 (净负荷残差, 价格对数残差)，k-medoids 削减为 n_scen 条带权情景。

    成对抽取意味着两种不确定性之间的相关结构不需要另行建模；标准化后拼接再聚类，
    保证两个维度在距离度量里权重可比。
    """
    ok = np.isfinite(net_resid).all(axis=1) & np.isfinite(price_logresid).all(axis=1)
    A = net_resid[ok]
    B = price_logresid[ok]
    if len(A) == 0:
        return net_mean[None, :].copy(), price_mean[None, :].copy(), np.array([1.0])
    sa = A.std() + 1e-9
    sb = B.std() + 1e-9
    Z = np.column_stack([A / sa, B / sb])
    if len(A) <= n_scen:
        med = np.arange(len(A)); cnt = np.ones(len(A), int)
    else:
        med, cnt = HS.kmedoids(Z, n_scen, seed=seed)
    w = cnt.astype(float) ** float(temper)
    w = w / w.sum()
    nets = net_mean[None, :] + float(shrink_net) * A[med]
    prices = price_mean[None, :] * np.exp(float(shrink_price) * B[med])
    return nets, np.maximum(prices, 1e-6), w


# ---------------------------------------------------------------- 终端库存价值（无量纲形状）
def fit_terminal_shape(bank, pbank, p, days, K=8, ngrid=21, capacity=12000.0):
    """一次 Bellman 回代，得到**归一化**的终端库存边际价值形状 shape_k（非增，均值≈1）。

    第三问里终端边际价值是常数 $v_E=p_{\\min}/\\eta$；这里电价逐日波动，所以把价值函数拆成
    「当日价格水平 $v_E(d)=\\hat p_{\\min}(d)/\\eta$」× 「与价格水平无关的形状 shape(E)」，
    形状用 1 月训练日的参数化 LP 拟合一次，之后逐日只缩放不重拟合。
    """
    import q4_stoch as _QS
    lo, hi = 0.1 * capacity, 0.9 * capacity
    grid = np.linspace(lo, hi, ngrid)
    edges = np.linspace(lo, hi, K + 1)
    curves = []
    for d in days:
        center = pbank.mean[d, 0]
        vE = float(center.min() / p.eta)
        flat = np.full(K, vE)
        net = bank.mean[d, 0]
        c = []
        for E in grid:
            out = _QS.solve_q4_deterministic(net, center, float(E), p, flat, edges)
            c.append(np.nan if out is None else out[2] / vE)       # 用 vE 归一化，消掉价格水平
        curves.append(c)
    C = np.nanmean(np.array(curves, float), axis=0)
    marg = np.maximum(-np.diff(C) / np.diff(grid), 0.0)
    marg = np.minimum.accumulate(marg)
    seg = []
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (grid[:-1] >= a - 1e-9) & (grid[:-1] < b - 1e-9)
        seg.append(float(marg[msk].mean()) if msk.any() else float(marg[-1]))
    shape = np.minimum.accumulate(np.array(seg))
    return shape, edges, dict(grid=grid.tolist(), normalized_cost=C.tolist(),
                              days=[str(DATES[d].date()) for d in days])
