"""第三问改进版主程序：结算口径统一 + 经济理论基准分位 + 历史校准 + 滚动 24 小时视野。

本文件对应的两轮修订：
（一）《第三问模型改进建议：结算口径与参数来源说明》
  建议 2.3 / 优先级 1  结算改为「取消部分仅承担 50% 违约成本」
  建议 3   / 优先级 2  交易电价一律取「该电量所属交付时段的分时电价」
  建议 4   / 优先级 3  多次调整按「最终有效购电量相对 0:00 基准计划的净额」结算一次
  建议 7/8 / 优先级 4  风险分位以经济理论基准分位 alpha0=1-1/5=0.80、alpha_h=1-1.5/5=0.70 为锚点
  建议 10  / 优先级 5  校准区间围绕理论锚点展开
  建议 11-14 / 优先级 6 时间窗口的结构来源
  建议 18/19 / 优先级 7 滚动 24 小时视野 + 经济终端库存价值 v_E = p_min/eta_c
  建议 17/20 / 优先级 8 不含无来源的 SOC 惩罚与调整量惩罚
  建议 21  / 优先级 9 仅保留极小数值退化惩罚 epsilon

（二）《第三问改进版：最终修正清单》
  P0-1 扩大午夜风险分位搜索范围，并在最优解落到搜索边界时自动继续向下/向上延展，
       直到最优参数位于已测试区域内部（`calibrate()`）。
  P0-2 报告中删除「未使用跨日预测优化」的错误描述。
  P0-3 0.80/0.70 统一改称「经济理论基准分位 / 无储能局部单阶段理论临界分位」。
  P1-4 弱化「17 天校准得到唯一最优点」的表述，改为报告稳定低成本邻域（`stability_map()`）。
  P1-5 增加滚动 / 扩展窗口时序验证（`rolling_validation()`），以多个验证窗口的平均成本选参。
  P1-6/7 岭正则系数归入「预测子模型工程超参数」，并按**预测精度**（MAE/RMSE）而非最终账单选择。
  P2-9 增加逐次调整结算（`settlements='stepwise'`）作为结构敏感性。
  P2-11 输出风险参数邻域的成本热力图与稳定区间。
  P2-12 报告参数变化下「6+12+18 仍优于不更新」的比例。

时间口径：所有预测只使用决策时点之前已经可获得的信息（严格因果）。
"""
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from itertools import product
import json, hashlib, time, platform
import numpy as np
import pandas as pd
import openpyxl
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
DT = 1 / 6
HOURS = np.arange(1, 145) / 6
ISSUES = (0, 6, 12, 18)
DATES = pd.date_range('2025-01-01', periods=365)
SELECTED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']

# ---------------------------------------------------------------- 参数体系
# 一级：题目直接给定（禁止调参）  m=5, a=1.5, b=0.5, B=12000, Pmax=5000, SOC 10%--90%
# 二级：理论推导参数  alpha0^base=1-1/m=0.80、alpha_h^base=1-a/m=0.70、v_E=p_min/eta_c
# 三级：历史数据校准参数  最终 alpha0 / alpha_h、误差窗口、负荷衰减、日内偏差窗口
# 四级：数值与预测超参数  岭正则系数、epsilon（不具经济含义）
THEORY_ALPHA0 = 1 - 1 / 5.0            # 0.80 —— 经济理论基准分位
THEORY_ALPHA_UPDATE = 1 - 1.5 / 5.0    # 0.70 —— 经济理论基准分位
CALIB_ALPHA0 = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
CALIB_ALPHA_UPDATE = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
RIDGE_SLOPE = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20, 1e6)     # 趋势项岭正则候选（1e6 表示趋势项被完全压缩）
RIDGE_WEEKDAY = (0.00625, 0.0125, 0.025, 0.05, 0.10, 0.20, 0.40)   # 星期效应岭正则候选


@dataclass(frozen=True)
class Params:
    alpha0: float = THEORY_ALPHA0              # 三级：由历史校准确定，默认取理论基准分位
    alpha_update: float = THEORY_ALPHA_UPDATE  # 三级
    window: int = 28                           # 三级：4 个完整周周期
    decay_days: float = 14.0                   # 三级：2 个完整周周期
    bias_window: int = 6                       # 三级：6 x 10min = 1h
    bias_decay: float = 6.0                    # 题目时间结构：相邻预测发布间隔
    slope: float = 0.2                         # 四级：趋势项岭正则（按预测精度选择）
    weekday: float = 0.05                      # 四级：星期效应岭正则（按预测精度选择）
    eta: float = 0.9                           # 一级
    capacity: float = 12000.0                  # 一级
    power: float = 5000.0                      # 一级
    emergency: float = 5.0                     # 一级
    up: float = 1.5                            # 一级
    down: float = 0.5                          # 一级
    settlements: str = 'net_refund'            # net_refund（主）| legacy_gross | stepwise
    terminal: str = 'inventory_value'          # 主：经济终端库存价值
    target: float = 0.5                        # 仅 terminal='fixed_target' 对照口径使用
    horizon: str = 'rolling24'                 # 主：24 小时滚动视野
    reserve: float = 0.0                       # 主模型不含无来源库存保留系数
    correction: float = 1.0
    forecast_scale: float = 1.0
    threshold: float = 0.0
    interpolate: str = 'linear'
    epsilon: float = 1e-7                      # 四级：纯数值退化惩罚
    lam_delta: float = 0.0                     # 建模原则：不叠加调整量惩罚


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                            encoding='utf8')


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.10f')


def rows(path, sheet=0):
    w = openpyxl.load_workbook(path, read_only=True, data_only=True)
    s = w.worksheets[sheet]
    a = list(s.values)
    w.close()
    return a


def load():
    """读取附件1--3；返回 负荷 L、光伏 P、预报 F、分时电价 price、负荷预报种子 seed。"""
    a = rows(ROOT / 'data/附件1.xlsx')
    price = np.array([x[1] for x in a[1:]], float)
    ll = rows(ROOT / 'data/附件2.xlsx')
    pp = rows(ROOT / 'data/附件2.xlsx', 1)
    L = np.array([x[1:] for x in ll[1:]], float)
    P = np.array([x[1:] for x in pp[1:]], float)
    f = rows(ROOT / 'data/附件3.xlsx')
    F = np.array([x[2:] for x in f[1:]], float).reshape(365, 4, 24)
    for i in range(365):
        assert pd.Timestamp(ll[i + 1][0]) == DATES[i] and pd.Timestamp(pp[i + 1][0]) == DATES[i]
        assert pd.Timestamp(f[1 + 4 * i][0]) == DATES[i]
        for k, h in enumerate(ISSUES):
            assert f[1 + 4 * i + k][1] == f'{h}:00'
    assert L.shape == P.shape == (365, 144) and F.shape == (365, 4, 24)
    assert all(np.isfinite(x).all() and (x >= 0).all() for x in (L, P, F, price))
    audit = {
        'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data').glob('*.xlsx')},
        'arrays': {n: {'shape': list(x.shape), 'min': float(x.min()), 'max': float(x.max()),
                       'missing': int(np.isnan(x).sum())}
                   for n, x in [('load', L), ('pv', P), ('forecast', F), ('price', price)]},
        'price_min': float(price.min()), 'price_max': float(price.max()), 'price_unique': int(len(set(price.tolist()))),
        'terminal_inventory_value_vE': float(price.min() / 0.9),
        'forecast_horizon_hours': 24,
        'forecast_horizon_note': '附件3 每个发布时刻给出其后 24 小时的逐时光伏预报，因此跨午夜 24 小时滚动视野可直接使用当前已发布的次日预测部分实现',
        'dates_checked': 365, 'forecast_issue_rows_checked': 1460,
        'time_convention': '10 minute interval ending at source timestamp; interpolation at right endpoint',
        'template_correction': 'B:EO relabeled 00:00-00:10 through 23:50-24:00; original template retained',
    }
    dump('data_audit.json', audit)
    return L, P, F, price, np.array([x[2] for x in a[1:]], float)


# ---------------------------------------------------------------- 负荷预报（预测子模型）
def load_forecast(L, d, seed, window=28, decay_days=14.0, horizon_days=0, slope=.2, weekday=.05, eps=1e-8):
    """严格因果的负荷基准预报：局部线性趋势 + 星期效应 + 指数时间权重。

    slope / weekday 是**预测子模型的岭正则超参数**（四级参数），用于控制有限样本下
    趋势项与星期效应的估计方差，不具有经济含义；其取值按预测精度（MAE/RMSE）选取，
    而不是按最终购电账单选取，以避免预测模型与决策模型围绕账单过度拟合。
    """
    if d <= 0:
        return [seed.copy() for _ in range(horizon_days + 1)]
    ids = np.arange(max(0, d - window), d)
    lag = (ids - d) / 7
    wd = np.array([DATES[i].weekday() for i in ids])
    X = np.column_stack([np.ones(len(ids)), lag, *[(wd == j) * 1. for j in range(1, 7)]])
    w = np.exp((ids - d) / decay_days)
    penalty = np.array([eps, slope, *([weekday] * 6)])
    beta = np.linalg.solve(X.T @ (w[:, None] * X) + np.diag(penalty), X.T @ (w[:, None] * L[ids]))
    out = []
    for off in range(horizon_days + 1):
        target_wd = (DATES[0].weekday() + d + off) % 7
        x = np.array([1, off / 7, *[float(target_wd == j) for j in range(1, 7)]])
        out.append(np.maximum(x @ beta, 0))
    return out


def select_ridge(L, seed, p, folds=((7, 15), (15, 23), (23, 31))):
    """用**滚动原点时间序列交叉验证**选择岭正则超参数（清单 §6.4）。

    对每个候选 (slope, weekday)，逐日执行「只用该日之前的数据拟合 → 预测该日全天负荷」，
    即拟合窗口随原点滚动扩展（rolling-origin CV）；误差按验证区块汇总，主判据为整体 RMSE，
    并以 MAE 作为次级判据。该准则只使用预测误差，不使用任何购电成本信息，
    因此不会让预测子模型与决策模型围绕最终账单一起过度拟合。
    """
    table, fold_rows = [], []
    for slope, weekday in product(RIDGE_SLOPE, RIDGE_WEEKDAY):
        errs, per_fold = [], []
        for s, e in folds:
            e_fold = []
            for d in range(s, e):
                pred = load_forecast(L, d, seed, p.window, p.decay_days, 0, slope, weekday)[0]
                e_fold.append(pred - L[d])
            e_fold = np.concatenate(e_fold)
            errs.append(e_fold)
            per_fold.append(dict(train_window=f'2025-01-01/{DATES[s - 1].date()}', val_block=f'{DATES[s].date()}/{DATES[e - 1].date()}',
                                 samples=int(e_fold.size), mae_kw=float(np.mean(abs(e_fold))),
                                 rmse_kw=float(np.sqrt(np.mean(e_fold ** 2)))))
        err = np.concatenate(errs)
        table.append(dict(slope=slope, weekday=weekday, samples=int(err.size),
                          mae_kw=float(np.mean(abs(err))), rmse_kw=float(np.sqrt(np.mean(err ** 2))),
                          worst_block_rmse_kw=float(max(f['rmse_kw'] for f in per_fold))))
        for f, (s, e) in zip(per_fold, folds):
            fold_rows.append(dict(slope=slope, weekday=weekday, fold=folds.index((s, e)) + 1, **f))
    df = pd.DataFrame(table).sort_values(['rmse_kw', 'mae_kw']).reset_index(drop=True)
    csv('ridge_selection.csv', df.to_dict('records'))
    csv('ridge_cv_folds.csv', fold_rows)
    best = df.iloc[0]
    print('RIDGE selected', best.slope, best.weekday, best.rmse_kw, flush=True)
    return float(best.slope), float(best.weekday), df



class Bank:
    """滚动视野预报库：对每个 (日 d, 发布时刻 k) 给出覆盖 24 小时的规划净负荷，并按提前量组织残差库。"""

    def __init__(self, L, P, F, seed, p):
        self.p = p
        self.L = L
        self.P = P
        self.mean = np.full((365, 4, 144), np.nan)
        self.pv = self.mean.copy()
        self.load = self.mean.copy()
        for d in range(365):
            base = load_forecast(L, d, seed, p.window, p.decay_days, 1, p.slope, p.weekday)
            for k, h in enumerate(ISSUES):
                t = h * 6
                n1 = 144 - t
                anchor = P[d, t - 1] if t else (P[d - 1, -1] if d else 0.)
                hrs = h + HOURS
                if p.interpolate == 'linear':
                    pv = np.interp(hrs, h + np.arange(25), np.r_[anchor, F[d, k]])
                else:
                    off = np.minimum(np.ceil(hrs - h).astype(int) - 1, 23)
                    pv = F[d, k, off]
                pv = np.maximum(pv * p.forecast_scale, 0)
                bias = float(np.median(L[d, max(0, t - p.bias_window):t] - base[0][max(0, t - p.bias_window):t])) if t else 0.
                decay = np.exp(-(hrs - h) / p.bias_decay)
                lf = np.empty(144)
                lf[:n1] = base[0][t:] + p.correction * bias * decay[:n1]
                lf[n1:] = base[1][:t] + p.correction * bias * decay[n1:]
                lf = np.maximum(lf, 0)
                self.pv[d, k] = pv
                self.load[d, k] = lf
                self.mean[d, k] = (lf - pv) * DT
        actual = (L - P) * DT
        self.residual = np.full((365, 4, 144), np.nan)
        for d in range(365):
            for k, h in enumerate(ISSUES):
                t = h * 6
                n1 = 144 - t
                self.residual[d, k, :n1] = actual[d, t:] - self.mean[d, k, :n1]
                if d + 1 <= 364:
                    self.residual[d, k, n1:] = actual[d + 1, :t] - self.mean[d, k, n1:]
        self.cache = {}

    def refresh(self):
        actual = (self.L - self.P) * DT
        self.mean = (self.load - self.pv) * DT
        self.residual = np.full((365, 4, 144), np.nan)
        for d in range(365):
            for k, h in enumerate(ISSUES):
                t = h * 6
                n1 = 144 - t
                self.residual[d, k, :n1] = actual[d, t:] - self.mean[d, k, :n1]
                if d + 1 <= 364:
                    self.residual[d, k, n1:] = actual[d + 1, :t] - self.mean[d, k, n1:]
        self.cache = {}

    def stale_pv_issues(self, ks):
        """把指定发布时刻的光伏预报替换为「该区间所属日历日 0:00 发布的版本」。"""
        for d in range(365):
            for k in ks:
                t = ISSUES[k] * 6
                n1 = 144 - t
                self.pv[d, k, :n1] = self.pv[d, 0, t:]
                if d + 1 <= 364:
                    self.pv[d, k, n1:] = self.pv[d + 1, 0, :t]
        self.refresh()

    def risk(self, d, k, alpha):
        key = (d, k, alpha)
        if key not in self.cache:
            if d >= 7:
                err = self.residual[max(1, d - self.p.window):d, k, :]
                adj = np.nanquantile(err, alpha, axis=0)
                adj = np.nan_to_num(adj, nan=0.0)
            else:
                adj = np.zeros(144)
            self.cache[key] = self.mean[d, k] + adj
        return self.cache[key]

    def actual_horizon(self, P, d, k):
        t = ISSUES[k] * 6
        n1 = 144 - t
        a = np.full(144, np.nan)
        a[:n1] = P[d, t:]
        if d + 1 < P.shape[0]:
            a[n1:] = P[d + 1, :t]
        return a


# ---------------------------------------------------------------- 规划 LP
def solve(net, price, initial, p, vE=0.0, original=None, tail_from=None):
    """滚动视野线性规划：x = [q, c, dis, spill, E, u, v]。

    已纳入合同的段按净额结算：q_t - u_t + v_t = q_t^0；
      · net_refund / legacy_gross：基准为当日 0:00 基准计划 q^0；
      · stepwise：基准为「本次发布前正在执行的有效计划」，用于逐次调整结算对照。
    尚未纳入合同的次日展望段按普通交付电价计价；视野末端库存按 v_E 计价。
    """
    n = len(net)
    idx = np.arange(n)
    rr, cc, vv = [], [], []

    def put(r, c, v):
        rr.extend(np.asarray(r).tolist())
        cc.extend(np.asarray(c).tolist())
        vv.extend(np.broadcast_to(v, len(r)).tolist())

    put(idx, idx, 1); put(idx, n + idx, -1); put(idx, 2 * n + idx, 1); put(idx, 3 * n + idx, -1)
    put(n + idx, 4 * n + idx, 1); put(n + idx, n + idx, -p.eta); put(n + idx, 2 * n + idx, 1 / p.eta)
    put(n + idx[1:], 4 * n + idx[:-1], -1)
    rhs = np.r_[net, initial, np.zeros(n - 1)]

    cost = np.zeros(7 * n)
    cost[n:3 * n] = p.epsilon
    bounds = [(0, None)] * (4 * n) + [(.1 * p.capacity, .9 * p.capacity)] * n + [(0, 0)] * (2 * n)
    for i in range(n):
        bounds[n + i] = bounds[2 * n + i] = (0, p.power * DT)

    if p.terminal == 'fixed_target':
        tgt = p.target * p.capacity
        bounds[5 * n - 1] = (tgt, tgt)
    else:
        cost[5 * n - 1] = -vE

    if original is None:
        cost[:n] = price
    else:
        m = n if tail_from is None else tail_from
        jj = np.arange(m)
        put(2 * n + jj, jj, 1); put(2 * n + jj, 5 * n + jj, -1); put(2 * n + jj, 6 * n + jj, 1)
        rhs = np.r_[rhs, original]
        cost[5 * n:5 * n + m] = p.up * price[:m]
        coef = -p.down if p.settlements in ('net_refund', 'stepwise') else p.down
        cost[6 * n:6 * n + m] = coef * price[:m]
        cost[m:n] = price[m:]
        bounds[5 * n:5 * n + m] = [(0, None)] * m   # 增购量 u >= 0
        bounds[6 * n:6 * n + m] = [(0, None)] * m   # 减购量 v >= 0
    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), 7 * n)).tocsr()
    fit = linprog(cost, A_eq=A, b_eq=rhs, bounds=bounds, method='highs')
    if not fit.success:
        return None
    assert np.max(np.abs(A @ fit.x - rhs)) < 1e-5
    q, c, dis, spill, E, u, v = fit.x.reshape(7, n)
    return q, np.r_[initial, E], float(fit.fun)


def settle_coef(p):
    return -p.down if p.settlements in ('net_refund', 'stepwise') else p.down


def execute(q, net, soc, ref, p):
    if q >= net:
        c = min(q - net, p.power * DT, (.9 * p.capacity - soc) / p.eta)
        dis = z = 0.
        spill = q - net - c
    else:
        reserve = .1 * p.capacity + p.reserve * (ref - .1 * p.capacity)
        dis = min(net - q, p.power * DT, max(0, soc - reserve) * p.eta)
        c = spill = 0.
        z = net - q - dis
    return c, dis, z, spill, soc + p.eta * c - dis / p.eta


def run(L, P, price, bank, p, mask=(6, 12, 18), start=31, end=365, initial=6000., detail=False):
    """滚动回测：每个发布时刻优化 24 小时视野，只执行紧随其后的 6 小时。

    结算口径：
      · net_refund：最终有效购电量相对 0:00 基准计划净额结算一次（主口径）；
      · legacy_gross：原费照付，减购另加 50% 罚金（旧解释对照）；
      · stepwise：逐次调整分别计费，基准为上一次有效计划（逐次结算对照）。
    """
    vE = float(price.min() / p.eta)
    dcoef = settle_coef(p)
    stepwise = (p.settlements == 'stepwise')
    soc = initial
    daily, intervals, decisions = [], [], []
    for day in range(start, end):
        begin = soc
        pred = solve(bank.risk(day, 0, p.alpha0), price, soc, p, vE)
        if pred is None:
            raise RuntimeError('day-ahead infeasible')
        q0, refday, _ = pred
        final = q0.copy()
        c = np.zeros(144); dis = c.copy(); z = c.copy(); spill = c.copy()
        step_up_cost = 0.0; step_dn_cost = 0.0; step_up_kwh = 0.0; step_dn_kwh = 0.0
        states = [soc]
        for k, h in enumerate(ISSUES):
            t = h * 6
            n1 = 144 - t
            exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
            if h in mask:
                if p.horizon == 'day':
                    tail = n1
                    risk_k = bank.risk(day, k, p.alpha_update)[:n1]
                    priceH = price[t:]
                    exec_ref = refday[t:]
                else:
                    tail = n1
                    risk_k = bank.risk(day, k, p.alpha_update)
                    priceH = np.r_[price[t:], price[:t]]
                    exec_ref = np.r_[refday[t:], np.full(t, refday[-1])]
                base = final[t:] if stepwise else q0[t:]
                candidate = solve(risk_k, priceH, soc, p, vE, base, tail_from=tail)
                if candidate is None:
                    raise RuntimeError('adjustment infeasible')
                qn, rn, _ = candidate

                def expected(q, rr):
                    ss = soc
                    zz = []
                    for j, nn in enumerate(risk_k):
                        _, _, ee, _, ss = execute(q[j], nn, ss, rr[j + 1], p)
                        zz.append(ee)
                    dd = q[:tail] - base
                    adj = price[t:] @ (p.up * np.maximum(dd, 0) + dcoef * np.maximum(-dd, 0))
                    future = priceH[tail:] @ q[tail:]
                    return adj + future + p.emergency * (priceH @ zz) - vE * (ss - soc)

                old = np.r_[final[t:], risk_k[tail:]]
                gain = expected(old, exec_ref) - expected(qn, rn)
                accept = (p.threshold <= 0 or gain > p.threshold)
                if accept:
                    if stepwise:
                        d = qn[:tail] - final[t:]
                        step_up_cost += float(price[t:] @ (p.up * np.maximum(d, 0)))
                        step_dn_cost += float(price[t:] @ (dcoef * np.maximum(-d, 0)))
                        step_up_kwh += float(np.maximum(d, 0).sum())
                        step_dn_kwh += float(np.maximum(-d, 0).sum())
                    final[t:] = qn[:tail]
                    refday[t:] = rn[:tail + 1]
                    exec_ref = rn
                if detail:
                    decisions.append(dict(date=str(DATES[day].date()), issue=h, estimated_gain=gain, accepted=accept,
                                          history_end=str(DATES[day - 1].date()), observed_end=f'{h:02d}:00',
                                          up_kwh=float(np.maximum(qn[:tail] - base, 0).sum()),
                                          down_kwh=float(np.maximum(base - qn[:tail], 0).sum())))
            for j in range(36):
                tt = t + j
                c[tt], dis[tt], z[tt], spill[tt], soc = execute(final[tt], (L[day, tt] - P[day, tt]) * DT, soc, exec_ref[j + 1], p)
                states.append(soc)
        states = np.array(states)
        delta = final - q0
        pc = float(price @ q0)
        if stepwise:
            uc, dc = step_up_cost, step_dn_cost
            up_kwh, down_kwh = step_up_kwh, step_dn_kwh
        else:
            uc = float(price @ (p.up * np.maximum(delta, 0)))
            dc = float(price @ (dcoef * np.maximum(-delta, 0)))
            up_kwh = float(np.maximum(delta, 0).sum())
            down_kwh = float(np.maximum(-delta, 0).sum())
        ec = float(p.emergency * (price @ z))
        cash = pc + uc + dc + ec
        residual = final + z + dis - c - spill - (L[day] - P[day]) * DT
        dyn = np.diff(states) - p.eta * c + dis / p.eta
        assert abs(residual).max() < 1e-5 and abs(dyn).max() < 1e-5
        assert states.min() >= .1 * p.capacity - 1e-5 and states.max() <= .9 * p.capacity + 1e-5
        assert min(c.min(), dis.min(), z.min(), spill.min(), final.min()) >= -1e-5
        assert max(c.max(), dis.max()) <= p.power * DT + 1e-5 and np.minimum(c, dis).max() < 1e-5
        daily.append(dict(date=str(DATES[day].date()), plan_cost=pc, up_cost=uc, down_net_cost=dc, emergency_cost=ec,
                          total_cost=cash, inventory_adjusted_cost=cash + vE * (begin - soc),
                          plan_kwh=q0.sum(), final_kwh=final.sum(), emergency_kwh=z.sum(),
                          up_kwh=up_kwh, down_kwh=down_kwh, spill_kwh=spill.sum(),
                          soc_start=begin, soc_end=soc,
                          balance_error=abs(residual).max(), dynamics_error=abs(dyn).max(),
                          soc_min=states.min(), soc_max=states.max()))
        if detail:
            for tt in range(144):
                intervals.append(dict(date=str(DATES[day].date()), t=tt, price=price[tt], load_kw=L[day, tt], pv_kw=P[day, tt],
                                      plan=q0[tt], adjusted=final[tt], charge=c[tt], discharge=dis[tt], emergency=z[tt],
                                      spill=spill[tt], soc_start=states[tt], soc_end=states[tt + 1],
                                      plan_cost=price[tt] * q0[tt], up_cost=price[tt] * p.up * max(delta[tt], 0),
                                      down_net_cost=price[tt] * dcoef * max(-delta[tt], 0),
                                      emergency_cost=price[tt] * p.emergency * z[tt]))
    return pd.DataFrame(daily), pd.DataFrame(intervals), pd.DataFrame(decisions)


def summary(df, **kw):
    return dict(**kw, total_cost=df.total_cost.sum(), adjusted_cost=df.inventory_adjusted_cost.sum(),
                emergency_cost=df.emergency_cost.sum(), emergency_kwh=df.emergency_kwh.sum(),
                up_kwh=df.up_kwh.sum(), down_kwh=df.down_kwh.sum(), plan_cost=df.plan_cost.sum(),
                up_cost=df.up_cost.sum(), down_net_cost=df.down_net_cost.sum(),
                plan_kwh=df.plan_kwh.sum(), final_kwh=df.final_kwh.sum(),
                daily_var95=df.total_cost.quantile(.95),
                daily_cvar95=df.total_cost[df.total_cost >= df.total_cost.quantile(.95)].mean(),
                final_soc=df.soc_end.iloc[-1])


# ---------------------------------------------------------------- 参数搜索
def calibrate(L, P, price, bank, p, initial, start=14, end=31, grid0=None, gridh=None, max_expand=3):
    """风险分位校准，并做搜索边界自动延展：最优解落在边界时继续向下/向上扩展后重测。"""
    grid0 = list(grid0 or CALIB_ALPHA0)
    gridh = list(gridh or CALIB_ALPHA_UPDATE)
    records, log = {}, []
    for step in range(max_expand + 1):
        for a0, au in product(grid0, gridh):
            pp = replace(p, alpha0=a0, alpha_update=au)
            df, _, _ = run(L, P, price, bank, pp, start=start, end=end, initial=initial)
            records[(a0, au)] = summary(df, alpha0=a0, alpha_update=au)
        rows_ = list(records.values())
        best = min(rows_, key=lambda x: x['adjusted_cost'])
        lo0, hi0 = min(grid0), max(grid0)
        loh, hih = min(gridh), max(gridh)
        at_edge = []
        if abs(best['alpha0'] - lo0) < 1e-9:
            at_edge.append('alpha0_low')
        if abs(best['alpha0'] - hi0) < 1e-9:
            at_edge.append('alpha0_high')
        if abs(best['alpha_update'] - loh) < 1e-9:
            at_edge.append('alpha_update_low')
        if abs(best['alpha_update'] - hih) < 1e-9:
            at_edge.append('alpha_update_high')
        log.append(dict(step=step, grid_alpha0=[lo0, hi0], grid_alpha_update=[loh, hih],
                        n_candidates=len(grid0) * len(gridh),
                        best_alpha0=best['alpha0'], best_alpha_update=best['alpha_update'],
                        best_adjusted_cost=best['adjusted_cost'], at_edge=at_edge))
        print(f'CALIB step={step} grid0=[{lo0},{hi0}] best=({best["alpha0"]},{best["alpha_update"]}) edge={at_edge}', flush=True)
        if not at_edge or step == max_expand:
            break
        if 'alpha0_low' in at_edge and lo0 > 0.26:
            grid0 = [round(lo0 - 0.05, 4)] + grid0
        elif 'alpha0_high' in at_edge and hi0 < 0.985:
            grid0 = grid0 + [round(hi0 + 0.05, 4)]
        if 'alpha_update_low' in at_edge and loh > 0.31:
            gridh = [round(loh - 0.05, 4)] + gridh
        elif 'alpha_update_high' in at_edge and hih < 0.96:
            gridh = gridh + [round(hih + 0.05, 4)]
    return list(records.values()), best, log, grid0, gridh


def rolling_validation(L, P, price, bank, p, warm, grid0, gridh, blocks=((7, 15), (15, 23), (23, 31))):
    """滚动 / 扩展窗口时序验证（清单 §5.3 方案B）。

    第 1 折：训练 1/1—1/7 → 验证 1/8—1/15；
    第 2 折：训练窗口扩展为 1/1—1/15 → 验证 1/16—1/23；
    第 3 折：训练窗口扩展为 1/1—1/23 → 验证 1/24—1/31。
    对每个候选参数在三个验证区块上分别回放，按**三个区间的平均库存校正成本**选参。
    全部区块都位于正式评价期（2 月 1 日）之前，因此不引入评价期信息。
    每个区块的初始库存取自以理论锚点为参数、仅使用该区块之前数据滚动得到的储能轨迹，
    保证验证段本身不含未来信息。
    """
    rows_ = []
    for a0, au in product(grid0, gridh):
        pp = replace(p, alpha0=a0, alpha_update=au)
        costs, detail = [], []
        for i, (s, e) in enumerate(blocks, 1):
            df, _, _ = run(L, P, price, bank, pp, start=s, end=e, initial=warm.soc_start.iloc[s])
            costs.append(df.inventory_adjusted_cost.sum())
            detail.append(dict(fold=i, train_window=f'2025-01-01/{DATES[s - 1].date()}',
                               validation_block=f'{DATES[s].date()}/{DATES[e - 1].date()}',
                               adjusted_cost=df.inventory_adjusted_cost.sum(),
                               total_cost=df.total_cost.sum(), down_kwh=df.down_kwh.sum()))
        rows_.append(dict(alpha0=a0, alpha_update=au, n_folds=len(blocks),
                          mean_adjusted_cost=float(np.mean(costs)),
                          worst_block_cost=float(np.max(costs)),
                          blocks=json.dumps(detail, ensure_ascii=False)))
    dfv = pd.DataFrame(rows_).sort_values('mean_adjusted_cost').reset_index(drop=True)
    csv('rolling_validation.csv', dfv.to_dict('records'))
    best = dfv.iloc[0]
    print('ROLLING-VAL best', best.alpha0, best.alpha_update, best.mean_adjusted_cost, flush=True)
    return dfv, best



def stability_map(L, P, price, bank, p, initial, records):
    """风险参数邻域的成本稳定性：全网格回放 6+12+18，并与同 alpha0 的不更新对照配对比较。

    不更新策略只使用 0:00 基准计划，其成本只依赖 alpha0、与 alpha_update 无关，
    因此只对每个 alpha0 计算一次，即可与全网格严格同源配对。
    """
    none_cost = {}
    for a0 in sorted({c['alpha0'] for c in records}):
        pp = replace(p, alpha0=a0)
        df, _, _ = run(L, P, price, bank, pp, mask=(), initial=initial)
        none_cost[a0] = df.inventory_adjusted_cost.sum()
    rows_ = []
    for c in records:
        pp = replace(p, alpha0=c['alpha0'], alpha_update=c['alpha_update'])
        df, _, _ = run(L, P, price, bank, pp, mask=(6, 12, 18), initial=initial)
        cost = df.inventory_adjusted_cost.sum()
        rows_.append(dict(alpha0=c['alpha0'], alpha_update=c['alpha_update'],
                          full_adjusted_cost=cost, none_adjusted_cost=none_cost[c['alpha0']],
                          saving=none_cost[c['alpha0']] - cost,
                          beats_none=bool(cost < none_cost[c['alpha0']])))
        print('STABILITY', c['alpha0'], c['alpha_update'], round(cost), flush=True)
    csv('stability_map.csv', rows_)
    return rows_


def main():
    OUT.exists() or OUT.mkdir(parents=True)
    clock = time.time()
    L, P, F, price, seed = load()
    p = Params()
    p_min = float(price.min())
    vE = p_min / p.eta

    # ---------- 四级：岭正则超参数按预测精度选择（不使用购电账单） ----------
    slope, weekday, _ = select_ridge(L, seed, p)
    p = replace(p, slope=slope, weekday=weekday)
    bank = Bank(L, P, F, seed, p)

    warm, _, _ = run(L, P, price, bank, p, mask=(), start=0, end=31)
    csv('warmup_january.csv', warm)

    # ---------- 三级：风险分位历史校准（含搜索边界自动延展） ----------
    records, best_sw, calib_log, grid0, gridh = calibrate(L, P, price, bank, p, warm.soc_start.iloc[14])
    csv('calibration.csv', records)
    dump('calibration_grid_extension.json', calib_log)

    # ---------- 三级：滚动 / 扩展窗口时序验证（P1-5） ----------
    _, best_rv = rolling_validation(L, P, price, bank, p, warm, grid0, gridh)
    p_sw = replace(p, alpha0=best_sw['alpha0'], alpha_update=best_sw['alpha_update'])
    p_rv = replace(p, alpha0=float(best_rv.alpha0), alpha_update=float(best_rv.alpha_update))
    p = p_rv                       # 主口径采用多窗口平均成本选出的参数
    warm_sel, _, _ = run(L, P, price, bank, p, mask=(), start=0, end=31)
    csv('warmup_january_selected.csv', warm_sel)
    initial = warm_sel.soc_end.iloc[-1]
    p_theory = replace(p, alpha0=THEORY_ALPHA0, alpha_update=THEORY_ALPHA_UPDATE)
    print('CALIBRATED', asdict(p), 'elapsed', time.time() - clock, flush=True)

    # ---------- 参数邻域稳定性与「是否仍需日内更新」比例（P2-11 / P2-12） ----------
    srows = stability_map(L, P, price, bank, p, initial, records)
    beat = float(np.mean([r['beats_none'] for r in srows]))
    costs = np.array([r['full_adjusted_cost'] for r in srows])
    stable_frac = float(np.mean(costs <= costs.min() * 1.002))
    interior = (0 < grid0.index(p.alpha0) < len(grid0) - 1) and (0 < gridh.index(p.alpha_update) < len(gridh) - 1)

    dump('parameters.json', {
        'selected': asdict(p),
        'theory_base': {'alpha0_base': THEORY_ALPHA0, 'alpha_update_base': THEORY_ALPHA_UPDATE,
                        'name': '经济理论基准分位 / 无储能局部单阶段理论临界分位',
                        'alpha0_source': '1 - p/(m p) = 1 - 1/5',
                        'alpha_update_source': '1 - a p /(m p) = 1 - 1.5/5',
                        'role': '仅作为完整储能滚动优化模型的理论锚点，不视为完整动态问题的理论最优值'},
        'selection': {'primary_rule': 'rolling/expanding multi-window validation on pre-evaluation January blocks',
                      'primary_alpha0': float(best_rv.alpha0), 'primary_alpha_update': float(best_rv.alpha_update),
                      'single_window_rule': '2025-01-15/2025-01-31',
                      'single_window_alpha0': best_sw['alpha0'], 'single_window_alpha_update': best_sw['alpha_update'],
                      'single_window_cost_adjusted': best_sw['adjusted_cost'],
                      'folds': [{'fold': 1, 'train': '2025-01-01/2025-01-07', 'validation': '2025-01-08/2025-01-15'},
                                {'fold': 2, 'train': '2025-01-01/2025-01-15', 'validation': '2025-01-16/2025-01-23'},
                                {'fold': 3, 'train': '2025-01-01/2025-01-23', 'validation': '2025-01-24/2025-01-31'}],
                      'folds_note': '训练窗口随折次扩展，选参依据为三个验证区间的平均库存校正成本；全部区块早于 2025-02-01',
                      'grid_extension_log': calib_log,
                      'alpha0_grid': grid0, 'alpha_update_grid': gridh,
                      'optimum_interior': bool(interior)},
        'ridge_hyperparameters': {'slope': slope, 'weekday': weekday,
                                  'class': '预测子模型工程超参数（四级，无经济含义）',
                                  'selected_by': 'rolling-origin time-series cross-validation of the day-ahead load forecast '
                                                 '(expanding fit window, validation blocks 2025-01-08/15, 01-16/23, 01-24/31); primary criterion RMSE, secondary MAE',
                                  'candidates_slope': list(RIDGE_SLOPE), 'candidates_weekday': list(RIDGE_WEEKDAY),
                                  'folds': [[7, 15], [15, 23], [23, 31]]},

        'terminal_inventory_value_vE': vE, 'price_min': p_min, 'eta': p.eta,
        'selection_period': '2025-01-08/2025-01-31 (pre-evaluation only)',
        'evaluation_period': '2025-02-01/2025-12-31',
        'initial_feb1': initial, 'parameter_candidates': len(records),
        'stability': {'fraction_full_beats_none': beat, 'fraction_within_0.2pct_of_min': stable_frac,
                      'n_candidates': len(srows)},
        'optimization_horizon': 'rolling 24 hours from each issue; only the next 6 hours are executed',
        'time_structure': {'error_window_days': 28, 'error_window_source': '4 complete weekly cycles',
                           'load_decay_days': 14, 'load_decay_source': '2 complete weekly cycles',
                           'intraday_bias_window_intervals': 6, 'intraday_bias_window_minutes': 60,
                           'bias_decay_hours': 6, 'bias_decay_source': 'interval between successive forecast issues'},
        'settlement': {'main': 'net_refund: plan fully priced, increases at 1.5x, cancellations bear only 0.5x, settled once on the net change against the 00:00 baseline',
                       'formula': 'C_t = p_t q_t^0 + 1.5 p_t (q_t-q_t^0)^+ - 0.5 p_t (q_t^0-q_t)^+ + 5 p_t z_t',
                       'price_convention': 'delivery-period TOU tariff p_t for the interval the energy belongs to',
                       'sensitivity_variants': ['legacy_gross: plan fully priced and 0.5x penalty charged on top',
                                                'stepwise: each issue settled separately against the preceding effective plan']},
        'no_unsourced_penalties': {'lambda_delta': 0.0, 'soc_penalty': 0.0, 'reserve': 0.0,
                                   'epsilon': p.epsilon, 'epsilon_role': 'numerical degeneracy tie-break only'},
        'model_boundary': 'deterministic rolling-horizon optimization over the information set available at each issue; NOT a full multi-stage stochastic program',
        'qualification': 'chronological historical simulation; every calibrated parameter uses pre-evaluation data only'})
    print('STABILITY full-superior fraction', beat, 'stable neighbourhood fraction', stable_frac,
          'interior', interior, flush=True)

    # ---------- 日内更新组合消融 ----------
    runs, comparisons = {}, []
    for bits in product([False, True], repeat=3):
        mask = tuple(h for h, b in zip((6, 12, 18), bits) if b)
        label = 'none' if not mask else '+'.join(map(str, mask))
        df, iv, dec = run(L, P, price, bank, p, mask=mask, initial=initial, detail=all(bits))
        runs[label] = df
        comparisons.append(summary(df, strategy=label))
        csv(f'daily_{label}.csv', df)
        if all(bits):
            csv('intervals.csv', iv)
            csv('decisions.csv', dec)
        print('ABLATION', label, df.total_cost.sum(), flush=True)
    csv('strategy_comparison.csv', comparisons)
    full = runs['6+12+18']

    # ---------- 结构变体 ----------
    variants = [('main_net_refund', p, True),
                ('legacy_gross_settlement', replace(p, settlements='legacy_gross'), False),
                ('stepwise_settlement', replace(p, settlements='stepwise'), False),
                ('fixed_target_soc', replace(p, terminal='fixed_target'), False),
                ('day_horizon_only', replace(p, horizon='day'), True),
                ('theory_quantiles', p_theory, False),
                ('single_window_params', p_sw, False),
                ('no_intraday_bias_correction', replace(p, correction=0.), True)]
    vrows = []
    for label, pp, rebuild in variants:
        bb = Bank(L, P, F, seed, pp) if rebuild else bank
        df, _, _ = run(L, P, price, bb, pp, initial=initial)
        csv(f'variant_daily_{label}.csv', df)
        vrows.append(summary(df, variant=label, alpha0=pp.alpha0, alpha_update=pp.alpha_update,
                             settlement=pp.settlements, terminal=pp.terminal))
    csv('variant_comparison.csv', vrows)
    legacy = next(r for r in vrows if r['variant'] == 'legacy_gross_settlement')
    stepwise_row = next(r for r in vrows if r['variant'] == 'stepwise_settlement')

    # ---------- 光伏信息消融 ----------
    stale = Bank(L, P, F, seed, p)
    stale.stale_pv_issues(range(4))
    stale_df, _, _ = run(L, P, price, stale, p, initial=initial)
    csv('daily_stale_pv.csv', stale_df)
    csv('pure_pv_update_value.csv', [dict(stale_pv_cost=stale_df.total_cost.sum(),
                                          fresh_pv_cost=full.total_cost.sum(),
                                          cash_saving=stale_df.total_cost.sum() - full.total_cost.sum(),
                                          inventory_adjusted_saving=stale_df.inventory_adjusted_cost.sum() - full.inventory_adjusted_cost.sum())])

    # ---------- 单因素敏感性 ----------
    sens = []
    variants_sens = (
        [('alpha0', x) for x in grid0] +
        [('alpha_update', x) for x in gridh] +
        [('eta', x) for x in (.85, float(np.sqrt(.9)), .95)] +
        [('capacity', x) for x in (9600., 14400.)] +
        [('power', x) for x in (4000., 6000.)] +
        [('window', x) for x in (14, 56)] +
        [('decay_days', x) for x in (7., 28.)] +
        [('bias_window', x) for x in (3, 12)] +
        [('bias_decay', x) for x in (3., 12.)] +
        [('slope', x) for x in (0.2, 3.2)] +
        [('weekday', x) for x in (0.0125, 0.1)] +
        [('reserve', x) for x in (0., .25, .5)] +
        [('up', x) for x in (1.2, 1.8)] +
        [('emergency', x) for x in (3., 7.)] +
        [('forecast_scale', x) for x in (.9, 1.1)] +
        [('correction', x) for x in (0.,)] +
        [('threshold', x) for x in (20., 100.)] +
        [('interpolate', x) for x in ('hold',)] +
        [('epsilon', x) for x in (1e-8, 1e-6)] +
        [('settlements', x) for x in ('legacy_gross', 'stepwise')] +
        [('terminal', x) for x in ('fixed_target',)] +
        [('horizon', x) for x in ('day',)]
    )
    bank_affecting = {'window', 'decay_days', 'bias_window', 'bias_decay', 'forecast_scale', 'correction',
                      'interpolate', 'slope', 'weekday'}
    for name, value in variants_sens:
        if getattr(p, name) == value:
            continue
        pp = replace(p, **{name: value})
        bb = Bank(L, P, F, seed, pp) if name in bank_affecting else bank
        ini = initial * pp.capacity / p.capacity
        df, _, _ = run(L, P, price, bb, pp, initial=ini)
        label = f'{name}={value}'
        csv(f'sensitivity_daily_{label}.csv', df)
        sens.append(summary(df, parameter=name, value=value, baseline=getattr(p, name)))
        print('SENSITIVITY', label, df.total_cost.sum(), flush=True)
    csv('sensitivity.csv', sens)

    # ---------- bootstrap / 边际信息价值 / 预报精度 ----------
    rng = np.random.default_rng(20260911)
    bootstrap = []
    base = runs['none'].inventory_adjusted_cost.to_numpy()
    for label, df in runs.items():
        if label == 'none':
            continue
        delta = base - df.inventory_adjusted_cost.to_numpy()
        n = len(delta)
        for block in (7, 14, 28):
            starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
            ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
            samples = delta[ix].sum(axis=1)
            bootstrap.append(dict(strategy=label, block_days=block, annual_saving=delta.sum(),
                                  ci_low=np.quantile(samples, .025), ci_high=np.quantile(samples, .975),
                                  fraction_positive=float((samples > 0).mean()),
                                  daily_win_rate=float((delta > 0).mean())))
    csv('bootstrap.csv', bootstrap)

    marginal = []
    for h in (6, 12, 18):
        label = '+'.join(str(x) for x in (6, 12, 18) if x != h)
        marginal.append(dict(issue=h, cash_saving=runs[label].total_cost.sum() - full.total_cost.sum(),
                             inventory_adjusted_saving=runs[label].inventory_adjusted_cost.sum() - full.inventory_adjusted_cost.sum()))
    csv('marginal_release_value.csv', marginal)

    metrics = []
    for k, h in enumerate(ISSUES):
        t = h * 6
        tgt = np.array([bank.actual_horizon(P, d, k) for d in range(31, 365)])
        for lo, hi, label in [(0, 36, 'executed_next_6h'), (0, 144, 'full_24h_horizon')]:
            err = bank.pv[31:, k, lo:hi] - tgt[:, lo:hi]
            metrics.append(dict(issue=h, target=label, samples=int(err.size), mae_kw=float(np.nanmean(abs(err))),
                                rmse_kw=float(np.sqrt(np.nanmean(err ** 2)))))
        tgt6 = np.array([P[d, t:t + 36] for d in range(31, 365)])
        err_k = bank.pv[31:, k, :36] - tgt6
        err_0 = bank.pv[31:, 0, t:t + 36] - tgt6
        metrics.append(dict(issue=h, target='same_targets_vs_midnight', samples=int(err_k.size),
                            mae_kw=float(np.nanmean(abs(err_k))), rmse_kw=float(np.sqrt(np.nanmean(err_k ** 2))),
                            midnight_rmse_same_targets=float(np.sqrt(np.nanmean(err_0 ** 2)))))
    csv('forecast_accuracy.csv', metrics)

    # ---------- 因果性与约束检查 ----------
    day, k, t = 78, 2, 72
    LL, PP, FF = L.copy(), P.copy(), F.copy()
    LL[day, t:] += 12345; LL[day + 1:] += 23456
    PP[day, t:] += 3333; PP[day + 1:] += 4444
    FF[day, k + 1:] += 5555; FF[day + 1:] += 6666
    altered = Bank(LL, PP, FF, seed, p)
    causal_diff = float(np.max(abs(bank.risk(day, k, p.alpha_update) - altered.risk(day, k, p.alpha_update))))
    assert causal_diff == 0
    assert legacy['down_kwh'] < 1e-6, '旧解释下减购应被支配为零'
    check = {'causality_future_perturbation_max_diff': causal_diff,
             'cross_midnight_horizon_uses_published_nextday_forecast': True,
             'balance_max_error': float(full.balance_error.max()),
             'soc_dynamics_max_error': float(full.dynamics_error.max()),
             'soc_min': float(full.soc_min.min()), 'soc_max': float(full.soc_max.max()),
             'cross_day_soc_max_gap': float(abs(full.soc_start.to_numpy()[1:] - full.soc_end.to_numpy()[:-1]).max()),
             'main_down_kwh': float(full.down_kwh.sum()), 'main_up_kwh': float(full.up_kwh.sum()),
             'legacy_down_kwh': float(legacy['down_kwh']),
             'stepwise_cost_minus_net': float(stepwise_row['total_cost'] - full.total_cost.sum()),
             'dates': len(full), 'interval_rows': len(iv),
             'optimization_horizon': p.horizon,
             'alpha0_interior': bool(0 < grid0.index(p.alpha0) < len(grid0) - 1),
             'alpha_update_interior': bool(0 < gridh.index(p.alpha_update) < len(gridh) - 1),
             'fraction_full_beats_none': beat, 'all_assertions_passed': True,
             'elapsed_seconds': time.time() - clock,
             'python': platform.python_version(), 'scipy': scipy.__version__}
    dump('checks.json', check)
    print('COMPLETE', check, flush=True)


if __name__ == '__main__':
    main()
