"""补充敏感性：预报内部系数，以及改变价格倍数后「重新校准政策」与「固定政策」的分离。

原则：
  · 只用决策时点之前可获得的数据，不用 2—12 月结果反向选参；
  · 岭正则系数属预测子模型超参数（四级），不赋予经济含义，其基准值取 q3.select_ridge()
    按预测精度选出的值；
  · 价格倍数改变时，经济理论基准分位随之改变（alpha0^base = 1 - 1/m、alpha_h^base = 1 - a/m），
    因此重新校准的搜索区间围绕新的理论锚点展开。
"""
import json
from dataclasses import replace
import numpy as np
import pandas as pd
import q3


def theory_grid(center, half=0.15, step=0.05, lo=0.10, hi=0.95):
    """围绕经济理论基准分位的有限邻域网格。"""
    vals = np.round(np.arange(center - half, center + half + 1e-9, step), 4)
    return tuple(float(v) for v in vals if lo - 1e-9 <= v <= hi + 1e-9)


def build(L, P, F, seed, p, decay=14., slope=None, weekday=None, bias_window=6, bias_decay=6., epsilon=1e-8):
    """重算负荷预报库：显式暴露回归系数，用于预报内部参数的敏感性。

    slope / weekday 缺省时取 Params 中按预测精度选出的值。
    """
    slope = p.slope if slope is None else slope
    weekday = p.weekday if weekday is None else weekday
    bank = q3.Bank(L, P, F, seed, p)
    for d in range(365):
        base = q3.load_forecast(L, d, seed, p.window, decay, 1, slope, weekday, epsilon)
        for k, h in enumerate(q3.ISSUES):
            t = h * 6
            n1 = 144 - t
            bias = float(np.median(L[d, max(0, t - bias_window):t] - base[0][max(0, t - bias_window):t])) if t else 0.
            hrs = h + q3.HOURS
            decayv = np.exp(-(hrs - h) / bias_decay)
            lf = np.empty(144)
            lf[:n1] = base[0][t:] + p.correction * bias * decayv[:n1]
            lf[n1:] = base[1][:t] + p.correction * bias * decayv[n1:]
            bank.load[d, k] = np.maximum(lf, 0)
    bank.refresh()
    return bank


def main():
    pars = json.loads((q3.OUT / 'parameters.json').read_text())
    p = q3.Params(**pars['selected'])
    L, P, F, price, seed = q3.load()
    bank = q3.Bank(L, P, F, seed, p)
    full = pd.read_csv(q3.OUT / 'daily_6+12+18.csv')
    base_cost = full.total_cost.sum()
    ridge = pd.read_csv(q3.OUT / 'ridge_selection.csv')

    # ---- 预报内部参数：按预测精度看到的候选区间 + 对经济结果的完整回放 ----
    out = []
    defaults = dict(decay=14., slope=p.slope, weekday=p.weekday, bias_window=6, bias_decay=6., epsilon=1e-8)
    variants = [('decay', [7., 28.]),
                ('slope', [0.20, 3.20]),
                ('weekday', [0.0125, 0.10]),
                ('bias_window', [3, 12]), ('bias_decay', [3., 12.]), ('epsilon', [1e-6, 1e-9])]
    for name, vals in variants:
        for value in vals:
            kw = dict(defaults)
            kw[name] = value
            b = build(L, P, F, seed, p, **kw)
            df, _, _ = q3.run(L, P, price, b, p, initial=pars['initial_feb1'])
            q3.csv(f'extra_daily_{name}={value}.csv', df)
            rec = q3.summary(df, parameter=name, value=value, baseline=defaults[name])
            # 预测精度侧的可比指标（同一天内仅在预报层评价，不涉及购电成本）
            rec['ridge_rmse_kw'] = float(ridge[(ridge.slope == (value if name == 'slope' else defaults['slope']))
                                              & (ridge.weekday == (value if name == 'weekday' else defaults['weekday']))]
                                         .rmse_kw.iloc[0]) if name in ('slope', 'weekday') else np.nan
            out.append(rec)
            print('FORECAST-SENS', name, value, df.total_cost.sum(), flush=True)
    q3.csv('forecast_parameter_sensitivity.csv', out)

    # ---- 价格倍数改变：冻结政策 vs 重新校准 ----
    warm = pd.read_csv(q3.OUT / 'warmup_january.csv')
    sens = pd.read_csv(q3.OUT / 'sensitivity.csv')
    retuned, candidates = [], []
    for name, val, m, a in [('emergency', 3., 3., p.up), ('emergency', 7., 7., p.up),
                            ('up', 1.2, p.emergency, 1.2), ('up', 1.8, p.emergency, 1.8)]:
        pp = replace(p, **{name: val})
        g0 = theory_grid(1 - 1 / m)
        gh = theory_grid(1 - a / m)
        best = None
        for a0, au in q3.product(g0, gh):
            cand = replace(pp, alpha0=a0, alpha_update=au)
            df, _, _ = q3.run(L, P, price, bank, cand, start=14, end=31, initial=warm.soc_start.iloc[14])
            s = q3.summary(df, parameter=name, value=val, alpha0=a0, alpha_update=au)
            candidates.append(s)
            if best is None or s['adjusted_cost'] < best['adjusted_cost']:
                best = s
        pp = replace(pp, alpha0=best['alpha0'], alpha_update=best['alpha_update'])
        df, _, _ = q3.run(L, P, price, bank, pp, initial=pars['initial_feb1'])
        q3.csv(f'retuned_daily_{name}={val}.csv', df)
        frozen = sens[(sens.parameter == name) & (pd.to_numeric(sens.value, errors='coerce') == val)].iloc[0]
        retuned.append(dict(parameter=name, value=val, theory_alpha0=1 - 1 / m, theory_alpha_update=1 - a / m,
                            alpha0=pp.alpha0, alpha_update=pp.alpha_update,
                            frozen_cost=float(frozen.total_cost), total_cost=df.total_cost.sum(),
                            adjusted_cost=df.inventory_adjusted_cost.sum(),
                            retuning_saving=float(frozen.total_cost) - df.total_cost.sum()))
        print('RETUNED', name, val, pp.alpha0, pp.alpha_update, flush=True)
    q3.csv('economic_retuning.csv', retuned)
    q3.csv('economic_retuning_calibration.csv', candidates)
    print('extra_sensitivity done, base_cost', base_cost, flush=True)


if __name__ == '__main__':
    main()
