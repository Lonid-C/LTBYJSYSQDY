"""生成图表、月度/指定日期汇总表与结果报告。"""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
FIG = ROOT / 'figures'
SELECTED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
LABELS = {'alpha0': '午夜风险分位', 'alpha_update': '日内风险分位', 'eta': '单向效率', 'capacity': '电池容量',
          'power': '最大功率', 'window': '历史误差窗口', 'decay_days': '负荷衰减天数', 'bias_window': '日内偏差窗口',
          'bias_decay': '偏差衰减小时', 'slope': '趋势项岭正则', 'weekday': '星期效应岭正则',
          'reserve': '库存保留系数', 'up': '增购倍数', 'emergency': '紧急购电倍数',
          'forecast_scale': '光伏预报倍率', 'correction': '负荷偏差校正', 'threshold': '调整门槛',
          'interpolate': '插值方式', 'epsilon': '数值惩罚', 'settlements': '结算口径', 'terminal': '终端条件',
          'horizon': '优化视野', 'decay': '指数衰减天数'}
VALUE_LABELS = {'legacy_gross': '旧解释(照付+罚金)', 'stepwise': '逐次调整结算', 'fixed_target': '人工固定日末SOC',
                'day': '仅当日24:00', 'hold': '分段保持'}
VLABEL = {'main_net_refund': '主口径\n净额退款结算', 'legacy_gross_settlement': '旧解释\n原费照付+50%罚金',
          'stepwise_settlement': '逐次调整\n分别计费', 'fixed_target_soc': '人工固定\n日末SOC=50%',
          'day_horizon_only': '视野仅到\n当日24:00', 'theory_quantiles': '经济理论\n基准分位0.80/0.70',
          'single_window_params': '单窗口校准\n参数', 'no_intraday_bias_correction': '关闭负荷\n日内偏差校正'}


def vlabel(v):
    s = str(v)
    return VALUE_LABELS.get(s, s)


def table(frame, digits=3):
    def fmt(v):
        return f'{v:,.{digits}f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(fmt, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def clock(t):
    return f'{t // 6:02d}:{(t % 6) * 10:02d}'


def interval(t):
    return f'{clock(t)}-{clock(t + 1)}'


def setup_font():
    for font in ['/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc']:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})


def main():
    FIG.exists() or FIG.mkdir(parents=True)
    setup_font()
    pars = json.loads((R / 'parameters.json').read_text())
    checks = json.loads((R / 'checks.json').read_text())
    calib_log = json.loads((R / 'calibration_grid_extension.json').read_text())
    p = pars['selected']
    ridge = pd.read_csv(R / 'ridge_selection.csv')
    val = pd.read_csv(R / 'rolling_validation.csv')
    stab = pd.read_csv(R / 'stability_map.csv')
    comp = pd.read_csv(R / 'strategy_comparison.csv')
    variants = pd.read_csv(R / 'variant_comparison.csv')
    daily = pd.read_csv(R / 'daily_6+12+18.csv')
    iv = pd.read_csv(R / 'intervals.csv')
    sens = pd.read_csv(R / 'sensitivity.csv')
    boot = pd.read_csv(R / 'bootstrap.csv')
    marg = pd.read_csv(R / 'marginal_release_value.csv')
    pure = pd.read_csv(R / 'pure_pv_update_value.csv')
    calib = pd.read_csv(R / 'calibration.csv')

    base = comp[comp.strategy == 'none'].iloc[0]
    full = comp[comp.strategy == '6+12+18'].iloc[0]
    variants['cost_change_yuan'] = variants.total_cost - variants[variants.variant == 'main_net_refund'].total_cost.iloc[0]
    variants['cost_change_percent'] = 100 * variants.cost_change_yuan / variants[variants.variant == 'main_net_refund'].total_cost.iloc[0]
    variants['adjusted_change_percent'] = 100 * (variants.adjusted_cost - variants[variants.variant == 'main_net_refund'].adjusted_cost.iloc[0]) / variants[variants.variant == 'main_net_refund'].adjusted_cost.iloc[0]
    variants.to_csv(R / 'variant_with_changes.csv', index=False, encoding='utf-8-sig')
    main = variants[variants.variant == 'main_net_refund'].iloc[0]
    legacy = variants[variants.variant == 'legacy_gross_settlement'].iloc[0]
    stepwise = variants[variants.variant == 'stepwise_settlement'].iloc[0]
    fixedt = variants[variants.variant == 'fixed_target_soc'].iloc[0]
    dayh = variants[variants.variant == 'day_horizon_only'].iloc[0]
    theory = variants[variants.variant == 'theory_quantiles'].iloc[0]
    swin = variants[variants.variant == 'single_window_params'].iloc[0]
    nobias = variants[variants.variant == 'no_intraday_bias_correction'].iloc[0]

    comp['saving_yuan'] = base.total_cost - comp.total_cost
    comp['saving_percent'] = 100 * comp.saving_yuan / base.total_cost
    comp.to_csv(R / 'comparison_with_savings.csv', index=False, encoding='utf-8-sig')
    sens['cost_change_yuan'] = sens.total_cost - full.total_cost
    sens['cost_change_percent'] = 100 * sens.cost_change_yuan / full.total_cost
    sens.to_csv(R / 'sensitivity_with_changes.csv', index=False, encoding='utf-8-sig')

    month = daily.assign(month=daily.date.str[:7]).groupby('month')[
        ['plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost', 'total_cost']].sum().reset_index()
    d0 = pd.read_csv(R / 'daily_none.csv')
    month['no_update_cost'] = d0.assign(month=d0.date.str[:7]).groupby('month').total_cost.sum().values
    month['saving'] = month.no_update_cost - month.total_cost
    month.to_csv(R / 'monthly.csv', index=False, encoding='utf-8-sig')

    t1, t2, events = [], [], []
    for date, g in iv.groupby('date', sort=True):
        g = g.sort_values('t').reset_index(drop=True)
        for start in range(0, 144, 24):
            piece = g.iloc[start:start + 24]
            if date in SELECTED:
                t2.append(dict(date=date, block=f'{clock(start)}-{clock(start + 24)}', charge=piece.charge.sum(),
                               discharge=piece.discharge.sum(), soc_00=g.soc_start.iloc[0], soc_24=g.soc_end.iloc[-1]))
        j, found = 0, False
        while j < 144:
            if g.emergency.iloc[j] <= 1e-7:
                j += 1
                continue
            start = j
            while j < 144 and g.emergency.iloc[j] > 1e-7:
                j += 1
            events.append(dict(date=date, interval=f'{clock(start)}-{clock(j)}', emergency_kwh=g.emergency.iloc[start:j].sum()))
            found = True
        if not found:
            events.append(dict(date=date, interval='无', emergency_kwh=0.))
        if date in SELECTED:
            for t in (60, 72, 84, 96, 108, 120):
                row = g.iloc[t]
                t1.append(dict(date=date, interval=interval(t), plan_kwh=row.plan, adjusted_kwh=row.adjusted,
                               emergency_kwh=row.emergency))
    t1 = pd.DataFrame(t1); t2 = pd.DataFrame(t2); ev = pd.DataFrame(events)
    for name, df in [('selected_table1.csv', t1), ('selected_table2.csv', t2), ('emergency_events.csv', ev),
                     ('selected_table3.csv', ev[ev.date.isin(SELECTED)]),
                     ('selected_daily.csv', daily[daily.date.isin(SELECTED)])]:
        df.to_csv(R / name, index=False, encoding='utf-8-sig')

    def save(fig, name):
        if os.environ.get('Q3_REPORT_ONLY') == '1':
            plt.close(fig)
            return
        fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(FIG / (name + '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def ch(name, v):
        sub = sens[sens.parameter == name].copy()
        num = pd.to_numeric(sub.value, errors='coerce')
        mask = (num - float(v)).abs() < 1e-9
        if not mask.any():
            mask = sub.value.astype(str) == str(v)
        return sub.loc[mask, 'cost_change_percent'].iloc[0]

    def axis_index(seq, target):
        for i, v in enumerate(seq):
            if abs(float(v) - float(target)) < 1e-9:
                return i
        raise ValueError(f'{target} not in {list(seq)}')

    def ends(name):
        sub = sens[sens.parameter == name].copy()
        sub['numeric'] = pd.to_numeric(sub.value, errors='coerce')
        sub = sub.dropna(subset=['numeric']).sort_values('numeric')
        return sub.iloc[0], sub.iloc[-1]

    alo, ahi = ends('alpha0')
    hlo, hhi = ends('alpha_update')
    sens_show = sens.copy()
    sens_show['parameter'] = [LABELS.get(n, n) for n in sens_show.parameter]
    sens_show['value'] = [vlabel(v) for v in sens_show.value]

    # ---------------- 图1：策略费用分解 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xx = np.arange(len(comp))
    axes[0].bar(xx, comp.total_cost / 1e6, color='#325b88')
    axes[0].axhline(base.total_cost / 1e6, color='#c46161', ls='--', lw=1, label='不更新对照')
    axes[0].set_xticks(xx, ['不更新' if s == 'none' else s + '时' for s in comp.strategy], rotation=20)
    axes[0].set_ylabel('2—12 月总购电费 / 百万元'); axes[0].legend(frameon=False)
    keys = [('plan_cost', '午夜计划费', '#325b88'), ('up_cost', '增购费(1.5倍)', '#d2a451'),
            ('down_net_cost', '减购净额(退款-0.5倍)', '#7aa87a'), ('emergency_cost', '紧急购电费(5倍)', '#c46161')]
    bottom_pos = np.zeros(len(variants)); bottom_neg = np.zeros(len(variants))
    vx = np.arange(len(variants))
    for key, lab, col in keys:
        vals = variants[key].to_numpy() / 1e6
        pos = np.maximum(vals, 0); neg = np.minimum(vals, 0)
        axes[1].bar(vx, pos, bottom=bottom_pos, label=lab, color=col); bottom_pos += pos
        axes[1].bar(vx, neg, bottom=bottom_neg, color=col); bottom_neg += neg
    axes[1].axhline(0, color='gray', lw=.7)
    axes[1].set_xticks(vx, [VLABEL.get(v, v) for v in variants.variant], fontsize=8)
    axes[1].set_ylabel('2—12 月费用 / 百万元'); axes[1].legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, '策略费用分解')

    # ---------------- 图2：预报更新价值 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(marg.issue.astype(str), marg.cash_saving / 1e4, color='#325b88')
    axes[0].axhline(0, color='gray', lw=.7)
    axes[0].set_xlabel('条件边际：关闭后再加入的发布时刻 / 时'); axes[0].set_ylabel('条件边际节约 / 万元')
    axes[1].plot(np.arange(11), month.saving / 1e4, 'o-', color='#325b88')
    axes[1].axhline(0, color='gray', lw=.7)
    axes[1].set_xticks(np.arange(11), [str(i) for i in range(2, 13)])
    axes[1].set_xlabel('月份'); axes[1].set_ylabel('全部更新相对不更新节约 / 万元')
    fig.tight_layout(); save(fig, '预报更新价值')

    # ---------------- 图3：单因素敏感性 ----------------
    top = sens.sort_values('cost_change_percent')
    fig, ax = plt.subplots(figsize=(10, max(6, .26 * len(top))))
    ax.barh(np.arange(len(top)), top.cost_change_percent,
            color=['#c46161' if x > 0 else '#325b88' for x in top.cost_change_percent])
    ax.set_yticks(np.arange(len(top)), [f'{LABELS.get(n, n)}={vlabel(v)}' for n, v in zip(top.parameter, top.value)], fontsize=8)
    ax.axvline(0, color='gray', lw=.8)
    ax.set_xlabel('相对主策略现金费用变化 / %')
    fig.tight_layout(); save(fig, '参数敏感性')

    # ---------------- 图4：指定日期调度 ----------------
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for i, date in enumerate(SELECTED):
        g = iv[iv.date == date]; t = np.arange(144) / 6
        axes[i, 0].step(t, g.plan, label='午夜计划', where='post', lw=1)
        axes[i, 0].step(t, g.adjusted, label='调整后最终', where='post', lw=1)
        axes[i, 0].fill_between(t, g.emergency, step='post', alpha=.35, color='#c46161', label='紧急购电')
        axes[i, 0].set_ylabel(date + '\n电量 / kWh'); axes[i, 0].set_xlim(0, 24)
        axes[i, 1].plot(np.arange(145) / 6, np.r_[g.soc_start.iloc[0], g.soc_end], color='#325b88')
        axes[i, 1].axhline(1200, color='gray', ls='--'); axes[i, 1].axhline(10800, color='gray', ls='--')
        axes[i, 1].set_ylim(0, 12000); axes[i, 1].set_xlim(0, 24); axes[i, 1].set_ylabel('储电量 / kWh')
        for h in (6, 12, 18):
            axes[i, 0].axvline(h, color='gray', alpha=.3); axes[i, 1].axvline(h, color='gray', alpha=.3)
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[-1, 0].set_xlabel('时刻 / 时'); axes[-1, 1].set_xlabel('时刻 / 时')
    fig.tight_layout(); save(fig, '指定日期调度')

    # ---------------- 图5：口径与终端条件对比 ----------------
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    vx = np.arange(len(variants))
    colors = ['#325b88' if v == 'main_net_refund' else '#c46161' for v in variants.variant]
    ax.bar(vx, variants.total_cost / 1e6, color=colors)
    for i, cost in enumerate(variants.total_cost):
        ax.text(i, cost / 1e6 + .02, f'{variants.cost_change_percent.iloc[i]:+.3f}%', ha='center', fontsize=8.5)
    ax.set_xticks(vx, [VLABEL.get(v, v) for v in variants.variant], fontsize=8.5)
    ax.set_ylabel('2—12 月总购电费 / 百万元'); ax.set_ylim(0, variants.total_cost.max() / 1e6 * 1.13)
    ax.axhline(main.total_cost / 1e6, color='gray', ls='--', lw=.8)
    fig.tight_layout(); save(fig, '口径与终端条件对比')

    # ---------------- 图6：风险分位校准曲面与稳定区间 ----------------
    piv = calib.pivot_table(index='alpha0', columns='alpha_update', values='adjusted_cost')
    cmin = piv.values.min()
    fig, ax = plt.subplots(figsize=(max(7.4, 1.0 * piv.shape[1]), max(5.0, .62 * piv.shape[0])))
    im = ax.imshow(piv.values / 1e4, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(piv.shape[1]), [f'{c:g}' for c in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]), [f'{i:g}' for i in piv.index])
    ax.set_xlabel(r'日内风险分位 $\alpha_h$'); ax.set_ylabel(r'午夜风险分位 $\alpha_0$')
    for a in range(piv.shape[0]):
        for b in range(piv.shape[1]):
            v = piv.values[a, b]
            ax.text(b, a, f'{v / 1e4:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if v < piv.values.mean() else '#18324d')
            if v <= cmin * 1.002:
                ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, edgecolor='#c46161', lw=2))
    ax.plot(axis_index(piv.columns, p['alpha_update']), axis_index(piv.index, p['alpha0']), marker='*',
            ms=19, color='#c46161', label=f'主口径选参 ({p["alpha0"]:g}, {p["alpha_update"]:g})')
    if 0.70 in [round(float(c), 6) for c in piv.columns] and 0.80 in [round(float(i), 6) for i in piv.index]:
        ax.plot(axis_index(piv.columns, 0.70), axis_index(piv.index, 0.80), marker='o', mfc='none', ms=14,
                mec='white', mew=2, label='经济理论基准分位 (0.80, 0.70)')
    ax.plot([], [], ls='none', marker='s', mfc='none', mec='#c46161', mew=2,
            label='最低成本 0.2% 邻域（稳定区间）')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.colorbar(im, ax=ax, label='校准目标（库存校正费用）/ 万元')
    fig.tight_layout(); save(fig, '风险分位校准曲面')

    # ---------------- 图7：参数邻域稳定性 ----------------
    sp = stab.pivot_table(index='alpha0', columns='alpha_update', values='saving')
    beat = float(mean_bool(stab.beats_none))
    fig, ax = plt.subplots(figsize=(max(7.4, 1.0 * sp.shape[1]), max(4.6, .62 * sp.shape[0])))
    im = ax.imshow(sp.values / 1e4, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(np.arange(sp.shape[1]), [f'{c:g}' for c in sp.columns])
    ax.set_yticks(np.arange(sp.shape[0]), [f'{i:g}' for i in sp.index])
    ax.set_xlabel(r'日内风险分位 $\alpha_h$'); ax.set_ylabel(r'午夜风险分位 $\alpha_0$')
    for a in range(sp.shape[0]):
        for b in range(sp.shape[1]):
            v = sp.values[a, b]
            ax.text(b, a, f'{v / 1e4:.1f}', ha='center', va='center', fontsize=7)
            if v <= 0:
                ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, edgecolor='black', lw=2.2))
    ax.set_title(f'每个参数组合下「6/12/18 全更新 − 不更新」的库存校正节约（万元）；'
                 f'{beat:.1%} 的参数组合下全更新仍然优于不更新', fontsize=10)
    fig.colorbar(im, ax=ax, label='节约 / 万元')
    fig.tight_layout(); save(fig, '参数邻域稳定性')

    # ---------------- 报告 ----------------
    bc = boot[(boot.strategy == '6+12+18') & (boot.block_days == 7)].iloc[0]
    bestpost = comp.loc[comp.total_cost.idxmin()]
    tv = calib_log[-1]
    rb = pd.read_csv(R / 'ridge_selection.csv').iloc[0]
    slope_txt = r'10^{6}' if p['slope'] >= 1e5 else f"{p['slope']:g}"
    fold_txt = '；'.join(f"第 {f['fold']} 折 训练 {f['train']} → 验证 {f['validation']}" for f in pars['selection']['folds'])
    rcv = pd.read_csv(R / 'ridge_cv_folds.csv')
    rcv_sel = rcv[(rcv.slope == p['slope']) & (rcv.weekday == p['weekday'])].sort_values('fold')
    rcv_tab = table(rcv_sel[['fold', 'train_window', 'val_block', 'samples', 'mae_kw', 'rmse_kw']].rename(
        columns={'fold': '折次', 'train_window': '拟合窗口（只用该窗口之前的数据）', 'val_block': '验证区块',
                 'samples': '样本数', 'mae_kw': 'MAE/kW', 'rmse_kw': 'RMSE/kW'}))
    calpiv = calib.pivot_table(index='alpha0', columns='alpha_update', values='adjusted_cost')
    _ia = axis_index(calpiv.index, p['alpha0']); _ib = axis_index(calpiv.columns, p['alpha_update'])
    _nb = calpiv.values[max(0, _ia - 1):_ia + 2, max(0, _ib - 1):_ib + 2].ravel()
    nb_spread = 100 * (_nb.max() / _nb.min() - 1)
    nb_of_best = 100 * (calpiv.values[_ia, _ib] / _nb.min() - 1)

    def chv(name, v):
        return ch(name, v)

    out = rf'''# 第三问量化模型（改进版）：计算结果与敏感性分析

> 本版按两轮修订完成：《第三问模型改进建议：结算口径与参数来源说明》与《第三问改进版：最终修正清单》。核心特征：统一结算口径、以经济理论基准分位为锚点的历史校准、搜索边界自动延展、滚动多窗口时序验证、跨午夜 24 小时滚动优化、参数邻域稳定性检验。

## 1. 主结果与范围

主口径下，2—12 月 {checks['dates']} 天全部启用 6/12/18 时更新的总购电费为 **{full.total_cost:,.2f} 元**，不更新对照为 **{base.total_cost:,.2f} 元**，节约 **{base.total_cost - full.total_cost:,.2f} 元（{100 * (base.total_cost - full.total_cost) / base.total_cost:.3f}%）**；库存校正后节约 **{base.adjusted_cost - full.adjusted_cost:,.2f} 元**。

评价期共同起点为 2 月 1 日库存 {pars['initial_feb1']:,.3f} kWh；主策略年末库存 {full.final_soc:,.3f} kWh，不更新为 {base.final_soc:,.3f} kWh。不更新对照同样使用附件 3 的午夜光伏预报，不能等同于第二问策略。

主策略参数在正式评价期之前的 1 月窗口内选定后整年冻结。需要说明的是：本报告所报告的“最优参数”是指在**已测试候选区域内**取得最低校准成本的组合，而不是对完整动态问题全局最优参数的断言。

## 2. 结算口径

### 2.1 主口径公式

设 $q_t^0$ 为 0:00 基准计划购电量，$q_t$ 为该交付时段最终有效购电量，$z_t$ 为实际运行后的紧急购电量，$p_t$ 为该交付时段分时电价，$(x)^+=\max(x,0)$：

$$C=\sum_t\left[p_tq_t^0+1.5p_t(q_t-q_t^0)^+-0.5p_t(q_t^0-q_t)^++5p_tz_t\right].$$

计划部分按 $p_t$ 正常结算；超出部分按 1.5 倍；被取消的部分只承担 50% 违约成本（等价于 $C_t=p_tq_t+0.5p_t(q_t^0-q_t)$）；实际缺口按 5 倍紧急购电。交易电价一律取该电量所属交付时段，且多次日内调整按最终有效购电量相对 0:00 基准计划的**净额结算一次**。

### 2.2 三种结算解释的对照

| 解释 | 规则 | 2—12 月总费 / 元 | 相对主口径 | 累计减购 / kWh |
|---|---|---:|---:|---:|
| 主口径：净额退款结算 | 取消部分仅承担 50%，净额结算一次 | {main.total_cost:,.2f} | {main.cost_change_percent:+.3f}% | {main.down_kwh:,.3f} |
| 旧解释：原费照付＋50% 罚金 | 计划费全额照付，减购另加罚金 | {legacy.total_cost:,.2f} | {legacy.cost_change_percent:+.3f}% | {legacy.down_kwh:,.3f} |
| 逐次调整结算 | 每次调整相对上一次有效计划分别计费 | {stepwise.total_cost:,.2f} | {stepwise.cost_change_percent:+.3f}% | {stepwise.down_kwh:,.3f} |

旧解释下减购在数学上被严格支配，因此减购量恒为 0（{legacy.down_kwh:,.3f} kWh），题目专门设置的 50% 违约价格失去决策意义；逐次计费把“反复调整”的成本显式计入，因此比净额结算更贵 {stepwise.total_cost - main.total_cost:,.2f} 元（{stepwise.cost_change_percent:+.3f}%），这正是两种计费规则的差别所在。

需要强调：这三种是**不同的合同解释或计费规则**，不是同一合同下的算法优劣。更重要的是，任何一种解释下都改变不了核心结论——不更新策略不产生任何调整，因此在三种解释下费用完全相同（{base.total_cost:,.2f} 元），而全更新在三种解释下分别节约 {100 * (base.total_cost - main.total_cost) / base.total_cost:.3f}%、{100 * (base.total_cost - legacy.total_cost) / base.total_cost:.3f}%、{100 * (base.total_cost - stepwise.total_cost) / base.total_cost:.3f}%。“引入日内预测有价值”这一结论对结算解释不敏感（清单 §11 S1）。

### 2.3 交易电价与净额结算

交易电价解释为**该电量所属交付时段的分时电价** $p_t$，而非下单时刻价格：6:00 决定增购 18:30 的 100 kWh，按 $1.5p_{{18:30}}\times100$ 结算。6:00、12:00、18:00 的修改不重复收费，$100\to120\to105$ 只按 $105-100=5$ 结算。

## 3. 费用构成

| 费用项 | 2—12 月金额 / 元 | 说明 |
|---|---:|---|
| 午夜计划费 | {full.plan_cost:,.2f} | $\sum p_tq_t^0$ |
| 增购费 | {full.up_cost:,.2f} | $\sum 1.5p_t(q_t-q_t^0)^+$，累计增购 {full.up_kwh:,.3f} kWh |
| 减购净额 | {full.down_net_cost:,.2f} | $-\sum 0.5p_t(q_t^0-q_t)^+$，累计减购 {full.down_kwh:,.3f} kWh |
| 紧急购电费 | {full.emergency_cost:,.2f} | $\sum 5p_tz_t$，累计 {full.emergency_kwh:,.3f} kWh |
| **合计** | **{full.total_cost:,.2f}** | 实际现金费用 |

规划 LP 中的极小吞吐惩罚 $\varepsilon$ 不计入实际费用，它不是经济参数。

## 4. 参数体系：四级来源

### 4.1 经济理论基准分位（二级参数）

为避免直接主观指定风险偏好参数，本文构造**忽略储能跨时耦合的局部单阶段边际成本模型**：午夜计划阶段正常购电单位边际成本为 $p_t$、预测不足后紧急购电为 $5p_t$，由非对称损失的临界分位条件得到

$$\alpha_0^{{\mathrm{{base}}}}=1-\frac{{1}}{{5}}=0.80,\qquad \alpha_h^{{\mathrm{{base}}}}=1-\frac{{1.5}}{{5}}=0.70 .$$

这两个数是**经济理论基准分位**（无储能局部单阶段条件下的理论临界分位）。完整第三问还包含储能跨时转移、SOC 状态、分时电价差、0.5/1.5 倍调整价格、后续 6/12/18 时继续更新、预测误差与跨日滚动优化，因此 0.80/0.70 **只作为完整储能滚动优化模型的理论锚点，不视为完整动态问题的理论最优值**；最终运行参数通过严格限制于评价期之前的历史数据校准。直接使用理论基准分位而不校准的费用为 {theory.total_cost:,.2f} 元（相对主口径 {theory.cost_change_percent:+.3f}%）。

### 4.2 历史校准：搜索范围扩大与边界自动延展（三级参数）

原搜索集合为 ${{0.60,\dots,0.90}}$，最优解曾落在下边界 0.60，说明搜索被截断。本版把候选集合扩展为

$$\alpha_0\in\{{0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90\}},\qquad \alpha_h\in\{{0.50,0.55,0.60,0.65,0.70,0.75,0.80\}},$$

并加入**搜索边界自动延展**：若最优解再次落在边界，则继续向下/向上扩展 0.05 并重测，直至最优参数位于已测试区域内部。本版校准共执行 {len(calib_log)} 轮：{"首轮即满足最优解内部性，未触发额外延展" if len(calib_log) == 1 else "因最优解曾落在边界而触发了延展"}；最终区间为 $\alpha_0\in[{tv['grid_alpha0'][0]:g},{tv['grid_alpha0'][1]:g}]$、$\alpha_h\in[{tv['grid_alpha_update'][0]:g},{tv['grid_alpha_update'][1]:g}]$，最优参数 $({p['alpha0']:g},{p['alpha_update']:g})$ **位于区域内部**（`alpha0_interior={checks['alpha0_interior']}`、`alpha_update_interior={checks['alpha_update_interior']}`），因此不存在因最优解落在预设边界而产生的截断偏差。延展机制与每一步的网格记录见 `results/calibration_grid_extension.json`。

### 4.3 滚动 / 扩展窗口时序验证（选参规则）

仅用 17 天窗口选参，样本偏少。本版增加**滚动 / 扩展窗口时序验证**（清单 §5.3 方案B）：{fold_txt}，训练窗口随折次扩展、每个折次都只用该折之前的数据，并按**三个验证区块的平均库存校正成本**选参。主口径采用该规则：$({pars['selection']['primary_alpha0']:g},{pars['selection']['primary_alpha_update']:g})$。作为对照，单窗口（1 月 15—31 日）规则选出 $({pars['selection']['single_window_alpha0']:g},{pars['selection']['single_window_alpha_update']:g})$，其在评价期的结果为 **{swin.total_cost:,.2f} 元**（相对主口径 {swin.cost_change_percent:+.3f}%），说明两种事前选参规则给出非常接近的结果。

校准曲面还显示最优附近的成本差异很小：主口径选参 $({p['alpha0']:g},{p['alpha_update']:g})$ 与其**相邻候选**（八邻域）的校准费用最大相差 {nb_spread:.3f}%，所选点本身相对邻域最低值相差 {nb_of_best:.3f}%。因此不宜把某一组合描述为具有唯一最优性。

因此本文的表述是：**历史校准表明目标函数在风险参数的一个邻域内形成稳定的低成本区域**；本文选取该区域内样本内平均成本最低的组合作为主模型设置，并通过邻域敏感性、滚动多窗口验证与样本外全年回放检验主要结论不依赖单一参数点，而不是宣称某个参数点具有唯一最优性。

### 4.4 参数邻域稳定性（清单 P2-11 / P2-12）

对全部 {len(stab)} 个风险参数组合在评价期逐一回放，结果见 `stability_map.csv`、`figures/参数邻域稳定性.*`。其中 **{checks['fraction_full_beats_none']:.1%} 的参数组合下“6/12/18 全更新”仍然优于不更新**；最低成本 0.2% 邻域占全部组合的 {pars['stability']['fraction_within_0.2pct_of_min']:.1%}。这说明“是否应当引入日内预测”这一核心结论不依赖某一个参数点。

### 4.5 时间窗口参数的结构来源（三级参数）

| 参数 | 取值 | 来源类型 | 结构解释 |
|---|---:|---|---|
| 历史误差窗口 $W$ | 28 天 | 时间结构 | $28=4\times7$，覆盖完整四周；居民负荷具周周期，四周为每个星期位置提供多个样本，同时不至于对近期误差变化反应迟钝 |
| 负荷预报时间衰减 | 14 天 | 时间结构 | $14=2\times7$，两个完整周周期；权重 $w_i=\exp(-(d-i)/14)$ |
| 日内负荷偏差窗口 | 6 段 = 60 分钟 | 信息结构 | 用过去 1 小时真实偏差的中位数估计短期偏差，取中位数以减弱异常点影响 |
| 偏差影响衰减 | 6 小时 | 题目结构 | 题目预测发布时刻为 0:00/6:00/12:00/18:00，相邻间隔恰为 6 小时 |

### 4.6 电池物理参数（题目给定，不参与调参）

$$B=12000\text{{ kWh}},\quad P^{{\max}}=5000\text{{ kW}},\quad SOC_{{\min}}=10\%B,\quad SOC_{{\max}}=90\%B,\quad \eta_c=\eta_d=0.9 .$$

主模型按题目最自然的单向效率口径取值（往返 $0.9^2=81\%$），另把“90% 指往返效率”的解释 $\eta_c=\eta_d=\sqrt{{0.9}}\approx0.9487$ 作为效率口径敏感性。

### 4.7 预测子模型超参数（四级参数，按预测精度选择）

负荷预报中趋势项与星期效应的岭正则系数（$\lambda_{{\mathrm{{slope}}}}$、$\lambda_{{\mathrm{{weekday}}}}$）属于**预测子模型的数值超参数，不具有直接经济含义**。其作用是控制有限历史样本下趋势项与星期效应的估计方差。本版不再沿用手工设定值，而是**按预测精度选择**（清单 §6.4）：在评价期之前的 1 月 8—31 日上做**滚动原点时间序列交叉验证**——对每个候选组合，拟合窗口随预测原点逐日滚动扩展、每次只用该日之前的数据拟合，再预测该日全天负荷；误差按三个验证区块汇总，主判据为整体 RMSE、次级判据为 MAE，取 RMSE 最小者。

候选网格 $\lambda_{{\mathrm{{slope}}}}\in\{{{', '.join(str(v) for v in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])}\}}$（另含趋势项被完全压缩的极限值 $10^6$）、$\lambda_{{\mathrm{{weekday}}}}\in\{{{', '.join(str(v) for v in [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4])}\}}$，按 RMSE 选出 $\lambda_{{\mathrm{{slope}}}}={slope_txt}$、$\lambda_{{\mathrm{{weekday}}}}={p['weekday']:g}$（整体 RMSE {rb.rmse_kw:,.3f} kW、MAE {rb.mae_kw:,.3f} kW）。所选取值的分折结果如下，全部候选与分折明细见 `ridge_selection.csv`、`ridge_cv_folds.csv`。

{rcv_tab}

预测精度准则与经济成本准则的一致性可以交叉核对：`results/forecast_parameter_sensitivity.csv` 中给出这些取值在经济评价期的完整回测费用，以及同一取值下的预测 RMSE，两者在本数据上方向一致（按精度选出的取值同时是低成本取值）。

这样做的意义是把两类参数分开：**预测模型参数按预测准确性选择，风险决策参数按最终经济成本选择**，避免预测模型与决策模型的所有参数全部围绕最终账单过度拟合（清单 §6.4）。经济侧对这两个参数的敏感性见第 9 节，可见按精度选出的取值在经济评价期同样是低成本的。

### 4.8 不再存在的参数

| 被剔除的参数 | 处理 | 理由 |
|---|---|---|
| 调整量惩罚 $\lambda_\Delta$ | 置 0 | 题目已用 0.5/1.5/5 倍价格显式刻画交易代价，再叠加固定手续费属重复计价 |
| 固定 SOC 目标惩罚 | 置 0 | 由滚动 24 小时优化 + 经济终端库存价值取代 |
| 库存保留系数 | 置 0 | 无数据来源；仅作为敏感性检验 |
| LP 吞吐惩罚 $\varepsilon$ | 保留 $10^{{-7}}$ | 纯数值退化惩罚，只在经济成本相同的多个最优解中选择吞吐更小的解 |

对 $\varepsilon$ 的验证（清单 §8）：$10^{{-8}}$ 与 $10^{{-6}}$ 下总现金费用、计划购电量、调整购电量与紧急购电量均与基准一致（见表 9 节）。

## 5. 优化视野与终端条件

优化视野为**每个发布时刻向后滚动 24 小时**，但只执行紧随其后的 6 小时：

| 发布时刻 | 优化视野 | 实际执行 |
|---|---|---|
| 0:00 | 0:00 → 次日 0:00 | 0:00—6:00 |
| 6:00 | 6:00 → 次日 6:00 | 6:00—12:00 |
| 12:00 | 12:00 → 次日 12:00 | 12:00—18:00 |
| 18:00 | 18:00 → 次日 18:00 | 18:00—24:00 |

附件 3 在每个发布时刻都给出其后 24 小时的逐时光伏预报，因此**跨越午夜的视野使用当前决策时点已经发布的次日预测部分**，不需要为次日时段凭空指定预测误差。视野末段（次日尚未交付时段）在本次发布时尚无基准计划，因此按普通交付电价计价；只有属于当日、已由 0:00 基准计划覆盖的时段才适用 1.5 倍 / 0.5 倍的调整价格。终端使用经济库存价值 $v_E=p_{{\min}}/\eta_c={pars['terminal_inventory_value_vE']:.6f}$ 元/kWh，取代人工固定的 $E_{{24}}=0.5B$。

结构敏感性：

| 结构设定 | 总购电费 / 元 | 库存校正费用 / 元 | 相对主口径（校正后） | 年末库存 / kWh |
|---|---:|---:|---:|---:|
| 主口径：24 小时滚动视野 + 库存价值 $v_E$ | {main.total_cost:,.2f} | {main.adjusted_cost:,.2f} | {main.adjusted_change_percent:+.3f}% | {main.final_soc:,.3f} |
| 视野只到当日 24:00 | {dayh.total_cost:,.2f} | {dayh.adjusted_cost:,.2f} | {dayh.adjusted_change_percent:+.3f}% | {dayh.final_soc:,.3f} |
| 人工固定 $E_{{24}}=0.5B$ | {fixedt.total_cost:,.2f} | {fixedt.adjusted_cost:,.2f} | {fixedt.adjusted_change_percent:+.3f}% | {fixedt.final_soc:,.3f} |

把视野延长到次日，使库存校正费用从 {dayh.adjusted_cost:,.2f} 元降到 {main.adjusted_cost:,.2f} 元（{dayh.adjusted_change_percent:+.3f}%），紧急购电量由 {dayh.emergency_kwh:,.0f} kWh 降到 {main.emergency_kwh:,.0f} kWh——只优化到当日 24:00 时，晚峰之后的电池余量在视野末端没有价值，模型会倾向放空；延伸到次日同一时刻后放空的代价被正确计价。

主口径下日末库存常落在接近下界的位置，这是给定电价结构的推论而非求解误差：$p_{{\min}}={pars['price_min']:.4f}$ 元/kWh 与峰段 1.3952 元/kWh 价差很大，往返效率 81% 下每日深循环套利仍然有利，而晚间峰段是当日最后一段高价区间。若实际运行要求保留备用裕度，应把备用容量写成显式的风险约束，而不是靠人工终端 SOC 隐式实现。

## 6. 是否引入日内预报

固定已校准参数，对全部 8 种日内更新组合逐日回放：

{table(comp[['strategy', 'total_cost', 'emergency_cost', 'down_kwh', 'up_kwh', 'daily_cvar95', 'saving_percent']].rename(columns={'strategy': '使用的更新时刻', 'total_cost': '总费/元', 'emergency_cost': '紧急费/元', 'down_kwh': '累计减购/kWh', 'up_kwh': '累计增购/kWh', 'daily_cvar95': '日成本CVaR95/元', 'saving_percent': '相对不更新节约/%'}))}

三次全开时，分别关闭某个时刻再比较，得到条件边际价值：

{table(marg.rename(columns={'issue': '时刻', 'cash_saving': '现金边际节约/元', 'inventory_adjusted_saving': '库存校正边际节约/元'}))}

正数表示该时刻更新在此策略体系中省钱。三个数不能简单相加成全更新总收益——储能状态与多次预报之间存在交互。

**纯光伏预报更新的识别**：固定负荷校正、SOC 观测与三个重算时点，只把日内光伏预报换回午夜版本，费用为 {pure.stale_pv_cost.iloc[0]:,.2f} 元，现金节约 {pure.cash_saving.iloc[0]:,.2f} 元。普通“关闭 18 时更新”同时失去负荷校正与 SOC 再优化，因此不能把它的收益全部解释为 18 时的光伏预报价值。逐次发布的纯信息价值：

{table(pd.read_csv(R / 'pure_release_value.csv').rename(columns={'issue': '发布时刻', 'pv_information_cash_value': '新光伏信息现金价值/元', 'pv_information_inventory_adjusted_value': '库存校正信息价值/元'}))}

当前数据只提供 0、6、12、18 时预报，因此只能回答“是否使用这三个日内更新”，不能声称新增 3、9、15、21 时预报的价值。

## 7. 预测误差与收益的不等价

{table(pd.read_csv(R / 'forecast_accuracy.csv').fillna('—').rename(columns={'issue': '发布时刻', 'target': '目标范围', 'samples': '样本数', 'mae_kw': 'MAE/kW', 'rmse_kw': 'RMSE/kW', 'midnight_rmse_same_targets': '同目标午夜RMSE/kW'}))}

“目标范围”列中 `executed_next_6h` 为发布后实际执行的 6 小时，`full_24h_horizon` 为整个 24 小时视野，`same_targets_vs_midnight` 为在**同一批目标时刻**上与当日 0:00 预报比较（最后一列仅在这类行上有意义，其余行标 —）。“更小 RMSE”与“更低结算费”并不等价：高价时段缺口、误差符号、储能饱和与调整成本共同决定经济效果。

## 8. 参数敏感性（冻结政策参数下的完整重算）

每次只改变一项，其余已校准参数冻结，并重新构造预测（若受影响）、优化、物理执行与实际结算。

{table(sens_show[['parameter', 'value', 'total_cost', 'cost_change_percent', 'emergency_kwh', 'down_kwh']].rename(columns={'parameter': '参数', 'value': '扰动值', 'total_cost': '总费/元', 'cost_change_percent': '费用变化/%', 'emergency_kwh': '紧急电量/kWh', 'down_kwh': '累计减购/kWh'}))}

**怎样解释这些结果**

1. **风险分位在可扰动区间两端都变差。** 午夜分位两端 {alo.numeric:g} 与 {ahi.numeric:g} 的费用变化为 {alo.cost_change_percent:+.3f}% / {ahi.cost_change_percent:+.3f}%；日内分位两端 {hlo.numeric:g}/{hhi.numeric:g} 为 {hlo.cost_change_percent:+.3f}%/{hhi.cost_change_percent:+.3f}%。过低显著增加紧急购电，过高显著增加正常购电与弃电，说明校准值位于合理区间内部。
2. **不必每次都要求高置信上界。** 分位越高，紧急电量越少（见表中紧急电量列），但正常购电量与弃电量越大；风险参数应由费用与尾部损失共同评价，而不是“置信水平越高越好”。
3. **容量比功率更关键。** 容量 ±20% 时费用变化 {ch('capacity', 9600):+.3f}% / {ch('capacity', 14400):+.3f}%，功率 ±20% 时为 {ch('power', 4000):+.3f}% / {ch('power', 6000):+.3f}%。题目没有设备投资成本，不能据此推投资回收期。
4. **效率口径必须写清楚。** 单向效率 0.90→0.85 费用变化 {ch('eta', 0.85):+.3f}%，0.90→0.95 为 {ch('eta', 0.95):+.3f}%；这是两个单向效率同时变化，不能误写成往返效率只变了 5 个百分点。往返效率口径 $\sqrt{{0.9}}$ 的行见表中。
5. **时间窗口稳健。** 误差窗口 14/56 天为 {ch('window', 14):+.3f}%/{ch('window', 56):+.3f}%；负荷衰减 7/28 天为 {ch('decay_days', 7.0):+.3f}%/{ch('decay_days', 28.0):+.3f}%；日内偏差窗口 30/120 分钟为 {ch('bias_window', 3):+.3f}%/{ch('bias_window', 12):+.3f}%。
6. **预测子模型超参数的影响有限且方向一致。** 趋势项岭正则取 0.2/3.2 时费用变化 {ch('slope', 0.2):+.3f}%/{ch('slope', 3.2):+.3f}%；星期效应岭正则取 0.0125/0.1 时为 {ch('weekday', 0.0125):+.3f}%/{ch('weekday', 0.1):+.3f}%。这些系数不具经济含义，其影响通过敏感性说明，不赋予业务解释。
7. **库存保留系数与调整门槛没有证据支持。** 保留系数 0.25/0.50 为 {ch('reserve', 0.25):+.3f}%/{ch('reserve', 0.5):+.3f}%；金额门槛 20/100 元为 {ch('threshold', 20.0):+.5f}%/{ch('threshold', 100.0):+.5f}%。
8. **数值惩罚不改变经济结果。** $\varepsilon=10^{{-8}}/10^{{-6}}$ 的费用变化为 {ch('epsilon', 1e-08):+.5f}%/{ch('epsilon', 1e-06):+.5f}%，且计划、调整与紧急购电量均不变，验证它只是求解器的退化打破项。
9. **负荷信息是日内收益的重要来源。** 关闭日内负荷偏差校正使费用变化 {nobias.cost_change_percent:+.3f}%。

## 9. 统计稳健性与月度表现

对逐日配对的库存校正费用差做 2000 次区块 bootstrap。7 日区块下，全更新节约的 95% 区间为 **[{bc.ci_low:,.2f}, {bc.ci_high:,.2f}] 元**，重采样节约为正的比例 {bc.fraction_positive:.4f}，原始逐日胜率 {bc.daily_win_rate:.3%}。（该比例不是严格的频率学派 p 值。）

{table(boot[boot.strategy == '6+12+18'][['block_days', 'annual_saving', 'ci_low', 'ci_high', 'fraction_positive']].rename(columns={'block_days': '区块/日', 'annual_saving': '累计节约/元', 'ci_low': '95%下界', 'ci_high': '95%上界', 'fraction_positive': '正收益重采样比例'}))}

由于只有一个年份，季节变化并非平稳重复抽样，区间只刻画历史路径的重采样稳定性，不保证下一年收益。

{table(month[['month', 'total_cost', 'no_update_cost', 'saving']].rename(columns={'month': '月份', 'total_cost': '更新总费/元', 'no_update_cost': '不更新总费/元', 'saving': '节约/元'}))}

## 10. 题目指定日期结果

### 表 1：指定 10 分钟段购电（三类口径分列）

{table(t1.rename(columns={'date': '日期', 'interval': '时间段', 'plan_kwh': '午夜计划/kWh', 'adjusted_kwh': '调整后最终/kWh', 'emergency_kwh': '紧急/kWh'}))}

### 全日电量与分项费用

{table(daily[daily.date.isin(SELECTED)][['date', 'plan_kwh', 'final_kwh', 'emergency_kwh', 'plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost', 'total_cost']].rename(columns={'date': '日期', 'plan_kwh': '计划/kWh', 'final_kwh': '调整后/kWh', 'emergency_kwh': '紧急/kWh', 'plan_cost': '计划费/元', 'up_cost': '增购费/元', 'down_net_cost': '减购净额/元', 'emergency_cost': '紧急费/元', 'total_cost': '总费/元'}))}

### 表 2：四小时充放电与日初日末库存

{table(t2.rename(columns={'date': '日期', 'block': '时段', 'charge': '充电/kWh', 'discharge': '放电/kWh', 'soc_00': '0时SOC/kWh', 'soc_24': '24时SOC/kWh'}))}

### 表 3：紧急购电连续时段

{table(ev[ev.date.isin(SELECTED)].rename(columns={'date': '日期', 'interval': '时间段', 'emergency_kwh': '紧急/kWh'}))}

紧急事件只合并同一天内首尾相邻且紧急量大于 $10^{{-7}}$ kWh 的区间，该阈值仅过滤数值噪声。其他日期完整保存在 `emergency_events.csv` 与提交表中。

## 11. 约束检查与结果文件

- {checks['dates']} 个评价日、{checks['interval_rows']} 条 10 分钟记录，源数据日期完整。
- 供需平衡最大误差 {checks['balance_max_error']:.3e} kWh；SOC 动力学最大误差 {checks['soc_dynamics_max_error']:.3e} kWh；跨日 SOC 衔接差 {checks['cross_day_soc_max_gap']:.3e} kWh。
- 实际 SOC 范围 [{checks['soc_min']:.3f}, {checks['soc_max']:.3f}] kWh，所有场景均执行功率、上下界、非负与充放电互斥断言。
- 未来数据扰动后当前预测最大变化 {checks['causality_future_perturbation_max_diff']:.3e}，说明已检查时刻的预测不会读取未来实际值或未发布预报；跨午夜视野使用的是当前发布时点已经给出的次日预测部分。
- 结算口径自检：旧解释下累计减购 {checks['legacy_down_kwh']:.6f} kWh（被支配为零），主口径下累计减购 {checks['main_down_kwh']:,.3f} kWh；逐次结算总费高于净额结算 {checks['stepwise_cost_minus_net']:,.2f} 元。
- 参数层自检：$\alpha_0$ 最优值位于搜索区域内部 = {checks['alpha0_interior']}，$\alpha_h$ 最优值位于搜索区域内部 = {checks['alpha_update_interior']}。
- 使用 Python {checks['python']}、SciPy {checks['scipy']}，主程序耗时 {checks['elapsed_seconds']:.1f} 秒。

提交文件 `result3.xlsx` 保留四个业务工作表。计划表最后一列为午夜计划费；调整表最后一列为最终全日总费（含计划费、增购费、减购净额与紧急费）。调整表里的每段数值是调整后的最终有效购电量，不是带符号增减差额。充放电表保存全期六个四小时块及 0 时、24 时真实 SOC（原模板把第二块的“时刻”误标为 24:00，本版已改为只在最后一个块标注 24:00）。

原模板把第一段写成 00:10—00:20、最后一段写成次日 00:00—00:10，本版明确校正到当天 00:00—24:00，数据时段不平移；原模板另存于 `data/result3_原模板.xlsx` 供检查。

关键数据文件：`calibration.csv`（风险参数全网格校准记录）、`calibration_grid_extension.json`（搜索边界延展过程）、`rolling_validation.csv`（多窗口时序验证）、`stability_map.csv`（参数邻域稳定性与“是否仍需日内更新”比例）、`ridge_selection.csv`（岭正则按预测精度的选择）、`parameters.json`（参数登记与四级来源）、`variant_comparison.csv`（结算口径/终端条件/视野/选参规则对比）、`intervals.csv`（逐段账本）、`daily_*.csv`（各策略日账）、`sensitivity_daily_*.csv`（每次扰动日账）、`bootstrap.csv`（区间）、`checks.json`（验证）。七张图均使用本报告中的同一批结果表。

## 12. 可复现性与模型边界

在安装 numpy、pandas、scipy、openpyxl、matplotlib 的 Python 环境依次运行：

```bash
python code/q3.py
python code/extra_sensitivity.py
python code/pure_release.py
python code/deliver.py
python code/export_xlsx.py
python code/verify.py
python code/compare_versions.py
```

离线 HTML 报告由 `node code/build_report.mjs` 生成（`marked` 与 `katex` 已随包内置在 `qa/`）。只阅读结论时不必运行任何程序。

**模型边界（必须如实陈述）**：本文采用严格因果的 24 小时滚动优化视野，在 0:00、6:00、12:00、18:00 各预测发布时点仅使用当时已经获得的光伏预测、历史负荷信息与实际储能状态，对未来 24 小时进行优化，并仅执行至下一次信息更新时间；跨越午夜时使用当前发布信息中已经包含的次日预测部分。但各次优化仍属于**基于当前信息集的确定性滚动优化**，并未显式构造未来预测修订的联合概率场景树，因此**不应表述为完整的多阶段随机规划模型**；库存价值 $v_E$ 与视野末段计价都只是经济价值的近似。此外，岭正则系数、插值方式与 $\varepsilon$ 属工程设定，本文通过补充敏感性说明其影响，但不宣称已穷尽所有联合敏感性。全部校准与预测只使用决策时点之前已经可获得的数据，但整个数据集来自同一年，因此结论应表述为“历史模拟下的稳健性”，而不是“对未来年份的收益保证”。
'''

    if (R / 'forecast_parameter_sensitivity.csv').exists():
        extra = pd.read_csv(R / 'forecast_parameter_sensitivity.csv')
        extra['cost_change_percent'] = 100 * (extra.total_cost / full.total_cost - 1)
        out += '\n\n## 13. 预报内部系数的补充敏感性\n\n'
        out += '为追溯参数依据，进一步分别检验指数衰减、趋势项岭正则、星期效应岭正则、日内偏差观测窗口、日内偏差衰减与截距数值正则。仍为固定风险策略参数的完整回测，不用 2—12 月结果反向选参。\n\n'
        cols = ['parameter', 'value', 'baseline', 'total_cost', 'cost_change_percent']
        if 'ridge_rmse_kw' in extra.columns:
            cols = cols + ['ridge_rmse_kw']
        out += table(extra[cols].rename(columns={'parameter': '参数', 'value': '扰动值', 'baseline': '基准值',
                                                 'total_cost': '总费/元', 'cost_change_percent': '费用变化/%',
                                                 'ridge_rmse_kw': '该取值下的预测RMSE/kW'}), digits=6)
        out += '\n\ndecay 为历史指数衰减天数；slope/weekday 为岭正则系数（预测子模型超参数，无经济含义，本节同时给出其在预测精度准则下的 RMSE 供对照）；bias_window 为最近观测区间数（每区间 10 分钟）；bias_decay 为日内偏差衰减小时数；epsilon 为截距数值正则，只检验数值稳定性。\n'
        econ = pd.read_csv(R / 'economic_retuning.csv')
        econ['cost_change_percent'] = 100 * (econ.total_cost / full.total_cost - 1)
        out += '\n## 14. 改变价格倍数后重新校准政策会怎样\n\n'
        out += '紧急倍数或增购倍数改变时，经济理论基准分位随之变为 $1-1/m$ 与 $1-a/m$。本版按新的理论锚点重新生成搜索区间，在评价期之前的 1 月窗口重新校准，再在 2—12 月评价，从而把“固定政策下账单变化”和“适应新价格后的决策变化”分开。\n\n'
        out += table(econ[['parameter', 'value', 'theory_alpha0', 'theory_alpha_update', 'alpha0', 'alpha_update',
                           'frozen_cost', 'total_cost', 'retuning_saving']].rename(
            columns={'parameter': '参数', 'value': '情景值', 'theory_alpha0': '理论锚点午夜分位', 'theory_alpha_update': '理论锚点日内分位',
                     'alpha0': '重校准午夜分位', 'alpha_update': '重校准日内分位', 'frozen_cost': '冻结政策费/元',
                     'total_cost': '重校准费/元', 'retuning_saving': '重校准节约/元'}))
        out += '\n\n重校准在 1 月更优不保证全年也更优。若表中重校准节约为负，按原样报告，不能事后改回全年最低者再声称策略预先有效。全部情景校准结果保存在 `economic_retuning_calibration.csv`。\n'

    elastic = []
    for name in ['eta', 'capacity', 'power', 'window', 'decay_days', 'bias_window', 'slope', 'weekday', 'up',
                 'emergency', 'forecast_scale', 'alpha0']:
        subset = sens[sens.parameter == name].copy()
        subset['numeric'] = pd.to_numeric(subset.value, errors='coerce')
        subset = subset.dropna(subset=['numeric']).sort_values('numeric')
        if len(subset) < 2:
            continue
        low, high = subset.iloc[0], subset.iloc[-1]
        if low.numeric > 0 and high.numeric > low.numeric:
            e = np.log(high.total_cost / low.total_cost) / np.log(high.numeric / low.numeric)
            elastic.append(dict(parameter=name, low_value=low.numeric, high_value=high.numeric, arc_elasticity=e))
    elastic = pd.DataFrame(elastic)
    elastic.to_csv(R / 'elasticities.csv', index=False, encoding='utf-8-sig')
    out += '\n\n## 15. 无量纲弧弹性\n\n用两侧扰动的对数斜率比较不同单位的参数，公式为 $[\\ln C(\\theta_+)-\\ln C(\\theta_-)]/[\\ln\\theta_+-\\ln\\theta_-]$。这是所列区间上的弧弹性，不是精确局部导数。\n\n'
    out += table(elastic.rename(columns={'parameter': '参数', 'low_value': '低值', 'high_value': '高值', 'arc_elasticity': '费用弧弹性'}), digits=5)

    dst = ROOT / 'reports/RESULTS_REPORT.md'
    dst.parent.exists() or dst.parent.mkdir(parents=True)
    dst.write_text(out, encoding='utf8')
    (R / 'RESULTS_REPORT.md').write_text(out, encoding='utf8')
    print(comp[['strategy', 'total_cost', 'saving_percent']].to_string(index=False))


def mean_bool(s):
    return np.mean([bool(x) if not isinstance(x, str) else x.lower() == 'true' for x in s])


if __name__ == '__main__':
    main()
