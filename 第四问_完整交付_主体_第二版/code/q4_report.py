"""第四问图表。"""
from __future__ import annotations
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
FIG = ROOT / 'figures4'
C = dict(base='#8C8C8C', hyb='#1F6FB4', acc='#C1533C', ok='#2E8B57', warn='#C9962C', grid='#DDDDDD')


def setup():
    for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/System/Library/Fonts/STHeiti Light.ttc',
              '/System/Library/Fonts/PingFang.ttc',
              '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'figure.dpi': 140,
                         'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': .6,
                         'axes.axisbelow': True})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig); print('figure', name, flush=True)


def fig_price():
    import q4_core as Q4C
    L, P, F, tou, seed, V = Q4C.load4()
    hp = json.loads((R / 'price_hp.json').read_text(encoding='utf8'))
    pb = Q4C.PriceBank(V, tou, hp)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    hrs = np.arange(144) / 6
    for d, col in ((59, C['hyb']), (171, C['acc']), (354, C['ok'])):
        ax[0].plot(hrs, V[d], lw=1.3, color=col, alpha=.9, label=f'{d+1} 日实际')
    ax[0].plot(hrs, tou, lw=2.0, color='#333', ls='--', label='附件 1 固定分时电价')
    ax[0].set_xlabel('时刻 / h'); ax[0].set_ylabel('电价 / (元/kWh)')
    ax[0].set_title('附件 4：波动电价的日内形状'); ax[0].legend(frameon=False, fontsize=8)
    m = (V / tou[None, :]).mean(1)
    ax[1].plot(np.arange(1, 366), m, lw=1.0, color=C['hyb'])
    ax[1].set_xlabel('日序'); ax[1].set_ylabel('日价格乘子 $m_d$')
    ax[1].set_title('价格水平的日间波动（周期性明显）')
    e = np.concatenate([pb.mean[d, 0] - pb.actual_horizon(d, 0) for d in range(31, 364)])
    e = e[np.isfinite(e)]
    e7 = np.concatenate([V[d - 7] - V[d] for d in range(31, 364)])
    e0 = np.concatenate([tou - V[d] for d in range(31, 364)])
    names = ['附件1 分时形状', '上周同日', '因果预报（本方案）']
    mae = [np.abs(x).mean() for x in (e0, e7, e)]
    rmse = [np.sqrt((x ** 2).mean()) for x in (e0, e7, e)]
    xp = np.arange(3)
    ax[2].bar(xp - .19, mae, .36, color=C['base'], label='MAE')
    ax[2].bar(xp + .19, rmse, .36, color=C['hyb'], label='RMSE')
    for i in range(3):
        ax[2].text(xp[i] - .19, mae[i] + .002, f'{mae[i]:.4f}', ha='center', fontsize=7.5)
        ax[2].text(xp[i] + .19, rmse[i] + .002, f'{rmse[i]:.4f}', ha='center', fontsize=7.5)
    ax[2].set_xticks(xp); ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_ylabel('误差 / (元/kWh)'); ax[2].set_ylim(0, max(rmse) * 1.28)
    ax[2].set_title('价格预报精度（越低越好）'); ax[2].legend(frameon=False, fontsize=8)
    save(fig, 'Q4_电价结构与预报')


def fig_alpha():
    d = pd.read_csv(R / 'q4_alpha_calibration.csv')
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        g = d[d.problem == prob].sort_values('alpha')
        ax[i].plot(g.alpha, g.train_fitness / 1e3, marker='o', ms=3.5, lw=1.8, color=C['hyb'])
        a = float(sel[f'{prob}_quantile']['alpha'])
        y = float(g[np.isclose(g.alpha, a)].train_fitness.iloc[0]) / 1e3
        ax[i].scatter([a], [y], s=80, color=C['acc'], zorder=5)
        ax[i].annotate(f'选中 α={a:.3f}', (a, y), textcoords='offset points', xytext=(8, 10),
                       fontsize=8.5, color=C['acc'])
        ax[i].axvline(0.60, color=C['base'], ls=':', lw=1.3)
        ax[i].text(0.605, ax[i].get_ylim()[1], ' 第三问沿用 α=0.60', va='top', fontsize=7.5, color=C['base'])
        ax[i].set_xlabel('风险分位 α'); ax[i].set_ylabel('1 月训练适应度 / 千元')
        ax[i].set_title(f'{prob}：分位法的一维标定（曲线单峰、最优在网格内部）')
    save(fig, 'Q4_分位标定')


def fig_convergence():
    import glob
    parts = [pd.read_csv(f) for f in sorted(glob.glob(str(R / 'q4_search_convergence*.csv')))]
    d = pd.concat(parts, ignore_index=True)
    d['problem'] = d.problem.str.replace('-wide', '', regex=False).str.replace('-structural', '', regex=False)
    s = json.loads((R / 'q4_experiment_summary.json').read_text(encoding='utf8'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for i, prob in enumerate(('Q4-3', 'Q4-2')):
        g = d[d.problem == prob]
        for alg, col in (('GA', C['hyb']), ('NM', '#7B4EA3')):
            gg = g[g.algorithm == alg].sort_values('evaluations')
            if len(gg):
                ax[i].plot(gg.evaluations, gg.best.cummin(), lw=1.8, marker='o', ms=3, color=col, label=alg)
        base = s['baseline_training_fitness']['q4_3' if prob == 'Q4-3' else 'q4_2']
        ax[i].axhline(base, color=C['base'], ls='--', lw=1.4, label='分位法对照基线')
        ax[i].set_xlabel('目标函数评估次数'); ax[i].set_ylabel('训练适应度 / 元')
        ax[i].set_title(f'{prob}：L3 参数寻优收敛'); ax[i].legend(frameon=False, fontsize=8)
    save(fig, 'Q4_寻优收敛')


def fig_oos():
    d = pd.read_csv(R / 'q4_out_of_sample.csv')
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    fig.subplots_adjust(wspace=0.55)
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        g = d[d.problem == prob].sort_values('adjusted_cost')
        names = [m.split('（')[0] for m in g.method]
        def colour(m):
            if m.startswith('M') or (prob == 'Q4-3' and m.startswith('H')):
                return C['hyb']          # 该问题的主策略
            if m.startswith('H'):
                return '#7B4EA3'         # 并列报告的 HYBRID（未胜出）
            if m.startswith('U'):
                return C['warn']
            if m.startswith('W'):
                return C['ok']
            return C['base']
        cols = [colour(m) for m in g.method]
        ax[i].barh(names, g.adjusted_cost / 1e6, color=cols)
        for j, (v, gp) in enumerate(zip(g.adjusted_cost / 1e6, g.gap_vs_baseline_percent)):
            ax[i].text(v * 0.985, j, f'{v:.3f}M ({gp:+.2f}%)', va='center', ha='right',
                       fontsize=8.5, color='white', fontweight='bold')
        ax[i].set_xlim(0, g.adjusted_cost.max() / 1e6 * 1.06)
        ax[i].set_xlabel('2—12 月库存校正费用 / 百万元')
        ax[i].set_title(f'{prob}：历史回放全年费用（越低越好）')
    save(fig, 'Q4_样本外对比')


def fig_monthly():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        m = pd.read_csv(R / f'q4_monthly_{prob}.csv')
        ax[i].bar(m.month, m.saving / 1e3, color=[C['ok'] if v > 0 else C['acc'] for v in m.saving])
        ax[i].axhline(0, color='#333', lw=.8); ax[i].set_xticks(m.month)
        ax[i].set_xlabel('月份'); ax[i].set_ylabel('相对基线节省 / 千元')
        ax[i].set_title(f'{prob}：逐月节省分解')
    save(fig, 'Q4_逐月节省')


def fig_structure():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0))
    main = {'Q4-2': ('M2', '分位法 α 重标定（主策略）'), 'Q4-3': ('H3', 'HYBRID-4L（主策略）')}
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        b = pd.read_csv(R / f'q4_daily_{prob}_B0.csv')
        pref, lab = main[prob]
        h = pd.read_csv(R / f'q4_daily_{prob}_{pref}.csv')
        parts = ['plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost']
        labels = ['计划购电', '增购(1.5×)', '减购净额(−0.5×)', '紧急购电(5×)']
        x = np.arange(len(parts))
        ax[i].bar(x - .2, [b[c].sum() / 1e6 for c in parts], .38, color=C['base'], label='α=0.60 沿用值')
        ax[i].bar(x + .2, [h[c].sum() / 1e6 for c in parts], .38, color=C['hyb'], label=lab)
        ax[i].set_xticks(x); ax[i].set_xticklabels(labels, fontsize=8)
        ax[i].set_ylabel('百万元'); ax[i].set_title(f'{prob}：费用结构分解')
        ax[i].legend(frameon=False, fontsize=8)
    save(fig, 'Q4_费用结构')


def fig_saa():
    d = pd.read_csv(R / 'q4_saa_convergence.csv')
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        g = d[d.problem == prob].sort_values('n_scen')
        ax[i].plot(g.n_scen, g.train_fitness / 1e3, marker='o', lw=1.8, color=C['hyb'], label='三块平均')
        ax[i].plot(g.n_scen, g.worst_block / 1e3, marker='s', ms=4, lw=1.2, ls='--',
                   color=C['acc'], label='最差验证块')
        ax[i].axvline(12, color=C['base'], ls=':', lw=1.4)
        ax[i].text(12.2, ax[i].get_ylim()[1], ' 选定 S=12', va='top', fontsize=8, color=C['base'])
        ax[i].set_xlabel('情景数 S（k-medoids 削减后）'); ax[i].set_ylabel('1 月训练适应度 / 千元')
        ax[i].set_title(f'{prob}：SAA 情景数扫描'); ax[i].legend(frameon=False, fontsize=8)
    save(fig, 'Q4_情景数收敛')


def fig_info():
    d = pd.read_csv(R / 'q4_information_value.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for i, prob in enumerate(('Q4-2', 'Q4-3')):
        r = d[d.problem == prob].iloc[0]
        steps = ['当作固定分时电价', '确定性基线\n（分位法）', '主策略', 'HYBRID\n（情景随机 LP）',
                 '+完美电价', '滚动完美信息\n对照', '全期完美信息\n真下界']
        vals = [r.price_assumed_fixed, r.baseline_deterministic, r.main_strategy,
                r.hybrid_stochastic, r.perfect_price, r.rolling_perfect_information,
                r.full_horizon_lower_bound]
        cols = [C['base'], C['base'], C['hyb'], '#7B4EA3', C['warn'], '#6B8E23', C['ok']]
        ax[i].bar(range(len(vals)), np.array(vals) / 1e6, color=cols)
        for j, v in enumerate(vals):
            ax[i].text(j, v / 1e6 + 0.03, f'{v/1e6:.3f}M', ha='center', fontsize=7.5)
        ax[i].set_xticks(range(len(vals))); ax[i].set_xticklabels(steps, fontsize=7)
        lo = min(vals) / 1e6
        ax[i].set_ylim(lo * 0.97, max(vals) / 1e6 * 1.03)
        ax[i].set_ylabel('2—12 月库存校正费用 / 百万元')
        ax[i].set_title(f'{prob}：固定策略费用对照\n'
                        f'融合节省={r.hybrid_saving_vs_quantile:,.0f} 元　完美价格替换={r.frozen_price_replacement_saving:,.0f} 元　'
                        f'距真下界={r.gap_to_lower_bound:,.0f} 元（{r.gap_to_lower_bound_percent:.1f}%）',
                        fontsize=8.5)
    save(fig, 'Q4_信息价值阶梯')


def fig_ablation():
    d = pd.read_csv(R / 'q4_ablation_sensitivity.csv')
    a = d[d.block == 'A 发布时点'].copy()
    order = ['不更新', '6:00', '12:00', '18:00', '6:00+12:00', '6:00+18:00', '12:00+18:00',
             '6:00+12:00+18:00']
    a['k'] = a.case.apply(lambda c: order.index(c) if c in order else 99)
    a = a.sort_values('k')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
    base = float(a[a.case == '不更新'].adjusted_cost.iloc[0])
    ax[0].bar(range(len(a)), (base - a.adjusted_cost) / 1e3,
              color=[C['hyb'] if c == '6:00+12:00+18:00' else C['base'] for c in a.case])
    ax[0].set_xticks(range(len(a))); ax[0].set_xticklabels(a.case, rotation=30, ha='right', fontsize=8)
    ax[0].set_ylabel('相对"不更新"的节省 / 千元')
    ax[0].set_title('A 发布时点开关：每个决策机会值多少')
    for i, v in enumerate((base - a.adjusted_cost) / 1e3):
        ax[0].text(i, v + 8, f'{v:,.0f}', ha='center', fontsize=7.5)
    b = d[d.block == 'B 信息更新']
    main_adj = float(a[a.case == '6:00+12:00+18:00'].adjusted_cost.iloc[0])
    lab = ['主策略\n（全更新）'] + [c.split('（')[0] for c in b.case]
    val = [main_adj] + list(b.adjusted_cost)
    ax[1].bar(range(len(val)), np.array(val) / 1e6,
              color=[C['hyb']] + [C['base']] * len(b))
    ax[1].set_xticks(range(len(val))); ax[1].set_xticklabels(lab, fontsize=7.5)
    ax[1].set_ylim(min(val) / 1e6 * 0.985, max(val) / 1e6 * 1.005)
    ax[1].set_ylabel('库存校正费用 / 百万元')
    ax[1].set_title('B 信息更新拆分：光伏 / 负荷 / 电价各自的增量')
    save(fig, 'Q4_消融矩阵')


def fig_sensitivity():
    d = pd.read_csv(R / 'q4_ablation_sensitivity.csv')
    main = float(d[(d.block == 'A 发布时点') & (d.case == '6:00+12:00+18:00')].adjusted_cost.iloc[0])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    c = d[d.block == 'C 结算口径']
    lab = ['净额退款\n（主口径）'] + [x.split('：')[0] for x in c.case]
    val = [main] + list(c.adjusted_cost)
    ax[0].bar(range(len(val)), np.array(val) / 1e6, color=[C['hyb']] + [C['warn']] * len(c))
    ax[0].set_xticks(range(len(val))); ax[0].set_xticklabels(lab, fontsize=8)
    ax[0].set_ylabel('库存校正费用 / 百万元'); ax[0].set_ylim(min(val) / 1e6 * 0.96, max(val) / 1e6 * 1.02)
    ax[0].set_title('C 结算口径（不同合同，不是算法改进）')
    dd = d[d.block == 'D 物理参数']
    ax[1].barh(range(len(dd)), (dd.adjusted_cost - main) / 1e3,
               color=[C['ok'] if v < 0 else C['acc'] for v in dd.adjusted_cost - main])
    ax[1].set_yticks(range(len(dd))); ax[1].set_yticklabels(dd.case, fontsize=8)
    ax[1].axvline(0, color='#333', lw=.8)
    ax[1].set_xlabel('相对主策略的费用变化 / 千元'); ax[1].set_title('D 物理参数压力（冻结策略）')
    t = pd.read_csv(R / 'q4_param_sensitivity_train.csv')
    t = t[np.isfinite(t.d_fitness)]
    piv = t.pivot_table(index='param', columns='rel', values='d_fitness')
    piv = piv.reindex(piv.abs().max(axis=1).sort_values().index)
    y = np.arange(len(piv))
    ax[2].barh(y - .18, piv[-0.2] / 1e3, .34, color=C['base'], label='−20%')
    ax[2].barh(y + .18, piv[0.2] / 1e3, .34, color=C['hyb'], label='+20%')
    ax[2].set_yticks(y); ax[2].set_yticklabels(piv.index, fontsize=8)
    ax[2].axvline(0, color='#333', lw=.8)
    ax[2].set_xlabel('1 月训练适应度变化 / 千元'); ax[2].set_title('E 经验参数逐项扰动')
    ax[2].legend(frameon=False, fontsize=8)
    save(fig, 'Q4_敏感性')


def main():
    setup()
    fig_price(); fig_saa(); fig_alpha(); fig_convergence(); fig_oos(); fig_monthly()
    fig_structure(); fig_info(); fig_ablation(); fig_sensitivity()
    print('done', flush=True)


if __name__ == '__main__':
    main()
