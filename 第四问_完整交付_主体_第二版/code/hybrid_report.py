"""分层融合算法（HYBRID-4L）的图表与实验报告生成。"""
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
R = ROOT / 'results'
FIG = ROOT / 'figures'
REP = ROOT / 'reports'
C = dict(base='#8C8C8C', hyb='#1F6FB4', acc='#C1533C', ok='#2E8B57', grid='#DDDDDD')


def setup_font():
    cands = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
             '/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc',
             '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
    for f in cands:
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
    plt.close(fig)
    print('figure', name, flush=True)


def fig_convergence():
    d = pd.read_csv(R / 'hybrid_search_convergence.csv')
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {'GA': C['hyb'], 'DE': C['acc'], 'SA': C['ok'], 'NM': '#7B4EA3'}
    for alg, grp in d.groupby('algorithm'):
        g = grp.sort_values('evaluations')
        ax.plot(g.evaluations, g.best.cummin(), lw=1.8, color=colors.get(alg, '#333'),
                marker='o', ms=3, label=alg)
    ax.set_xlabel('目标函数评估次数'); ax.set_ylabel('训练适应度（三块平均库存校正费用 / 元）')
    ax.set_title('L3 参数寻优：算法组合的收敛对比（同一严格因果适应度）')
    ax.legend(frameon=False)
    save(fig, 'HYBRID_优化器收敛')


def fig_terminal_value():
    seg = pd.read_csv(R / 'hybrid_terminal_value_segments.csv')
    cur = pd.read_csv(R / 'hybrid_terminal_value_curve.csv')
    s = json.loads((R / 'hybrid_experiment_summary.json').read_text(encoding='utf8'))
    vE = s['terminal_value']['vE_linear']
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(cur.soc_kwh, cur.future_cost, color=C['hyb'], lw=1.9)
    ax[0].set_xlabel('视野末端库存 E / kWh'); ax[0].set_ylabel('未来 24h 最优成本 C(E) / 元')
    ax[0].set_title('一次 Bellman 回代得到的成本函数（凸）')
    x = np.repeat(seg[['seg_lo', 'seg_hi']].to_numpy().ravel(), 1)
    y = np.repeat(seg.marginal_value.to_numpy(), 2)
    ax[1].step(np.r_[seg.seg_lo.to_numpy(), seg.seg_hi.iloc[-1]],
               np.r_[seg.marginal_value.to_numpy(), seg.marginal_value.iloc[-1]],
               where='post', color=C['hyb'], lw=1.9, label='分段线性终端边际价值 v(E)')
    ax[1].axhline(vE, color=C['base'], ls='--', lw=1.5, label=f'改进版常数 $v_E=p_{{\\min}}/\\eta$ = {vE:.4f}')
    ax[1].set_xlabel('库存 E / kWh'); ax[1].set_ylabel('边际价值 / (元/kWh)')
    ax[1].set_title('L2 终端库存价值：常数 → 分段线性')
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, 'HYBRID_终端价值函数')


def fig_out_of_sample():
    d = pd.read_csv(R / 'hybrid_out_of_sample.csv')
    d = d.sort_values('adjusted_cost')
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    names = [m.split('（')[0] for m in d.method]
    cols = [C['hyb'] if m.startswith('A3') else C['base'] for m in d.method]
    ax[0].barh(names, d.adjusted_cost / 1e6, color=cols)
    for i, (v, g) in enumerate(zip(d.adjusted_cost / 1e6, d.gap_vs_baseline_percent)):
        ax[0].text(v + 0.02, i, f'{v:.3f}M ({g:+.2f}%)', va='center', fontsize=8)
    ax[0].set_xlabel('2—12 月库存校正费用 / 百万元')
    ax[0].set_xlim(0, d.adjusted_cost.max() / 1e6 * 1.22)
    ax[0].set_title('样本外全年费用（越低越好）')
    m = pd.read_csv(R / 'hybrid_monthly.csv')
    ax[1].bar(m.month, m.saving / 1e3, color=[C['ok'] if v > 0 else C['acc'] for v in m.saving])
    ax[1].axhline(0, color='#333', lw=.8)
    ax[1].set_xlabel('月份'); ax[1].set_ylabel('相对基线节省 / 千元')
    ax[1].set_title('逐月节省分解（正为 HYBRID 更省）')
    ax[1].set_xticks(m.month)
    save(fig, 'HYBRID_样本外对比')


def fig_pareto():
    tr = pd.read_csv(R / 'hybrid_pareto_front_train.csv')
    oo = pd.read_csv(R / 'hybrid_pareto_out_of_sample.csv')
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].scatter(tr.train_cost / 1e3, tr.train_cvar90 / 1e3, color=C['hyb'], s=26, zorder=3)
    ax[0].plot(tr.sort_values('train_cost').train_cost / 1e3,
               tr.sort_values('train_cost').train_cvar90 / 1e3, color=C['hyb'], lw=1.2, alpha=.6)
    ax[0].set_xlabel('训练窗口平均费用 / 千元'); ax[0].set_ylabel('训练窗口 CVaR90 / 千元')
    ax[0].set_title('L4 NSGA-II：训练期成本—风险 Pareto 前沿')
    ax[1].scatter(oo.adjusted_cost / 1e6, oo.cvar95 / 1e3, color=C['acc'], s=34, zorder=3)
    for _, r in oo.iterrows():
        ax[1].annotate(f'β={r.cvar_beta:.2f}', (r.adjusted_cost / 1e6, r.cvar95 / 1e3),
                       textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax[1].set_xlabel('样本外全年费用 / 百万元'); ax[1].set_ylabel('样本外日 CVaR95 / 千元')
    ax[1].set_title('前沿解的样本外表现')
    save(fig, 'HYBRID_多目标前沿')


def fig_structure():
    b = pd.read_csv(R / 'hybrid_daily_A0.csv')
    h = pd.read_csv(R / 'hybrid_daily_A3.csv')
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    parts = ['plan_cost', 'up_cost', 'down_net_cost', 'emergency_cost']
    labels = ['计划购电', '增购(1.5×)', '减购净额(−0.5×)', '紧急购电(5×)']
    x = np.arange(len(parts))
    ax[0].bar(x - .2, [b[c].sum() / 1e6 for c in parts], .38, color=C['base'], label='基线 LP-MPC')
    ax[0].bar(x + .2, [h[c].sum() / 1e6 for c in parts], .38, color=C['hyb'], label='HYBRID-4L')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel('百万元'); ax[0].set_title('费用结构分解'); ax[0].legend(frameon=False, fontsize=8)
    ax[1].scatter(b.total_cost / 1e3, h.total_cost / 1e3, s=6, alpha=.45, color=C['hyb'])
    lim = [min(b.total_cost.min(), h.total_cost.min()) / 1e3, max(b.total_cost.max(), h.total_cost.max()) / 1e3]
    ax[1].plot(lim, lim, color='#333', lw=.9, ls='--')
    ax[1].set_xlabel('基线日费用 / 千元'); ax[1].set_ylabel('HYBRID 日费用 / 千元')
    ax[1].set_title('逐日配对比较（点在对角线下方即更省）')
    dd = (b.total_cost - h.total_cost).to_numpy() / 1e3
    ax[2].hist(dd, bins=40, color=C['hyb'], alpha=.85)
    ax[2].axvline(0, color='#333', lw=.9)
    ax[2].set_xlabel('每日节省 / 千元'); ax[2].set_ylabel('天数')
    ax[2].set_title(f'节省分布：{(dd > 0).sum()} / {len(dd)} 天更省')
    save(fig, 'HYBRID_结构分解')


def fig_saa():
    d = pd.read_csv(R / 'hybrid_saa_convergence.csv')
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(d.n_scen, d.fold1_adjusted_cost / 1e3, marker='o', color=C['hyb'], lw=1.8)
    ax.set_xlabel('情景数 S（k-medoids 削减后）')
    ax.set_ylabel('验证块 1 库存校正费用 / 千元')
    ax.set_title('SAA 情景数收敛（S 作为数值离散参数，按收敛而非账单选取）')
    save(fig, 'HYBRID_情景数收敛')


def main():
    setup_font()
    fig_saa(); fig_convergence(); fig_terminal_value(); fig_out_of_sample()
    fig_structure(); fig_pareto()
    print('done', flush=True)


if __name__ == '__main__':
    main()
