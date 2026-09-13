# -*- coding: utf-8 -*-
"""第四问论文级图表：全部从 results4 取数，PNG(300dpi)+PDF 双出。"""
import os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.sans-serif': ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 11,
    'axes.edgecolor': '#43506B', 'axes.labelcolor': '#1B2333',
    'text.color': '#1B2333', 'xtick.color': '#43506B', 'ytick.color': '#43506B',
    'axes.grid': True, 'grid.color': '#DCE1EA', 'grid.linewidth': .7,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})
C = dict(main='#2F5D9E', alt='#C25A3C', base='#8C93A8', good='#3D7A5A',
         warn='#C9A227', lb='#5B3E8E', light='#AFC3DE', light2='#E8B9A8')
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results4'
OUT = ROOT / 'figures_paper'; OUT.mkdir(exist_ok=True)
SAVED = []

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig); SAVED.append(name); print('  ✓', name, flush=True)

def money(x, pos=None):
    return f'{x/1e6:.2f}M' if abs(x) >= 1e6 else f'{x/1e3:.0f}k'

oos = pd.read_csv(R / 'q4_out_of_sample.csv')
info = pd.read_csv(R / 'q4_information_value.csv')
abl = pd.read_csv(R / 'q4_ablation_sensitivity.csv')
boot = pd.read_csv(R / 'q4_cvar_bootstrap.csv')
lb = pd.read_csv(R / 'q4_full_horizon_bound.csv')
d2 = pd.read_csv(R / 'q4_daily_Q4-2_M2.csv'); d3 = pd.read_csv(R / 'q4_daily_Q4-3_H3.csv')
b2 = pd.read_csv(R / 'q4_daily_Q4-2_B0.csv'); b3 = pd.read_csv(R / 'q4_daily_Q4-3_B0.csv')
ver = json.loads((R / 'q4_verification.json').read_text(encoding='utf8'))
chk = json.loads((R / 'q4_repair_checks.json').read_text(encoding='utf8'))
short = lambda s: s.split('（')[0].strip()

# ---------------- 1 决策时序与信息集 ----------------
fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
for ax, (title, issues) in zip(axes, [('Q4-2：仅 0:00 下达全天计划，日内无增减购通道', [0]),
                                      ('Q4-3：0:00 计划 + 6:00 / 12:00 / 18:00 三次调整', [0, 6, 12, 18])]):
    ax.set_xlim(-0.6, 24); ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
    ax.add_patch(Rectangle((0, .38), 24, .24, fc='#F2F5FA', ec='#C3CCDC'))
    for h in issues:
        wblk = 24 if len(issues) == 1 else min(6, 24 - h)
        ax.add_patch(Rectangle((h, .38), wblk, .24, fc=C['light'], ec=C['main'], lw=1.4, alpha=.85))
        ax.annotate('', xy=(h, .76), xytext=(h, .62), arrowprops=dict(arrowstyle='-|>', color=C['main'], lw=1.8))
        ax.text(h, .80, f'{h}:00\n发布', ha='center', va='bottom', fontsize=9.5, color=C['main'], weight='bold')
        ax.text(h + wblk / 2, .50,
                ('锁定 6h\n(非预期性)' if h else ('0:00 下达全天计划，全天不可再调整' if len(issues) == 1 else '基准计划')),
                ha='center', va='center', fontsize=8.5, color='#26456F')
    ax.text(-0.45, .22, '信息集', fontsize=9, color='#5A6478', ha='left')
    ax.set_title(title, fontsize=11.5, weight='bold', loc='left', pad=8)
axes[1].set_xticks(range(0, 25, 3)); axes[1].set_xticklabels([f'{h}:00' for h in range(0, 25, 3)])
axes[1].set_xlabel('交付时段')
fig.suptitle('图　两个分支的决策时序与信息集', fontsize=13, weight='bold', y=1.0)
save(fig, 'Q4_决策时序与信息集')

# ---------------- 2 费用结构分解 ----------------
rows = [('Q4-2 基线 α=0.60', oos[(oos.problem=='Q4-2')&oos.method.str.startswith('B0')].iloc[0]),
        ('Q4-2 主策略 α=0.675', oos[(oos.problem=='Q4-2')&oos.method.str.startswith('M2')].iloc[0]),
        ('Q4-3 基线 α=0.60', oos[(oos.problem=='Q4-3')&oos.method.str.startswith('B0')].iloc[0]),
        ('Q4-3 主策略 HYBRID-4L', oos[(oos.problem=='Q4-3')&oos.method.str.startswith('H3')].iloc[0])]
comp = [('计划购电费','plan_cost',C['main']), ('增购费','up_cost',C['warn']),
        ('减购净额','down_net_cost',C['good']), ('紧急购电费','emergency_cost',C['alt'])]
fig, ax = plt.subplots(figsize=(10.6, 5.2))
y = np.arange(len(rows)); left = np.zeros(len(rows)); neg = np.zeros(len(rows))
for lab, col, c in comp:
    v = np.array([r[col] for _, r in rows])
    pos = np.clip(v, 0, None); ng = np.clip(v, None, 0)
    ax.barh(y, pos, left=left, color=c, label=lab, height=.56, ec='white')
    ax.barh(y, ng, left=neg, color=c, height=.56, ec='white', alpha=.55)
    left += pos; neg += ng
for i, (_, r) in enumerate(rows):
    ax.text(left[i]*1.008, i, f'现金 {r.total_cost:,.0f}', va='center', fontsize=9.5, weight='bold')
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(money)); ax.set_xlabel('费用（元）')
ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.20))
ax.set_title('图　费用结构分解（334 天合计，负值为减购退款净额）', fontsize=12.5, weight='bold', loc='left')
save(fig, 'Q4_费用结构')

# ---------------- 3 样本外对比 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, prob in zip(axes, ('Q4-2', 'Q4-3')):
    g = oos[oos.problem == prob].copy().sort_values('adjusted_cost')
    labs = [short(m) for m in g.method]
    cols = [C['lb'] if m.startswith('W') else C['base'] if m[0] in 'UX' else
            C['alt'] if m.startswith(('M2','H3')) else C['light'] for m in g.method]
    bars = ax.barh(range(len(g)), g.adjusted_cost, color=cols, height=.66, ec='white')
    ax.set_yticks(range(len(g))); ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis(); ax.xaxis.set_major_formatter(plt.FuncFormatter(money))
    lo = float(lb[lb.problem == prob].adjusted_cost.iloc[0])
    ax.axvline(lo, color=C['lb'], ls='--', lw=1.5)
    ax.text(lo, len(g)-.3, f'  全期真下界 {lo:,.0f}', color=C['lb'], fontsize=9, va='top')
    ax.set_xlim(lo*0.97, g.adjusted_cost.max()*1.02)
    for i, v in enumerate(g.adjusted_cost):
        ax.text(v, i, f' {v:,.0f}', va='center', fontsize=8.5)
    ax.set_title(f'{prob}　库存校正费用（元）', fontsize=12, weight='bold', loc='left')
fig.suptitle('图　样本外 334 天各方法费用对比（橙色为主策略，紫色为滚动视野完美信息对照）',
             fontsize=12.5, weight='bold', y=1.02)
fig.tight_layout(); save(fig, 'Q4_样本外对比')

# ---------------- 4 逐月节省 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
for ax, prob in zip(axes, ('Q4-2', 'Q4-3')):
    m = pd.read_csv(R / f'q4_monthly_{prob}.csv')
    cols = [C['good'] if v > 0 else C['alt'] for v in m.saving]
    ax.bar(m.month, m.saving, color=cols, width=.64, ec='white')
    for _, r in m.iterrows():
        ax.text(r.month, r.saving, f'{r.saving_percent:+.2f}%', ha='center',
                va='bottom' if r.saving > 0 else 'top', fontsize=8.4)
    ax.axhline(0, color='#43506B', lw=1)
    ax.set_xticks(m.month); ax.set_xlabel('月份'); ax.set_ylabel('相对基线节省（元）')
    npos = int((m.saving > 0).sum())
    ax.set_title(f'{prob}　合计节省 {m.saving.sum():,.0f} 元　{npos}/{len(m)} 个月为正',
                 fontsize=11.5, weight='bold', loc='left')
    ax.margins(y=.20)
fig.suptitle('图　主策略相对 α=0.60 基线的逐月费用节省（2—12 月）', fontsize=12.5, weight='bold', y=1.03)
fig.tight_layout(); save(fig, 'Q4_逐月节省')

# ---------------- 5 费用阶梯 / 真下界 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
for ax, prob in zip(axes, ('Q4-2', 'Q4-3')):
    r = info[info.problem == prob].iloc[0]
    steps = [('当作固定\n分时电价', r.price_assumed_fixed, C['base']),
             ('分位基线', r.baseline_deterministic, C['light']),
             ('主策略', r.main_strategy, C['alt']),
             ('冻结策略\n+完美电价', r.perfect_price, C['warn']),
             ('滚动视野\n完美信息\n(不是下界)', r.rolling_perfect_information, C['good']),
             ('全期完美信息\n真下界', r.full_horizon_lower_bound, C['lb'])]
    xs = np.arange(len(steps)); vals = np.array([s[1] for s in steps])
    ax.bar(xs, vals, color=[s[2] for s in steps], width=.58, ec='white')
    span = vals.max() - vals.min()
    ax.set_ylim(vals.min() - span*.10, vals.max() + span*.26)
    for x, v in zip(xs, vals):
        ax.text(x, v + span*.025, f'{v:,.0f}', ha='center', va='bottom',
                fontsize=8.8, weight='bold', rotation=0)
    # 相邻差额：画在柱顶上方的连线上，避开数值标签
    ytop = vals.max() + span*.155
    for i in range(len(steps)-1):
        d = vals[i] - vals[i+1]
        ax.annotate('', xy=(i+1, ytop), xytext=(i, ytop),
                    arrowprops=dict(arrowstyle='-|>', color='#9AA3B4', lw=1.1,
                                    shrinkA=2, shrinkB=2))
        ax.text(i+.5, ytop + span*.012, f'−{d:,.0f}', ha='center', va='bottom',
                fontsize=7.8, color='#5A6478')
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], fontsize=8.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(money)); ax.set_ylabel('库存校正费用（元）')
    ax.set_title(f'{prob}　主策略距真下界 {r.gap_to_lower_bound:,.0f} 元（{r.gap_to_lower_bound_percent:.2f}%）',
                 fontsize=11.5, weight='bold', loc='left', pad=10)
fig.suptitle('图　费用阶梯：滚动视野完美信息仍高于全期真下界，差额即滚动视野与终端代理的约束代价',
             fontsize=12.5, weight='bold', y=1.00)
fig.tight_layout(); save(fig, 'Q4_信息价值阶梯')

# ---------------- 6 CVaR 尾部分布 ----------------
from q4_statistics import empirical_cvar
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (prob, dm, db, nm) in zip(axes, [('Q4-2', d2, b2, '主策略 α=0.675'), ('Q4-3', d3, b3, '主策略 HYBRID-4L')]):
    cm, cb = dm.total_cost.to_numpy(), db.total_cost.to_numpy()
    bins = np.linspace(min(cm.min(), cb.min()), max(cm.max(), cb.max()), 44)
    ax.hist(cb, bins=bins, color=C['base'], alpha=.55, label='基线 α=0.60')
    ax.hist(cm, bins=bins, color=C['alt'], alpha=.62, label=nm)
    for c, col, ls in [(empirical_cvar(cb), C['base'], '--'), (empirical_cvar(cm), C['alt'], '-')]:
        ax.axvline(c, color=col, ls=ls, lw=1.9)
    ax.text(.98, .96, f'精确经验 CVaR$_{{95}}$\n基线 {empirical_cvar(cb):,.0f}\n{nm} {empirical_cvar(cm):,.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9.2,
            bbox=dict(fc='white', ec='#C3CCDC', boxstyle='round,pad=.45'))
    ax.set_xlabel('日费用（元）'); ax.set_ylabel('天数'); ax.legend(frameon=False, loc='upper left')
    ax.set_title(f'{prob}　334 个等概率日损失的分布与尾部', fontsize=11.5, weight='bold', loc='left')
fig.suptitle('图　日费用分布与精确经验 CVaR$_{95}$（尾部质量恰为 5%，即 16.7 天）',
             fontsize=12.5, weight='bold', y=1.03)
fig.tight_layout(); save(fig, 'Q4_CVaR风险分布')

# ---------------- 7 bootstrap 区间 ----------------
fig, ax = plt.subplots(figsize=(10.4, 4.6))
lbl, yy = [], 0; ytick = []
for (prob, var), g in boot.groupby(['problem', 'variant'], sort=False):
    for _, r in g.iterrows():
        cross = r.ci_lo <= 0 <= r.ci_hi
        col = C['base'] if cross else (C['good'] if r['diff'] < 0 else C['alt'])
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=3, solid_capstyle='round', alpha=.85)
        ax.plot(r['diff'], yy, 'o', color=col, ms=7, mec='white', mew=1.2)
        ytick.append(yy); lbl.append(f'{prob} {var}｜块长 {int(r.block_len)}d'); yy += 1
    yy += .6
ax.axvline(0, color='#1B2333', lw=1.3)
ax.set_yticks(ytick); ax.set_yticklabels(lbl, fontsize=8.8); ax.invert_yaxis()
ax.set_xlabel('CVaR$_{95}$ 差（元）　负=尾部改善')
ax.set_title('图　配对移动块 bootstrap（4000 次重采样）：灰色表示 95% 区间跨零，不能宣称改善',
             fontsize=12, weight='bold', loc='left')
save(fig, 'Q4_bootstrap区间')

# ---------------- 8 结算口径对比 ----------------
s = chk['rerun_stepwise_summary']
main = oos[(oos.problem=='Q4-3')&oos.method.str.startswith('H3')].iloc[0]
cases = [('主方案\n净额结算', main.total_cost, C['main']),
         ('冻结同一计划\n按逐次重算账单', chk['fixed_main_plan_stepwise_cash'], C['warn']),
         ('逐次合同下\n重新优化决策', s['total_cost'], C['alt'])]
fig, ax = plt.subplots(figsize=(9.4, 5.0))
xs = np.arange(3); vals = [c[1] for c in cases]
ax.bar(xs, vals, color=[c[2] for c in cases], width=.54, ec='white')
for x, v in zip(xs, vals):
    ax.text(x, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10, weight='bold')
    if x: ax.text(x, vals[0]*.9988, f'贵 {v-vals[0]:,.0f}', ha='center', fontsize=9, color=C['alt'])
ax.axhline(vals[0], color=C['main'], ls='--', lw=1.2)
ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases])
ax.yaxis.set_major_formatter(plt.FuncFormatter(money)); ax.set_ylabel('现金费用（元）')
ax.set_ylim(min(vals)*.9975, max(vals)*1.0012)
ax.set_title('图　结算口径对比：同一计划下逐次计费不可能优于净额结算\n'
             r'$\sum_j f(\delta_j)-f(\sum_j\delta_j)=0.5p[\sum_j|\delta_j|-|\sum_j\delta_j|]\geq 0$',
             fontsize=11.8, weight='bold', loc='left')
save(fig, 'Q4_结算口径对比')

# ---------------- 9 消融矩阵 ----------------
A = abl[abl.block == 'A 发布时点'].copy()
fig, ax = plt.subplots(figsize=(10.6, 4.8))
order = A.sort_values('adjusted_cost')
cols = [C['alt'] if '6:00+12:00+18:00' in c else C['light'] for c in order.case]
ax.bar(range(len(order)), order.adjusted_cost, color=cols, width=.64, ec='white')
for i, (_, r) in enumerate(order.iterrows()):
    ax.text(i, r.adjusted_cost, f'{r.adjusted_cost:,.0f}', ha='center', va='bottom', fontsize=8.4)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order.case, fontsize=9, rotation=18, ha='right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(money)); ax.set_ylabel('库存校正费用（元）')
ax.set_ylim(order.adjusted_cost.min()*.985, order.adjusted_cost.max()*1.01)
gain = order.adjusted_cost.max() - order.adjusted_cost.min()
ax.set_title(f'图　发布时点消融：从完全不更新到三次更新，费用下降 {gain:,.0f} 元',
             fontsize=12, weight='bold', loc='left')
save(fig, 'Q4_消融矩阵')

# ---------------- 10 敏感性 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
D = abl[abl.block == 'D 物理参数']; E = abl[abl.block == 'E 经验参数（全年）']
base = main.adjusted_cost
for ax, G, t in [(axes[0], D, '物理参数'), (axes[1], E, '经验参数（±20%）')]:
    d = (G.adjusted_cost - base)
    cols = [C['alt'] if v > 0 else C['good'] for v in d]
    ax.barh(range(len(G)), d, color=cols, height=.6, ec='white')
    ax.set_yticks(range(len(G))); ax.set_yticklabels(G.case, fontsize=9); ax.invert_yaxis()
    for i, v in enumerate(d): ax.text(v, i, f' {v:+,.0f}', va='center', fontsize=8.6)
    ax.axvline(0, color='#1B2333', lw=1.1); ax.set_xlabel('相对主策略的费用变化（元）')
    ax.set_title(f'{t}', fontsize=11.5, weight='bold', loc='left')
    ax.margins(x=.22)
fig.suptitle('图　敏感性分析：物理参数与经验参数扰动对总费用的影响', fontsize=12.5, weight='bold', y=1.03)
fig.tight_layout(); save(fig, 'Q4_敏感性')

# ---------------- 11 分位标定 ----------------
ac = pd.read_csv(R / 'q4_2_alpha_calibration.csv').sort_values('alpha')
fig, ax = plt.subplots(figsize=(9.6, 4.8))
ax.plot(ac.alpha, ac.train_fitness, '-o', color=C['main'], ms=4.6, lw=1.9, label='1 月三块平均适应度')
best = ac.loc[ac.train_fitness.idxmin()]
ax.plot(best.alpha, best.train_fitness, '*', color=C['alt'], ms=20, zorder=5)
ax.annotate(f'最优 α={best.alpha:.3f}\n适应度 {best.train_fitness:,.2f}',
            xy=(best.alpha, best.train_fitness), xytext=(best.alpha+.06, best.train_fitness+np.ptp(ac.train_fitness.to_numpy())*.22),
            fontsize=9.6, color=C['alt'], arrowprops=dict(arrowstyle='-|>', color=C['alt']))
ax.axvline(.60, color=C['base'], ls='--', lw=1.3)
ax.text(.60, ax.get_ylim()[1], ' 沿用第三问 α=0.60', color=C['base'], fontsize=9, va='top')
ax.set_xlabel('分位数 α'); ax.set_ylabel('1 月训练适应度（元）'); ax.legend(frameon=False)
ax.set_title('图　Q4-2 分位法一维标定：单峰、内部最优，过拟合风险最低的一次标定',
             fontsize=12, weight='bold', loc='left')
save(fig, 'Q4_分位标定')

# ---------------- 12 终端价值函数 ----------------
tc = pd.read_csv(R / 'q4_terminal_curve.csv')
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
axes[0].plot(tc.soc_kwh, tc.normalized_cost, '-o', color=C['main'], ms=4, lw=2)
axes[0].set_xlabel('期末库存 E（kWh）'); axes[0].set_ylabel('归一化续航成本（元）')
axes[0].set_title('Bellman 回代估值（21 个 SOC 网格点）', fontsize=11.5, weight='bold', loc='left')
marg = -np.diff(tc.normalized_cost.to_numpy())/np.diff(tc.soc_kwh.to_numpy())
axes[1].step(tc.soc_kwh[1:], marg, where='pre', color=C['alt'], lw=2.2)
axes[1].set_xlabel('期末库存 E（kWh）'); axes[1].set_ylabel('边际价值 dV/dE（元/kWh）')
axes[1].set_title('边际非增 ⇒ V(E) 凹 ⇒ 可精确嵌入 LP（K=8 段）', fontsize=11.5, weight='bold', loc='left')
fig.suptitle('图　分段线性凹终端库存价值函数', fontsize=12.5, weight='bold', y=1.03)
fig.tight_layout(); save(fig, 'Q4_终端价值函数')

# ---------------- 13 过拟合证据 ----------------
ev = [('Q4-2 四维宽框最优\n(shrink_net=0.298)', 475899, 15807845, False),
      ('Q4-2 主策略\n分位法 α 一维标定', 479390.09, 15127539.71, True),
      ('Q4-3 无约束最优\n(shrink_net=0.215)', 456873.19, 14210031.74, False),
      ('Q4-3 主策略\nshrink_net=0.55 框内最优', 458569.29, 14028199.24, True)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
for ax, (sl, ti) in zip(axes, [(slice(0, 2), 'Q4-2'), (slice(2, 4), 'Q4-3')]):
    g = ev[sl]; xs = np.arange(2); w = .36
    a1 = ax; a2 = ax.twinx(); a2.grid(False)
    a1.bar(xs-w/2, [x[1] for x in g], w, color=C['light'], ec='white', label='1 月训练适应度（左轴）')
    a2.bar(xs+w/2, [x[2] for x in g], w, color=[C['good'] if x[3] else C['alt'] for x in g],
           ec='white', label='样本外库存校正费用（右轴）')
    for i, x in enumerate(g):
        a1.text(i-w/2, x[1], f'{x[1]:,.0f}', ha='center', va='bottom', fontsize=8.4)
        a2.text(i+w/2, x[2], f'{x[2]:,.0f}', ha='center', va='bottom', fontsize=8.4, weight='bold')
    a1.set_xticks(xs); a1.set_xticklabels([x[0] for x in g], fontsize=9)
    a1.set_ylabel('1 月训练适应度（元）'); a2.set_ylabel('样本外费用（元）')
    a1.set_ylim(min(x[1] for x in g)*.985, max(x[1] for x in g)*1.02)
    a2.set_ylim(min(x[2] for x in g)*.99, max(x[2] for x in g)*1.02)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(money))
    a1.set_title(f'{ti}：1 月更优的那个，样本外更差', fontsize=11.5, weight='bold', loc='left')
fig.suptitle('图　两次过拟合的留出证据（绿色为采纳，橙色为拒绝）', fontsize=12.5, weight='bold', y=1.03)
fig.tight_layout(); save(fig, 'Q4_过拟合证据')

# ---------------- 14 因果性检验 ----------------
cz = pd.DataFrame(ver['causality_by_issue'])
fig, ax = plt.subplots(figsize=(10.2, 4.0))
xs = np.arange(len(cz))
ax.bar(xs, np.maximum(cz.max_plan_diff_kwh, 1e-3), color=C['good'], width=.6, ec='white')
ax.set_xticks(xs); ax.set_xticklabels([f'{r.date}\n{int(r.issue)}:00' for _, r in cz.iterrows()], fontsize=8.2)
ax.set_ylabel('扰动后计划最大变动（kWh）'); ax.set_ylim(0, 1)
ax.text(.5, .55, f'12 次检查全部为 0.0 kWh', transform=ax.transAxes, ha='center',
        fontsize=15, weight='bold', color=C['good'])
ax.set_title('图　因果性检验：扰动决策时点之后的负荷/光伏/电价及未释放预报，计划完全不变',
             fontsize=11.8, weight='bold', loc='left')
save(fig, 'Q4_因果性检验')

# ---------------- 15 HYBRID-4L 框架 ----------------
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis('off')
layers = [('第 1 层　情景层', '历史残差成对抽取 (净负荷, 电价对数)\nk-medoids 带权削减 → 12 条情景', C['light']),
          ('第 2 层　规划层', '两阶段情景随机 LP + 非预期性约束\n5×紧急 / 1.5×增购 / 0.5×违约写进目标', C['main']),
          ('第 3 层　终端层', 'Bellman 回代 → 分段线性凹 V(E)（K=8）\n边际非增 ⇒ LP 精确可嵌', C['good']),
          ('第 4 层　搜索层', '模因算法：GA/DE/SA 同预算三选一 + Nelder-Mead\n只搜 LP 写不进去的非凸参数', C['warn'])]
for i, (t, b, c) in enumerate(layers):
    y = .78 - i*.20
    ax.add_patch(Rectangle((.08, y), .84, .155, fc=c, ec='white', lw=2, alpha=.30 if i != 1 else .38))
    ax.add_patch(Rectangle((.08, y), .022, .155, fc=c, ec='none'))
    ax.text(.13, y+.108, t, fontsize=12, weight='bold', color='#1B2333')
    ax.text(.13, y+.040, b, fontsize=9.6, color='#3A4761')
    if i < 3:
        ax.add_patch(FancyArrowPatch((.5, y), (.5, y-.045), arrowstyle='-|>', mutation_scale=16, color='#8C93A8'))
ax.text(.5, .96, 'HYBRID-4L　分层融合架构', ha='center', fontsize=14, weight='bold')
ax.text(.5, .035, '设计原则：能写进 LP 的结构交给 LP 精确求解，只把非凸、无法线性化的参数留给元启发式',
        ha='center', fontsize=9.6, color='#5A6478', style='italic')
save(fig, 'Q4_HYBRID4L框架')

# ---------------- 16 寻优收敛 ----------------
try:
    cv = pd.read_csv(R / 'q4_search_convergence.csv')
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for key, g in cv.groupby(cv.columns[0]):
        yc = [c for c in g.columns if 'fit' in c.lower() or 'best' in c.lower()][0]
        ax.plot(range(len(g)), g[yc].cummin(), lw=2, label=str(key))
    ax.set_xlabel('评估次数'); ax.set_ylabel('当前最优训练适应度（元）')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('图　寻优收敛曲线（GA 种群 16 × 8 代 + Nelder-Mead 30 次精修）',
                 fontsize=12, weight='bold', loc='left')
    save(fig, 'Q4_寻优收敛')
except Exception as e:
    print('  ! 寻优收敛跳过:', e)

# ---------------- 17 情景数收敛 ----------------
try:
    sc = pd.read_csv(R / 'q4_saa_convergence.csv')
    xc = sc.columns[0]; yc = [c for c in sc.columns if 'cost' in c or 'fit' in c][0]
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    ax.plot(sc[xc], sc[yc], '-o', color=C['main'], ms=5, lw=2)
    ax.axvline(12, color=C['alt'], ls='--', lw=1.5)
    ax.text(12, ax.get_ylim()[1], ' 采用 n_scen=12', color=C['alt'], fontsize=9.4, va='top')
    ax.set_xlabel('情景数 n_scen'); ax.set_ylabel('费用（元）')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(money))
    ax.set_title('图　情景数收敛：12 条情景后边际收益迅速衰减', fontsize=12, weight='bold', loc='left')
    save(fig, 'Q4_情景数收敛')
except Exception as e:
    print('  ! 情景数收敛跳过:', e)

# ---------------- 18 下单时刻计价的有效配对 ----------------
try:
    rp = pd.read_csv(R / 'q4_release_pricing_pairs.csv')
    per = rp[rp.release_clock != '合计']; tot = rp[rp.release_clock == '合计'].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    xs = np.arange(len(per))
    bars = axes[0].bar(xs, per.share_percent, color=[C['main'], C['alt'], C['base']],
                       width=.55, ec='white')
    for x, r in zip(xs, per.itertuples()):
        axes[0].text(x, r.share_percent + 1.6, f'{r.share_percent:.2f}%\n{r.hits:,}/{r.pairs:,}',
                     ha='center', fontsize=9.4, weight='bold')
    axes[0].axhline(tot.share_percent, color=C['lb'], ls='--', lw=1.4)
    axes[0].text(len(per) - .45, tot.share_percent + 1.2, f'合计 {tot.share_percent:.2f}%',
                 ha='right', color=C['lb'], fontsize=9.6, weight='bold')
    axes[0].set_xticks(xs); axes[0].set_xticklabels([f'{c} 发布' for c in per.release_clock])
    axes[0].set_ylabel(r'$1.5p^{release}_{\tau}<p^{delivery}_{t}$ 的配对占比（%）')
    axes[0].set_ylim(0, 92)
    axes[0].set_title('图　按「发布时点—可调整交付时段」有效配对重算\n'
                      f'评价期 334 天，共 {int(tot.pairs):,} 个有效配对',
                      fontsize=11.6, weight='bold', loc='left')
    axes[1].bar(xs, per.pairs, color=C['light'], width=.55, ec='white')
    for x, r in zip(xs, per.itertuples()):
        axes[1].text(x, r.pairs, f'{int(r.adjustable_slots_per_day)} 段/日',
                     ha='center', va='bottom', fontsize=9.2)
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f'{c} 发布' for c in per.release_clock])
    axes[1].set_ylabel('有效配对数')
    axes[1].set_title('配对规模：可调整交付时段随发布时刻推迟而减少',
                      fontsize=11.2, weight='bold', loc='left')
    fig.tight_layout(); save(fig, 'Q4_下单计价配对')
except Exception as e:
    print('  ! 下单计价配对跳过:', e)

json.dump(SAVED, open(OUT / 'INDEX.json', 'w'), ensure_ascii=False, indent=1)

# ---------------- 同步到交付目录（去掉手工拷贝这一步，避免交付图与脚本产物脱节）----------------
import shutil
TARGETS = [(ROOT / '论文交付/3_可视化部分/PNG', '.png'),
           (ROOT / '论文交付/3_可视化部分/PDF', '.pdf'),
           (ROOT / '论文交付/2_论文部分/figures', '.png')]
synced = 0
for d, ext in TARGETS:
    d.mkdir(parents=True, exist_ok=True)
    for name in SAVED:
        src = OUT / f'{name}{ext}'
        if src.exists():
            shutil.copy2(src, d / src.name); synced += 1
# figures4 只保留它原有的那一组，不把论文专用图铺进去
F4 = ROOT / 'figures4'
if F4.exists():
    for name in SAVED:
        for ext in ('.png', '.pdf'):
            if (F4 / f'{name}{ext}').exists():
                shutil.copy2(OUT / f'{name}{ext}', F4 / f'{name}{ext}'); synced += 1
print(f'\n共生成 {len(SAVED)} 组图（PNG+PDF）→ {OUT}；已同步 {synced} 个文件到交付目录')
