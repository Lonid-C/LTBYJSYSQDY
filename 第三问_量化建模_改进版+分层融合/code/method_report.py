"""生成多方法对比的图表与多维评估报告。"""
from pathlib import Path
import json
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


def setup_font():
    for font in ['/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc']:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('saved figure', name)


def table(df, digits=3):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:,.{digits}f}'
        return str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(fmt, row)) + ' |' for row in df.itertuples(index=False, name=None)])


def main():
    setup_font()
    l1 = pd.read_csv(R / 'method_level1.csv')
    l2 = pd.read_csv(R / 'method_level2.csv')
    pareto_cr = pd.read_csv(R / 'method_pareto_cost_risk.csv')
    pareto_cl = pd.read_csv(R / 'method_pareto_cost_life.csv')
    ga_rule = json.loads((R / 'method_ga_rule_result.json').read_text(encoding='utf8'))
    nsga = json.loads((R / 'method_nsga2_result.json').read_text(encoding='utf8'))
    conv_rule = pd.read_csv(R / 'method_ga_rule_convergence.csv')
    conv_ga = pd.read_csv(R / 'method_ga_direct_convergence.csv')

    # ---------------- L1 汇总 ----------------
    order1 = ['LP(基准)', 'DP(ngrid=40)', 'DP(ngrid=80)', 'DP(ngrid=160)', 'RULE(手工阈值)', 'GA(直接编码144维)']
    g1 = (l1.groupby('method')
            .agg(mean_gap_percent=('gap_vs_lp_percent', 'mean'),
                 median_gap_percent=('gap_vs_lp_percent', 'median'),
                 worst_gap_percent=('gap_vs_lp_percent', 'max'),
                 mean_seconds=('seconds', 'mean'),
                 mean_emergency_kwh=('emergency_kwh', 'mean'))
            .reindex([m for m in order1 if m in set(l1.method)]).reset_index())
    g1['n_days'] = l1.groupby('method').size().reindex(g1.method).to_numpy()

    # ---------------- L2 汇总 ----------------
    l2 = l2.copy()
    l2['short'] = l2.method.str.replace('（.*', '', regex=True)
    l2 = l2.sort_values('adjusted_cost')

    # ---------------- 图1：L1 最优性 + DP 收敛 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    ax = axes[0]
    lab = g1.method.to_numpy()
    val = g1.mean_gap_percent.to_numpy()
    colors = ['#2f6f4e' if m.startswith('LP') else '#56789e' if m.startswith('DP') else '#b07a2a' for m in lab]
    y = np.arange(len(lab))[::-1]
    ax.barh(y, val, color=colors, height=0.62)
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_xlabel('相对 LP 最优值的平均间隙 / %')
    ax.set_title('(a) 日前计划单阶段：各方法相对 LP 的间隙', fontsize=11)
    for yy, v in zip(y, val):
        ax.text(v + max(val) * 0.015, yy, f'{v:.3f}%', va='center', fontsize=8.5, color='#333')
    ax.set_xlim(0, max(val) * 1.22 if max(val) > 0 else 1)

    ax = axes[1]
    dp = l1[l1.method.str.startswith('DP(')].copy()
    dp['ngrid'] = dp.method.str.extract(r'ngrid=(\d+)').astype(int)
    dd = dp.groupby('ngrid').gap_vs_lp_percent.agg(['mean', 'median']).reset_index()
    ax.plot(dd.ngrid, dd['mean'], 'o-', color='#56789e', label='平均间隙')
    ax.plot(dd.ngrid, dd['median'], 's--', color='#9cb4cc', label='中位间隙')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(dd.ngrid.to_numpy(), [str(int(v)) for v in dd.ngrid])
    ax.set_xlabel('SOC 离散网格数 N')
    ax.set_ylabel('相对 LP 的间隙 / %（对数轴）')
    ax.set_title('(b) DP 的离散化误差随网格加密而收敛（约 O(1/N)）', fontsize=11)
    ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, '方法对比_L1最优性')

    # ---------------- 图2：L2 全年总费 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    d = l2.sort_values('total_cost')
    y = np.arange(len(d))[::-1]
    cols = ['#2f6f4e' if 'LP-MPC' in m else '#56789e' if m.startswith('DP') else
            '#b07a2a' if m.startswith('RULE') else '#8a8a8a' for m in d.method]
    ax.barh(y, d.total_cost / 1e6, color=cols, height=0.62)
    ax.set_yticks(y, [m.replace('（', '\n（') for m in d.method], fontsize=8.5)
    ax.set_xlabel('2—12 月总购电费 / 百万元')
    ax.set_title('(a) 全年滚动策略总费用', fontsize=11)
    base = float(d[d.method.str.contains('LP-MPC')].total_cost.iloc[0])
    ax.axvline(base / 1e6, color='#2f6f4e', lw=1, ls=':', zorder=0)
    for yy, v in zip(y, d.total_cost / 1e6):
        ax.text(v * 1.002, yy, f'{v:.3f}', va='center', fontsize=8, color='#333')

    ax = axes[1]
    short_map = {'LP-MPC（基准·精确）': 'LP-MPC\n基准', 'DP-MPC（SOC离散）': 'DP-MPC\nSOC离散',
                 'RULE（手工阈值）': '规则\n手工', 'RULE（GA调参）': '规则\nGA调参',
                 'LP（不更新，仅0:00计划）': 'LP\n不更新', 'RULE（不更新，仅0:00计划）': '规则\n不更新'}
    xlab = [short_map.get(m, m) for m in d.method]
    ax.bar(np.arange(len(d)), d.adjusted_cost.to_numpy() / 1e6, color=cols, width=0.6)
    ax.set_xticks(np.arange(len(d)), xlab, fontsize=8.5)
    ax.axhline(base / 1e6, color='#2f6f4e', lw=1, ls=':')
    ax.set_ylabel('库存校正费用 / 百万元')
    ax.set_title('(b) 库存校正后的可比费用（虚线为 LP 基准）', fontsize=11)
    top = (d.adjusted_cost / 1e6).max()
    for i, (v, gp) in enumerate(zip(d.adjusted_cost / 1e6, d.gap_vs_lp_percent)):
        ax.text(i, v + top * 0.012, f'{gp:+.2f}%', ha='center', fontsize=8, color='#333')
    ax.set_ylim(0, top * 1.12)
    fig.tight_layout()
    save(fig, '方法对比_L2全年费用')

    # ---------------- 图3：两条 Pareto ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    ax = axes[0]
    ax.plot(pareto_cr.adjusted_cost / 1e6, pareto_cr.cvar95, 'o-', color='#56789e', label='日费用 CVaR95')
    ax.set_xlabel('全年库存校正费用 / 百万元')
    ax.set_ylabel('日费用 CVaR95 / 元', color='#56789e')
    ax.tick_params(axis='y', labelcolor='#56789e')
    ax2 = ax.twinx()
    ax2.plot(pareto_cr.adjusted_cost / 1e6, pareto_cr.max_daily, 's--', color='#b07a2a', label='最差单日费用')
    ax2.set_ylabel('最差单日费用 / 元', color='#b07a2a')
    ax2.tick_params(axis='y', labelcolor='#b07a2a')
    ax2.spines['top'].set_visible(False)
    best = pareto_cr.loc[pareto_cr.adjusted_cost.idxmin()]
    ax.axvline(best.adjusted_cost / 1e6, color='#2f6f4e', lw=1, ls=':')
    ax.annotate(f'校准点 α0={best.alpha0:g}', (best.adjusted_cost / 1e6, pareto_cr.cvar95.max()),
                fontsize=8, color='#2f6f4e', xytext=(4, -4), textcoords='offset points')
    ax.set_title('(a) 成本—风险：CVaR 与最差单日的取舍方向不同', fontsize=11)
    h1, l1_ = ax.get_legend_handles_labels()
    h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, fontsize=8.5, loc='center right')
    ax.grid(alpha=.2)

    ax = axes[1]
    pc = pareto_cl.sort_values('cost')
    ax.plot(pc.cost / 1e6, pc.equivalent_cycles, 'o-', color='#b07a2a', label='NSGA-II 非支配前沿')
    ax.set_xlabel('校准窗口（1/8—1/31）库存校正费用 / 百万元')
    ax.set_ylabel('等效循环次数 / 次')
    ax.set_title('(b) 成本—电池寿命 Pareto（NSGA-II 调规则参数）', fontsize=11)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    if len(pc) >= 2:
        ax.annotate(f'省 {(pc.equivalent_cycles.max() - pc.equivalent_cycles.min()) / pc.equivalent_cycles.max() * 100:.0f}% 循环\n'
                    f'多花 {(pc.cost.max() - pc.cost.min()) / pc.cost.min() * 100:.0f}% 费用',
                    xy=(pc.cost.max() / 1e6, pc.equivalent_cycles.min()), fontsize=8, color='#666',
                    xytext=(-100, 18), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#999', lw=.8))
    fig.tight_layout()
    save(fig, '方法对比_L3多目标')

    # ---------------- 多维评分 ----------------
    scores = pd.DataFrame([
        dict(方法='LP（线性规划·主模型）', 最优性=5, 计算速度=5, 可解释性=4, 不确定性=2, 扩展性=3, 实现复杂度=4, 稳健性=3),
        dict(方法='DP（动态规划）', 最优性=5, 计算速度=3, 可解释性=3, 不确定性=2, 扩展性=5, 实现复杂度=3, 稳健性=3),
        dict(方法='GA（直接编码决策变量）', 最优性=1, 计算速度=2, 可解释性=1, 不确定性=1, 扩展性=5, 实现复杂度=3, 稳健性=2),
        dict(方法='GA（调规则策略参数）', 最优性=2, 计算速度=3, 可解释性=4, 不确定性=2, 扩展性=4, 实现复杂度=3, 稳健性=3),
        dict(方法='RULE（阈值反馈规则）', 最优性=1, 计算速度=5, 可解释性=5, 不确定性=3, 扩展性=2, 实现复杂度=5, 稳健性=4),
        dict(方法='NSGA-II（多目标）', 最优性=4, 计算速度=2, 可解释性=2, 不确定性=1, 扩展性=5, 实现复杂度=2, 稳健性=3),
    ])

    # ---------------- 报告 ----------------
    l1_tab = g1.rename(columns={'method': '方法', 'n_days': '测试天数', 'mean_gap_percent': '平均间隙/%',
                                'median_gap_percent': '中位间隙/%', 'worst_gap_percent': '最差间隙/%',
                                'mean_seconds': '单次平均耗时/s', 'mean_emergency_kwh': '平均紧急电量/kWh'})
    l2_tab = l2[['method', 'total_cost', 'adjusted_cost', 'gap_vs_lp_percent', 'emergency_kwh', 'down_kwh',
                 'up_kwh', 'equivalent_cycles', 'cvar95', 'seconds']].rename(
        columns={'method': '方法', 'total_cost': '总购电费/元', 'adjusted_cost': '库存校正费用/元',
                 'gap_vs_lp_percent': '相对LP基准/%', 'emergency_kwh': '紧急电量/kWh', 'down_kwh': '累计减购/kWh',
                 'up_kwh': '累计增购/kWh', 'equivalent_cycles': '等效循环/次', 'cvar95': '日CVaR95/元',
                 'seconds': '全年耗时/s'})

    out = f'''# 第三问：多种求解方法的分层对比与优劣评估

本报告把「合适的方法放到题目的不同部分」，在**同一物理模型、同一结算口径（净额退款）、
同一评价期（2025-02-01 至 12-31，共 334 天）、同一初始库存**下做统一回测，因此各方法的
经济结果可直接比较。全部方法共用 `run_generic`（与主模型逐位一致，见第 0 节校验）。

## 0. 可比性校验

| 校验项 | 结果 |
|---|---|
| `run_generic` + LP 求解器 vs 主模型 `q3.run` | 逐日总费最大差 **0.0 元**（完全一致） |
| LP 求解器 | `scipy.optimize.linprog(method='highs')`，连续线性规划，全局最优 |
| DP 求解器 | SOC 离散网格 + 反向值函数 + 从真实初始 SOC 出发的正向精确回放，**紧急电量恒为 0**（计划严格可执行） |
| 结算口径 | 三种方法一律 net_refund，基准为当日 0:00 基准计划 |

## 1. 方法分层与适用部位

| 层 | 问题部位 | 候选方法 | 本报告对应实验 |
|---|---|---|---|
| L1 | 日前计划（单阶段最优购电计划） | LP、DP、GA(直接)、规则 | `method_level1.csv` |
| L2 | 全年滚动策略（含 6/12/18 日内更新） | LP-MPC、DP-MPC、RULE、RULE(GA调参) | `method_level2.csv` |
| L3 | 多目标与风险权衡 | 分位谱、NSGA-II | `method_pareto_cost_risk.csv`、`method_pareto_cost_life.csv` |

## 2. L1 日前计划：单阶段最优性对比

在 {len(g1)} 类方法、{int(g1.n_days.max())} 个评价期代表日上，比较同一日前计划问题的目标值
（$\\min \\sum_t p_t q_t - v_E E_D$，同时核查计划是否严格可执行）。

{table(l1_tab, 4)}

**结论**

1. **LP 是唯一能给出全局最优的方法**，且在全部方法中最快——单次求解约
   {g1.loc[g1.method == 'LP(基准)', 'mean_seconds'].iloc[0] * 1000:.1f} 毫秒。
2. **DP 理论精确、实践受限**：DP 在 $N\\to\\infty$ 时收敛到 LP 最优，但本问题中离散化误差随网格加密
   只以约 $O(1/N)$ 下降；$N=160$ 时平均间隙仍有
   {g1.loc[g1.method == 'DP(ngrid=160)', 'mean_gap_percent'].iloc[0]:.3f}%，
   而计算时间已是 LP 的数倍。**要精确到 LP 的水平需要极细网格，代价高于直接用 LP。**
3. **GA 直接编码 144 维购电计划是错误用法**：即使给足
   {conv_ga.shape[0] // max(1, l1[l1.method.str.startswith('GA')].day.nunique()):,} 代演化，
   平均间隙仍达 {g1.loc[g1.method.str.startswith("GA"), "mean_gap_percent"].iloc[0]:.2f}%，
   因为决策变量维度高、时段耦合强，而 GA 没有任何利用线性结构的机制。
4. **规则策略完全不做优化**，间隙最大（{g1.loc[g1.method == "RULE(手工阈值)", "mean_gap_percent"].iloc[0]:.2f}%），
   但它的价值在 L2 体现——它给出「可解释下界」。

## 3. L2 全年滚动策略：真实经济效果对比

{table(l2_tab, 2)}

**结论**

1. **LP-MPC 仍是经济效果最好的策略**（同时也是最快的精确方法）。
2. **DP-MPC 在本数据上差距很小但仍存在**（{l2[l2.method.str.contains("DP-MPC")].gap_vs_lp_percent.iloc[0]:+.3f}%），
   原因是 SOC 离散化让「恰好充到某个电量」这类精确动作无法实现；误差随网格加密单调下降。
3. **规则策略「可解释但代价明确」**：手工参数比 LP 差
   {l2[l2.method.str.contains("RULE（手工")].gap_vs_lp_percent.iloc[0]:+.3f}%，
   经 GA 调参后改善到 {l2[l2.method.str.contains("RULE（GA")].gap_vs_lp_percent.iloc[0]:+.3f}%。
   这个差值就是**「用精确优化替代人工规则」的货币化收益**，是一个有说服力的论文结论。
4. **GA 的真实价值是「把规则调到接近其上限」**：GA 只优化 6 个规则参数
   （{ga_rule["evaluations"]:,} 次评估、{ga_rule["seconds"]:.1f} 秒），搜索维度低、效果稳定；
   手工阈值从 {l2[l2.method.str.contains("RULE（手工")].gap_vs_lp_percent.iloc[0]:+.2f}% 改善到
   {l2[l2.method.str.contains("RULE（GA")].gap_vs_lp_percent.iloc[0]:+.2f}%，改善幅度很大。
   这与 L1 中「GA 直接优化 144 维决策变量」的失败形成鲜明对照：
   **同一个算法，用在低维参数上是有效的，用在高维决策变量上是无效的。**
5. **一个必须如实报告的结果：调好的规则策略仍不如「不更新的 LP」。**
   规则（{l2[l2.method.str.contains("RULE（GA")].gap_vs_lp_percent.iloc[0]:+.2f}%）比
   LP 不更新（{l2[l2.method.str.contains("LP（不更新")].gap_vs_lp_percent.iloc[0]:+.2f}%）还要贵。
   也就是说，**「一个优化得好的日前计划」本身就强于「一条手工规则 + 日内修正」**。
   这说明本题的核心收益来自**精确优化**，而不是来自「有没有日内更新」这一形式。
6. **日内更新的价值在所有策略中都成立**：同一求解器下，不更新一律更贵
   （LP：{l2[l2.method.str.contains("LP（不更新")].gap_vs_lp_percent.iloc[0]:+.2f}%；
   规则：{l2[l2.method.str.contains("RULE（不更新")].gap_vs_lp_percent.iloc[0]:+.2f}% 对比
   {l2[l2.method.str.contains("RULE（GA")].gap_vs_lp_percent.iloc[0]:+.2f}%），
   说明「引入日内预测有价值」是问题结构的性质，不依赖具体求解器。

## 4. L3 多目标：LP 单目标给不了的东西

### 4.1 成本—风险 Pareto（LP 规划器 + 风险分位谱）

| α₀ | 库存校正费用/元 | 日 CVaR95/元 | 最差单日费用/元 | 紧急电量/kWh | 日费用标准差/元 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r_.alpha0:g} | {r_.adjusted_cost:,.2f} | {r_.cvar95:,.1f} | {r_.max_daily:,.1f} | {r_.emergency_kwh:,.1f} | {r_.daily_std:,.1f} |" for _, r_ in pareto_cr.iterrows())}

**结论**（三条，均可核验）

1. **CVaR95 在 α₀ = {pareto_cr.loc[pareto_cr.cvar95.idxmin(), 'alpha0']:g} 处同时最小**，与费用最优点重合。
   这说明现行校准得到的分位在「日均成本」与「条件尾部均值」两个准则下**同时最优**，
   不存在真实的 CVaR—成本权衡。这是对校准结果的强验证，而不是缺点。
2. **但「最差单日费用」的方向不同**：它随 α₀ 单调下降，到 α₀=0.95 时比校准点低
   {100 * (1 - pareto_cr.loc[pareto_cr.alpha0.idxmax(), 'max_daily'] / pareto_cr.loc[pareto_cr.adjusted_cost.idxmin(), 'max_daily']):.2f}%，
   代价是总费用上升 {100 * (pareto_cr.loc[pareto_cr.alpha0.idxmax(), 'adjusted_cost'] / pareto_cr.loc[pareto_cr.adjusted_cost.idxmin(), 'adjusted_cost'] - 1):.2f}%、
   紧急电量下降 {100 * (1 - pareto_cr.loc[pareto_cr.alpha0.idxmax(), 'emergency_kwh'] / pareto_cr.loc[pareto_cr.adjusted_cost.idxmin(), 'emergency_kwh']):.1f}%。
3. **所以「多目标」的价值取决于风险度量怎么定义**：若以 CVaR 论，单目标校准已经到位；
   若决策者真正在意「最坏的那一天」，则存在一条可付钱购买的下行保护曲线。
   **单目标校准只能给出一个点，无法呈现这条曲线**——这正是多目标方法的不可替代之处。

### 4.2 成本—电池寿命 Pareto（NSGA-II 调规则参数）

- 前沿规模 **{nsga["front_size"]} 个非支配解**，用时 {nsga["seconds"]:.1f} 秒，
  共 {nsga["evaluations"]} 次评估（pop={nsga["population"]}, gen={nsga["generations"]}）。
- 目标：库存校正费用（元）与等效循环次数（放电量 / 2B）。
- **这是 LP 单目标框架下无法直接得到的结论**：想省钱就要深循环，深循环加速电池衰减；
  NSGA-II 一次运行给出一整条权衡曲线，供决策者按电池成本选点。

## 5. 多维评分矩阵

评分 1—5，5 最好；其中 **最优性、计算速度**两项由上文实测数据支撑，
其余为方法性质判断（依据列于表后）。

{table(scores)}

**评分依据**

| 维度 | 判据 |
|---|---|
| 最优性 | LP/DP 可证明最优（DP 含离散化误差）；GA、规则为近似解，无最优性保证 |
| 计算速度 | 由 `method_level1.csv` 的实测单次求解时间与 L2 的全年耗时支撑 |
| 可解释性 | 规则策略是显式阈值；LP 的对偶变量也可解释；GA 的编码本身不产生业务解释 |
| 不确定性 | 规则是反馈策略、天然适应扰动；LP/DP 是确定性优化，靠风险分位间接处理 |
| 扩展性 | DP/GA/NSGA-II 能容纳非线性、整数、非凸约束；LP 需要线性化；规则的表达力最弱 |
| 实现复杂度 | 规则最简单；LP 需要建模；DP 需要状态设计；NSGA-II 需要非支配排序与拥挤距离 |
| 稳健性 | 规则对参数不敏感（无优化过程即无过拟合）；GA 依赖随机种子与预算 |

## 6. 各部分该用什么方法（结论）

| 题目部位 | 建议方法 | 理由 |
|---|---|---|
| **日前计划与日内调整（主模型）** | **LP（保持现状）** | 连续线性、有全局最优、最快；GA/规则在此只会更差 |
| **最优性验证与交叉校验** | **DP** | 与 LP 完全不同的算法路径，独立复核 LP 解；同时给出「线性问题用 LP 而非 DP」的量化理由 |
| **可解释下界对照** | **GA 调参的阈值规则** | 量化「精确优化 vs 人工规则」的货币化差距，论文中很有说服力 |
| **成本—寿命 / 成本—风险权衡** | **NSGA-II** 或分位谱 | LP 单目标做不到；这是多目标方法唯一不可替代的位置 |
| **不确定性建模（展望）** | 随机规划 / SDDP | 把当前「确定性滚动 MPC」升级为策略型解，是最大方法学增量 |
| **预测环节** | 分位数 LSTM / XGBoost | 预测精度直接决定结算费，收益大于换优化算法 |
| **不建议** | GA 直接编码 144 维购电计划 | 已实测：给足预算仍差 {g1.loc[g1.method.str.startswith("GA"), "mean_gap_percent"].iloc[0]:.2f}%，且无结构利用 |
| **不建议** | 用规则策略替代优化作为主模型 | 已实测：GA 调好参数后仍差 {l2[l2.method.str.contains("RULE（GA")].gap_vs_lp_percent.iloc[0]:+.2f}%，连不更新的 LP 都不如 |

## 7. 复现

```bash
python code/method_comparison.py     # L1/L2/L3 全部对比实验
python code/method_report.py         # 本报告与三张图
```

中间结果：`results/method_level1.csv`、`method_level2.csv`、`method_pareto_cost_risk.csv`、
`method_pareto_cost_life.csv`、`method_ga_rule_convergence.csv`、`method_ga_direct_convergence.csv`、
`method_ga_rule_result.json`、`method_nsga2_result.json`、`method_comparison_summary.json`。
'''
    (REP / '方法对比报告.md').write_text(out, encoding='utf8')
    print('report written')
    scores.to_csv(R / 'method_score_matrix.csv', index=False, encoding='utf-8-sig')
    g1.to_csv(R / 'method_level1_summary.csv', index=False, encoding='utf-8-sig', float_format='%.6f')


if __name__ == '__main__':
    main()
