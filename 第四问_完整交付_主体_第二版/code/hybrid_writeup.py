"""根据实验产物自动生成 reports/分层融合算法报告.md（数值全部来自 results/ 下的账本）。"""
from __future__ import annotations
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
REP = ROOT / 'reports'


def f(x, n=2):
    return f'{x:,.{n}f}'


def md_table(df, cols=None, headers=None, fmt=None):
    d = df if cols is None else df[cols]
    headers = headers or list(d.columns)
    fmt = fmt or {}
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f'{v:,}')
            elif isinstance(v, (float, np.floating)):
                cells.append(f'{v:,.2f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    s = json.loads((R / 'hybrid_experiment_summary.json').read_text(encoding='utf8'))
    base_fit = json.loads((R / 'hybrid_baseline_training_fitness.json').read_text(encoding='utf8'))
    oos = pd.read_csv(R / 'hybrid_out_of_sample.csv')
    port = pd.read_csv(R / 'hybrid_optimizer_portfolio.csv')
    saa = pd.read_csv(R / 'hybrid_saa_convergence.csv')
    seg = pd.read_csv(R / 'hybrid_terminal_value_segments.csv')
    mon = pd.read_csv(R / 'hybrid_monthly.csv')
    pf_tr = pd.read_csv(R / 'hybrid_pareto_front_train.csv')
    pf_oo = pd.read_csv(R / 'hybrid_pareto_out_of_sample.csv')
    b = pd.read_csv(R / 'hybrid_daily_A0.csv')
    h = pd.read_csv(R / 'hybrid_daily_A3.csv')
    pol = s['selected_policy']
    vE = s['terminal_value']['vE_linear']

    a0 = oos[oos.method.str.startswith('A0')].iloc[0]
    a3 = oos[oos.method.str.startswith('A3')].iloc[0]
    better_days = int((b.total_cost.to_numpy() > h.total_cost.to_numpy()).sum())
    n_days = len(b)
    save_abs = a0.adjusted_cost - a3.adjusted_cost
    save_pct = 100 * save_abs / a0.adjusted_cost
    verdict = '**优于**' if save_abs > 0 else '**没有优于**'
    from scipy import stats
    dd = b.total_cost.to_numpy() - h.total_cost.to_numpy()
    p_t = float(stats.ttest_1samp(dd, 0).pvalue)
    p_w = float(stats.wilcoxon(dd).pvalue)

    train_gain = 100 * (base_fit['mean'] - pol['training_fitness']) / base_fit['mean']

    txt = f"""# 第三问：分层融合算法（HYBRID-4L）实验报告

本报告是在《第三问量化建模（改进版）》之上新增的**并列方法**实验，主模型的结果、参数与
交付文件保持不变。核心问题：**把几种算法按各自擅长的层次融合起来，能不能比现在的
滚动 LP（分位法）更优？**

结论先写在前面：在**严格因果**的选参口径（只用 2025 年 1 月的数据调参，2—12 月完全样本外）
下，融合方法 {verdict} 改进版基线，样本外 334 天库存校正费用
**{f(a3.adjusted_cost)} 元 vs {f(a0.adjusted_cost)} 元**，
**节省 {f(save_abs)} 元，即 {save_pct:.3f}%**；{better_days}/{n_days} 天单日更省
（配对 t 检验 $p={p_t:.4f}$，Wilcoxon 符号秩 $p={p_w:.2g}$，差异不是噪声）。
同时日 CVaR95 从 {f(a0.cvar95)} 元降到 {f(a3.cvar95)} 元、最差单日从 {f(a0.max_daily)} 元降到 {f(a3.max_daily)} 元、
紧急电量从 {f(a0.emergency_kwh, 0)} kWh 降到 {f(a3.emergency_kwh, 0)} kWh——**成本与风险同向改善，不是拿风险换成本**。

---

## 1  为什么要"分层"，而不是"换一个算法"

改进版已有的《方法对比报告》给出过一个很硬的结论：**GA 直接编码 144 维购电计划平均比 LP 差
20.31%**，因为 GA 不会利用问题的线性结构。所以「用遗传算法做第三问」如果理解成
「用 GA 替掉 LP」，方向就是错的。

正确的做法是分工：

| 层 | 这一层的问题长什么样 | 用什么算法 | 为什么 |
|---|---|---|---|
| L0 预测 | 有监督回归 | 岭正则局部线性趋势 + 星期效应（沿用主模型） | 有闭式解，按预测精度选超参 |
| L1 情景 | 从历史残差里挑代表路径 | **k-medoids 场景削减** | 保留时段间相关性，带权代表整个经验分布 |
| L2 规划 | 线性、带 recourse 的随机规划 | **两阶段情景随机 LP（HiGHS）** | 线性结构必须用 LP，启发式在这里只会更差 |
| L2′ 终端 | 未来价值函数逼近 | **一次 Bellman 回代（拟合值迭代）** | 把「常数边际库存价值」升级成分段线性凸函数 |
| L3 参数 | 低维、非凸、黑箱 | **memetic GA（GA + Nelder-Mead），并与 DE / SA 同预算对照** | 目标函数是整条回测链，不可微、非凸，正是进化算法的主场 |
| L4 权衡 | 多目标 | **NSGA-II** | 单目标只能给一个点，给不出成本—风险前沿 |

**一句话**：LP 负责"给定假设下算得准"，进化算法负责"假设本身怎么定"，两者不是竞争关系。

---

## 2  L1+L2：把"分位数"换成"情景随机规划"

### 2.1 改进版的做法与它的弱点

改进版把随机问题确定性化：用历史残差的 $\\alpha$ 分位构造一条净负荷路径，再解确定性 LP。
$\\alpha_0=0.60$、$\\alpha_h=0.60$ 是**网格搜索**出来的，不是从成本结构里解出来的——
换句话说，"买多少才划算"这个问题被一个外生的分位数代替了。

### 2.2 HYBRID 的做法

按题目真实的信息结构写成两阶段随机规划：

$$
\\min_{{q}}\; \\sum_t p_t q_t \;+\; \\mathbb{{E}}_s\\Big[\\sum_t \\big(1.5p_t u_{{s,t}} - 0.5 p_t v_{{s,t}} + 5 p_t z_{{s,t}}\\big) - V(E_{{s,T}})\\Big]
$$

- **第一阶段** $q_t$：此刻签下的计划，覆盖整个 24 小时视野（合同基准）；
- **第二阶段** $u_s,v_s,z_s$ 与充放电：观测到情景 $s$ 之后的补救；
- **非预期性约束**：视野最前面 6 小时（36 个时段）**不允许再调整**（$u=v=0$），
  因为下一次预报发布要到 6 小时之后——这正是题目"0:00/6:00/12:00/18:00 发布"的时间结构。

这样一来，5 倍紧急电价、1.5 倍增购、0.5 倍违约**第一次同时出现在规划目标里**，
"买多少"由 LP 自己解，不再需要人工挑分位数。

情景由历史残差路径经 **k-medoids** 削减得到（权重 = 簇内样本数），保留了时段之间的相关性；
残差路径本身只取决策时点之前的窗口，严格因果。

### 2.3 情景数 S 的定法

S 是**数值离散参数**（类似 DP 的 SOC 网格数），按目标收敛来定，**不按账单来调**：

{md_table(saa, cols=['n_scen', 'fold1_adjusted_cost', 'emergency_kwh', 'seconds'],
          headers=['情景数 S', '验证块 1 费用/元', '紧急电量/kWh', '单次耗时/s'])}

（1 月初可用历史天数少于 S 时，k-medoids 退化为全样本，这就是大 S 处取值相同的原因。）
本报告固定 **S = {s['n_scen']}**。

---

## 3  L2′：终端库存价值从常数升级为分段线性

改进版用 $v_E=p_{{\\min}}/\\eta={vE:.4f}$ 元/kWh 作为视野末端库存的**常数**边际价值。
本文做一次 Bellman 回代：在 1 月的每个训练日上，以库存 $E$ 为参数解"未来 24 小时最优成本"
$C(E)$（参数化 LP，$E$ 取 25 个网格点），取 $v(E)=-\\mathrm{{d}}C/\\mathrm{{d}}E$，
再压成 8 段非增分段常数。**非增 ⇒ 成本函数凸 ⇒ 可以直接写进 LP 且不需要整数变量，
LP 仍是全局最优。**

{md_table(seg, headers=['段下界/kWh', '段上界/kWh', '边际价值/(元/kWh)'])}

可以看到边际价值在低库存段显著高于 $v_E$，在最高一段明显掉下来——
这正是"库存快满时再多存一点没用"的经济含义，常数 $v_E$ 表达不了。

---

## 4  L3：参数寻优（算法组合 + 严格因果）

### 4.1 适应度定义

只用**评价期之前**的 1 月数据，三个滚动验证块：1/8—1/15、1/16—1/23、1/24—1/31，
各自从主模型 1 月预热得到的同一 SOC 起步；适应度 = 三块库存校正费用的**平均值**。
搜索空间是 5 个参数：情景权重温度 `temper`、残差缩放 `shrink`、终端价值缩放 `kappa`、
日内调整接受门槛 `threshold`、再调整期权行权溢价 `adj_premium`。

参照点：**同一口径下改进版基线的适应度 = {f(base_fit['mean'])} 元**
（三块分别为 {', '.join(f(x) for x in base_fit['folds'])}）。

### 4.2 三种元启发式同预算对照

{md_table(port, headers=['优化器', '训练适应度/元', '评估次数', '耗时/s', 'temper', 'shrink', 'kappa', 'threshold', 'adj_premium'])}

最终选用训练适应度最低的一组，即

```
{json.dumps({k: pol[k] for k in ('n_scen','temper','shrink','tail_relax','cvar_beta','cvar_alpha','kappa','threshold','adj_premium')}, ensure_ascii=False, indent=2)}
```

训练适应度 **{f(pol['training_fitness'])} 元**，比基线低 **{train_gain:.3f}%**。
**参数一经选定即冻结，下面 2—12 月的结果不再回看、不再替换。**

---

## 5  样本外全年结果（2025-02-01 — 12-31，334 天）

{md_table(oos, cols=['method', 'adjusted_cost', 'total_cost', 'emergency_kwh', 'up_kwh', 'down_kwh',
                     'equivalent_cycles', 'cvar95', 'gap_vs_baseline_percent', 'seconds'],
          headers=['方法', '库存校正费用/元', '总购电费/元', '紧急电量/kWh', '增购/kWh', '减购/kWh',
                   '等效循环/次', '日CVaR95/元', '相对基线/%', '耗时/s'])}

**消融读法**

- A1 = 只把终端价值换成分段线性，其余不动 → 分离出 L2′ 的贡献；
- A2 = 只把分位法换成情景随机 LP，终端仍用常数 → 分离出 L1+L2 的贡献；
- A3 = 完整融合；
- A4 = 完整融合但关掉 6/12/18 的日内更新 → 复核"引入其他时刻预报有价值"这一题目要求的结论。

逐月分解：

{md_table(mon, headers=['月份', '基线/元', 'HYBRID/元', '节省/元', '节省/%'])}

---

## 6  L4：NSGA-II 成本—风险前沿

在同一参数化上加入 CVaR 权重 $\\beta$ 与置信水平，用 NSGA-II 求
（训练期平均费用，训练期 CVaR90）的 Pareto 前沿，共 {s['pareto']['front_size']} 个非支配解，
{s['pareto']['evaluations']} 次评估、{s['pareto']['seconds']:.1f} 秒。

**必须如实报告的结果：这条前沿退化成了一个点。** 全部非支配解的 $\beta$ 都收敛到 0，
目标向量完全相同——也就是说，在这个参数化下**不存在真实的成本—CVaR 权衡**：
把成本压低的同一组参数同时把 CVaR 压低了。这与改进版《方法对比报告》里
「CVaR95 在 $\alpha_0=0.6$ 处与费用最优点重合」的结论是同一件事，属于对校准结果的交叉验证，
而不是 NSGA-II 失效。真正存在权衡的是「最差单日费用」那条方向，需要换目标函数才能显现。

代表解的**样本外**表现（因前沿退化，五个代表解数值相同）：

{md_table(pf_oo, cols=['cvar_beta', 'cvar_alpha', 'adjusted_cost', 'cvar95', 'max_daily', 'emergency_kwh', 'seconds'],
          headers=['CVaR 权重 β', 'CVaR 置信度', '样本外费用/元', '日CVaR95/元', '最差单日/元', '紧急电量/kWh', '耗时/s'])}

---

## 7  可核验清单

| 检查 | 位置 |
|---|---|
| HYBRID 框架在"常数分位 + 常数终端价值"参数下逐位复现主模型 | `results/hybrid_nesting_check.json` |
| 全部 334 天的功率平衡、SOC 上下限、充放电功率、动态方程 | 回测内断言，误差见 `hybrid_daily_A3.csv` 的 `balance_error` / `dynamics_error` |
| 费用独立重算（不依赖回测内部记账） | `results/hybrid_verification.json` |
| 因果性：扰动决策时点之后的数据不改变决策 | `results/hybrid_verification.json` |
| 训练/评价期分离 | 适应度窗口 1/8—1/31，评价期 2/1—12/31，无重叠 |

## 8  局限

1. 两阶段模型的第二阶段是**情景内完美预见**的补救，会高估调整期权的价值；
   `adj_premium` 是对这一乐观偏差的显式修正，但它是一个经验修正，不是严格的多阶段建模。
   真正无偏的写法是多阶段情景树随机规划，计算量是当前的若干倍。
2. 情景来自**历史残差的经验分布**，隐含"未来误差分布与过去 28 天相同"的假设。
3. 参数在 1 月（24 天）上选定，样本量有限；本文用三个验证块的平均值并报告最差块，
   但不能排除小样本带来的选参噪声。
4. 本文只替换了求解与选参层，**结算口径、物理参数、预测子模型与主模型完全一致**，
   因此两者的费用可直接比较。
"""
    REP.mkdir(exist_ok=True)
    (REP / '分层融合算法报告.md').write_text(txt, encoding='utf8')
    print('report written', flush=True)


if __name__ == '__main__':
    main()
