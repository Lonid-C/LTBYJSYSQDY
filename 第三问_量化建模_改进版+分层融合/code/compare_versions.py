"""生成《原版 / 第一轮改进版 / 第二轮改进版》三代对照报告。

原版数值来自原交付包 third-question 结果；第一轮改进版数值来自上一轮本目录 results/ 的实测
输出（作为常量登记，便于离线核对）；第二轮（本轮）数值从当前 results/ 读取。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'

# ---- 原交付包（第三问_量化建模_完整交付.zip）2—12 月实测值 ----
LEGACY = dict(
    none_total=14559779.4751451444, full_total=13440139.2799133062, full_final_soc=5867.9668253119,
    emergency_kwh=41358.4868423830, up_kwh=351741.7425716933, down_kwh=0.0,
    alpha0=0.65, alpha_update=0.5, initial=7291.237353593605,
    grid='0.35/0.50/0.65/0.80/0.90 x 0.35/0.50/0.60/0.70/0.80（50 组，无理论中心）',
    ci_low=625866.6204272751, ci_high=1211756.7608738136, elapsed=203.7)

# ---- 第一轮改进版（上一轮交付）2—12 月实测值 ----
ROUND1 = dict(
    none_total=14969809.57, full_total=13435399.87, emergency_kwh=81931.689413,
    up_kwh=563508.553832, down_kwh=735114.394206, final_soc=1236.245681, initial=1904.712849,
    alpha0=0.60, alpha_update=0.55, legacy_total=13426636.39, stepwise_total=None,
    fixed_total=13420255.06, theory_total=13489333.11, dayh_total=13467889.92,
    grid='0.60..0.90 x 0.50..0.80（49 组，最优落在下边界）',
    ci_low=1137462.448, ci_high=2016074.784, elapsed=578.7, slope=0.2, weekday=0.05)


def table(frame, digits=3):
    def fmt(v):
        return f'{v:,.{digits}f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(fmt, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def main():
    pars = json.loads((R / 'parameters.json').read_text())
    checks = json.loads((R / 'checks.json').read_text())
    p = pars['selected']
    sel = pars['selection']
    comp = pd.read_csv(R / 'strategy_comparison.csv')
    variants = pd.read_csv(R / 'variant_comparison.csv')
    boot = pd.read_csv(R / 'bootstrap.csv')
    base = comp[comp.strategy == 'none'].iloc[0]
    full = comp[comp.strategy == '6+12+18'].iloc[0]
    main_row = variants[variants.variant == 'main_net_refund'].iloc[0]
    legacy_row = variants[variants.variant == 'legacy_gross_settlement'].iloc[0]
    step_row = variants[variants.variant == 'stepwise_settlement'].iloc[0]
    fixed_row = variants[variants.variant == 'fixed_target_soc'].iloc[0]
    theory_row = variants[variants.variant == 'theory_quantiles'].iloc[0]
    swin_row = variants[variants.variant == 'single_window_params'].iloc[0]
    b = boot[(boot.strategy == '6+12+18') & (boot.block_days == 7)].iloc[0]
    save1 = ROUND1['none_total'] - ROUND1['full_total']
    save2 = base.total_cost - full.total_cost
    slope_txt = r'10^{6}' if p['slope'] >= 1e5 else f"{p['slope']:g}"
    fold_txt = '；'.join(f"第 {f['fold']} 折 训练 {f['train']} → 验证 {f['validation']}" for f in sel['folds'])

    out = f'''# 第三问：原版 / 第一轮改进版 / 第二轮改进版 对照

本文件用于审阅两轮修订的效果。原版数值取自 `第三问_量化建模_完整交付.zip`；第一轮改进版数值为上一轮本目录实测输出；第二轮（本轮）数值取自当前 `results/`。两轮修订依据分别是《第三问模型改进建议：结算口径与参数来源说明》与《第三问改进版：最终修正清单》。

## 1. 关键指标

| 指标 | 原版 | 第一轮改进版 | 第二轮改进版（本轮） |
|---|---|---|---|
| 结算口径 | 原费照付＋增购1.5倍＋减购另加0.5倍罚金 | 净额结算＋取消部分仅承担0.5倍 | 同左，并新增逐次调整结算对照 |
| 优化视野 | 每个发布时刻只优化到当日24:00 | 发布时刻→次日同一时刻（24h滚动） | 同左，且报告与代码一致 |
| 终端条件 | 人工固定 $E_{{24}}=0.5B$ | 经济库存价值 $-v_EE_D$ | 同左 |
| 风险分位命名 | 直接校准，无理论来源 | 称为理论值 | **经济理论基准分位**（局部单阶段理论临界分位） |
| 校准网格 | {LEGACY['grid']} | {ROUND1['grid']} | {f"{len(sel['alpha0_grid'])}×{len(sel['alpha_update_grid'])} 组，区间 {sel['alpha0_grid'][0]:g}–{sel['alpha0_grid'][-1]:g} × {sel['alpha_update_grid'][0]:g}–{sel['alpha_update_grid'][-1]:g}；最优位于区域内部"} |
| 选参规则 | 1月15—31日单窗口 | 1月15—31日单窗口 | **滚动多窗口时序验证**（3 个评价期前窗口的平均成本） |
| 选定风险参数 | $\\alpha_0={LEGACY['alpha0']}$, $\\alpha_h={LEGACY['alpha_update']}$ | $\\alpha_0={ROUND1['alpha0']}$, $\\alpha_h={ROUND1['alpha_update']}$ | $\\alpha_0={p['alpha0']:g}$, $\\alpha_h={p['alpha_update']:g}$ |
| 岭正则（预测超参数） | 手工设定 | 手工设定 0.2 / 0.05 | **按预测精度选择**：$\\lambda_{{slope}}={slope_txt}$, $\\lambda_{{weekday}}={p['weekday']:g}$ |
| 2—12 月总购电费（全更新） | {LEGACY['full_total']:,.2f} 元 | {ROUND1['full_total']:,.2f} 元 | {full.total_cost:,.2f} 元 |
| 不更新对照 | {LEGACY['none_total']:,.2f} 元 | {ROUND1['none_total']:,.2f} 元 | {base.total_cost:,.2f} 元 |
| 节约 | {LEGACY['none_total'] - LEGACY['full_total']:,.2f} 元（7.690%） | {save1:,.2f} 元（{100 * save1 / ROUND1['none_total']:.3f}%） | {save2:,.2f} 元（{100 * save2 / base.total_cost:.3f}%） |
| 累计减购量 | {LEGACY['down_kwh']:,.0f} kWh（被严格支配） | {ROUND1['down_kwh']:,.0f} kWh | {full.down_kwh:,.0f} kWh |
| 累计增购量 | {LEGACY['up_kwh']:,.0f} kWh | {ROUND1['up_kwh']:,.0f} kWh | {full.up_kwh:,.0f} kWh |
| 紧急购电量 | {LEGACY['emergency_kwh']:,.0f} kWh | {ROUND1['emergency_kwh']:,.0f} kWh | {full.emergency_kwh:,.0f} kWh |
| 2月1日初始库存 | {LEGACY['initial']:,.1f} kWh | {ROUND1['initial']:,.1f} kWh | {pars['initial_feb1']:,.1f} kWh |
| 7日区块 bootstrap 95% 区间 | [{LEGACY['ci_low']:,.0f}, {LEGACY['ci_high']:,.0f}] 元 | [{ROUND1['ci_low']:,.0f}, {ROUND1['ci_high']:,.0f}] 元 | [{b.ci_low:,.0f}, {b.ci_high:,.0f}] 元 |

> 三代之间的总费用不可直接当作“同一合同下的算法优劣”：结算口径、终端条件、优化视野与选参规则都已改变。可比的是**同一合同内部**的结构对比与**同一目标函数内**的参数校准。另外三代的 2 月 1 日起点库存不同，这一差异本身对应数百元量级的库存价值。

## 2. 第一轮修订（结算口径与参数来源）的落地结论

### 2.1 主结算公式

$$C=\\sum_t\\left[p_tq_t^0+1.5p_t(q_t-q_t^0)^+-0.5p_t(q_t^0-q_t)^++5p_tz_t\\right].$$

旧解释采用「原计划费用照付，减购另加 50% 罚金」，使减购在数学上被严格支配（100→80 需付 $110p_t$，高于不调整的 $100p_t$），因此减购量恒为 0，题目专门设置的 50% 违约价格形同虚设。修正后减购在合同上重新具有经济意义，本轮实测累计减购 {full.down_kwh:,.0f} kWh、减购净额 {full.down_net_cost:,.2f} 元。

### 2.2 三种结算解释的对照（本轮新增逐次结算，对应清单 §11 S1/S2）

| 结算解释 | 2—12 月总费 / 元 | 累计减购 / kWh | 相对主口径 |
|---|---:|---:|---:|
| 主口径：净额退款结算 | {main_row.total_cost:,.2f} | {main_row.down_kwh:,.3f} | 0.000% |
| 旧解释：原费照付＋0.5 倍罚金 | {legacy_row.total_cost:,.2f} | {legacy_row.down_kwh:,.3f} | {100 * (legacy_row.total_cost / main_row.total_cost - 1):+.3f}% |
| 逐次调整结算 | {step_row.total_cost:,.2f} | {step_row.down_kwh:,.3f} | {100 * (step_row.total_cost / main_row.total_cost - 1):+.3f}% |

核心结论对结算解释不敏感：不更新策略不产生任何调整，因此三种解释下它的费用完全相同（{base.total_cost:,.2f} 元），而全更新在三种解释下分别节约 {100 * save2 / base.total_cost:.3f}%、{100 * (base.total_cost - legacy_row.total_cost) / base.total_cost:.3f}%、{100 * (base.total_cost - step_row.total_cost) / base.total_cost:.3f}%。

## 3. 第二轮修订（最终修正清单）的落地结论

### 3.1 P0-1 风险分位搜索范围扩大与边界自动延展

第一轮的最优解 $\\alpha_0=0.60$ 恰好落在搜索区间下边界，只能说明“在已测候选中 0.60 最好”，不能说明邻域已覆盖完整。本轮把候选集合扩展为 $\\alpha_0\\in\\{{{','.join(f'{v:g}' for v in sel['alpha0_grid'])}\\}}$、$\\alpha_h\\in\\{{{','.join(f'{v:g}' for v in sel['alpha_update_grid'])}\\}}$，并加入**边界自动延展**机制：若最优解仍落在边界，继续向下/向上扩展 0.05 后重测，直至最优参数位于已测试区域内部。本轮最优参数 $({p['alpha0']:g},{p['alpha_update']:g})$ 已位于区域内部（$\\alpha_0$ 内部 = {checks['alpha0_interior']}，$\\alpha_h$ 内部 = {checks['alpha_update_interior']}），因此不再存在因搜索边界造成的截断偏差。延展过程见 `results/calibration_grid_extension.json`。

### 3.2 P0-3 术语修正：0.80/0.70 的地位

0.80 与 0.70 是**忽略储能跨时耦合的局部单阶段边际成本模型**给出的理论临界分位，本文统一称为「经济理论基准分位」，仅作为完整储能滚动优化模型的**理论锚点**，不再称为完整模型的理论最优分位。直接使用该锚点而不校准的费用为 {theory_row.total_cost:,.2f} 元（相对主口径 {100 * (theory_row.total_cost / main_row.total_cost - 1):+.3f}%）。

### 3.3 P1-4/P1-5 弱化唯一最优表述 + 滚动多窗口时序验证

本轮新增滚动多窗口时序验证：{fold_txt}，训练窗口随折次扩展、每个折次都只用该折之前的数据，并按三个验证区块的平均库存校正成本选参。主口径即为该规则选出的 $({sel['primary_alpha0']:g},{sel['primary_alpha_update']:g})$；单窗口规则（1月15—31日）选出的 $({sel['single_window_alpha0']:g},{sel['single_window_alpha_update']:g})$ 在评价期的结果为 {swin_row.total_cost:,.2f} 元（相对主口径 {100 * (swin_row.total_cost / full.total_cost - 1):+.3f}%）。两种事前选参规则给出非常接近的结果，因此报告中改为“某一参数邻域形成稳定低成本区域”，而不是断言某个点唯一最优。

### 3.4 P2-11/P2-12 参数邻域稳定性与“是否仍需日内更新”的比例

对全部 {len(sel['alpha0_grid']) * len(sel['alpha_update_grid'])} 个风险参数组合在评价期逐一回放（`stability_map.csv`、`figures/参数邻域稳定性.*`）：**{checks['fraction_full_beats_none']:.1%} 的参数组合下“6/12/18 全更新”仍然优于不更新**，最低成本 0.2% 邻域占全部组合的 {pars['stability']['fraction_within_0.2pct_of_min']:.1%}。这说明“是否应当引入日内预测”这一核心结论不依赖单一参数点。

### 3.5 P1-6/P1-7 预测子模型超参数按预测精度选择

岭正则系数属于预测子模型的数值超参数，不具经济含义。本轮不再手工设定，而是按**预测精度**选择：在评价期之前的 1 月 8—31 日上逐候选做“只用当日之前数据拟合、预测当日全天负荷”的时序校核，比较 MAE/RMSE，取 RMSE 最小者，得到 $\\lambda_{{slope}}={slope_txt}$、$\\lambda_{{weekday}}={p['weekday']:g}$。这样把两类参数彻底分开：**预测模型参数按预测准确性选择，风险决策参数按最终经济成本选择**。完整候选见 `ridge_selection.csv`。

需要如实说明：预测精度准则把趋势项的正则强度推到了上限（$\\lambda_{{slope}}=10^6$，等价于在基准预报中取消局部线性趋势项），说明本文历史窗口内的局部趋势对预测精度没有正贡献，日历（星期）效应与日内偏差校正承担了主要解释力。这是数据层面的结论，而不是人为设定。

### 3.6 P0-2 报告与代码一致性

第一轮报告中仍保留了「没有把附件 3 中跨到次日的部分用于跨日优化」的表述，与代码已实现的跨午夜 24 小时滚动视野矛盾，本轮已删除并改写为与代码一致的描述（见 `RESULTS_REPORT.md` 第 5 节与第 12 节），同时明确模型边界：**各次优化仍是基于当前信息集的确定性滚动优化，不是完整多阶段随机规划**。

## 4. 结构敏感性一览（本轮）

| 结构设定 | 总购电费 / 元 | 库存校正费用 / 元 | 相对主口径（校正后） |
|---|---:|---:|---:|
| 主口径：净额结算 + 24h 滚动视野 + 库存价值 + 多窗口选参 | {main_row.total_cost:,.2f} | {main_row.adjusted_cost:,.2f} | {100 * (main_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 旧解释：原费照付＋0.5 倍罚金 | {legacy_row.total_cost:,.2f} | {legacy_row.adjusted_cost:,.2f} | {100 * (legacy_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 逐次调整分别计费 | {step_row.total_cost:,.2f} | {step_row.adjusted_cost:,.2f} | {100 * (step_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 人工固定 $E_{{24}}=0.5B$ | {fixed_row.total_cost:,.2f} | {fixed_row.adjusted_cost:,.2f} | {100 * (fixed_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 视野仅到当日 24:00 | {variants[variants.variant == 'day_horizon_only'].iloc[0].total_cost:,.2f} | {variants[variants.variant == 'day_horizon_only'].iloc[0].adjusted_cost:,.2f} | {100 * (variants[variants.variant == 'day_horizon_only'].iloc[0].adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 经济理论基准分位 0.80/0.70（不校准） | {theory_row.total_cost:,.2f} | {theory_row.adjusted_cost:,.2f} | {100 * (theory_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |
| 单窗口校准参数（$\\alpha_h={sel['single_window_alpha_update']:g}$） | {swin_row.total_cost:,.2f} | {swin_row.adjusted_cost:,.2f} | {100 * (swin_row.adjusted_cost / main_row.adjusted_cost - 1):+.3f}% |

## 5. 与原版一致、未改动的部分

- 因果信息约束：任何预测只使用决策时点之前已可获得的信息；`checks.json` 中未来数据扰动后预测变化为 {checks['causality_future_perturbation_max_diff']:.3e}。
- 分位数残差库取「当前日之前完整历史日」的残差，并按**提前量**组织。
- 8 种日内更新组合的消融框架、条件边际价值、纯光伏信息消融、区块 bootstrap。
- 逐段账本、逐日账本、约束断言与提交表结构。
- 数据时段不平移；原模板另存于 `data/result3_原模板.xlsx`。

## 6. 工程性修正

| 项 | 原版 | 本轮 |
|---|---|---|
| 充放电表“时刻”列 | 第二个 4 小时块被标为 24:00 | 只在最后一个块标 24:00，首块标 00:00 |
| 工作表汇总列 | 依赖第三方表格工具的公式重算与缓存 | 直接写入数值，保证任意查看器可显示，并由 `verify.py` 独立重算校验 |
| 结算重算自检 | 独立脚本按旧口径重算 | `verify.py` 按**主口径**逐日重算总费与四个分项，并校验 $v_E$ |
| 参数层自检 | 无 | 断言旧解释减购为 0；断言最优参数位于搜索区域内部 |

## 7. 复现

```bash
python code/q3.py               # 主模型（含校准、边界延展、多窗口验证、稳定性图、敏感性）
python code/extra_sensitivity.py
python code/pure_release.py
python code/deliver.py
python code/export_xlsx.py
python code/verify.py
python code/compare_versions.py
node code/build_report.mjs
```

本轮约束检查：平衡最大误差 {checks['balance_max_error']:.3e} kWh、SOC 动力学最大误差 {checks['soc_dynamics_max_error']:.3e} kWh、跨日 SOC 衔接差 {checks['cross_day_soc_max_gap']:.3e} kWh、实际 SOC 范围 [{checks['soc_min']:.3f}, {checks['soc_max']:.3f}] kWh、主程序耗时 {checks['elapsed_seconds']:.1f} 秒。
'''
    (ROOT / 'reports/改进前后对比.md').write_text(out, encoding='utf8')
    print('saved reports/改进前后对比.md')


if __name__ == '__main__':
    main()
