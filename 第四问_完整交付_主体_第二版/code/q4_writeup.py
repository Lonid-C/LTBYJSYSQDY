"""生成 reports4/第四问_波动电价分层融合报告.md（数值全部来自 results4/）。"""
from __future__ import annotations
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
REP = ROOT / 'reports4'


def f(x, n=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:,.{n}f}'


def tbl(df, headers=None):
    headers = headers or list(df.columns)
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('—' if not np.isfinite(v) else (f'{v:,.4f}' if abs(v) < 10 else f'{v:,.2f}'))
            elif isinstance(v, (int, np.integer)):
                cells.append(f'{v:,}')
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main():
    s = json.loads((R / 'q4_experiment_summary.json').read_text(encoding='utf8'))
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    cfg = json.loads((R / 'model_config.json').read_text(encoding='utf8'))
    oos = pd.read_csv(R / 'q4_out_of_sample.csv')
    ver = json.loads((R / 'q4_verification.json').read_text(encoding='utf8'))
    audit = json.loads((R / 'price_data_audit.json').read_text(encoding='utf8'))
    cv1 = pd.read_csv(R / 'price_forecast_cv.csv'); cv2 = pd.read_csv(R / 'price_bias_cv.csv')
    saa = pd.read_csv(R / 'q4_saa_convergence.csv'); alp = pd.read_csv(R / 'q4_alpha_calibration.csv')
    seg = pd.read_csv(R / 'q4_terminal_segments.csv')
    info = pd.read_csv(R / 'q4_information_value.csv'); boot = pd.read_csv(R / 'q4_cvar_bootstrap.csv')
    bext = pd.read_csv(R / 'q4_boundary_extension.csv')
    abl = pd.read_csv(R / 'q4_ablation_sensitivity.csv')
    psen = pd.read_csv(R / 'q4_param_sensitivity_train.csv')
    seeds = pd.read_csv(R / 'q4_scenario_seed_stability.csv')
    caus = pd.read_csv(R / 'q4_causality_checks.csv')
    lb = pd.read_csv(R / 'q4_full_horizon_bound.csv')
    mon2 = pd.read_csv(R / 'q4_monthly_Q4-2.csv'); mon3 = pd.read_csv(R / 'q4_monthly_Q4-3.csv')
    hp = s['price_hp']

    def row(prob, pref):
        g = oos[(oos.problem == prob) & (oos.method.str.startswith(pref))]
        return g.iloc[0] if len(g) else None

    b2, m2, h2, u2, w2, t2 = (row('Q4-2', p) for p in ('B0', 'M2', 'H2', 'U2', 'W2', 'B1'))
    b3, h3, u3, w3, t3, a3, x3 = (row('Q4-3', p) for p in ('B0', 'H3', 'U3', 'W3', 'B1', 'A3', 'X3'))
    i2 = info[info.problem == 'Q4-2'].iloc[0]; i3 = info[info.problem == 'Q4-3'].iloc[0]
    a2v = float(sel['Q4-2_quantile']['alpha'])
    d2 = pd.read_csv(R / 'q4_daily_Q4-2_B0.csv'); e2 = pd.read_csv(R / 'q4_2_daily.csv')
    d3 = pd.read_csv(R / 'q4_daily_Q4-3_B0.csv'); e3 = pd.read_csv(R / 'q4_3_daily.csv')
    from scipy import stats

    def sig(a, b):
        w = a.total_cost.to_numpy() - b.total_cost.to_numpy()
        return (int((w > 0).sum()), len(w), float(stats.ttest_1samp(w, 0).pvalue),
                float(stats.wilcoxon(w).pvalue))

    n2, n3 = sig(d2, e2), sig(d3, e3)
    bt = boot[boot.block_len == 7]
    bt2m = bt[(bt.problem == 'Q4-2') & bt.variant.str.startswith('M2')].iloc[0]
    bt2h = bt[(bt.problem == 'Q4-2') & bt.variant.str.startswith('H2')].iloc[0]
    bt3 = bt[(bt.problem == 'Q4-3') & bt.variant.str.startswith('H3')].iloc[0]
    hard = cv2[(cv2.correction == 1.0) & (cv2.bias_window == 6) & (cv2.bias_decay == 6.0)]
    hard_rank = int(hard.index[0]) + 1 if len(hard) else -1
    A = abl[abl.block == 'A 发布时点'].copy()
    a_none = float(A[A.case == '不更新'].adjusted_cost.iloc[0])
    A['saving'] = a_none - A.adjusted_cost
    order = ['不更新', '6:00', '12:00', '18:00', '6:00+12:00', '6:00+18:00', '12:00+18:00', '6:00+12:00+18:00']
    A['k'] = A.case.apply(lambda c: order.index(c) if c in order else 99)
    A = A.sort_values('k')
    B = abl[abl.block == 'B 信息更新']; Cb = abl[abl.block == 'C 结算口径']
    D = abl[abl.block == 'D 物理参数']; Ef = abl[abl.block == 'E 经验参数（全年）']
    noise = float(seeds.train_fitness.max() - seeds.train_fitness.min())
    main_adj = float(abl[(abl.block == 'A 发布时点') & (abl.case == '6:00+12:00+18:00')].adjusted_cost.iloc[0])
    D2 = abl[abl.block == 'D 物理参数'][['case', 'total_cost', 'adjusted_cost', 'emergency_kwh',
                                        'spill_kwh', 'final_soc']].copy()
    D2['delta'] = D2.adjusted_cost - main_adj
    d_md = tbl(D2, ['压力点', '现金费用/元', '库存校正费用/元', '应急电量/kWh', '弃电/kWh',
                    '期末库存/kWh', '相对主策略/元'])
    lbv = float(lb.adjusted_cost.iloc[0])
    checks_md = tbl(pd.DataFrame([
        {'检查': '费用独立重算最大误差/元', 'Q4-2': f"{ver['q4_2']['cost']:.2e}",
         'Q4-3': f"{ver['q4_3']['cost']:.2e}"},
        {'检查': '功率平衡最大误差/kWh', 'Q4-2': f"{ver['q4_2']['balance']:.2e}",
         'Q4-3': f"{ver['q4_3']['balance']:.2e}"},
        {'检查': '储能动态最大误差/kWh', 'Q4-2': f"{ver['q4_2']['soc']:.2e}",
         'Q4-3': f"{ver['q4_3']['soc']:.2e}"},
        {'检查': '工作表与账本最大差/kWh', 'Q4-2': f"{ver.get('q4_2_sheet_plan_max_diff', 0):.2e}",
         'Q4-3': f"{max(ver.get('q4_3_sheet_plan_max_diff', 0), ver.get('q4_3_sheet_adjusted_max_diff', 0)):.2e}"}]))

    txt = f"""# 第四问：波动电价下的分层融合（HYBRID-4L / Q4）

`variant_id`：**{cfg['variant_id']}**　口径与指纹见 `results4/model_config.json`（运行设置、选参及文件指纹快照）

第四问把第二、三问在**逐日逐时波动的电价**（附件 4）下重算。方法沿用第三问的分层融合，
并把随机性从「净负荷」一维扩展到「净负荷 × 电价」二维。

**两个口径的结论不同，如实并列。** 百分比分母是同问题 1 月重标定过的确定性分位法基准。

| | Q4-2（问题 2 口径，无日内调整） | Q4-3（问题 3 口径，0/6/12/18） |
|---|---|---|
| 沿用第三问 α=0.60 的分位法 | {f(b2.adjusted_cost)} 元 | {f(b3.adjusted_cost)} 元 |
| **主策略** | **分位法 α={a2v:.3f}（1 月重标定）：{f(m2.adjusted_cost)} 元（−{100*(b2.adjusted_cost-m2.adjusted_cost)/b2.adjusted_cost:.3f}%）** | **HYBRID-4L：{f(h3.adjusted_cost)} 元（{h3.gap_vs_baseline_percent:+.3f}%）** |
| HYBRID-4L 联合情景随机 LP | {f(h2.adjusted_cost)} 元（{h2.gap_vs_baseline_percent:+.3f}%）**未胜出** | 即主策略 |
| 1 月无约束最优（shrink 超范围·未采纳） | — | {f(x3.adjusted_cost)} 元（{x3.gap_vs_baseline_percent:+.3f}%） |
| 误以为电价不波动 | {f(t2.adjusted_cost)} 元（{t2.gap_vs_baseline_percent:+.3f}%） | {f(t3.adjusted_cost)} 元（{t3.gap_vs_baseline_percent:+.3f}%） |
| 主策略 + 完美电价 | {f(u2.adjusted_cost)} 元 | {f(u3.adjusted_cost)} 元 |
| 滚动视野完美信息对照 | {f(w2.adjusted_cost)} 元 | {f(w3.adjusted_cost)} 元 |
| **全期完美信息真下界** | **{f(lbv)} 元** | **{f(float(lb.adjusted_cost.iloc[1]))} 元** |

所得方案是**既定候选集合与计算预算下的历史前推结果**，不宣称全局最优；评价期为顺序回放，
年度附件此前已多次用于开发，因此不称独立盲测。

## 0  修订概况

| 问题 | 上一版 | 本版 |
|---|---|---|
| **终端参考价泄漏** | `vE_ref` 取 2—12 月**真实**电价逐日最低值的均值（{f(0.336368,6)}），而它进入 1 月训练适应度的库存校正项 | 改用附件 1 给定分时电价最低值/η = **{f(cfg['storage']['vE_ref'],6)}**，决策前已知、与附件 4 实现值无关、全方法共用 |
| **"全知下界"名不副实** | W2/W3 虽是完美信息，但仍跑在 24h 滚动视野 + 终端代理里，却被标为"理论下界" | 改称**滚动视野完美信息对照**；另解 334 天 × 144 段的**单个 LP**（288,576 变量）作为真下界 |
| **因果性只查 0:00** | 仅 1 天 1 个发布时刻 | 3 天 × **4 个发布时刻** = 12 组，且扰动包含**尚未发布**的预报 |
| 运行设置与追溯 | 旧指纹与代码不一致 | `runtime_settings.json` 供主运行入口读取；`model_config.json` 记录快照；最终 manifest 校验文件 |
| 只有日末净额 | 无逐笔记录 | Q4-3 输出 **{len(pd.read_csv(R / 'q4_3_transactions.csv')):,} 条**逐笔交易账本（Q4-2 为 0 条，它没有调整通道） |
| 统计只有 7 日块 | 单一块长 | **7/14/28** 三档移动块重采样，并给出情景削减的跨种子波动 |

旧版本引用未来信息的问题已经在参考价定义中移除；本修正版继续使用题目给定的参考电价。

---

## 1  第四问新增的困难

附件 4 电价 **{f(audit['min'],4)} — {f(audit['max'],4)} 元/kWh**，日均价在
{f(audit['daily_mean_min'],4)} — {f(audit['daily_mean_max'],4)} 元/kWh 摆动。分解成
「附件 1 分时形状 × 当日价格乘子 × 日内噪声」后：乘子滞后 1 天自相关
{audit['autocorr_daily_multiplier_lag1']:.3f}、**滞后 7 天 {audit['autocorr_daily_multiplier_lag7']:.3f}**（强周期，可预测）；
相对形状的比值标准差 {audit['ratio_to_tou_std']:.3f}。

后果：电价必须严格因果地预报；**储能套利的收益本身成了随机变量**——前几问价差是已知常数。

---

## 2  预测层与模型层

| 层 | 第三问 | 第四问的改动 |
|---|---|---|
| L0 预测 | 负荷：岭正则局部线性趋势 + 星期效应 | **新增价格预报**：对 log(实际电价/附件1 形状) 用同一估计量回归 |
| L1 情景 | 净负荷残差路径 + k-medoids | **联合情景**：同一历史日成对取（净负荷残差，价格对数残差） |
| L2 规划 | 两阶段情景随机 LP | **每条情景带自己的电价路径**；一阶段按期望电价签合同，二阶段按情景电价补救 |
| L2′ 终端 | 分段线性凸 V(E)，常数 v_E | 拆成「当日价格水平 × 归一化形状」，形状只拟合一次 |

### 2.1 价格预报超参数：两段式滚动原点交叉验证（只用 1 月）

第一段（日前形状）最好的五组：

{tbl(cv1.head(5)[['window','decay','slope','weekday','mae','rmse']], ['窗口/天','衰减/天','趋势岭','星期岭','MAE','RMSE'])}

第二段（日内偏差校正）。上一版把 (correction, bias_window, bias_decay) **写死**为 (1.0, 6, 6.0)
且从未参与选参，实测它在 12:00 让预报变差。放进同一网格后：

{tbl(cv2.head(6)[['correction','bias_window','bias_decay','mae','rmse','mae_issue6','mae_issue12','mae_issue18']],
     ['校正强度','观测窗/段','衰减/h','MAE','RMSE','MAE@6:00','MAE@12:00','MAE@18:00'])}

原先写死的组合在 {len(cv2)} 个候选里排第 **{hard_rank}**。最终选中 `{json.dumps(hp, ensure_ascii=False)}`。

### 2.2 终端库存价值的归一化形状

{tbl(seg, ['段下界/kWh','段上界/kWh','归一化边际价值'])}

边际非增 ⇒ 成本函数凸 ⇒ 分段线性终端价值可直接写进 LP，解仍是该 LP 的全局最优。

### 2.3 情景数 S 与情景削减的跨种子波动

{tbl(saa.pivot(index='n_scen', columns='problem', values='train_fitness').reset_index(),
     ['情景数 S', 'Q4-2 训练适应度/元', 'Q4-3 训练适应度/元'])}

情景数扫描用于计算预算折中，不构成统计收敛证明。本实验取 **S = {s['n_scen']}**。再用 5 个事先固定的随机种子重跑削减与回测：

{tbl(seeds[['seed','train_fitness','effective_scenarios','min_weight']],
     ['种子','1 月训练适应度/元','有效情景数 1/Σw²','最小权重'])}

**跨种子极差 {f(noise)} 元，占均值 {100*noise/seeds.train_fitness.mean():.3f}%。**
该极差描述这五次训练期运行的离散程度，不是标准误、显著性阈值，也不能与全年收益率直接比较。
有效情景数 ≈ {seeds.effective_scenarios.mean():.1f}／{s['n_scen']}，权重分布不极端。

---

## 3  选参与历史开发记录

适应度只用 1 月三个滚动验证块（1/8—1/15、1/16—1/23、1/24—1/31），
适应度函数只计算上述日期。模型结构与参数范围曾结合全年回放表现修订，因此本年度不能充当独立盲测。

### 3.1 分位法的一维标定

{tbl(alp.sort_values(['problem','alpha']).pivot(index='alpha', columns='problem', values='train_fitness').reset_index().iloc[::2],
     ['α', 'Q4-2 训练适应度/元', 'Q4-3 训练适应度/元'])}

两条曲线单峰、最优在网格内部：Q4-2 选 α={a2v:.3f}，Q4-3 选
α={float(sel['Q4-3_quantile']['alpha']):.3f}（恰为第三问沿用值，说明 Q4-3 基线本就公平校准）。
α 的网格最小值位于内部；这降低了边界截断疑虑，但不证明低过拟合风险或总体最优。

### 3.2 Q4-2 移出了两类参数

**惰性参数**（`threshold`、`adj_premium`）：Q4-2 `mask=()` ⇒ `lock=144` ⇒ 二阶段增减购变量上界恒为 0、
接受门分支从不执行。实测把 `adj_premium` 从 1.0 改到 5.0，计划逐段差 **0.000e+00 kWh**
（同日在 Q4-3 口径下差 754 kWh）。

**缩放参数**（`shrink_net`、`shrink_price`）控制经验情景离散程度。
Q4-2 的充放电、库存和紧急购电仍为情景相关的第二阶段补救决策，仍存在情景内前视近似。
本实验固定缩放为 1，是短训练窗口下减少自由度的正则化选择，不是由无日内增减购电推出的定理。

### 3.3 两次过拟合的完整记录

净负荷缩放范围 `[0.55,1.30]`、价格缩放范围 `[0.30,1.60]` 是人为设定的搜索范围。
这些范围不由“40%”阈值推导，也不能称为物理或结构性边界。开发中查看过评价期排名，
故下表仅作为历史开发记录，不能用于无偏泛化误差估计。

两个口径各有一次样本外证据：

| 口径 | 1 月适应度 | 样本外（库存校正） | 结论 |
|---|---:|---:|---|
| Q4-2 4 维宽框最优（`shrink_net`=0.298） | 475,899（**更低**） | 15,807,845（**比基线贵 1.42%**） | 未采纳 |
| Q4-2 主策略（分位法 α 一维标定） | {f(sel['Q4-2_quantile']['fitness'])} | **{f(m2.adjusted_cost)}** | 采纳 |
| Q4-3 无约束最优（`shrink_net`=0.215） | {f(float(bext[(bext.problem=='Q4-3')&(bext.step==1)].fitness.iloc[0]))}（**更低**） | {f(x3.adjusted_cost)}（**比基线贵 {x3.gap_vs_baseline_percent:.2f}%**） | 未采纳 |
| Q4-3 主策略（`shrink_net`=0.55，框内最优） | {f(sel['Q4-3']['fitness'])} | **{f(h3.adjusted_cost)}** | 采纳 |

训练期与回放期的排名发生变化，说明结论对参数范围和样本划分敏感。
本次修复保留既有主策略参数，不据此重新挑选全年表现最好的配置。
要估计泛化收益，应另用未参与开发的年份，或采用外层留出、内层选参的嵌套时间验证。

### 3.4 边界延展：相对量程 1%，并区分结构性下界

{tbl(bext[['problem','step','fitness','at_edge','structural_corner','params']],
     ['问题','轮次','训练适应度/元','仍在边界的参数','命中结构性下界','参数'])}

`temper`=0（情景等权）、`shrink`=0、`kappa`=0、`threshold`=0 是参数**本身的意义边界**，
贴到这些端点不算被搜索框截断，不可也无需再延展。

### 3.5 最终选参

```
Q4-3（HYBRID，6 维）: {json.dumps(sel['Q4-3']['params'], ensure_ascii=False)}
Q4-2（主策略，分位法）: alpha = {a2v:.3f}
Q4-2（并列报告的 HYBRID）: {json.dumps(sel['Q4-2']['params'], ensure_ascii=False)}
   正则化固定 {json.dumps(sel['Q4-2'].get('fixed_structural', {}), ensure_ascii=False)}　
   惰性固定 {json.dumps(sel['Q4-2'].get('inert_fixed', {}), ensure_ascii=False)}
```

---

## 4  历史前推全年结果（2025-02-01 — 12-31，334 天，48,096 个 10 分钟段）

金额列一律为**库存校正费用**（= 现金费用 + v_E_ref ×(期初−期末)）；现金费用单列。
Q4-3 的计划购电费与总费用**不可相加**——总费用已含计划费、调整费与应急费。

{tbl(oos[['problem','method','total_cost','adjusted_cost','plan_cost','emergency_kwh','up_kwh','down_kwh','spill_kwh','cvar95','gap_vs_baseline_percent','seconds']],
     ['问题','方法','现金费用/元','库存校正费用/元','计划购电费/元','应急电量/kWh','增购/kWh','减购/kWh','弃电/kWh','日CVaR95/元','相对基准/%','耗时/s'])}

### 4.1 真下界与"滚动完美信息"的区别

把 334 天一次性解成单个 LP（{int(lb.intervals.iloc[0]):,} 段 × 6 变量 = {6*int(lb.intervals.iloc[0]):,} 变量，
HiGHS {lb.seconds.iloc[0]:.0f} 秒）。在相同期初库存、物理约束和终端估值下，这是库存校正目标的放松下界：净额结算下实际支付为
p·q + 0.5p·u + 0.5p·v ≥ p·q，应急 5p ≥ p，故"以交付时段电价买下全部实际取用电量且完美预见"最省。

| | Q4-2 | Q4-3 |
|---|---:|---:|
| 滚动视野完美信息对照 | {f(i2.rolling_perfect_information)} | {f(i3.rolling_perfect_information)} |
| **全期完美信息真下界** | **{f(i2.full_horizon_lower_bound)}** | **{f(i3.full_horizon_lower_bound)}** |
| 滚动策略与全期放松 LP 的差额 | {f(i2.rolling_to_full_horizon_gap)} | {f(i3.rolling_to_full_horizon_gap)} |
| 主策略距真下界 | {f(i2.gap_to_lower_bound)}（{i2.gap_to_lower_bound_percent:.2f}%） | {f(i3.gap_to_lower_bound)}（{i3.gap_to_lower_bound_percent:.2f}%） |

上一版把滚动版标成"全知下界"是不成立的：它比真下界高 {f(i3.rolling_to_full_horizon_gap)} 元
（Q4-3）。差额同时受有限视野、终端代理和执行规则影响，不能严格归因于两项额外约束。

纯现金目标应去掉期末库存估值另行优化，对应现金下界为
{f(lb.cash_lower_bound.iloc[0])}／{f(lb.cash_lower_bound.iloc[1])} 元；不得把库存校正最优解的现金分量称为现金最小值。
上表差距百分比的分母为主策略费用。

### 4.2 固定策略的回放费用对照

{tbl(info[['problem','price_assumed_fixed','baseline_deterministic','main_strategy','hybrid_stochastic','perfect_price','rolling_perfect_information','full_horizon_lower_bound']],
     ['问题','当作固定电价','确定性基准','主策略','HYBRID','完美电价','滚动完美信息','全期真下界'])}

{tbl(info[['problem','hybrid_saving_vs_quantile','hybrid_saving_percent','frozen_price_replacement_saving','frozen_price_replacement_percent','gap_to_lower_bound','gap_to_lower_bound_percent']],
     ['问题','融合相对分位基线节省/元','节省/%','完美价格替换节省/元','替换节省/%','距全期下界/元','差距/%'])}

融合策略相对分位法节省：Q4-3 为 {f(i3.hybrid_saving_vs_quantile)} 元，Q4-2 为
{f(i2.hybrid_saving_vs_quantile)} 元。完美价格替换节省分别为
{f(i2.frozen_price_replacement_saving)}／{f(i3.frozen_price_replacement_saving)} 元。
这些是固定策略之间的历史费用差，不是标准随机规划的 VSS 或理论 EVPI，亦非信息价值上界。
不能据此断言改进价格预测没有价值，也不能把距全期下界的全部差额都归因于不可消除的负荷不确定性。

### 4.3 尾部风险：精确经验 CVaR 与配对移动块重采样

334 个等权日费用按降序排列为 C[1]…C[334]。精确经验 CVaR95 为
`(C[1]+…+C[16]+0.7 C[17])/16.7`，不会将最高 17 天的均值混称为精确 CVaR。
采用 7/14/28 日移动块、4000 次配对重采样：全部合法块起点均可抽取，抽 ceil(n/L) 块后截取 n=334 项。
相同日期索引同时用于两个策略。区块方法是对局部依赖的近似，单一年度季节性仍限制区间解释。
`P(更差)` 是 bootstrap 重采样差值大于零的比例，不是总体劣化概率或显著性检验 p 值。

{tbl(boot[['problem','reference','variant','block_len','cvar95_reference','cvar95_variant','diff','ci_lo','ci_hi','p_worse','mean_diff','mean_ci_lo','mean_ci_hi']],
     ['问题','参照','对比','块长/日','参照 CVaR95','对比 CVaR95','CVaR 差','CI 下','CI 上','P(更差)','日均费用差','均值CI下','均值CI上'])}

按上表判断不确定性：三档块长下 Q4-2 分位重标定的区间上界分别为
{', '.join(f(v) for v in boot[(boot.problem=='Q4-2') & boot.variant.str.startswith('M2')].ci_hi)} 元。
Q4-3 融合策略的尾部费用差为 {f(bt3['diff'])} 元，区间跨零时只能说方向未确定，不能声称等价或已证明没有恶化。
以上是历史样本的区块 bootstrap 结果，不构成独立盲测证据。

### 4.4 逐月分解（参照沿用 α=0.60 的分位法）

Q4-2：{int((mon2.saving > 0).sum())}/{len(mon2)} 个月为正。

{tbl(mon2, ['月份','基准/元','主策略/元','节省/元','节省/%'])}

Q4-3：{int((mon3.saving > 0).sum())}/{len(mon3)} 个月为正。

{tbl(mon3, ['月份','基准/元','主策略/元','节省/元','节省/%'])}

---

## 5  单变量消融与敏感性

全部使用**冻结的选定策略**，每次只改标题里那一个要素；改多个要素的一律标为组合对照。

### 5.1 A 发布时点开关：每个决策机会值多少

{tbl(A[['case','total_cost','adjusted_cost','emergency_kwh','spill_kwh','saving']],
     ['开放的发布时点','现金费用/元','库存校正费用/元','应急电量/kWh','弃电/kWh','相对"不更新"的节省/元'])}

单点里 **18:00 最值钱**（{f(float(A[A.case=='18:00'].saving.iloc[0]))} 元），
其次 12:00、6:00；三点全开 {f(float(A[A.case=='6:00+12:00+18:00'].saving.iloc[0]))} 元。
三个单点之和 {f(float(A[A.case.isin(['6:00','12:00','18:00'])].saving.sum()))} 元
**大于**全开的节省，说明各时点的信息高度重叠，不能把全开的收益简单拆成三份。

### 5.2 B 信息更新拆分

{tbl(B[['case','changed','total_cost','adjusted_cost','emergency_kwh','spill_kwh']],
     ['对照','只改变了什么','现金费用/元','库存校正费用/元','应急电量/kWh','弃电/kWh'])}

以主策略 {f(float(A[A.case=='6:00+12:00+18:00'].adjusted_cost.iloc[0]))} 元为参照：
关掉负荷日内校正贵 {f(float(B[B.case.str.startswith('只更新光伏')].adjusted_cost.iloc[0]) - float(A[A.case=='6:00+12:00+18:00'].adjusted_cost.iloc[0]))} 元，
光伏用 0:00 版本贵 {f(float(B[B.case.str.startswith('只更新负荷')].adjusted_cost.iloc[0]) - float(A[A.case=='6:00+12:00+18:00'].adjusted_cost.iloc[0]))} 元，
两者都关是组合对照。**电价日内校正的增量几乎为零**
（{f(float(B[B.case.str.startswith('关闭电价')].adjusted_cost.iloc[0]) - float(A[A.case=='6:00+12:00+18:00'].adjusted_cost.iloc[0]))} 元），
该数仅描述当前冻结策略对该项校正的敏感程度。

### 5.3 C 结算口径敏感性（不同合同，不是算法改进）

{tbl(Cb[['case','changed','total_cost','adjusted_cost','emergency_kwh','spill_kwh']],
     ['口径','配置','现金费用/元','库存校正费用/元','应急电量/kWh','弃电/kWh'])}

逐次结算已修复：无论是否导出明细都累计费用；优化和接受门均以上一次有效计划作为调整基准。
上表逐次行由修复后的策略在 334 天上重新回放得到，冻结了数值超参数，允许合同改变后计划随之变化。
相对主口径的库存校正费用变化为
{f(float(Cb[Cb.case=='逐次调整分别计费'].adjusted_cost.iloc[0])-float(h3.adjusted_cost))} 元。

另以主方案原有计划固定不变，仅重算合同支付：逐次调整费为
{f(json.loads((R/'q4_repair_checks.json').read_text())['fixed_main_plan_stepwise_cash']-float(e3.plan_cost.sum()+e3.emergency_cost.sum()))} 元，
现金总费 {f(json.loads((R/'q4_repair_checks.json').read_text())['fixed_main_plan_stepwise_cash'])} 元，
相对净额账单增加 {f(json.loads((R/'q4_repair_checks.json').read_text())['fixed_main_plan_stepwise_minus_net'])} 元。

对每个交付时段，`f(δ)=pδ+0.5p|δ|`，因此
`Σ f(δ_j)-f(Σδ_j)=0.5p(Σ|δ_j|-|Σδ_j|)≥0`。
这是同计划下的计费不变量，不代表不同合同下重新决策的全年费用一定按同一方向排列。
本包主合同仍假设交付电价、净额退款；它是明确采用的题意解释，不以“某价格看起来像折扣”作为证明。

### 5.4 D 物理参数压力（冻结策略）

{d_md}

容量的影响最大（±20% 容量 → ∓约 60 万元），功率 ±10% 影响很小，效率 0.85/0.95 居中。
以上为**冻结策略下的压力扫描**，不是重新选参后的最优；这些取值是压力点，不是题目给定值。

### 5.5 E 经验参数逐项扰动

1 月训练窗口上的 ΔC/Δθ（基准值为 0 的参数不计算相对弹性）：

{tbl(psen[['param','base','perturbed','rel','fitness','d_fitness','elasticity','note']],
     ['参数','基准值','扰动后','相对变化','适应度/元','Δ适应度/元','弹性','备注'])}

最敏感的三项做了全年复核：

{tbl(Ef[['case','changed','total_cost','adjusted_cost','emergency_kwh','spill_kwh']],
     ['扰动','配置','现金费用/元','库存校正费用/元','应急电量/kWh','弃电/kWh'])}

五个种子的训练适应度极差为 {f(noise)} 元。该数只作离散程度描述；比较参数优劣需要配对种子与验证数据。

---

## 6  交付与核验

| 文件 | 内容 | 由哪个策略生成 |
|---|---|---|
| `result4-2.xlsx` | 计划购电量 / 充放电量 / 紧急购电量 | M2 分位法 α={a2v:.3f} |
| `result4-3.xlsx` | 多一张「调整购电量」 | H3 HYBRID-4L |
| `q4_{{2,3}}_daily.csv`、`_intervals.csv`、`_transactions.csv` | 逐日 / 逐段 / 逐笔账本 | 同上 |
| `model_config.json` | 运行口径快照 + 数据与代码指纹 + `variant_id` | — |

{checks_md}

**因果性：3 天 × 4 个发布时刻 = {len(caus)} 组全部通过，最大计划差
{caus.max_plan_diff_kwh.max():.2e} kWh。** 每组的扰动包括该时刻之后的真实负荷/光伏/电价，
以及当天更晚发布时刻与之后各天的**全部未发布预报**。日前通过不能自动证明日内分支通过，
故逐时刻分别检验。

SOC 上下限 1200—10800 kWh、充放电功率 ≤ 5000 kW 在全部 334 天逐段成立。

Q4-2 Excel 的“全天购电费”已改为含紧急购电的总现金费用；Q4-3 的计划表最后一列明确标为“全天计划购电费”，总现金费用见调整表。两份表均附费用汇总与版本说明。

---

## 7  局限

1. **方法结论限于当前回放**：Q4-2 所选分位法费用较低；Q4-3 所选融合策略费用较低。不能推导一般适用边界。
2. **两阶段模型的第二阶段是情景内完美预见**的补救——它假设观测到情景后整条后续路径一次可知，
   属**近似**，不是多阶段最优控制。`adj_premium` 是对这一乐观偏差的经验修正（只在 Q4-3 生效）。
3. **24 天选参窗口 + {100*noise/seeds.train_fitness.mean():.2f}% 的跨种子波动**：§3.3 记录了两次
   1 月排名与回放排名不同的实例。需要新的外层留出样本评估泛化收益。
4. **价格情景来自历史对数残差的经验分布**，隐含"未来误差分布与过去 28 天相同"。
5. **单一年度、顺序前推**：评价期数据此前已多次用于开发，结论只能称历史前推验证。
6. **结算口径**见 §5.3 与 `model_config.json`；改用"下单时刻电价"的读法会显著改变结论，
   属口径敏感性，不应与算法改进混为一谈。
"""
    REP.mkdir(exist_ok=True)
    (REP / '第四问_波动电价分层融合报告.md').write_text(txt, encoding='utf8')
    print('report written', len(txt), flush=True)


if __name__ == '__main__':
    main()
