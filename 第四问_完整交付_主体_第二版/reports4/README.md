# 第四问：波动电价下的分层融合（HYBRID-4L / Q4）

**variant_id：`Q4-e2549ddb99`**　口径与指纹见 `results4/model_config.json`（运行设置、选参及文件指纹快照）

**先读** `reports4/第四问_波动电价分层融合报告.md`。

## 一句话结论

两个口径的答案不同，报告如实并列（样本外 2025-02-01 — 12-31，334 天，库存校正费用，元）：

| | Q4-2（无日内调整） | Q4-3（0/6/12/18） |
|---|---:|---:|
| 沿用第三问 α=0.60 的分位法 | 15,585,898 | 14,145,416 |
| **主策略** | **分位法 α=0.675：15,127,540（−2.94%）** | **HYBRID-4L：14,028,199（−0.83%）** |
| HYBRID-4L | 15,152,826（+0.17%，未胜出） | 即主策略 |
| 全期完美信息真下界 | 12,841,236 → 真下界 12,782,824 | 12,782,827 |

原因是结构性的：两阶段随机规划的核心优势是给"发布时刻可调整的购电决策"正确定价，而 Q4-2 没有这类日内调整通道。
注意：Q4-2 **仍有 recourse**（逐情景的充放电、弃电、库存与紧急购电变量），缺的是**日内增减购通道**。
**有日内调整通道时用随机规划、没有时用一维标定的分位法**——这条适用边界只适用于本文的数据与预算，是主要结论之一。

## 本版（审阅意见落地）相对上一版的改动

1. **修掉终端参考价泄漏**：`vE_ref` 原取 2—12 月真实电价最低值均值，却进入 1 月训练适应度；
   改用附件 1 给定分时电价最低值/η = 0.412556（事前已知、全方法共用）。全链重跑。
2. **"全知下界"改正**：原 W2/W3 仍跑在滚动视野 + 终端代理里，不构成下界；改称
   **滚动视野完美信息对照**，另解 334 天单个 LP（288,576 变量）作为**真下界**。
   滚动结构的额外约束代价实测 4.8—5.8 万元。
3. **因果性扩到 3 天 × 4 个发布时刻 = 12 组**，扰动含尚未发布的预报，全部 0.0 kWh 通过。
4. **口径冻结与可追溯**：`model_config.json` + `variant_id`；Q4-3 输出 12,586 条逐笔交易账本。
5. **补齐消融与敏感性**：发布时点 8 种组合、信息更新拆分、结算口径、物理参数压力、
   六个经验参数逐项扰动、情景削减多种子（给出 0.37% 的噪声地板）。
6. **统计升级**：CVaR95 用 7/14/28 三档移动块重采样，并单列日均费用差的同法区间。

## 交付件

| 文件 | 内容 | 由哪个策略生成 |
|---|---|---|
| `result4-2.xlsx` | 计划购电量 / 充放电量 / 紧急购电量（3 表） | 分位法 α=0.675 |
| `result4-3.xlsx` | 多一张「调整购电量」（4 表） | HYBRID-4L |
| `results4/q4_2_selected_table{1,2,3}.csv` | 论文表 1/2/3（四个指定日期），Q4-2 | |
| `results4/q4_3_selected_table{1,2,3}.csv` | 论文表 1/2/3（四个指定日期），Q4-3 | |

## 逐文件用途索引

| 文件 | 用途 |
|---|---|
| `results4/model_config.json` | **口径单一真源**：结算/信息/储能/评价/数值约定 + 数据与代码指纹 + variant_id |
| `results4/price_bias_cv.csv` | 价格预报第二段超参网格（日内偏差校正）的 1 月 CV |
| `results4/price_data_audit.json` | 附件4 电价的数据指纹与结构统计 |
| `results4/price_forecast_cv.csv` | 价格预报第一段超参网格（日前形状）的 1 月 CV |
| `results4/price_hp.json` | 最终选中的价格预报超参数 |
| `results4/q4_2_alpha.log` | 运行日志 |
| `results4/q4_2_daily.csv` | Q4-2 主策略逐日账本 |
| `results4/q4_2_intervals.csv` | Q4-2 逐段账本（result4-2.xlsx 的数据来源） |
| `results4/q4_2_structural.log` | 运行日志 |
| `results4/q4_2_transactions.csv` | Q4-2 逐笔交易账本（0 条：该口径无调整通道） |
| `results4/q4_2_wide.log` | 运行日志 |
| `results4/q4_3_daily.csv` | Q4-3 主策略逐日账本 |
| `results4/q4_3_decisions.csv` | Q4-3 每次日内调整的估计收益、是否接受、增减购量 |
| `results4/q4_3_intervals.csv` | Q4-3 逐段账本（result4-3.xlsx 的数据来源） |
| `results4/q4_3_transactions.csv` | Q4-3 逐笔交易账本：每次调整的 old/new/delta/交付价/发布价/归因费 |
| `results4/q4_ablation.log` | 运行日志 |
| `results4/q4_ablation_sensitivity.csv` | A 发布时点 / B 信息更新 / C 结算口径 / D 物理参数 / E 经验参数（全年） |
| `results4/q4_alpha.log` | 运行日志 |
| `results4/q4_alpha_calibration.csv` | 分位法 α 的一维标定网格（两问） |
| `results4/q4_boundary.log` | 运行日志 |
| `results4/q4_boundary_extension.csv` | 边界延展逐轮记录（相对量程 1% 判据 + 结构性下界标注） |
| `results4/q4_causality_checks.csv` | **因果性 3 天 × 4 发布时刻 = 12 组**的逐组结果 |
| `results4/q4_chain1.log` | 运行日志 |
| `results4/q4_chain2.log` | 运行日志 |
| `results4/q4_cvar_bootstrap.csv` | 尾部风险 CVaR95 的 7/14/28 日移动块重采样 |
| `results4/q4_experiment.log` | 运行日志 |
| `results4/q4_experiment_summary.json` | 主实验汇总（含 variant_id） |
| `results4/q4_export.log` | 运行日志 |
| `results4/q4_finalize.log` | 运行日志 |
| `results4/q4_full_horizon_bound.csv` | **全期完美信息真下界**（334 天单个 LP，288,576 变量） |
| `results4/q4_information_value.csv` | 回放费用差 / 冻结策略下完美价格替换收益 / 滚动约束代价 / 距真下界的拆分（**不是** VSS/EVPI） |
| `results4/q4_monthly_Q4-2.csv` | Q4-2 逐月节省分解 |
| `results4/q4_monthly_Q4-3.csv` | Q4-3 逐月节省分解 |
| `results4/q4_out_of_sample.csv` | 样本外全年各方法对照总表 |
| `results4/q4_param_sensitivity_train.csv` | 六个经验参数各 ±20% 的 1 月 ΔC/Δθ 与弹性 |
| `results4/q4_saa_convergence.csv` | 情景数 S 的 1 月训练适应度收敛（两问） |
| `results4/q4_scenario_seed_stability.csv` | 情景削减多种子稳定性、有效情景数 1/Σw²、最小权重 |
| `results4/q4_search_convergence.csv` | L3 参数寻优收敛记录 |
| `results4/q4_search_convergence_q4_2_structural.csv` | Q4-2 结构受限（shrink=1）寻优记录 |
| `results4/q4_search_convergence_q4_2_wide.csv` | Q4-2 加宽框寻优记录（已弃用的尝试） |
| `results4/q4_selected_policy.json` | 最终选参（含被弃用尝试与无约束最优的留痕） |
| `results4/q4_terminal_curve.csv` | 一次 Bellman 回代得到的归一化成本曲线 C(E) |
| `results4/q4_terminal_segments.csv` | 压成 8 段后的非增边际价值形状 |
| `results4/q4_terminal_shape.npz` | 终端库存价值的归一化分段形状与端点（二进制） |
| `results4/q4_verification.json` | 独立复核（账本重算/物理约束/因果性/工作表一致性） |
| `results4/warmup_q4_2.csv` | Q4-2 的 1 月预热轨迹（给出 2/1 初始库存） |
| `results4/warmup_q4_3.csv` | Q4-3 的 1 月预热轨迹 |
| `results4/q4_daily_<问题>_<前缀>.csv` | 各对照方法的逐日账本（B0/M2/B1/H2/H3/U*/W*/A3/X3） |

| 图 | 内容 |
|---|---|
| `figures4/Q4_电价结构与预报.png\|pdf` | 附件4 日内形状、价格乘子波动、三种预报器精度 |
| `figures4/Q4_情景数收敛.png\|pdf` | 情景数 S 的 SAA 收敛 |
| `figures4/Q4_分位标定.png\|pdf` | 分位法 α 一维标定（所测试网格内的最优点位于网格内部） |
| `figures4/Q4_寻优收敛.png\|pdf` | GA/Nelder-Mead 收敛与对照基线 |
| `figures4/Q4_样本外对比.png\|pdf` | 样本外全年各方法费用 |
| `figures4/Q4_逐月节省.png\|pdf` | 逐月节省分解 |
| `figures4/Q4_费用结构.png\|pdf` | 计划/增购/减购/紧急四项费用结构 |
| `figures4/Q4_信息价值阶梯.png\|pdf` | 七档费用阶梯（含滚动完美信息与全期真下界）；不使用 VSS/EVPI 命名 |
| `figures4/Q4_消融矩阵.png\|pdf` | A 发布时点开关 + B 信息更新拆分 |
| `figures4/Q4_敏感性.png\|pdf` | C 结算口径 + D 物理参数 + E 经验参数 |

## 复现

```bash
python code/run_q4_experiment.py     # 价格两段选参、SAA 扫描、L3 寻优、样本外（2 核约 57 分钟）
python code/q4_alpha_calibrate.py    # 分位法 α 一维标定（两问）
python code/q4_boundary_extend.py    # 边界延展（可带问题名强制再确认一轮）
python code/q4_finalize.py           # 按当前选参重跑全部分支并重建所有下游账本
python code/q4_lower_bound.py        # 全期完美信息真下界
python code/q4_export.py             # result4-2/4-3.xlsx 与指定日期表
python code/q4_verify.py             # 账本重算 / 物理约束 / 四时刻因果性 / 工作表一致性
python code/q4_ablation_sensitivity.py   # 消融与敏感性套件（约 2 小时，可断点续跑）
python code/q4_report.py             # 十张图
python code/q4_writeup.py            # 报告
```

> **每次选参变动后必须跑 `q4_finalize.py`**，它按当前选参重建全部下游账本，
> 避免"只更新了一部分"造成的不自洽；`q4_lower_bound.py` 需在 finalize 之后跑，
> 否则信息价值表里的真下界列会是空的。

## 口径

决策端严格因果；账单端一律用附件 4 的**实际电价**，调整电量按**交付时段**电价计费。
参数取值只用 1 月三个滚动验证块标定，冻结后 2—12 月顺序前推回放；参数范围与模型结构曾结合历史回放修订，故**非独立盲测**。
完整约定见 `results4/model_config.json`。
