# 方案

要依次调用这些 skill，按照里面要求完成任务。

用户偏好：
- 排版引擎：本次不生成完整论文，暂不指定
- 竞赛类型：2026 年高教社杯全国大学生数学建模竞赛（国赛）
- 论文语言：中文
- 子问题数量：题面共 4 个顶层问题；本次仅解答问题 1、问题 2

## 本次目标

在既有问题 1、问题 2 模型上增量补齐论文证据链：完成全年/日内 EDA、LP 选择依据、baseline、可行性与最优性评价、端到端模型比较、参数敏感性及月/季度稳健性；保留并复核附件 5 的 `result1.xlsx` 与 `result2.xlsx`。

## workflow

| step | skills | 本次范围 | 主要产物 |
| --- | --- | --- | --- |
| 1 | 赛题分析与建模设计 - `2analysis-modeling` | 问题 1、问题 2 | `reports/ANALYSIS_MODELING_REPORT.md` |
| 2 | 编程实现和图表生成 - `3coding-visual` | 问题 1、问题 2；全部代码采用 Jupyter Notebook；补充 EDA/Evaluation/敏感性 | `code/problem1.ipynb`、`code/problem2.ipynb`、`results/`、`reports/RESULTS_REPORT.md`、`figures/*.pdf`、`result1.xlsx`、`result2.xlsx` |
| 3 | 流程与架构图绘制 - `4drawio` | 本次不执行 | `figures/*.drawio`、`reports/DRAWIO_REPORT.md` |
| 4 | 竞赛论文撰写 - `5writing` | 不生成整篇论文；仅输出可直接粘贴的 Q1/Q2 论文模块 | `reports/PAPER_MODULES_Q1_Q2.md` |
| 5 | 验证和验收 - `6verity` | 不启动完整论文验收；仅对本次模型、代码与结果文件做专项校验 | `reports/VERIFY_REPORT.md`（若后续扩展为完整论文） |

## 建模方向

- 问题 1：以 10 分钟为离散时间步，建立含购电、充电、放电和储能状态的线性规划；最小化全天购电费并满足功率平衡、储能上下限、充放电功率、效率与日首尾储电量相等约束。
- 问题 2：将每天 0:00 的计划看作面向不确定负载与光伏的日 ahead 决策。比较确定性基线与基于历史同类时刻误差分布的风险控制方案，用计划购电费与 5 倍紧急购电费的期望/回测成本选择策略；全年逐日滚动更新储能状态并回代实际数据计算紧急购电。
- 问题 1 论文链：EDA → 净负荷分析 → LP 选择依据 → 模型建立 → 求解 → Baseline → Evaluation → 敏感性。
- 问题 2 论文链：数据质量 → 全年 EDA → 周期性证据 → Weekly/Trend → 残差情景 → 两阶段 SAA → 全年滚动回测 → 策略/Oracle 对照 → 端到端选参 → 月/季稳健性 → 验收。
- 对“90% 充放电效率”的解释、光伏余电能否弃电/充电、问题 2 是否允许利用当天实际值等关键歧义先做敏感性预检，再固定实现口径。

## 项目目录结构

```text
.
├── plan.md
├── todo.md
├── reports/
│   ├── ANALYSIS_MODELING_REPORT.md
│   └── RESULTS_REPORT.md
├── code/
│   ├── problem1.ipynb
│   └── problem2.ipynb
├── code/outputs/
├── results/
├── figures/
│   └── *.pdf
├── result1.xlsx
└── result2.xlsx
```

## 风险控制

- 题面公式、时间边界和单位同时用 PDF 文本提取与页面渲染核对。
- 读取全部工作表并检查时间网格、日期连续性、缺失、重复、异常、公式和单位。
- 任何优化解均逐时段回代功率平衡、储能递推、上下界、功率限额和日首尾条件。
- 对问题 2 严格限制每天 0:00 可使用的信息，避免以当天实际负载/光伏反推计划造成数据泄漏。
- 所有论文可引用数值均写入机器可读结果文件，并与 Excel 模板逐格核对。
