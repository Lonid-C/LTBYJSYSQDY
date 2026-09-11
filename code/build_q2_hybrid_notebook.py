"""由受版本控制的模型源码生成完整 Q2 融合实验 notebook。"""
from pathlib import Path
import nbformat as nb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code/problem2_hybrid.ipynb"
SOURCE = (ROOT / "code/q2_hybrid.py").read_text(encoding="utf-8")
SOURCE = SOURCE.split('\nif __name__ == "__main__":', 1)[0]


def md(text):
    return nb.v4.new_markdown_cell(text.strip())


def code(text):
    cell = nb.v4.new_code_cell(text.strip())
    cell.metadata["colab"] = {"base_uri": "https://localhost:8080/"}
    return cell


book = nb.v4.new_notebook()
book.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "accelerator": "CPU",
    "cumcm": {"section": "Q2 hybrid planning", "no_result2_overwrite": True},
})
book.cells = [
md(r'''
# C题问题二：日前购电—日内因果储能融合模型及消融实验

本 Notebook 是现有 Q2 的增量补充，不覆盖正式 `result2.xlsx`，也不删除已有预测消融和 P0—P6 规划实验。它重点修正原模型中“每天 0:00 同时固定全天购电和全部储能动作、每天实际 SOC 强制回到 6000 kWh”的偏强假设。

信息时序为：每天 0:00 只锁定全天计划购电量与参考 SOC；每个 10 分钟时槽开始后，只使用当前已经观测到的实际净负荷和实际 SOC 决定充电、放电、弃电和紧急购电。任何函数都不得读取后续时槽的真实值。
'''),
md(r'''
## 1. 问题分析与模型选择

日前购电单价已知，临时购电单价是同一时槽正常电价的 5 倍。只使用点预测会低估误差尾部，只使用完全固定的两阶段 SAA 又限制了储能的实时纠偏能力。因此本节将“日前不可撤销决策”和“日内可适应决策”分离：

- 日前变量：$q_{d,t}$，即计划购电量；
- 日内变量：$c_{d,t},v_{d,t},z_{d,t},s_{d,t}$，分别为充电、放电、紧急购电和弃电；
- 状态变量：$E_{d,t}$，为储能电量；
- 参考变量：$E^{\rm ref}_{d,t}$，由日初 LP 给出，只用于构造保留线，不等于实际 SOC。

在预测曲线给定时，目标函数、功率平衡、SOC 递推、功率和容量边界均为线性，因此日初问题使用 LP，可以直接获得有限模型的全局最优解。GA、PSO 不会提高线性模型的最优性证明；LSTM 等方法属于预测端，也不能代替规划端的可行性约束。
'''),
md(r'''
## 2. 单位、物理约束与参数来源

附件数据为功率 kW，采样间隔为 $\Delta t=1/6$ h。所有规划和执行变量使用 kWh：

$$N_{d,t}=\Delta t(L_{d,t}-P_{d,t}).$$

题目物理参数固定为：容量 12000 kWh，保护区间 $[1200,10800]$ kWh，最大充放电功率 5000 kW，即每槽上限 $5000/6$ kWh，充放电效率均为 0.9。紧急购电倍率 5 来自题意。

统计参数不是随意给定：残差窗口 14/21/28/42/56 天对应 2/3/4/6/8 个完整周；$\alpha$ 先在 $[0.50,0.95]$ 上以 0.05 粗搜，再在局部以 0.01 细搜；$\rho\in\{0,0.25,0.5,0.75,1\}$ 覆盖完全灵活到完全跟随参考库存；0/3/5 槽平滑用于检验局部分位数噪声，其中三点核为 $(1,2,1)/4$，五点核为 $(1,4,6,4,1)/16$，均为对称且归一化的二项式核，不引入相位移动。每月配置仅由月初之前 56 个已结束日期选择，11—12 月冻结 10 月规则。

$\alpha=0.8$ 仅是经济学锚点。在忽略储能和跨时槽耦合的单时槽问题

$$\min_q\;p_tq+5p_t\,\mathbb E[(N_t-q)_+]$$

中，临界分位数满足 $F_N(q)=1-p_t/(5p_t)=0.8$。完整系统存在 SOC 耦合，因此最终 $\alpha$ 必须由历史端到端成本选择，不能预先宣称 0.8 最优。
'''),
md(r'''
## 3. 日初参考规划 LP

对点预测、残差分位预测或条件残差分位预测得到的风险净负荷 $\widetilde N_{d,t}$，求解

$$\min \sum_{t=1}^{144}p_tq_{d,t},$$

满足

$$q_{d,t}+v^{\rm ref}_{d,t}-c^{\rm ref}_{d,t}-s^{\rm ref}_{d,t}=\widetilde N_{d,t},$$

$$E^{\rm ref}_{d,t+1}=E^{\rm ref}_{d,t}+\eta_c c^{\rm ref}_{d,t}-\frac{v^{\rm ref}_{d,t}}{\eta_d},$$

$$1200\le E^{\rm ref}_{d,t}\le10800,$$

$$0\le c^{\rm ref}_{d,t},v^{\rm ref}_{d,t}\le5000/6,$$

$$q_{d,t},s^{\rm ref}_{d,t}\ge0.$$

参考轨迹日末取 6000 kWh，这是规划边界假设，不强迫真实执行日末回到 6000。实际 SOC 跨日传递。
'''),
md(r'''
## 4. 风险净负荷与日内因果执行

普通残差分位策略为

$$\widetilde N_{d,t}(\alpha,W)=\widehat N_{d,t}+Q_\alpha\{N_{j,t}-\widehat N_{j,t}:d-W\le j<d\}.$$

条件策略从过去最多 120 天中，依次使用季节周期、星期周期及预测负荷/光伏日总电量选择相似日；候选特征的尺度只用历史池估计。

日内保留线定义为

$$R_{d,t}(\rho)=E_{\min}+\rho(E^{\rm ref}_{d,t+1}-E_{\min}).$$

若计划购电高于当前实际净负荷，则在功率和容量范围内充电，剩余弃电；若出现缺口，则

$$v_{d,t}=\min\{-\delta_{d,t},P_{\max}\Delta t,
\eta_d[E_{d,t}-R_{d,t}(\rho)]_+\},$$

$$z_{d,t}=-\delta_{d,t}-v_{d,t},$$

其中 $\delta_{d,t}=q_{d,t}-N_{d,t}$。紧急电量只补当前供电缺口，不用于事后给电池充电。
'''),
md(r'''
## 5. H0—H5 机制消融

|编号|模型|唯一主要变化|回答的问题|
|---|---|---|---|
|H0|原加权 SAA 固定执行|购电和储能动作均在日初锁定，日循环 SOC|旧模型基准|
|H1-fixedQ|沿用 H0 的购电与参考轨迹，改为因果执行|只改变执行权限|日内纠偏的价值|
|H1-replan|SAA 按实际日初 SOC 重算参考计划|再加入跨日状态反馈|连续 SOC 的价值|
|H2|点预测 LP + 因果执行|不加残差安全余量|不确定性建模是否必要|
|H3|残差分位 LP + 因果执行|学习 $\alpha,W,\rho$ 与平滑|主要融合候选|
|H4|条件残差分位 LP + 因果执行|相似日替代纯时间窗口|条件选场景是否有用|
|H5|使用真实净负荷的连续期 Oracle LP|完美信息|理论下界与预测价值空间|

H0 与 H1-fixedQ 使用相同的 $q$，用于识别执行灵活性的贡献。H2—H4 共用同一个因果执行器，防止把执行差异误归因于预测风险模型。H5 为每个策略匹配其实际年末 SOC；另报告库存修正成本，不制造在线策略必能精确回到 6000 的假象。
'''),
md(r'''
## 6. 评价体系与验收规则

经济指标包括计划费、紧急费、总成本、计划与紧急购电量、弃电量、紧急时槽/日期/事件频率、日成本 CVaR$_{90}$/CVaR$_{95}$、最大日成本、最终 SOC、库存修正成本和 Oracle 差距。

库存修正成本为

$$C_{\rm adj}=C_{\rm actual}-\frac{p_{\min}}{\eta_c}(E_{\rm end}-E_{\rm start}).$$

该指标只用于校正不同年末库存，不替代真实支付账单。采用 7 日移动块 Bootstrap、2000 次、随机种子 2026 计算日成本差的 95% 区间。开发期为 6—10 月；11—12 月只检查 10 月冻结规则。差异小于 0.1% 时优先理论锚点附近、较短窗口和更简单结构。

逐时检查功率平衡、SOC 递推、容量和功率边界、非负性与同时充放电；LP 还检查原始矩阵残差和对偶间隙。因果性通过“任意扰动未来实际值，当前以前动作保持不变”的前缀测试验证。
'''),
md("## 7. 完整可复现模型源码\n\n下方单元格嵌入本次实际运行的全部模型源码，避免 Notebook 文字、公式与外部脚本版本不一致。"),
code("from pathlib import Path\nimport sys\nPROJECT_ROOT = Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path.cwd().parent\nif str(PROJECT_ROOT / 'code') not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT / 'code'))\nprint('PROJECT_ROOT =', PROJECT_ROOT)"),
code(SOURCE),
md("## 8. Colab 小规模试运行、完整实验与检查点\n\n先执行 3 日试运行和因果前缀测试；通过后自动进行逐月选参、334 日回测、匹配终态 Oracle、Bootstrap 和图表生成。所有阶段写入 `results/q2_hybrid_v1` 检查点，正式结果表不被覆盖。"),
code("ROOT = PROJECT_ROOT\ns, summary, monthly, selection, oracle, boot, figure_manifest, manifest = run_full(ROOT)"),
md("## 9. 实验结果、图表含义与数据驱动结论\n\n本单元格只读取刚刚实际计算的结果生成结论，不提前写入任何‘某模型最好’的判断。"),
code("render_results(s, summary, monthly, selection, oracle, boot, figure_manifest, manifest)"),
]

for i, cell in enumerate(book.cells):
    cell.id = f"q2-hybrid-{i:02d}"
nb.validate(book)
nb.write(book, OUT)
print(OUT)
