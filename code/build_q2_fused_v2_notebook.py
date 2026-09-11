"""生成 Q2 参数来源增强 V2 全链路 Colab notebook。"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code/problem2_fused_v2.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"name": OUT.name, "provenance": [], "gpuType": "A100"},
})


def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# C题问题二：参数来源增强 V2 全年实验

本笔记本在现有 Team-A 预测—储能调度方案上补齐参数来源、时间因果性、后置验证和终端 SOC 核算。本次不读取 V1 的 daily/summary/cache 结果，证据链为

\[
\text{数据审计}\to\text{因果预测与残差}\to\text{风险净负荷}
\to\text{日前 LP}\to\text{日内因果执行}
\to\text{滚动搜索/独立验收}\to\text{全年回测}\to\text{稳健性检验}.
\]

正式 `result2.xlsx` 不被覆盖；代码和原始数据 SHA256 写入运行清单。
""")

md(r"""
## 1. 问题分析、数据清洗和信息集

附件2给出365天、每天144个10分钟时槽的负荷 $L_{d,t}$ 和光伏 $P_{d,t}$，$\Delta t=1/6$ h。规划与执行统一使用

\[
N_{d,t}=(L_{d,t}-P_{d,t})\Delta t\quad(\mathrm{kWh/slot}).
\]

每天0:00的决策只能使用 $\mathcal I_d=\{L_{j,t},P_{j,t}:j<d\}$ 及已知电价。清洗检查表头、日期连续性、144时槽完整性、缺失/非有限值和非负性；不主观删除真实尖峰。
""")

code("""
from pathlib import Path
import os, sys, json, subprocess
import numpy as np, pandas as pd
from IPython.display import display, Markdown, Image
PROJECT=Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path.cwd()
os.chdir(PROJECT); sys.path.insert(0,str(PROJECT/'code/q2_v2'))
print('project =',PROJECT); print('python =',sys.version)
""")

code("""
from utils import load_data, data_audit, BATTERY, DT, T
data=load_data(); audit=data_audit(data)
display(pd.DataFrame([{'field':k,**v} for k,v in audit.items() if isinstance(v,dict) and 'shape' in v]))
display(pd.DataFrame([audit['battery']]))
print(json.dumps({'source_directory':audit['source_directory'],
                  'source_sha256':audit['source_sha256'],
                  'time_convention':audit['time_convention']},ensure_ascii=False,indent=2))
""")

md(r"""
## 2. 因果预测、残差与风险输入

保留 V2 的 Team-A 预测主干：负荷成员为周期外推与带星期虚拟变量的趋势回归；光伏成员为5日均值、指数均值与晴空形状分解。成员权重仅用过去10个已结束日的误差学习：

\[
w_{m,d}=\frac{\exp(-8\tilde e_{m,d})}{\sum_j\exp(-8\tilde e_{j,d})},\qquad
\tilde e_{m,d}=e_{m,d}/\bar e_d.
\]

记 $\widehat N_{d,t}=(\widehat L_{d,t}-\widehat P_{d,t})\Delta t$，残差 $r_{j,t}=N_{j,t}-\widehat N_{j,t}$。规划输入为

\[
\widetilde N_{d,t}=\widehat N_{d,t}+S_s\!\left(Q_\alpha\{r_{j,t}:d-W\le j<d\}\right),
\]

其中 $S_s$ 是0/3/5点对称平滑。条件残差只用历史内的季节、星期和当日预测日总量选相似日，再以 $b\in[0,0.5]$ 与全局残差收缩，防止小样本条件分位数完全支配结果。
""")

md(r"""
## 3. 日前线性规划与日内执行

时槽 $t$ 的决策变量为计划购电 $q_t$、充电 $c_t$、放电 $v_t$、弃电 $s_t$ 和时槽末 SOC $E_t$。

\[
\min \sum_{t=1}^{144}p_tq_t,
\qquad q_t+v_t-c_t-s_t=\widetilde N_{d,t},
\]
\[
E_t=E_{t-1}+\eta_cc_t-v_t/\eta_d,
\quad 0\le c_t,v_t\le5000\Delta t,
\]
\[
q_t,s_t\ge0,\quad1200\le E_t\le10800,
\quad E_0=E_d^{actual},\quad E_{144}=E_{target}.
\]

$\eta_c=\eta_d=0.9$。目标、平衡、SOC递推和边界都是线性的，因此 HiGHS LP 能对当前有限问题提供全局最优解和可行性证据，无需用 GA/PSO 替代。LSTM是预测器，也不是此线性调度问题的求解器。

实际执行每个时槽只读当前 $N_{d,t}$，缺电且储能不能再放时以 $z_t\ge0$ 紧急购电，实现成本是
\[
C_d=\sum_t p_tq_t+5p_tz_t.
\]
""")

md(r"""
## 4. 参数来源、`near()` 与 CVaR 修正

参数明确分成题设物理量、历史数据学习量、事前协议和风险/经济口径。每月部署前用42天搜索、14天独立成本验收和1天embargo。

原 `score <= best*(1+0.1%)` 的带宽没有统计来源，已删除。现以搜索期最低费用候选 $m^*$ 为参考，对 $\delta_d=C_{m,d}-C_{m^*,d}$ 做7日循环块、2000次 Bootstrap；只有当 $\bar\delta$ 的90%区间下界 $\le0$ 时，候选才进入简约性打破平局集。这只是搜索正则化，不等于证明两模型等价。

14天下 $14(1-0.95)=0.7<1$，精确经验 CVaR95 必然等于worst-day。因此14天只评估平均成本改善，风险非劣性改用同一截止点前56天，此时尾部质量为 $56(1-0.95)=2.8$天。
""")

code("""
from q2_fused import PARAMETER_REGISTRY
display(pd.DataFrame(PARAMETER_REGISTRY))
""")

md(r"""
## 5. 终端 SOC、库存价值和敏感性

$E_{target}=6000$ kWh 来自初始SOC和容量50%，是日前LP参考轨迹的边界假设，不是Q2明确要求实际每天回到6000。实际日末 SOC 被下一天继承。核算费用为

\[
C^{adj}=C^{raw}-v_E(E_{end}-E_{start}).
\]

主口径 $v_E=p_{min}/\eta_c$ 是用最低电价补回1 kWh电池内能量的成本；同时报告 $v_E=0$ 和 $v_E=\eta_dp_{max}$。库存价值只改核算，不改LP。对 $E_{target}\in\{1200,4800,6000,7200,10800\}$ 全年实际重跑：端点是题设保护边界，4800/7200是中心值的±10%容量扰动。
""")

md(r"""
## 6. 对照与评价体系

对照是旧GitHub的 **Ridge + H1-replan**。冻结其预测明细以保持比较器不变，但不读取V1逐日费用、SOC、summary或cache；在本次环境中重新求解21日场景、14日半衰期SAA和全年H1因果执行。

评价包括计划费、紧急费、总费用、紧急电量/时槽/日/事件频率、弃电量、CVaR90/95、最坏日、计算时间和7/14/28日块Bootstrap。分报2—12月全期、6—10月开发期和11—12月冻结检查期。

验收检查电量平衡、SOC递推/边界/跨日连续、功率上界、非负性、同时充放电、费用重算、参数更新限制和11—12月冻结。改加当日及未来观测的时间前缀检验要求当日决策输入完全不变。
""")

code("""
# 主计算：仅清空本版专属输出目录，再从头跑 user/fused/Ridge/敏感性/验收。
cmd=[sys.executable,'code/q2_v2/q2_fused.py','--stage','all']
print('RUN:', ' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
""")

md(r"""
## 7. 规划端扩展的统一场景构造

下面三项实验全部重新使用本次 V2 的点预测、残差和当月已通过验收的配置，不读取旧版
`q2_planning_v1` 结果。设历史完整日残差向量为
$\boldsymbol r_j=\boldsymbol N_j-\widehat{\boldsymbol N}_j$。全局残差池与条件相似日池分别
分配质量 $1-b$ 与 $b$，池内等权；重复日期合并后归一化为 $w_s$。场景为

\[
\boldsymbol N_d^{(s)}=\widehat{\boldsymbol N}_d+\boldsymbol r_s,
\qquad w_s>0,\quad\sum_s w_s=1.
\]

这里原风险净负荷的分位数 $\alpha$ 不再进入风险中性经验分布：$\alpha$ 是单轨迹安全裕度参数，
而SAA需要保留全部历史场景及其概率，二者不能重复保守化。条件池的 `blend`、特征和邻居数仍
继承当月已验收配置。所有完整日残差路径整体抽取，保留日内相关性。
""")

md(r"""
## 8. P4：CVaR风险前沿与对偶风险价格

两阶段 SAA 的第一阶段变量是日前计划购电 $q_t$、充放电 $c_t,v_t$ 和SOC $E_t$；场景 $s$
的第二阶段变量是紧急购电 $z_{s,t}$。风险中性目标为

\[
RP=\min_x\sum_s w_s C_s(x),\qquad
C_s(x)=\sum_t p_tq_t+5p_tz_{s,t}.
\]

采用 Rockafellar--Uryasev 线性化：引入VaR辅助变量 $\zeta$ 和超额损失 $\xi_s\ge0$，

\[
\operatorname{CVaR}_{\beta}(C)=\zeta+\frac{1}{1-\beta}\sum_sw_s\xi_s,
\quad \xi_s\ge C_s-\zeta.
\]

分别取 $\beta=0.90,0.95$。风险预算不是随机指定：先求风险中性解的CVaR上端 $U$，再求
最小CVaR解的下端 $L$，然后令 $B_k=(1-\gamma_k)U+\gamma_kL$，
$\gamma_k\in\{0,0.25,0.5,0.75,1\}$，得到完整可行前沿。预算约束的非负对偶乘子
$\lambda_{risk}$ 表示预算再收紧一元的局部边际成本；同时求
$\min\mathbb E[C]+\lambda_{risk}\operatorname{CVaR}_{\beta}(C)$ 核对KKT等价。
它不是数据学习出的用户风险偏好，因此只报告前沿，不事后挑一个最好点。
""")

md(r"""
## 9. P5：VSS与EVPI

在同一天、同一场景集、同一初始与终端SOC下计算：随机规划解 $RP$；用均值场景先求决策、
再放回全部场景评价的 $EEV$；每个场景都预先知道后分别求解的等待观望值 $WS$。于是

\[
VSS=EEV-RP,\qquad EVPI=RP-WS,
\qquad WS\le RP\le EEV.
\]

$VSS$ 回答考虑多场景值多少钱，$EVPI$ 回答完美预测最多还值多少钱。这两个量属于经验场景
分布内的理论量，不能拿实际回测的策略费用差冒充。逐日检查上述序关系。
""")

md(r"""
## 10. P6：2/3日滚动SOC与跨日规划

在每天0点求未来 $H\in\{2,3\}$ 日规划，只执行首日144个时槽，再以实际SOC进入次日重规划。
第2/3日不能读取未来真实值，故递归生成 V2 成员预测，并把组合权重冻结在预测原点；未来未知
残差创新取条件均值0。场景残差改为连续 $H$ 日历史块，且块末日严格小于决策日，从而保留
跨日相关性并防止泄漏。

远端SOC边界取1200、6000、10800 kWh，分别代表保护下界、50%参考点和保护上界；这三档来自
题设电池边界而非调参。滚动SOC实验检验“每天参考终点6000”是否造成短视。由于题意是否允许
跨日重规划存在解释空间，P6作为边界扩展，不自动替代H1主模型。
""")

code("""
# 全新运行规划端扩展；程序按月写检查点，并保存原始逐日/逐时结果。
cmd=[sys.executable,'code/q2_v2/planning_extensions_v2.py','--stage','all']
print('RUN:', ' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
""")

code("""
# 只使用本次输出构建论文图表和数据驱动报告。
subprocess.run([sys.executable,'code/q2_v2/build_delivery_v2.py'],check=True)
subprocess.run([sys.executable,'code/q2_v2/build_extensions_delivery_v2.py'],check=True)
""")

md("""## 11. 全年结果、成本分解与月度稳定性

以下数字由本次 Colab 输出现场读取，不手工填写。""")
code("""
R=PROJECT/'results/q2_fused_v2'; F=PROJECT/'figures/q2_fused_v2'
summary=pd.read_csv(R/'summary.csv'); display(summary.sort_values(['period','total_cost']))
display(Image(filename=str(F/'01_cost_and_risk.png')))
display(Image(filename=str(F/'02_monthly_stability.png')))
""")

md("""## 12. 参数更新、`near()` 和 Validation/CVaR 证据

表中保留每月候选、在位参数、Bootstrap区间、14天成本验收和56天风险验收。""")
code("""
display(pd.read_csv(R/'validation_diagnostics.csv'))
scores=pd.read_csv(R/'fused_scores.csv')
display(scores.groupby(['month','stage']).agg(candidates=('score','size'),
    indistinguishable=('near_indistinguishable','sum'),best_score=('score','min')).reset_index())
display(Image(filename=str(F/'03_parameter_path.png')))
display(Image(filename=str(F/'04_validation_and_cvar.png')))
""")

md("""## 13. $E_{target}$ 与期末库存价值敏感性

$E_{target}$ 敏感性是真实重跑，期末价值敏感性是对同一SOC轨迹的核算；两者不混淆。""")
code("""
target=pd.read_csv(R/'terminal_target_sensitivity.csv')
display(target[target.inventory_valuation=='replacement'].sort_values(['strategy','sensitivity_target']))
display(pd.read_csv(R/'inventory_value_sensitivity.csv').query("period=='full334'"))
display(Image(filename=str(F/'05_terminal_target_sensitivity.png')))
display(Image(filename=str(F/'06_inventory_value_sensitivity.png')))
""")

md("""## 14. Bootstrap、压力测试与验收

若费用差的区间跨过0，不将点估计的优势表述为稳健改进。冻结期是后期检查，但已被研究者查看，不夸大为真正未开封盲测。""")
code("""
display(pd.read_csv(R/'bootstrap.csv')); display(pd.read_csv(R/'stress.csv'))
display(Image(filename=str(F/'07_bootstrap_vs_ridge.png')))
verification=json.loads((R/'verification.json').read_text())
display(pd.DataFrame(verification['tests']))
assert verification['all_pass'] and all(x['passed'] for x in verification['tests'])
print('ALL VERIFICATION TESTS PASSED:',verification['count'])
""")

md("""## 15. CVaR、VSS/EVPI与滚动规划结果

先看理论信息价值，再看风险—费用前沿与影子价格，最后比较2/3日跨日滚动策略。每张图的
解释写在图下方和扩展报告中。""")
code("""
RX=PROJECT/'results/q2_fused_v2_extensions';FX=PROJECT/'figures/q2_fused_v2_extensions'
display(pd.read_csv(RX/'information_monthly.csv'))
display(Image(filename=str(FX/'08_vss_evpi_monthly.png')))
display(pd.read_csv(RX/'cvar_frontier_summary.csv'))
display(Image(filename=str(FX/'09_cvar_frontier.png')))
display(pd.read_csv(RX/'dual_risk_price.csv'))
display(Image(filename=str(FX/'10_dual_risk_price.png')))
display(pd.read_csv(RX/'rolling_summary.csv'))
display(Image(filename=str(FX/'11_rolling_horizon.png')))
xverify=json.loads((RX/'verification.json').read_text());display(pd.DataFrame(xverify['tests']))
assert xverify['all_pass'];display(Markdown((PROJECT/'reports/Q2_FUSED_V2_EXTENSIONS_RESULTS.md').read_text()))
""")

md("""## 16. 局限与论文使用边界

- 单年回测不等于跨年泛化保证；56天CVaR95仍只含2.8个等权尾部日，所以同时报告CVaR90、worst-day和压力情景。
- `near()` 只是搜索段简约化，不是候选等价的证明。
- Ridge预测明细被冻结以保持对照定义；本次全新生成其规划、执行、SOC、费用和检验。
- GPU主要提供Colab高性能主机；HiGHS LP和Bootstrap为CPU计算，不声称GPU直接加速LP。
- “全局最优”只指已给定有限样本的LP，不外推到未知真实分布。""")
code("""
run_manifest=json.loads((R/'run_manifest.json').read_text())
figure_manifest=json.loads((F/'figure_manifest.json').read_text())
print(json.dumps(run_manifest,ensure_ascii=False,indent=2)); display(pd.DataFrame(figure_manifest))
display(Markdown((PROJECT/'reports/Q2_FUSED_V2_RESULTS.md').read_text()))
""")

nbf.write(nb, OUT)
print(OUT)
