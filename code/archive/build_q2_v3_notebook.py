"""Build the self-contained Q2 V3 Colab notebook from the audited source module."""
from pathlib import Path
import nbformat as nbf

def project_root(start):
    """按只在仓库根存在的文件向上查找项目根。

    使本模块不依赖自身在 code/ 下的深度，归档移动后仍可运行。
    """
    for candidate in (start, *start.parents):
        if (candidate / 'C题/附件/附件2.xlsx').exists():
            return candidate
    raise FileNotFoundError('未找到含 C题/附件/附件2.xlsx 的项目根目录')



ROOT=project_root(Path(__file__).resolve().parent)
MODULE=ROOT/'code/archive/q2_v3/q2_dual_terminal.py'
OUT=ROOT/'code/problem2_v3_dual_terminal.ipynb'
nb=nbf.v4.new_notebook()
nb.metadata.update({'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'},
    'colab':{'name':OUT.name,'provenance':[],'gpuType':'L4'}})

def md(x):nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x):nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r'''
# C题问题二 V3：条件加权 SAA—对偶终端价值—H1 因果执行

本笔记本在现有 Q2 结果上增量修改，预测中心固定为已经完成全年公平回测的 Ridge，不重新堆叠预测算法。新的证据链为

\[
\text{Ridge中心预测}
\rightarrow\text{完整日条件残差场景}
\rightarrow\text{风险中性加权SAA}
\rightarrow\text{对偶学习终端价值}
\rightarrow\text{H1因果执行}
\rightarrow\text{端到端经济评价}.
\]

本版取消每日和远端的硬终端 SOC 等式。6000 kWh 仅是题设初始状态以及价值函数的加法零点；价值函数整体平移不会改变任何最优决策。正式 `result2.xlsx` 不覆盖。
''')

md(r'''
## 1. 问题分析与严格信息集

附件2包含365天、每日144个10分钟时槽的负荷和光伏。记

\[
N_{d,t}=L_{d,t}-P_{d,t},\qquad \Delta t=\frac16\ \mathrm h.
\]

日期 $d$ 的日前决策只能使用

\[
\mathcal I_d=\{L_{i,t},P_{i,t}:i<d\}
\]

和题设已知电价、储能物理参数。场景日期、参数搜索数据、价值函数训练数据的最大日期都必须严格小于决策日期。1月用于历史积累，2月1日以题给6000 kWh启动连续回测。
''')

code(r'''
from pathlib import Path
import os,sys,json,subprocess
import numpy as np,pandas as pd
from IPython.display import display,Image,Markdown
PROJECT=Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path.cwd()
os.chdir(PROJECT)
print('project =',PROJECT)
print('python =',sys.version)
''')

md(r'''
## 2. 数据读取、清洗与 Ridge 中心预测

数据检查包括日期连续、每天144时槽、缺失值、非有限值和物理非负性。负荷与光伏分别使用冻结的 Ridge 预测，再形成

\[
\widehat N_{d,t}=\max(\widehat L_{d,t},0)-\max(\widehat P_{d,t},0).
\]

采用 Ridge 的依据不是只看 MAE，而是此前相同执行器下的全年端到端购电费更低。2—3月严格使用已经声明的历史冷启动预测，不用真实当日值替换预测。
''')

code(r'''
sys.path.insert(0,str(PROJECT/'code/q2_v3'))
from q2_dual_terminal import V3,ROOT,OUT,FIG,REPORT,DT,T,BATTERY
preview=V3()
audit=pd.DataFrame({
    'item':['days','slots_per_day','load_finite','pv_finite','date_unique','date_monotone'],
    'value':[len(preview.dates),preview.load.shape[1],np.isfinite(preview.load).all(),np.isfinite(preview.pv).all(),
             preview.dates.is_unique,preview.dates.is_monotonic_increasing]})
display(audit)
print('SOC bounds =',BATTERY.minimum,BATTERY.maximum,'kWh; power/slot =',BATTERY.limit,'kWh')
''')

md(r'''
## 3. 完整日条件残差场景

历史预测残差为

\[
\varepsilon_{i,t}=N_{i,t}-\widehat N_{i,t}.
\]

场景使用完整144点残差曲线

\[
N_{d,t}^{(s)}=\widehat N_{d,t}+\varepsilon_{s,t},
\]

不逐时打乱，从而保留峰谷、光伏波动和爬坡的日内相关性。相似日距离只使用预测时已知的季节、星期、预测负荷/光伏电量和净负荷形状特征。全局场景的新近权重与条件场景相似度权重混合为

\[
\widetilde w_{d,i}^{G}=2^{-(d-i)/\tau},\qquad
\widetilde w_{d,i}^{C}=\exp\!\left[-\frac{D_{d,i}^2}{2h_d^2}\right].
\]

$h_d$ 取当时邻居距离的中位数，是由历史几何尺度自适应得到的带宽。窗口 $W$、半衰期 $\tau$、邻居数 $K$、条件混合质量 $b$ 和执行储备系数 $\rho$ 通过滚动成本验证选择。
''')

md(r'''
## 4. 风险中性加权 SAA

决策变量为计划购电 $q_t$、参考充电 $c_t$、参考放电 $v_t$、SOC $E_t$、场景紧急购电 $z_{s,t}$ 和未来成本变量 $\theta$。目标为

\[
\min\ \sum_t p_tq_t\Delta t
+\sum_sw_s\sum_t5p_tz_{s,t}\Delta t
+\theta.
\]

约束为

\[
q_t+v_t-c_t+z_{s,t}\ge N_{d,t}^{(s)},
\]

\[
E_t=E_{t-1}+\eta_c c_t\Delta t-\frac{v_t\Delta t}{\eta_d},
\]

\[
0\le c_t,v_t\le P_{\max},\quad
1200\le E_t\le10800,\quad q_t,z_{s,t}\ge0.
\]

目标和全部约束均为线性，因而有限场景问题是 LP；HiGHS 返回全局最优解及原始—对偶证书。
''')

md(r'''
## 5. 为什么不再设置 $E_T=6000$

每日硬等式会阻断跨日套利；把同一等式推迟到两三天以后，只是推迟末端效应。本版在当天和远端都不设置硬 SOC 等式，而加入未来运行成本

\[
\widehat V_{d+1}(E_{d,T}).
\]

6000只作为相对价值的零点：

\[
\widehat V(6000)=0.
\]

把价值函数整体加上任意常数不会改变最优解，因此这个零点不具有政策偏好含义。
''')

md(r'''
## 6. 周期相对价值迭代与对偶切平面

由于电价和用电具有显著周周期，用过去数据构造星期状态 $u\in\{0,\ldots,6\}$ 的经验分布，求平均成本 Bellman 方程

\[
V_u(e)=\min_x\left\{\mathbb E[C_u(x,N)]+V_{u+1}(E_T)\right\}-g_u,
\]

其中 $g_u$ 是归一化常数。对多个初始 SOC $e_j$ 求解 LP，初始 SOC 平衡约束的对偶变量为

\[
\pi_j=\frac{\partial V_u(e_j)}{\partial e_j},
\]

表示增加1 kWh库存对未来最优成本的边际影响。由此生成切平面

\[
\theta\ge V_u(e_j)+\pi_j(E_T-e_j),\qquad\forall j,
\]

以及分段线性凸函数

\[
\widehat V_u(E)=\max_j\{a_jE+b_j\}.
\]

斜率强制满足非正且单调不减，对应“更多库存不会增加未来成本”和“库存边际价值递减”。SOC网格属于数值离散：先使用9点，再用17点做网格收敛检查；停止阈值按人民币计算精度声明，不作为经济偏好参数。
''')

md(r'''
## 7. H1因果执行

日前 SAA 给出购电和参考 SOC 路径。实际时槽 $t$ 到来后，只使用当前实际净负荷、当前实际 SOC、日前购电和参考路径决定当期充放电与紧急购电：

\[
(c_t^{act},v_t^{act},z_t^{act})
=\mu_t(N_{1:t}^{act},E_{t-1}^{act},q_{1:144},E_{1:144}^{ref}).
\]

实际 SOC 跨日连续：

\[
E_{d+1,0}^{act}=E_{d,144}^{act}.
\]

年末不硬拉回6000；按相同学习价值函数进行库存归一化：

\[
C^{adj}=C^{raw}+\widehat V_{next}(E_{year,end})-\widehat V_{next}(6000).
\]
''')

md(r'''
## 8. 参数来源与防过拟合协议

每个正式月度更新使用

\[
42\text{日搜索}+14\text{日独立验证}+1\text{日embargo}.
\]

窗口候选均为完整周的整数倍；半衰期覆盖快、中、慢衰减和等权；条件混合质量覆盖纯全局至纯条件；$\rho$ 覆盖物理可行区间。搜索期对配对日成本做7日移动块 Bootstrap，统计不可区分时选结构更简单者。部署候选只有在随后14日上“原模型费用减候选费用”的95%区间下界大于0时才更新。原固定0.1%近似并列规则不再使用。
''')

md(r'''
## 9. 对照、经济价值与验收

正式比较：当前 `Ridge + 21日/14日半衰期SAA + H1`、条件场景但每日硬6000、以及新对偶终端价值策略。另计算同场景

\[
VSS=EEV-RP,\qquad EVPI=RP-WS,
\]

并检查 $WS\le RP\le EEV$。评价包括计划费、紧急费、总费、紧急购电量与频率、弃电量、最大日费、月度稳定性、年末SOC、7日块Bootstrap，以及功率平衡、SOC递推、边界、跨日连续和LP残差。
''')

# Embed the exact executable source so the delivered notebook contains the whole chain.
source=MODULE.read_text(encoding='utf-8').replace('from __future__ import annotations\n','')
source=source.replace("\nif __name__=='__main__':main()\n",'\n')
code("# 下面是与本次实验完全一致的可执行源码。\n__file__=str(PROJECT/'code/archive/q2_v3/q2_dual_terminal.py')\n"+source)

md('''## 10. 在本次环境从头运行 V3\n\n本单元只清理 V3 专属输出，不读取旧 V3 结果；现有 V2 和正式提交表不受影响。程序按月写参数、价值函数和策略检查点。''')
code("resume_or_full()")

md('''## 11. 参数更新轨迹与价值函数收敛''')
code(r'''
display(pd.DataFrame(json.loads((OUT/'selection.json').read_text())))
vi=pd.read_csv(OUT/'value_iteration.csv')
display(vi.groupby('origin').tail(1))
display(pd.read_csv(OUT/'terminal_value_curves.csv').head(20))
''')

md('''## 12. 全年费用、Bootstrap与VSS/EVPI''')
code(r'''
summary=pd.read_csv(OUT/'summary.csv');display(summary.sort_values(['period','adjusted_total_cost']))
display(pd.read_csv(OUT/'bootstrap.csv'))
information=pd.read_csv(OUT/'information_value.csv')
display(information[['RP','EEV','WS','VSS','EVPI']].sum().to_frame('sum'))
''')

md('''## 13. 图表与含义\n\n费用分解图显示不同策略的计划购电费和紧急购电费；月度图检查优势是否来自少数月份；终端价值曲线显示不同星期下SOC的边际经济价值，并检验曲线是否递减、凸且经过6000的相对零点。''')
code(r'''
for f in ['01_cost_comparison.png','02_monthly_stability.png','03_terminal_value.png']:
    display(Image(filename=str(FIG/f)))
''')

md('''## 14. 约束、因果性与运行清单''')
code(r'''
verification=json.loads((OUT/'verification.json').read_text());display(pd.DataFrame(verification['tests']))
manifest=json.loads((OUT/'run_manifest.json').read_text());print(json.dumps(manifest,ensure_ascii=False,indent=2))
assert verification['all_pass'] and manifest['complete']
''')

md('''## 15. 数据驱动结论\n\n以下文字由本次运行结果自动生成，不预写“新模型更优”。只有开发期改善、Bootstrap支持且冻结期不恶化时，才建议替换现有方案。''')
code(r'''
s=summary.set_index(['policy','period'])
new=s.loc[('v3_dual_terminal','full334')];base=s.loc[('ridge_fixed_saa_h1','full334')]
dev=s.loc[('v3_dual_terminal','development153')].adjusted_total_cost-s.loc[('ridge_fixed_saa_h1','development153')].adjusted_total_cost
hold=s.loc[('v3_dual_terminal','frozen61')].adjusted_total_cost-s.loc[('ridge_fixed_saa_h1','frozen61')].adjusted_total_cost
b=pd.read_csv(OUT/'bootstrap.csv');ci=b[(b.reference=='ridge_fixed_saa_h1')&(b.period=='full334')].iloc[0]
recommend=(dev<0 and hold<=0 and ci.ci_low95>0)
display(Markdown(f"""全年新模型调整后费用为 **{new.adjusted_total_cost:,.2f} 元**，现有 Ridge 基线为 **{base.adjusted_total_cost:,.2f} 元**。开发期费用差（新−旧）为 **{dev:,.2f} 元**，冻结期为 **{hold:,.2f} 元**；相对旧模型的日节省95%区间为 **[{ci.ci_low95:,.2f}, {ci.ci_high95:,.2f}] 元/日**。据预先声明规则，**{'建议替换' if recommend else '暂不建议替换'}**。"""))
''')

nbf.write(nb,OUT)
print(OUT)
