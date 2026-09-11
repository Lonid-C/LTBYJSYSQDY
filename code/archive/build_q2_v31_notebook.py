"""Build the self-contained Q2 V3.1 Colab notebook."""
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
MODULE=ROOT/'code/archive/q2_v31/q2_terminal_ablation.py'
OUT=ROOT/'code/problem2_v31_terminal_ablation.ipynb'
nb=nbf.v4.new_notebook()
nb.metadata.update({'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'},
    'colab':{'name':OUT.name,'provenance':[],'gpuType':'L4'}})

def md(x):nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x):nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r'''
# C题问题二 V3.1：终端条件消融与对偶价值修正

本笔记本只修正规划端终端处理，不重新训练预测模型。Ridge 负荷与光伏预测、历史完整日残差场景、月度验证后确定的场景参数、H1 因果执行器全部保持一致。这样五条策略之间唯一的结构差异是终端条件：

1. 日前参考终态固定为6000 kWh；
2. 日前参考终态等于当天实际期初 SOC；
3. 终态自由且没有未来价值；
4. 9点 SOC 网格的对偶支持切平面价值；
5. 17点 SOC 网格的对偶支持切平面价值。

正式提交工作簿不在本实验中覆盖。
''')

md(r'''
## 1. 问题、信息集与数据单位

第 (d) 天第 (t) 个10分钟时槽的净负荷功率为

\[
N_{d,t}=L_{d,t}-P_{d,t}\quad(\mathrm{kW}),
\qquad \Delta t=\frac16\ \mathrm h.
\]

进入规划模型的时槽净电量为

\[
n_{d,t}=N_{d,t}\Delta t\quad(\mathrm{kWh}).
\]

代码中的 (q_t,c_t,v_t,z_{s,t}) 均表示一个时槽内的电量（kWh），而 (E_t) 表示时槽末 SOC（kWh）。电价 (p_t) 的单位是元/kWh。因此目标函数中不再重复乘 ​(\Delta t)。日期 (d) 的决策只使用 (i<d) 的历史数据。
''')

code(r'''
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd
from IPython.display import display,Image,Markdown
PROJECT=Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path.cwd()
os.chdir(PROJECT)
print('project =',PROJECT)
print('python =',sys.version)
''')

md(r'''
## 2. 冻结的预测与场景模型

复用既有 Ridge 对负荷、光伏的预测，构造点预测净负荷

\[
\widehat N_{d,t}=\widehat L_{d,t}-\widehat P_{d,t}.
\]

历史完整日残差为

\[
\varepsilon_{i,t}=N_{i,t}-\widehat N_{i,t},
\]

第 (i) 条场景为

\[
N_{d,t}^{(i)}=\widehat N_{d,t}+\varepsilon_{i,t}.
\]

2—5月沿用21日窗口、14日半衰期；V3 的独立验证只接受了6月开始的14日窗口、7日半衰期，之后冻结。相似日组件没有通过验证，故本轮不使用。场景权重为

\[
w_{d,i}=\frac{2^{-(d-i)/\tau}}{\sum_j2^{-(d-j)/\tau}}.
\]

执行储备系数保持已验证的 ​(\rho=0.5)，终端消融不重新搜索其他规划参数。
''')

code(r'''
sys.path.insert(0,str(PROJECT/'code/q2_v31'))
from q2_terminal_ablation import *
preview=Experiment()
audit=pd.DataFrame({
    '检查项':['日期数','每日时槽','负荷有限','光伏有限','日期唯一','日期递增','预测文件哈希'],
    '结果':[len(preview.dates),preview.net.shape[1],np.isfinite(preview.study.load).all(),
          np.isfinite(preview.study.pv).all(),preview.dates.is_unique,preview.dates.is_monotonic_increasing,
          sha(PROJECT/'results/problem2_forecast_ablation_predictions.csv.gz')]})
display(audit)
''')

md(r'''
## 3. 风险中性有限场景线性规划

决策变量：计划购电量 (q_t)、参考充电量 (c_t)、参考放电量 (v_t)、SOC (E_t)、场景紧急购电量 (z_{s,t}) 和可选的终端未来成本 (\theta)。模型为

\[
\min\quad
\sum_{t=1}^{144}p_tq_t
+\sum_{s=1}^{S}w_s\sum_{t=1}^{144}5p_tz_{s,t}
+\theta,
\]

\[
q_t+v_t-c_t+z_{s,t}\ge N_{d,t}^{(s)}\Delta t,
\]

\[
E_t=E_{t-1}+\eta_c c_t-\frac{v_t}{\eta_d},
\]

\[
0\le c_t,v_t\le P_{\max}\Delta t=833.333\ \mathrm{kWh},
\quad1200\le E_t\le10800,
\quad q_t,z_{s,t}\ge0.
\]

当不使用未来价值时删去 (\theta)。目标函数和约束均为线性，因此每个有限场景问题都是 LP，求解器给出的原始—对偶证书对应有限场景问题的全局最优解。
''')

md(r'''
## 4. 四类终端机制

### 4.1 固定参考终态

\[
E_{144}=6000.
\]

它便于日间能量闭合，但将6000作为固定政策目标。

### 4.2 当日循环终态

\[
E_{144}=E_0.
\]

它只要求日前参考计划在一天内能量中性，不偏好某个固定 SOC。

### 4.3 自由终态

\[
1200\le E_{144}\le10800,
\]

且不加入后续价值。该模型揭示完全忽略跨日库存价值会产生多强的末端效应。

### 4.4 对偶学习终端价值

取消终端等式，在目标中加入下一星期状态的相对价值 (\widehat V_{u+1}(E_{144}))。6000 kWh 只作为函数加法零点 (\widehat V_u(6000)=0)，整体平移不改变决策。
''')

md(r'''
## 5. 真正由 LP 对偶变量生成支持切平面

在 SOC 网格点 (e_j) 上求 Bellman 子问题，得到最优目标值 (Q_u(e_j))。初始 SOC 所在等式右端的对偶变量

\[
\pi_j=\frac{\partial Q_u(e_j)}{\partial e_j}
\]

是凸最优值函数的一个次梯度。因此支持切平面直接写为

\[
\ell_j(E)=Q_u(e_j)+\pi_j(E-e_j),
\]

并构造

\[
\widehat V_u(E)=\max_j\ell_j(E)-\max_j\ell_j(6000).
\]

代码逐点验证所有切平面在网格上不超过对应 (Q_u)，并验证自身切平面在生成点与 (Q_u(e_j)) 相切。这里不再用相邻网格目标值的割线代替对偶切平面。
''')

md(r'''
## 6. 周期相对价值迭代与参数来源

星期状态 (u\in\{0,\ldots,6\}) 满足

\[
V_u(e)=\min_x\left\{\mathbb E[C_u(x,N)]+V_{u+1}(E_{144})\right\}-g_u.
\]

每月1日只用此前84天，即12个完整周，估计各星期净负荷分布；28天半衰期表示最近4周权重衰减一半。这两个数是历史经验分布的时间尺度，不代表用户风险偏好。迭代阈值 (10^{-6}) 元仅用于数值停止，最大50次是防止异常不收敛的保护上限。

SOC 网格分别使用9点和17点：步长分别为1200 kWh和600 kWh。网格节点数不是调参结果，而是数值离散收敛实验。除比较两条价值曲线外，还比较全年调整后费用和年末 SOC，避免只凭函数图宣称收敛。
''')

md(r'''
## 7. H1因果执行与双边库存价值

日前计划只确定购电与参考 SOC 路径。每个时槽到来后，执行器只读取当前真实净负荷、当前 SOC、日前购电量和当前参考 SOC，不访问未来真实值。实际 SOC 连续跨日继承。

对任意评价区间 ([a,b])，不同策略可能具有不同期初与期末库存。公平费用定义为

\[
C_{[a,b]}^{\rm adj}
=C_{[a,b]}^{\rm raw}
+\widehat V_{b+1}(E_b)
-\widehat V_a(E_{a-1}).
\]

即同时计入期末库存并扣除期初库存，统一采用更细的17点价值函数作为核算价格。全年各策略都从2月1日6000 kWh启动，因此全年期初项自然为0。
''')

source=MODULE.read_text(encoding='utf-8').replace('from __future__ import annotations\n','')
source=source.replace("\nif __name__ == '__main__':\n    full()\n",'\n')
code("# 以下是本次实验的完整可执行源码。\n__file__=str(PROJECT/'code/archive/q2_v31/q2_terminal_ablation.py')\n"+source)

md('''## 8. 从头运行 V3.1\n\n本单元删除并重建 V3.1 专属输出，预测文件只读，旧版规划结果不作为本次策略费用输入。''')
code('full()')

md('''## 9. 对偶切平面与价值迭代验收''')
code(r'''
dual_checks=pd.read_csv(OUT/'dual_cut_checks.csv')
display(dual_checks[['support_violation','own_cut_error','dual_min','dual_max']].agg(['min','max']))
iteration=pd.read_csv(OUT/'value_iteration.csv')
display(iteration.groupby(['grid_size','origin']).tail(1))
''')

md('''## 10. 9点—17点网格收敛''')
code(r'''
display(pd.read_csv(OUT/'grid_curve_convergence.csv').agg({
    'max_abs_value_difference':['max','mean'],
    'mean_abs_value_difference':['max','mean'],
    'relative_max_difference':['max','mean']}))
display(pd.read_csv(OUT/'grid_policy_convergence.csv'))
''')

md('''## 11. 终端机制费用与运行指标''')
code(r'''
summary=pd.read_csv(OUT/'summary.csv')
display(summary.sort_values(['period','adjusted_total_cost']))
display(pd.read_csv(OUT/'bootstrap.csv'))
''')

md('''## 12. 月度稳定性与 SOC 行为''')
code(r'''
display(pd.read_csv(OUT/'monthly.csv'))
daily=pd.read_csv(OUT/'daily.csv')
display(daily.groupby('policy').terminal_soc.agg(['min','median','mean','max','last']))
''')

md('''## 13. 图表及含义

- 相对费用图放大总费用的细小差异，零点对应全年调整后费用最低方案；
- 月度差额图检验优势是否集中在少数月份；
- 网格价值图检验9点和17点离散是否给出接近的未来成本曲线；
- SOC箱线图展示不同终端机制是否导致持续高库存或末端放空。
''')
code(r'''
for filename in ['01_adjusted_cost_delta.png','02_monthly_cost_delta.png',
                 '03_grid_value_comparison.png','04_soc_distribution.png']:
    display(Image(filename=str(FIG/filename)))
''')

md('''## 14. 约束、因果性与运行清单''')
code(r'''
verification=json.loads((OUT/'verification.json').read_text())
display(pd.DataFrame(verification['tests']))
manifest=json.loads((OUT/'run_manifest.json').read_text())
print(json.dumps(manifest,ensure_ascii=False,indent=2))
assert verification['all_pass'] and manifest['complete'] and manifest['solve_calls']>0
''')

md('''## 15. 数据驱动结论

下面的结论完全由本次结果生成。主要判断顺序为：双边库存价值修正后的全年费用、开发期与冻结期方向、配对7日块区间、9点与17点策略差异。若复杂方法与简单方法差异很小且区间跨0，优先选择结构更简单的终端机制。
''')
code(r'''
s=summary.set_index(['policy','period'])
full=s.xs('full334',level='period').sort_values('adjusted_total_cost')
best=full.iloc[0]
g9=s.loc[('dual_terminal_g9','full334')]
g17=s.loc[('dual_terminal_g17','full334')]
display(Markdown(f"""全年调整后费用最低的是 **{full.index[0]}**，为 **{best.adjusted_total_cost:,.2f} 元**。17点对偶价值策略为 **{g17.adjusted_total_cost:,.2f} 元**；9点与17点的全年费用差为 **{g9.adjusted_total_cost-g17.adjusted_total_cost:,.2f} 元**。最终是否采用对偶终端价值，应同时参考上表的冻结期结果和Bootstrap区间，而不是只依据模型复杂度。"""))
''')

nbf.write(nb,OUT)
print(OUT)
