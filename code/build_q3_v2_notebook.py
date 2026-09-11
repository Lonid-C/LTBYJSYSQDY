"""Build the self-contained Q3 aligned Colab notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'code/q3_v2/q3_aligned.py'
OUT=ROOT/'code/problem3_v2_aligned.ipynb'
nb=nbf.v4.new_notebook()
nb.metadata.update({'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'},
    'colab':{'name':OUT.name,'provenance':[],'gpuType':'L4'}})

def md(x):nb.cells.append(nbf.v4.new_markdown_cell(x.strip()))
def code(x):nb.cells.append(nbf.v4.new_code_cell(x.strip()))

md(r'''
# 第三问对齐版：多阶段 SAA + 合同调整

本笔记本把第三问建立在第二问最终模型（`dual_terminal_g17`）的同一套机器上，而不是另起炉灶：

- **0:00 契约计划**：与 Q2 完全相同的两阶段 SAA（冻结 Ridge 预测 + 完整日残差场景 + 17 点对偶终端价值），
  但场景补救价格换成 Q3 结算规则——缺口按上调 1.5p、盈余按取消 0.5p 计价（Q3 下调整是缺口的第一顺位补救，
  5 倍紧急购电只是最后一次调整之后的实时兜底）；
- **6:00 / 12:00 / 18:00 部分补救**：以实际储能状态与最新光伏预报重优化当日剩余时段，
  相对 0:00 契约按净额结算一次：$1.5p_t u_t - 0.5p_t v_t$，场景紧急项仍为 $5p_t z_{s,t}$；
- **执行**：每个时槽用 H1 因果执行器（储备系数 $\rho=0.5$，Q2 已验证），实际 SOC 跨日连续。

结算口径：$C=\sum_t\left[p_tq^0_t+1.5p_t(q_t-q^0_t)^+-0.5p_t(q^0_t-q_t)^++5p_tz_t\right]$，
等价于 $p_tq_t+0.5p_t|q_t-q^0_t|$。本次运行覆盖 `result3.xlsx`（按附件 5 模板复制填充，表头不改、按位置对应写入）。
''')

code(r'''
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd
from IPython.display import display,Markdown
PROJECT=Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path.cwd()
os.chdir(PROJECT)
for extra in ('code','code/q2_v2','code/q2_v32','code/q3_v2'):
    sys.path.insert(0,str(PROJECT/extra))
print('project =',PROJECT); print('python =',sys.version.split()[0])
''')

code(r'''
import q3_v2.q3_aligned as M
exp=M.Experiment(); banks=M.load_banks()
audit=pd.DataFrame({'检查项':['日期数','每日时槽','价值库月份数','价值库切平面数','预测文件哈希','价值库哈希'],
 '结果':[len(exp.dates),exp.net.shape[1],len(banks),len(banks[2][0]),
        M.sha(PROJECT/'results/problem2_forecast_ablation_predictions.csv.gz'),
        M.sha(PROJECT/'results/q2_v32_final/value_banks.json')]})
display(audit)
print('评价区间:',exp.dates[31].date(),'→',exp.dates[364].date())
''')

source=MODULE.read_text(encoding='utf-8').replace('from __future__ import annotations\n','')
source=source.replace("\nif __name__ == '__main__':\n    full()\n",'\n')
code("__file__=str(PROJECT/'code/q3_v2/q3_aligned.py')\n"+source)

md('''## 运行

删除并重建 `results/q3_v2_aligned/`，重算 `result3.xlsx`。主跑之外再跑一次
起点敏感性（2/1 起点取队友版按 1 月模拟得到的库存 1,520.655 kWh），用于量化起点口径差异。
''')
code('full()')

md('''## 费用与验收''')
code(r'''
summary=pd.read_csv(OUT/'summary.csv'); display(summary)
verification=json.loads((OUT/'verification.json').read_text())
display(pd.DataFrame(verification['tests']))
assert verification['all_pass']
''')

md('''## 遗传算法单日对照

在 4 个指定日期上，把与 0:00 层完全相同的决策问题交给实数编码遗传算法
（染色体 = 当日 144 槽的 q/c/dis，SOC 递推由染色体决定，越界罚金 10 倍最高电价）。
LP 与 GA 的目标函数完全相同，因此两个数可直接比较。
''')
code(r'''
ga=pd.read_csv(OUT/'ga_single_day.csv')
ga['GA/LP 耗时比']=(ga.ga_seconds/ga.lp_seconds).round(0)
display(ga)
display(Markdown(f"四个指定日期上 GA 相对 LP 的目标值差为 "
  f"**{ga.gap_relative.mean():.4%}**（平均），GA 耗时为 LP 的 **{ga['GA/LP 耗时比'].mean():,.0f} 倍**。"
  "LP 给出的是该有限场景问题的全局最优解且有对偶证书，遗传算法只能逼近、无法证明最优——"
  "这正是模型采用线性规划而非启发式搜索的实证依据。"))
''')

md('''## `result3.xlsx` 回读与运行清单''')
code(r'''
export=json.loads((OUT/'result3_export_audit.json').read_text())
print('days =',export['days'],'| events =',export['events'],'| passed =',export['passed'])
display(pd.Series({k:v for k,v in export.items() if k!='path'},name='value').to_frame())
from openpyxl import load_workbook
wb=load_workbook(PROJECT/'result3.xlsx')
tpl=load_workbook(PROJECT/'C题/附件/附件5/result3.xlsx')
same=all([c.value for c in tpl[s][1]]==[c.value for c in wb[s][1]] for s in wb.sheetnames)
display(pd.DataFrame([{'工作表':ws.title,'行数':ws.max_row,'列数':ws.max_column,
                       '表头与模板一致':([c.value for c in tpl[ws.title][1]]==[c.value for c in ws[1]])}
                      for ws in wb.worksheets]))
print('全部表头与附件 5 模板一致:',same)
manifest=json.loads((OUT/'run_manifest.json').read_text())
print(json.dumps(manifest,ensure_ascii=False,indent=2))
''')

nbf.write(nb,OUT)
print(OUT)
