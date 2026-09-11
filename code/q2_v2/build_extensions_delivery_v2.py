"""Build publication figures and a data-driven report for V2 planning extensions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/q2_fused_v2_extensions';F=ROOT/'figures/q2_fused_v2_extensions'
REPORT=ROOT/'reports/Q2_FUSED_V2_EXTENSIONS_RESULTS.md'

def table(x,digits=2):
    return x.to_markdown(index=False,floatfmt=f'.{digits}f')

def save(fig,name,meaning,manifest):
    for suffix in ('png','pdf'):fig.savefig(F/f'{name}.{suffix}',dpi=220 if suffix=='png' else None,bbox_inches='tight')
    plt.close(fig);manifest.append(dict(figure=name,meaning=meaning))

def main():
    required=['information_monthly.csv','cvar_frontier_summary.csv','dual_risk_price.csv','rolling_summary.csv','verification.json','run_manifest.json']
    missing=[x for x in required if not (R/x).exists()]
    if missing:raise FileNotFoundError(missing)
    F.mkdir(parents=True,exist_ok=True)
    for path in Path('/usr/share/fonts').rglob('*CJK*.ttc'):
        try:font_manager.fontManager.addfont(str(path))
        except Exception:pass
    names=[x.name for x in font_manager.fontManager.ttflist if 'Noto Sans CJK' in x.name or 'Source Han Sans' in x.name]
    plt.rcParams.update({'font.family':names[0] if names else 'DejaVu Sans','axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
    info=pd.read_csv(R/'information_monthly.csv');risk=pd.read_csv(R/'cvar_frontier_summary.csv')
    dual=pd.read_csv(R/'dual_risk_price.csv');roll=pd.read_csv(R/'rolling_summary.csv');manifest=[]

    fig,ax=plt.subplots(figsize=(8.4,4.2));x=np.arange(len(info));w=.36
    ax.bar(x-w/2,info.VSS/1e3,w,label='VSS');ax.bar(x+w/2,info.EVPI/1e3,w,label='EVPI')
    ax.set_xticks(x,info.month.astype(int));ax.set_xlabel('月份');ax.set_ylabel('理论价值（千元）');ax.legend();ax.grid(axis='y',alpha=.25)
    save(fig,'08_vss_evpi_monthly','按月展示同场景分布上的随机规划价值与完美信息价值；不是实际回测费用差。',manifest)

    fig,ax=plt.subplots(1,2,figsize=(10.8,4.1))
    for beta,g in risk.groupby('beta'):
        g=g.sort_values('risk_reduction_fraction');lab=f'β={beta:.2f}'
        ax[0].plot(g.scenario_cvar/1e6,g.scenario_expected/1e6,marker='o',label=lab)
        ax[1].plot(g.risk_reduction_fraction,g.risk_protection_cost/1e3,marker='o',label=lab)
    ax[0].set_xlabel('场景 CVaR（百万元）');ax[0].set_ylabel('场景期望费用（百万元）')
    ax[1].set_xlabel('风险预算收紧比例');ax[1].set_ylabel('实际回测风险保护代价（千元）')
    for a in ax:a.grid(alpha=.25);a.legend()
    save(fig,'09_cvar_frontier','左图是优化空间的期望—尾部风险前沿，右图是相同计划在真实观测下的费用代价。',manifest)

    fig,ax=plt.subplots(figsize=(8.6,4.1))
    for beta,g in dual.groupby('beta'):
        q=g.groupby('budget_index').lambda_risk.median();ax.plot(q.index,q.values,marker='o',label=f'β={beta:.2f}')
    ax.set_xlabel('风险预算档位');ax.set_ylabel('对偶风险价格（元/元CVaR）');ax.grid(alpha=.25);ax.legend()
    save(fig,'10_dual_risk_price','对偶乘子表示CVaR预算再收紧一元对目标的局部边际代价，不代表用户风险偏好。',manifest)

    q=roll.groupby(['policy']).agg(total_cost=('total_cost','sum'),emergency_cost=('emergency_cost','sum'),emergency_kwh=('emergency_kwh','sum')).reset_index()
    fig,ax=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(len(q))
    ax[0].bar(x,q.total_cost/1e6);ax[1].bar(x,q.emergency_kwh/1e3)
    for a in ax:a.set_xticks(x,q.policy,rotation=30);a.grid(axis='y',alpha=.25)
    ax[0].set_ylabel('全年总费用（百万元）');ax[1].set_ylabel('全年紧急购电（MWh）')
    save(fig,'11_rolling_horizon','比较2/3日滚动规划及远端SOC边界，检验跨日联动是否带来稳定经济价值。',manifest)

    (F/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    isall=pd.DataFrame([{'month':'合计',**{c:info[c].sum() for c in ['days','RP','EEV','WS','VSS','EVPI']}}])
    report=f"""# Q2 V2 规划端扩展：CVaR、VSS/EVPI 与滚动规划

本报告只读取本次 `q2_fused_v2_extensions` 新生成结果。CVaR与信息价值评价期为6—12月，
因为V2学习配置从6月开始部署；2/3日滚动规划覆盖2—12月。所有场景只由决策日前的完整
V2残差路径构造，条件场景与全局场景按当月已验收的blend混合。

## VSS与EVPI

{table(pd.concat([info,isall],ignore_index=True),2)}

这里 $VSS=EEV-RP$、$EVPI=RP-WS$，且逐日检查 $WS\le RP\le EEV$。
它们是同一经验场景分布内的理论经济量，不能与真实观测下的策略费用差混用。

## CVaR风险前沿与对偶风险价格

{table(risk[['beta','risk_reduction_fraction','scenario_expected','scenario_cvar','total_cost','emergency_cost','emergency_kwh','realized_cvar95','worst_day','risk_protection_cost','mean_risk_multiplier']],2)}

风险预算在风险中性解与最小CVaR解之间取五个等距位置。`dual_risk_price.csv` 的
`lambda_risk` 是约束的影子价格；`relative_gap` 验证预算形式和惩罚形式的KKT等价。
它不是从数据中“学出”的用户风险偏好，因此论文应展示前沿，由决策者选点。

## 2/3日滚动规划

{table(roll,2)}

未来第2/3日预测递归生成，组合权重冻结在当日0点；跨日残差使用连续历史块，且历史块末日
严格早于预测起点。每天只执行首日计划并继承实际SOC。1200/6000/10800分别检验保护下界、
50%参考点和保护上界。该模块是题意边界扩展，不自动替代H1主模型。

## 验收

`verification.json` 检查VSS/EVPI非负与序关系、CVaR预算、对偶等价、场景因果性、执行平衡和
每种滚动策略的SOC连续性。有限场景LP可声明全局最优，但不推广为未知真实分布下的全局最优。
"""
    REPORT.write_text(report,encoding='utf-8');print(REPORT)

if __name__=='__main__':main()
