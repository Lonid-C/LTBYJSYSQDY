"""只从本次新生成的 q2_fused_v2 输出构建图表与结果报告。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from utils import ROOT, RESULTS, FIGURES, REPORTS, markdown_table


def setup_style():
    for path in Path('/usr/share/fonts').rglob('*CJK*.ttc'):
        try: font_manager.fontManager.addfont(str(path))
        except Exception: pass
    names=[x.name for x in font_manager.fontManager.ttflist
           if 'Noto Sans CJK' in x.name or 'Source Han Sans' in x.name]
    plt.rcParams.update({'font.family':names[0] if names else 'DejaVu Sans',
                         'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10,
                         'figure.dpi':130,'savefig.bbox':'tight'})


def save(fig,name,meaning,manifest):
    for suffix in ['png','pdf']:
        fig.savefig(FIGURES/f'{name}.{suffix}',dpi=220 if suffix=='png' else None)
    plt.close(fig);manifest.append({'figure':name,'meaning':meaning})


def main():
    setup_style();FIGURES.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    required=['summary.csv','monthly.csv','bootstrap.csv','validation_diagnostics.csv',
              'terminal_target_sensitivity.csv','inventory_value_sensitivity.csv',
              'fused_selection.json','run_manifest.json','verification.json']
    missing=[x for x in required if not (RESULTS/x).exists()]
    if missing:raise FileNotFoundError(f'本次 V2 输出不完整: {missing}')
    manifest=[];summary=pd.read_csv(RESULTS/'summary.csv');monthly=pd.read_csv(RESULTS/'monthly.csv')
    boot=pd.read_csv(RESULTS/'bootstrap.csv');valid=pd.read_csv(RESULTS/'validation_diagnostics.csv')
    target=pd.read_csv(RESULTS/'terminal_target_sensitivity.csv');inventory=pd.read_csv(RESULTS/'inventory_value_sensitivity.csv')
    select=json.loads((RESULTS/'fused_selection.json').read_text())

    # 1. 全年成本分解和风险。
    keep=['ridge_h1_replan','baseline','user','fused','no_conditional','search_only']
    q=summary[(summary.period=='full334')&summary.strategy.isin(keep)].set_index('strategy').loc[[x for x in keep if x in set(summary.strategy)]]
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    ax[0].bar(q.index,q.planned_cost/1e6,label='计划购电费')
    ax[0].bar(q.index,q.emergency_cost/1e6,bottom=q.planned_cost/1e6,label='紧急购电费')
    ax[0].set_ylabel('费用（百万元）');ax[0].set_title('全年总费用分解');ax[0].legend();ax[0].tick_params(axis='x',rotation=28)
    x=np.arange(len(q));w=.36
    ax[1].bar(x-w/2,q.daily_cvar95/1e3,w,label='CVaR95')
    ax[1].bar(x+w/2,q.worst_day/1e3,w,label='最坏日')
    ax[1].set_xticks(x,q.index,rotation=28);ax[1].set_ylabel('日费用（千元）');ax[1].set_title('尾部风险与最坏日');ax[1].legend()
    fig.tight_layout();save(fig,'01_cost_and_risk','同时展示计划费、紧急费、CVaR95 和最坏日，避免只看总费用。',manifest)

    # 2. 与 Ridge+H1 的月度差额。
    p=monthly[monthly.strategy.isin(['ridge_h1_replan','fused','baseline'])].pivot(index='month',columns='strategy',values='total_cost')
    fig,ax=plt.subplots(figsize=(10,4.2))
    for name in ['fused','baseline']:
        ax.plot(p.index,(p[name]-p['ridge_h1_replan'])/1e3,marker='o',label=f'{name} - Ridge+H1')
    ax.axhline(0,color='black',lw=.8);ax.set_ylabel('月费用差（千元）');ax.set_title('月度稳定性：正值表示高于 Ridge+H1-replan')
    ax.tick_params(axis='x',rotation=30);ax.legend();fig.tight_layout();save(fig,'02_monthly_stability','检查优势是否只来自少数月份。',manifest)

    # 3. 参数更新轨迹。
    sf=pd.DataFrame([{'month':r['month'],**r['config'],'accepted':r['accepted']} for r in select['fused']])
    fig,ax=plt.subplots(2,2,figsize=(10,6.6),sharex=True)
    for a,col in zip(ax.ravel(),['alpha','rho','window','blend']):
        a.step(sf.month,sf[col],where='mid',marker='o');a.set_title(col);a.grid(alpha=.25)
        for _,r in sf.iterrows():
            if not r.accepted:a.scatter(r.month,r[col],marker='x',s=60,color='crimson')
    fig.suptitle('融合策略月度参数更新轨迹（× 为未通过验收）');fig.tight_layout()
    save(fig,'03_parameter_path','展示参数是否频繁波动、更新是否被后置验收拒绝。',manifest)

    # 4. 14日CVaR退化与56日风险门。
    v=valid[valid.strategy=='fused'].sort_values('month')
    fig,ax=plt.subplots(1,2,figsize=(11,4.1))
    ax[0].plot(v.month,v.validation_new_emergency_cvar95/1e3,marker='o',label='14日 CVaR95')
    ax[0].plot(v.month,v.validation_new_worst_emergency/1e3,marker='x',ls='--',label='14日 worst-day')
    ax[0].set_title('14日窗：CVaR95 退化为最坏日');ax[0].set_ylabel('紧急费（千元/日）');ax[0].legend()
    ax[1].plot(v.month,v.old_emergency_cvar95/1e3,marker='o',label='原策略')
    ax[1].plot(v.month,v.new_emergency_cvar95/1e3,marker='o',label='候选策略')
    ax[1].set_title('56日风险窗 CVaR95（尾部2.8日）');ax[1].legend()
    for a in ax:a.set_xlabel('部署月份');a.grid(alpha=.25)
    fig.tight_layout();save(fig,'04_validation_and_cvar','直观证明14日CVaR95信息量不足，并展示用于验收的56日风险证据。',manifest)

    # 5. 参考终端 SOC 实际重跑。
    t=target[target.inventory_valuation=='replacement']
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for name,g in t.groupby('strategy'):
        g=g.sort_values('sensitivity_target');ax[0].plot(g.sensitivity_target,g.total_cost/1e6,marker='o',label=name)
        ax[1].plot(g.sensitivity_target,g.emergency_kwh/1e3,marker='o',label=name)
    ax[0].set_ylabel('总费用（百万元）');ax[1].set_ylabel('紧急购电量（MWh）')
    for a in ax:a.set_xlabel(r'$E_{target}$ (kWh)');a.grid(alpha=.25)
    ax[0].legend();ax[0].set_title('费用对参考终点的敏感性');ax[1].set_title('紧急购电对参考终点的敏感性')
    fig.tight_layout();save(fig,'05_terminal_target_sensitivity','每个Etarget均重新规划与因果执行，而非事后换算。',manifest)

    # 6. 期末库存价值口径。
    iv=inventory[(inventory.period=='full334')&inventory.strategy.isin(['ridge_h1_replan','baseline','fused'])]
    order=['zero','replacement','avoidance_upper'];pp=iv.pivot(index='strategy',columns='inventory_valuation',values='adjusted_cost')
    fig,ax=plt.subplots(figsize=(9,4.2));x=np.arange(len(pp));w=.25
    for j,label in enumerate(order):ax.bar(x+(j-1)*w,pp[label]/1e6,w,label=label)
    ax.set_xticks(x,pp.index);ax.set_ylabel('期末价值调整后费用（百万元）');ax.set_title('期末库存价值三口径敏感性');ax.legend();fig.tight_layout()
    save(fig,'06_inventory_value_sensitivity','说明期末SOC价值是核算假设，不是凭空获得的唯一真值。',manifest)

    # 7. 与 Ridge 的配对区间。
    b=boot[(boot.reference=='ridge_h1_replan')&(boot.block==7)]
    fig,ax=plt.subplots(figsize=(8.5,4.1));x=np.arange(len(b));y=b.daily_saving_mean
    ax.errorbar(x,y,yerr=np.vstack([y-b.daily_saving_low95,b.daily_saving_high95-y]),fmt='o',capsize=5)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,b.period);ax.set_ylabel('Ridge+H1 减 V2（元/日）')
    ax.set_title('7日移动块 Bootstrap 95% 区间');fig.tight_layout()
    save(fig,'07_bootstrap_vs_ridge','区分观测差额与在周相关下仍可支持的稳健差异。',manifest)

    (FIGURES/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    full=summary[summary.period=='full334'].sort_values('total_cost')
    ridge=full[full.strategy=='ridge_h1_replan'].iloc[0];fused=full[full.strategy=='fused'].iloc[0]
    vb=valid[valid.strategy=='fused'][['month','accepted','relative_improvement','bootstrap_daily_low90',
      'bootstrap_daily_high90','validation_new_emergency_cvar95','validation_new_worst_emergency',
      'new_emergency_cvar95','risk_tail_mass_days']]
    target_main=t[['strategy','sensitivity_target','total_cost','adjusted_cost','emergency_kwh','daily_cvar95','terminal_soc']]
    report=f"""# Q2 参数来源增强 V2：全年重跑结果

> 本报告只读取 `results/q2_fused_v2` 本次运行的输出。`run_manifest.json` 声明
> `fresh_run=true` 且未读取 V1 daily/summary 结果。Ridge 预测明细被作为已冻结比较器输入，
> 其 H1-replan 规划、SOC 轨迹和费用已全部重新求解。

## 核心结果

{markdown_table(full[['strategy','total_cost','planned_cost','emergency_cost','emergency_kwh','daily_cvar95','worst_day','adjusted_cost']],2)}

V2 fused 全年实际费用为 **{fused.total_cost:,.2f} 元**，冻结 Ridge+H1-replan 比较器为
**{ridge.total_cost:,.2f} 元**，实际差额（Ridge 减 V2）为 **{ridge.total_cost-fused.total_cost:,.2f} 元**。
该句只是同日回测差额；是否能推荐替换，还要同时看开发期、冻结期和 Bootstrap 区间。

## `near()` 与验证门

原固定 0.1% 近似并列带宽已删除。现对搜索期候选与搜索期最低费用候选的逐日费用差
作 7 日循环移动块 Bootstrap；若“候选减最优”均值的 90% 区间下界不大于0，
则尚无证据说明它更差，才进入简约性打破平局集合。这只是搜索段正则化，部署仍需独立的
14 日费用改善证据和 56 日 CVaR95 非劣性。

{markdown_table(vb,4)}

14 日时，$14(1-0.95)=0.7<1$，故精确经验 CVaR95 必然等于最坏日，不能提供独立尾部信息。
56 日窗有 $56(1-0.95)=2.8$ 个等权尾部日，因此改用它检查风险非劣性。

## 期末 SOC 与库存价值

$E_{{target}}$ 是日前 LP 参考 SOC 轨迹的终点，不是实际执行必须回到的日末 SOC；
实际 SOC 逐日连续继承。主核算使用 $v_E=p_{{min}}/\\eta_c$，即以最低电价补回一单位电池内能量的成本；
同时报告 $v_E=0$ 和 $v_E=\\eta_d p_{{max}}$。它们是经济核算边界，不冒充为数据学习的风险偏好。

{markdown_table(target_main,2)}

## 稳健性与可复现性

- `bootstrap.csv` 给出 V2 相对 Ridge+H1-replan、固定 A 和原分阶段方案的 7/14/28 日块区间。
- `terminal_target_daily.csv.gz` 保留 5 个 $E_{{target}}$ 的逐日实际重跑轨迹。
- `inventory_value_sensitivity.csv` 保留三种库存价值口径。
- `verification.json` 检查时间前缀不变性、LP/执行物理可行性、SOC 连续性、费用重算、风险窗和参数冻结。
- 所有“全局最优”只针对已给定有限样本的线性规划，不外推到未知真实分布。
"""
    (REPORTS/'Q2_FUSED_V2_RESULTS.md').write_text(report,encoding='utf-8')
    print(REPORTS/'Q2_FUSED_V2_RESULTS.md')


if __name__=='__main__':main()
