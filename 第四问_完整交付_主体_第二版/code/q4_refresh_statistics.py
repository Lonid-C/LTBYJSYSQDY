"""Rebuild evaluation statistics from saved daily costs, without refitting policies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from q4_statistics import empirical_cvar,paired_bootstrap
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results4'

def refresh():
    oos=pd.read_csv(R/'q4_out_of_sample.csv')
    for i,r in oos.iterrows():
        f=R/f'q4_daily_{r.problem}_{r.method.split()[0]}.csv'
        daily=pd.read_csv(f)
        oos.loc[i,'cvar95']=empirical_cvar(daily.total_cost.to_numpy())
        oos.loc[i,'top17_mean']=np.sort(daily.total_cost.to_numpy())[-17:].mean()
        if str(r.method).startswith('U'):
            oos.loc[i,'note']='冻结策略的完美价格替换对照；不是理论信息价值上界'
    oos.to_csv(R/'q4_out_of_sample.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    lb=pd.read_csv(R/'q4_full_horizon_bound.csv')
    info=[]
    for prob,mainpref,basepref in [('Q4-2','M2','M2'),('Q4-3','H3','B0')]:
        def pick(pref): return float(oos[(oos.problem==prob)&oos.method.str.startswith(pref)].adjusted_cost.iloc[0])
        base,main,hyb,price,rolling,fixed=(pick(p) for p in [basepref,mainpref,'H'+prob[-1],'U'+prob[-1],'W'+prob[-1],'B1'])
        lower=float(lb[lb.problem==prob].adjusted_cost.iloc[0])
        info.append(dict(problem=prob,baseline_deterministic=base,main_strategy=main,
            hybrid_stochastic=hyb,perfect_price=price,rolling_perfect_information=rolling,
            full_horizon_lower_bound=lower,price_assumed_fixed=fixed,
            hybrid_saving_vs_quantile=base-hyb,hybrid_saving_percent=100*(base-hyb)/base,
            frozen_price_replacement_saving=main-price,
            frozen_price_replacement_percent=100*(main-price)/main,
            rolling_net_load_replacement_difference=price-rolling,
            rolling_to_full_horizon_gap=rolling-lower,gap_to_lower_bound=main-lower,
            gap_to_lower_bound_percent=100*(main-lower)/main,
            gap_denominator='main_strategy',interpretation='historical policy comparison, not theoretical VSS/EVPI'))
    pd.DataFrame(info).to_csv(R/'q4_information_value.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    rows=[]
    for prob,a,b,an,bn in [
        ('Q4-2','B0','M2','B0 分位法 α=0.60','M2 分位法 α 重标定'),
        ('Q4-2','M2','H2','M2 主策略','H2 HYBRID'),
        ('Q4-3','B0','H3','B0 分位法 α=0.60','H3 HYBRID 主策略')]:
        da=pd.read_csv(R/f'q4_daily_{prob}_{a}.csv'); db=pd.read_csv(R/f'q4_daily_{prob}_{b}.csv')
        assert da.date.tolist()==db.date.tolist()
        for block in [7,14,28]:
            r=paired_bootstrap(da.total_cost,db.total_cost,block)
            r.update(problem=prob,reference=an,variant=bn)
            rows.append(r)
    pd.DataFrame(rows).to_csv(R/'q4_cvar_bootstrap.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    f=R/'q4_experiment_summary.json'; s=json.loads(f.read_text())
    s.update(out_of_sample=oos.to_dict('records'),information_value=info,cvar_bootstrap=rows)
    s['risk_evaluation_note']='exact empirical CVaR95; 334 equal-probability daily losses; MBB has all valid starts and length 334; seasonal dependence remains a limitation'
    f.write_text(json.dumps(s,ensure_ascii=False,indent=2,default=float))
    print(pd.DataFrame(rows)[['problem','variant','block_len','diff','ci_lo','ci_hi']].to_string(index=False),flush=True)
    return info,rows

if __name__=='__main__': refresh()
