"""Rerun the affected settlement experiment and test billing invariance."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from dataclasses import replace
import json,time
import numpy as np
import pandas as pd
import q4_policy as Q
import run_q4_experiment as E
from q4_statistics import empirical_cvar,moving_block_indices
R=E.R

def main():
    g=E.prepare()
    sel=json.loads((R/'q4_selected_policy.json').read_text())
    pol=E.policy_from(sel['Q4-3']['x']); p=replace(g['p'],settlements='stepwise')
    def run(detail,end=365,policy=pol,param=p,start=31):
        return Q.run_q4(g['L'],g['P'],g['V'],g['bank'],g['rb'],g['pbank'],param,policy,
            E.G['shape'],E.G['edges'],g['vE_ref'],mask=(6,12,18),start=start,end=end,
            initial=E.G['init_q4_3'],mode='stoch',detail=detail)
    checks={}
    # detail flag must not change any daily result, including rejected/accepted gates.
    for settlement in ['stepwise','net_refund']:
        for threshold in [0.,1e12]:
            args=dict(end=33,policy=replace(pol,threshold=threshold),param=replace(g['p'],settlements=settlement))
            a,_,_,_=run(False,**args); b,_,_,tx=run(True,**args)
            err=float(np.abs(a.select_dtypes('number').to_numpy()-b.select_dtypes('number').to_numpy()).max())
            assert err<1e-8,(settlement,threshold,err)
            checks[f'detail_invariance_{settlement}_{threshold:g}']=err
    assert abs(empirical_cvar([1,2,3,4],.625)-(4+.5*3)/1.5)<1e-12
    assert empirical_cvar([7,7,7],.95)==7
    rng=np.random.default_rng(4)
    for block in [7,14,28]:
        ix=np.array([moving_block_indices(334,block,rng) for _ in range(100)])
        assert ix.shape==(100,334) and ix.min()>=0 and ix.max()==333
    checks['empirical_cvar_and_block_sampling']='passed'
    print('Regression checks passed; starting 334-day stepwise rerun',flush=True)
    t=time.time(); df,iv,dc,tx=run(True)
    for suffix,frame in [('daily',df),('intervals',iv),('decisions',dc),('transactions',tx)]:
        frame.to_csv(R/f'q4_stepwise_{suffix}.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    # Transactions independently reproduce every daily adjustment fee.
    fees=tx.groupby('date').fee_attributed.sum().reindex(df.date,fill_value=0).to_numpy()
    err=float(np.max(np.abs(fees-df.up_cost.to_numpy()-df.down_net_cost.to_numpy())))
    assert err<1e-6
    grouped=iv.groupby('date')
    netfee=grouped.apply(lambda z: float((z.price*(1.5*np.maximum(z.adjusted-z.plan,0)-.5*np.maximum(z.plan-z.adjusted,0))).sum()))
    assert np.min(fees-netfee.reindex(df.date).to_numpy())>-1e-6
    checks['stepwise_ledger_daily_max_error']=err
    checks['stepwise_ge_net_for_same_plan']=True
    checks['stepwise_days']=len(df); checks['stepwise_transactions']=len(tx)
    s=Q.summarize(df,block='C 结算口径',case='逐次调整分别计费',
        changed="settlements='stepwise'; prior accepted plan as baseline; billing independent of detail",seconds=time.time()-t)
    abl=pd.read_csv(R/'q4_ablation_sensitivity.csv')
    # Old rows without stored daily losses retain their explicitly named top-17 statistic.
    if 'cvar95' in abl:
        abl=abl.rename(columns={'cvar95':'top17_mean'})
    abl=abl[~((abl.block=='C 结算口径')&(abl.case=='逐次调整分别计费'))]
    abl=pd.concat([abl,pd.DataFrame([s])],ignore_index=True)
    abl.to_csv(R/'q4_ablation_sensitivity.csv',index=False,encoding='utf-8-sig',float_format='%.10f')
    # Fixed existing main-policy transactions: isolate the accounting effect from changed decisions.
    main=pd.read_csv(R/'q4_3_daily.csv'); oldtx=pd.read_csv(R/'q4_3_transactions.csv')
    delta=oldtx.delta.to_numpy(); pr=oldtx.delivery_price.to_numpy()
    fee=float(pr@(1.5*np.maximum(delta,0)-.5*np.maximum(-delta,0)))
    cash=float(main.plan_cost.sum()+main.emergency_cost.sum()+fee)
    checks['fixed_main_plan_stepwise_cash']=cash
    checks['fixed_main_plan_stepwise_adjusted']=cash+g['vE_ref']*(main.soc_start.iloc[0]-main.soc_end.iloc[-1])
    checks['fixed_main_plan_stepwise_minus_net']=fee-float(main.up_cost.sum()+main.down_net_cost.sum())
    checks['rerun_stepwise_summary']=s
    (R/'q4_repair_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=float))
    print(json.dumps(checks,ensure_ascii=False,indent=2,default=float),flush=True)

if __name__=='__main__': main()
