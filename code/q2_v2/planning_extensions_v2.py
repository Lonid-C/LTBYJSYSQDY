"""V2 planning extensions: CVaR frontier, VSS/EVPI and 2/3-day rolling horizon.

All experiments are rebuilt from the V2 forecast/residual bank.  The module never
reads q2_planning_v1 results.  Scenario inputs are kW for the sparse LP, while the
V2 forecast bank and causal executor use kWh/slot.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import argparse, hashlib, json, os, shutil, sys, time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'code'))
from q2_fused import Engine, Config, DEFAULT, cfgdict, lookup, DT, T, BATTERY, cv
from dispatch import execute_day, check_dispatch
from forecast_core import (LOAD_MEMBERS,PV_MEMBERS,_smooth,load_forecast_weekly,
    load_forecast_trend,pv_forecast_mean,pv_forecast_expw,pv_forecast_physics)
from q2_planning import Study, scenario_cost, cvar

OUT=ROOT/'results/q2_fused_v2_extensions'
SEED=20260911
BETAS=(.90,.95)
BUDGET_FRACTIONS=(0.,.25,.50,.75,1.)
FAIR_START=151                 # 2025-06-01: learned configurations begin deployment
ROLL_START=31                  # 2025-02-01
INITIAL=6000.
FAR_TARGETS=(1200.,6000.,10800.)

def atom(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def solver_for(engine):
    """Use only Study.solve; bypass Study.__init__ and every V1 artifact."""
    s=Study.__new__(Study);s.price=np.asarray(engine.price,float)
    return s

def selected(engine):
    obj=json.loads((ROOT/'results/q2_fused_v2/fused_selection.json').read_text())
    records=obj['fused'] if isinstance(obj,dict) else obj
    return lookup(engine,records),records

def _merge_ids(groups):
    mass={}
    for ids,total in groups:
        ids=np.asarray(ids,int)
        if not len(ids) or total<=0:continue
        for i in ids:mass[int(i)]=mass.get(int(i),0.)+float(total)/len(ids)
    ids=np.array(sorted(mass),int);w=np.array([mass[i] for i in ids],float);w/=w.sum()
    return ids,w

def scenario_set(engine,day,cfg,horizon=1,target=None):
    """V2 residual paths with the same global/conditional mixture as deployed V2.

    For H>1 each scenario is a consecutive historical H-day residual block ending
    strictly before the forecast origin, preserving intra- and inter-day dependence.
    """
    latest=day-horizon
    lo=max(1,day-cfg.window-horizon+1)
    global_ids=np.arange(lo,latest+1,dtype=int)
    global_ids=global_ids[np.array([np.isfinite(engine.bank.residual[i:i+horizon]).all() for i in global_ids])]
    blend=float(cfg.blend if cfg.features!='none' else 0.)
    groups=[(global_ids,1-blend)]
    if blend>0:
        candidates=np.arange(max(7,day-120),latest+1,dtype=int)
        dim={'season':2,'weekday':4,'level':6}[cfg.features]
        x=engine.feature[candidates,:dim];scale=x.std(0);scale[scale<1e-9]=1
        dist=np.sum(((x-engine.feature[day,:dim])/scale)**2,axis=1)
        cond=np.sort(candidates[np.argsort(dist,kind='stable')[:min(cfg.k,len(candidates))]])
        groups.append((cond,blend))
    ids,w=_merge_ids(groups)
    assert len(ids) and int(np.max(ids)+horizon-1)<day
    residual=np.stack([engine.bank.residual[i:i+horizon].reshape(-1) for i in ids])
    if target is None:
        assert horizon==1;target=engine.bank.net[day]
    target=np.asarray(target,float).reshape(-1)
    sc_energy=target[None,:]+residual
    assert sc_energy.shape==(len(ids),horizon*T) and np.isfinite(sc_energy).all()
    return sc_energy/DT,w,dict(scenario_days=ids.tolist(),n_scenarios=len(ids),
        effective_n=float(1/(w@w)),max_history_day=int(ids.max()+horizon-1),scenario_blend=blend)

def actual(engine,sol,day,initial,rho):
    q=sol['q'][:T];reference=sol['soc'][:T+1]
    ex=execute_day(q,engine.bank.actual_net[day],initial,reference,rho)
    z=ex['emergency'];price=engine.price;pc=float(price@q);ec=float(5*price@z)
    chk=check_dispatch(engine.bank.actual_net[day],q,ex['charge'],ex['discharge'],ex['spill'],ex['soc'],z)
    assert chk['pass']
    pos=z>1e-6
    return dict(planned_cost=pc,emergency_cost=ec,total_cost=pc+ec,
        adjusted_cost=pc+ec-engine.value*(float(ex['soc'][-1])-initial),
        planned_kwh=float(q.sum()),emergency_kwh=float(z.sum()),spill_kwh=float(ex['spill'].sum()),
        emergency_slots=int(pos.sum()),emergency_day=int(pos.any()),
        events=int(np.sum(pos&~np.r_[False,pos[:-1]])),initial_soc=float(initial),
        terminal_soc=float(ex['soc'][-1]),balance_residual=chk['balance_max_abs'],
        soc_residual=chk['soc_equation_max_abs'])

def common_initials():
    f=pd.read_csv(ROOT/'results/q2_fused_v2/fused_daily.csv')
    return dict(zip(f.day.astype(int),f.initial_soc.astype(float)))

def run_information(engine,solver,cfg_at,days):
    rows=[]
    for d in days:
        cfg=cfg_at(d);sc,w,meta=scenario_set(engine,d,cfg)
        rp=solver.solve(sc,w,initial=INITIAL,terminal=INITIAL)
        ev=solver.solve((w@sc)[None,:],initial=INITIAL,terminal=INITIAL)
        eev=float(w@scenario_cost(solver,ev,sc))
        ws_cost=[]
        for s in sc:ws_cost.append(solver.solve(s[None,:],initial=INITIAL,terminal=INITIAL)['expected'])
        ws=float(w@np.asarray(ws_cost));tol=.02
        assert ws<=rp['expected']+tol and rp['expected']<=eev+tol,(d,ws,rp['expected'],eev)
        rows.append(dict(day=d,date=str(engine.dates[d]),RP=rp['expected'],EEV=eev,WS=ws,
            VSS=eev-rp['expected'],EVPI=rp['expected']-ws,**meta,**asdict(cfg)))
        if pd.Timestamp(engine.dates[d]).is_month_end:
            pd.DataFrame(rows).to_csv(OUT/'information_value.partial.csv',index=False)
            print('information checkpoint',engine.dates[d],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'information_value.csv',index=False)
    return df

def run_risk(engine,solver,cfg_at,days,initials):
    rows=[];duals=[]
    for d in days:
        cfg=cfg_at(d);initial=float(initials[d]);sc,w,meta=scenario_set(engine,d,cfg)
        neutral=solver.solve(sc,w,initial=initial,terminal=INITIAL)
        neutral_cost=scenario_cost(solver,neutral,sc)
        for beta in BETAS:
            risk_min=solver.solve(sc,w,initial=initial,terminal=INITIAL,beta=beta,risk_only=True)
            upper=cvar(neutral_cost,w,beta);lower=float(risk_min['cvar'])
            for k,fraction in enumerate(BUDGET_FRACTIONS):
                budget=float((1-fraction)*upper+fraction*lower)
                sol=(neutral if k==0 else solver.solve(sc,w,initial=initial,terminal=INITIAL,
                                                       beta=beta,budget=budget+1e-5))
                sc_cost=scenario_cost(solver,sol,sc);sc_cvar=cvar(sc_cost,w,beta)
                row=dict(day=d,date=str(engine.dates[d]),beta=beta,budget_index=k,
                    risk_reduction_fraction=fraction,risk_budget=budget,scenario_expected=float(w@sc_cost),
                    scenario_cvar=sc_cvar,risk_multiplier=float(sol.get('risk_multiplier',0.)),
                    effective_tail_n=float((1-beta)/(w@w)),n_scenarios=meta['n_scenarios'],
                    max_history_day=meta['max_history_day'],solve_seconds=sol['seconds'],**actual(engine,sol,d,initial,cfg.rho))
                rows.append(row)
                # Monthly representative KKT check: constrained solution equals penalty form at dual lambda.
                if pd.Timestamp(engine.dates[d]).day==15 and k>0:
                    lam=row['risk_multiplier'];pen=solver.solve(sc,w,initial=initial,terminal=INITIAL,beta=beta,penalty=lam)
                    lhs=row['scenario_expected']+lam*row['scenario_cvar']
                    rhs=pen['expected']+lam*pen['cvar'];gap=abs(lhs-rhs)/max(1,abs(lhs))
                    duals.append(dict(date=row['date'],beta=beta,budget_index=k,lambda_risk=lam,
                        constrained_value=lhs,penalty_value=rhs,relative_gap=gap))
                    assert gap<1e-6,(d,beta,k,gap)
        if pd.Timestamp(engine.dates[d]).is_month_end:
            pd.DataFrame(rows).to_csv(OUT/'cvar_frontier_daily.partial.csv',index=False)
            print('risk checkpoint',engine.dates[d],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'cvar_frontier_daily.csv',index=False)
    pd.DataFrame(duals).to_csv(OUT/'dual_risk_price.csv',index=False)
    return df,pd.DataFrame(duals)

def recursive_target(engine,day,horizon):
    """Causal V2 recursive multi-day forecast; ensemble weights freeze at origin."""
    load=list(np.asarray(engine.data['load'][:day],float));pv=list(np.asarray(engine.data['pv'][:day],float))
    wl=engine.bank.weight_load[day].copy();wp=engine.bank.weight_pv[day].copy();out=[]
    for lead in range(horizon):
        j=day+lead
        if lead==0:
            lh=engine.bank.load_hat[j].copy();ph=engine.bank.pv_hat[j].copy();net=engine.bank.net[j].copy()
        else:
            hl=np.asarray(load);hp=np.asarray(pv);dates=engine.dates[:j];target_date=engine.dates[j]
            lm=[load_forecast_weekly(hl),load_forecast_trend(hl,dates,target_date,28)]
            pm=[pv_forecast_mean(hp,5),pv_forecast_expw(hp,3.,28),pv_forecast_physics(hp)]
            lh=np.maximum(_smooth(np.maximum(np.tensordot(wl,np.stack(lm),axes=1),0)),0)
            ph=np.maximum(_smooth(np.maximum(np.tensordot(wp,np.stack(pm),axes=1),0)),0)
            hist=np.asarray(pv[-14:]);ph[np.max(hist,axis=0)<=0]=0
            net=(lh-ph)*DT                 # future residual innovation has conditional mean zero
        load.append(lh);pv.append(ph);out.append(net)
    return np.concatenate(out)

def run_rolling(engine,solver,cfg_at):
    rows=[];slots=[]
    for horizon in (2,3):
        for far in FAR_TARGETS:
            policy=f'H{horizon}_E{int(far)}';state=INITIAL;rr=[]
            for d in range(ROLL_START,len(engine.dates)):
                h=min(horizon,len(engine.dates)-d);target=recursive_target(engine,d,h)
                cfg=cfg_at(d);sc,w,meta=scenario_set(engine,d,cfg,h,target)
                terminal=INITIAL if d+h==len(engine.dates) else far
                sol=solver.solve(sc,w,initial=state,terminal=terminal)
                metrics=actual(engine,sol,d,state,cfg.rho);state=metrics['terminal_soc']
                row=dict(policy=policy,day=d,date=str(engine.dates[d]),horizon_days=h,
                    far_terminal_soc=terminal,forecast_origin=str(engine.dates[d]),
                    max_history_day=meta['max_history_day'],n_scenarios=meta['n_scenarios'],
                    scenario_expected=sol['expected'],solve_seconds=sol['seconds'],**metrics)
                rr.append(row);rows.append(row)
                slots.append(pd.DataFrame(dict(policy=policy,date=row['date'],slot=np.arange(T),
                    q=sol['q'][:T],reference_soc_start=sol['soc'][:T],reference_soc_end=sol['soc'][1:T+1])))
                assert meta['max_history_day']<d
                if pd.Timestamp(engine.dates[d]).is_month_end:
                    pd.DataFrame(rows).to_csv(OUT/'rolling_daily.partial.csv',index=False)
                    print('rolling checkpoint',policy,engine.dates[d],flush=True)
            starts=np.asarray([r['initial_soc'] for r in rr[1:]]);ends=np.asarray([r['terminal_soc'] for r in rr[:-1]])
            assert np.max(np.abs(starts-ends),initial=0)<1e-7
    df=pd.DataFrame(rows);df.to_csv(OUT/'rolling_daily.csv',index=False)
    pd.concat(slots,ignore_index=True).to_csv(OUT/'rolling_slots.csv.gz',index=False,compression='gzip')
    return df

def summarize(info,risk,duals,rolling):
    isum=info.assign(month=pd.to_datetime(info.date).dt.month).groupby('month').agg(
        days=('day','size'),RP=('RP','sum'),EEV=('EEV','sum'),WS=('WS','sum'),VSS=('VSS','sum'),EVPI=('EVPI','sum')).reset_index()
    isum.to_csv(OUT/'information_monthly.csv',index=False)
    rr=[]
    for (beta,f),g in risk.groupby(['beta','risk_reduction_fraction']):
        rr.append(dict(beta=beta,risk_reduction_fraction=f,days=len(g),planned_cost=g.planned_cost.sum(),
            emergency_cost=g.emergency_cost.sum(),total_cost=g.total_cost.sum(),adjusted_cost=g.adjusted_cost.sum(),
            emergency_kwh=g.emergency_kwh.sum(),emergency_day_frequency=g.emergency_day.mean(),
            mean_daily_cost=g.total_cost.mean(),realized_cvar90=cv(g.total_cost,.90),
            realized_cvar95=cv(g.total_cost,.95),worst_day=g.total_cost.max(),
            scenario_expected=g.scenario_expected.sum(),scenario_cvar=g.scenario_cvar.sum(),
            mean_risk_multiplier=g.risk_multiplier.mean()))
    rsum=pd.DataFrame(rr);rsum.to_csv(OUT/'cvar_frontier_summary.csv',index=False)
    base=rsum[rsum.risk_reduction_fraction==0].set_index('beta')
    rsum['risk_protection_cost']=rsum.apply(lambda r:r.total_cost-base.loc[r.beta,'total_cost'],axis=1)
    rsum.to_csv(OUT/'cvar_frontier_summary.csv',index=False)
    roll=rolling.groupby('policy').agg(days=('day','size'),
        planned_cost=('planned_cost','sum'),emergency_cost=('emergency_cost','sum'),total_cost=('total_cost','sum'),
        adjusted_cost=('adjusted_cost','sum'),emergency_kwh=('emergency_kwh','sum'),
        emergency_day_frequency=('emergency_day','mean'),solve_seconds=('solve_seconds','sum'),
        start_soc=('initial_soc','first'),end_soc=('terminal_soc','last')).reset_index()
    roll['horizon_days']=roll.policy.str.extract(r'H(\d)').astype(int)
    roll['requested_far_terminal_soc']=roll.policy.str.extract(r'_E(\d+)').astype(float)
    roll.to_csv(OUT/'rolling_summary.csv',index=False)
    return isum,rsum,roll

def verify(engine,info,risk,duals,rolling):
    tests=[]
    def add(name,value,tol,mode='max'):
        passed=bool(value<=tol);tests.append(dict(test=name,value=float(value),tolerance=float(tol),passed=passed));assert passed,(name,value)
    add('VSS_nonnegative',max(0,float((-info.VSS).max())),.02)
    add('EVPI_nonnegative',max(0,float((-info.EVPI).max())),.02)
    add('information_order',max(0,float(np.maximum(info.WS-info.RP,info.RP-info.EEV).max())),.02)
    add('risk_budget',max(0,float((risk.scenario_cvar-risk.risk_budget).where(risk.budget_index>0,0).max())),2e-4)
    add('dual_equivalence',float(duals.relative_gap.max()),1e-6)
    add('risk_causality',max(0,float((risk.max_history_day-risk.day+1).max())),0)
    add('rolling_causality',max(0,float((rolling.max_history_day-rolling.day+1).max())),0)
    add('execution_balance',max(float(risk.balance_residual.max()),float(rolling.balance_residual.max())),1e-7)
    for policy,g in rolling.groupby('policy'):
        g=g.sort_values('day');add('soc_continuity_'+policy,float(np.max(np.abs(g.initial_soc.iloc[1:].to_numpy()-g.terminal_soc.iloc[:-1].to_numpy()))),1e-7)
    obj={'all_pass':all(x['passed'] for x in tests),'count':len(tests),'tests':tests}
    atom(OUT/'verification.json',obj);return obj

def pilot():
    engine=Engine();cfg_at,_=selected(engine);solver=solver_for(engine);initials=common_initials()
    days=[165,257]
    info=run_information(engine,solver,cfg_at,days)
    risk,duals=run_risk(engine,solver,cfg_at,days,initials)
    # Direct H2/H3 construction/solve checks without running the whole year.
    checks=[]
    for h in (2,3):
        d=166;cfg=cfg_at(d);target=recursive_target(engine,d,h);sc,w,meta=scenario_set(engine,d,cfg,h,target)
        sol=solver.solve(sc,w,initial=INITIAL,terminal=INITIAL)
        checks.append(dict(horizon=h,n_scenarios=len(sc),max_history_day=meta['max_history_day'],seconds=sol['seconds'],**sol['checks']))
    atom(OUT/'pilot.json',dict(days=days,checks=checks,information_rows=len(info),risk_rows=len(risk),dual_rows=len(duals)))
    print(json.dumps(json.loads((OUT/'pilot.json').read_text()),ensure_ascii=False,indent=2))

def all_run():
    if OUT.exists():shutil.rmtree(OUT)
    OUT.mkdir(parents=True);started=time.time();engine=Engine();cfg_at,selection=selected(engine);solver=solver_for(engine)
    days=range(FAIR_START,len(engine.dates));initials=common_initials()
    info=run_information(engine,solver,cfg_at,days)
    risk,duals=run_risk(engine,solver,cfg_at,days,initials)
    rolling=run_rolling(engine,solver,cfg_at)
    isum,rsum,rollsum=summarize(info,risk,duals,rolling);verification=verify(engine,info,risk,duals,rolling)
    manifest=dict(fresh_run=True,complete=True,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        seconds=time.time()-started,source_sha256=sha(__file__),q2_planning_sha256=sha(ROOT/'code/q2_planning.py'),
        raw_input_sha256={p.name:sha(p) for p in (ROOT/'C题/附件').glob('*.xlsx')},
        evaluation_period='2025-06-01--2025-12-31 for CVaR/VSS; 2025-02-01--2025-12-31 for H2/H3',
        betas=BETAS,budget_fractions=BUDGET_FRACTIONS,far_targets=FAR_TARGETS,
        scenario_rule='V2 global/conditional residual mixture; full consecutive paths; causal only',
        planning_solver='finite-scenario sparse LP via HiGHS',verification_count=verification['count'])
    atom(OUT/'run_manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','all'],default='all');a=ap.parse_args()
    (OUT).mkdir(parents=True,exist_ok=True)
    pilot() if a.stage=='pilot' else all_run()
