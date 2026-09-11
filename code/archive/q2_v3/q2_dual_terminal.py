"""Q2 V3: Ridge + conditional weighted SAA + dual terminal value + H1 execution.

The experiment is causal.  Every decision for day d uses only rows with index < d.
No daily or remote hard terminal-SOC equality is used by the V3 policy.  A periodic
relative-value iteration learns a convex piecewise-linear continuation value from
past net-load paths; LP equality duals supply its marginal slopes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import argparse, hashlib, itertools, json, os, shutil, sys, time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack, csr_matrix
from sklearn.isotonic import IsotonicRegression

def project_root(start):
    """按只在仓库根存在的文件向上查找项目根。

    使本模块不依赖自身在 code/ 下的深度，归档移动后仍可运行。
    """
    for candidate in (start, *start.parents):
        if (candidate / 'C题/附件/附件2.xlsx').exists():
            return candidate
    raise FileNotFoundError('未找到含 C题/附件/附件2.xlsx 的项目根目录')


HERE = Path(__file__).resolve().parent
ROOT = project_root(HERE)
sys.path.insert(0, str(ROOT/'code'))
sys.path.insert(0, str(ROOT/'code/q2_v2'))

from q2_planning import Study, weights_for
from dispatch import execute_day, check_dispatch
from utils import BATTERY, DT, T, INTERVALS

OUT = ROOT/'results/q2_v3_dual_terminal'
FIG = ROOT/'figures/q2_v3_dual_terminal'
REPORT = ROOT/'reports/Q2_V3_DUAL_TERMINAL_RESULTS.md'
SEED = 20260912
LO, HI, INITIAL = 1200., 10800., 6000.
CAP = 5000.*DT
EPS = 1e-8
SEARCH_DAYS, VALIDATION_DAYS, EMBARGO_DAYS = 42, 14, 1
BLOCK, BOOT_DRAWS = 7, 2000
WINDOWS = (14, 21, 28, 42, 56)
HALVES = (7, 14, 28, None)
FEATURES = ('season', 'weekday', 'level')
NEIGHBORS = (14, 21, 28, 42)
BLENDS = (0., .25, .5, .75, 1.)
RHOS = (0., .25, .5, .75, 1.)
VALUE_GRID = np.linspace(LO, HI, 9)  # numerical mesh; convergence is checked against 17 nodes
VALUE_TOL_YUAN = .10
VALUE_MAX_ITER = 40


def atom(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


@dataclass(frozen=True)
class Config:
    window:int=21
    half:int|None=14
    features:str='none'
    k:int=21
    blend:float=0.
    rho:float=.5


DEFAULT=Config()


def cfgdict(x):
    return Config(**{k:x[k] for k in Config.__dataclass_fields__})


def block_ci(x,seed=SEED,ci=.95):
    x=np.asarray(x,float);n=len(x);b=min(BLOCK,n);rng=np.random.default_rng(seed)
    m=int(np.ceil(n/b));starts=rng.integers(0,n,size=(BOOT_DRAWS,m))
    ids=((starts[:,:,None]+np.arange(b))%n).reshape(BOOT_DRAWS,-1)[:,:n]
    z=x[ids].mean(1);a=(1-ci)/2
    return float(x.mean()),float(np.quantile(z,a)),float(np.quantile(z,1-a))


def value_at(cuts, e):
    if not cuts:return 0.
    return float(max(a*float(e)+b for a,b in cuts))


def convex_cuts(grid, values):
    """Convex, non-increasing PWL fit; 6000 is only the additive zero reference."""
    grid=np.asarray(grid,float);values=np.asarray(values,float)
    slopes=np.diff(values)/np.diff(grid)
    slopes=IsotonicRegression(increasing=True,y_max=0.).fit_transform(np.arange(len(slopes)),slopes)
    fitted=np.r_[0.,np.cumsum(slopes*np.diff(grid))]
    fitted-=np.interp(INITIAL,grid,fitted)
    cuts=[(float(m),float(fitted[j]-m*grid[j])) for j,m in enumerate(slopes)]
    return fitted,cuts


class V3:
    def __init__(self):
        self.study=Study(ROOT)
        self.price=self.study.price
        self.dates=self.study.dates
        self.load,self.pv,self.net=self.study.load,self.study.pv,self.study.net
        self.fc=self.study.fc['Ridge'].copy()
        # Preserve the already-audited cold-start rule: no actual-value substitution.
        self.fc[:,:90]=self.study.legacy[:,:90]
        self.netfc=self.fc[0]-self.fc[1]
        doy=self.dates.dayofyear.to_numpy();dow=self.dates.dayofweek.to_numpy()
        self.context=np.column_stack([np.sin(2*np.pi*doy/365),np.cos(2*np.pi*doy/365),
            np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),
            self.fc[0].sum(1)*DT,self.fc[1].sum(1)*DT,
            self.netfc.max(1),self.netfc.min(1)])
        self.solve_calls=0

    def scenarios(self,d,cfg):
        lo=max(7,d-cfg.window);gids=np.arange(lo,d,dtype=int)
        mass={int(i):float(w) for i,w in zip(gids,weights_for(len(gids),cfg.half))}
        if cfg.features!='none' and cfg.blend>0:
            ids=np.arange(max(31,d-120),d,dtype=int)
            dim={'season':2,'weekday':4,'level':8}[cfg.features]
            x=self.context[ids,:dim];sd=x.std(0);sd[sd<1e-9]=1.
            distance=np.sqrt(np.sum(((x-self.context[d,:dim])/sd)**2,axis=1))
            order=np.argsort(distance,kind='stable')[:min(cfg.k,len(ids))]
            ids,dist=ids[order],distance[order]
            bandwidth=max(float(np.median(dist[dist>0])) if np.any(dist>0) else 1.,1e-9)
            local=np.exp(-.5*(dist/bandwidth)**2);local/=local.sum()
            for i in list(mass):mass[i]*=(1-cfg.blend)
            for i,w in zip(ids,local):mass[int(i)]=mass.get(int(i),0.)+cfg.blend*float(w)
        ids=np.array(sorted(mass),int);w=np.array([mass[i] for i in ids],float);w/=w.sum()
        residual=self.net[ids]-self.netfc[ids]
        sc=self.netfc[d][None,:]+residual
        assert ids.max()<d and np.isfinite(sc).all() and abs(w.sum()-1)<1e-10
        return sc,w,dict(min_history_day=int(ids.min()),max_history_day=int(ids.max()),
            n_scenarios=len(ids),effective_n=float(1/(w@w)))

    def solve(self,scen,w,initial,cuts=None,terminal=None):
        """Risk-neutral sparse LP with fixed terminal OR free terminal plus PWL value."""
        scen=np.atleast_2d(np.asarray(scen,float));S,H=scen.shape
        assert H%T==0 and np.isfinite(scen).all()
        w=np.asarray(w,float);w=w/w.sum();p=np.tile(self.price,H//T)
        use_value=cuts is not None
        n0=4*H+S*H;n=n0+int(use_value);theta=n0 if use_value else None
        t=np.arange(H);er=np.r_[t,t,t,t[1:]];ec=np.r_[H+t,2*H+t,3*H+t,3*H+t[:-1]]
        ev=np.r_[np.full(H,-BATTERY.eta_c),np.full(H,1/BATTERY.eta_d),np.ones(H),-np.ones(H-1)]
        Aeq=coo_matrix((ev,(er,ec)),shape=(H,n)).tocsr();beq=np.zeros(H);beq[0]=initial
        k=np.arange(S*H);slots=np.tile(t,S)
        Aub=coo_matrix((np.tile([-1.,1.,-1.,-1.],(S*H,1)).ravel(),
            (np.repeat(k,4),np.column_stack([slots,H+slots,2*H+slots,4*H+k]).ravel())),shape=(S*H,n)).tocsr()
        bub=-(scen*DT).ravel()
        if use_value:
            cr=[];cc=[];cv=[]
            for j,(a,b) in enumerate(cuts):cr.extend([j,j]);cc.extend([4*H-1,theta]);cv.extend([a,-1.])
            C=coo_matrix((cv,(cr,cc)),shape=(len(cuts),n)).tocsr()
            Aub=vstack([Aub,C],format='csr');bub=np.r_[bub,[-b for a,b in cuts]]
        obj=np.r_[p,np.full(2*H,EPS),np.zeros(H),(5*w[:,None]*p).ravel(),[1.] if use_value else []]
        eb=[(LO,HI)]*H
        if terminal is not None:eb[-1]=(float(terminal),float(terminal))
        bounds=[(0,None)]*H+[(0,CAP)]*(2*H)+eb+[(0,None)]*(S*H)+([(None,None)] if use_value else [])
        started=time.perf_counter();ans=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,
            method='highs',options={'presolve':True,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
        if not ans.success:
            ans=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs-ds',options={'presolve':False})
        seconds=time.perf_counter()-started;self.solve_calls+=1
        if not ans.success:raise RuntimeError(f'LP failed {ans.status}: {ans.message}')
        x=ans.x;q,c,v=x[:H].copy(),x[H:2*H].copy(),x[2*H:3*H].copy();soc=np.r_[initial,x[3*H:4*H]]
        raw=np.minimum(c,v).clip(0)
        if raw.max(initial=0)>1e-7:
            remove=np.minimum(c,v/BATTERY.eta_c**2).clip(0);c-=remove;v-=BATTERY.eta_c**2*remove
        emergency=np.maximum(scen*DT+c-v-q,0);physical=float(p@q+w@(emergency@(5*p)))
        tv=value_at(cuts,soc[-1]) if use_value else 0.
        eq=float(np.abs(Aeq@x-beq).max());ineq=float(max(0,np.max(Aub@x-bub)))
        assert eq<1e-5 and ineq<1e-5 and LO-1e-6<=soc.min()<=soc.max()<=HI+1e-6
        return dict(q=q,c=c,v=v,soc=soc,expected=physical,terminal_value=tv,objective=physical+tv,
            initial_dual=float(ans.eqlin.marginals[0]),seconds=seconds,eq_residual=eq,
            ineq_violation=ineq,solver_status=int(ans.status),simultaneous_kwh=float(raw.sum()))

    def run_interval(self,cfg,start,stop,initial=INITIAL):
        """Selection score under the common legacy terminal boundary; not the V3 final policy."""
        state=float(initial);rows=[]
        for d in range(start,stop):
            sc,w,_=self.scenarios(d,cfg);sol=self.solve(sc,w,state,terminal=INITIAL)
            ex=execute_day(sol['q'],self.net[d]*DT,state,sol['soc'],cfg.rho)
            pc=float(self.price@sol['q']);ec=float(5*self.price@ex['emergency'])
            rows.append(pc+ec);state=float(ex['soc'][-1])
        return np.asarray(rows)


def complexity(cfg):
    return (cfg.features!='none',cfg.window,cfg.half is not None,cfg.k,cfg.blend,abs(cfg.rho-.5))


def choose_search(v3,candidates,start,stop,seed):
    records=[]
    for cfg in candidates:
        daily=v3.run_interval(cfg,start,stop)
        records.append(dict(cfg=cfg,daily=daily,mean=float(daily.mean())))
    best=min(records,key=lambda x:x['mean']);eligible=[]
    for r in records:
        mean,lo,hi=block_ci(r['daily']-best['daily'],seed)
        r.update(delta_mean=mean,ci_low=lo,ci_high=hi,indistinguishable=bool(lo<=0))
        if lo<=0:eligible.append(r)
    chosen=min(eligible,key=lambda x:(complexity(x['cfg']),x['mean']))
    return chosen,records


def monthly_selection(v3):
    path=OUT/'selection.json';score_path=OUT/'selection_scores.csv'
    if path.exists():return json.loads(path.read_text())
    incumbent=DEFAULT;selected=[];scores=[]
    for month in range(6,11):
        origin=int(np.flatnonzero(v3.dates==pd.Timestamp(2025,month,1))[0])
        search_stop=origin-VALIDATION_DAYS-EMBARGO_DAYS;search_start=search_stop-SEARCH_DAYS
        val_start=origin-VALIDATION_DAYS-EMBARGO_DAYS;val_stop=origin-EMBARGO_DAYS
        global_candidates=[Config(w,h,rho=incumbent.rho) for w,h in itertools.product(WINDOWS,HALVES)]
        g,gr=choose_search(v3,global_candidates,search_start,search_stop,SEED+month*10)
        cond_candidates=[g['cfg']]+[replace(g['cfg'],features=f,k=k,blend=b) for f,k,b in itertools.product(FEATURES,NEIGHBORS,BLENDS[1:])]
        c,cr=choose_search(v3,cond_candidates,search_start,search_stop,SEED+month*10+1)
        rho_candidates=[replace(c['cfg'],rho=r) for r in RHOS]
        p,pr=choose_search(v3,rho_candidates,search_start,search_stop,SEED+month*10+2)
        proposal=p['cfg'];new=v3.run_interval(proposal,val_start,val_stop);old=v3.run_interval(incumbent,val_start,val_stop)
        mean,lo,hi=block_ci(old-new,SEED+month*100)
        accepted=bool(lo>0);chosen=proposal if accepted else incumbent
        selected.append(dict(month=month,origin=str(v3.dates[origin].date()),search_start=str(v3.dates[search_start].date()),
            search_end=str(v3.dates[search_stop-1].date()),validation_start=str(v3.dates[val_start].date()),
            validation_end=str(v3.dates[val_stop-1].date()),embargo_date=str(v3.dates[origin-1].date()),
            incumbent=asdict(incumbent),proposal=asdict(proposal),config=asdict(chosen),accepted=accepted,
            validation_saving_mean=mean,validation_ci_low95=lo,validation_ci_high95=hi))
        for stage,rows in [('global',gr),('conditional',cr),('rho',pr)]:
            for r in rows:scores.append(dict(month=month,stage=stage,mean_cost=r['mean'],delta_mean=r['delta_mean'],
                ci_low95=r['ci_low'],ci_high95=r['ci_high'],indistinguishable=r['indistinguishable'],**asdict(r['cfg'])))
        incumbent=chosen;atom(path,selected);pd.DataFrame(scores).to_csv(score_path,index=False)
        print('selection',month,asdict(chosen),'accepted',accepted,flush=True)
    return selected


def config_at(v3,selection,d):
    m=v3.dates[d].month
    if m<6:return DEFAULT
    row=next(x for x in selection if x['month']==min(m,10))
    return cfgdict(row['config'])


def weekday_distribution(v3,origin,weekday,window=84):
    ids=np.arange(max(7,origin-window),origin-EMBARGO_DAYS)
    ids=ids[v3.dates[ids].dayofweek==weekday]
    if len(ids)<3:
        ids=np.arange(max(7,origin-56),origin-EMBARGO_DAYS)
    w=weights_for(len(ids),28 if len(ids)>7 else None)
    return v3.net[ids],w,ids


def learn_value_bank(v3,origin,grid=VALUE_GRID,max_iter=VALUE_MAX_ITER,tol=VALUE_TOL_YUAN):
    """Periodic relative-value iteration over seven weekday states, no hard terminal SOC."""
    grid=np.asarray(grid,float);values={u:np.zeros(len(grid)) for u in range(7)};history=[];duals=[]
    for iteration in range(1,max_iter+1):
        updated={}
        for u in range(6,-1,-1):
            next_u=(u+1)%7
            next_values=updated[next_u] if next_u in updated else values[next_u]
            _,cuts=convex_cuts(grid,next_values)
            sc,w,ids=weekday_distribution(v3,origin,u)
            raw=[];ds=[]
            for e in grid:
                sol=v3.solve(sc,w,float(e),cuts=cuts);raw.append(sol['objective']);ds.append(sol['initial_dual'])
            fit,newcuts=convex_cuts(grid,raw);updated[u]=fit
            duals.extend(dict(origin=str(v3.dates[origin].date()),iteration=iteration,weekday=u,initial_soc=float(e),
                initial_soc_dual=float(d),history_min_day=int(ids.min()),history_max_day=int(ids.max())) for e,d in zip(grid,ds))
        gap=max(float(np.max(np.abs(updated[u]-values[u]))) for u in range(7));history.append(dict(
            origin=str(v3.dates[origin].date()),iteration=iteration,max_value_change=gap,converged=gap<=tol))
        values=updated
        if gap<=tol:break
    banks={u:convex_cuts(grid,values[u])[1] for u in range(7)}
    return banks,pd.DataFrame(history),pd.DataFrame(duals),values


def build_value_banks(v3):
    path=OUT/'value_banks.json'
    if path.exists():
        raw=json.loads(path.read_text());return {int(m):{int(u):[(float(a),float(b)) for a,b in c] for u,c in x.items()} for m,x in raw.items()}
    banks={};hist=[];duals=[];curves=[]
    for month in range(2,13):
        origin=int(np.flatnonzero(v3.dates==pd.Timestamp(2025,month,1))[0])
        bank,h,d,val=learn_value_bank(v3,origin)
        banks[month]=bank;hist.append(h);duals.append(d)
        for u,y in val.items():
            for e,z in zip(VALUE_GRID,y):curves.append(dict(month=month,weekday=u,soc=e,relative_future_cost=z))
        atom(path,{m:{u:c for u,c in b.items()} for m,b in banks.items()})
        pd.concat(hist,ignore_index=True).to_csv(OUT/'value_iteration.csv',index=False)
        pd.concat(duals,ignore_index=True).to_csv(OUT/'terminal_duals.csv',index=False)
        pd.DataFrame(curves).to_csv(OUT/'terminal_value_curves.csv',index=False)
        print('value bank',month,'iterations',len(h),'gap',h.iloc[-1].max_value_change,flush=True)
    return banks


def actual_row(v3,d,cfg,sol,state,policy):
    ex=execute_day(sol['q'],v3.net[d]*DT,state,sol['soc'],cfg.rho)
    chk=check_dispatch(v3.net[d]*DT,sol['q'],ex['charge'],ex['discharge'],ex['spill'],ex['soc'],ex['emergency'])
    if not chk['pass']:raise AssertionError(chk)
    pc=float(v3.price@sol['q']);ec=float(5*v3.price@ex['emergency']);z=ex['emergency'];pos=z>1e-7
    row=dict(policy=policy,day=d,date=str(v3.dates[d].date()),initial_soc=float(state),terminal_soc=float(ex['soc'][-1]),
        planned_cost=pc,emergency_cost=ec,total_cost=pc+ec,planned_kwh=float(sol['q'].sum()),
        emergency_kwh=float(z.sum()),spill_kwh=float(ex['spill'].sum()),charge_kwh=float(ex['charge'].sum()),
        discharge_kwh=float(ex['discharge'].sum()),emergency_slots=int(pos.sum()),emergency_day=int(pos.any()),
        events=int(np.sum(pos&~np.r_[False,pos[:-1]])),reference_terminal_soc=float(sol['soc'][-1]),
        terminal_value=float(sol['terminal_value']),lp_objective=float(sol['objective']),solve_seconds=float(sol['seconds']),
        lp_eq_residual=sol['eq_residual'],lp_ineq_violation=sol['ineq_violation'],balance_residual=chk['balance_max_abs'],
        soc_residual=chk['soc_equation_max_abs'],**asdict(cfg))
    slot=pd.DataFrame(dict(policy=policy,date=row['date'],slot=np.arange(T),interval=INTERVALS,price=v3.price,
        net_actual=v3.net[d]*DT,q=sol['q'],charge=ex['charge'],discharge=ex['discharge'],emergency=z,
        spill=ex['spill'],soc_start=ex['soc'][:-1],soc_end=ex['soc'][1:],reference_soc=sol['soc'][1:]))
    return row,slot,ex


def run_policies(v3,selection,banks):
    rows=[];slots=[]
    policies=('ridge_fixed_saa_h1','hard6000','v3_dual_terminal')
    for policy in policies:
        state=INITIAL
        for d in range(31,365):
            cfg=DEFAULT if policy=='ridge_fixed_saa_h1' else config_at(v3,selection,d)
            sc,w,meta=v3.scenarios(d,cfg)
            cuts=None if policy=='hard6000' else banks[v3.dates[d].month][int((v3.dates[d].dayofweek+1)%7)]
            if policy=='ridge_fixed_saa_h1':cuts=None
            sol=v3.solve(sc,w,state,cuts=cuts,terminal=INITIAL if policy in ('ridge_fixed_saa_h1','hard6000') else None)
            row,slot,ex=actual_row(v3,d,cfg,sol,state,policy);row.update(meta);rows.append(row);slots.append(slot)
            state=float(ex['soc'][-1])
            if v3.dates[d].is_month_end:
                pd.DataFrame(rows).to_csv(OUT/'daily.partial.csv',index=False);print('policy checkpoint',policy,row['date'],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'daily.csv',index=False)
    pd.concat(slots,ignore_index=True).to_csv(OUT/'slots.csv.gz',index=False,compression='gzip')
    return df


def run_information(v3,selection,banks):
    rows=[]
    for d in range(151,365):
        cfg=config_at(v3,selection,d);sc,w,meta=v3.scenarios(d,cfg)
        cuts=banks[v3.dates[d].month][int((v3.dates[d].dayofweek+1)%7)]
        rp=v3.solve(sc,w,INITIAL,cuts=cuts);ev=v3.solve((w@sc)[None,:],np.ones(1),INITIAL,cuts=cuts)
        eev=float(w@np.array([v3.solve(s[None,:],np.ones(1),INITIAL,cuts=cuts)['objective'] for s in sc])) if False else None
        # Evaluate the EV plan under all scenarios without reoptimizing its first-stage decisions.
        p=v3.price;costs=p@ev['q']+5*np.maximum(sc*DT+ev['c']-ev['v']-ev['q'],0)@p+value_at(cuts,ev['soc'][-1])
        eev=float(w@costs)
        ws=float(w@np.array([v3.solve(s[None,:],np.ones(1),INITIAL,cuts=cuts)['objective'] for s in sc]))
        assert ws<=rp['objective']+.05 and rp['objective']<=eev+.05,(d,ws,rp['objective'],eev)
        rows.append(dict(day=d,date=str(v3.dates[d].date()),RP=rp['objective'],EEV=eev,WS=ws,
            VSS=eev-rp['objective'],EVPI=rp['objective']-ws,**meta))
        if v3.dates[d].is_month_end:
            pd.DataFrame(rows).to_csv(OUT/'information_value.partial.csv',index=False);print('information',v3.dates[d],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'information_value.csv',index=False);return df


def summarize(v3,daily,banks):
    data=daily.copy();data['date']=pd.to_datetime(data.date)
    periods={'full334':('2025-02-01','2025-12-31'),'development153':('2025-06-01','2025-10-31'),'frozen61':('2025-11-01','2025-12-31')}
    summaries=[];boots=[]
    allframes=[data]
    for frame in allframes:
        for policy,g in frame.groupby('policy'):
            g=g.sort_values('date')
            for period,(a,b) in periods.items():
                q=g[g.date.between(a,b)]
                if len(q)!=len(pd.date_range(a,b)):continue
                next_date=pd.Timestamp(b)+pd.Timedelta(days=1)
                # At the Dec31 reporting boundary, the latest causally learned bank is December's;
                # do not wrap to a nonexistent January-2026 bank.
                bank_month=12 if next_date.year>2025 else int(next_date.month)
                cuts=banks[bank_month][int(next_date.dayofweek)]
                rate=value_at(cuts,q.iloc[-1].terminal_soc)-value_at(cuts,INITIAL)
                summaries.append(dict(policy=policy,period=period,days=len(q),planned_cost=q.planned_cost.sum(),
                    emergency_cost=q.emergency_cost.sum(),raw_total_cost=q.total_cost.sum(),terminal_value_adjustment=rate,
                    adjusted_total_cost=q.total_cost.sum()+rate,planned_kwh=q.planned_kwh.sum(),emergency_kwh=q.emergency_kwh.sum(),
                    spill_kwh=q.spill_kwh.sum(),emergency_slots=q.emergency_slots.sum(),emergency_day_frequency=q.emergency_day.mean(),
                    max_daily_cost=q.total_cost.max(),start_soc=q.iloc[0].initial_soc,end_soc=q.iloc[-1].terminal_soc))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    main=data[data.policy=='v3_dual_terminal'][['date','total_cost']]
    for refname in ['ridge_fixed_saa_h1','hard6000']:
        ref=data[data.policy==refname]
        pair=main.merge(ref[['date','total_cost']],on='date',suffixes=('_v3','_ref'))
        for period,(a,b) in periods.items():
            q=pair[pair.date.between(a,b)];mean,lo,hi=block_ci(q.total_cost_ref-q.total_cost_v3,SEED+len(q))
            boots.append(dict(reference=refname,period=period,days=len(q),daily_saving_mean=mean,ci_low95=lo,ci_high95=hi))
    pd.DataFrame(boots).to_csv(OUT/'bootstrap.csv',index=False)
    data['month']=data.date.dt.month
    data.groupby(['policy','month'])[['planned_cost','emergency_cost','total_cost','emergency_kwh','spill_kwh']].sum().reset_index().to_csv(OUT/'monthly.csv',index=False)
    return summary


def verify(v3,daily,info):
    tests=[]
    def add(name,value,tol):
        passed=bool(value<=tol);tests.append(dict(test=name,value=float(value),tolerance=float(tol),passed=passed));assert passed,(name,value)
    add('causality',max(0,float((daily.max_history_day-daily.day+1).max())),0)
    add('balance',float(daily.balance_residual.max()),1e-7);add('soc_recursion',float(daily.soc_residual.max()),1e-7)
    add('lp_eq',float(daily.lp_eq_residual.max()),1e-5);add('lp_ineq',float(daily.lp_ineq_violation.max()),1e-5)
    for p,g in daily.groupby('policy'):
        add('soc_continuity_'+p,float(np.max(np.abs(g.initial_soc.iloc[1:].to_numpy()-g.terminal_soc.iloc[:-1].to_numpy()))),1e-7)
    add('VSS_nonnegative',max(0,float((-info.VSS).max())),.05);add('EVPI_nonnegative',max(0,float((-info.EVPI).max())),.05)
    baseline=float(daily[daily.policy=='ridge_fixed_saa_h1'].total_cost.sum())
    add('ridge_baseline_regression',abs(baseline-13667441.212052785),1e-3)
    obj={'all_pass':all(x['passed'] for x in tests),'tests':tests,'solve_calls':v3.solve_calls}
    atom(OUT/'verification.json',obj);return obj


def figures_and_report():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','Arial Unicode MS','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    FIG.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(OUT/'summary.csv');monthly=pd.read_csv(OUT/'monthly.csv');curves=pd.read_csv(OUT/'terminal_value_curves.csv')
    q=summary[summary.period=='full334'].sort_values('adjusted_total_cost')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(q.policy,q.planned_cost/1e6,label='计划购电费');ax.bar(q.policy,q.emergency_cost/1e6,bottom=q.planned_cost/1e6,label='紧急购电费')
    ax.set_ylabel('费用/百万元');ax.tick_params(axis='x',rotation=15);ax.legend();fig.tight_layout();fig.savefig(FIG/'01_cost_comparison.pdf');fig.savefig(FIG/'01_cost_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for p,g in monthly.groupby('policy'):ax.plot(g.month,g.total_cost/1e6,marker='o',label=p)
    ax.set_xlabel('月份');ax.set_ylabel('总费用/百万元');ax.legend();fig.tight_layout();fig.savefig(FIG/'02_monthly_stability.pdf');fig.savefig(FIG/'02_monthly_stability.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));g=curves[(curves.month==10)&(curves.weekday.isin([0,2,4]))]
    for u,z in g.groupby('weekday'):ax.plot(z.soc,z.relative_future_cost,marker='o',label=f'星期{int(u)+1}')
    ax.axvline(INITIAL,color='k',ls='--',lw=1);ax.set_xlabel('SOC/kWh');ax.set_ylabel('相对未来成本/元');ax.legend();fig.tight_layout();fig.savefig(FIG/'03_terminal_value.pdf');fig.savefig(FIG/'03_terminal_value.png',dpi=180);plt.close(fig)
    manifest={'files':[p.name for p in sorted(FIG.iterdir())]};atom(FIG/'figure_manifest.json',manifest)
    info=pd.read_csv(OUT/'information_value.csv');boot=pd.read_csv(OUT/'bootstrap.csv');selection=json.loads((OUT/'selection.json').read_text())
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('# Q2 V3 对偶终端价值实验结果\n\n本报告由执行脚本根据本次输出自动生成，不包含预写结果。\n\n'+
        '## 全年与分期费用\n\n'+summary.to_markdown(index=False)+'\n\n## Bootstrap\n\n'+boot.to_markdown(index=False)+
        '\n\n## VSS / EVPI\n\n'+info[['RP','EEV','WS','VSS','EVPI']].sum().to_frame('sum').to_markdown()+
        '\n\n## 参数更新\n\n'+pd.DataFrame(selection).to_markdown(index=False)+'\n',encoding='utf-8')


def pilot():
    OUT.mkdir(parents=True,exist_ok=True);v=V3();d=180;sc,w,meta=v.scenarios(d,DEFAULT)
    fixed=v.solve(sc,w,INITIAL,terminal=INITIAL)
    zero=[(0.,0.)];free=v.solve(sc,w,INITIAL,cuts=zero)
    bank,h,duals,values=learn_value_bank(v,d,grid=np.linspace(LO,HI,5),max_iter=3,tol=0)
    tv=v.solve(sc,w,INITIAL,cuts=bank[int((v.dates[d].dayofweek+1)%7)])
    tests={'fixed_terminal':fixed['soc'][-1],'free_terminal':free['soc'][-1],'value_terminal':tv['soc'][-1],
        'max_history_day':meta['max_history_day'],'decision_day':d,'all_finite':bool(np.isfinite([fixed['objective'],free['objective'],tv['objective']]).all()),
        'pilot_value_iterations':len(h),'solve_calls':v.solve_calls}
    assert meta['max_history_day']<d and tests['all_finite'];atom(OUT/'pilot.json',tests);print(json.dumps(tests,indent=2))


def full():
    if OUT.exists():shutil.rmtree(OUT)
    if FIG.exists():shutil.rmtree(FIG)
    OUT.mkdir(parents=True);FIG.mkdir(parents=True)
    started=time.time();v=V3();pilot();v=V3()
    selection=monthly_selection(v);banks=build_value_banks(v);daily=run_policies(v,selection,banks)
    info=run_information(v,selection,banks);summary=summarize(v,daily,banks);verification=verify(v,daily,info);figures_and_report()
    manifest=dict(complete=True,started_at=pd.Timestamp.fromtimestamp(started,tz='UTC').isoformat(),
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),seconds=time.time()-started,solve_calls=v.solve_calls,
        notebook='code/problem2_v3_dual_terminal.ipynb',source_sha256=sha(__file__),
        prediction_sha256=sha(ROOT/'results/problem2_forecast_ablation_predictions.csv.gz'),
        daily_hard_terminal=False,remote_hard_terminal=False,all_checks_pass=verification['all_pass'])
    atom(OUT/'run_manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


def resume_or_full():
    """Resume finalization from completed raw checkpoints after a reporting-only failure."""
    required=[OUT/'selection.json',OUT/'value_banks.json',OUT/'daily.csv',OUT/'information_value.csv']
    if not all(x.exists() for x in required):
        return full()
    started=time.time();v=V3();selection=json.loads((OUT/'selection.json').read_text())
    raw=json.loads((OUT/'value_banks.json').read_text())
    banks={int(m):{int(u):[(float(a),float(b)) for a,b in c] for u,c in x.items()} for m,x in raw.items()}
    daily=pd.read_csv(OUT/'daily.csv');info=pd.read_csv(OUT/'information_value.csv')
    summary=summarize(v,daily,banks);verification=verify(v,daily,info);figures_and_report()
    manifest=dict(complete=True,resumed_from_raw_checkpoints=True,
        completed_at=pd.Timestamp.now(tz='UTC').isoformat(),finalization_seconds=time.time()-started,
        notebook='code/problem2_v3_dual_terminal.ipynb',source_sha256=sha(__file__),
        prediction_sha256=sha(ROOT/'results/problem2_forecast_ablation_predictions.csv.gz'),
        daily_hard_terminal=False,remote_hard_terminal=False,all_checks_pass=verification['all_pass'])
    atom(OUT/'run_manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],default='full');a=p.parse_args()
    pilot() if a.stage=='pilot' else full()


if __name__=='__main__':main()
