"""Frozen HYBRID + paired load/PV scenarios. No hyperparameter search."""
from pathlib import Path
from dataclasses import asdict
import argparse, hashlib, json, sys, platform
import numpy as np
import pandas as pd
import scipy
import q3
import hybrid_core as H
import hybrid_policy as HP
import hybrid_stoch as HS
import export_xlsx as EX

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
CACHE = ROOT / 'cache'

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')

def fingerprint():
    files = sorted(list((ROOT/'code').glob('*.py')) + list((ROOT/'data').glob('*.xlsx')) + list((ROOT/'assets').glob('*')))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)
    key = hashlib.sha256(json.dumps([hashes, versions], sort_keys=True).encode()).hexdigest()
    return key, dict(files=hashes, versions=versions, key=key)

def get_bank(L, P, F, seed, p, folder):
    path = folder/'forecast_bank.npz'
    hit = path.exists()
    if hit:
        with np.load(path, allow_pickle=False) as a:
            bank = q3.Bank.__new__(q3.Bank)
            bank.p, bank.L, bank.P, bank.cache = p, L, P, {}
            for name in ['load','pv','mean','residual']: setattr(bank, name, a[name])
    else:
        bank = q3.Bank(L, P, F, seed, p)
        np.savez_compressed(path, **{n:getattr(bank,n) for n in ['load','pv','mean','residual']})
    print('Forecast cache:', 'HIT (no Ridge fitting)' if hit else 'BUILT', flush=True)
    return bank, hit

def actual_horizons(X):
    a = np.full((365,4,144), np.nan)
    for d in range(365):
        for k,h in enumerate(q3.ISSUES):
            t=h*6
            a[d,k,:144-t]=X[d,t:]
            if t and d<364: a[d,k,144-t:]=X[d+1,:t]
    return a

def paired_errors(bank, L, P):
    return actual_horizons(L)-bank.load, actual_horizons(P)-bank.pv

class JointScenarios:
    def __init__(self, bank, el, ep, p, pol, folder):
        self.bank,self.el,self.ep,self.p,self.pol,self.folder=bank,el,ep,p,pol,folder
    def __call__(self,d,k):
        path=self.folder/f'joint_{d:03d}_{k}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as a:return a['net'],a['weight']
        ids=np.arange(max(1,d-self.p.window),d)
        valid=np.isfinite(self.el[ids,k]).all(1)&np.isfinite(self.ep[ids,k]).all(1)
        ids=ids[valid]
        # A historical 24h residual is used only once that entire horizon has elapsed.
        assert np.all(ids*144+q3.ISSUES[k]*6+144 <= d*144+q3.ISSUES[k]*6)
        if not len(ids):
            net=self.bank.mean[d,k][None].copy(); w=np.ones(1)
            ls=self.bank.load[d,k][None]; ps=self.bank.pv[d,k][None]; chosen=ids
        else:
            le,pe=self.el[ids,k],self.ep[ids,k]
            # One scale per channel, estimated solely from eligible historical errors.
            sl=max(float(np.std(le)),1.); sp=max(float(np.std(pe)),1.)
            joint=np.concatenate([le/sl,pe/sp],axis=1)
            med,cnt=HS.kmedoids(joint,min(self.pol.n_scen,len(ids)),seed=97*d+k)
            # Historical paired paths are never independently permuted.
            chosen=ids[med]
            ls=np.maximum(self.bank.load[d,k]+self.pol.shrink*le[med],0)
            ps=np.maximum(self.bank.pv[d,k]+self.pol.shrink*pe[med],0)
            net=(ls-ps)*q3.DT
            w=cnt.astype(float)**self.pol.temper; w/=w.sum()
        np.savez_compressed(path,net=net,weight=w,load_kw=ls,pv_kw=ps,source_day=chosen,
                            available_at_slot=chosen*144+q3.ISSUES[k]*6+144)
        return net,w

def diagnose(bank,el,ep):
    rows=[]
    for start,end,label in [(7,31,'training_january'),(31,365,'evaluation_feb_dec')]:
        for k,h in enumerate(q3.ISSUES):
            for block in range(4):
                sl=slice(36*block,36*(block+1))
                a,b=el[start:end,k,sl],ep[start:end,k,sl]
                # Operational, downscaled slot errors, not native hourly forecast accuracy.
                pv=bank.pv[start:end,k,sl]
                mask=np.isfinite(a)&np.isfinite(b)
                for name,err,m in [('load',a,mask),('pv_daylight',b,mask&(pv>10)),('net',a-b,mask)]:
                    x=err[m]
                    rows.append(dict(period=label,issue_hour=h,lead_block=block+1,channel=name,n=len(x),
                        mae_kw=float(np.mean(abs(x))) if len(x) else None,
                        rmse_kw=float(np.sqrt(np.mean(x*x))) if len(x) else None,
                        bias_actual_minus_forecast_kw=float(np.mean(x)) if len(x) else None))
    pd.DataFrame(rows).to_csv(OUT/'forecast_accuracy.csv',index=False)
    corr=[]
    for k,h in enumerate(q3.ISSUES):
        train=np.isfinite(el[7:31,k])&np.isfinite(ep[7:31,k])&(bank.pv[7:31,k]>10)
        la,pa=el[7:31,k][train],ep[7:31,k][train]
        lq,pq=np.quantile(la,.9),np.quantile(pa,.1)
        for s,e,label in [(7,31,'training_january'),(31,365,'evaluation_feb_dec')]:
            m=np.isfinite(el[s:e,k])&np.isfinite(ep[s:e,k])&(bank.pv[s:e,k]>10)
            a,b=el[s:e,k][m],ep[s:e,k][m]
            corr.append(dict(period=label,issue_hour=h,n=len(a),pearson=float(np.corrcoef(a,b)[0,1]),
                dangerous_joint_tail_frequency=float(np.mean((a>lq)&(b<pq))),
                load_threshold_kw=float(lq),pv_threshold_kw=float(pq)))
    pd.DataFrame(corr).to_csv(OUT/'joint_error_diagnostics.csv',index=False)


def causal_check(L,P,F,seed,p,bank,el,ep,pol,tv,folder):
    rows=[]
    # Rebuild from perturbed raw inputs: do not let cached predictions hide leakage.
    day=200
    for k,h in enumerate(q3.ISSUES):
        t=h*6; lc=L.copy();pc=P.copy();fc=F.copy()
        lc[day,t:]+=4000;lc[day+1:]+=4000
        pc[day,t:]+=3000;pc[day+1:]+=3000
        fc[day,k+1:]+=2000;fc[day+1:]+=2000
        altered=q3.Bank(lc,pc,fc,seed,p)
        le,pe=paired_errors(altered,lc,pc)
        aa=folder/f'causal_original_{k}';bb=folder/f'causal_changed_{k}'
        aa.mkdir(exist_ok=True);bb.mkdir(exist_ok=True)
        n,w=JointScenarios(bank,el,ep,p,pol,aa)(day,k)
        nn,ww=JointScenarios(altered,le,pe,p,pol,bb)(day,k)
        diff=float(np.max(abs(n-nn)));wd=float(np.max(abs(w-ww)))
        assert diff<1e-9 and wd<1e-12,(h,diff,wd)
        # Same observed state and contracts => identical current LP plan.
        prices=np.r_[price_global[t:],price_global[:t]]
        base=None if k==0 else np.full(144-t,500.)
        tail=0 if k==0 else 144-t
        args=dict(emergency_scale=HP.emergency_scale(pol),cvar_beta=pol.cvar_beta,
                  cvar_alpha=pol.cvar_alpha,adj_premium=pol.adj_premium)
        x=HS.solve_stoch2(n,w,prices,6000.,p,tv.scaled(pol.kappa),base,tail,36,**args)
        y=HS.solve_stoch2(nn,ww,prices,6000.,p,tv.scaled(pol.kappa),base,tail,36,**args)
        assert x is not None and y is not None
        delta=float(np.max(abs(x[0]-y[0])));assert delta<1e-7
        rows.append(dict(day=day,issue_hour=h,scenario_max_difference=diff,plan_max_difference=delta))
    write_json(OUT/'causal_checks.json',rows)

def audit(daily,iv,L,P,price,p):
    iv=iv.copy()
    iv['soc_start']=iv.soc_end-p.eta*iv.charge+iv.discharge/p.eta
    errs=[]
    for date,g in iv.groupby('date',sort=True):
        g=g.sort_values('t');d=int((pd.Timestamp(date)-q3.DATES[0]).days)
        r=daily[daily.date==date].iloc[0]
        balance=g.adjusted+g.emergency+g.discharge-g.charge-g.spill-(L[d]-P[d])*q3.DT
        delta=g.adjusted.to_numpy()-g.plan.to_numpy()
        cash=float(price@g.plan.to_numpy()+price@(p.up*np.maximum(delta,0)+q3.settle_coef(p)*np.maximum(-delta,0))+p.emergency*(price@g.emergency.to_numpy()))
        errs.append(max(abs(cash-r.total_cost),float(abs(balance).max()),abs(g.soc_start.iloc[0]-r.soc_start),abs(g.soc_end.iloc[-1]-r.soc_end)))
        assert np.max(abs(g.soc_start.to_numpy()[1:]-g.soc_end.to_numpy()[:-1]))<1e-5
    assert np.max(errs)<1e-4
    assert np.max(abs(daily.soc_start.to_numpy()[1:]-daily.soc_end.to_numpy()[:-1]))<1e-5
    return iv,dict(max_independent_ledger_error=float(max(errs)),days=len(daily),slots=len(iv))

def main():
    global price_global
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=31);ap.add_argument('--end',type=int,default=365)
    ap.add_argument('--skip-causal',action='store_true');args=ap.parse_args()
    if args.start!=31:raise ValueError('Frozen initial SOC is valid only for Feb 1 (start=31).')
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    key,manifest=fingerprint();folder=CACHE/key;folder.mkdir(exist_ok=True)
    q3.OUT=OUT
    L,P,F,price,seed=q3.load();price_global=price
    pars=json.loads((ROOT/'assets/parameters.json').read_text());sel=pars['selected']
    keys=('alpha0','alpha_update','window','decay_days','bias_window','bias_decay','slope','weekday','epsilon')
    p=q3.replace(q3.Params(),**{k:sel[k] for k in keys})
    saved=json.loads((ROOT/'assets/hybrid_selected_policy.json').read_text())
    pol=HP.Policy(**{k:saved[k] for k in HP.Policy.__dataclass_fields__})
    seg=pd.read_csv(ROOT/'assets/hybrid_terminal_value_segments.csv')
    tv=H.TerminalValue(tuple([*seg.seg_lo,seg.seg_hi.iloc[-1]]),tuple(seg.marginal_value))
    assert np.all(np.diff(tv.edges)>0) and np.all(np.diff(tv.marginals)<=1e-10)
    bank,hit=get_bank(L,P,F,seed,p,folder);el,ep=paired_errors(bank,L,P)
    np.savez_compressed(folder/'paired_errors.npz',load_error_kw=el,pv_error_kw=ep)
    diagnose(bank,el,ep)
    if not args.skip_causal:causal_check(L,P,F,seed,p,bank,el,ep,pol,tv,folder)
    joint=JointScenarios(bank,el,ep,p,pol,folder)
    summaries=[]
    for label,provider in [('inherited_hybrid',None),('joint_hybrid',joint)]:
        dest=OUT/label;dest.mkdir(exist_ok=True)
        d,iv,dec=HP.run_stoch2(L,P,price,bank,H.RiskBank(bank,p.window),p,pol,tv,
            start=args.start,end=args.end,initial=float(pars['initial_feb1']),detail=True,scenario_provider=provider)
        iv,checks=audit(d,iv,L,P,price,p)
        d.to_csv(dest/'daily_6+12+18.csv',index=False);iv.to_csv(dest/'intervals.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
        write_json(dest/'checks.json',checks)
        s=H.summarize(d,method=label);summaries.append(s)
        print(label,s,flush=True)
        if label=='joint_hybrid':
            EX.ROOT=ROOT;EX.R=dest;EX.main()
            import openpyxl
            wb=openpyxl.load_workbook(ROOT/'result3.xlsx',data_only=True,read_only=True)
            cost=sum(r[-1] for r in list(wb['调整购电量'].values)[1:])
            emergency=sum(r[-1] for r in list(wb['紧急购电量'].values)[1:])
            assert abs(cost-d.total_cost.sum())<1e-4
            assert abs(emergency-d.emergency_kwh.sum())<1e-4
            for sheet,col in [('计划购电量','plan'),('调整购电量','adjusted')]:
                values=np.array([r[1:145] for r in list(wb[sheet].values)[1:]],float)
                assert np.max(abs(values-iv[col].to_numpy().reshape(-1,144)))<1e-7
            wb.close();write_json(dest/'workbook_checks.json',dict(cost=cost,emergency_kwh=emergency,all_plan_cells_match=True))
    pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False)
    manifest.update(parameters=asdict(p),policy=asdict(pol),forecast_cache_hit=hit,training='inherited January assets; no new tuning',
                    evaluation=[args.start,args.end],status='executed',causal_checks_run=not args.skip_causal)
    write_json(OUT/'run_manifest.json',manifest)
    (ROOT/'reports/RESULTS_REPORT.md').write_text('# Q3 联合情景版本运行结果\n\n'+pd.DataFrame(summaries).to_markdown(index=False)+'\n\n同一初始库存、价格及参数；差异包括联合聚类与逐通道非负裁剪，不能解释为纯相关性收益。\n数据见 results/comparison.csv；约束与账本见各子目录 checks.json；工作簿对应 joint_hybrid。\n',encoding='utf8')

if __name__=='__main__':main()
