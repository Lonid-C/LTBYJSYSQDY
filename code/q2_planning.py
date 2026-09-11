"""Q2 P0--P6: causal planning experiments. Execute only in Colab.

All energies are AC-bus kWh per slot; scenario arrays contain net power kW.
The notebook embeds this source so formulas, code and results remain together.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DT = 1 / 6
T = 144
ETA = .9
LO, HI, INITIAL = 1200., 10800., 6000.
CAP = 5000 * DT
EPS = 1e-8
WINDOWS = [14, 21, 28, 42, 56]
HALVES = [7, 14, 28, None]
NEIGHBORS = [14, 21, 28, 42]
SEED = 2026


def atom_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cvar(values, weights=None, beta=.9):
    """Exact empirical upper-tail mean, including fractional boundary mass."""
    values = np.asarray(values, float)
    w = np.ones(len(values)) / len(values) if weights is None else np.asarray(weights, float)
    idx = np.argsort(values)[::-1]
    before = np.r_[0., np.cumsum(w[idx])[:-1]]
    mass = np.minimum(w[idx], np.maximum(0., 1-beta-before))
    return float(values[idx] @ mass / (1-beta))


def weights_for(n, half):
    w = np.ones(n) if half is None else 2. ** (-np.arange(n, 0, -1)/half)
    return w / w.sum()


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root/'results/q2_planning_v1'
        self.fig = self.root/'figures/q2_planning_v1'
        self.out.mkdir(parents=True, exist_ok=True)
        self.fig.mkdir(parents=True, exist_ok=True)
        ref = pd.read_excel(self.root/'C题/附件/附件1.xlsx')
        self.price = ref['电价'].to_numpy(float)
        lf = pd.read_excel(self.root/'C题/附件/附件2.xlsx', sheet_name='小区负载', index_col=0)
        pf = pd.read_excel(self.root/'C题/附件/附件2.xlsx', sheet_name='光伏发电实际功率', index_col=0)
        self.dates = pd.DatetimeIndex(pd.to_datetime(lf.index))
        self.load, self.pv = lf.to_numpy(float), pf.to_numpy(float)
        self.net = self.load-self.pv
        self.ref = (ref['小区负载'].to_numpy(float), ref['光伏发电预测功率'].to_numpy(float))
        raw = self.root/'results/problem2_forecast_ablation_predictions.csv.gz'
        manual = np.full((2, 365, T), np.nan)
        for d in range(7, 365):
            ix = [d-7*k for k in range(1, 5) if d-7*k >= 0]
            w = np.array([.4, .3, .2, .1])[:len(ix)]; w /= w.sum()
            manual[0, d] = sum(a*self.load[j] for a, j in zip(w, ix))
            manual[1, d] = self.pv[d-7:d].mean(axis=0)
        self.legacy = manual.copy()
        manual[0, 7:90] = self.load[:83]
        manual[1, 7:90] = self.pv[:83]
        self.fc = {'ManualWeekly': manual, 'Ridge': manual.copy()}
        for part in pd.read_csv(raw, chunksize=200000):
            part = part[(part.method=='Ridge') & part.target.isin(['load', 'pv'])]
            if len(part):
                ds = (pd.to_datetime(part.date)-self.dates[0]).dt.days.to_numpy()
                tg = part.target.map({'load':0, 'pv':1}).to_numpy()
                self.fc['Ridge'][tg, ds, part.slot.to_numpy()] = part.y_pred.to_numpy()
        assert all(np.isfinite(x[:, 7:]).all() for x in self.fc.values())
        self.context = {}
        for method, fc in self.fc.items():
            doy = self.dates.dayofyear.to_numpy()
            dow = self.dates.dayofweek.to_numpy()
            self.context[method] = np.column_stack([
                np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
                np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
                fc[0].sum(axis=1)*DT, fc[1].sum(axis=1)*DT])
        source_hash = file_hash(self.root/'code/q2_planning.py')
        self.fingerprint = hashlib.sha256((source_hash+file_hash(raw)+
            file_hash(self.root/'C题/附件/附件1.xlsx')+
            file_hash(self.root/'C题/附件/附件2.xlsx')).encode()).hexdigest()
        self.cache = self.out/'checkpoints'/self.fingerprint[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.jobs = min(4, max(1, os.cpu_count() or 1))
        self.ridge_models = {}
        self.progress('initialized')

    def progress(self, phase, **kw):
        atom_json(self.out/'progress.json', {'phase': phase, 'utc':pd.Timestamp.now(tz='UTC').isoformat(),
            'fingerprint': self.fingerprint, **kw})
        print(phase, kw, flush=True)

    def solve(self, scen, w=None, initial=INITIAL, terminal=INITIAL,
              beta=None, budget=None, penalty=0., risk_only=False):
        """Sparse LP; retain the solver's original SOC and dual certificate.

        risk_only minimizes CVaR. A finite budget constrains CVaR while minimizing
        expected cost. penalty uses expected cost + penalty*CVaR.
        """
        scen = np.atleast_2d(np.asarray(scen, float)); S, H = scen.shape
        assert H % T == 0 and np.isfinite(scen).all()
        w = np.ones(S)/S if w is None else np.asarray(w, float)
        assert len(w)==S and np.all(w>0) and abs(w.sum()-1)<1e-10
        p = np.tile(self.price, H//T)
        n0 = 4*H+S*H
        risk = beta is not None
        n = n0+1+S if risk else n0
        t = np.arange(H)
        er = np.r_[t,t,t,t[1:]]
        ec = np.r_[H+t,2*H+t,3*H+t,3*H+t[:-1]]
        ev = np.r_[np.full(H,-ETA),np.full(H,1/ETA),np.ones(H),-np.ones(H-1)]
        Aeq = coo_matrix((ev,(er,ec)), shape=(H,n)).tocsr()
        beq = np.zeros(H); beq[0] = initial
        k = np.arange(S*H); slots = np.tile(t,S)
        Aub = coo_matrix((np.tile([-1.,1.,-1.,-1.],(S*H,1)).ravel(),
            (np.repeat(k,4),np.column_stack([slots,H+slots,2*H+slots,4*H+k]).ravel())),
            shape=(S*H,n)).tocsr()
        bub = -(scen*DT).ravel()
        expected = np.r_[p,np.full(2*H,EPS),np.zeros(H),(5*w[:,None]*p).ravel(),np.zeros(n-n0)]
        rv = np.zeros(n)
        budget_row = None
        if risk:
            rv[n0] = 1; rv[n0+1:] = w/(1-beta)
            # Scenario total cost <= zeta + xi_s, includes deterministic plan cost.
            rows = np.repeat(np.arange(S),2*H+2)
            cols = np.concatenate([np.r_[t,4*H+s*H+t,n0,n0+1+s] for s in range(S)])
            vals = np.tile(np.r_[p,5*p,-1.,-1.],S)
            cr = coo_matrix((vals,(rows,cols)),shape=(S,n)).tocsr()
            Aub = vstack([Aub,cr],format='csr'); bub = np.r_[bub,np.zeros(S)]
            if budget is not None:
                budget_row = len(bub)
                Aub = vstack([Aub,csr_matrix(rv[None,:])],format='csr')
                bub = np.r_[bub,float(budget)]
        obj = rv+EPS*np.r_[np.zeros(H),np.ones(2*H),np.zeros(n-3*H)] if risk_only else expected+penalty*rv
        bounds = ([(0,None)]*H+[(0,CAP)]*(2*H)+[(LO,HI)]*(H-1)+
                  [(terminal,terminal)]+[(0,None)]*(S*H))
        if risk: bounds += [(None,None)]+[(0,None)]*S
        started = time.perf_counter()
        # SciPy 所携 HiGHS 版本在不同 Colab 镜像上偶发返回 status=4 /
        # "HiGHS Status 0: Not Set"。这是求解器驱动问题，不是模型状态。
        # 同一 LP 在双单纯形与内点法间可确定性退避，最终仍须通过
        # 下方原始/对偶、驻点和互补残差检查。
        ans = None; solver_method = None
        attempts=[('highs',{'presolve':True,'dual_feasibility_tolerance':1e-8,
                            'primal_feasibility_tolerance':1e-8}),
                  ('highs-ds',{'presolve':False}),('highs-ipm',{'presolve':True})]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options')
            for method,options in attempts:
                candidate=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,
                                  method=method,options=options)
                if candidate.success:
                    ans=candidate;solver_method=method;break
                ans=candidate
        seconds = time.perf_counter()-started
        if not ans.success: raise RuntimeError(f'LP failed: {ans.status} {ans.message}; beta={beta} B={budget}')
        x = ans.x
        lower = np.array([-np.inf if a is None else a for a,b in bounds])
        upper = np.array([np.inf if b is None else b for a,b in bounds])
        fl, fu = np.isfinite(lower), np.isfinite(upper)
        deq, dub = ans.eqlin.marginals, ans.ineqlin.marginals
        dl, du = ans.lower.marginals, ans.upper.marginals
        dual = float(beq@deq+bub@dub+lower[fl]@dl[fl]+upper[fu]@du[fu])
        stationarity = obj-Aeq.T@deq-Aub.T@dub-dl-du
        comp = max(np.max(np.abs(dub*(bub-Aub@x))),
                   np.max(np.abs(dl[fl]*(x[fl]-lower[fl]))),
                   np.max(np.abs(du[fu]*(upper[fu]-x[fu]))))
        checks = {'eq_residual':float(np.abs(Aeq@x-beq).max()),
            'ineq_violation':float(max(0,np.max(Aub@x-bub))),
            'bound_violation':float(max(0,np.max(lower[fl]-x[fl]),np.max(x[fu]-upper[fu]))),
            'soc_recursion':float(np.max(np.abs(x[3*H:4*H]-(initial+np.cumsum(ETA*x[H:2*H]-x[2*H:3*H]/ETA))))),
            'dual_gap':float(abs(obj@x-dual)), 'dual_gap_relative':float(abs(obj@x-dual)/max(1,abs(obj@x))),
            'stationarity':float(np.abs(stationarity).max()),'complementarity':float(comp),
            'simultaneous_kwh':float(np.minimum(x[H:2*H],x[2*H:3*H]).clip(0).sum())}
        assert max(checks[k] for k in ['eq_residual','ineq_violation','bound_violation','soc_recursion'])<1e-5, checks
        assert checks['dual_gap_relative']<1e-7 and checks['stationarity']<1e-6, checks
        assert checks['complementarity']/max(1,abs(obj@x))<1e-7, checks
        q,c,v = x[:H].copy(),x[H:2*H].copy(),x[2*H:3*H].copy()
        # With unrestricted spill, remove a simultaneous charging cycle while
        # preserving each SOC increment. Physical spill absorbs surplus power.
        raw_sim = np.minimum(c,v).clip(0)
        if raw_sim.max(initial=0)>1e-7:
            remove = np.minimum(c,v/ETA**2).clip(0)
            c -= remove; v -= ETA**2*remove
        physical_cost = p@q+5*np.maximum(scen*DT+c-v-q,0)@p
        risk_value = cvar(physical_cost,w,beta) if risk else None
        if budget is not None: assert risk_value <= budget+1e-4, (risk_value,budget)
        assert np.abs(initial+np.cumsum(ETA*c-v/ETA)-x[3*H:4*H]).max()<1e-5
        assert min(q.min(),c.min(),v.min())>=-1e-7
        return {'q':q,'c':c,'v':v,'soc':np.r_[initial,x[3*H:4*H]],'x':x,
            'scenarios':scen,'weights':w,
            'risk_parameters':np.array([beta if risk else np.nan,budget if budget is not None else np.nan,penalty,float(risk_only)]),
            'dual_eq':deq,'dual_ub':dub,'dual_lower':dl,'dual_upper':du,
            'checks':checks,'expected':float(w@physical_cost),'cvar':risk_value,
            'objective':float(ans.fun),'dual_objective':dual,'seconds':seconds,'solver_method':solver_method,
            'risk_multiplier':float(max(0,-dub[budget_row])) if budget_row is not None else 0.,
            'postprocessed':bool(raw_sim.max(initial=0)>1e-7)}

    def realized(self, sol, net, take=T):
        q,c,v = (sol[k][:take] for k in ['q','c','v'])
        net = np.asarray(net).ravel()[:take]
        p = np.tile(self.price,take//T)
        u = np.maximum(net*DT+c-v-q,0)
        spill = np.maximum(q+v-net*DT-c,0)
        pos = u>1e-7
        balance = q+v+u-net*DT-c-spill
        assert np.abs(balance).max()<1e-6
        return {'planned_cost':float(p@q),'emergency_cost':float(5*p@u),
            'total_cost':float(p@q+5*p@u),'planned_kwh':float(q.sum()),
            'emergency_kwh':float(u.sum()),'spill_kwh':float(spill.sum()),
            'charge_kwh':float(c.sum()),'discharge_kwh':float(v.sum()),
            'emergency_slots':int(pos.sum()),'emergency_day':int(pos.any()),
            'events':int((pos & ~np.r_[False,pos[:-1]]).sum()),
            'initial_soc':float(sol['soc'][0]),'terminal_soc':float(sol['soc'][take]),
            'balance_residual':float(np.abs(balance).max()),
            'peak_price_emergency_kwh':float(u[p>=np.quantile(p,.75)].sum()),
            'solve_seconds':sol['seconds'],**sol['checks']}

    def scenarios(self, method, d, cfg, forecast=None):
        fc = self.fc[method]; netfc = fc[0]-fc[1]
        if cfg['kind']=='point': return netfc[d][None,:],np.ones(1),{'max_history_day':d-1,'n':1}
        if cfg['kind']=='conditional':
            hist = np.arange(max(31,d-120),d)
            dims = {'season':2,'weekday':4,'level':6}[cfg['features']]
            x = self.context[method][hist,:dims]
            sd = x.std(axis=0); sd[sd<1e-8]=1
            distance = np.sum(((x-self.context[method][d,:dims])/sd)**2,axis=1)
            hist = np.sort(hist[np.argsort(distance,kind='stable')[:min(cfg['k'],len(hist))]])
            w = weights_for(len(hist),None)
            f = fc[:,d] if forecast is None else forecast
            l = np.maximum(f[0]+self.load[hist]-fc[0,hist],0)
            p = np.maximum(f[1]+self.pv[hist]-fc[1,hist],0)
            scen = l-p
        else:
            hist = np.arange(max(7,d-cfg['window']),d)
            w = weights_for(len(hist),cfg['half'])
            target = netfc[d] if forecast is None else forecast[0]-forecast[1]
            scen = target+self.net[hist]-netfc[hist]
            if cfg.get('project_components'):
                f=fc[:,d] if forecast is None else forecast
                scen=np.maximum(f[0]+self.load[hist]-fc[0,hist],0)-np.maximum(f[1]+self.pv[hist]-fc[1,hist],0)
        assert hist.max()<d
        return scen,w,{'max_history_day':int(hist.max()),'min_history_day':int(hist.min()),
            'n':len(hist),'effective_n':float(1/(w@w)),
            'native_residual_fraction':float(np.mean(hist>=90))}

    def day(self, method, d, cfg, label=None):
        key = hashlib.sha256(json.dumps([method,int(d),cfg],sort_keys=True).encode()).hexdigest()
        path = self.cache/f'{key}.json'
        if path.exists(): return json.loads(path.read_text())
        sc,w,meta = self.scenarios(method,d,cfg)
        sol = self.solve(sc,w)
        row = {'method':method,'d':int(d),'date':str(self.dates[d].date()),
               'config':json.dumps(cfg,sort_keys=True),**meta,**self.realized(sol,self.net[d])}
        np.savez_compressed(self.cache/f'{key}.npz',**{k:sol[k] for k in
            ['x','soc','q','c','v','dual_eq','dual_ub','dual_lower','dual_upper']})
        atom_json(path,row)
        return row

    def map_days(self, method, days, cfg):
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            return list(pool.map(lambda d:self.day(method,int(d),cfg), days))


BASE = {'kind':'time','window':21,'half':14}


def candidate_configs():
    time_cfg = [{'kind':'time','window':w,'half':h} for w in WINDOWS for h in HALVES]
    cond = [{'kind':'conditional','features':f,'k':k} for f in ['season','weekday','level'] for k in NEIGHBORS]
    return time_cfg+cond


def select_config(rows):
    best = min(r['mean_cost'] for r in rows)
    near = [r for r in rows if r['mean_cost']<=best*1.001]
    def simplicity(r):
        c = r['cfg']
        return (c['kind']!='time',c.get('half') is not None,
                c.get('window',c.get('k')),{'season':0,'weekday':1,'level':2}.get(c.get('features'),0),r['mean_cost'])
    return min(near,key=simplicity)


def run_pilot(study):
    s = study; rows = []; tests=[]
    for method in s.fc:
        for d in [151,258]:
            for cfg in [BASE,{'kind':'time','window':56,'half':None},
                        {'kind':'conditional','features':'level','k':21}]:
                row=s.day(method,d,cfg); rows.append(row)
    d=166
    sc,w,_=s.scenarios('Ridge',d,{'kind':'time','window':21,'half':None})
    a=s.solve(sc,w); rng=np.random.default_rng(SEED)
    shuffled=np.column_stack([rng.permutation(sc[:,t]) for t in range(T)])
    b=s.solve(shuffled,w)
    tests.append({'test':'equal_weight_permutation','gap':abs(a['objective']-b['objective'])})
    assert abs(a['objective']-b['objective'])<1e-4
    f=s.fc['Ridge'][0,d]-s.fc['Ridge'][1,d]; hist=np.arange(d-21,d)
    r=s.net[hist]-(s.fc['Ridge'][0,hist]-s.fc['Ridge'][1,hist])
    shifted=(f+100.)+(r-100.)
    tests.append({'test':'bias_cancellation','gap':float(np.max(np.abs(shifted-(f+r))))})
    assert np.allclose(shifted,f+r,atol=1e-10)
    risk=s.solve(sc,w,beta=.9,risk_only=True)
    B=(cvar(scenario_cost(s,a,sc),w,.9)+risk['cvar'])/2
    budget=s.solve(sc,w,beta=.9,budget=B+1e-6)
    penal=s.solve(sc,w,beta=.9,penalty=budget['risk_multiplier'])
    gap=abs((budget['expected']+budget['risk_multiplier']*budget['cvar'])-
            (penal['expected']+budget['risk_multiplier']*penal['cvar']))
    tests.append({'test':'risk_dual_equivalence','gap':gap})
    assert gap<.1
    for nd in [2,3]:
        sol=s.solve(np.tile(sc,(1,nd)),w,initial=4800,terminal=6000)
        tests.append({'test':f'horizon_{nd}days','gap':sol['checks']['eq_residual']})
    pd.DataFrame(rows).to_csv(s.out/'pilot.csv',index=False)
    atom_json(s.out/'pilot_tests.json',tests)
    s.progress('pilot_passed',median_lp_seconds=float(np.median([r['solve_seconds'] for r in rows])),
               cpu_workers=s.jobs,tests=tests)
    return pd.DataFrame(rows),tests


def scenario_cost(s, sol, sc):
    H=sc.shape[1]; p=np.tile(s.price,H//T)
    return p@sol['q']+5*np.maximum(sc*DT+sol['c']-sol['v']-sol['q'],0)@p


def run_selection(s):
    """Monthly selection uses only dates ending before that month's origin."""
    selection=[]; scores=[]
    configs=candidate_configs()
    for method in s.fc:
        for month in range(6,11):
            origin=int(np.flatnonzero(s.dates==pd.Timestamp(2025,month,1))[0])
            all_rows=[]
            for j,cfg in enumerate(configs):
                rr=s.map_days(method,range(origin-56,origin),cfg)
                for cal in [28,56]:
                    sample=rr[-cal:]
                    row={'method':method,'month':month,'origin':str(s.dates[origin].date()),
                         'calibration_days':cal,'last_training_date':str(s.dates[origin-1].date()),
                         'cfg':cfg,'mean_cost':float(np.mean([r['total_cost'] for r in sample]))}
                    all_rows.append(row); scores.append(row)
                if j%8==0: s.progress('selection',method=method,month=month,candidate=j+1,total=len(configs))
            for cal in [28,56]:
                for family in ['time','conditional','all','season','weekday','level']:
                    opts=[r for r in all_rows if r['calibration_days']==cal and
                          (family=='all' or r['cfg']['kind']==family or r['cfg'].get('features')==family)]
                    selection.append({**select_config(opts),'family':family})
            atom_json(s.out/'selection.json',selection)
            atom_json(s.out/'calibration_scores.json',scores)
    return selection


def chosen_at(selection,method,d,cal=28,family='all'):
    month=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(d))
    if month.month<6: return BASE
    return next(r['cfg'] for r in selection if r['method']==method and
                r['month']==min(month.month,10) and r['calibration_days']==cal and r['family']==family)


def run_baselines(s, selection):
    rows=[]; regression=[]
    # Exact legacy regression uses pre-cold-start Manual forecasts, not rounded CSV.
    netlegacy=s.legacy[0]-s.legacy[1]
    for d in range(31,365):
        hist=np.arange(d-21,d)
        sol=s.solve(netlegacy[d]+s.net[hist]-netlegacy[hist],weights_for(21,14))
        regression.append(s.realized(sol,s.net[d])['total_cost'])
    total=float(sum(regression)); assert abs(total-14539240.537931435)<1e-3,total
    atom_json(s.out/'legacy_regression.json',{'cost':total,'reference':14539240.537931435,'error':total-14539240.537931435})
    for method in s.fc:
        for policy in ['P0','P1','P2_equal21','P2_projected21','P2_learned28','P2_learned56','P3_learned28','P3_learned56',
                       'P3_stage_season','P3_stage_weekday','P3_stage_level','SAA_selected']:
            for d in range(31,365):
                cfg=({'kind':'point'} if policy=='P0' else
                     {'kind':'time','window':21,'half':None} if policy=='P2_equal21' else
                     {'kind':'time','window':21,'half':None,'project_components':True} if policy=='P2_projected21' else
                     BASE if policy=='P1' else chosen_at(selection,method,d,
                     cal=56 if policy.endswith('56') else 28,
                     family=policy.split('_')[-1] if policy.startswith('P3_stage') else
                     'time' if policy.startswith('P2') else 'conditional' if policy.startswith('P3') else 'all'))
                rows.append({**s.day(method,d,cfg),'policy':policy})
            s.progress('baselines',method=method,policy=policy)
        for d in range(31,365):
            sol=s.solve(s.net[d][None,:]); row=s.realized(sol,s.net[d])
            cert=save_certificate(s,'P5_daily',method,d,sol)
            rows.append({'method':method,'d':d,'date':str(s.dates[d].date()),'policy':'P5_daily','certificate':cert,**row})
    df=pd.DataFrame(rows); df.to_csv(s.out/'daily_baselines.csv',index=False)
    return df


def run_calibration(s, selection):
    rows=[]; daily=[]
    for method in s.fc:
        for d in range(151,365):
            cfg=chosen_at(selection,method,d); sc,w,meta=s.scenarios(method,d,cfg)
            for tau in [.5,.8,.9,.95]:
                order=np.argsort(sc,axis=0); sv=np.take_along_axis(sc,order,axis=0)
                sw=w[order]; ix=(np.cumsum(sw,axis=0)<tau).sum(axis=0).clip(max=len(w)-1)
                quant=sv[ix,np.arange(T)]; err=s.net[d]-quant
                # Bins use predicted PV and past predicted-PV tertiles only.
                past=s.context[method][max(31,d-120):d,5]
                pvbin=int(np.searchsorted(np.quantile(past,[1/3,2/3]),s.context[method][d,5]))
                for t in range(T):
                    rows.append({'method':method,'date':str(s.dates[d].date()),'month':s.dates[d].month,
                        'slot':t,'tau':tau,'quantile_kw':quant[t],'actual_kw':s.net[d,t],
                        'covered':float(err[t]<=0),'pinball':float(max(tau*err[t],(tau-1)*err[t])),
                        'upper_exceedance_kw':float(max(err[t],0)),'price':s.price[t],
                        'pv_bin':pvbin,'effective_n':meta['effective_n'],
                        'tail_effective_n':(1-tau)*meta['effective_n']})
            daily.append({'method':method,'date':str(s.dates[d].date()),'month':s.dates[d].month,
                'residual_bias':float(np.mean(s.net[d]-(s.fc[method][0,d]-s.fc[method][1,d]))),
                'scenario_spread':float(np.mean(np.std(sc,axis=0))),**meta})
    df=pd.DataFrame(rows); df.to_csv(s.out/'quantile_detail.csv.gz',index=False,compression='gzip')
    summary=df.groupby(['method','month','tau','pv_bin']).agg(coverage=('covered','mean'),
        pinball=('pinball','mean'),exceedance=('upper_exceedance_kw','mean'),
        effective_n=('effective_n','mean'),tail_effective_n=('tail_effective_n','mean')).reset_index()
    summary.to_csv(s.out/'calibration.csv',index=False)
    pd.DataFrame(daily).to_csv(s.out/'residual_scale.csv',index=False)
    return summary


def save_certificate(s,label,method,d,sol):
    folder=s.out/'certificates'; folder.mkdir(exist_ok=True)
    safe=hashlib.sha256(f'{label}|{method}|{d}'.encode()).hexdigest()[:20]
    np.savez_compressed(folder/f'{safe}.npz',**{k:sol[k] for k in
        ['x','q','c','v','soc','scenarios','weights','risk_parameters','dual_eq','dual_ub','dual_lower','dual_upper']})
    return safe


def run_risk(s,selection):
    rows=[]; dual_rows=[]; dependence=[]
    for method in s.fc:
        for month in range(6,13):
            marker=s.cache/f'risk_{method}_{month}.json'
            if marker.exists():
                packed=json.loads(marker.read_text()); rows+=packed['rows']; dual_rows+=packed['dual']; dependence+=packed['dependence']; continue
            mr=[]; md=[]; dep=[]
            for d in np.flatnonzero(s.dates.month==month):
                sc,w,meta=s.scenarios(method,int(d),chosen_at(selection,method,d))
                neutral=s.solve(sc,w)
                for beta in [.9,.95]:
                    rmin=s.solve(sc,w,beta=beta,risk_only=True)
                    upper=cvar(scenario_cost(s,neutral,sc),w,beta); lower=rmin['cvar']
                    budgets=np.linspace(upper,lower,5)
                    lambdas=[0.]
                    budget_solutions=[]
                    for k,B in enumerate(budgets):
                        sol=s.solve(sc,w,beta=beta,budget=float(B+1e-5))
                        lambdas.append(sol['risk_multiplier']); budget_solutions.append(sol)
                        label=f'P4_b{beta}_budget{k}'
                        cert=save_certificate(s,label,method,int(d),sol)
                        mr.append({'method':method,'d':int(d),'date':str(s.dates[d].date()),'policy':label,
                            'beta':beta,'budget_index':k,'budget':B,'scenario_expected':sol['expected'],
                            'scenario_cvar':sol['cvar'],'risk_multiplier':sol['risk_multiplier'],
                            'effective_tail_n':(1-beta)/(w@w),'certificate':cert,**s.realized(sol,s.net[d])})
                    for lam in sorted(set(round(x,8) for x in lambdas)):
                        sol=s.solve(sc,w,beta=beta,penalty=lam)
                        cert=save_certificate(s,f'penalty{beta}_{lam}',method,int(d),sol)
                        md.append({'method':method,'date':str(s.dates[d].date()),'beta':beta,'lambda':lam,
                            'expected':sol['expected'],'cvar':sol['cvar'],'certificate':cert,**s.realized(sol,s.net[d])})
                        for bs in budget_solutions:
                            if abs(bs['risk_multiplier']-lam)<1e-7:
                                val=bs['expected']+lam*bs['cvar']; other=sol['expected']+lam*sol['cvar']
                                assert abs(val-other)/max(1,abs(val))<1e-6,(val,other,lam)
                    if s.dates[d].day==15:
                        eq=np.ones(len(sc))/len(sc); rng=np.random.default_rng(SEED+int(d))
                        sh=np.column_stack([rng.permutation(sc[:,t]) for t in range(T)])
                        en=s.solve(sc,eq); sn=s.solve(sh,eq)
                        er=s.solve(sc,eq,beta=beta,risk_only=True); sr=s.solve(sh,eq,beta=beta,risk_only=True)
                        assert abs(en['objective']-sn['objective'])<1e-4
                        dep.append({'method':method,'date':str(s.dates[d].date()),'beta':beta,
                            'neutral_pairing_gap':en['objective']-sn['objective'],
                            'original_min_cvar':er['cvar'],'shuffled_min_cvar':sr['cvar']})
            atom_json(marker,{'rows':mr,'dual':md,'dependence':dep}); rows+=mr; dual_rows+=md; dependence+=dep
            s.progress('risk_checkpoint',method=method,month=month)
    pd.DataFrame(rows).to_csv(s.out/'daily_risk.csv',index=False)
    pd.DataFrame(dual_rows).to_csv(s.out/'risk_penalty.csv',index=False)
    pd.DataFrame(dependence).to_csv(s.out/'dependence.csv',index=False)
    return pd.DataFrame(rows)


def features(y,d,dates):
    columns=[y[d-k] for k in [1,2,3,7,14,21,28]]
    for W in [3,7,14,28]: columns += [y[d-W:d].mean(axis=0),y[d-W:d].std(axis=0)]
    columns += [np.mean([y[d-7*k] for k in range(1,5)],axis=0)]
    dow=np.zeros((T,7)); dow[:,dates[d].dayofweek]=1
    a=2*np.pi*dates[d].dayofyear/365
    day=np.tile([np.sin(a),np.cos(a)],(T,1)); slot=np.arange(T)/T
    harmonic=np.column_stack([fn(2*np.pi*k*slot) for k in range(1,4) for fn in [np.sin,np.cos]])
    return np.column_stack(columns+[dow,day,harmonic])


def recursive_forecast(s,method,d,nd):
    ys=[s.load[:d].copy(),s.pv[:d].copy()]; future=[s.fc[method][:,d]]
    models=[]
    if method=='Ridge':
        origin=int(np.flatnonzero(s.dates==pd.Timestamp(2025,s.dates[d].month,1))[0])
        # Before April all methods explicitly share the existing lag-7 warm-up.
        if d>=90:
            for target,y in enumerate([s.load,s.pv]):
                key=(origin,target)
                if key not in s.ridge_models:
                    X=np.vstack([features(y,j,s.dates) for j in range(28,origin)])
                    Y=y[28:origin].ravel()
                    model=make_pipeline(StandardScaler(),Ridge(alpha=100. if target==0 else 10.))
                    model.fit(X,Y); s.ridge_models[key]=model
                models.append(s.ridge_models[key])
    for lead in range(1,nd):
        ys=[np.vstack([y,future[-1][target]]) for target,y in enumerate(ys)]
        day=d+lead; out=[]
        for target,y in enumerate(ys):
            if d<90: pred=y[day-7]
            elif method=='ManualWeekly':
                pred=(sum(w*y[day-7*k] for k,w in enumerate([.4,.3,.2,.1],1))
                      if target==0 else y[day-7:day].mean(axis=0))
            else: pred=models[target].predict(features(y,day,s.dates))
            out.append(np.maximum(pred,0))
        future.append(np.array(out))
    return future


def run_rolling(s,selection):
    rows=[]; trajectory=[]; oracles=[]
    for method in s.fc:
        for horizon,start in [(2,31),(3,31),(2,151),(3,151)]:
            for terminal in ([1200.,6000.,10800.] if start==31 else [6000.]):
                prefix='P6' if start==31 else 'P6J'
                policy=f'{prefix}_H{horizon}_E{int(terminal)}'; soc=INITIAL
                marker=s.cache/f'rolling_{method}_{policy}.json'
                if marker.exists():
                    done=json.loads(marker.read_text()); rows+=done['rows']; trajectory+=done['trajectory']; continue
                partial=marker.with_suffix('.partial.json')
                previous=json.loads(partial.read_text()) if partial.exists() else {'rows':[],'trajectory':[]}
                rr=previous['rows']; tr=previous['trajectory']
                if rr: soc=rr[-1]['terminal_soc']
                for d in range(start+len(rr),365):
                    nd=min(horizon,365-d); fc=recursive_forecast(s,method,d,nd)
                    cfg=chosen_at(selection,method,d)
                    ss=[]
                    for forecast in fc:
                        sc,w,meta=s.scenarios(method,d,cfg,forecast=forecast); ss.append(sc)
                    scenarios=np.concatenate(ss,axis=1)
                    # Common year-end storage condition applies whenever horizon reaches Dec31.
                    target=INITIAL if d+nd==365 else terminal
                    sol=s.solve(scenarios,w,initial=soc,terminal=target)
                    m=s.realized(sol,s.net[d]); cert=save_certificate(s,policy,method,d,sol)
                    assert abs(m['initial_soc']-soc)<1e-6
                    soc=m['terminal_soc']
                    rr.append({'method':method,'policy':policy,'d':d,'date':str(s.dates[d].date()),
                        'horizon_days':nd,'far_terminal_soc':target,'max_actual_history_day':d-1,
                        'forecast_origin':str(s.dates[d].date()),'certificate':cert,**m})
                    for t in range(T):
                        tr.append({'method':method,'policy':policy,'date':str(s.dates[d].date()),'slot':t,
                            'q':sol['q'][t],'c':sol['c'][t],'v':sol['v'][t],
                            'soc_start':sol['soc'][t],'soc_end':sol['soc'][t+1]})
                    if s.dates[d].is_month_end:
                        atom_json(partial,{'rows':rr,'trajectory':tr})
                        s.progress('rolling_month_checkpoint',method=method,policy=policy,month=s.dates[d].month)
                assert abs(soc-INITIAL)<1e-5
                atom_json(marker,{'rows':rr,'trajectory':tr}); rows+=rr; trajectory+=tr
                s.progress('rolling_checkpoint',method=method,policy=policy)
    # Continuous-SOC perfect-information bound, not a sum of cyclic daily Oracles.
    for start,label in [(31,'full334'),(151,'fair214')]:
        marker=s.cache/f'continuous_oracle_{label}.json'
        if marker.exists(): oracles.append(json.loads(marker.read_text())); continue
        sol=s.solve(s.net[start:].reshape(1,-1),initial=INITIAL,terminal=INITIAL)
        row={'period':label,**s.realized(sol,s.net[start:].ravel(),take=(365-start)*T)}
        save_certificate(s,'P5_continuous',label,start,sol); atom_json(marker,row); oracles.append(row)
    pd.DataFrame(rows).to_csv(s.out/'daily_rolling.csv',index=False)
    pd.DataFrame(trajectory).to_csv(s.out/'rolling_slots.csv.gz',index=False,compression='gzip')
    pd.DataFrame(oracles).to_csv(s.out/'continuous_oracle.csv',index=False)
    return pd.DataFrame(rows)


def run_information(s,selection):
    rows=[]
    # All fair-period days: conditional distribution VSS/EVPI, not realized differences.
    for method in s.fc:
        for month in range(6,13):
            marker=s.cache/f'information_{method}_{month}.json'
            if marker.exists(): rows+=json.loads(marker.read_text()); continue
            rr=[]
            for d in np.flatnonzero(s.dates.month==month):
                sc,w,meta=s.scenarios(method,int(d),chosen_at(selection,method,d))
                rp=s.solve(sc,w); ev=s.solve((w@sc)[None,:])
                save_certificate(s,'information_RP',method,int(d),rp)
                save_certificate(s,'information_EV',method,int(d),ev)
                eev=float(w@scenario_cost(s,ev,sc))
                with ThreadPoolExecutor(max_workers=s.jobs) as pool:
                    wait_and_see=list(pool.map(lambda x:s.solve(x[None,:]),sc))
                ws=float(w@np.array([x['expected'] for x in wait_and_see]))
                np.savez_compressed(s.cache/f'WS_{method}_{int(d)}.npz',scenarios=sc,weights=w,
                    **{key:np.stack([x[key] for x in wait_and_see]) for key in
                       ['x','dual_eq','dual_ub','dual_lower','dual_upper']})
                assert ws<=rp['expected']+.01 and rp['expected']<=eev+.01,(ws,rp['expected'],eev)
                rr.append({'method':method,'date':str(s.dates[d].date()),'RP':rp['expected'],
                    'EEV':eev,'WS':ws,'VSS':eev-rp['expected'],'EVPI':rp['expected']-ws,**meta})
            atom_json(marker,rr); rows+=rr; s.progress('information_checkpoint',method=method,month=month)
    df=pd.DataFrame(rows); df.to_csv(s.out/'information_value.csv',index=False); return df


def bootstrap(delta,block=7,reps=2000):
    a=np.asarray(delta,float); rng=np.random.default_rng(SEED); n=len(a)
    starts=rng.integers(0,n-block+1,size=(reps,int(np.ceil(n/block))))
    ix=(starts[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:n]
    sampled=a[ix].mean(axis=1)
    return float(a.mean()),float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))


def summarize(s):
    baseline=pd.read_csv(s.out/'daily_baselines.csv')
    risk=pd.read_csv(s.out/'daily_risk.csv'); rolling=pd.read_csv(s.out/'daily_rolling.csv')
    data=pd.concat([baseline,risk,rolling],ignore_index=True); data['date']=pd.to_datetime(data.date)
    periods={'full334':('2025-02-01','2025-12-31'),
             'fair214':('2025-06-01','2025-12-31'),'development153':('2025-06-01','2025-10-31'),
             'later61':('2025-11-01','2025-12-31')}
    summaries=[]; boot=[]
    for (method,policy),g in data.groupby(['method','policy']):
        g=g.sort_values('date')
        for period,(start,end) in periods.items():
            q=g[g.date.between(start,end)]; expected=len(pd.date_range(start,end))
            if len(q)!=expected: continue
            row={'method':method,'policy':policy,'period':period,'days':len(q)}
            for col in ['planned_cost','emergency_cost','total_cost','planned_kwh','emergency_kwh','spill_kwh','events','solve_seconds']:
                row[col]=float(q[col].sum())
            row.update(cvar90=cvar(q.total_cost,beta=.9),cvar95=cvar(q.total_cost,beta=.95),
                max_daily_cost=float(q.total_cost.max()),emergency_slot_frequency=q.emergency_slots.sum()/(T*len(q)),
                emergency_day_frequency=q.emergency_day.mean(),emergency_events_per_day=q.events.sum()/len(q),
                total_purchased_kwh=row['planned_kwh']+row['emergency_kwh'],
                start_soc=q.iloc[0].initial_soc,end_soc=q.iloc[-1].terminal_soc)
            summaries.append(row)
        if policy!='P1' and policy!='P5_daily':
            base=baseline[(baseline.method==method)&(baseline.policy=='P1')].copy()
            base['date']=pd.to_datetime(base.date)
            pair=g.merge(base[['date','total_cost']],on='date',suffixes=('','_base'))
            for period,(start,end) in list(periods.items())[1:]:
                q=pair[pair.date.between(start,end)]
                if len(q)<7:continue
                mean,lo,hi=bootstrap(q.total_cost_base-q.total_cost)
                boot.append({'method':method,'policy':policy,'period':period,'improvement_mean':mean,'ci_low':lo,'ci_high':hi,
                    'interpretation':'exploratory paired block interval; not selection-adjusted'})
    summary=pd.DataFrame(summaries)
    summary.to_csv(s.out/'summary.csv',index=False)
    empirical=[]
    for (method,period),g in summary.groupby(['method','period']):
        by=g.set_index('policy')
        if not {'P0','P1','P5_daily'}.issubset(by.index): continue
        oracle=float(by.loc['P5_daily','total_cost'])
        empirical.append({'method':method,'period':period,
            'actual_P0_minus_P1':by.loc['P0','total_cost']-by.loc['P1','total_cost'],
            'actual_P1_minus_daily_Oracle':by.loc['P1','total_cost']-oracle,
            'interpretation':'realized backtest differences, NOT theoretical VSS/EVPI'})
        for policy,r in by.iterrows():
            if not policy.startswith('P6'): assert r.total_cost>=oracle-.01,(policy,period)
    pd.DataFrame(empirical).to_csv(s.out/'empirical_economic_gaps.csv',index=False)
    continuous=pd.read_csv(s.out/'continuous_oracle.csv').set_index('period')
    comparisons=[]
    for _,r in summary.iterrows():
        if (r.policy.startswith('P6_') and r.period=='full334') or (r.policy.startswith('P6J_') and r.period=='fair214'):
            bound=float(continuous.loc[r.period,'total_cost'])
            assert abs(r.start_soc-INITIAL)<1e-5 and abs(r.end_soc-INITIAL)<1e-5
            assert r.total_cost>=bound-.01
            comparisons.append({'method':r.method,'policy':r.policy,'period':r.period,
                'actual_cost':r.total_cost,'matched_continuous_oracle':bound,
                'actual_oracle_gap':r.total_cost-bound,'oracle_regret_rate':(r.total_cost-bound)/bound})
    pd.DataFrame(comparisons).to_csv(s.out/'matched_rolling_oracle.csv',index=False)
    pd.DataFrame(boot).to_csv(s.out/'bootstrap.csv',index=False)
    data['month']=data.date.dt.month
    monthly=data.groupby(['method','policy','month'])[['planned_cost','emergency_cost','total_cost','emergency_kwh','spill_kwh']].sum().reset_index()
    monthly.to_csv(s.out/'monthly.csv',index=False)
    neutral=data[data.policy=='SAA_selected'][['method','date','total_cost']]
    rp=data[data.policy.str.startswith('P4')].merge(neutral,on=['method','date'],suffixes=('','_neutral'))
    riskcost=rp.groupby(['method','policy']).agg(mean_risk_protection_cost=('total_cost', 'mean'),
        mean_neutral_cost=('total_cost_neutral','mean')).reset_index()
    riskcost['delta_mean_cost']=riskcost.mean_risk_protection_cost-riskcost.mean_neutral_cost
    riskcost.to_csv(s.out/'risk_protection_cost.csv',index=False)
    checkcols=['eq_residual','ineq_violation','bound_violation','soc_recursion','dual_gap_relative','stationarity','complementarity','simultaneous_kwh']
    checks=data.groupby(['method','policy'])[checkcols].max()
    checks.to_csv(s.out/'checks_summary.csv')
    manifest={'fingerprint':s.fingerprint,'complete':True,'rows':len(data),'policies':sorted(data.policy.unique()),
        'methods':list(s.fc),'checks_max':data[checkcols].max().to_dict(),
        'claims':{'later_period_is_untouched':False,'daily_joint_dependence_used_by_risk_neutral':False,
                  'official_result2_overwritten':False,'lp_optimality_not_distribution_optimality':True},
        'ridge_input_precision':'saved predictions rounded to 6 decimals; legacy Manual reconstructed exactly',
        'initialization':'January battery idle; Feb1 6000 kWh',
        'completed_at':pd.Timestamp.now(tz='UTC').isoformat()}
    atom_json(s.out/'manifest.json',manifest)
    return summary,monthly


def parameter_registry(s):
    rows=[
        ('DT','1/6 h','数据','10分钟功率转电量；来自附件采样间隔','固定'),
        ('T','144','数据','24小时/10分钟','固定'),
        ('capacity','12000 kWh','题目','附录1','固定'),
        ('LO/HI','1200/10800 kWh','题目','附录1储能保护边界','固定'),
        ('CAP','5000/6 kWh/槽','题目+单位换算','5000 kW乘时间间隔','固定'),
        ('ETA','0.9/0.9','题意解释','充电和放电各90%；往返效率81%','沿用原解释'),
        ('INITIAL','6000 kWh','题目+初始化假设','1月1日6000；1月电池闲置，2月1日继承6000','所有策略共同'),
        ('emergency_multiplier','5','题目','Q2按时刻电价5倍紧急购电','固定'),
        ('daily_cyclic_SOC','6000 kWh','结构假设','P0—P5用于同边界比较；不是Q2明确强制','P6独立检验'),
        ('far_terminal_SOC','1200/6000/10800','结构敏感性','保护边界与原初始电量；不是已学习最优','P6逐一比较'),
        ('horizon_days','2/3','结构敏感性','跨过一个/两个午夜；每天只执行第一天','P6比较48/72小时'),
        ('window','14/21/28/42/56天','验证选择','覆盖2/3/4/6/8个完整周；保留原21天','月初历史费用验证'),
        ('half_life','7/14/28/无限天','验证选择','一周/两周/四周/不遗忘；rho=2^(-1/h)','月初历史费用验证'),
        ('conditional_pool','最多120天','工程结构','约四个月候选历史；不得包含当日或未来','固定并声明未证明最优'),
        ('neighbors','14/21/28/42','验证选择','有效样本量与相似性之间折中','月初历史费用验证'),
        ('conditional_features','季节/加星期/加负荷光伏预测','机制消融','依次增加信息；无天气与未来观测','历史候选池标准化'),
        ('calibration_days','28/56天','验证协议','最近四周主设置，八周敏感性；均早于决策月','月度外层验证'),
        ('near_tie','0.1%','预设决策规则','近似同价优先等权、少场景和简单结构','不事后调整阈值'),
        ('CVaR_beta','0.90/0.95','风险口径','关注最坏10%/5%日总费用；不是用户偏好的学习结果','分别报告'),
        ('CVaR_B','两端点间五个等距预算','内生风险前沿','端点来自风险中性和最小CVaR解；五点用于作图','每日从可实现范围生成'),
        ('lambda_risk','0及预算对偶乘子','优化输出','风险预算边际价格；非凭空指定的风险偏好','实际惩罚LP复核'),
        ('quantiles','0.5/0.8/0.9/0.95','诊断口径','中位数、简化经济分位、两种尾部风险水平','不混称覆盖保证'),
        ('Ridge_alpha','load=100; PV=10','既有3月验证','只用于P6递归未来输入；不重新搜索架构','沿用且注明网格局限'),
        ('EPS','1e-8 元/kWh','数值','沿用原吞吐打破平局项，不是真实退化费','报告主费用不含EPS'),
        ('HiGHS_tolerance','primal/dual 1e-8','数值','求解精度；独立可行性阈值1e-5','原始矩阵与对偶验证'),
        ('bootstrap','block=7,reps=2000,seed=2026','统计协议','保留周内相关性；探索性区间，不作选参目标','所有策略共用'),
        ('parallel_workers',str(s.jobs),'运行环境','最多4个CPU线程并行任务；单个HiGHS线程数1','不按结果调参'),
    ]
    df=pd.DataFrame(rows,columns=['parameter','value','source_type','reason','selection_or_check'])
    df.to_csv(s.out/'parameters.csv',index=False)
    return df


def figures_and_report(s):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import seaborn as sns
    from IPython.display import display, Markdown, HTML
    for path in Path('/usr/share/fonts').rglob('*CJK*.ttc'):
        font_manager.fontManager.addfont(str(path))
    fonts=[f.name for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
    if not fonts: raise RuntimeError('中文字体未安装，不能验收论文图')
    plt.rcParams.update({'font.family':fonts[0],'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
    summary=pd.read_csv(s.out/'summary.csv'); monthly=pd.read_csv(s.out/'monthly.csv')
    calibration=pd.read_csv(s.out/'calibration.csv'); info=pd.read_csv(s.out/'information_value.csv')
    boot=pd.read_csv(s.out/'bootstrap.csv')
    def table(df,title):
        display(Markdown('#### '+title))
        display(HTML('<div style="max-height:480px;overflow:auto">'+df.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</div>'))
    def emit(fig,name,caption):
        fig.savefig(s.fig/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(s.fig/f'{name}.png',dpi=140,bbox_inches='tight')
        display(fig); plt.close(fig); display(Markdown(caption))
    table(parameter_registry(s),'参数来源与选择方式')
    table(pd.DataFrame(json.loads((s.out/'selection.json').read_text())),'每月冻结的配置与历史选参费用')
    table(summary,'所有规划方案、全部评价期间的经济结果')
    table(pd.read_csv(s.out/'checks_summary.csv'),'原始LP与物理执行检查（按方法/方案取最不利值）')
    table(boot,'7日移动块Bootstrap：相对P1的日成本改善（正值为节省）')
    policies=['P0','P1','P2_equal21','P2_learned28','P3_learned28','SAA_selected','P5_daily']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=summary[(summary.method==method)&(summary.period=='fair214')&summary.policy.isin(policies)].set_index('policy').loc[policies]
        ax.bar(q.index,q.planned_cost/1e6,label='计划购电费')
        ax.bar(q.index,q.emergency_cost/1e6,bottom=q.planned_cost/1e6,label='紧急购电费')
        ax.tick_params(axis='x',rotation=35); ax.set_ylabel(f'{method}，6—12月费用（百万元）'); ax.legend()
    emit(fig,'p0_p5_cost','图：相同日循环边界下的P0—P5费用分解。观察场景、权重和条件信息能否降低真实补购费用；Oracle仅为同边界下界。')
    scores=pd.DataFrame(json.loads((s.out/'calibration_scores.json').read_text()))
    scores['window']=scores.cfg.map(lambda x:x.get('window',np.nan))
    scores['half']=scores.cfg.map(lambda x:str(x.get('half','conditional')))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=scores[(scores.method==method)&(scores.calibration_days==28)&scores.window.notna()]
        heat=q.groupby(['window','half']).mean_cost.mean().unstack()
        sns.heatmap(heat,annot=True,fmt='.0f',ax=ax,cmap='viridis_r',cbar_kws={'label':'历史校准日均费用（元）'})
        ax.set_xlabel('半衰期（None=等权）'); ax.set_ylabel(f'{method}，窗口天数')
    emit(fig,'parameter_heatmap','图：不同窗口和半衰期在历史校准段的费用。每月选择依据仅来自该月之前；图中跨月平均不作为重新选参依据。')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=calibration[calibration.method==method]
        for tau in sorted(q.tau.unique()):
            curve=q[q.tau==tau].groupby('month').coverage.mean()
            ax.plot(curve.index,curve,marker='o',label=f'分位数{tau}')
            ax.axhline(tau,color='gray',lw=.5,ls='--')
        ax.set_xlabel('月份'); ax.set_ylabel(f'{method}，分组平均覆盖率'); ax.legend(fontsize=8)
    emit(fig,'scenario_calibration','图：按月份与光伏水平分组后的覆盖率均值。虚线为名义分位数；组均值不等于按样本量加权的总体覆盖，完整分组样本另见校准表。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,method in zip(axes,s.fc):
        for beta in [.9,.95]:
            q=summary[(summary.method==method)&(summary.period=='fair214')&summary.policy.str.startswith(f'P4_b{beta}')].sort_values('policy')
            ax.plot(q.total_cost/q.days,q['cvar90' if beta==.9 else 'cvar95'],marker='o',label=f'β={beta}')
            for _,r in q.iterrows():ax.annotate(r.policy[-1],(r.total_cost/r.days,r['cvar90' if beta==.9 else 'cvar95']),fontsize=8)
        ax.set_xlabel('实际日均费用（元）'); ax.set_ylabel(f'{method}，实际日费用CVaR（元）'); ax.legend()
    emit(fig,'risk_frontier','图：五个风险预算策略的样本外平均费用与尾部费用。样本内风险预算并不保证样本外曲线单调，也不能保证所有CVaR策略都更经济。')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=info[info.method==method].copy(); q['month']=pd.to_datetime(q.date).dt.month
        q=q.groupby('month')[['VSS','EVPI']].mean()
        q.plot.bar(ax=ax); ax.set_xlabel('月份'); ax.set_ylabel(f'{method}，场景内日均信息价值（元）')
    emit(fig,'information_value','图：在每个决策日同一经验场景分布、同一日循环边界内计算的VSS和EVPI。它们不是实际回测费用差，也不保证真实未知分布下相同收益。')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=monthly[(monthly.method==method)&(monthly.month>=6)&monthly.policy.isin(policies[:-1])]
        mat=q.pivot(index='policy',columns='month',values='total_cost')
        mat=mat.subtract(mat.loc['P1'],axis=1)/1000
        sns.heatmap(mat,center=0,cmap='RdBu_r',ax=ax,cbar_kws={'label':'相对P1费用差（千元）'})
        ax.set_xlabel('月份'); ax.set_ylabel(method)
    emit(fig,'monthly_stability','图：月度相对P1费用差，负值为节省。11—12月是已经查看过的后期验证，不作为新的独立封存集。')
    slots=pd.read_csv(s.out/'rolling_slots.csv.gz')
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    for ax,method in zip(axes,s.fc):
        for policy in ['P6_H2_E6000','P6_H3_E6000']:
            q=slots[(slots.method==method)&(slots.policy==policy)&slots.date.between('2025-06-19','2025-06-23')]
            ax.plot(np.arange(len(q))/T,q.soc_end,label=policy,lw=1)
        ax.axhline(LO,color='gray',ls='--'); ax.axhline(HI,color='gray',ls='--')
        ax.set_xlabel('自6月19日起经过天数'); ax.set_ylabel(f'{method}，SOC（kWh）'); ax.legend()
    emit(fig,'rolling_soc','图：连续五天SOC轨迹。跨日状态继承前日末值；48/72小时远端边界仍可能影响第一天决策，故需同时查看终端敏感性。')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,s.fc):
        q=summary[(summary.method==method)&(summary.period=='full334')&summary.policy.str.startswith('P6')]
        for H in [2,3]:
            z=q[q.policy.str.contains(f'H{H}')].copy(); z['end']=z.policy.str.split('_E').str[-1].astype(int); z=z.sort_values('end')
            ax.plot(z.end,z.total_cost/1e6,marker='o',label=f'{24*H}小时')
        ax.set_xlabel('远端终端电量（kWh）'); ax.set_ylabel(f'{method}，2—12月费用（百万元）'); ax.legend()
    emit(fig,'terminal_sensitivity','图：相同2月1日初始SOC与12月31日末SOC下，远端边界和滚动时域的费用影响。全年合计是连续状态公平比较；6月分段费用须同时看分段起始SOC。')
    table(pd.read_csv(s.out/'risk_protection_cost.csv'),'风险保护代价：P4实际日均费用减同场景风险中性策略')
    table(info.groupby('method')[['RP','EEV','WS','VSS','EVPI']].sum().reset_index(),'经验场景分布内信息价值（6—12月合计）')
    table(pd.read_csv(s.out/'continuous_oracle.csv'),'连续SOC完美信息Oracle；起末均6000 kWh')
    table(pd.read_csv(s.out/'empirical_economic_gaps.csv'),'实际回测费用差：不得冒充理论VSS或EVPI')
    table(pd.read_csv(s.out/'matched_rolling_oracle.csv'),'P6连续状态及6月共同重启的匹配Oracle差距')
    table(pd.read_csv(s.out/'dependence.csv'),'等权场景配对打乱：风险中性目标与CVaR变化')
    table(pd.read_csv(s.out/'risk_penalty.csv').groupby(['method','beta']).agg(
        solves=('lambda','size'),lambda_min=('lambda','min'),lambda_max=('lambda','max'),
        mean_expected=('expected','mean'),mean_cvar=('cvar','mean')).reset_index(),
        '预算对偶导出的惩罚LP核对；逐日每个乘子保存在risk_penalty.csv')
    mechanism=pd.read_csv(s.out/'quantile_detail.csv.gz')
    mechanism=mechanism[mechanism.tau==.8].copy()
    mechanism['price_band']=np.where(mechanism.price>=np.quantile(s.price,.75),'高电价','其余时段')
    table(mechanism.groupby(['method','price_band']).agg(quantile_kw=('quantile_kw','mean'),
        coverage=('covered','mean'),upper_exceedance_kw=('upper_exceedance_kw','mean'),
        pinball=('pinball','mean')).reset_index(),'经济机制：80%分位数、高价上穿量与校准')
    later=summary[summary.period=='later61'].set_index(['method','policy'])
    lines=['### 实際结果与适用边界','以下结论自动读取已完成的规划结果；金额为真实回测费用，除非注明场景内。']
    for method in s.fc:
        q=summary[(summary.method==method)&(summary.period=='development153')&summary.policy.isin(policies[:-1])].sort_values('total_cost')
        win=q.iloc[0]; base=q[q.policy=='P1'].iloc[0]
        lines.append(f'- {method}：6—10月表内最低费用策略为{win.policy}，费用{win.total_cost:,.2f}元；P1为{base.total_cost:,.2f}元，差额{base.total_cost-win.total_cost:,.2f}元。对应策略11—12月费用为{later.loc[(method,win.policy),"total_cost"]:,.2f}元。此排名为已记录策略比较，不重新更改后期策略。')
    lines += ['- 所有统计结论仅针对本年数据与明确列出的信息集、费用结构和边界。计算出的有限场景LP最优性不等于真实分布全局最优。',
              '- 原预测端Ridge结果仍是固定规划器下的候选证据；本轮未修复ETS月内更新、PCA维数选择或LSTM内部验证等预测比较局限，因此不扩展为预测模型普遍优劣结论。',
              '- 日循环方案与跨日方案可行域不同，必须匹配Oracle和起末电量；CVaR风险水平是决策口径，对偶风险价格不是从数据自动发现的用户偏好。',
              '- P3历史池120天、相似性特征及网格范围是可解释但未证明最优的结构设置。每个参数的性质见登记表。',
              '- 所有用于本轮规划的实际观测、预测记录和校准标签必须早于决策日。']
    conclusion='\n\n'.join(lines)
    display(Markdown(conclusion)); (s.out/'conclusions.md').write_text(conclusion)
    report=s.root/'reports/RESULTS_REPORT.md'; report.parent.mkdir(exist_ok=True)
    with report.open('a') as f:
        f.write('\n\n## Q2规划端P0—P6实际运行补充\n'+conclusion+'\n')
        for p in sorted(s.fig.glob('*.pdf')): f.write(f'\n- 图表：{p.relative_to(s.root)}；绘图数据位于results/q2_planning_v1。')
    s.progress('completed',files=len(list(s.out.glob('*'))),pdfs=len(list(s.fig.glob('*.pdf'))))
    return summary
