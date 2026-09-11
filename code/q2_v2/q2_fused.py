"""A + staged dynamic calibration + shrinkage similar-days + temporal acceptance gate.

Standalone package: only the two raw input workbooks and bundled support modules.
Run --stage user, then --stage fusion, then --stage evaluate (or --stage all).
No notebook instructions are executed and no external q2_planning.py is required.
"""
from pathlib import Path
from dataclasses import dataclass,asdict,replace
import argparse,json,time,itertools,shutil,sys,hashlib,platform
import numpy as np
import pandas as pd
from utils import load_data,data_audit,BATTERY,DT,T,INTERVALS,write_csv,write_json
from dispatch import optimize_day,execute_day,check_dispatch
from forecast_core import OptimizedBank
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/q2_fused_v2'
FIG=ROOT/'figures/q2_fused_v2'
# ---- Parameter domains and protocol constants ---------------------------------
# Candidate grids are declared ex ante. Window lengths are integer multiples of the
# 7-day electricity-use cycle; rho spans [0,1] uniformly; smoothing compares no/weak/moderate smoothing.
ALPHAS=np.round(np.arange(.5,.951,.05),2)
WINDOWS=[14,21,28,42,56]
RHOS=[0.,.25,.5,.75,1.]
SMOOTHS=[0,3,5]
BLENDS=[0.,.1,.2,.3,.4,.5]  # conditional residual receives at most half the weight because its local sample is smaller
NEIGHBOR_K=[14,21,28,42]
FEATURES=['season','weekday','level']
MONTHS=[6,7,8,9,10]
SEED=20260911
SEARCH_DAYS=42          # 6 complete weeks
VALIDATION_DAYS=14      # 2 complete weeks
EMBARGO_DAYS=1          # one-day information isolation before deployment
BOOTSTRAP_BLOCK=7       # preserve weekly dependence
BOOTSTRAP_DRAWS=2000
BOOTSTRAP_CI=.90
CVAR_ALPHA=.95
RISK_VALIDATION_DAYS=56  # 95% tail mass = 2.8 days, unlike 14-day CVaR95 which equals max-day
NEAR_BLOCK=7
NEAR_DRAWS=2000
NEAR_CI=.90
ETARGETS=[1200.,4800.,6000.,7200.,10800.]

PARAMETER_REGISTRY=[
 dict(parameter='battery/SOC/power/efficiency',category='题设物理参数',source='题目与附件直接给定',selection='不参与寻优'),
 dict(parameter='alpha',category='数据学习参数',source='预先声明候选域 0.50–0.95；搜索段拟合、后置验证段验收',selection='滚动时间验证'),
 dict(parameter='window',category='结构候选参数',source='14/21/28/42/56 天，均围绕 7 日周期构造',selection='滚动时间验证'),
 dict(parameter='rho',category='数据学习参数',source='[0,1] 等距离散为 0/0.25/0.5/0.75/1',selection='滚动时间验证'),
 dict(parameter='smooth',category='结构候选参数',source='0/3/5 点：无平滑、弱平滑、中等平滑',selection='滚动时间验证'),
 dict(parameter='similar-day blend',category='数据学习参数',source='0–0.5 以 0.1 为步长；条件残差样本更小，故不允许权重超过全局残差',selection='搜索段选择+验证段验收'),
 dict(parameter='similar-day K',category='结构候选参数',source='14/21/28/42 天，均为完整周的整数倍',selection='搜索段选择+验证段验收'),
 dict(parameter='42/14/1 temporal split',category='实验协议参数',source='6 周搜索 + 2 周成本后置验证 + 1 日信息隔离',selection='事前固定，不按全年结果调节'),
 dict(parameter='near tie rule',category='统计选择规则',source='删除固定0.1%带宽；用7日循环块Bootstrap 90%区间判断候选相对最优者是否可区分',selection='搜索段内只作简约性规则，部署仍须独立验证'),
 dict(parameter='one-grid update limit',category='稳定性规则',source='每次正式更新仅允许在预先声明网格上移动一档',selection='事前固定，不按全年结果调节'),
 dict(parameter='bootstrap block=7',category='数值/统计参数',source='保留周周期相关结构',selection='事前固定'),
 dict(parameter='bootstrap CI=90%',category='统计口径参数',source='有限14日验证窗下的保守证据门槛',selection='事前固定并披露局限'),
 dict(parameter='CVaR alpha=95%',category='风险偏好参数',source='风险管理口径；不是由样本拟合出的参数',selection='使用56个历史日，尾部质量2.8日；14日CVaR95仅作退化诊断'),
 dict(parameter='risk validation=56 days',category='实验协议参数',source='8个完整周且CVaR95包含2.8个尾部日，避免14日CVaR95严格等于worst-day',selection='事前固定；不作为成本收益验证窗'),
 dict(parameter='inventory value',category='经济核算参数',source='主值为最低电价补库成本 p_min/eta_c；另报0与eta_d*p_max上下口径',selection='三口径敏感性，不声称精确市场终值'),
 dict(parameter='reference E_target',category='边界结构参数',source='6000为题给初始SOC和容量50%；1200/4800/7200/10800覆盖保护边界与±10%容量',selection='6000主报告，五档实际重跑敏感性'),
 dict(parameter='SciPy=1.15.3',category='数值环境参数',source='Colab SciPy 1.16.3 对旧SAA稀疏LP返回HiGHS Status Not Set；1.15.3通过常数/真实场景试运行',selection='锁定环境并仍检查原始/对偶残差'),
 dict(parameter='forecast_core inherited constants',category='继承模型参数',source='来自已审计A预测主干；本版为保持基准可比性不重新事后寻优',selection='保留并在报告中单列敏感性/局限'),
]

@dataclass(frozen=True)
class Config:
    alpha:float=.8
    rho:float=.5
    window:int=28
    smooth:int=3
    features:str='none'
    k:int=21
    blend:float=0.

DEFAULT=Config()
def cfgdict(d):return Config(**{k:d[k] for k in Config.__dataclass_fields__})
def cv(x,alpha=.95):
    x=np.asarray(x,float);x=np.sort(x)[::-1];mass=len(x)*(1-alpha)
    if len(x)==0 or mass<=0:raise ValueError('CVaR 需要非空样本且 alpha<1')
    whole=int(np.floor(mass));frac=mass-whole
    return float((x[:whole].sum()+(x[whole]*frac if whole<len(x) else 0))/mass)
def circular_block_interval(x,block=7,draws=2000,ci=.90,seed=SEED):
    """循环移动块 Bootstrap 的均值区间；保留周内依赖。"""
    x=np.asarray(x,float);n=len(x)
    if n==0 or not np.isfinite(x).all():raise ValueError('Bootstrap 样本非法')
    block=min(int(block),n);rng=np.random.default_rng(seed);blocks=int(np.ceil(n/block))
    starts=rng.integers(0,n,size=(int(draws),blocks))
    ids=((starts[:,:,None]+np.arange(block))%n).reshape(int(draws),-1)[:,:n]
    means=x[ids].mean(1);tail=(1-ci)/2
    return float(x.mean()),float(np.quantile(means,tail)),float(np.quantile(means,1-tail))
def smooth(x,width):
    if not width:return x.copy()
    kernel={3:np.array([1,2,1])/4,5:np.array([1,4,6,4,1])/16}[width]
    return np.convolve(np.pad(x,(width//2,width//2),mode='edge'),kernel,'valid')
def json_records(frame):return frame.replace({np.nan:None}).to_dict('records')

class Engine:
    def __init__(self,data=None):
        self.data=data if data is not None else load_data();self.bank=OptimizedBank(self.data,pv_window=5)
        self.dates=self.data['dates'];self.price=self.data['price']
        # 单位期末 SOC 的三种透明核算口径。主值是用最低电价补回 1 kWh
        # 电池内能量的最低采购成本；上口径是高价时放出 1 kWh SOC 可避免的购电费。
        self.inventory_values={'zero':0.,'replacement':float(self.price.min()/BATTERY.eta_c),
          'avoidance_upper':float(BATTERY.eta_d*self.price.max())}
        self.value=self.inventory_values['replacement']
        doy=np.array([d.timetuple().tm_yday for d in self.dates]);dow=np.array([d.weekday() for d in self.dates])
        self.feature=np.column_stack([np.sin(2*np.pi*doy/365),np.cos(2*np.pi*doy/365),
          np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),
          np.maximum(self.bank.net+self.bank.pv_hat*DT,0).sum(1),(self.bank.pv_hat*DT).sum(1)])
        self.margin_cache={};self.score_cache={};self.checks=[];self.lp_calls=0

    def neighbors(self,day,features,k):
        ids=np.arange(max(7,day-120),day);dim={'season':2,'weekday':4,'level':6}[features]
        x=self.feature[ids,:dim];scale=x.std(0);scale[scale<1e-9]=1
        dist=np.sum(((x-self.feature[day,:dim])/scale)**2,axis=1)
        return np.sort(ids[np.argsort(dist,kind='stable')[:min(k,len(ids))]])

    def risk(self,day,cfg):
        key=(day,cfg.alpha,cfg.window,cfg.smooth,cfg.features,cfg.k,cfg.blend)
        if key in self.margin_cache:return self.margin_cache[key]
        ids=np.arange(max(1,day-cfg.window),day);pool=self.bank.residual[ids]
        finite=np.isfinite(pool).all(1);ids=ids[finite];pool=pool[finite]
        assert len(pool) and ids.max()<day
        margin=np.quantile(pool,cfg.alpha,axis=0)
        if cfg.blend and cfg.features!='none':
            chosen=self.neighbors(day,cfg.features,cfg.k);assert chosen.max()<day
            cond=np.quantile(self.bank.residual[chosen],cfg.alpha,axis=0)
            margin=(1-cfg.blend)*margin+cfg.blend*cond
        net=self.bank.net[day]+smooth(margin,cfg.smooth)
        assert np.isfinite(net).all();self.margin_cache[key]=net
        return net

    def replay(self,start,stop,cfg=DEFAULT,lookup=None,keep=False,initial=6000.,
               e_target=6000.,inventory_value=None):
        """按日重规划并因果执行。

        e_target 只约束每日 LP 的参考轨迹终点，实际执行的日末 SOC 不被
        强制等于它，并在下一天继承。inventory_value 只用于跨策略公平核算，
        不进入 LP 目标，因而不改变决策。
        """
        value=self.value if inventory_value is None else float(inventory_value)
        state=float(initial);rows=[];slots=[];before=time.perf_counter()
        for day in range(start,stop):
            c=lookup(day) if lookup else cfg
            plan=optimize_day(self.risk(day,c),self.price,state,float(e_target));self.lp_calls+=1
            ex=execute_day(plan.purchase,self.bank.actual_net[day],state,plan.soc,c.rho)
            z=ex['emergency'];pc=float(self.price@plan.purchase);ec=float(5*self.price@z)
            row=dict(date=str(self.dates[day]),day=day,initial_soc=state,terminal_soc=float(ex['soc'][-1]),
              total_cost=pc+ec,planned_cost=pc,emergency_cost=ec,emergency_kwh=float(z.sum()),spill_kwh=float(ex['spill'].sum()),
              planned_kwh=float(plan.purchase.sum()),charge_kwh=float(ex['charge'].sum()),
              discharge_kwh=float(ex['discharge'].sum()),emergency_slots=int(np.sum(z>1e-6)),
              emergency_day=int(np.any(z>1e-6)),events=int(np.sum((z>1e-6)&~np.r_[False,z[:-1]>1e-6])),
              adjusted_cost=pc+ec-value*(ex['soc'][-1]-state),inventory_value_rate=value,
              reference_terminal_target=float(e_target),**asdict(c))
            rows.append(row)
            if keep:
                chk=check_dispatch(self.bank.actual_net[day],plan.purchase,ex['charge'],ex['discharge'],ex['spill'],ex['soc'],z)
                assert chk['pass'];self.checks.append(dict(date=row['date'],**chk))
                slots.append(pd.DataFrame(dict(date=row['date'],slot=np.arange(T),interval=INTERVALS,price=self.price,
                  net_actual=self.bank.actual_net[day],q=plan.purchase,reserve=BATTERY.minimum+c.rho*(plan.soc[1:]-BATTERY.minimum),
                  charge=ex['charge'],discharge=ex['discharge'],emergency=z,spill=ex['spill'],soc_start=ex['soc'][:-1],soc_end=ex['soc'][1:])))
            state=float(ex['soc'][-1])
        return pd.DataFrame(rows),pd.concat(slots,ignore_index=True) if slots else None,time.perf_counter()-before

    def score(self,cfg,start,stop):
        key=(cfg,start,stop)
        if key not in self.score_cache:
            f,_,_=self.replay(start,stop,cfg)
            self.score_cache[key]=(float(f.adjusted_cost.sum()),f)
        return self.score_cache[key]

def near(records,stage):
    """在搜索段用配对周块 Bootstrap 判定“统计上无法区分”的简约候选集。

    原 V2 的 score <= best*(1+0.1%) 是固定工程带宽，没有由样本波动导出。
    现在对每个候选与搜索段最低成本候选的逐日调整成本差作配对块抽样；
    若候选“更差”差值的 90% 区间下界不大于 0，则尚不能证明它比最优者更差，
    允许再按事前简约性次序打破平局。真正部署仍须通过后置验证门。
    """
    bestrow=min(records,key=lambda r:r['score']);pool=[]
    base=np.asarray(bestrow['_daily'],float)
    for j,r in enumerate(records):
        delta=np.asarray(r['_daily'],float)-base
        mean,low,high=circular_block_interval(delta,NEAR_BLOCK,NEAR_DRAWS,NEAR_CI,
          SEED+{'coarse':1000,'local':2000,'conditional':3000}[stage])
        r.update(near_reference_score=float(bestrow['score']),near_delta_daily_mean=mean,
          near_ci_low90=low,near_ci_high90=high,near_indistinguishable=bool(low<=0.))
        if low<=0.:pool.append(r)
    if not pool:pool=[bestrow]
    if stage=='coarse':key=lambda r:(abs(r['alpha']-.8),r['window'],r['score'])
    elif stage=='conditional':key=lambda r:(r['blend']>0,{'none':0,'season':1,'weekday':2,'level':3}[r['features']],abs(r['alpha']-.8),abs(r['rho']-.5),r['k'],r['score'])
    else:key=lambda r:(abs(r['alpha']-.8),abs(r['rho']-.5),r['smooth'],r['score'])
    return min(pool,key=key)

def search(engine,start,stop,conditional=False):
    records=[]
    def scan(configs,stage):
        part=[]
        for c in configs:
            score,f=engine.score(c,start,stop)
            row=dict(stage=stage,fit_start=str(engine.dates[start]),fit_end=str(engine.dates[stop-1]),score=score,
              _daily=f.adjusted_cost.to_numpy(copy=True),**asdict(c))
            records.append(row);part.append(row)
        return part
    coarse=scan([Config(alpha=float(a),window=w) for a in ALPHAS for w in WINDOWS],'coarse')
    one=near(coarse,'coarse')
    fine=np.unique(np.round(np.clip(np.arange(one['alpha']-.04,one['alpha']+.041,.01),.5,.95),2))
    local=scan([Config(float(a),r,one['window'],sm) for a,r,sm in itertools.product(fine,RHOS,SMOOTHS)],'local')
    normal=cfgdict(near(local,'local'))
    best=normal
    if conditional:
        alternatives=scan([replace(normal,features=f,k=k,blend=b) for f,k,b in itertools.product(FEATURES,NEIGHBOR_K,BLENDS[1:])],'conditional')
        pscore,pframe=engine.score(normal,start,stop)
        plain=dict(score=pscore,_daily=pframe.adjusted_cost.to_numpy(copy=True),**asdict(normal))
        best=cfgdict(near([plain]+alternatives,'conditional'))
    clean=pd.DataFrame([{k:v for k,v in r.items() if k!='_daily'} for r in records])
    return normal,best,clean

def origin(engine,month):return next(i for i,d in enumerate(engine.dates) if d.month==month and d.day==1)
def lookup(engine,selection):
    # Key by month to preserve the supplied June–October / November–December protocol.
    configs={int(r['month']):cfgdict(r['config']) for r in selection}
    return lambda day: DEFAULT if engine.dates[day].month<6 else configs[min(engine.dates[day].month,10)]

def _nearest(grid,value):
    return min(grid,key=lambda z:abs(float(z)-float(value)))

def limit_change(proposed,incumbent):
    """Stability constraint defined by model structure, not ad-hoc numeric deltas.

    Each deployed parameter may move at most one adjacent position on its declared policy grid.
    Alpha is therefore governed by the coarse policy grid (0.05 spacing), while rho/window/smooth
    use their own declared grids.  Conditional structure/blend is not forced on if the incumbent
    has no conditional layer; it still has to pass the independent validation gate.
    """
    def step(grid,new,old):
        n=_nearest(grid,new);o=_nearest(grid,old)
        i,j=grid.index(n),grid.index(o)
        return grid[j+int(np.clip(i-j,-1,1))]
    return replace(proposed,
      alpha=float(step(list(ALPHAS),proposed.alpha,incumbent.alpha)),
      rho=float(step(RHOS,proposed.rho,incumbent.rho)),
      window=int(step(WINDOWS,proposed.window,incumbent.window)),
      smooth=int(step(SMOOTHS,proposed.smooth,incumbent.smooth)))

def gate(engine,proposal,incumbent,start,stop,economic_threshold=0.):
    """Paired temporal acceptance gate.

    Main rule deliberately avoids the former 1.1×CVaR+1 yuan tolerance.  A candidate is accepted
    only when (i) paired block-bootstrap evidence supports lower adjusted cost and (ii) its observed
    emergency-cost CVaR95 is non-inferior to the incumbent.  `economic_threshold` is retained only
    for pre-declared ablations; the main fused model uses zero.
    """
    _,new=engine.score(proposal,start,stop);_,old=engine.score(incumbent,start,stop)
    delta=old.adjusted_cost.to_numpy()-new.adjusted_cost.to_numpy()
    _,low,high=circular_block_interval(delta,BOOTSTRAP_BLOCK,BOOTSTRAP_DRAWS,BOOTSTRAP_CI,SEED+start)
    relative=float(delta.sum()/abs(old.adjusted_cost.sum()))
    # 14 日样本下 (1-0.95)*14=0.7 日，精确经验 CVaR95 必然退化为最坏日。
    # 因此 14 日窗口只验证平均成本改善；风险非劣性在同一截止点的
    # 56 个历史日上计算，使 95% 尾部质量为 2.8 日。
    risk_start=max(1,stop-RISK_VALIDATION_DAYS)
    _,newrisk=engine.score(proposal,risk_start,stop);_,oldrisk=engine.score(incumbent,risk_start,stop)
    risk_old=cv(oldrisk.emergency_cost,CVAR_ALPHA);risk_new=cv(newrisk.emergency_cost,CVAR_ALPHA)
    diagnostic_old=cv(old.emergency_cost,CVAR_ALPHA);diagnostic_new=cv(new.emergency_cost,CVAR_ALPHA)
    gain_ok=relative>economic_threshold and low>0
    risk_ok=risk_new<=risk_old+1e-9
    passed=gain_ok and risk_ok
    return passed,dict(validation_start=str(engine.dates[start]),validation_end=str(engine.dates[stop-1]),
      incumbent_adjusted=float(old.adjusted_cost.sum()),candidate_adjusted=float(new.adjusted_cost.sum()),
      relative_improvement=relative,bootstrap_daily_low90=low,bootstrap_daily_high90=high,
      validation_observations=len(old),validation_tail_mass_days=(1-CVAR_ALPHA)*len(old),
      validation_old_emergency_cvar95=diagnostic_old,validation_new_emergency_cvar95=diagnostic_new,
      validation_old_worst_emergency=float(old.emergency_cost.max()),validation_new_worst_emergency=float(new.emergency_cost.max()),
      risk_validation_start=str(engine.dates[risk_start]),risk_validation_end=str(engine.dates[stop-1]),
      risk_observations=len(oldrisk),risk_tail_mass_days=(1-CVAR_ALPHA)*len(oldrisk),
      old_emergency_cvar95=risk_old,new_emergency_cvar95=risk_new,accepted=passed,
      gate_rule='bootstrap_cost14_improvement_and_CVaR95_56_noninferiority',
      reason='accepted' if passed else ('economic_gain_below_declared_threshold' if relative<=economic_threshold else 'uncertain_cost_gain' if low<=0 else 'emergency_cvar_inferiority'))

def run_user():
    e=Engine();selection=[];scores=[];start=time.perf_counter()
    for m in MONTHS:
        o=origin(e,m);normal,_,f=search(e,o-56,o,False)
        selection.append(dict(month=m,decision_date=str(e.dates[o]),config=asdict(normal),history_start=str(e.dates[o-56]),history_end=str(e.dates[o-1])))
        scores.append(f.assign(month=m));write_json(OUT/'user_selection.json',selection)
        write_csv(pd.concat(scores),OUT/'user_scores.csv');print('user',m,asdict(normal),round(time.perf_counter()-start,1),flush=True)
    f,d,sec=e.replay(31,365,lookup=lookup(e,selection),keep=True)
    write_csv(f,OUT/'user_daily.csv');write_csv(d,OUT/'user_slots.csv');write_csv(pd.DataFrame(e.checks),OUT/'user_checks.csv')
    write_json(OUT/'user_run.json',dict(seconds=time.perf_counter()-start,lp_calls=e.lp_calls,scope='faithful H3 staged search with TeamA forecast; not all SAA/Ridge branches'))

def write_parameter_registry():
    frame=pd.DataFrame(PARAMETER_REGISTRY)
    write_csv(frame,OUT/'parameter_registry.csv')
    write_json(OUT/'parameter_registry.json',PARAMETER_REGISTRY)

def run_fusion():
    write_parameter_registry();e=Engine();start=time.perf_counter();selections={k:[] for k in ['fused','no_conditional','search_only','loose_gate','strict_gate']};scores=[]
    current={k:DEFAULT for k in selections}
    for m in MONTHS:
        o=origin(e,m);normal,best,f=search(e,o-SEARCH_DAYS-VALIDATION_DAYS-EMBARGO_DAYS,o-VALIDATION_DAYS-EMBARGO_DAYS,True);scores.append(f.assign(month=m))
        for name in selections:
            incumbent=current[name];candidate=normal if name=='no_conditional' else best
            if name=='search_only':proposed=candidate;chosen=candidate;g=dict(accepted=True,reason='fit_only_ablation')
            else:
                proposed=limit_change(candidate,incumbent)
                accepted,g=gate(e,proposed,incumbent,o-VALIDATION_DAYS-EMBARGO_DAYS,o-EMBARGO_DAYS,economic_threshold={'strict_gate':.005}.get(name,0.))
                chosen=proposed if accepted else incumbent
            selections[name].append(dict(month=m,decision_date=str(e.dates[o]),fit_start=str(e.dates[o-SEARCH_DAYS-VALIDATION_DAYS-EMBARGO_DAYS]),fit_end=str(e.dates[o-VALIDATION_DAYS-EMBARGO_DAYS-1]),
               embargo_date=str(e.dates[o-EMBARGO_DAYS]),config=asdict(chosen),unrestricted_candidate=asdict(candidate),
               limited_candidate=asdict(proposed),incumbent=asdict(incumbent),**g))
            current[name]=chosen
        write_json(OUT/'fused_selection.json',selections);write_csv(pd.concat(scores),OUT/'fused_scores.csv')
        print('fused',m,selections['fused'][-1],round(time.perf_counter()-start,1),flush=True)
    for name,sel in selections.items():
        f,d,seconds=e.replay(31,365,lookup=lookup(e,sel),keep=True)
        write_csv(f,OUT/f'{name}_daily.csv');write_csv(d,OUT/f'{name}_slots.csv')
    base,slots,sec=e.replay(31,365,keep=True);write_csv(base,OUT/'baseline_daily.csv');write_csv(slots,OUT/'baseline_slots.csv')
    write_csv(pd.DataFrame(e.checks),OUT/'fused_checks.csv')
    write_json(OUT/'fused_run.json',dict(seconds=time.perf_counter()-start,lp_calls=e.lp_calls,
      note='parameter domains/protocol fixed ex ante; main gate uses bootstrap cost evidence plus CVaR95 non-inferiority; ablations are not post-hoc model promotion'))

def _file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def _ridge_study():
    """读取既有 Ridge 预测明细，但从原始日序列重做全部 H1-replan 规划/执行。

    与旧 V1 的 daily/summary/cache 没有任何读依赖；预测 CSV 是已冻结的
    GitHub 比较器输入，其 SHA256 写入清单。
    """
    code_root=str(ROOT/'code')
    if code_root not in sys.path:sys.path.insert(0,code_root)
    from q2_planning import Study
    return Study(ROOT)

def run_ridge_h1_replan(e_target=6000.,save=True,study=None):
    """用冻结 Ridge 预测重新求解每日 21 场景、14 天半衰期 SAA，因果执行并继承 SOC。"""
    s=study if study is not None else _ridge_study()
    code_root=str(ROOT/'code')
    if code_root not in sys.path:sys.path.insert(0,code_root)
    from q2_planning import weights_for
    fc=s.fc['Ridge'].copy();fc[:,:90]=s.legacy[:,:90]
    netfc=fc[0]-fc[1];state=6000.;rows=[];slots=[];checks=[];started=time.perf_counter()
    value=float(s.price.min()/BATTERY.eta_c)
    for d in range(31,365):
        hist=np.arange(d-21,d);scen=netfc[d]+s.net[hist]-netfc[hist]
        sol=s.solve(scen,weights_for(len(hist),14),initial=state,terminal=float(e_target))
        ex=execute_day(sol['q'],s.net[d]*DT,state,sol['soc'],.5)
        chk=check_dispatch(s.net[d]*DT,sol['q'],ex['charge'],ex['discharge'],ex['spill'],ex['soc'],ex['emergency'])
        if not chk['pass']:raise AssertionError(chk)
        pc=float(s.price@sol['q']);ec=float(5*s.price@ex['emergency']);date=str(s.dates[d].date())
        z=ex['emergency']
        rows.append(dict(date=date,day=d,initial_soc=state,terminal_soc=float(ex['soc'][-1]),
          total_cost=pc+ec,planned_cost=pc,emergency_cost=ec,planned_kwh=float(sol['q'].sum()),
          emergency_kwh=float(z.sum()),spill_kwh=float(ex['spill'].sum()),charge_kwh=float(ex['charge'].sum()),
          discharge_kwh=float(ex['discharge'].sum()),emergency_slots=int(np.sum(z>1e-6)),
          emergency_day=int(np.any(z>1e-6)),events=int(np.sum((z>1e-6)&~np.r_[False,z[:-1]>1e-6])),
          adjusted_cost=pc+ec-value*(ex['soc'][-1]-state),inventory_value_rate=value,
          reference_terminal_target=float(e_target),alpha=np.nan,rho=.5,window=21,smooth=np.nan,
          features='Ridge residual SAA',k=21,blend=np.nan,solve_seconds=float(sol['seconds'])))
        checks.append(dict(date=date,**chk,lp_eq_residual=sol['checks']['eq_residual'],
          lp_dual_gap_relative=sol['checks']['dual_gap_relative'],lp_stationarity=sol['checks']['stationarity']))
        slots.append(pd.DataFrame(dict(date=date,slot=np.arange(T),interval=INTERVALS,price=s.price,
          net_actual=s.net[d]*DT,q=sol['q'],reserve=BATTERY.minimum+.5*(sol['soc'][1:]-BATTERY.minimum),
          charge=ex['charge'],discharge=ex['discharge'],emergency=z,spill=ex['spill'],
          soc_start=ex['soc'][:-1],soc_end=ex['soc'][1:])))
        state=float(ex['soc'][-1])
    frame=pd.DataFrame(rows);detail=pd.concat(slots,ignore_index=True);audit=pd.DataFrame(checks)
    if save:
        write_csv(frame,OUT/'ridge_h1_replan_daily.csv');write_csv(detail,OUT/'ridge_h1_replan_slots.csv')
        write_csv(audit,OUT/'ridge_h1_replan_checks.csv')
        prediction=ROOT/'results/problem2_forecast_ablation_predictions.csv.gz'
        write_json(OUT/'ridge_h1_replan_run.json',dict(seconds=time.perf_counter()-started,
          recomputed_planning=True,reused_v1_daily_results=False,prediction_artifact=str(prediction.relative_to(ROOT)),
          prediction_sha256=_file_sha(prediction),comparison_definition='Ridge + 21-day/14-day-half-life SAA + H1 causal replanning'))
    return frame,detail,audit

def metrics(frame,name,period):
    f=frame
    return dict(strategy=name,period=period,days=len(f),total_cost=f.total_cost.sum(),planned_cost=f.planned_cost.sum(),
      emergency_cost=f.emergency_cost.sum(),planned_kwh=f.planned_kwh.sum(),emergency_kwh=f.emergency_kwh.sum(),
      spill_kwh=f.spill_kwh.sum(),charge_kwh=f.charge_kwh.sum(),discharge_kwh=f.discharge_kwh.sum(),
      emergency_slots=f.emergency_slots.sum(),emergency_slot_frequency=f.emergency_slots.sum()/(T*len(f)),
      emergency_day_frequency=f.emergency_day.mean(),events=f.events.sum(),daily_cvar90=cv(f.total_cost,.90),
      daily_cvar95=cv(f.total_cost,.95),worst_day=f.total_cost.max(),initial_soc=f.iloc[0].initial_soc,
      terminal_soc=f.iloc[-1].terminal_soc,inventory_value_rate=f.inventory_value_rate.iloc[0],
      adjusted_cost=f.total_cost.sum()-f.inventory_value_rate.iloc[0]*(f.iloc[-1].terminal_soc-f.iloc[0].initial_soc))

PERIODS=[('full334','2025-02-01','2025-12-31'),('development153','2025-06-01','2025-10-31'),
         ('frozen61','2025-11-01','2025-12-31')]

def run_sensitivities():
    """实际重跑 E_target；期末价值不改决策，在同一轨迹上用三种透明口径重算。"""
    e=Engine();fs=json.loads((OUT/'fused_selection.json').read_text());ridge_study=_ridge_study()
    target_daily=[];target_summary=[]
    for target in ETARGETS:
        runs={}
        if target==6000.:
            for name in ['baseline','fused','ridge_h1_replan']:
                runs[name]=pd.read_csv(OUT/f'{name}_daily.csv')
        else:
            runs['baseline']=e.replay(31,365,e_target=target)[0]
            runs['fused']=e.replay(31,365,lookup=lookup(e,fs['fused']),e_target=target)[0]
            runs['ridge_h1_replan']=run_ridge_h1_replan(target,False,ridge_study)[0]
        for name,f in runs.items():
            q=f.copy();q['strategy']=name;q['sensitivity_target']=target;target_daily.append(q)
            base=metrics(q,name,'full334');base['sensitivity_target']=target
            for valuation,rate in e.inventory_values.items():
                row={**base,'inventory_valuation':valuation,'inventory_value_rate':rate}
                row['adjusted_cost']=row['total_cost']-rate*(row['terminal_soc']-row['initial_soc'])
                target_summary.append(row)
        print('target sensitivity',target,flush=True)
    td=pd.concat(target_daily,ignore_index=True);write_csv(td,OUT/'terminal_target_daily.csv.gz')
    write_csv(pd.DataFrame(target_summary),OUT/'terminal_target_sensitivity.csv')
    values=[]
    for name in ['baseline','user','fused','no_conditional','search_only','loose_gate','strict_gate','ridge_h1_replan']:
        f=pd.read_csv(OUT/f'{name}_daily.csv')
        for period,start,stop in PERIODS:
            q=f[f.date.between(start,stop)]
            for label,rate in e.inventory_values.items():
                values.append(dict(strategy=name,period=period,inventory_valuation=label,inventory_value_rate=rate,
                  raw_total_cost=float(q.total_cost.sum()),initial_soc=float(q.iloc[0].initial_soc),
                  terminal_soc=float(q.iloc[-1].terminal_soc),terminal_delta=float(q.iloc[-1].terminal_soc-q.iloc[0].initial_soc),
                  adjusted_cost=float(q.total_cost.sum()-rate*(q.iloc[-1].terminal_soc-q.iloc[0].initial_soc))))
    write_csv(pd.DataFrame(values),OUT/'inventory_value_sensitivity.csv')

def evaluate():
    write_parameter_registry();e=Engine();names=['baseline','user','fused','no_conditional','search_only','loose_gate','strict_gate','ridge_h1_replan'];allrows=[];summaries=[];tests=[]
    def check(name,ok,**extra):
        if not ok:raise AssertionError((name,extra))
        tests.append(dict(test=name,passed=True,**extra))
    for name in names:
        f=pd.read_csv(OUT/f'{name}_daily.csv');allrows.append(f.assign(strategy=name))
        check(name+' complete days',len(f)==334 and f.date.nunique()==334)
        check(name+' continuous SOC',np.allclose(f.initial_soc.to_numpy()[1:],f.terminal_soc.to_numpy()[:-1],atol=1e-6,rtol=0))
        check(name+' settlement',np.max(np.abs(f.total_cost-f.planned_cost-f.emergency_cost))<1e-6)
        for period,start,stop in PERIODS:summaries.append(metrics(f[f.date.between(start,stop)],name,period))
    daily=pd.concat(allrows,ignore_index=True);summary=pd.DataFrame(summaries);write_csv(summary,OUT/'summary.csv')
    observed=float(summary[(summary.strategy=='baseline')&(summary.period=='full334')].total_cost.iloc[0])
    check('A fixed regression reproduced from raw inputs',abs(observed-13820986.576531284)<1e-5,observed=observed)
    from user_forecast_reference import build_team_a_forecast
    fc,_=build_team_a_forecast(e.data['load'],e.data['pv'],e.dates);ported=(fc[0]-fc[1])*DT
    diff=float(np.max(np.abs(ported[31:]-e.bank.net[31:])))
    check('user TeamA forecast matches audited A',diff<1e-7,max_difference_kwh=diff)
    fs=json.loads((OUT/'fused_selection.json').read_text());validation=[]
    for name,selection in fs.items():
        for r in selection:
            validation.append(dict(strategy=name,**{k:v for k,v in r.items() if not isinstance(v,dict)}))
        if name=='search_only':continue
        for r in selection:
            check(f'{name} selection chronology month {r["month"]}',r['fit_end']<r['validation_start']<=r['validation_end']<r['embargo_date']<r['decision_date'])
            check(f'{name} 14-day CVaR95 degeneracy recorded month {r["month"]}',
              abs(r['validation_old_emergency_cvar95']-r['validation_old_worst_emergency'])<1e-7 and
              abs(r['validation_new_emergency_cvar95']-r['validation_new_worst_emergency'])<1e-7)
            check(f'{name} 56-day risk window month {r["month"]}',r['risk_observations']==56 and abs(r['risk_tail_mass_days']-2.8)<1e-9)
            a,b=cfgdict(r['limited_candidate']),cfgdict(r['incumbent'])
            check(f'{name} step limit month {r["month"]}',abs(list(ALPHAS).index(_nearest(list(ALPHAS),a.alpha))-list(ALPHAS).index(_nearest(list(ALPHAS),b.alpha)))<=1 and abs(RHOS.index(_nearest(RHOS,a.rho))-RHOS.index(_nearest(RHOS,b.rho)))<=1)
            o=origin(e,r['month']);passed,g=gate(e,a,b,o-VALIDATION_DAYS-EMBARGO_DAYS,o-EMBARGO_DAYS,economic_threshold={'strict_gate':.005}.get(name,0.))
            check(f'{name} gate replay month {r["month"]}',bool(passed)==r['accepted'] and abs(g['candidate_adjusted']-r['candidate_adjusted'])<1e-6)
        f=pd.read_csv(OUT/f'{name}_daily.csv');oct=cfgdict(selection[-1]['config']);later=f[f.date>='2025-11-01']
        check(name+' November December parameter freeze',all(all(abs(float(row[k])-float(getattr(oct,k)))<1e-8 if k!='features' else row[k]==oct.features for k in asdict(oct)) for row in later.to_dict('records')))
    write_csv(pd.DataFrame(validation),OUT/'validation_diagnostics.csv')
    scores=pd.read_csv(OUT/'fused_scores.csv')
    check('near uses statistical fields',{'near_ci_low90','near_ci_high90','near_indistinguishable'}.issubset(scores.columns))
    # 全链路时间前缀测试：改变当日及未来观测，不应改变当日决策输入。
    for day in [151,243,334]:
        changed={**e.data,'load':e.data['load'].copy(),'pv':e.data['pv'].copy()};changed['load'][day:]*=3;changed['pv'][day:]=0
        other=Engine(changed);prefix={**changed,'dates':changed['dates'][:day+1],'load':changed['load'][:day+1],'pv':changed['pv'][:day+1]};short=Engine(prefix)
        for cfg in [DEFAULT,Config(.85,.25,42,5,'level',21,.3)]:
            q=e.risk(day,cfg);x=other.risk(day,cfg);p=short.risk(day,cfg)
            check(f'future and prefix risk invariant {day} {cfg.features}',np.array_equal(q,x) and np.array_equal(q,p))
            plan=optimize_day(q,e.price,6000);actual=e.bank.actual_net[day].copy();alt=actual.copy();alt[72:]+=1e6
            one=execute_day(plan.purchase,actual,6000,plan.soc,cfg.rho);two=execute_day(plan.purchase,alt,6000,plan.soc,cfg.rho)
            check(f'intraday prefix {day} {cfg.features}',all(np.array_equal(one[k][:72],two[k][:72]) for k in ['charge','discharge','emergency','spill']))
        for cfg in [DEFAULT,Config(.85,.25,42,5,'level',21,.3)]:
            x=e.score(cfg,day-15,day-1)[0];y=other.score(cfg,day-15,day-1)[0]
            check(f'past validation unaffected by future {day} {cfg.features}',abs(x-y)<1e-7)
    for filename in ['user_checks.csv','fused_checks.csv','ridge_h1_replan_checks.csv']:
        f=pd.read_csv(OUT/filename);check(filename+' physical',bool(f['pass'].all()),rows=len(f))
    ts=pd.read_csv(OUT/'terminal_target_sensitivity.csv');check('Etarget five-level coverage',set(np.round(ts.sensitivity_target,6))==set(ETARGETS))
    iv=pd.read_csv(OUT/'inventory_value_sensitivity.csv');check('inventory three-value coverage',set(iv.inventory_valuation)==set(e.inventory_values))
    # 配对周块 Bootstrap：主比较是新 V2 与冻结 Ridge+H1-replan。
    boot=[]
    for period,start,stop in PERIODS:
        p=daily[daily.date.between(start,stop)].pivot(index='date',columns='strategy',values='total_cost')
        for ref in ['ridge_h1_replan','baseline','user']:
            x=(p[ref]-p['fused']).to_numpy()
            for block in [7,14,28]:
                mean,lo,hi=circular_block_interval(x,block,5000,.95,SEED+block+len(x))
                boot.append(dict(period=period,reference=ref,candidate='fused',block=block,
                  daily_saving_mean=mean,daily_saving_low95=lo,daily_saving_high95=hi,
                  observed_total_saving=float(x.sum()),relative_saving=float(x.sum()/p[ref].sum())))
    write_csv(pd.DataFrame(boot),OUT/'bootstrap.csv')
    monthly=daily.assign(month=daily.date.str[:7]).groupby(['strategy','month'])[['planned_cost','emergency_cost','total_cost','emergency_kwh','spill_kwh']].sum().reset_index();write_csv(monthly,OUT/'monthly.csv')
    stress=[]
    for name in ['baseline','fused','ridge_h1_replan']:
        slots=pd.read_csv(OUT/f'{name}_slots.csv');q=slots.q.to_numpy().reshape(334,T);r=slots.reserve.to_numpy().reshape(334,T)
        for label,lm,pm,vol in [('实际',1,1,1),('负荷+10%',1.1,1,1),('光伏-30%',1,.7,1),('波动1.5倍',1,1,1.5),('联合冲击',1.1,.7,1.5)]:
            state=6000.;costs=[];em=0.
            for k,day in enumerate(range(31,365)):
                actual=(e.data['load'][day]*lm-e.data['pv'][day]*pm)*DT;net=e.bank.net[day]+vol*(actual-e.bank.net[day])
                ex=execute_day(q[k],net,state,np.r_[state,r[k]],1.);state=ex['soc'][-1]
                check_dispatch_result=check_dispatch(net,q[k],ex['charge'],ex['discharge'],ex['spill'],ex['soc'],ex['emergency'])
                if not check_dispatch_result['pass']:raise AssertionError(check_dispatch_result)
                costs.append(float(e.price@(q[k]+5*ex['emergency'])));em+=ex['emergency'].sum()
            stress.append(dict(strategy=name,scenario=label,total_cost=sum(costs),daily_cvar90=cv(costs,.90),daily_cvar95=cv(costs,.95),emergency_kwh=em))
    write_csv(pd.DataFrame(stress),OUT/'stress.csv')
    import scipy
    manifest=dict(fresh_run=True,complete=True,generated_utc=pd.Timestamp.now(tz='UTC').isoformat(),
      python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
      code_sha256=_file_sha(Path(__file__)),
      inputs={str(p.relative_to(ROOT)):_file_sha(p) for p in [ROOT/'C题/附件/附件1.xlsx',ROOT/'C题/附件/附件2.xlsx',ROOT/'results/problem2_forecast_ablation_predictions.csv.gz']},
      v1_result_files_used=False,ridge_prediction_artifact_used=True,official_result2_overwritten=False,
      interpretation='finite-sample LP optimality only; no claim of optimality under the unknown population distribution')
    write_json(OUT/'run_manifest.json',manifest)
    write_json(OUT/'verification.json',dict(tests=tests,all_pass=True,count=len(tests),retrospective_not_unseen=True))
    print(summary.to_string(index=False));print('verification',len(tests),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','user','fusion','ridge','sensitivity','evaluate'],default='all');a=p.parse_args()
    if a.stage=='all':
        # 用户明确要求废弃旧 V2 结果重算；只清理专属 V2 输出目录。
        for path in [OUT,FIG]:
            if path.exists():shutil.rmtree(path)
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    if a.stage=='all':data_audit(load_data())
    if a.stage in ['all','user']:run_user()
    if a.stage in ['all','fusion']:run_fusion()
    if a.stage in ['all','ridge']:run_ridge_h1_replan()
    if a.stage in ['all','sensitivity']:run_sensitivities()
    if a.stage in ['all','evaluate']:evaluate()

if __name__=='__main__':main()
