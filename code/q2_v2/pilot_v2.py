"""Colab 小规模试运行：在全年任务前检查数据、LP、Ridge对照和修正后统计规则。"""
from pathlib import Path
import json
import sys
import time

import numpy as np

PROJECT=Path('/content/CUMCM_2026_Last_Dance') if Path('/content/CUMCM_2026_Last_Dance').exists() else Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'code/q2_v2'))

from q2_fused import (ROOT, Engine, Config, DEFAULT, cv, near,
                      circular_block_interval, ETARGETS, T, DT)
from dispatch import optimize_day, execute_day, check_dispatch


def main():
    tests=[];e=Engine();started=time.perf_counter()
    def check(name,ok,**kw):
        if not ok:raise AssertionError((name,kw))
        tests.append({'test':name,'passed':True,**kw})

    # 14日CVaR95退化是数学性质，56日则包含2.8日尾部质量。
    a=np.arange(14.,dtype=float);b=np.arange(56.,dtype=float)
    check('cvar14_equals_worst',abs(cv(a,.95)-a.max())<1e-12,value=cv(a,.95))
    check('cvar56_uses_fractional_three_day_tail',a.max()>0 and b.max()>cv(b,.95)>b[-3],value=cv(b,.95))

    # near 不再读取固定费用百分比，只读配对日差的块区间。
    synthetic=[]
    for j,shift in enumerate([0.,.01,10.]):
        synthetic.append(dict(score=float(1000+42*shift),_daily=np.arange(42.)+shift,
          alpha=[.75,.8,.85][j],rho=.5,window=28,smooth=3,features='none',k=21,blend=0.))
    chosen=near(synthetic,'coarse')
    check('near_statistical_fields',all('near_ci_low90' in x for x in synthetic),chosen_alpha=chosen['alpha'])

    # 决定性LP和因果执行的小样本。
    lp_seconds=[]
    for target in [1200.,6000.,10800.]:
        for d in [151,243]:
            t=time.perf_counter();plan=optimize_day(e.risk(d,DEFAULT),e.price,6000.,target);lp_seconds.append(time.perf_counter()-t)
            ex=execute_day(plan.purchase,e.bank.actual_net[d],6000.,plan.soc,.5)
            chk=check_dispatch(e.bank.actual_net[d],plan.purchase,ex['charge'],ex['discharge'],ex['spill'],ex['soc'],ex['emergency'])
            check(f'deterministic_dispatch_{d}_{int(target)}',chk['pass'],**chk)

    # 冻结Ridge预测的21场景SAA接口。
    code_root=str(ROOT/'code')
    if code_root not in sys.path:sys.path.insert(0,code_root)
    from q2_planning import Study, weights_for
    s=Study(ROOT);fc=s.fc['Ridge'].copy();fc[:,:90]=s.legacy[:,:90];netfc=fc[0]-fc[1]
    saa_seconds=[]
    for d in [151,243]:
        hist=np.arange(d-21,d);scen=netfc[d]+s.net[hist]-netfc[hist]
        t=time.perf_counter();sol=s.solve(scen,weights_for(21,14),initial=6000.,terminal=6000.);saa_seconds.append(time.perf_counter()-t)
        ex=execute_day(sol['q'],s.net[d]*DT,6000.,sol['soc'],.5)
        chk=check_dispatch(s.net[d]*DT,sol['q'],ex['charge'],ex['discharge'],ex['spill'],ex['soc'],ex['emergency'])
        check(f'ridge_saa_{d}',chk['pass'] and sol['checks']['dual_gap_relative']<1e-7,
              dual_gap_relative=sol['checks']['dual_gap_relative'])

    # 接口级防泄漏检验。
    day=151;changed={**e.data,'load':e.data['load'].copy(),'pv':e.data['pv'].copy()}
    changed['load'][day:]*=2;changed['pv'][day:]=0;other=Engine(changed)
    check('forecast_future_invariant',np.array_equal(e.risk(day,Config(.85,.25,42,5,'level',21,.3)),
                                                     other.risk(day,Config(.85,.25,42,5,'level',21,.3))))

    med_det=float(np.median(lp_seconds));med_saa=float(np.median(saa_seconds))
    # 这是保守粗估；真实搜索存在缓存复用。
    estimate_seconds=100000*med_det+1700*med_saa+180
    result={'all_pass':True,'tests':tests,'deterministic_lp_median_seconds':med_det,
            'ridge_saa_lp_median_seconds':med_saa,'rough_full_seconds':estimate_seconds,
            'rough_full_minutes':estimate_seconds/60,'hardware_note':'HiGHS/Bootstrap are CPU-bound; GPU model affects host allocation more than LP arithmetic.',
            'elapsed_seconds':time.perf_counter()-started,'targets':ETARGETS}
    path=ROOT/'tmp/q2_fused_v2_pilot.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
