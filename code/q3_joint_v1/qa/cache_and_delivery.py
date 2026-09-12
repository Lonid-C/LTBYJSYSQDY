"""Post-run verification; no optimization or model fitting."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import openpyxl
r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(r/'code'))
import run_joint as J
import q3
import hybrid_policy as HP
key,_=J.fingerprint();folder=r/'cache'/key
assert folder.exists(), 'Cache does not match this source/environment'
m=json.loads((r/'results/run_manifest.json').read_text())
p=q3.Params(**m['parameters'])
q3.OUT=r/'results'
L,P,F,price,seed=q3.load()
original=q3.Bank
class NoTraining:
    def __new__(cls,*args,**kwargs):
        if args or kwargs:raise AssertionError('Unexpected Ridge refit')
        return object.__new__(original)
q3.Bank=NoTraining
try:bank,hit=J.get_bank(L,P,F,seed,p,folder)
finally:q3.Bank=original
assert hit
assert HP.next_lock(0,())==144
assert HP.next_lock(0,(12,))==72
assert HP.next_lock(6,(6,18))==72
assert HP.next_lock(18,(6,12,18))==36
iv=pd.read_csv(r/'results/joint_hybrid/intervals.csv')
w=openpyxl.load_workbook(r/'result3.xlsx',read_only=True,data_only=True)
rows=list(w['充放电量'].values)[1:]
expected=[]
for date,g in iv.groupby('date',sort=True):
    g=g.sort_values('t').reset_index(drop=True)
    for t in range(0,144,24):
        b=g.iloc[t:t+24]
        expected.append((date,b.charge.sum(),b.discharge.sum(),g.soc_start.iloc[0] if t==0 else g.soc_end.iloc[-1] if t==120 else None))
assert len(rows)==len(expected)
for actual,exp in zip(rows,expected):
    assert actual[0]==exp[0]
    np.testing.assert_allclose(actual[2:4],exp[1:3],atol=1e-7,rtol=0)
    if exp[3] is not None:assert abs(actual[5]-exp[3])<1e-7
for date,g in iv.groupby('date',sort=True):
    val=sum(row[2] for row in list(w['紧急购电量'].values)[1:] if row[0]==date)
    assert abs(val-g.emergency.sum())<1e-7
w.close()
checks=dict(cache_load_without_training=True,lock_schedule_checks=True,workbook_charge_discharge_soc=True,workbook_daily_emergency=True)
(r/'results/additional_checks.json').write_text(json.dumps(checks,indent=2))
print(checks)
# Also deliver the reproduced frozen baseline, regardless of comparison outcome.
import export_xlsx as EX
base=r/'results/inherited_hybrid'
EX.ROOT=base; EX.R=base; EX.main()
import shutil
shutil.copy2(base/'result3.xlsx',r/'result3_inherited.xlsx')
w=openpyxl.load_workbook(r/'result3_inherited.xlsx',read_only=True,data_only=True)
d=pd.read_csv(base/'daily_6+12+18.csv')
assert abs(sum(row[-1] for row in list(w['调整购电量'].values)[1:])-d.total_cost.sum())<1e-4
w.close()
print('Frozen baseline workbook exported and cost verified')
# Production delivery stays with the reproduced second-package main model.
shutil.copy2(r/'result3.xlsx',r/'result3_joint.xlsx')
shutil.copy2(r/'result3_inherited.xlsx',r/'result3.xlsx')
with (r/'reports/RESULTS_REPORT.md').open('a',encoding='utf8') as f:
    f.write('\n## 交付定位\n\n工程完善包以已复现第二版HYBRID为主模型：根目录result3.xlsx与result3_inherited.xlsx相同；联合情景候选另存result3_joint.xlsx。不依据评价期表现重新调参或自动换主模型。缓存无训练读取及工作簿检查见results/additional_checks.json。\n')
print('Main result3.xlsx = frozen second-package HYBRID; joint candidate = result3_joint.xlsx')
