"""Launch the V3.2 final notebook in a detached Colab worker."""
from pathlib import Path
import json,os,subprocess,sys,textwrap,time
S=Path('/content/q3_v2_status.json');L=Path('/content/q3_v2_worker.log')
worker=r'''
from pathlib import Path
import json,os,subprocess,sys,time,traceback,tarfile
P=Path('/content/CUMCM_2026_Last_Dance');S=Path('/content/q3_v2_status.json');L=Path('/content/q3_v2_worker.log')
def status(phase,**kw):S.write_text(json.dumps({'phase':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),**kw},ensure_ascii=False,indent=2))
try:
    os.chdir(P);status('executing_notebook')
    with L.open('a') as log:
        fit=subprocess.run([sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute','code/problem3_v2_aligned.ipynb',
          '--output','problem3_v2_aligned_output.ipynb','--output-dir','code','--ExecutePreprocessor.timeout=43200'],stdout=log,stderr=log,text=True)
    if fit.returncode:raise RuntimeError(f'nbconvert return code {fit.returncode}')
    status('packing_outputs')
    targets=[P/'code/problem3_v2_aligned_output.ipynb',P/'results/q3_v2_aligned',P/'result3.xlsx',
             P/'reports/Q3_V2_ALIGNED_RESULTS.md']
    missing=[str(x) for x in targets if not x.exists()]
    if missing:raise FileNotFoundError(missing)
    archive=Path('/content/q3_v2_outputs.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for x in targets:tf.add(x,arcname=str(x.relative_to(P)))
    status('completed',archive=str(archive),output_notebook=str(targets[0]))
except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc())
'''
path=Path('/content/q3_v2_worker.py');path.write_text(textwrap.dedent(worker),encoding='utf-8')
S.write_text(json.dumps({'phase':'launching','time':time.strftime('%Y-%m-%d %H:%M:%S')}))
with L.open('a') as log:proc=subprocess.Popen([sys.executable,str(path)],stdout=log,stderr=log,start_new_session=True,close_fds=True)
print(json.dumps({'launched':True,'pid':proc.pid,'status':str(S),'log':str(L)}))
