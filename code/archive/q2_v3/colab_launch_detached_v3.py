"""Launch the V3 notebook in a detached Colab process."""
from pathlib import Path
import json,os,subprocess,sys,textwrap,time
P=Path('/content/CUMCM_2026_Last_Dance');S=Path('/content/q2_v3_remote_status.json');L=Path('/content/q2_v3_remote_worker.log')
worker=r'''
from pathlib import Path
import json,os,subprocess,sys,time,traceback,tarfile
P=Path('/content/CUMCM_2026_Last_Dance');S=Path('/content/q2_v3_remote_status.json');L=Path('/content/q2_v3_remote_worker.log')
def status(phase,**kw):S.write_text(json.dumps({'phase':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),**kw},ensure_ascii=False,indent=2))
try:
    os.chdir(P);status('executing_notebook')
    with L.open('a') as log:
        fit=subprocess.run([sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute','code/problem2_v3_dual_terminal.ipynb',
          '--output','problem2_v3_dual_terminal_output.ipynb','--output-dir','code','--ExecutePreprocessor.timeout=43200'],stdout=log,stderr=log,text=True)
    if fit.returncode:raise RuntimeError(f'nbconvert return code {fit.returncode}')
    status('packing_outputs')
    targets=[P/'code/problem2_v3_dual_terminal_output.ipynb',P/'results/q2_v3_dual_terminal',P/'figures/q2_v3_dual_terminal',P/'reports/Q2_V3_DUAL_TERMINAL_RESULTS.md']
    missing=[str(x) for x in targets if not x.exists()]
    if missing:raise FileNotFoundError(missing)
    archive=Path('/content/q2_v3_outputs.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for x in targets:tf.add(x,arcname=str(x.relative_to(P)))
    status('completed',archive=str(archive),output_notebook=str(targets[0]))
except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc())
'''
path=Path('/content/q2_v3_remote_worker.py');path.write_text(textwrap.dedent(worker),encoding='utf-8')
S.write_text(json.dumps({'phase':'launching','time':time.strftime('%Y-%m-%d %H:%M:%S')}))
with L.open('a') as log:proc=subprocess.Popen([sys.executable,str(path)],stdout=log,stderr=log,start_new_session=True,close_fds=True)
print(json.dumps({'launched':True,'pid':proc.pid,'status':str(S),'log':str(L)}))
