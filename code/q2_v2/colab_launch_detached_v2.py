"""Launch the full executed notebook as a detached process inside Colab."""
from pathlib import Path
import json, os, subprocess, sys, textwrap, time

PROJECT=Path('/content/CUMCM_2026_Last_Dance')
STATUS=Path('/content/q2_v2_remote_status.json')
LOG=Path('/content/q2_v2_remote_worker.log')
worker=r'''
from pathlib import Path
import json, os, subprocess, sys, time, traceback
P=Path('/content/CUMCM_2026_Last_Dance');S=Path('/content/q2_v2_remote_status.json');L=Path('/content/q2_v2_remote_worker.log')
def status(phase,**kw):S.write_text(json.dumps({'phase':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw},ensure_ascii=False,indent=2))
try:
    os.chdir(P);status('executing_notebook',pid=os.getpid())
    with L.open('a') as log:
        fit=subprocess.run([sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute',
          'code/problem2_fused_v2.ipynb','--output','problem2_fused_v2_output.ipynb','--output-dir','code',
          '--ExecutePreprocessor.timeout=10800'],stdout=log,stderr=log,text=True)
    if fit.returncode:raise RuntimeError(f'nbconvert return code {fit.returncode}')
    status('packing_outputs')
    with L.open('a') as log:subprocess.run([sys.executable,'code/q2_v2/colab_pack_v2.py'],check=True,stdout=log,stderr=log,text=True)
    status('completed',output_notebook=str(P/'code/problem2_fused_v2_output.ipynb'),archive='/content/q2_fused_v2_outputs.tar.gz')
except Exception as e:
    status('failed',error=repr(e),traceback=traceback.format_exc())
'''
if not PROJECT.exists():raise FileNotFoundError(PROJECT)
path=Path('/content/q2_v2_remote_worker.py');path.write_text(textwrap.dedent(worker),encoding='utf-8')
STATUS.write_text(json.dumps({'phase':'launching','time':time.strftime('%Y-%m-%d %H:%M:%S')}))
with LOG.open('a') as log:
    proc=subprocess.Popen([sys.executable,str(path)],stdout=log,stderr=log,start_new_session=True,close_fds=True)
print(json.dumps({'launched':True,'pid':proc.pid,'status':str(STATUS),'log':str(LOG)}))
