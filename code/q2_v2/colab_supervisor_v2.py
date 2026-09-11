"""Detached local supervisor: execute, download, validate, sync, and always stop the Colab VM."""
from pathlib import Path
import json
import shutil
import subprocess
import tarfile
import time
import traceback

import nbformat

ROOT=Path(__file__).resolve().parents[2]
SESSION='cumcm-q2-v2-extensions'
STATUS=ROOT/'tmp/q2_v2_extensions_background_status.json'
ARCHIVE=ROOT/'tmp/q2_v2_extensions_outputs.tar.gz'
NOTEBOOK=ROOT/'code/problem2_fused_v2.ipynb'
OUTPUT_NOTEBOOK=ROOT/'code/problem2_fused_v2_output.ipynb'


def write_status(phase,**extra):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({'phase':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**extra},
                                 ensure_ascii=False,indent=2),encoding='utf-8')


def run(args,**kw):
    return subprocess.run(args,cwd=ROOT,check=True,text=True,**kw)


def safe_extract(archive):
    root=ROOT.resolve()
    with tarfile.open(archive,'r:gz') as tf:
        for m in tf.getmembers():
            target=(ROOT/m.name).resolve()
            if root!=target and root not in target.parents:raise RuntimeError(f'unsafe path {m.name}')
            if m.issym() or m.islnk():raise RuntimeError(f'link member {m.name}')
        for path in [ROOT/'results/q2_fused_v2',ROOT/'figures/q2_fused_v2',
                     ROOT/'results/q2_fused_v2_extensions',ROOT/'figures/q2_fused_v2_extensions']:
            if path.exists():shutil.rmtree(path)
        tf.extractall(ROOT,filter='data')


def validate_notebook(path):
    nb=nbformat.read(path,as_version=4);errors=[];missing=[]
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        # colab-cli 回传的 notebook 保留完整 outputs，但当前版本不回填
        # execution_count。因此以“有输出且无 error output”作为已执行证据。
        if cell.execution_count is None and not cell.get('outputs'):missing.append(i)
        for out in cell.get('outputs',[]):
            if out.get('output_type')=='error':errors.append((i,out.get('ename'),out.get('evalue')))
    if missing or errors:raise RuntimeError({'unexecuted_cells':missing,'error_outputs':errors})
    return {'cells':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells)}


def main():
    write_status('running_notebook',session=SESSION)
    ok=False
    try:
        run(['colab','exec','-s',SESSION,'-f',str(NOTEBOOK.relative_to(ROOT)),'--timeout','10800'])
        write_status('notebook_executed',session=SESSION)
        run(['colab','exec','-s',SESSION,'-f','code/q2_v2/colab_pack_v2.py','--timeout','900'])
        run(['colab','download','-s',SESSION,'/content/q2_fused_v2_outputs.tar.gz',str(ARCHIVE)])
        safe_extract(ARCHIVE)
        if not OUTPUT_NOTEBOOK.exists():raise FileNotFoundError(OUTPUT_NOTEBOOK)
        notebook_audit=validate_notebook(OUTPUT_NOTEBOOK)
        verification=json.loads((ROOT/'results/q2_fused_v2/verification.json').read_text())
        manifest=json.loads((ROOT/'results/q2_fused_v2/run_manifest.json').read_text())
        xverification=json.loads((ROOT/'results/q2_fused_v2_extensions/verification.json').read_text())
        xmanifest=json.loads((ROOT/'results/q2_fused_v2_extensions/run_manifest.json').read_text())
        if not verification.get('all_pass') or not manifest.get('fresh_run') or not manifest.get('complete'):
            raise RuntimeError('verification or fresh-run manifest failed')
        if not xverification.get('all_pass') or not xmanifest.get('fresh_run') or not xmanifest.get('complete'):
            raise RuntimeError('extension verification or fresh-run manifest failed')
        shutil.copy2(OUTPUT_NOTEBOOK,NOTEBOOK)
        shutil.copy2(OUTPUT_NOTEBOOK,ROOT/'code/problem2_hybrid.ipynb')
        # Commit only this run's explicit deliverables; preserve unrelated user changes.
        run(['git','add','code/problem2_fused_v2.ipynb','code/problem2_hybrid.ipynb',
             'results/q2_fused_v2','figures/q2_fused_v2','reports/Q2_FUSED_V2_RESULTS.md',
             'results/q2_fused_v2_extensions','figures/q2_fused_v2_extensions',
             'reports/Q2_FUSED_V2_EXTENSIONS_RESULTS.md'])
        committed=subprocess.run(['git','commit','-m','Add executed Q2 V2 planning extensions'],cwd=ROOT,text=True,capture_output=True)
        if committed.returncode not in (0,1):raise RuntimeError(committed.stderr)
        run(['git','push','origin','HEAD'])
        write_status('completed_and_synced',session=SESSION,notebook=notebook_audit,
                     verification_count=verification['count']+xverification['count'],git_commit=committed.stdout.strip())
        ok=True
    except Exception as exc:
        write_status('failed',session=SESSION,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        stopped=subprocess.run(['colab','stop','-s',SESSION],cwd=ROOT,text=True,capture_output=True)
        current=json.loads(STATUS.read_text()) if STATUS.exists() else {}
        current.update({'vm_stop_returncode':stopped.returncode,'vm_stop_stdout':stopped.stdout,
                        'vm_stop_stderr':stopped.stderr,'success':ok})
        STATUS.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
