"""Extract the V3.2 source bundle and install the pinned Colab environment."""
from pathlib import Path
import os,platform,shutil,subprocess,sys,tarfile
ARCHIVE=Path('/content/q3_v2_source.tar.gz');DEST=Path('/content/CUMCM_2026_Last_Dance')
if not ARCHIVE.exists():raise FileNotFoundError(ARCHIVE)
DEST.mkdir(parents=True,exist_ok=True);root=DEST.resolve()
with tarfile.open(ARCHIVE,'r:gz') as tf:
    for m in tf.getmembers():
        target=(DEST/m.name).resolve()
        if root!=target and root not in target.parents:raise RuntimeError(m.name)
        if m.issym() or m.islnk():raise RuntimeError('links are not allowed')
    tf.extractall(DEST,filter='data')
subprocess.run([sys.executable,'-m','pip','install','-q','openpyxl','scikit-learn','matplotlib','pandas','scipy==1.15.3','nbformat','tabulate','jupyter'],check=True)
if not any(Path('/usr/share/fonts').rglob('*CJK*.ttc')):
    subprocess.run(['apt-get','update','-qq'],check=True,timeout=300)
    subprocess.run(['apt-get','install','-y','-qq','fonts-noto-cjk'],check=True,timeout=300)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True) \
    if shutil.which('nvidia-smi') else None
print('python',platform.python_version());print('cpu',os.cpu_count())
print('gpu',gpu.stdout.strip() if gpu is not None and gpu.returncode==0 else 'none')
