"""Run frozen baseline/joint comparison and produce the baseline-led delivery."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for script in ['code/run_joint.py','qa/cache_and_delivery.py']:
    subprocess.run([sys.executable,script],cwd=root,check=True)
