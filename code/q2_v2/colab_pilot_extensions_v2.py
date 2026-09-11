"""Remote-only pilot launcher for the V2 planning extensions."""
from pathlib import Path
import os, shutil, subprocess, sys

PROJECT=Path('/content/CUMCM_2026_Last_Dance')
if not PROJECT.exists():raise FileNotFoundError(PROJECT)
os.chdir(PROJECT)
base=PROJECT/'results/q2_fused_v2';base.mkdir(parents=True,exist_ok=True)
for name in ('fused_selection.json','fused_daily.csv'):
    staged=Path('/content')/name
    if staged.exists() and not (base/name).exists():shutil.copy2(staged,base/name)
fit=subprocess.run([sys.executable,'code/q2_v2/planning_extensions_v2.py','--stage','pilot'],
                   text=True,capture_output=True)
print(fit.stdout);print(fit.stderr)
fit.check_returncode()
