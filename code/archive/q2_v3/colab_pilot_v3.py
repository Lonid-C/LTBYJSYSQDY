"""Run the V3 smoke test in the prepared Colab project."""
from pathlib import Path
import os,subprocess,sys
P=Path('/content/CUMCM_2026_Last_Dance')
os.chdir(P)
subprocess.run([sys.executable,'code/archive/q2_v3/q2_dual_terminal.py','--stage','pilot'],check=True)
