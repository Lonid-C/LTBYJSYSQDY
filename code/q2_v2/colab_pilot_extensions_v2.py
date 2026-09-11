"""Remote-only pilot launcher for the V2 planning extensions."""
from pathlib import Path
import os, subprocess, sys

PROJECT=Path('/content/CUMCM_2026_Last_Dance')
if not PROJECT.exists():raise FileNotFoundError(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable,'code/q2_v2/planning_extensions_v2.py','--stage','pilot'],check=True)
