from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
import tarfile

archive = Path('/content/q4_three_improvements_input.tar.gz')
target = Path('/content/q4_run')
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True)
with tarfile.open(archive, 'r:gz') as tf:
    tf.extractall(target, filter='data')

print('python', sys.version)
print('platform', platform.platform())
print('logical_cpus', os.cpu_count())
print('memory')
print(Path('/proc/meminfo').read_text().splitlines()[0])
try:
    import numpy, scipy, pandas, tqdm, openpyxl
    print('versions', numpy.__version__, scipy.__version__, pandas.__version__, tqdm.__version__, openpyxl.__version__)
except Exception as exc:
    print('dependency_error', repr(exc))
    raise
print('extracted', target)
