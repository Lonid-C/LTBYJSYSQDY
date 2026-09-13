import os
import subprocess
import sys

root = '/content/q4_run'
env = {**os.environ, 'Q4_WORKERS': '6', 'PYTHONUNBUFFERED': '1'}
command = [
    sys.executable,
    f'{root}/第四问_Q4三项专项改进_v1/code/run_q4_three_improvements.py',
    '--smoke-only',
]
print('COMMAND', ' '.join(command), flush=True)
subprocess.run(command, cwd=root, env=env, check=True)
