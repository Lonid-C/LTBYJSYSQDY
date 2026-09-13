import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('/content/q4_run')
experiment = root / '第四问_Q4三项专项改进_v1'
logs = experiment / 'logs'
logs.mkdir(parents=True, exist_ok=True)
master = logs / 'remote_master.log'
command = [sys.executable, str(experiment / 'code/run_remote_pipeline.py')]
stream = master.open('a', encoding='utf-8', buffering=1)
process = subprocess.Popen(
    command,
    cwd=root,
    env={**os.environ, 'Q4_WORKERS': '6', 'PYTHONUNBUFFERED': '1'},
    stdout=stream,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(experiment / 'remote_pid.json').write_text(
    json.dumps({'pid': process.pid, 'started_at': time.time(), 'command': command}, indent=2),
    encoding='utf-8',
)
print(json.dumps({'pid': process.pid, 'log': str(master), 'workers': 6}), flush=True)
