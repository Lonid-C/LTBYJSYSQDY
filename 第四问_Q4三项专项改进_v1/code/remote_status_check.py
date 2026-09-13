import json
import os
from pathlib import Path

experiment = Path('/content/q4_run/第四问_Q4三项专项改进_v1')
pid_info = json.loads((experiment / 'remote_pid.json').read_text(encoding='utf-8'))
pid = int(pid_info['pid'])
alive = Path(f'/proc/{pid}').exists()
status_file = experiment / 'remote_status.json'
status = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else None
log_file = experiment / 'logs/remote_master.log'
text = log_file.read_text(encoding='utf-8', errors='replace') if log_file.exists() else ''
print(json.dumps({'pid': pid, 'alive': alive, 'status': status, 'log_bytes': len(text.encode())}, ensure_ascii=False, indent=2))
print('--- LOG TAIL ---')
print('\n'.join(text.splitlines()[-35:]))
