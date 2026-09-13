"""Pack remote Q4 outputs, checkpoints, logs, and status for local synchronization."""
from pathlib import Path
import tarfile


ROOT = Path("/content/q4_run/第四问_Q4三项专项改进_v1")
ARCHIVE = Path("/content/q4_three_improvements_artifacts.tar.gz")

with tarfile.open(ARCHIVE, "w:gz") as bundle:
    for name in ("results", "checkpoints", "logs", "remote_status.json", "remote_pid.json"):
        path = ROOT / name
        if path.exists():
            bundle.add(path, arcname=name)

print(f"{ARCHIVE}\t{ARCHIVE.stat().st_size} bytes")
