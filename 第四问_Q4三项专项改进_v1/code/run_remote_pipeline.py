"""Remote entry point with durable logs and status markers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "remote_status.json"


def write_status(stage, **extra):
    STATUS.write_text(
        json.dumps({"stage": stage, "time": time.time(), **extra}, indent=2),
        encoding="utf-8",
    )


def run(command, log_name):
    log_path = LOGS / log_name
    write_status("running", command=command, log=str(log_path))
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    workers = os.environ.get("Q4_WORKERS", "6")
    write_status("starting", workers=int(workers))
    try:
        run(
            [sys.executable, "第四问_Q4三项专项改进_v1/code/run_q4_three_improvements.py",
             "--workers", workers, "--resamples", "4000"],
            "numerical_pipeline.log",
        )
        run(
            [sys.executable, "第四问_Q4三项专项改进_v1/code/run_q4_causality_expanded.py",
             "--workers", workers, "--days", "48"],
            "causality_pipeline.log",
        )
    except Exception as exc:
        write_status("failed", error=repr(exc))
        raise
    write_status("complete")


if __name__ == "__main__":
    main()
