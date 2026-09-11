"""本机后台监督器：执行、下载、验证、同步并最终停止 Colab VM。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import traceback

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
SESSION = os.environ.get("Q2_COLAB_SESSION", "cumcm-q2-hybrid")
CLI = shutil.which("colab") or "/Users/a1234/.local/bin/colab"
STATE = ROOT / "tmp/q2_hybrid_background_status.json"
LOG = ROOT / "tmp/q2_hybrid_background.log"
DOWNLOAD = ROOT / "tmp/q2_hybrid_outputs.tar.gz"
MANIFEST = ROOT / "tmp/q2_hybrid_outputs_manifest.json"
state = {"session": SESSION, "phase": "starting", "pid": os.getpid(), "vm_stopped": False}


def update(**kwargs):
    state.update(kwargs)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    os.replace(temp, STATE)


def call(*args, timeout=21660):
    with LOG.open("a") as log:
        log.write("\nCOMMAND " + repr(args) + "\n"); log.flush()
        result = subprocess.run([CLI, *args], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"colab command failed: {args}, code={result.returncode}")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_outputs():
    call("exec", "-s", SESSION, "-f", "code/colab_pack_q2_hybrid.py", "--timeout", "600", timeout=660)
    call("download", "-s", SESSION, "/content/q2_hybrid_outputs.tar.gz", str(DOWNLOAD), timeout=1200)
    call("download", "-s", SESSION, "/content/q2_hybrid_outputs_manifest.json", str(MANIFEST), timeout=180)
    manifest = json.loads(MANIFEST.read_text())
    if sha(DOWNLOAD) != manifest["archive_sha256"]:
        raise AssertionError("output archive hash mismatch")
    stage = ROOT / "tmp/q2_hybrid_sync"
    stage.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DOWNLOAD) as tf:
        for member in tf.getmembers():
            target = (stage / member.name).resolve()
            if not target.is_relative_to(stage.resolve()) or not member.isfile():
                raise ValueError(member.name)
        tf.extractall(stage, filter="data")
    for item in manifest["files"]:
        source = stage / item["path"]
        if sha(source) != item["sha256"]:
            raise AssertionError(item["path"])
        target = ROOT / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(manifest["files"])


def verify_notebook():
    output = ROOT / "code/problem2_hybrid_output.ipynb"
    if not output.exists():
        raise FileNotFoundError(output)
    book = nb.read(output, as_version=4); nb.validate(book)
    errors = [o for c in book.cells if c.cell_type == "code" for o in c.outputs
              if o.output_type == "error"]
    if errors:
        raise AssertionError(errors[0])
    code_cells = [c for c in book.cells if c.cell_type == "code"]
    if not all(c.outputs for c in code_cells[-2:]):
        raise AssertionError("final experiment cells have no outputs")
    backup = ROOT / "code/problem2_hybrid_source.ipynb"
    if not backup.exists():
        shutil.copy2(ROOT / "code/problem2_hybrid.ipynb", backup)
    shutil.copy2(output, ROOT / "code/problem2_hybrid.ipynb")
    return len(code_cells)


def main():
    outcome = "failed"
    try:
        update(phase="executing_notebook")
        call("exec", "-s", SESSION, "-f", "code/problem2_hybrid.ipynb", "--timeout", "21600")
        update(phase="syncing")
        file_count = sync_outputs()
        code_cells = verify_notebook()
        manifest = json.loads((ROOT / "results/q2_hybrid_v1/manifest.json").read_text())
        if not manifest["complete"] or not manifest["checks_all_pass"]:
            raise AssertionError(manifest)
        update(phase="verified", synced_files=file_count, verified_code_cells=code_cells,
               experiment_runtime_seconds=manifest["runtime_seconds"])
        outcome = "completed"
    except BaseException as exc:
        update(phase="failed", error=repr(exc), traceback=traceback.format_exc())
        try:
            synced = sync_outputs()
            update(failure_snapshot_files=synced)
        except BaseException as sync_error:
            update(failure_sync_error=repr(sync_error))
    finally:
        try:
            call("stop", "-s", SESSION, timeout=120)
            update(vm_stopped=True)
        except BaseException as stop_error:
            update(stop_error=repr(stop_error))
        update(phase=outcome)


if __name__ == "__main__":
    main()
