"""Colab 远端打包：只收集融合实验输出，不接触正式结果表。"""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path("/content/CUMCM_2026_Last_Dance")
archive = Path("/content/q2_hybrid_outputs.tar.gz")
manifest_path = Path("/content/q2_hybrid_outputs_manifest.json")
folders = [root / "results/q2_hybrid_v1", root / "figures/q2_hybrid_v1"]
files = sorted(p for folder in folders if folder.exists() for p in folder.rglob("*") if p.is_file())
with tarfile.open(archive, "w:gz") as tf:
    for path in files:
        tf.add(path, arcname=str(path.relative_to(root)))
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
manifest = {"archive_sha256": sha(archive), "files": [
    {"path": str(p.relative_to(root)), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("Q2_HYBRID_PACKED", len(files), archive.stat().st_size)
