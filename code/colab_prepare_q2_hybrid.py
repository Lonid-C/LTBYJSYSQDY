"""Colab 远端准备：安全解包本轮最小输入归档。"""
from pathlib import Path
import tarfile

archive = Path("/content/q2_hybrid_input.tar.gz")
destination = Path("/content/CUMCM_2026_Last_Dance")
destination.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as tf:
    for member in tf.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError(f"unsafe archive member: {member.name}")
    tf.extractall(destination, filter="data")
root = destination
required = [root / "code/q2_planning.py", root / "code/q2_hybrid.py",
            root / "C题/附件/附件1.xlsx", root / "C题/附件/附件2.xlsx",
            root / "results/problem2_forecast_ablation_predictions.csv.gz"]
assert all(p.exists() for p in required), [str(p) for p in required if not p.exists()]
print("Q2_HYBRID_PREPARED", root)
