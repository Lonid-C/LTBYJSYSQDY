"""Pack the local inputs required by the Q2 V3.2 final Colab run."""
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'tmp/q2_v32_source.tar.gz'
MEMBERS = [
    'C题/附件/附件1.xlsx',
    'C题/附件/附件2.xlsx',
    'C题/附件/附件5/result2.xlsx',
    'code/q2_planning.py',
    'code/q2_v2/dispatch.py',
    'code/q2_v2/utils.py',
    'code/q2_v32/__init__.py',
    'code/q2_v32/q2_final.py',
    'code/problem2_v32_final.ipynb',
    'results/problem2_forecast_ablation_predictions.csv.gz',
]


def main():
    missing = [m for m in MEMBERS if not (ROOT / m).exists()]
    if missing:
        raise FileNotFoundError(missing)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, 'w:gz') as tf:
        for name in MEMBERS:
            tf.add(ROOT / name, arcname=name)
    print(f'{ARCHIVE}  {ARCHIVE.stat().st_size/1048576:.2f} MB  {len(MEMBERS)} members')


if __name__ == '__main__':
    main()
