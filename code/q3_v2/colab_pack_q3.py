"""Pack the local inputs required by the Q3 aligned Colab run."""
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'tmp/q3_v2_source.tar.gz'
MEMBERS = [
    'C题/附件/附件1.xlsx',
    'C题/附件/附件2.xlsx',
    'C题/附件/附件3.xlsx',
    'C题/附件/附件5/result3.xlsx',
    'code/q2_planning.py',
    'code/q2_v2/dispatch.py',
    'code/q2_v2/utils.py',
    'code/q2_v32/__init__.py',
    'code/q2_v32/q2_final.py',
    'code/q3_v2/__init__.py',
    'code/q3_v2/q3_aligned.py',
    'code/problem3_v2_aligned.ipynb',
    'results/problem2_forecast_ablation_predictions.csv.gz',
    'results/q2_v32_final/value_banks.json',
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
