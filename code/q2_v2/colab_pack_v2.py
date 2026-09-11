"""在 Colab 完成后打包仅属于本次 V2 的输出。"""
from pathlib import Path
import tarfile

ROOT=Path('/content/CUMCM_2026_Last_Dance')
ARCHIVE=Path('/content/q2_fused_v2_outputs.tar.gz')


def main():
    targets=[ROOT/'results/q2_fused_v2',ROOT/'figures/q2_fused_v2',
             ROOT/'reports/Q2_FUSED_V2_RESULTS.md',
             ROOT/'results/q2_fused_v2_extensions',ROOT/'figures/q2_fused_v2_extensions',
             ROOT/'reports/Q2_FUSED_V2_EXTENSIONS_RESULTS.md']
    missing=[str(x) for x in targets if not x.exists()]
    if missing:raise FileNotFoundError(missing)
    with tarfile.open(ARCHIVE,'w:gz') as tf:
        for path in targets:tf.add(path,arcname=str(path.relative_to(ROOT)))
    print(ARCHIVE,ARCHIVE.stat().st_size)


if __name__=='__main__':main()
