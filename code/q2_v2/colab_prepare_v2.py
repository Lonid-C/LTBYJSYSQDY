"""在新 Colab VM 上安全解压本次最小项目包并检查环境。"""
from pathlib import Path
import os
import platform
import subprocess
import sys
import tarfile


ARCHIVE=Path('/content/q2_fused_v2_source.tar.gz')
DEST=Path('/content/CUMCM_2026_Last_Dance')


def main():
    if not ARCHIVE.exists():raise FileNotFoundError(ARCHIVE)
    DEST.mkdir(parents=True,exist_ok=True);root=DEST.resolve()
    with tarfile.open(ARCHIVE,'r:gz') as tf:
        for member in tf.getmembers():
            target=(DEST/member.name).resolve()
            if root!=target and root not in target.parents:
                raise RuntimeError(f'拒绝越界路径: {member.name}')
            if member.issym() or member.islnk():
                raise RuntimeError(f'拒绝链接成员: {member.name}')
        tf.extractall(DEST,filter='data')
    # SciPy 1.16.3 的当前 Colab 组合会使旧 SAA 稀疏 LP 驱动返回
    # "HiGHS Status 0: Not Set"；1.15.3 已用常数场景和真实场景双重试运行验证。
    subprocess.run([sys.executable,'-m','pip','install','-q','openpyxl','scikit-learn',
                    'matplotlib','pandas','scipy==1.15.3','nbformat'],check=True)
    subprocess.run(['apt-get','update','-qq'],check=True)
    subprocess.run(['apt-get','install','-y','-qq','fonts-noto-cjk'],check=True)
    print('python',platform.python_version())
    print('cpu_count',os.cpu_count())
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                       text=True,capture_output=True)
    print('gpu',gpu.stdout.strip() if gpu.returncode==0 else 'none')
    print('project',DEST)


if __name__=='__main__':main()
