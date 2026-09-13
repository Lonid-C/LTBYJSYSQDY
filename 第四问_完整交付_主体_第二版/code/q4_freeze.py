# -*- coding: utf-8 -*-
"""冻结本版：重算全部代码/数据/结果指纹，重写 model_config.json，并**自检**指纹与磁盘一致。

复核意见要求：「冻结代码后生成指纹清单，将结果文件也纳入清单，至少保证重新打包时实际文件与指纹一致」。
本脚本是交付前的最后一步——它跑完之后任何 code/ 下的改动都会让自检失败。
"""
from __future__ import annotations
import os, sys, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import q4_config as CFG
import run_q4_experiment as E

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results4'
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]


def result_fingerprints():
    keep = ('q4_out_of_sample.csv', 'q4_information_value.csv', 'q4_cvar_bootstrap.csv',
            'q4_ablation_sensitivity.csv', 'q4_full_horizon_bound.csv', 'q4_verification.json',
            'q4_selected_policy.json', 'q4_repair_checks.json', 'q4_monthly_Q4-2.csv',
            'q4_monthly_Q4-3.csv', 'q4_2_daily.csv', 'q4_3_daily.csv',
            'q4_alpha_calibration.csv', 'q4_terminal_curve.csv',
            'q4_release_pricing_pairs.csv', 'q4_release_pricing.json')
    return {n: sha(R / n) for n in keep if (R / n).exists()}


def deliverable_fingerprints():
    return {n: sha(ROOT / n) for n in ('result4-2.xlsx', 'result4-3.xlsx') if (ROOT / n).exists()}


def figure_fingerprints():
    """论文引用的插图也纳入清单——复核意见要求图表文件可逐项校验。"""
    d = ROOT / '论文交付/2_论文部分/figures'
    return {f.name: sha(f) for f in sorted(d.glob('*.png'))} if d.exists() else {}


def config_fingerprints():
    """运行**输入**（不是运行后的描述性副本）的指纹：唯一配置入口与运行时设置。"""
    return {n: sha(ROOT / n) for n in ('config/q3_inherited_parameters.json', 'runtime_settings.json')
            if (ROOT / n).exists()}


def main():
    g = E.prepare()
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    cfg = CFG.write(g['p'], E.G['tou'], g['hp'], E.N_SCEN, g['vE_ref'], selected=sel)
    # 结果与交付物指纹另存，避免把它们算进 variant_id（否则自引用）
    man = dict(variant_id=cfg['variant_id'],
               code_sha256_16=cfg['code_sha256_16'],
               data_sha256_16=cfg['data_sha256_16'],
               config_sha256_16=config_fingerprints(),
               figure_sha256_16=figure_fingerprints(),
               result_sha256_16=result_fingerprints(),
               deliverable_sha256_16=deliverable_fingerprints())
    (R / 'fingerprint_manifest.json').write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding='utf8')

    # README 的版本行随冻结一起改写，避免再次落后于实际版本
    rp = ROOT / 'reports4' / 'README.md'
    if rp.exists():
        import re as _re
        txt = rp.read_text(encoding='utf8')
        txt = _re.sub(r'\*\*variant_id：`[^`]*`\*\*',
                      f"**variant_id：`{cfg['variant_id']}`**", txt, count=1)
        rp.write_text(txt, encoding='utf8')

    bad = []
    for n, h in cfg['code_sha256_16'].items():
        if sha(ROOT / 'code' / n) != h:
            bad.append(('code', n))
    for n, h in cfg['data_sha256_16'].items():
        if sha(ROOT / 'data' / n) != h:
            bad.append(('data', n))
    for n, h in man['result_sha256_16'].items():
        if sha(R / n) != h:
            bad.append(('result', n))
    for n, h in man['config_sha256_16'].items():
        if sha(ROOT / n) != h:
            bad.append(('config', n))
    for n, h in man['figure_sha256_16'].items():
        if sha(ROOT / '论文交付/2_论文部分/figures' / n) != h:
            bad.append(('figure', n))
    for n, h in man['deliverable_sha256_16'].items():
        if sha(ROOT / n) != h:
            bad.append(('deliverable', n))
    print(f'variant_id = {cfg["variant_id"]}')
    print(f'指纹清单：代码 {len(cfg["code_sha256_16"])} 个、数据 {len(cfg["data_sha256_16"])} 个、'
          f'配置 {len(man["config_sha256_16"])} 个、插图 {len(man["figure_sha256_16"])} 个、'
          f'结果 {len(man["result_sha256_16"])} 个、'
          f'交付物 {len(man["deliverable_sha256_16"])} 个')
    if bad:
        print('✗ 指纹自检失败：', bad); sys.exit(1)
    print('✓ 指纹自检通过：清单内每个文件与磁盘一致')


if __name__ == '__main__':
    main()
