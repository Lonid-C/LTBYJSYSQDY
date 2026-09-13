"""Q4-2 主策略的风险分位标定：只有 1 个自由参数 α，在 1 月三个验证块上选，随后冻结。

为什么 Q4-2 的主策略是分位法而不是 HYBRID：见报告 §3.3。简言之，Q4-2 没有第二阶段调整通道，
两阶段情景随机规划最大的优势（正确给 recourse 定价）在这一口径下并不存在；实测它也确实赢不了
调好的分位法。α 是一维参数，在 24 天窗口上标定的过拟合风险远低于多维策略向量。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import q4_policy as Q4P
import run_q4_experiment as E

R = E.R
GRID = np.round(np.arange(0.40, 0.925, 0.025), 4)


def calibrate(label, mask, tag):
    rows = []
    for a in GRID:
        t0 = time.time()
        m, blocks = E.fold_eval(Q4P.Q4Policy(n_scen=E.N_SCEN, alpha=float(a)), mask, tag, mode='det')
        rows.append(dict(problem=label, alpha=float(a), train_fitness=m, worst_block=max(blocks),
                         block1=blocks[0], block2=blocks[1], block3=blocks[2],
                         seconds=time.time() - t0))
        print(f'  [{label}] alpha={a:.3f} 训练适应度={m:,.1f}', flush=True)
    df = pd.DataFrame(rows).sort_values('train_fitness').reset_index(drop=True)
    best = float(df.alpha.iloc[0])
    interior = bool(GRID[0] + 1e-9 < best < GRID[-1] - 1e-9)
    print(f'  [{label}] 选中 alpha={best:.3f}（网格内部={interior}），'
          f'训练适应度 {df.train_fitness.iloc[0]:,.1f}；'
          f'第三问沿用值 α=0.60 为 {float(df[df.alpha == 0.60].train_fitness.iloc[0]):,.1f}', flush=True)
    return df, best, interior


def main():
    E.prepare()
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    allrows = []
    for label, mask, tag in (('Q4-2', (), 'q4_2'), ('Q4-3', (6, 12, 18), 'q4_3')):
        df, best, interior = calibrate(label, mask, tag)
        allrows.append(df)
        sel[f'{label}_quantile'] = dict(strategy='quantile', alpha=best,
                                        fitness=float(df.train_fitness.iloc[0]),
                                        grid=[float(x) for x in GRID], optimum_interior=interior,
                                        note='分位法对照/主策略的 α 在 1 月三个验证块上重标定，'
                                             '与 HYBRID 使用同一窗口、同一适应度，保证可比')
    pd.concat(allrows).to_csv(R / 'q4_alpha_calibration.csv', index=False,
                              encoding='utf-8-sig', float_format='%.6f')
    (R / 'q4_selected_policy.json').write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
