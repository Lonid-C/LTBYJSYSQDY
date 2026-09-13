# -*- coding: utf-8 -*-
"""下单（发布）时刻计价口径的**有效配对**重算。

复核意见 §7：不能把某一个 12:00 发布价的 1.5 倍与全年所有交付时段电价比较，
再据此声称"低于 57.6% 的交付价格"。正确的比较对象是每个发布时刻 τ 与该时刻
**实际允许调整**的未来交付时段 t：

    Pr( 1.5 · p_release(τ) < p_delivery(t) | t ∈ T(τ) )

配对规则（与 q4_policy.run_q4 的调整窗口完全一致）：
  · 发布时刻 τ ∈ {6:00, 12:00, 18:00}（0:00 是下达原始计划，不是调整，故不计入）；
  · 发布时刻 h 对应的时段下标 t0 = 6h；该时刻可调整的交付时段为同一日的 t ∈ [t0, 144)，
    即 q4_policy 中 `base = final[t:]`、`tail = 144 - t` 所覆盖的区间；
  · 发布价 p_release(τ) = V[d, t0]，即发布时刻当段已公布的实际电价；
  · 日期 d 取评价期 2025-02-01—2025-12-31（下标 31..364），与全文口径一致。

输出 results4/q4_release_pricing_pairs.csv（逐发布时刻与合计），
供论文与报告直接引用，不得手工硬编码。
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hashlib
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
START, END = 31, 365          # 评价期 2025-02-01 .. 2025-12-31
UP = 1.5                      # 增购溢价倍数 mu^+
ISSUES = (0, 6, 12, 18)       # 与 q3.ISSUES / q4_policy 的发布时点定义一致


def load_prices():
    """读取附件 4 的 365x144 实际电价，并核对数据指纹与冻结清单一致。"""
    f = ROOT / 'data/附件4.xlsx'
    ws = openpyxl.load_workbook(f, read_only=True, data_only=True).worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    V = np.array([r[1:145] for r in rows[1:]], float)
    assert V.shape == (365, 144) and np.isfinite(V).all() and (V > 0).all(), V.shape
    man = R / 'fingerprint_manifest.json'
    if man.exists():
        want = json.loads(man.read_text(encoding='utf8'))['data_sha256_16'].get('附件4.xlsx')
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        assert want in (None, got), (want, got)
    return V


def pairs(V, start=START, end=END, up=UP, issues=tuple(h for h in ISSUES if h > 0)):
    """逐"发布时点—可调整交付时段"配对统计。"""
    rows, hits, total = [], 0, 0
    for h in issues:
        t0 = h * 6
        rel = V[start:end, t0]                 # 发布时刻已公布的实际电价
        dlv = V[start:end, t0:144]             # 该时刻可调整的交付时段
        flag = (up * rel[:, None] < dlv)
        n, k = int(flag.size), int(flag.sum())
        ratio = (up * rel[:, None]) / dlv
        rows.append(dict(release_clock=f'{h:02d}:00', release_slot=t0,
                         adjustable_slots_per_day=144 - t0, days=end - start,
                         pairs=n, hits=k, share_percent=100.0 * k / n,
                         release_price_mean=float(rel.mean()),
                         premium_over_delivery_ratio_mean=float(ratio.mean()),
                         premium_over_delivery_ratio_median=float(np.median(ratio))))
        hits += k; total += n
    rows.append(dict(release_clock='合计', release_slot=-1, adjustable_slots_per_day=-1,
                     days=end - start, pairs=total, hits=hits,
                     share_percent=100.0 * hits / total,
                     release_price_mean=float('nan'),
                     premium_over_delivery_ratio_mean=float('nan'),
                     premium_over_delivery_ratio_median=float('nan')))
    return pd.DataFrame(rows)


def main():
    V = load_prices()
    df = pairs(V)
    df['rule'] = ('发布时刻 τ∈{6:00,12:00,18:00}；可调整交付时段 t∈[6h,144)（同日）；'
                  '发布价=V[d,6h]；判据 1.5·p_release < p_delivery；评价期 334 天')
    df.to_csv(R / 'q4_release_pricing_pairs.csv', index=False,
              encoding='utf-8-sig', float_format='%.10f')
    tot = df.iloc[-1]
    # 论文中被替换掉的旧口径：0.6648 与全年所有时段电价比较，仅供对照留痕，不再引用。
    legacy = 100.0 * float((0.6648 < V).mean())
    out = dict(total_pairs=int(tot.pairs), total_hits=int(tot.hits),
               share_percent=float(tot.share_percent),
               per_release={r.release_clock: dict(pairs=int(r.pairs), hits=int(r.hits),
                                                  share_percent=float(r.share_percent))
                            for r in df.itertuples() if r.release_clock != '合计'},
               legacy_unpaired_share_percent=legacy,
               note='旧稿 57.6% 既非有效配对结果，也复现不出未配对比较值，已作废')
    (R / 'q4_release_pricing.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
    print(df.to_string(index=False), flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
