"""Shared runtime settings and a fingerprinted description of the frozen run."""
from __future__ import annotations
from pathlib import Path
import sys, json, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'


INHERITED = ROOT / 'config/q3_inherited_parameters.json'
LEGACY_INHERITED = ROOT / 'results/parameters.json'


def runtime_settings():
    return json.loads((ROOT / 'runtime_settings.json').read_text(encoding='utf8'))


def inherited_parameters():
    """第三问继承来的预报/风险参数的**唯一读取入口**。

    本交付包不含第三问的 results/ 目录；优先读包内 config/q3_inherited_parameters.json，
    仅当在完整仓库中运行时才回退到 results/parameters.json，避免同一组参数散落两处。
    """
    if INHERITED.exists():
        return json.loads(INHERITED.read_text(encoding='utf8'))
    if LEGACY_INHERITED.exists():
        return json.loads(LEGACY_INHERITED.read_text(encoding='utf8'))
    raise FileNotFoundError(f'缺少继承参数配置：{INHERITED}')

def apply_runtime(p):
    from dataclasses import replace
    return replace(p, **runtime_settings()['params'])


def build(p, tou, hp, n_scen, vE_ref):
    return {
        'settlement': {
            'plan_price': '交付时段实际电价 V[d,t]（附件 4）',
            'adjustment_pricing': 'delivery',
            'adjustment_pricing_note': '本包明确采用交付时段实际电价。下单时刻报价属于另一种合同解释；不能因其可能出现折扣就排除。',
            'refund': p.settlements in ('net_refund', 'stepwise'),
            'settlement_basis': p.settlements,
            'settlement_basis_note': '日末按最终有效购电量相对 0:00 基准计划的净额一次结清；'
                                     'stepwise 变体按每次调整相对上一次有效计划分别计费，见口径敏感性。',
            'emergency_multiplier': float(p.emergency),
            'up_multiplier': float(p.up),
            'down_multiplier': float(p.down),
            'free_spill_allowed': True,
            'free_spill_note': '允许无成本弃置外购电（spill ≥ 0 无罚项）；不允许反向售电（q ≥ 0，无负购电）。',
            'reverse_sale_allowed': False,
        },
        'information': {
            'price_visibility': '决策时点之前已实现的电价可见；未来电价不可见，必须因果预报',
            'forecast_releases': [0, 6, 12, 18],
            'executed_block_hours': 6,
            'horizon_hours': 24,
            'causality_note': '每个发布时刻的决策只用该时刻之前已可获得的负荷、光伏、电价与预报',
        },
        'storage': {
            'capacity_kwh': float(p.capacity), 'power_kw': float(p.power),
            'eta_one_way': float(p.eta),
            'soc_lo_kwh': 0.1 * float(p.capacity), 'soc_hi_kwh': 0.9 * float(p.capacity),
            'initial_2025_01_01_kwh': 6000.0,
            'terminal_rule': '库存校正费用 = 现金费用 + vE_ref × (期初 − 期末)',
            'vE_ref': float(vE_ref),
            'vE_ref_source': '附件 1 分时电价最低值 / η —— 题目直接给定、决策前已知、'
                             '与附件 4 的任何实现值无关；全部方法共用同一常数。',
        },
        'evaluation': {
            'period': '2025-02-01/2025-12-31', 'days': 334, 'intervals': 334 * 144,
            'selection_window': '2025-01-08/2025-01-31（三个滚动验证块）',
            'selection_note': '参数在评价期之前选定并冻结；评价期为顺序前推回放，不宣称独立盲测',
        },
        'numerics': {'n_scen': int(n_scen), 'epsilon': float(p.epsilon),
                     'price_forecast_hp': hp, 'lp_solver': 'scipy.optimize.linprog(method="highs")'},
    }


def data_fingerprints():
    out = {}
    for name in ('附件1.xlsx', '附件2.xlsx', '附件3.xlsx', '附件4.xlsx'):
        f = ROOT / 'data' / name
        if f.exists():
            out[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def code_fingerprints():
    out = {}
    # *.html 是 q4_build_paper_html.py 的样式模板，输出依赖它们，必须一并纳入指纹
    for f in sorted(list((ROOT / 'code').glob('*.py')) + list((ROOT / 'code').glob('*.mjs'))
                    + list((ROOT / 'code').glob('*.html'))):
        out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    for n in ('hybrid_core.py', 'hybrid_stoch.py', 'hybrid_search.py', 'q3.py',
              'run_q4_experiment.py'):
        f = ROOT / 'code' / n
        if f.exists():
            out[n] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def write(p, tou, hp, n_scen, vE_ref, selected=None):
    cfg = build(p, tou, hp, n_scen, vE_ref)
    cfg['data_sha256_16'] = data_fingerprints()
    cfg['runtime_settings'] = runtime_settings()
    cfg['revision'] = '20260912-corrected-v1'
    cfg['risk_statistic'] = 'exact empirical CVaR95; paired 7/14/28-day MBB; resample length 334'
    cfg['provenance_note'] = 'Main net-refund policy parameters retained; settlement variant rerun; risk statistics recomputed. Historical development is not independent blind testing.'
    cfg['code_sha256_16'] = code_fingerprints()
    if selected is not None:
        cfg['selected_policy'] = selected
    blob = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode()
    cfg['variant_id'] = 'Q4-' + hashlib.sha256(blob).hexdigest()[:10]
    (R / 'model_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf8')
    return cfg


def variant_id():
    f = R / 'model_config.json'
    if not f.exists():
        return 'unset'
    return json.loads(f.read_text(encoding='utf8')).get('variant_id', 'unset')
