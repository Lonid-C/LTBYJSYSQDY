"""两版对比运行器：队友原版 vs 融合版，同一校准-验证-评价协议。

协议（两版完全对称）：
  select_ridge → 1 月 warmup → 风险分位校准（含边界自动延展）→ 滚动/扩展窗口
  时序验证选参 → warmup_selected → 全年消融（none/6/12/18 组合）→ 结构变体
  （8 个）→ 旧光伏信息对照 → 块 bootstrap → 因果性检查。

跳过 stability_map 与 ~50 项单因素敏感性（约 120 个全年当量的纯分析性回放，
不影响校准路径与主数字；原版完整套件证据已在队友仓库 results/ 中）。
"""
import json, sys, time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import q3

ROOT = q3.ROOT
STATUS = ROOT.parent / 'q3_two_status.json'


def write_status(d):
    Path(STATUS).write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def core_pipeline(bank_cls, run_fn, out_dir, variant_label, extra_note=''):
    out_dir.mkdir(parents=True, exist_ok=True)
    q3.OUT = out_dir
    clock = time.time()
    L, P, F, price, seed = q3.load()
    p = q3.Params()

    slope, weekday, _ = q3.select_ridge(L, seed, p)
    p = q3.replace(p, slope=slope, weekday=weekday)
    bank = bank_cls(L, P, F, seed, p)

    warm, _, _ = run_fn(L, P, price, bank, p, mask=(), start=0, end=31)
    q3.csv('warmup_january.csv', warm)

    records, best_sw, calib_log, grid0, gridh = q3.calibrate(L, P, price, bank, p, warm.soc_start.iloc[14])
    q3.csv('calibration.csv', records)
    q3.dump('calibration_grid_extension.json', calib_log)

    _, best_rv = q3.rolling_validation(L, P, price, bank, p, warm, grid0, gridh)
    p_sw = q3.replace(p, alpha0=best_sw['alpha0'], alpha_update=best_sw['alpha_update'])
    p_rv = q3.replace(p, alpha0=float(best_rv.alpha0), alpha_update=float(best_rv.alpha_update))
    p = p_rv
    warm_sel, _, _ = run_fn(L, P, price, bank, p, mask=(), start=0, end=31)
    q3.csv('warmup_january_selected.csv', warm_sel)
    initial = warm_sel.soc_end.iloc[-1]
    p_theory = q3.replace(p, alpha0=q3.THEORY_ALPHA0, alpha_update=q3.THEORY_ALPHA_UPDATE)
    print(f'[{variant_label}] CALIBRATED alpha0={p.alpha0} alpha_update={p.alpha_update} '
          f'initial={initial:.2f} elapsed={time.time()-clock:.0f}s', flush=True)

    runs, comparisons = {}, []
    for bits in product([False, True], repeat=3):
        mask = tuple(h for h, b in zip((6, 12, 18), bits) if b)
        label = 'none' if not mask else '+'.join(map(str, mask))
        df, iv, dec = run_fn(L, P, price, bank, p, mask=mask, initial=initial, detail=all(bits))
        runs[label] = df
        comparisons.append(q3.summary(df, strategy=label))
        q3.csv(f'daily_{label}.csv', df)
        if all(bits):
            q3.csv('intervals.csv', iv)
            q3.csv('decisions.csv', dec)
        print(f'[{variant_label}] ABLATION', label, round(df.total_cost.sum()), flush=True)
    q3.csv('strategy_comparison.csv', comparisons)
    full = runs['6+12+18']

    variants = [('main_net_refund', p, True),
                ('legacy_gross_settlement', q3.replace(p, settlements='legacy_gross'), False),
                ('stepwise_settlement', q3.replace(p, settlements='stepwise'), False),
                ('fixed_target_soc', q3.replace(p, terminal='fixed_target'), False),
                ('day_horizon_only', q3.replace(p, horizon='day'), True),
                ('theory_quantiles', p_theory, False),
                ('single_window_params', p_sw, False),
                ('no_intraday_bias_correction', q3.replace(p, correction=0.), True)]
    vrows = []
    for label, pp, rebuild in variants:
        bb = bank_cls(L, P, F, seed, pp) if rebuild else bank
        df, _, _ = run_fn(L, P, price, bb, pp, initial=initial)
        q3.csv(f'variant_daily_{label}.csv', df)
        vrows.append(q3.summary(df, variant=label, alpha0=pp.alpha0, alpha_update=pp.alpha_update,
                                settlement=pp.settlements, terminal=pp.terminal))
        print(f'[{variant_label}] VARIANT', label, round(df.total_cost.sum()), flush=True)
    q3.csv('variant_comparison.csv', vrows)
    legacy = next(r for r in vrows if r['variant'] == 'legacy_gross_settlement')
    stepwise_row = next(r for r in vrows if r['variant'] == 'stepwise_settlement')

    stale = bank_cls(L, P, F, seed, p)
    stale.stale_pv_issues(range(4))
    stale_df, _, _ = run_fn(L, P, price, stale, p, initial=initial)
    q3.csv('daily_stale_pv.csv', stale_df)
    q3.csv('pure_pv_update_value.csv', [dict(stale_pv_cost=stale_df.total_cost.sum(),
                                             fresh_pv_cost=full.total_cost.sum(),
                                             cash_saving=stale_df.total_cost.sum() - full.total_cost.sum(),
                                             inventory_adjusted_saving=stale_df.inventory_adjusted_cost.sum() - full.inventory_adjusted_cost.sum())])

    rng = np.random.default_rng(20260911)
    bootstrap = []
    base = runs['none'].inventory_adjusted_cost.to_numpy()
    for label, df in runs.items():
        if label == 'none':
            continue
        delta = base - df.inventory_adjusted_cost.to_numpy()
        n = len(delta)
        for block in (7, 14, 28):
            starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
            ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
            samples = delta[ix].sum(axis=1)
            bootstrap.append(dict(strategy=label, block_days=block, annual_saving=delta.sum(),
                                  ci_low=np.quantile(samples, .025), ci_high=np.quantile(samples, .975),
                                  fraction_positive=float((samples > 0).mean()),
                                  daily_win_rate=float((delta > 0).mean())))
    q3.csv('bootstrap.csv', bootstrap)

    day, k, t = 78, 2, 72
    LL, PP, FF = L.copy(), P.copy(), F.copy()
    LL[day, t:] += 12345; LL[day + 1:] += 23456
    PP[day, t:] += 3333; PP[day + 1:] += 4444
    FF[day, k + 1:] += 5555; FF[day + 1:] += 6666
    altered = bank_cls(LL, PP, FF, seed, p)
    causal_diff = float(np.max(abs(bank.risk(day, k, p.alpha_update) - altered.risk(day, k, p.alpha_update))))
    assert causal_diff == 0
    assert legacy['down_kwh'] < 1e-6, '旧解释下减购应被支配为零'

    check = {'variant': variant_label, 'extra_note': extra_note,
             'causality_future_perturbation_max_diff': causal_diff,
             'balance_max_error': float(full.balance_error.max()),
             'soc_dynamics_max_error': float(full.dynamics_error.max()),
             'soc_min': float(full.soc_min.min()), 'soc_max': float(full.soc_max.max()),
             'cross_day_soc_max_gap': float(abs(full.soc_start.to_numpy()[1:] - full.soc_end.to_numpy()[:-1]).max()),
             'main_down_kwh': float(full.down_kwh.sum()), 'main_up_kwh': float(full.up_kwh.sum()),
             'legacy_down_kwh': float(legacy['down_kwh']),
             'stepwise_cost_minus_net': float(stepwise_row['total_cost'] - full.total_cost.sum()),
             'dates': len(full),
             'alpha0_interior': bool(0 < grid0.index(p.alpha0) < len(grid0) - 1),
             'alpha_update_interior': bool(0 < gridh.index(p.alpha_update) < len(gridh) - 1),
             'protocol': 'core (stability_map/sensitivity skipped; 不影响校准路径与主数字)',
             'all_assertions_passed': True,
             'elapsed_seconds': time.time() - clock}
    q3.dump('checks.json', check)
    print(f'[{variant_label}] COMPLETE', round(full.total_cost.sum()), f'{time.time()-clock:.0f}s', flush=True)
    return dict(label=variant_label, alpha0=p.alpha0, alpha_update=p.alpha_update,
                initial_feb1=initial, total_cost=check and float(full.total_cost.sum()),
                check=check)


def main():
    fusion_only = 'fusion-only' in sys.argv
    if fusion_only:
        write_status({'phase': 'fusion-running', 'note': 'fusion-only（原版结果沿用已有运行）'})
    else:
        write_status({'phase': 'original-running', 'started': str(pd.Timestamp.now())})
        r1 = core_pipeline(q3.Bank, q3.run, ROOT / 'results_rerun_original',
                           'original', '队友原版：自有岭预报 + 线性 v_E=p_min/eta')
        write_status({'phase': 'fusion-running', 'original': r1})

    import q3_fusion as Fm          # 原版阶段结束后才导入/打补丁
    q3.run = Fm.run_f               # calibrate/rolling_validation 内部改走融合 run
    r2 = core_pipeline(Fm.FusionBank, Fm.run_f, ROOT / 'results_fusion',
                       'fusion', '融合版：Q2 冻结 Ridge 负荷预测 + Q2 17 点对偶终端切平面')
    write_status({'phase': 'completed', **({} if fusion_only else {'original': r1}), 'fusion': r2,
                  'finished': str(pd.Timestamp.now())})
    print('TWO-VERSIONS-COMPLETE', flush=True)


if __name__ == '__main__':
    main()
