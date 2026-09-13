"""第四问的单变量消融、口径敏感性与参数/物理敏感性。可断点续跑（已完成的行会跳过）。

审阅意见要求的对照矩阵：
  A 发布时点开关      6/12/18 的全部 8 种组合 —— 每个决策机会的边际价值
  B 信息更新拆分      只更新光伏 / 只更新负荷 / 都不更新 / 电价日内校正开关
  C 结算口径敏感性    净额退款（主）/ 旧口径原费照付 / 逐次调整分别计费
  D 物理参数压力      容量 ±20%、功率 ±10%、效率 0.85/0.90/0.95（冻结策略）
  E 经验参数扰动      六个参数各 ±20%，先在 1 月窗口给 ΔC/Δθ，再对最敏感的做全年复核
  F 情景削减稳定性    多个固定随机种子 + 最小权重 + 有效情景数 1/Σw²

全部使用**冻结的选定策略**，只改标题里那一个要素；改多个要素的一律标为组合对照。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from q3 import Params, replace, Bank, DT
import q4_core as Q4C
import q4_policy as Q4P
import run_q4_experiment as E

R = E.R
OUT = R / 'q4_ablation_sensitivity.csv'


def load_done():
    if OUT.exists():
        df = pd.read_csv(OUT)
        return df, set(zip(df.block, df.case))
    return pd.DataFrame(), set()


def save(rows):
    old, _ = load_done()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.6f')


def summarize(df, block, case, changed, seconds):
    s = Q4P.summarize(df, method=case, problem='Q4-3', seconds=seconds)
    return dict(block=block, case=case, changed=changed, **{k: v for k, v in s.items()
                                                            if k not in ('method', 'problem', 'note')})


def main():
    g = E.prepare()
    sel = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    pol = E.policy_from(sel['Q4-3']['x'])
    p0, shape, edges0 = g['p'], E.G['shape'], E.G['edges']
    init = E.G['init_q4_3']
    _, done = load_done()
    rows = []

    def run(block, case, changed, *, mask=(6, 12, 18), pol_=None, p_=None, bank=None, pbank=None,
            edges=None):
        if (block, case) in done:
            print(f'  跳过（已完成）{block} / {case}', flush=True)
            return
        t0 = time.time()
        df, _, _, _ = Q4P.run_q4(g['L'], g['P'], g['V'], bank or g['bank'], g['rb'], pbank or g['pbank'],
                                 p_ or p0, pol_ or pol, shape, edges if edges is not None else edges0,
                                 g['vE_ref'], mask=mask, start=31, end=365, initial=init, mode='stoch', detail=True)
        r = summarize(df, block, case, changed, time.time() - t0)
        import hashlib
        case_id = hashlib.sha256((block + case).encode()).hexdigest()[:12]
        df.to_csv(R / f'ablation_daily_{case_id}.csv', index=False, encoding='utf-8-sig')
        rows.append(r); save([r]); rows.clear()
        print(f'  {block} | {case}: 现金 {r["total_cost"]:,.0f} 库存校正 {r["adjusted_cost"]:,.0f} '
              f'应急 {r["emergency_kwh"]:,.0f} 弃电 {r["spill_kwh"]:,.0f} ({time.time()-t0:.0f}s)', flush=True)

    # ---------------- A 发布时点开关 ----------------
    print('=== A 发布时点开关矩阵 ===', flush=True)
    for mask in [(), (6,), (12,), (18,), (6, 12), (6, 18), (12, 18), (6, 12, 18)]:
        name = '不更新' if not mask else '+'.join(f'{h}:00' for h in mask)
        run('A 发布时点', name, f'mask={mask}', mask=mask)

    # ---------------- B 信息更新拆分 ----------------
    print('=== B 信息更新拆分 ===', flush=True)
    bank_noload = Bank(g['L'], g['P'], g['F'], g['seed'], replace(p0, correction=0.0))
    bank_stale = Bank(g['L'], g['P'], g['F'], g['seed'], p0)
    bank_stale.stale_pv_issues([1, 2, 3])
    bank_both = Bank(g['L'], g['P'], g['F'], g['seed'], replace(p0, correction=0.0))
    bank_both.stale_pv_issues([1, 2, 3])
    hp_nc = dict(g['hp']); hp_nc['correction'] = 0.0
    pb_nc = Q4C.PriceBank(g['V'], g['tou'], hp_nc)
    run('B 信息更新', '只更新光伏（关负荷日内校正）', 'bank.correction=0', bank=bank_noload)
    run('B 信息更新', '只更新负荷（光伏用 0:00 版本）', 'stale_pv_issues=[1,2,3]', bank=bank_stale)
    run('B 信息更新', '两者都不更新', '组合对照：correction=0 且 stale PV', bank=bank_both)
    run('B 信息更新', '关闭电价日内偏差校正', 'price hp.correction=0', pbank=pb_nc)

    # ---------------- C 结算口径 ----------------
    print('=== C 结算口径敏感性 ===', flush=True)
    run('C 结算口径', '旧口径：原费照付＋罚金', "settlements='legacy_gross'",
        p_=replace(p0, settlements='legacy_gross'))
    run('C 结算口径', '逐次调整分别计费', "settlements='stepwise'",
        p_=replace(p0, settlements='stepwise'))

    # ---------------- D 物理参数压力 ----------------
    print('=== D 物理参数压力（冻结策略）===', flush=True)
    for cap in (9600.0, 14400.0):
        e = np.linspace(0.1 * cap, 0.9 * cap, len(edges0))
        run('D 物理参数', f'容量 {cap:.0f} kWh', f'capacity={cap}', p_=replace(p0, capacity=cap), edges=e)
    for pw in (4500.0, 5500.0):
        run('D 物理参数', f'功率 {pw:.0f} kW', f'power={pw}', p_=replace(p0, power=pw))
    for eta in (0.85, 0.95):
        run('D 物理参数', f'效率 {eta}', f'eta={eta}', p_=replace(p0, eta=eta))

    # ---------------- E 经验参数扰动（先 1 月窗口） ----------------
    print('=== E 经验参数扰动：1 月窗口 ΔC/Δθ ===', flush=True)
    base_fit, _ = E.fold_eval(pol, (6, 12, 18), 'q4_3')
    erows = []
    for key in Q4P.Q4_KEYS:
        theta = getattr(pol, key)
        for sgn in (-0.2, 0.2):
            if abs(theta) < 1e-12:
                erows.append(dict(param=key, base=theta, perturbed=theta, rel=sgn,
                                  fitness=np.nan, d_fitness=np.nan, elasticity=np.nan,
                                  note='基准值为 0，不计算相对弹性'))
                continue
            newv = theta * (1 + sgn)
            pp = Q4P.Q4Policy(**{**pol.__dict__, key: newv})
            f_, _ = E.fold_eval(pp, (6, 12, 18), 'q4_3')
            erows.append(dict(param=key, base=theta, perturbed=newv, rel=sgn, fitness=f_,
                              d_fitness=f_ - base_fit,
                              elasticity=(f_ - base_fit) / base_fit / sgn, note=''))
            print(f'  {key} {sgn:+.0%}: 适应度 {f_:,.1f}（{f_-base_fit:+,.1f}）', flush=True)
    pd.DataFrame(erows).to_csv(R / 'q4_param_sensitivity_train.csv', index=False,
                               encoding='utf-8-sig', float_format='%.6f')
    ed = pd.DataFrame([r for r in erows if np.isfinite(r.get('d_fitness', np.nan))])
    top = ed.reindex(ed.d_fitness.abs().sort_values(ascending=False).index).head(3)
    print('=== E 最敏感的三项做全年复核 ===', flush=True)
    for _, r in top.iterrows():
        pp = Q4P.Q4Policy(**{**pol.__dict__, r.param: r.perturbed})
        run('E 经验参数（全年）', f'{r.param} {r.rel:+.0%}', f'{r.param}={r.perturbed:.4f}', pol_=pp)

    # ---------------- F 情景削减稳定性 ----------------
    print('=== F 情景削减：多种子与有效情景数 ===', flush=True)
    frows = []
    for sd in (0, 1, 2, 3, 4):
        # 注意：必须把种子传进 run_q4（它决定 k-medoids 的初始化），
        # 只在权重统计里换种子而回测仍用默认种子，会得到"零方差"的假象。
        costs = []
        for s0, e0 in E.FOLDS:
            dfk, _, _, _ = Q4P.run_q4(g['L'], g['P'], g['V'], g['bank'], g['rb'], g['pbank'], p0,
                                      pol, shape, edges0, g['vE_ref'], mask=(6, 12, 18),
                                      start=s0, end=e0, initial=float(E.G['warm_q4_3'][s0]),
                                      mode='stoch', seed=sd)
            costs.append(float(dfk.inventory_adjusted_cost.sum()))
        f_, blocks = float(np.mean(costs)), costs
        ws = []
        for day in range(31, 365, 17):
            for k in range(4):
                _, _, w = Q4C.joint_scenarios(
                    g['bank'].mean[day, k], g['bank'].residual[day - p0.window:day, k, :],
                    g['pbank'].mean[day, k], g['pbank'].resid_window(day, k, p0.window),
                    pol.n_scen, seed=sd + 97 * day + k, temper=pol.temper,
                    shrink_net=pol.shrink_net, shrink_price=pol.shrink_price)
                ws.append(w)
        W = np.array([w for w in ws if len(w) == pol.n_scen])
        frows.append(dict(seed=sd, train_fitness=f_, worst_block=max(blocks),
                          min_weight=float(W.min()), mean_min_weight=float(W.min(axis=1).mean()),
                          effective_scenarios=float((1 / (W ** 2).sum(axis=1)).mean()),
                          samples=int(len(W))))
        print(f'  seed={sd} 适应度 {f_:,.1f} 有效情景数 {frows[-1]["effective_scenarios"]:.2f} '
              f'最小权重 {frows[-1]["min_weight"]:.4f}', flush=True)
    fd = pd.DataFrame(frows)
    fd.to_csv(R / 'q4_scenario_seed_stability.csv', index=False, encoding='utf-8-sig',
              float_format='%.6f')
    print(f'  跨种子适应度 均值 {fd.train_fitness.mean():,.1f} 标准差 {fd.train_fitness.std(ddof=1):,.1f}',
          flush=True)
    print('=== 全部完成 ===', flush=True)


if __name__ == '__main__':
    main()
