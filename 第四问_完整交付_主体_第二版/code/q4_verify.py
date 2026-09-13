"""第四问独立复核：账本重算、物理约束、严格因果性、工作表与账本一致性。"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import openpyxl

import q3
from q3 import Params, replace, Bank, DATES, DT
import hybrid_core as H
import q4_core as Q4C
import q4_policy as Q4P
import q4_stoch as Q4S
import q4_config as CFG

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'


def main():
    L, P, F, tou, seed, V = Q4C.load4()
    pars = CFG.inherited_parameters()   # 唯一配置入口，见 q4_config.inherited_parameters
    sel = pars['selected']
    p = replace(Params(), **{k: sel[k] for k in ('alpha0', 'alpha_update', 'window', 'decay_days',
                                                 'bias_window', 'bias_decay', 'slope', 'weekday', 'epsilon')})
    p = CFG.apply_runtime(p)
    hp = json.loads((R / 'price_hp.json').read_text(encoding='utf8'))
    pbank = Q4C.PriceBank(V, tou, hp)
    bank = Bank(L, P, F, seed, p)
    z = np.load(R / 'q4_terminal_shape.npz')
    shape, edges = z['shape'], z['edges']
    polj = json.loads((R / 'q4_selected_policy.json').read_text(encoding='utf8'))
    out = {}
    idx = {str(x.date()): i for i, x in enumerate(DATES)}
    coef = q3.settle_coef(p)

    # ---- 1 账本独立重算 + 物理约束 ----
    for tag, sheetfile, has_adj in (('q4_2', ROOT / 'result4-2.xlsx', False),
                                    ('q4_3', ROOT / 'result4-3.xlsx', True)):
        iv = pd.read_csv(R / f'{tag}_intervals.csv')
        dl = pd.read_csv(R / f'{tag}_daily.csv')
        err = dict(cost=0.0, balance=0.0, soc=0.0, power=0.0, price=0.0)
        prev = float(dl.soc_start.iloc[0])
        for date, g in iv.groupby('date', sort=True):
            g = g.sort_values('t'); di = idx[date]
            plan = g.plan.to_numpy(); adj = g.adjusted.to_numpy()
            c = g.charge.to_numpy(); dis = g.discharge.to_numpy()
            zz = g.emergency.to_numpy(); sp = g.spill.to_numpy(); soc = g.soc_end.to_numpy()
            pr = V[di]
            err['price'] = max(err['price'], float(np.abs(g.price.to_numpy() - pr).max()))
            d = adj - plan
            cost = float(pr @ plan) + float(pr @ (p.up * np.maximum(d, 0))) \
                + float(pr @ (coef * np.maximum(-d, 0))) + float(p.emergency * (pr @ zz))
            row = dl[dl.date == date].iloc[0]
            err['cost'] = max(err['cost'], abs(cost - row.total_cost))
            net = (L[di] - P[di]) * DT
            err['balance'] = max(err['balance'], float(np.abs(adj + zz + dis - c - sp - net).max()))
            st = np.r_[prev, soc]
            err['soc'] = max(err['soc'], float(np.abs(np.diff(st) - p.eta * c + dis / p.eta).max()))
            err['power'] = max(err['power'], float(max(c.max(), dis.max()) - p.power * DT))
            assert soc.min() >= .1 * p.capacity - 1e-4 and soc.max() <= .9 * p.capacity + 1e-4
            if not has_adj:
                err['cost'] = max(err['cost'], float(np.abs(d).max()))      # Q4-2 无调整通道
            prev = soc[-1]
        out[tag] = {k: float(v) for k, v in err.items()}
        out[f'{tag}_pass'] = bool(err['cost'] < 1e-3 and err['balance'] < 1e-3 and err['soc'] < 1e-3
                                  and err['power'] < 1e-6 and err['price'] < 1e-9)
        # ---- 工作表与账本一致 ----
        wb = openpyxl.load_workbook(sheetfile, read_only=True, data_only=True)
        ws = wb['计划购电量']; rowsx = list(ws.values)
        dmax = 0.0
        for r in rowsx[1:]:
            date = str(pd.Timestamp(r[0]).date())
            g = iv[iv.date == date].sort_values('t')
            dmax = max(dmax, float(np.abs(np.array(r[1:145], float) - g.plan.to_numpy()).max()))
        out[f'{tag}_sheet_plan_max_diff'] = dmax
        if has_adj:
            ws = wb['调整购电量']; rowsx = list(ws.values); amax = 0.0
            for r in rowsx[1:]:
                date = str(pd.Timestamp(r[0]).date())
                g = iv[iv.date == date].sort_values('t')
                amax = max(amax, float(np.abs(np.array(r[1:145], float) - g.adjusted.to_numpy()).max()))
            out[f'{tag}_sheet_adjusted_max_diff'] = amax
        # ---- 工作表的**费用列**与账本一致 ----
        # 复核意见 §10.1：只证明「计划量与逐段表最大差 0 kWh」不足以说明费用列正确，
        # 必须同时核对末列合计与「费用汇总与版本」页的四个分项。
        cost_err = {}
        cost_sheets = [('计划购电量', 'plan_cost' if has_adj else 'total_cost')]
        if has_adj:
            cost_sheets.append(('调整购电量', 'total_cost'))
        for sheet, col in cost_sheets:
            vals = [r[-1] for r in list(wb[sheet].values)[1:]]
            cost_err[f'{sheet}_末列合计_减_{col}'] = float(
                abs(float(np.array(vals, float).sum()) - float(dl[col].sum())))
        summ = {r[0]: r[1] for r in wb['费用汇总与版本'].values}
        for item, col in (('计划购电费', 'plan_cost'), ('增购费', 'up_cost'),
                          ('减购净额', 'down_net_cost'), ('紧急购电费', 'emergency_cost')):
            cost_err[f'汇总页_{item}'] = float(abs(float(summ[item]) - float(dl[col].sum())))
        out[f'{tag}_sheet_cost_max_diff'] = max(cost_err.values())
        out[f'{tag}_sheet_cost_detail'] = cost_err
        wb.close()

    # ---- 2 严格因果性：四个发布时刻全覆盖 ----
    # 日前通过不能自动证明日内分支也通过：6/12/18 的决策多用了「当天 h:00 之前已观测到的
    # 负荷与电价」做日内偏差校正，且以 0:00 基准计划为合同基准，链路与 0:00 不同。
    # 对每个发布时刻分别扰动：① 该时刻之后的真实负荷/光伏/电价；② **尚未发布**的预报
    #（当天更晚的发布时刻与之后各天的全部预报）。决策必须逐位不变。
    pol = Q4P.Q4Policy(**{**polj['Q4-3']['params'], 'n_scen': 12,
                          **polj['Q4-3'].get('fixed_structural', {}),
                          **polj['Q4-3'].get('inert_fixed', {})})
    esc = Q4P.emergency_scale()
    caus = []
    rng = np.random.default_rng(0)

    def decide(bk, pb, day, k, soc, base):
        """复刻 run_q4 在发布时刻 k 的决策（不含执行），返回计划向量。"""
        t = Q4P.ISSUES[k] * 6
        center = pb.mean[day, k]
        nets, prices, w = Q4C.joint_scenarios(
            bk.mean[day, k], bk.residual[max(1, day - p.window):day, k, :], center,
            pb.resid_window(day, k, p.window), pol.n_scen, seed=97 * day + k,
            temper=pol.temper, shrink_net=pol.shrink_net, shrink_price=pol.shrink_price)
        vE = float(center.min() / p.eta)
        tvm = pol.kappa * shape * vE
        if base is None:
            out = Q4S.solve_q4(nets, prices, w, soc, p, tvm, edges, None, 0, Q4P.EXEC_STEPS,
                               emergency_scale=esc, adj_premium=pol.adj_premium)
        else:
            out = Q4S.solve_q4(nets, prices, w, soc, p, tvm, edges, base, 144 - t, Q4P.EXEC_STEPS,
                               emergency_scale=esc, adj_premium=pol.adj_premium)
        return np.asarray(out[0], float)

    for day in (120, 200, 300):
        soc0 = 5000.0
        q0 = decide(bank, pbank, day, 0, soc0, None)
        for k in range(4):
            t = Q4P.ISSUES[k] * 6
            base = None if k == 0 else q0[t:]
            ref = decide(bank, pbank, day, k, soc0, base)
            # 扰动：决策时点之后的真实数据 + 尚未发布的预报
            L2, P2, F2, V2 = L.copy(), P.copy(), F.copy(), V.copy()
            L2[day, t:] *= 1 + .4 * rng.random(144 - t)
            P2[day, t:] *= 1 + .4 * rng.random(144 - t)
            V2[day, t:] *= 1 + .6 * rng.random(144 - t)
            L2[day + 1:] *= 1 + .4 * rng.random(L2[day + 1:].shape)
            P2[day + 1:] *= 1 + .4 * rng.random(P2[day + 1:].shape)
            V2[day + 1:] *= 1 + .6 * rng.random(V2[day + 1:].shape)
            F2[day, k + 1:] *= 1.7                      # 当天更晚发布、此刻还看不到的预报
            F2[day + 1:] *= 1.7                         # 之后各天的全部预报
            bk2 = Bank(L2, P2, F2, seed, p)
            pb2 = Q4C.PriceBank(V2, tou, hp)
            base2 = None if k == 0 else decide(bk2, pb2, day, 0, soc0, None)[t:]
            prb = decide(bk2, pb2, day, k, soc0, base2)
            d = float(np.abs(ref - prb).max())
            caus.append(dict(date=str(DATES[day].date()), issue=Q4P.ISSUES[k],
                             max_plan_diff_kwh=d, passed=bool(d < 1e-6)))
            print(f'  因果性 {DATES[day].date()} {Q4P.ISSUES[k]:02d}:00 最大计划差 {d:.3e} kWh', flush=True)
    pd.DataFrame(caus).to_csv(R / 'q4_causality_checks.csv', index=False, encoding='utf-8-sig',
                              float_format='%.10f')
    out['causality_by_issue'] = caus
    out['causality_max_plan_diff_kwh'] = float(max(c['max_plan_diff_kwh'] for c in caus))
    out['causality_pass'] = bool(all(c['passed'] for c in caus))

    out['all_pass'] = bool(out['q4_2_pass'] and out['q4_3_pass'] and out['causality_pass']
                           and out.get('q4_2_sheet_plan_max_diff', 1) < 1e-6
                           and out.get('q4_3_sheet_plan_max_diff', 1) < 1e-6
                           and out.get('q4_3_sheet_adjusted_max_diff', 1) < 1e-6
                           and out.get('q4_2_sheet_cost_max_diff', 1) < 1e-4
                           and out.get('q4_3_sheet_cost_max_diff', 1) < 1e-4)
    (R / 'q4_verification.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
