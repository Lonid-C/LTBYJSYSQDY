"""每次发布的纯光伏信息价值：保留负荷修正、SOC 观测与 LP 重算，只把某一次的光伏预报换回该日历日 0:00 的版本。"""
import json
import pandas as pd
import q3


def main():
    pars = json.loads((q3.OUT / 'parameters.json').read_text())
    p = q3.Params(**pars['selected'])
    L, P, F, price, seed = q3.load()
    full = pd.read_csv(q3.OUT / 'daily_6+12+18.csv')
    out = []
    for k, h in enumerate(q3.ISSUES[1:], 1):
        b = q3.Bank(L, P, F, seed, p)
        b.stale_pv_issues([k])
        df, _, _ = q3.run(L, P, price, b, p, initial=pars['initial_feb1'])
        q3.csv(f'pure_pv_remove_{h}.csv', df)
        out.append(dict(issue=h,
                        pv_information_cash_value=df.total_cost.sum() - full.total_cost.sum(),
                        pv_information_inventory_adjusted_value=df.inventory_adjusted_cost.sum() - full.inventory_adjusted_cost.sum()))
        print('PURE-PV', h, out[-1], flush=True)
    q3.csv('pure_release_value.csv', out)


if __name__ == '__main__':
    main()
