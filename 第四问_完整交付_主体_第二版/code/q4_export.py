"""第四问交付：生成 result4-2.xlsx / result4-3.xlsx 与论文用的指定日期表。

result4-2 与 result2 同构（计划购电量 / 充放电量 / 紧急购电量）；
result4-3 与 result3 同构（多一张「调整购电量」）。全天购电费按**附件 4 的实际电价**计算。
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results4'
SELECTED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
HEAD_FILL = PatternFill('solid', fgColor='325B88')
HEAD_FONT = Font(name='Arial', size=10, bold=True, color='FFFFFF')
BODY_FONT = Font(name='Arial', size=10)
THIN = Side(style='thin', color='D8DEE6')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def clock(t):
    return f'{t // 6:02d}:{(t % 6) * 10:02d}'


def interval(t):
    return f'{clock(t)}-{clock(t + 1)}'


def style_header(ws, ncol):
    for j in range(1, ncol + 1):
        c = ws.cell(row=1, column=j)
        c.fill = HEAD_FILL; c.font = HEAD_FONT
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = 'B2'
    ws.sheet_view.showGridLines = False


def autosize(ws, ncol, width=17.0, first_width=None):
    for j in range(1, ncol + 1):
        ws.column_dimensions[get_column_letter(j)].width = (first_width if j == 1 and first_width else width)


def build(iv, daily, adjusted_sheet, out, meta=None):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    # 「全天购电费」的口径：题目模板要求该列反映当日**实际支付**。
    #  Q4-3 有「调整购电量」表承载最终计划与总现金费用，计划表那一列只含计划合同费用，
    #  故显式更名为「全天计划购电费」；Q4-2 没有调整表，计划表就是唯一的购电表，
    #  该列必须是**含紧急购电在内的当日总现金费用**，否则全簿合计与报告对不上。
    plan_cost_key = 'plan_cost' if adjusted_sheet else 'total_cost'
    plan_cost_name = '全天计划购电费' if adjusted_sheet else '全天购电费（含紧急购电）'
    sheets = [('计划购电量', 'plan', plan_cost_key, plan_cost_name)]
    if adjusted_sheet:
        sheets.append(('调整购电量', 'adjusted', 'total_cost', '全天购电费（含紧急购电）'))
    for name, key, costkey, costname in sheets:
        header = ['日期\\时间'] + [interval(t) for t in range(144)] + ['全天购电量', costname]
        ws = wb.create_sheet(name); ws.append(header)
        for date, g in iv.groupby('date', sort=True):
            g = g.sort_values('t')
            d = daily[daily.date == date].iloc[0]
            ws.append([date] + [float(x) for x in g[key].tolist()] + [float(g[key].sum()), float(d[costkey])])
        style_header(ws, len(header)); autosize(ws, len(header))
        for i in range(2, ws.max_row + 1):
            ws.cell(row=i, column=1).number_format = 'yyyy-mm-dd'
            ws.cell(row=i, column=1).alignment = Alignment(horizontal='center')
            ws.cell(row=i, column=1).font = BODY_FONT
            for j in range(2, len(header) + 1):
                cell = ws.cell(row=i, column=j)
                cell.number_format = '#,##0.000'; cell.font = BODY_FONT
        ws.column_dimensions[get_column_letter(len(header) - 1)].width = 22
        ws.column_dimensions[get_column_letter(len(header))].width = 22

    ws = wb.create_sheet('充放电量')
    ws.append(['日期', '时间段', '充电量', '放电量', '时刻', '储电量'])
    for date, g in iv.groupby('date', sort=True):
        g = g.sort_values('t').reset_index(drop=True)
        d = daily[daily.date == date].iloc[0]
        for start in range(0, 144, 24):
            piece = g.iloc[start:start + 24]
            mark = '00:00' if start == 0 else ('24:00' if start == 120 else '')
            soc = float(d.soc_start) if start == 0 else (float(g.soc_end.iloc[-1]) if start == 120 else None)
            ws.append([date, f'{clock(start)}-{clock(start + 24)}', float(piece.charge.sum()),
                       float(piece.discharge.sum()), mark, soc])
    style_header(ws, 6)
    for i in range(2, ws.max_row + 1):
        ws.cell(row=i, column=1).number_format = 'yyyy-mm-dd'
        ws.cell(row=i, column=6).number_format = '#,##0.000'
        for j in range(1, 7):
            ws.cell(row=i, column=j).font = BODY_FONT
        ws.cell(row=i, column=2).alignment = Alignment(horizontal='center')
        ws.cell(row=i, column=5).alignment = Alignment(horizontal='center')
    autosize(ws, 6, first_width=12.0); ws.column_dimensions['B'].width = 22

    ws = wb.create_sheet('紧急购电量')
    ws.append(['日期', '购电时间段', '购电量'])
    for date, g in iv.groupby('date', sort=True):
        g = g.sort_values('t').reset_index(drop=True)
        j, found = 0, False
        while j < 144:
            if g.emergency.iloc[j] <= 1e-7:
                j += 1; continue
            s0 = j
            while j < 144 and g.emergency.iloc[j] > 1e-7:
                j += 1
            ws.append([date, f'{clock(s0)}-{clock(j)}', float(g.emergency.iloc[s0:j].sum())]); found = True
        if not found:
            ws.append([date, '无', 0.0])
    style_header(ws, 3)
    for i in range(2, ws.max_row + 1):
        ws.cell(row=i, column=1).number_format = 'yyyy-mm-dd'
        ws.cell(row=i, column=3).number_format = '#,##0.000'
        for j in range(1, 4):
            ws.cell(row=i, column=j).font = BODY_FONT
        ws.cell(row=i, column=2).alignment = Alignment(horizontal='center')
    autosize(ws, 3, first_width=12.0); ws.column_dimensions['B'].width = 24

    # ---- 费用汇总与版本：把总现金费用、库存校正项与版本指纹落到簿子里，便于单独核对 ----
    if meta:
        ws = wb.create_sheet('费用汇总与版本')
        plan = float(daily.plan_cost.sum()); up = float(daily.up_cost.sum())
        dn = float(daily.down_net_cost.sum()); em = float(daily.emergency_cost.sum())
        cash = float(daily.total_cost.sum())
        e0 = float(daily.soc_start.iloc[0]); e1 = float(daily.soc_end.iloc[-1])
        corr = meta['vE_ref'] * (e0 - e1)
        rows = [('第四问费用汇总与版本', None),
                ('版本 variant_id', meta['variant_id']),
                ('分支', meta['branch']), ('主策略', meta['strategy']),
                ('评价期', '2025-02-01 至 2025-12-31（334 天 / 48,096 段）'),
                ('结算口径', '计划与调整均按交付时段实际电价；日末净额退款；紧急购电 5 倍'),
                ('项目', '金额（元）'),
                ('计划购电费', plan), ('增购费', up), ('减购净额', dn), ('紧急购电费', em),
                ('总现金费用', None),
                ('库存校正项 v_E×(期初−期末)', corr),
                ('库存校正费用', None),
                ('期初库存（kWh）', e0), ('期末库存（kWh）', e1),
                ('库存参考价 v_E（元/kWh）', meta['vE_ref']),
                ('说明', '「总现金费用」已包含计划、增购、减购净额与紧急购电四项，即当日实际支付总额。'),
                ('各表末列口径', meta['col_note']),
                ('核对方式', '本页 B8:B11 逐项之和 = B12「总现金费用」；B12 与各逐日明细表末列的合计一致。'
                              '仅核对「计划量与逐段表最大差 0 kWh」不足以证明费用列正确，须同时核对本页四个分项。'),
                ('数据来源', f'本包 results4/{meta["tag"]}_daily.csv 与 {meta["tag"]}_intervals.csv；附件 1 至 4。')]
        for r in rows:
            ws.append(list(r))
        ws['B12'] = '=SUM(B8:B11)'
        ws['B14'] = '=B12+B13'
        for cell in ('B16', 'B17', 'B18'):   # 说明类长文本换行显示
            ws[cell].alignment = Alignment(wrap_text=True, vertical='top')
        ws.sheet_view.showGridLines = False
        ws.column_dimensions['A'].width = 34; ws.column_dimensions['B'].width = 62
        ws['A1'].font = Font(name='Arial', size=14, bold=True)
        for j in (1, 2):
            c = ws.cell(row=7, column=j)
            c.fill = HEAD_FILL; c.font = HEAD_FONT
            c.alignment = Alignment(horizontal='center', vertical='center')
        for i in range(2, ws.max_row + 1):
            for j in (1, 2):
                c = ws.cell(row=i, column=j)
                if i != 7:
                    c.font = BODY_FONT
                c.alignment = Alignment(vertical='center', wrap_text=(j == 2 and i >= 18))
            ws.row_dimensions[i].height = 20
        for i in range(8, 18):
            ws.cell(row=i, column=2).number_format = '#,##0.00'
        ws.cell(row=17, column=2).number_format = '0.0000000000'
        for i in (12, 14):
            ws.cell(row=i, column=1).font = Font(name='Arial', size=10, bold=True)
            ws.cell(row=i, column=2).font = Font(name='Arial', size=10, bold=True)
        # 自检：汇总页的现金合计必须与逐日表一致
        assert abs((plan + up + dn + em) - cash) < 1e-3, (plan + up + dn + em, cash)

    wb.save(out)
    return wb.sheetnames, [wb[s].max_row for s in wb.sheetnames]


def selected_tables(iv, daily, tag):
    """论文表 1 / 表 2 / 表 3（四个指定日期）。"""
    t1, t2, t3 = [], [], []
    for date in SELECTED:
        g = iv[iv.date == date].sort_values('t')
        if g.empty:
            continue
        d = daily[daily.date == date].iloc[0]
        q = g.adjusted.to_numpy() if 'adjusted' in g else g.plan.to_numpy()
        for h in (10, 12, 14, 16, 18, 20):
            t = h * 6
            t1.append(dict(date=date, interval=f'{clock(t)}-{clock(t+1)}', purchase_kwh=float(q[t])))
        t1.append(dict(date=date, interval='全天购电量', purchase_kwh=float(q.sum())))
        t1.append(dict(date=date, interval='全天购电费(元)', purchase_kwh=float(d.total_cost)))
        gg = g.reset_index(drop=True)
        for start in range(0, 144, 24):
            piece = gg.iloc[start:start + 24]
            t2.append(dict(date=date, block=f'{clock(start)}-{clock(start+24)}',
                           charge_kwh=float(piece.charge.sum()), discharge_kwh=float(piece.discharge.sum())))
        t2.append(dict(date=date, block='0:00 储电量', charge_kwh=float(d.soc_start), discharge_kwh=np.nan))
        t2.append(dict(date=date, block='24:00 储电量', charge_kwh=float(gg.soc_end.iloc[-1]), discharge_kwh=np.nan))
        e = gg.emergency.to_numpy(); j = 0; any_e = False
        while j < 144:
            if e[j] <= 1e-7:
                j += 1; continue
            s0 = j
            while j < 144 and e[j] > 1e-7:
                j += 1
            t3.append(dict(date=date, interval=f'{clock(s0)}-{clock(j)}', emergency_kwh=float(e[s0:j].sum())))
            any_e = True
        if not any_e:
            t3.append(dict(date=date, interval='无', emergency_kwh=0.0))
    for name, rowsx in (('table1', t1), ('table2', t2), ('table3', t3)):
        pd.DataFrame(rowsx).to_csv(R / f'q4_{tag}_selected_{name}.csv', index=False,
                                   encoding='utf-8-sig', float_format='%.6f')
    return len(t1), len(t2), len(t3)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import q4_config as CFG
    vid = CFG.variant_id()
    log = [f'variant_id: {vid}']
    import json as _json
    cfgj = _json.loads((R / 'model_config.json').read_text(encoding='utf8'))
    vE = float(cfgj['storage']['vE_ref'])
    strat = {'q4_2': '分位法 α=0.675（1 月重标定）', 'q4_3': 'HYBRID-4L 联合情景两阶段随机 LP'}
    branch = {'q4_2': 'Q4-2：沿用第二问决策规则（仅 0:00 计划，日内无增减购）',
              'q4_3': 'Q4-3：沿用第三问决策规则（0:00 计划 + 6/12/18 三次调整）'}
    for tag, adj, out in (('q4_2', False, ROOT / 'result4-2.xlsx'),
                          ('q4_3', True, ROOT / 'result4-3.xlsx')):
        iv = pd.read_csv(R / f'{tag}_intervals.csv')
        daily = pd.read_csv(R / f'{tag}_daily.csv')
        col_note = ('「计划购电量」末列 =「全天购电费（含紧急购电）」= 计划购电费 + 紧急购电费 '
                    '= 当日总现金费用（Q4-2 无增/减购，故两者相等）。'
                    if not adj else
                    '「计划购电量」末列 =「全天计划购电费」，**仅含计划合同费用**；'
                    '「调整购电量」末列 =「全天购电费（含紧急购电）」= 计划 + 增购 + 减购净额 + 紧急购电 '
                    '= 当日总现金费用。')
        meta = dict(variant_id=vid, vE_ref=vE, tag=tag, strategy=strat[tag], branch=branch[tag],
                    col_note=col_note)
        names, rowsn = build(iv, daily, adj, out, meta)
        n1, n2, n3 = selected_tables(iv, daily, tag)
        log.append(f'{out.name}: sheets={names} rows={rowsn} selected_rows=({n1},{n2},{n3})')
        print(log[-1], flush=True)
    (R / 'q4_export.log').write_text('\n'.join(log) + '\n', encoding='utf8')
    print('variant_id', vid, flush=True)


if __name__ == '__main__':
    main()
