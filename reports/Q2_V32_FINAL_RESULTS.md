# Q2 V3.2 最终版运行结果

本报告由本次 Colab 输出自动生成。

## 费用（双边库存价值修正）

| policy            | period         |   days |   planned_cost |   emergency_cost |   raw_total_cost |   start_inventory_value |   end_inventory_value |   inventory_adjustment |   adjusted_total_cost |   planned_kwh |   emergency_kwh |        spill_kwh |   emergency_slots |   emergency_day_frequency |   max_daily_cost |   start_soc |   end_soc |
|:------------------|:---------------|-------:|---------------:|-----------------:|-----------------:|------------------------:|----------------------:|-----------------------:|----------------------:|--------------:|----------------:|-----------------:|------------------:|--------------------------:|-----------------:|------------:|----------:|
| dual_terminal_g17 | full334        |    334 |    1.32918e+07 |         345186   |      1.3637e+07  |            -1.36424e-12 |             -1207.54  |               -1207.54 |           1.36358e+07 |   2.17767e+07 |        56089.7  |      2.66327e+06 |               527 |                  0.362275 |          84176.1 |     6000    |   8544.19 |
| dual_terminal_g17 | development153 |    153 |    6.1949e+06  |         194939   |      6.38984e+06 |         -1809.95        |               204.428 |                2014.38 |           6.39185e+06 |   1.00255e+07 |        31309.7  |      1.38875e+06 |               328 |                  0.470588 |          84176.1 |     9820.82 |   5504.04 |
| dual_terminal_g17 | frozen61       |     61 |    2.90408e+06 |          22882.1 |      2.92697e+06 |           204.428       |             -1207.54  |               -1411.97 |           2.92555e+06 |   4.6908e+06  |         4640.49 | 285691           |                49 |                  0.262295 |          71230.7 |     5504.04 |   8544.19 |

## 验收

| test                             |       value |   tolerance | passed   |
|:---------------------------------|------------:|------------:|:---------|
| causality                        | 0           |       0     | True     |
| balance                          | 1.7053e-13  |       1e-07 | True     |
| soc_recursion                    | 9.09495e-13 |       1e-07 | True     |
| lp_eq                            | 1.81899e-12 |       1e-05 | True     |
| lp_ineq                          | 1.13687e-13 |       1e-05 | True     |
| dual_support                     | 2.91038e-11 |       1e-05 | True     |
| dual_own_cut                     | 7.27596e-12 |       1e-05 | True     |
| value_iteration_convergence      | 0           |       0     | True     |
| soc_continuity_dual_terminal_g17 | 0           |       1e-07 | True     |

## result2.xlsx 回读校验

|                                     |       value |
|:------------------------------------|------------:|
| plan_shape_ok                       | 1           |
| storage_shape_ok                    | 1           |
| event_shape_ok                      | 1           |
| plan_max_abs_error_kwh              | 4.54747e-13 |
| total_kwh_max_abs_error             | 7.27596e-12 |
| total_cost_max_abs_error_yuan       | 2.18279e-11 |
| storage_charge_max_abs_error_kwh    | 5.45697e-12 |
| storage_discharge_max_abs_error_kwh | 9.09495e-13 |
| soc_start_max_abs_error_kwh         | 5.45697e-12 |
| soc_end_max_abs_error_kwh           | 5.45697e-12 |
| event_energy_abs_error_kwh          | 7.27596e-12 |
| formula_error_count                 | 0           |
| pass                                | 1           |
