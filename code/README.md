# code/ —— 代码入口地图

本目录前后经历多轮迭代，因此文件较多。**看这一页就够，不用逐个翻。**

论文与交付件只认下面标 ★ 的文件；其余是研究记录或早期路线，保留是为了支撑论文论证和可追溯性。

---

## 一、三层分级

### ★ 最终版（论文与交付件的唯一口径）

| 文件 | 角色 |
| --- | --- |
| `problem1.ipynb` | **问题 1 最终模型**。均值日确定性 LP，产出 `result1.xlsx`。 |
| `q2_v32/q2_final.py` | **问题 2 最终生产版**。只求解最终采用的 `dual_terminal_g17` 策略，并写出 `result2.xlsx`。 |
| `problem2_v32_final.ipynb` | 上述模型的 Colab 源 notebook（由 `build_q2_v32_notebook.py` 生成，自包含）。 |
| `problem2_v32_final_output.ipynb` | 上述 notebook 的 **Colab 执行版**，含完整输出。论文取证看这个。 |
| `build_q2_v32_notebook.py` | 生成 `problem2_v32_final.ipynb`。 |
| `q2_v32/colab_pack_v32.py` | 打包 Colab 上传所需的输入（附件、模型、notebook、预测文件）。 |
| `q2_v32/colab_prepare_v32.py` | 在 Colab 上解包并安装运行环境。 |
| `q2_v32/colab_launch_v32.py` | 在 Colab 上后台执行 notebook 并打包产物。 |

### ★ 共享运行时（最终版依赖，不要随意改动）

| 文件 | 角色 |
| --- | --- |
| `q2_planning.py` | 提供 `Study`（数据读取、Ridge 预测、正/余弦与滞后特征）与 `weights_for`（指数时间权重）。**只复用这两个接口**；该模块自身还含一套早期两阶段 SAA 主实验，其费用数字属于旧口径。 |
| `q2_v2/dispatch.py` | **H1 逐时因果执行器**：日前计划 → 日内实际充放电 / 弃电 / 紧急购电。含 `execute_day` 与 `check_dispatch`。 |
| `q2_v2/utils.py` | 统一参数与单位：`BATTERY`、`T=144`、`DT=1/6`、`INTERVALS`（自然日区间标签）。 |

### 研究记录（保留原位，被多处交叉引用）

| 文件 / 目录 | 角色 | 为何不归档 |
| --- | --- | --- |
| `archive/` | 已归档的研究代码（q2_v3、q2_v31 及其 builder） | 见 `archive/README.md` |
| `q2_hybrid.py` | 早期「预测 × 策略」融合消融（H0–H4） | 被 9 处引用，且是预测端筛选证据 |
| `q2_hybrid_colab_pilot.py`、`q2_hybrid_colab_supervisor.py` | 上述实验的 Colab 驱动 | 与 notebook 路径耦合 |
| `colab_pack_q2_hybrid.py`、`colab_prepare_q2_hybrid.py` | 上述实验的打包 / 环境脚本 | 同上 |
| `build_q2_hybrid_notebook.py`、`build_q2_fused_v2_notebook.py` | 早期 notebook 的生成器 | 生成物在 `code/` 根部，移动会让路径失真 |
| `q2_v2/` 中除 `dispatch.py`、`utils.py` 外的文件 | 融合 V2 实验、CVaR 前沿、VSS/EVPI、滚动规划、预测器核心 | 与 `q2_fused.py`、`planning_extensions_v2.py` 互引较多 |
| `problem2*.ipynb`（除 v3.2 外） | 早期各版本 notebook | 都被对应的 `colab_*` / `build_*` 脚本按文件名引用 |
| `html_思路/problem1.html` | 问题 1 的建模思路稿 | 文档 |

---

## 二、notebook 命名约定

- `X.ipynb` —— 源 notebook（未执行，`execution_count` 为空），由 `build_*.py` 生成。
- `X_output.ipynb` —— 从 Colab 下载的**已执行**版本，含完整输出。

入库的 notebook 必须是 Colab 执行版（首个代码单元回显的项目根目录为 `/content`）。本地重跑产生的输出不得入库。

> 注意：`problem2_fused_v2_output.ipynb`、`problem2_fused_v2.ipynb`、`problem2_hybrid.ipynb` 三者**字节完全相同**，但都被脚本按名字引用，因此都保留。看着像冗余，实际不能删。

---

## 三、依赖关系

```text
problem1.ipynb                     （独立，只读附件 1）
    └── result1.xlsx

q2_final.py ★
    ├── q2_planning.py      → Study, weights_for
    ├── q2_v2/dispatch.py   → execute_day, check_dispatch
    ├── q2_v2/utils.py      → BATTERY, DT, T, INTERVALS
    └── 读 results/problem2_forecast_ablation_predictions.csv.gz（冻结的 Ridge 预测，48.6 MB）
    └── 产出 result2.xlsx + results/q2_v32_final/

archive/q2_v3/q2_dual_terminal.py      ┐
archive/q2_v31/q2_terminal_ablation.py ┘ 同样依赖上面三个共享模块
                                          ROOT 按 C题/附件/附件2.xlsx 向上查找，可随归档移动
```

---

## 四、改动前的三条约束

1. **不要移动 `q2_v2/dispatch.py` 与 `q2_v2/utils.py`**。最终版用 `sys.path.insert(ROOT/'code/q2_v2')` 硬编码定位它们，挪走会让最终版直接 import 失败。
2. **不要改 `q2_planning.py` 的 `Study`/`weights_for` 语义**。后果是整个 Q2 的费用数字全体漂移。改动后请用 README 里的强回归锚点验证：`q2_final.py` 的重跑必须与 `results/q2_v31_terminal_ablation/` 中 `dual_terminal_g17` 的记录逐位一致。
3. **不要改动已执行 notebook 的源码单元**。它们是 Colab 运行证据，改源码会让输出与代码的对应关系失真。

---

## 五、新增实验（如问题 3）怎么放

参照 v3.2 的做法，为每一轮独立的建模思路建一个 `q2_vNN/` 目录，内含：

```text
q2_vNN/
├── __init__.py
├── q2_<模型名>.py        # 模型主体：可 import，也可直接 python 运行
├── colab_pack_vNN.py     # 本地打包输入（只在本地跑）
├── colab_prepare_vNN.py  # 远端解包 + 装环境
└── colab_launch_vNN.py   # 远端后台执行 + 打包产物
```

外加 `code/build_q2_vNN_notebook.py` 与生成出的 `code/problem2_vNN_*.ipynb`。
产物落到 `results/q2_vNN_*/`、`figures/q2_vNN_*/`，报告写 `reports/`。

问题 3 的起手点：复制 `q2_v32/q2_final.py`，把 `write_result2` 改成 `write_result3`（多一张「调整购电量」表），并接入附件 3 的整点预报降尺度。详见根目录 `README.md` 第五节。
