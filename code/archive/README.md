# code/archive —— 研究记录归档

本目录存放**已被最终版取代**的研究过程代码。它们支撑论文中的论证（终端机制消融、信息价值、参数来源），但不是生产口径，**不要在新工作中依赖它们**。

最终版代码在 `code/q2_v32/`（问题 2）与 `code/problem1.ipynb`（问题 1），共享运行时在 `code/q2_v2/` 与 `code/q2_planning.py`。完整分层见 `code/README.md`。

---

## 归档内容

| 路径 | 内容 | 产出证据 |
| --- | --- | --- |
| `q2_v3/` | 对偶终端价值的首次实现 + 月度参数滚动验证 + VSS/EVPI | `results/q2_v3_dual_terminal/`、`figures/q2_v3_dual_terminal/`、`reports/Q2_V3_DUAL_TERMINAL_RESULTS.md` |
| `q2_v31/` | 终端条件消融：5 种终端处理 × 3 个分期、9/17 点网格收敛、配对 7 日块 Bootstrap | `results/q2_v31_terminal_ablation/`、`figures/q2_v31_terminal_ablation/`、`reports/Q2_V31_TERMINAL_ABLATION_RESULTS.md` |
| `build_q2_v3_notebook.py` | 由 `q2_v3/q2_dual_terminal.py` 生成 Colab notebook | `code/problem2_v3_dual_terminal.ipynb` |
| `build_q2_v31_notebook.py` | 由 `q2_v31/q2_terminal_ablation.py` 生成 Colab notebook | `code/problem2_v31_terminal_ablation.ipynb` |

归档时**没有**移动 `results/`、`figures/`、`reports/` 下的任何产物——两个模块的 `OUT` / `FIG` / `REPORT` 仍然指向原路径，因此归档代码若重跑，产物会落在原位，不会分裂成两份。

---

## 证据链：源码哈希

两个模块的 `run_manifest.json` 记录了产出结果时的源码 SHA256。**归档时修改了路径定位，因此当前哈希与记录值不同**，这是唯一差异。

| 模块 | 归档前（= run_manifest 记录值） | 归档后（当前） |
| --- | --- | --- |
| `q2_v3/q2_dual_terminal.py` | `8776f516a39e8bb119451f0b3c75d9ff47a2c866f03c8b5d5a45c26be3012567` | `e17299d7a85156df2c033543180018d8f0239f4f83f6680b46dc540f8ade43e4` |
| `q2_v31/q2_terminal_ablation.py` | `bdd65fe87b24956c8057976628ba03a502ca2d44ab0d1286d5f434ad6b209bed` | `062476cf482410e29ae840ae210ea6f0f6e3b5c812f5751625dab903b44d131f` |

归档前（与 run_manifest 逐字一致）的版本可从 git 历史取回并校验：

```bash
git show c6e4029:code/q2_v3/q2_dual_terminal.py   | shasum -a 256
git show c6e4029:code/q2_v31/q2_terminal_ablation.py | shasum -a 256
```

### 归档时做的最小改动

1. **两个模块**：把 `ROOT = HERE.parents[1]` 换成向上查找项目根的函数——按只在仓库根存在的 `C题/附件/附件2.xlsx` 逐级向上定位，因此模块不再依赖自身在 `code/` 下的深度。

   ```python
   def project_root(start):
       for candidate in (start, *start.parents):
           if (candidate / 'C题/附件/附件2.xlsx').exists():
               return candidate
       raise FileNotFoundError('未找到含 C题/附件/附件2.xlsx 的项目根目录')
   ```

2. **两个 builder**：同样的 ROOT 定位；`MODULE` 与注入 notebook 的 `__file__` / `sys.path` 路径改为 `code/archive/...`。
3. **`q2_v3/colab_pilot_v3.py`**：模块路径改为 `code/archive/q2_v3/q2_dual_terminal.py`。

除此之外**没有改动任何逻辑**——决策、目标函数、约束、随机数种子全部原样。冒烟测试确认两个模块仍可导入，`ROOT` 正确定位，`Experiment()` 能构造完整场景（14 条，权重和为 1.0）。

---

## 未改动：已执行 notebook

`code/problem2_v3_dual_terminal.ipynb`、`code/problem2_v3_dual_terminal_output.ipynb`、`code/problem2_v31_terminal_ablation.ipynb`、`code/problem2_v31_terminal_ablation_output.ipynb` **保留在 `code/` 未移动**，且其源码中的导入路径仍指向归档前的位置（`PROJECT/'code/q2_v3'`、`PROJECT/'code/q2_v31'`）。

这是**有意为之**：`*_output.ipynb` 是 Colab 执行证据，改动其源码单元会使输出与代码的对应关系失真。这些 notebook 是移动前的产物，记录的是当时的运行。

**如需复跑**，二选一：

- 用归档后的 builder 重新生成 notebook（已指向 archive 路径）：
  ```bash
  python code/archive/build_q2_v31_notebook.py   # 写出 code/problem2_v31_terminal_ablation.ipynb
  ```
- 或把该 notebook 第一个代码单元里的 `sys.path.insert(0, str(PROJECT/'code/q2_v31'))` 改为 `.../'code/archive/q2_v31'`。

---

## 注意

- 本目录下的代码**不属于**论文的生产口径。论文与交付件引用的是 `code/q2_v32/q2_final.py` 与 `code/problem1.ipynb`。
- 早期管线各自带有自己的费用数字（例如被取代的 H0 = 14 539 240.54 元、`q2_v3` 的 VSS/EVPI 一套结果），**引用时必须注明来源**，不要与最终版的 13 635 767.16 元混用。
- VSS / EVPI 在本项目中有**两套**结果，场景配置不同、不可互换：`q2_v3` 一套（RP 9 568 832.12 / EEV 10 474 478.50 / WS 8 389 482.00 / VSS 905 646.39 / EVPI 1 179 350.11），`results/q2_fused_v2_extensions` 一套（VSS 953 071.74 / EVPI 1 462 732.14）。两者都只在**同一经验场景分布内**成立。
