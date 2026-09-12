# Q3 第二版HYBRID工程完善包（含联合情景候选）

来源：第二版 HYBRID 交付包。本目录为当前第三问主交付版本；主模型为 inherited_hybrid，joint_hybrid 仅为扩展对照。仓库根目录 result3.xlsx 与本目录主模型工作簿一致。

完整交付入口：`run_delivery.py`（求解子入口：`code/run_joint.py`）；Colab notebook：`code/problem3_joint.ipynb`。

## 本版实现
- 沿用历史截止明确的 Ridge 负荷预测、附件3四发布时刻预报、两阶段随机 LP、8段凹库存价值、因果物理执行与交付电价净额结算。
- 负荷/光伏残差按同一历史日和发布时刻成对保留，以两个通道的历史标准差缩放后聚类为最多12条代表路径；不独立抽样。预测加收缩残差后逐通道截断为非负，再计算净负荷。
- 只采用在当前发布时刻已完整观测完毕的历史24小时误差路径。
- 默认固定附件中的已选参数、终端价值CSV和2月1日初始SOC。不加载外来pickle，不重新运行参数搜索或终端价值训练。
- 首次生成全年因果滚动预测缓存；以后相同数据/代码/参数/环境直接读取，避免重新拟合Ridge。历史新增数据会使指纹失效，不能复用旧缓存掩盖变化。
- 提供分时刻和提前量预报误差、训练/评价分期联合尾部诊断；评价期诊断仅用于报告，不反馈选参。
- 修正无调整策略的可调整时间锁定；默认仅对照继承HYBRID与联合情景两种策略，不跑搜索或大型消融。
- 扰动未来实际值与未发布预报，在0/6/12/18分别检查重建后的情景和当前LP计划；同一已知SOC和合同条件。
- 独立核对物理账本，导出两个版本的四表工作簿，主交付保留第二版HYBRID。

## 运行
仅在Colab运行：
```
%cd /content/q3_joint_v1
!python run_delivery.py
```
初次运行下载 `cache/`；重新上传或保存在Google Drive的独立项目目录，才能跨Colab虚拟机复用。缓存不是只保留在临时 `/content` 就能永久存在。

## 输出
`results/comparison.csv`、`forecast_accuracy.csv`、`joint_error_diagnostics.csv`、`causal_checks.json`、`run_manifest.json`；两策略独立账本；`result3.xlsx`对应继承的第二版HYBRID，`result3_joint.xlsx`对应joint_hybrid候选；`cache/<fingerprint>/`含预报和联合情景；`reports/RESULTS_REPORT.md`记录实测。

## 数学与结论边界
本版是滚动两阶段补救近似，不是完整多阶段最优。无Wasserstein DRO；不宣称误差天然负相关。继承CVaR权重为0，风险指标只作评价。联合版同时改变聚类空间和非负裁剪，差值不等于纯相关性收益。没有运行完不得声称优于旧版。
诊断中的光伏误差是沿用插值后的10分钟运行误差，包含降尺度误差，不等于附件3原始整点预报准确率。该映射仍需结合题面口径说明。
Q3 工作簿采用自然日表头 00:00—24:00，按位置对应附件数据，不平移实际数值。Q1/Q2 保持原有模板标签约定。

另提供 `result3_inherited.xlsx`：已复现的原HYBRID基线工作簿，避免新候选表现不佳时缺少可交付的原方案。

最终文件定位：完整运行notebook或run_delivery.py后，`result3.xlsx`为已复现第二版HYBRID主模型；`result3_joint.xlsx`为联合情景候选。单独调用code/run_joint.py只是中间求解步骤，需继续运行qa/cache_and_delivery.py才能完成最终导出。
