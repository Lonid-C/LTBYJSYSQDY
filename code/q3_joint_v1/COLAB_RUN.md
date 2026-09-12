# Colab复用说明

这是已冻结配置的候选版本，不需要GPU，也不运行参数搜索。

1. 将交付ZIP上传Colab文件区，解压到 `/content/`，得到 `/content/q3_joint_v1/`。
2. 安装依赖：`pip install numpy pandas scipy openpyxl nbformat nbclient ipykernel tabulate`。为了复用随包缓存，优先按 `results/run_manifest.json` 中记录的版本安装numpy/pandas/scipy，并使用相同Python版本；不匹配会自动生成新缓存，不会误用。
3. 打开 `code/problem3_joint.ipynb`，全部运行。第一格进入上述路径。
4. 下载整个目录或将cache复制到自己的Google Drive项目目录。跨虚拟机运行前必须恢复cache；只留在/content会随虚拟机销毁而丢失。
5. 结果包含两个策略的统一口径对照；`result3.xlsx`对应继承基线，`result3_joint.xlsx`对应联合情景候选。
6. 运行后将执行版notebook保存为 `code/problem3_joint_output.ipynb`；确认所有代码格执行、无error输出。核验完成后停止Colab会话。

## 已保存的资产
- assets/hybrid_selected_policy.json：选定策略参数。
- assets/hybrid_terminal_value_segments.csv：终端价值边际和分段端点。
- cache/<指纹>/forecast_bank.npz：全部滚动预测输出，复用时不需要再次拟合负荷Ridge。
- cache/<指纹>/paired_errors.npz：负荷、光伏成对误差。
- cache/<指纹>/joint_*.npz：联合情景、权重、历史来源和可用时间。

缓存只用于完全相同版本的回放；新数据或新代码不能宣称无需更新模型。没有把OAuth、会话状态等凭据放入包内。

最终文件定位：完整运行notebook或run_delivery.py后，`result3.xlsx`为已复现第二版HYBRID主模型；`result3_joint.xlsx`为联合情景候选。单独调用code/run_joint.py只是中间求解步骤，需继续运行qa/cache_and_delivery.py才能完成最终导出。
