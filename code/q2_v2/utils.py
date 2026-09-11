"""统一参数、单位、输入读取与文件输出。原始 Excel 始终只读。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/q2_fused_v2"
OUTPUTS = ROOT / "results/q2_fused_v2"
FIGURES = ROOT / "figures/q2_fused_v2"
REPORTS = ROOT / "reports"
T = 144
DT = 1 / 6  # h，10 分钟
TOL = 1e-6
SELECTED_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


@dataclass(frozen=True)
class Battery:
    capacity: float = 12000.0
    minimum: float = 1200.0
    maximum: float = 10800.0
    power: float = 5000.0  # 母线侧 kW
    eta_c: float = 0.9
    eta_d: float = 0.9
    initial: float = 6000.0

    @property
    def limit(self) -> float:
        return self.power * DT


BATTERY = Battery()


def clock_text(index: int) -> str:
    minutes = 10 * int(index)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


INTERVALS = [f"{clock_text(t)}-{clock_text(t + 1)}" for t in range(T)]
BLOCKS = [f"{4 * i}:00-{4 * (i + 1)}:00" for i in range(6)]


def ensure_dirs() -> None:
    for path in (RESULTS, OUTPUTS, FIGURES, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2,
                               default=json_default, allow_nan=False), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def load_rows(path: Path, sheet: str | int = 0) -> list:
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws = workbook.worksheets[sheet] if isinstance(sheet, int) else workbook[sheet]
        return list(ws.values)
    finally:
        workbook.close()


def check_time_headers(values) -> None:
    if len(values) != T:
        raise ValueError("每天必须恰好 144 个时间标签")
    for t, value in enumerate(values, 1):
        if t == T:
            if str(value).strip() not in ("0:00+1", "00:00+1", "24:00"):
                raise ValueError(f"无法识别日末标签：{value}")
        else:
            if isinstance(value, time):
                minutes = value.hour * 60 + value.minute
            elif isinstance(value, str):
                parts = value.strip().split(":")
                if len(parts) != 2:
                    raise ValueError(f"无法识别时间标签：{value}")
                minutes = 60 * int(parts[0]) + int(parts[1])
            else:
                raise ValueError(f"无法识别时间标签类型：{type(value)}")
            if minutes != 10 * t:
                raise ValueError(f"时间顺序不正确：第 {t} 个为 {value}")


def load_data() -> dict:
    # 压缩包原版使用 data/raw，本项目的原始附件位于 C题/附件。
    # 只在这两个明确的只读位置中选择，避免误读历史结果。
    candidates = [ROOT / "C题/附件", ROOT / "data/raw"]
    source = next((p for p in candidates if (p / "附件1.xlsx").exists()
                   and (p / "附件2.xlsx").exists()), None)
    if source is None:
        raise FileNotFoundError("未找到附件1.xlsx 和附件2.xlsx")
    one = load_rows(source / "附件1.xlsx")
    if list(one[0]) != ["时间", "电价", "小区负载", "光伏发电预测功率"]:
        raise ValueError("附件 1 表头不匹配")
    check_time_headers([r[0] for r in one[1:]])
    a = np.array([r[1:] for r in one[1:]], dtype=float)
    loads = load_rows(source / "附件2.xlsx", "小区负载")
    solars = load_rows(source / "附件2.xlsx", "光伏发电实际功率")
    check_time_headers(loads[0][1:])
    check_time_headers(solars[0][1:])
    dates = [r[0].date() for r in loads[1:]]
    dates_pv = [r[0].date() for r in solars[1:]]
    expected = [date(2025, 1, 1) + timedelta(days=i) for i in range(365)]
    if dates != expected or dates != dates_pv:
        raise ValueError("附件 2 日期缺失、重复或顺序异常")
    load = np.array([r[1:] for r in loads[1:]], dtype=float)
    pv = np.array([r[1:] for r in solars[1:]], dtype=float)
    for name, x, shape in (("附件1", a, (T, 3)), ("负荷", load, (365, T)),
                           ("光伏", pv, (365, T))):
        if x.shape != shape or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError(f"{name} 存在非法数据或形状错误")
    if np.any(a[:, 0] <= 0):
        raise ValueError("当前模型要求严格正电价")
    return dict(price=a[:, 0], q1_load=a[:, 1], q1_pv=a[:, 2],
                load=load, pv=pv, dates=dates, source_dir=source)


def data_audit(data: dict) -> dict:
    records = {}
    for name in ("price", "q1_load", "q1_pv", "load", "pv"):
        x = data[name]
        records[name] = dict(shape=list(x.shape), missing=int(np.isnan(x).sum()),
                             minimum=float(x.min()), maximum=float(x.max()))
    records["battery"] = asdict(BATTERY)
    records["interval_hours"] = DT
    records["time_convention"] = "右端标签：00:10 对应 00:00-00:10；24:00 对应 23:50-24:00"
    source = Path(data["source_dir"])
    records["source_directory"] = str(source.relative_to(ROOT))
    records["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.glob("*.xlsx"))}
    write_json(OUTPUTS / "data_audit.json", records)
    write_csv(pd.DataFrame({"sample_end": [clock_text(t + 1) for t in range(T)],
                            "physical_interval": INTERVALS,
                            "array_index": np.arange(T)}), OUTPUTS / "time_mapping.csv")
    return records


def environment() -> dict:
    import scipy
    import matplotlib
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                openpyxl=openpyxl.__version__, matplotlib=matplotlib.__version__)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def render(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:,.{digits}f}"
        return str(x)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(render(x) for x in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)
