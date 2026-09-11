"""Colab 试运行：在完整后台实验前验证 3 日机制与两项费用回归。"""
from pathlib import Path
import sys

root = Path("/content/CUMCM_2026_Last_Dance")
sys.path.insert(0, str(root / "code"))
import importlib  # noqa: E402
import q2_hybrid as hybrid  # noqa: E402
hybrid = importlib.reload(hybrid)
HybridStudy, PolicyConfig, run_pilot = hybrid.HybridStudy, hybrid.PolicyConfig, hybrid.run_pilot

s = HybridStudy(root)
pilot, tests = run_pilot(s)
team, _ = s.simulate("TeamA", "H3_quantile", 31, 365,
                     constant=PolicyConfig(.8, .5, 28, 3))
team_cost = float(team.total_cost.sum())
print("Q2_HYBRID_PILOT_OK")
print("pilot_lp_seconds", float(pilot.seconds.sum()))
print("team_a_port_cost", team_cost)
print("team_a_reference_error", team_cost - 13820986.576531284)
