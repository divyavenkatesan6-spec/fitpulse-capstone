"""Run 8 freeze-request scenarios through the agent and save outputs/agent_results.csv
Usage (from the fitpulse folder):  python run_agent_demo.py
"""
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "freeze_ledger.jsonl").unlink(missing_ok=True)      # start every demo with a clean ledger
shutil.rmtree(ROOT / "state", ignore_errors=True)

from src.freeze_agent import run_agent                                  # noqa: E402  (import after cleanup)

with sqlite3.connect(ROOT / "data" / "fitpulse.db") as con:
    def member(plan):
        return con.execute("select member_id from members where plan=? limit 1", (plan,)).fetchone()[0]

standard, premium = member("Standard"), member("Premium")
TODAY = "2025-07-10"                      # next billing date is 2025-08-01 (22 days away)

scenarios = [
    ("standard_30_days", f"Member {standard} wants to freeze for 30 days. Today is {TODAY}."),
    ("premium_first_freeze", f"Member {premium} wants to freeze for 20 days. Today is {TODAY}."),
    ("premium_second_freeze", f"Member {premium} wants to freeze for 20 days. Today is {TODAY}."),
    ("too_short", f"Member {standard} wants to freeze for 5 days. Today is {TODAY}."),
    ("too_long", f"Member {standard} wants to freeze for 90 days. Today is {TODAY}."),
    ("too_late", f"Member {standard} wants to freeze for 30 days. Today is 2025-07-30."),
    ("unknown_member", f"Member M9999 wants to freeze for 30 days. Today is {TODAY}."),
    ("missing_info", "Please freeze my membership."),
    ("confused_agent_loop", f"[LOOP] Member {standard} freeze 30 days. Today is {TODAY}."),
]
rows = []
for name, text in scenarios:
    r = run_agent(text, scenario=name)
    r["tools_used"] = " > ".join(r["tools_used"])
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs" / "agent_results.csv", index=False)
print(df[["scenario", "status", "steps", "answer"]].to_string(index=False))
