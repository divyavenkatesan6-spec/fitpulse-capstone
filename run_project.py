"""Run the whole FitPulse project in order.   Usage:  python run_project.py
(Run `python generate_data.py` once first to create the raw data.)
"""
import subprocess
import sys

STEPS = [
    ("PART A - audit, clean, validate", ["src/data_pipeline.py"]),
    ("SQL queries", ["run_sql.py"]),
    ("PART B - segments and churn model", ["src/member_models.py"]),
    ("PART C - ticket triage", ["src/triage.py"]),
    ("PART D1 - policy search", ["src/policy_rag.py"]),
    ("PART D2 - freeze agent", ["run_agent_demo.py"]),
]
for title, cmd in STEPS:
    print(f"\n{'=' * 8} {title} {'=' * 8}")
    subprocess.run([sys.executable] + cmd, check=True)
print("\nProject finished. Look in the outputs/ folder.")
