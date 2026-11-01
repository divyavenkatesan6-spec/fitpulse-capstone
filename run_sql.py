"""Run every query in sql/queries.sql and print the results.  Usage: python run_sql.py"""
import re
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
text = (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
text = re.sub(r"--[^\n]*", "", text)                      # remove comment lines
queries = [q.strip() for q in text.split(";") if q.strip()]
titles = re.findall(r"-- (Q\d\..*)", (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8"))

with sqlite3.connect(ROOT / "data" / "fitpulse.db") as con:
    for title, q in zip(titles, queries):
        print("\n" + title)
        print(pd.read_sql_query(q, con).head(12).to_string(index=False))
