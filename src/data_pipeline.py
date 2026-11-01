"""Part A - Data pipeline: audit -> clean -> validate -> save (CSV + SQLite).

Run from the project root:   python src/data_pipeline.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]       # the fitpulse folder
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "fitpulse.db"


# ---------------------------------------------------------------- 1. AUDIT
def audit():
    """Look at the raw data and count the problems. Nothing is changed here."""
    members = pd.read_csv(RAW / "members.csv", keep_default_na=False)
    checkins = pd.read_csv(RAW / "checkins.csv", keep_default_na=False)
    payments = pd.read_csv(RAW / "payments.csv", keep_default_na=False)

    for name, df in [("members", members), ("checkins", checkins), ("payments", payments)]:
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

    age = pd.to_numeric(members.age, errors="coerce")
    print("\nMEMBERS")
    print("  different city spellings :", members.city.nunique())
    print("  different gender spellings:", members.gender.nunique())
    print("  blank ages               :", int((members.age.astype(str).str.strip() == "").sum()))
    print("  impossible ages (<16 or >90):", int(((age < 16) | (age > 90)).sum()))

    dur = pd.to_numeric(checkins.duration_min, errors="coerce")
    cal = pd.to_numeric(checkins.calories, errors="coerce")
    print("\nCHECKINS")
    print("  exact duplicate rows     :", int(checkins.duplicated().sum()))
    print("  negative durations       :", int((dur < 0).sum()))
    print("  missing/bad calories     :", int(cal.isna().sum()))
    joined = checkins.merge(members[["member_id", "join_date"]], on="member_id")
    before = pd.to_datetime(joined.checkin_ts).dt.normalize() < pd.to_datetime(joined.join_date)
    print("  check-ins before join date:", int(before.sum()))

    amt = pd.to_numeric(payments.amount_inr, errors="coerce")
    print("\nPAYMENTS")
    print("  exact duplicate rows     :", int(payments.duplicated().sum()))
    print("  unreadable amounts       :", int(amt.isna().sum()))


# ---------------------------------------------------------------- 2. CLEAN
def clean_data():
    """Apply one rule at a time. Each rule is explained in the comments."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    plans = pd.read_csv(RAW / "plans.csv")
    members = pd.read_csv(RAW / "members.csv", keep_default_na=False)
    checkins = pd.read_csv(RAW / "checkins.csv", keep_default_na=False)
    payments = pd.read_csv(RAW / "payments.csv", keep_default_na=False)

    # ---- MEMBERS
    # Rule 1: city -> remove spaces and use Title Case ("  PUNE " -> "Pune")
    members["city"] = members.city.str.strip().str.title()
    # Rule 2: gender -> male / female, blank becomes "unknown"
    members["gender"] = (members.gender.str.strip().str.lower()
                         .map({"male": "male", "m": "male", "female": "female", "f": "female"})
                         .fillna("unknown"))
    # Rule 3: impossible ages (<16 or >90) are not real, so make them missing...
    members["age"] = pd.to_numeric(members.age, errors="coerce")
    members.loc[~members.age.between(16, 90), "age"] = np.nan
    # ...then fill every missing age with the median age of that member's city
    members["age"] = members.age.fillna(members.groupby("city").age.transform("median")).round().astype(int)
    members["join_date"] = pd.to_datetime(members.join_date)

    # ---- CHECKINS
    checkins["checkin_ts"] = pd.to_datetime(checkins.checkin_ts)
    # Rule 4: remove rows that are exact copies of another row
    checkins = checkins.drop_duplicates().copy()
    # Rule 5: a check-in before the member joined is impossible -> drop it
    checkins = checkins.merge(members[["member_id", "join_date"]], on="member_id", how="left")
    checkins = checkins[checkins.checkin_ts.dt.normalize() >= checkins.join_date].drop(columns="join_date").copy()
    # Rule 6: negative duration is impossible -> make it missing, fill with the median
    # duration of the same workout type
    checkins["duration_min"] = pd.to_numeric(checkins.duration_min, errors="coerce")
    checkins.loc[checkins.duration_min < 0, "duration_min"] = np.nan
    checkins["duration_min"] = checkins.duration_min.fillna(
        checkins.groupby("workout_type").duration_min.transform("median"))
    # Rule 7: missing / "N/A" calories -> estimate = duration x typical calories-per-minute
    # for that workout type (typical = median of the rows that have a real value)
    checkins["calories"] = pd.to_numeric(checkins.calories, errors="coerce")
    per_min = (checkins.calories / checkins.duration_min).groupby(checkins.workout_type).transform("median")
    checkins["calories"] = checkins.calories.fillna(checkins.duration_min * per_min).round().astype(int)
    checkins["duration_min"] = checkins.duration_min.round().astype(int)

    # ---- PAYMENTS
    # Rule 8: remove exact duplicate payment rows
    payments = payments.drop_duplicates().copy()
    # Rule 9: unreadable amount ("error") -> use the monthly fee of the member's plan
    fee = plans.set_index("plan").monthly_fee_inr
    expected = payments.member_id.map(members.set_index("member_id").plan).map(fee)
    payments["amount_inr"] = pd.to_numeric(payments.amount_inr, errors="coerce").fillna(expected).astype(int)

    # ---- SAVE
    members["join_date"] = members.join_date.dt.strftime("%Y-%m-%d")
    checkins["checkin_ts"] = checkins.checkin_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    tables = {"plans": plans, "members": members, "checkins": checkins, "payments": payments}
    for name, df in tables.items():
        df.to_csv(CLEAN / f"{name}.csv", index=False)
    with sqlite3.connect(DB) as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)
    print("clean rows ->", {k: len(v) for k, v in tables.items()})
    return tables


# ---------------------------------------------------------------- 3. VALIDATE
def validate():
    """If any check fails, Python stops with an AssertionError. That is the point."""
    m = pd.read_csv(CLEAN / "members.csv")
    c = pd.read_csv(CLEAN / "checkins.csv")
    p = pd.read_csv(CLEAN / "payments.csv")
    assert m.member_id.is_unique, "duplicate member ids"
    assert m.city.nunique() == 5, "city names are not standardised"
    assert set(m.gender) <= {"male", "female", "unknown"}, "bad gender value"
    assert m.age.between(16, 90).all(), "age out of range"
    assert not c.duplicated().any(), "duplicate check-ins remain"
    assert (c.duration_min > 0).all(), "non-positive duration"
    assert (c.calories > 0).all(), "non-positive calories"
    assert p.amount_inr.notna().all() and (p.amount_inr > 0).all(), "bad payment amount"
    assert not p.duplicated().any(), "duplicate payments remain"
    assert set(c.member_id) <= set(m.member_id), "check-in for unknown member"
    with sqlite3.connect(DB) as con:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"plans", "members", "checkins", "payments"} <= tables, "table missing in database"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    print("===== AUDIT =====")
    audit()
    print("\n===== CLEAN =====")
    clean_data()
    print("\n===== VALIDATE =====")
    validate()
