"""Part B - Member segments (K-means) and churn prediction.

Run from the project root:   python src/member_models.py

Two ideas to remember:
1. SEGMENTS group members by behaviour (no "right answer" - unsupervised).
2. CHURN MODEL predicts a future event. To avoid cheating (data leakage) the features
   only use information from BEFORE the cutoff date, and the label (did the member
   come back?) is measured AFTER the cutoff.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-07-01")          # the day after the last check-in


def load():
    members = pd.read_csv(CLEAN / "members.csv", parse_dates=["join_date"])
    checkins = pd.read_csv(CLEAN / "checkins.csv", parse_dates=["checkin_ts"])
    payments = pd.read_csv(CLEAN / "payments.csv")
    return members, checkins, payments


# ====================================================== PART B1: SEGMENTS
def member_segments(k=4):
    """Group members by Recency, Frequency and average workout length."""
    members, checkins, _ = load()
    feat = checkins.groupby("member_id").agg(
        last_visit=("checkin_ts", "max"),
        frequency=("checkin_id", "nunique"),
        avg_minutes=("duration_min", "mean")).reset_index()
    feat["recency"] = (SNAPSHOT - feat.last_visit).dt.days      # days since last visit
    feat = feat.drop(columns="last_visit")

    cols = ["recency", "frequency", "avg_minutes"]
    X = StandardScaler().fit_transform(feat[cols])               # put columns on one scale

    # Try K = 2..8 and compare. Silhouette: higher is better (max 1).
    diag = []
    for n in range(2, 9):
        km = KMeans(n_clusters=n, n_init=10, random_state=42).fit(X)
        diag.append({"k": n, "inertia": round(km.inertia_, 1),
                     "silhouette": round(silhouette_score(X, km.labels_), 3)})
    diag = pd.DataFrame(diag)

    feat["segment_id"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    profile = feat.groupby("segment_id").agg(
        members=("member_id", "size"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), avg_minutes=("avg_minutes", "mean")).round(1).reset_index()

    # Name each segment from its numbers: most visits = Champions, longest gap = At Risk
    names = {}
    names[profile.sort_values("frequency").segment_id.iloc[-1]] = "Champions"
    remaining = profile[~profile.segment_id.isin(names)]
    names[remaining.sort_values("recency").segment_id.iloc[-1]] = "At Risk"
    for sid in profile.segment_id:
        row = profile.set_index("segment_id").loc[sid]
        if sid in names:
            continue
        # recently active -> Regulars; not seen for 3+ weeks -> Fading
        names[sid] = "Fading" if row.recency > 21 else "Regulars"
    feat["segment_name"] = feat.segment_id.map(names)
    profile["segment_name"] = profile.segment_id.map(names)

    OUT.mkdir(exist_ok=True)
    feat.to_csv(OUT / "member_segments.csv", index=False)
    explained = PCA(n_components=2).fit(X).explained_variance_ratio_.round(3)
    return feat, diag, profile, explained


# ====================================================== PART B2: CHURN MODEL
def build_features(cutoff, with_label):
    """One row per member, using ONLY data before `cutoff`.
    Label (if asked): churned = 1 when the member has NO visit in the 30 days after cutoff."""
    members, checkins, payments = load()
    cutoff = pd.Timestamp(cutoff)
    past = checkins[checkins.checkin_ts < cutoff]
    d30 = past[past.checkin_ts >= cutoff - pd.Timedelta(days=30)]
    d60 = past[(past.checkin_ts >= cutoff - pd.Timedelta(days=60)) &
               (past.checkin_ts < cutoff - pd.Timedelta(days=30))]

    f = past.groupby("member_id").agg(
        visits_total=("checkin_id", "nunique"), avg_minutes=("duration_min", "mean"),
        avg_calories=("calories", "mean"), last_visit=("checkin_ts", "max")).reset_index()
    f["days_since_last_visit"] = (cutoff - f.last_visit).dt.days
    f = f.drop(columns="last_visit")
    f = f.merge(d30.groupby("member_id").size().rename("visits_last_30d"), on="member_id", how="left")
    f = f.merge(d60.groupby("member_id").size().rename("visits_prev_30d"), on="member_id", how="left")
    f[["visits_last_30d", "visits_prev_30d"]] = f[["visits_last_30d", "visits_prev_30d"]].fillna(0)
    f["visit_trend"] = f.visits_last_30d - f.visits_prev_30d          # falling = warning sign

    pay = payments[payments.month < cutoff.strftime("%Y-%m")]
    failed = pay[pay.status == "failed"].groupby("member_id").size().rename("failed_payments")
    f = f.merge(failed, on="member_id", how="left").fillna({"failed_payments": 0})

    f = f.merge(members[["member_id", "plan", "city", "age", "gender", "join_date"]], on="member_id")
    f = f[f.join_date < cutoff].copy()
    f["tenure_days"] = (cutoff - f.join_date).dt.days
    f = f.drop(columns="join_date")

    if with_label:
        future = checkins[(checkins.checkin_ts >= cutoff) &
                          (checkins.checkin_ts < cutoff + pd.Timedelta(days=30))]
        f["churned"] = (~f.member_id.isin(future.member_id)).astype(int)
    return f


NUMERIC = ["visits_total", "avg_minutes", "avg_calories", "days_since_last_visit", "visits_last_30d",
           "visits_prev_30d", "visit_trend", "failed_payments", "age", "tenure_days"]
CATEGORICAL = ["plan", "city", "gender"]


def build_churn_model():
    """Train on cutoff 2025-06-01 (label = June activity), then score members at 2025-07-01."""
    data = build_features("2025-06-01", with_label=True)
    X, y = data[NUMERIC + CATEGORICAL], data.churned
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # The Pipeline learns scaling/encoding from TRAIN rows only -> no leakage into the test set.
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    candidates = [
        ("logistic", LogisticRegression(max_iter=1000), {"model__C": [0.01, 0.1, 1, 10]}),
        ("forest", RandomForestClassifier(n_estimators=150, random_state=42),
         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10]})]
    results, fitted = [], {}
    for name, est, grid in candidates:
        search = GridSearchCV(Pipeline([("prep", prep), ("model", est)]), grid, cv=5, scoring="f1")
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        fitted[name] = search
        results.append({"model": name, "best_params": search.best_params_,
                        "precision": round(precision_score(y_test, pred), 3),
                        "recall": round(recall_score(y_test, pred), 3),
                        "f1": round(f1_score(y_test, pred), 3),
                        "confusion_matrix": confusion_matrix(y_test, pred).tolist()})
    results = pd.DataFrame(results)
    best_name = results.sort_values("f1", ascending=False).model.iloc[0]
    best = fitted[best_name]

    OUT.mkdir(exist_ok=True)
    joblib.dump(best, OUT / "churn_model.joblib")
    results.to_csv(OUT / "churn_model_results.csv", index=False)

    # Use the saved model on TODAY's data (cutoff 2025-07-01, no label yet) -> who is at risk in July?
    today = build_features("2025-07-01", with_label=False)
    today["churn_probability"] = best.predict_proba(today[NUMERIC + CATEGORICAL])[:, 1].round(3)
    watch = today.sort_values("churn_probability", ascending=False)
    watch[["member_id", "plan", "city", "visits_last_30d", "days_since_last_visit",
           "churn_probability"]].head(25).to_csv(OUT / "retention_watchlist.csv", index=False)

    summary = {"members_modelled": len(data), "churn_rate": round(float(y.mean()), 3),
               "baseline_accuracy_always_predict_stay": round(accuracy_score(y_test, np.zeros(len(y_test))), 3),
               "best_model": best_name, "best_model_accuracy": round(accuracy_score(y_test, best.predict(X_test)), 3)}
    return summary, results, watch.head(10)


if __name__ == "__main__":
    seg, diag, profile, var = member_segments(k=4)
    print("K-MEANS DIAGNOSTICS (higher silhouette = better separated)\n", diag.to_string(index=False))
    print("\nSEGMENT PROFILE\n", profile.to_string(index=False))
    print("\nPCA explained variance (2 components):", var, "total:", round(float(var.sum()), 3))
    summary, results, top = build_churn_model()
    print("\nCHURN MODEL SUMMARY\n", summary)
    print("\nMODEL COMPARISON\n", results.to_string(index=False))
    print("\nTOP 10 AT-RISK MEMBERS FOR JULY\n", top[["member_id", "plan", "city", "visits_last_30d",
                                                       "days_since_last_visit", "churn_probability"]].to_string(index=False))
