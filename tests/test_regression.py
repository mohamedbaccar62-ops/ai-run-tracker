"""Regression test: compare two model versions on the same data.

Trains v1 and v2 fresh via run_pipeline on a small fixed fixture dataset
using temporary MLflow tracking and SQLite paths (pytest tmp_path), so the
test never pollutes real runs or the dashboard.

Fails if v2's F1 or recall drops by more than REGRESSION_THRESHOLD
(relative) compared to v1. Both versions must run on identical data
(verified by identical data_hash) with the same seed/split semantics.
"""

from __future__ import annotations

import json

import pandas as pd

from ai_run_tracker.run_pipeline import run_pipeline

REGRESSION_THRESHOLD = 0.10  # fail if v2 drops >10% relative vs v1
METRICS_TO_CHECK = ("f1", "recall")

V1_CONFIG = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "test_size": 0.2,
    "stratify": True,
}

V2_CONFIG = {
    "n_estimators": 150,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "test_size": 0.2,
    "stratify": True,
}


def _make_fixture(tmp_path) -> str:
    """Small clean fixture dataset (~120 rows, no deliberate defects)."""
    import random

    rng = random.Random(7)
    rows = []
    for i in range(120):
        cid = f"FIX-{i:04d}"
        tenure = rng.randint(1, 60)
        attendance = round(rng.uniform(0, 1), 2)
        payment = round(rng.uniform(20, 200), 2)
        last_login = rng.randint(0, 365)
        support_calls = rng.randint(0, 10)
        churn_prob = 0.1 + 0.4 * (tenure < 6) + 0.3 * (attendance < 0.3) + 0.2 * (support_calls > 5)
        churned = int(rng.random() < min(churn_prob, 0.95))
        rows.append(
            {
                "customer_id": cid,
                "subscription_type": rng.choice(["monthly", "yearly"]),
                "tenure_months": tenure,
                "attendance_rate": attendance,
                "payment_amount": payment,
                "last_login_days_ago": last_login,
                "support_calls": support_calls,
                "churned": churned,
            }
        )
    path = tmp_path / "fixture.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_f1_recall_no_regression(tmp_path):
    fixture = _make_fixture(tmp_path)
    tracking_uri = str(tmp_path / "mlruns")
    db_path = str(tmp_path / "runs.db")

    r1 = run_pipeline("v1", V1_CONFIG, fixture, db_path=db_path, tracking_uri=tracking_uri)
    r2 = run_pipeline("v2", V2_CONFIG, fixture, db_path=db_path, tracking_uri=tracking_uri)

    assert r1["status"] == "SUCCESS", r1.get("error_message")
    assert r2["status"] == "SUCCESS", r2.get("error_message")

    # Same dataset -> identical data_hash
    assert r1["data_hash"] == r2["data_hash"], (
        f"data_hash mismatch: v1={r1['data_hash']} v2={r2['data_hash']}"
    )

    m1 = r1["metrics"]
    m2 = r2["metrics"]
    for metric in METRICS_TO_CHECK:
        v1 = m1[metric]
        v2 = m2[metric]
        floor = v1 * (1 - REGRESSION_THRESHOLD)
        assert v2 >= floor, (
            f"{metric} regression: v1={v1:.4f} v2={v2:.4f} "
            f"(floor={floor:.4f}, threshold={REGRESSION_THRESHOLD:.0%})"
        )
