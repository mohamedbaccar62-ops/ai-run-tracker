"""Generate a synthetic customer churn dataset with deliberate defects.

Outputs data/customers.csv (~500 rows). Columns:
- customer_id: unique identifier (some rows deliberately missing or duplicated)
- subscription_type: monthly / yearly
- tenure_months: how long the customer has been subscribed
- attendance_rate: fraction of sessions attended (0..1, some invalid > 1)
- payment_amount: monthly payment (some negative / missing)
- last_login_days_ago: days since last login (some future / negative)
- support_calls: number of support calls
- churned: target, 0 or 1

Defects are intentional so Phase 2 (data quality) has something to flag.
No real personal data is used anywhere.
"""

from __future__ import annotations

import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(ROOT, "data", "customers.csv")

N_ROWS = 500
RNG_SEED = 42


def _defect_row(row: pd.Series, rng: random.Random) -> pd.Series:
    """Mutate a single row to introduce one deliberate defect."""
    defect = rng.choice(
        ["missing_id", "duplicate_id", "negative_amount", "future_date", "invalid_attendance", "clean"]
    )
    if defect == "missing_id":
        row["customer_id"] = None
    elif defect == "duplicate_id":
        row["customer_id"] = row["customer_id"]  # dup handled at frame level
    elif defect == "negative_amount":
        row["payment_amount"] = -abs(rng.uniform(10, 100))
    elif defect == "future_date":
        row["last_login_days_ago"] = -rng.randint(1, 30)
    elif defect == "invalid_attendance":
        row["attendance_rate"] = rng.uniform(1.01, 1.5)
    return row


def generate() -> pd.DataFrame:
    rng = random.Random(RNG_SEED)
    rows: list[dict] = []
    for i in range(N_ROWS):
        cid = f"CUST-{i:04d}"
        sub = rng.choice(["monthly", "yearly"])
        tenure = rng.randint(1, 60)
        attendance = round(rng.uniform(0, 1), 2)
        payment = round(rng.uniform(20, 200), 2)
        last_login = rng.randint(0, 365)
        support_calls = rng.randint(0, 10)
        # Simple heuristic target so the model has a signal to learn.
        churn_prob = 0.1 + 0.4 * (tenure < 6) + 0.3 * (attendance < 0.3) + 0.2 * (support_calls > 5)
        churned = int(rng.random() < min(churn_prob, 0.95))
        rows.append(
            {
                "customer_id": cid,
                "subscription_type": sub,
                "tenure_months": tenure,
                "attendance_rate": attendance,
                "payment_amount": payment,
                "last_login_days_ago": last_login,
                "support_calls": support_calls,
                "churned": churned,
            }
        )

    df = pd.DataFrame(rows)

    # --- inject defects ---
    missing_mask = [rng.random() < 0.06 for _ in range(N_ROWS)]
    df.loc[missing_mask, "customer_id"] = None

    dup_idx = df.sample(n=8, random_state=rng.randint(0, 10 ** 9)).index
    df.loc[dup_idx, "customer_id"] = df.loc[dup_idx, "customer_id"].shift(1)
    df.loc[dup_idx[0], "customer_id"] = df.loc[dup_idx[-1], "customer_id"]

    neg_mask = [rng.random() < 0.05 for _ in range(N_ROWS)]
    df.loc[neg_mask, "payment_amount"] = df.loc[neg_mask, "payment_amount"].apply(
        lambda v: -abs(v)
    )

    future_mask = [rng.random() < 0.04 for _ in range(N_ROWS)]
    df.loc[future_mask, "last_login_days_ago"] = df.loc[future_mask, "last_login_days_ago"].apply(
        lambda v: -rng.randint(1, 30)
    )

    bad_att_mask = [rng.random() < 0.04 for _ in range(N_ROWS)]
    df.loc[bad_att_mask, "attendance_rate"] = df.loc[bad_att_mask, "attendance_rate"].apply(
        lambda v: round(rng.uniform(1.01, 1.5), 2)
    )

    missing_pay_mask = [rng.random() < 0.05 for _ in range(N_ROWS)]
    df.loc[missing_pay_mask, "payment_amount"] = None

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT}")
    return df


if __name__ == "__main__":
    generate()
