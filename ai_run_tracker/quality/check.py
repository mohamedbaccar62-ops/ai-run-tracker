"""Data quality rules for AI Run Tracker.

check_quality(df) validates a raw customer dataframe and returns
(report, clean_df). Critical rules block a run on any single failure;
warning rows are dropped below a row-fraction threshold, otherwise they
also block. This keeps the design explicit for interview explainability.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd

# Fraction of bad rows above which a warning rule also blocks the run.
WARNING_BLOCK_THRESHOLD = 0.05

CRITICAL_RULES = ["missing_id", "duplicate_id"]
WARNING_RULES = ["negative_amount", "future_date", "invalid_attendance"]


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]


def check_quality(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Run all quality rules on df.

    Returns (report, clean_df) where clean_df has offending rows removed
    for warning rules below the blocking threshold.
    """
    report: dict = {
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "input_rows": int(len(df)),
        "input_hash": _hash(df),
        "rules": {},
        "rows_removed": 0,
        "blocked": False,
        "block_reasons": [],
    }

    clean = df.copy()

    # --- critical: missing id ---
    missing_id = clean["customer_id"].isna()
    n_missing = int(missing_id.sum())
    report["rules"]["missing_id"] = {
        "level": "critical",
        "violations": n_missing,
        "blocking": bool(n_missing > 0),
    }
    if n_missing > 0:
        report["blocked"] = True
        report["block_reasons"].append(f"missing_id: {n_missing} rows")
        clean = clean[~missing_id]

    # --- critical: duplicate id ---
    dup_mask = clean["customer_id"].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    report["rules"]["duplicate_id"] = {
        "level": "critical",
        "violations": n_dup,
        "blocking": bool(n_dup > 0),
    }
    if n_dup > 0:
        report["blocked"] = True
        report["block_reasons"].append(f"duplicate_id: {n_dup} rows")
        clean = clean[~dup_mask]

    # --- warning: negative amount ---
    neg_mask = clean["payment_amount"].astype(float) < 0
    n_neg = int(neg_mask.sum())
    neg_blocking = (n_neg / len(clean)) > WARNING_BLOCK_THRESHOLD if len(clean) else False
    report["rules"]["negative_amount"] = {
        "level": "warning",
        "violations": n_neg,
        "blocking": bool(neg_blocking),
    }
    if neg_blocking:
        report["blocked"] = True
        report["block_reasons"].append(f"negative_amount: {n_neg} rows (>5%)")
    else:
        clean = clean[~neg_mask]
        report["rows_removed"] += n_neg

    # --- warning: future date (negative days ago) ---
    future_mask = clean["last_login_days_ago"].astype(float) < 0
    n_future = int(future_mask.sum())
    future_blocking = (n_future / len(clean)) > WARNING_BLOCK_THRESHOLD if len(clean) else False
    report["rules"]["future_date"] = {
        "level": "warning",
        "violations": n_future,
        "blocking": bool(future_blocking),
    }
    if future_blocking:
        report["blocked"] = True
        report["block_reasons"].append(f"future_date: {n_future} rows (>5%)")
    else:
        clean = clean[~future_mask]
        report["rows_removed"] += n_future

    # --- warning: invalid attendance ---
    bad_att = clean["attendance_rate"].astype(float)
    att_mask = (bad_att < 0) | (bad_att > 1)
    n_att = int(att_mask.sum())
    report["rules"]["invalid_attendance"] = {
        "level": "warning",
        "violations": n_att,
        "blocking": False,
    }
    clean = clean[~att_mask]
    report["rows_removed"] += n_att

    report["output_rows"] = int(len(clean))
    report["clean_hash"] = _hash(clean)
    return report, clean


def report_to_markdown(report: dict) -> str:
    lines = [
        "# Data Quality Report",
        f"Checked at: {report['checked_at']}",
        f"Input rows: {report['input_rows']}  |  Output rows: {report['output_rows']}",
        f"Input hash: {report['input_hash']}  |  Clean hash: {report['clean_hash']}",
        f"Rows removed: {report['rows_removed']}",
        f"Blocked: **{'YES' if report['blocked'] else 'NO'}**",
        "",
        "## Rules",
    ]
    if report["block_reasons"]:
        lines += ["", "### Block reasons", *[f"- {r}" for r in report["block_reasons"]]]
    for name, rule in report["rules"].items():
        lines.append(
            f"- {name}: {rule['violations']} violations ({rule['level']}, "
            f"blocking={rule['blocking']})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    sys.path.insert(0, ".")
    from ai_run_tracker.data.generate import generate

    df = generate()
    report, clean = check_quality(df)
    print(json.dumps(report, indent=2))
    print(report_to_markdown(report))
