"""Pipeline orchestrator: quality check -> train -> SQLite run record.

Exposes run_pipeline() so tests (and other callers) can drive it
programmatically with temporary MLflow/SQLite paths. main() is the CLI
wrapper around run_pipeline().
train.py never touches SQLite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load .env locally (no external dependency) so N8N_WEBHOOK_URL is available.
def _load_env() -> None:
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

from ai_run_tracker.runs_db import connect, insert_run  # noqa: E402
from ai_run_tracker.training.train import run_training  # noqa: E402
from ai_run_tracker.alerts.alerter import check_and_alert  # noqa: E402


def run_pipeline(
    model_version: str,
    config: dict | None = None,
    data_path: str | None = None,
    db_path: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "churn-run-tracker",
) -> dict:
    """Run one training pipeline iteration and record it in SQLite.

    Always writes one row to SQLite (in a finally block), even when the
    run is blocked or crashes. Returns the result dict from run_training.
    """
    data_path = data_path or os.path.join(ROOT, "data", "customers.csv")
    db_path = db_path or os.path.join(ROOT, "runs.db")
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    start = time.time()

    conn = connect(db_path)
    try:
        result = run_training(
            model_version,
            config,
            data_path,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )

        total_duration = time.time() - start
        duration = result.get("duration_seconds") or total_duration

        run_info = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_version": model_version,
            "status": result["status"],
            "dq_status": result.get("dq_status"),
            "dq_warning_count": result.get("dq_warning_count"),
            "duration_seconds": duration,
            "error_message": result.get("error_message"),
        }
        insert_run(
            conn,
            run_id=run_id,
            timestamp=timestamp,
            model_version=model_version,
            status=run_info["status"],
            dq_status=run_info["dq_status"],
            dq_warning_count=run_info["dq_warning_count"],
            duration_seconds=run_info["duration_seconds"],
            error_message=run_info["error_message"],
            mlflow_run_id=result.get("mlflow_run_id"),
        )
        # --- alerts: never let an alert problem change the run result ---
        try:
            check_and_alert(conn, run_info)
        except Exception as exc:  # noqa: BLE001
            print(f"[alert] check_and_alert raised ({exc}), continuing")
        return result
    except Exception as exc:  # noqa: BLE001 - still write a FAILED row
        total_duration = time.time() - start
        run_info = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_version": model_version,
            "status": "FAILED",
            "dq_status": None,
            "dq_warning_count": None,
            "duration_seconds": total_duration,
            "error_message": str(exc),
        }
        insert_run(
            conn,
            run_id=run_id,
            timestamp=timestamp,
            model_version=model_version,
            status="FAILED",
            dq_status=None,
            dq_warning_count=None,
            duration_seconds=total_duration,
            error_message=str(exc),
            mlflow_run_id=None,
        )
        try:
            check_and_alert(conn, run_info)
        except Exception as exc2:  # noqa: BLE001
            print(f"[alert] check_and_alert raised ({exc2}), continuing")
        return {
            "mlflow_run_id": None,
            "status": "FAILED",
            "dq_status": None,
            "metrics": {},
            "error_message": str(exc),
            "duration_seconds": total_duration,
            "data_hash": None,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full training pipeline with SQLite record")
    parser.add_argument("--model-version", required=True, help="e.g. v1, v2")
    parser.add_argument("--config", default=None, help="JSON config overriding defaults")
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "customers.csv"))
    parser.add_argument("--db", default=None, help="SQLite database path (default runs.db)")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    db_path = args.db or os.path.join(ROOT, "runs.db")
    result = run_pipeline(
        args.model_version,
        json.load(open(args.config)) if args.config else None,
        args.data,
        db_path=db_path,
        tracking_uri=args.tracking_uri,
    )

    print(f"Pipeline run: {result['status']} (dq={result.get('dq_status')})")
    if result.get("error_message"):
        print(f"  error: {result['error_message']}")
    if result.get("mlflow_run_id"):
        print(f"  mlflow_run_id: {result['mlflow_run_id']}")

    if result["status"] == "BLOCKED":
        return 2
    if result["status"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
