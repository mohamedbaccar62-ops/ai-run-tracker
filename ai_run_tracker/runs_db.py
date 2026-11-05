"""SQLite store for ML run records and alerts.

One row per run: run_id, timestamp, model_version, status, dq_status,
dq_warning_count, duration_seconds, error_message, mlflow_run_id. Flat
columns per the recorded design decision (no JSON blob).

An `alerts` table records every alert sent to n8n so the weekly report
can count REPEATED_FAILURE alerts and other alert history.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs.db")

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL,
    dq_status TEXT,
    dq_warning_count INTEGER,
    duration_seconds REAL,
    error_message TEXT,
    mlflow_run_id TEXT
)
"""

ALTER_RUNS_ADD_DQ_WARNING = (
    "ALTER TABLE runs ADD COLUMN dq_warning_count INTEGER"
)

CREATE_ALERTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT
)
"""

INSERT_RUN_SQL = """
INSERT INTO runs (run_id, timestamp, model_version, status, dq_status,
                  dq_warning_count, duration_seconds, error_message, mlflow_run_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_ALERT_SQL = """
INSERT INTO alerts (event_type, run_id, timestamp, payload)
VALUES (?, ?, ?, ?)
"""


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_RUNS_TABLE_SQL)
    conn.execute(CREATE_ALERTS_TABLE_SQL)
    # Backfill dq_warning_count column on pre-existing databases.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    if "dq_warning_count" not in cols:
        conn.execute(ALTER_RUNS_ADD_DQ_WARNING)
    conn.commit()
    return conn


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    timestamp: str,
    model_version: str,
    status: str,
    dq_status: str | None,
    dq_warning_count: int | None,
    duration_seconds: float | None,
    error_message: str | None,
    mlflow_run_id: str | None,
) -> None:
    conn.execute(
        INSERT_RUN_SQL,
        (
            run_id,
            timestamp,
            model_version,
            status,
            dq_status,
            dq_warning_count,
            duration_seconds,
            error_message,
            mlflow_run_id,
        ),
    )
    conn.commit()


def insert_alert(
    conn: sqlite3.Connection,
    event_type: str,
    run_id: str,
    timestamp: str,
    payload: dict,
) -> None:
    import json

    conn.execute(
        INSERT_ALERT_SQL,
        (event_type, run_id, timestamp, json.dumps(payload)),
    )
    conn.commit()


def init_db(db_path: str = DEFAULT_DB) -> None:
    connect(db_path)


if __name__ == "__main__":  # pragma: no cover
    init_db()
    print(f"Initialized {DEFAULT_DB}")
