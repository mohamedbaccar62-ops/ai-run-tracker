"""Alerting via n8n webhook.

Reads N8N_WEBHOOK_URL env var (default http://localhost:5678/webhook/pipeline-alert).
A failed webhook call is logged as a warning and never crashes or changes
the status of a pipeline run. HTTP calls use a 5-second timeout.
"""

from __future__ import annotations

import os
import sqlite3

import requests

from ai_run_tracker.runs_db import insert_alert

DEFAULT_WEBHOOK_URL = "http://localhost:5678/webhook/pipeline-alert"
TIMEOUT_SECONDS = 5
SLOW_FALLBACK_SECONDS = 10.0
REPEATED_FAILURE_WINDOW = 10
REPEATED_FAILURE_THRESHOLD = 3


def webhook_url() -> str:
    return os.environ.get("N8N_WEBHOOK_URL", DEFAULT_WEBHOOK_URL)


def send_alert(event_type: str, payload: dict) -> None:
    """POST an alert to the n8n webhook. Never raises."""
    url = webhook_url()
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            print(f"[alert] {event_type}: webhook HTTP {resp.status_code}, continuing")
        else:
            print(f"[alert] {event_type}: sent to n8n")
    except Exception as exc:  # noqa: BLE001 - alert failure must not crash pipeline
        print(f"[alert] {event_type}: webhook unreachable ({exc}), continuing")


def _recent_rows(conn: sqlite3.Connection, limit: int):
    return conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC, rowid DESC LIMIT ?", (limit,)
    ).fetchall()


def _successful_durations(conn: sqlite3.Connection, limit: int = 20):
    rows = conn.execute(
        "SELECT duration_seconds FROM runs WHERE status='SUCCESS' "
        "AND duration_seconds IS NOT NULL "
        "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def check_and_alert(conn: sqlite3.Connection, run_info: dict) -> None:
    """Inspect the just-recorded run and send any applicable alerts.

    run_info keys: run_id, timestamp, model_version, status, dq_status,
    duration_seconds, error_message.
    """
    status = run_info["status"]
    duration = run_info["duration_seconds"] or 0.0

    base_payload = {
        "event_type": None,
        "run_id": run_info["run_id"],
        "timestamp": run_info["timestamp"],
        "status": status,
        "model_version": run_info["model_version"],
        "duration_seconds": duration,
        "error_message": run_info.get("error_message"),
        "dq_status": run_info.get("dq_status"),
    }

    alerts_to_send: list[tuple[str, dict]] = []

    # --- BLOCKED / FAILED ---
    if status == "BLOCKED":
        alerts_to_send.append(("BLOCKED", {**base_payload, "event_type": "BLOCKED"}))
    elif status == "FAILED":
        alerts_to_send.append(("FAILED", {**base_payload, "event_type": "FAILED"}))

    # --- SLOW: duration > 2x median of last 20 successful (fallback 10s) ---
    durations = _successful_durations(conn)
    if len(durations) >= 20:
        threshold = sorted(durations)[len(durations) // 2] * 2
    else:
        threshold = SLOW_FALLBACK_SECONDS
    if duration > threshold:
        alerts_to_send.append(
            ("SLOW", {**base_payload, "event_type": "SLOW", "threshold_seconds": threshold})
        )

    # --- REPEATED_FAILURE: 3 failures in last 10 runs, alert once per crossing ---
    recent = _recent_rows(conn, REPEATED_FAILURE_WINDOW)
    prior_failures = sum(1 for r in recent[1:] if r[3] != "SUCCESS")
    is_failure = status != "SUCCESS"
    total_failures = prior_failures + (1 if is_failure else 0)
    if (
        is_failure
        and total_failures >= REPEATED_FAILURE_THRESHOLD
        and prior_failures < REPEATED_FAILURE_THRESHOLD
    ):
        alerts_to_send.append(
            (
                "REPEATED_FAILURE",
                {
                    **base_payload,
                    "event_type": "REPEATED_FAILURE",
                    "recent_failure_count": total_failures,
                },
            )
        )

    # Send + persist every alert. Alert failures never crash the pipeline.
    for event_type, payload in alerts_to_send:
        send_alert(event_type, payload)
        try:
            insert_alert(
                conn,
                event_type=event_type,
                run_id=run_info["run_id"],
                timestamp=run_info["timestamp"],
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[alert] failed to persist {event_type} ({exc}), continuing")
