"""Weekly report: trends and recurring issues for the last 7 days vs the
previous 7 days.

Reads runs.db (read-only) and writes markdown to
reports/weekly/weekly_YYYY-MM-DD.md. Markdown only -- no new dependencies.
Optional --send posts a WEEKLY_REPORT summary to the n8n webhook.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT, "runs.db")
OUT_DIR = os.path.join(ROOT, "reports", "weekly")

FAILURE_RATE_THRESHOLD = 0.5  # "needs attention" if failure rate exceeds this


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _window_rows(conn: sqlite3.Connection, start: datetime, end: datetime):
    return conn.execute(
        "SELECT * FROM runs WHERE timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp",
        (start.isoformat(), end.isoformat()),
    ).fetchall()


def _alerts_in_window(conn: sqlite3.Connection, start: datetime, end: datetime, event_type: str | None = None):
    if event_type:
        return conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE event_type = ? AND timestamp >= ? AND timestamp < ?",
            (event_type, start.isoformat(), end.isoformat()),
        ).fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= ? AND timestamp < ?",
        (start.isoformat(), end.isoformat()),
    ).fetchone()[0]


def _metrics(rows, conn, start, end):
    total = len(rows)
    failures = sum(1 for r in rows if r["status"] != "SUCCESS")
    failure_rate = (failures / total) if total else 0.0

    # slow runs: duration > 2x median of last 20 successful runs (fallback 10s)
    dur_rows = conn.execute(
        "SELECT duration_seconds FROM runs WHERE status='SUCCESS' AND duration_seconds IS NOT NULL "
        "ORDER BY timestamp DESC, rowid DESC LIMIT 20"
    ).fetchall()
    durations = sorted([r[0] for r in dur_rows if r[0] is not None])
    threshold = (durations[len(durations) // 2] * 2) if len(durations) >= 20 else 10.0
    slow = sum(1 for r in rows if r["duration_seconds"] is not None and r["duration_seconds"] > threshold)

    warning_counts = [r["dq_warning_count"] for r in rows if r["dq_warning_count"] is not None]
    avg_warnings = (sum(warning_counts) / len(warning_counts)) if warning_counts else 0.0

    runs_per_day: dict[str, int] = {}
    for r in rows:
        day = _parse_ts(r["timestamp"]).date().isoformat()
        runs_per_day[day] = runs_per_day.get(day, 0) + 1

    return {
        "total": total,
        "failures": failures,
        "failure_rate": failure_rate,
        "slow": slow,
        "avg_warnings": avg_warnings,
        "runs_per_day": runs_per_day,
        "threshold": threshold,
    }


def _direction(cur: float, prev: float) -> str:
    if cur > prev:
        return "up"
    if cur < prev:
        return "down"
    return "same"


def _top_messages(rows, status_filter: str | None, limit: int = 5):
    msgs = Counter()
    for r in rows:
        if status_filter and r["status"] != status_filter:
            continue
        msg = r["error_message"] or "(no message)"
        msgs[msg] += 1
    return msgs.most_common(limit)


def build_report(db_path: str) -> str:
    conn = _connect(db_path)
    try:
        now = datetime.now(timezone.utc)
        end = now
        start = now - timedelta(days=7)
        prev_end = start
        prev_start = start - timedelta(days=7)

        cur_rows = _window_rows(conn, start, end)
        prev_rows = _window_rows(conn, prev_start, prev_end)

        cur = _metrics(cur_rows, conn, start, end)
        prev = _metrics(prev_rows, conn, prev_start, prev_end)

        repeated_failures = _alerts_in_window(conn, start, end, "REPEATED_FAILURE")
    finally:
        conn.close()

    lines: list[str] = []
    stamp = start.date().isoformat()
    lines.append(f"# Weekly AI Run Report — week of {stamp}")
    lines.append(f"Window: last 7 days ({start.date()} to {end.date()}), vs previous 7 days.")
    lines.append("")

    # --- Needs attention ---
    lines.append("## Needs attention")
    attention = []
    if cur["failure_rate"] > FAILURE_RATE_THRESHOLD:
        attention.append(
            f"- Failure rate {cur['failure_rate']:.0%} exceeds {FAILURE_RATE_THRESHOLD:.0%} threshold."
        )
    if cur["slow"] > 0:
        attention.append(f"- {cur['slow']} slow run(s) in the window.")
    if repeated_failures > 0:
        attention.append(f"- {repeated_failures} REPEATED_FAILURE alert(s) in the window.")
    if not attention:
        lines.append("- None.")
    else:
        lines.extend(attention)
    lines.append("")

    # --- Trends ---
    lines.append("## Trends (last 7 days vs previous 7 days)")
    lines.append(f"- Total runs: {cur['total']} ({_direction(cur['total'], prev['total'])} vs {prev['total']})")
    lines.append(
        f"- Failure rate (FAILED + BLOCKED): {cur['failure_rate']:.1%} "
        f"({_direction(cur['failure_rate'], prev['failure_rate'])} vs {prev['failure_rate']:.1%})"
    )
    lines.append(
        f"- Slow runs (>{cur['threshold']:.0f}s): {cur['slow']} "
        f"({_direction(cur['slow'], prev['slow'])} vs {prev['slow']})"
    )
    lines.append(
        f"- Avg data-quality warnings per run: {cur['avg_warnings']:.2f} "
        f"({_direction(cur['avg_warnings'], prev['avg_warnings'])} vs {prev['avg_warnings']:.2f})"
    )
    lines.append("")
    if cur["runs_per_day"]:
        lines.append("### Runs per day")
        for day in sorted(cur["runs_per_day"]):
            lines.append(f"- {day}: {cur['runs_per_day'][day]}")
        lines.append("")

    # --- Recurring issues ---
    lines.append("## Recurring issues")
    lines.append(f"- REPEATED_FAILURE alerts: {repeated_failures}")

    blocked = [r for r in cur_rows if r["status"] == "BLOCKED"]
    if blocked:
        lines.append("- Top block reasons (blocked runs):")
        for msg, count in _top_messages(blocked, None):
            lines.append(f"  - {count}x {msg}")
    else:
        lines.append("- Top block reasons: none.")

    failed = [r for r in cur_rows if r["status"] == "FAILED"]
    if failed:
        lines.append("- Top error messages (failed runs):")
        for msg, count in _top_messages(failed, "FAILED"):
            lines.append(f"  - {count}x {msg}")
    else:
        lines.append("- Top error messages: none.")
    lines.append("")

    if cur["total"] == 0:
        lines.append("_No runs recorded in this window._")

    summary = {
        "total_runs": cur["total"],
        "failures": cur["failures"],
        "failure_rate": cur["failure_rate"],
        "slow_runs": cur["slow"],
        "repeated_failures": repeated_failures,
        "avg_warnings": cur["avg_warnings"],
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
    }
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly AI run report (markdown)")
    parser.add_argument("--db", default=DB_PATH, help="SQLite runs database path")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    parser.add_argument("--send", action="store_true", help="POST summary to n8n webhook")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Missing database: {args.db}")
        return 1

    report, summary = build_report(args.db)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"weekly_{datetime.now(timezone.utc).date().isoformat()}.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")

    if args.send:
        import requests

        from ai_run_tracker.alerts.alerter import webhook_url

        url = webhook_url()
        payload = {"event_type": "WEEKLY_REPORT", "report": out_path, **summary}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            print(f"[weekly] webhook HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"[weekly] webhook unreachable ({exc}), continuing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
