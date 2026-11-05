"""Streamlit dashboard for AI Run Tracker & Monitor.

Reads run history read-only from SQLite runs.db and the latest data-quality
report from reports/. Never queries MLflow. Shows runs-per-day, run
status/failures, slow runs, and the latest quality report.

Refresh: st.cache_data(ttl=30) plus a "Refresh now" button that clears the
cache and reruns. A "last updated" timestamp is shown on the page.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(ROOT, "runs.db")
REPORTS_DIR = os.path.join(ROOT, "reports")
QUALITY_MD = os.path.join(REPORTS_DIR, "quality_report.md")

SLOW_FALLBACK_SECONDS = 10.0


@st.cache_data(ttl=30)
def load_runs(db_path: str) -> pd.DataFrame:
    """Load all runs read-only. Empty DataFrame if DB missing or unreadable."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp DESC", conn)
        conn.close()
    except (sqlite3.DatabaseError, sqlite3.OperationalError):
        return pd.DataFrame()
    return df


@st.cache_data(ttl=30)
def load_quality_report(md_path: str) -> str | None:
    """Return the latest quality report markdown, or None if missing."""
    if not os.path.exists(md_path):
        return None
    with open(md_path) as f:
        return f.read()


def slow_threshold(success_durations: pd.Series) -> float:
    """2x median of successful runs, or 10s fallback if fewer than 20."""
    if len(success_durations) >= 20:
        return float(success_durations.median() * 2)
    return SLOW_FALLBACK_SECONDS


def main() -> None:
    st.set_page_config(page_title="AI Run Tracker & Monitor", layout="wide")
    st.title("AI Run Tracker & Monitor")

    left, right = st.columns([3, 1])
    with left:
        st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with right:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    runs = load_runs(DB_PATH)
    if runs.empty:
        st.info(
            "No runs yet. Run `python3 -m ai_run_tracker.run_pipeline --model-version v1` "
            "to generate data."
        )
        return

    runs["timestamp"] = pd.to_datetime(runs["timestamp"])
    runs["date"] = runs["timestamp"].dt.date

    # --- Runs per day ---
    st.subheader("Runs per day")
    per_day = runs.groupby("date").size().reset_index(name="count")
    st.bar_chart(per_day.set_index("date"))

    # --- Status / failures ---
    st.subheader("Run status")
    st.bar_chart(runs["status"].value_counts())
    failures = runs[runs["status"] != "SUCCESS"]
    st.metric("Failed / blocked runs", int(len(failures)))

    # --- Slow runs ---
    st.subheader("Slow runs")
    success = runs[runs["status"] == "SUCCESS"]
    threshold = slow_threshold(success["duration_seconds"])
    slow = runs[runs["duration_seconds"] > threshold]
    st.caption(
        f"Slow threshold: >{threshold:.1f}s "
        f"(2x median of last 20 successful runs, or 10s fallback)"
    )
    if slow.empty:
        st.write("No slow runs.")
    else:
        st.dataframe(
            slow[["timestamp", "model_version", "duration_seconds", "status", "error_message"]],
            use_container_width=True,
        )

    # --- Latest data quality report ---
    st.subheader("Latest data quality report")
    md = load_quality_report(QUALITY_MD)
    if md:
        st.markdown(md)
    else:
        st.info("No quality report yet. Run the pipeline to generate `reports/quality_report.md`.")

    # --- Recent runs table ---
    st.subheader("Recent runs")
    st.dataframe(
        runs.head(20)[
            ["timestamp", "model_version", "status", "dq_status", "duration_seconds", "mlflow_run_id"]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
