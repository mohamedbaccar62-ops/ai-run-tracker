"""Train a Random Forest churn model with MLflow tracking.

train.py only trains and logs to MLflow. It never touches SQLite.
Call run_training() directly (used by run_pipeline.py) or use the CLI.
Returns a dict with mlflow_run_id, status, dq_status, metrics, error_message.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TRACKING_URI = os.path.join(ROOT, "mlruns")
EXPERIMENT = "churn-run-tracker"

DEFAULT_CONFIG = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "test_size": 0.2,
    "stratify": True,
}

FEATURES = [
    "subscription_type",
    "tenure_months",
    "attendance_rate",
    "payment_amount",
    "last_login_days_ago",
    "support_calls",
]
TARGET = "churned"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_config(path: str | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            config.update(json.load(f))
    return config


def _file_hash(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_training(
    model_version: str,
    config: dict | None = None,
    data_path: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Train one model version and log to MLflow.

    Returns a dict with keys: mlflow_run_id, status, dq_status, metrics,
    error_message, duration_seconds, data_hash.
    """
    config = dict(config or DEFAULT_CONFIG)
    data_path = data_path or os.path.join(ROOT, "data", "customers.csv")
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI
    experiment_name = experiment_name or EXPERIMENT
    start = time.time()
    result: dict = {
        "mlflow_run_id": None,
        "status": "FAILED",
        "dq_status": None,
        "dq_warning_count": 0,
        "metrics": {},
        "error_message": None,
        "duration_seconds": 0.0,
        "data_hash": None,
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # --- data quality check (Phase 2) ---
    from ai_run_tracker.quality.check import check_quality, report_to_markdown

    df = pd.read_csv(data_path)
    report, clean = check_quality(df)
    data_hash = _file_hash(data_path)
    result["data_hash"] = data_hash
    dq_status = (
        "blocked"
        if report["blocked"]
        else ("passed_with_warnings" if report["rows_removed"] > 0 else "passed")
    )
    result["dq_status"] = dq_status
    dq_warning_count = sum(
        1 for r in report["rules"].values() if r["level"] == "warning" and r["violations"] > 0
    )
    result["dq_warning_count"] = dq_warning_count

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "quality_report.json"), "w") as f:
        f.write(json.dumps(report, indent=2))
    with open(os.path.join(ROOT, "reports", "quality_report.md"), "w") as f:
        f.write(report_to_markdown(report))

    try:
        with mlflow.start_run() as run:
            result["mlflow_run_id"] = run.info.run_id

            # --- tags ---
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("data_hash", data_hash)
            mlflow.set_tag("dq_status", dq_status)
            mlflow.set_tag("dq_warning_count", dq_warning_count)
            mlflow.set_tag("git_commit", _git_commit())

            # --- params ---
            for key in ("n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state"):
                mlflow.log_param(key, config[key])
            mlflow.log_param("test_size", config["test_size"])
            mlflow.log_param("train_size", 1 - config["test_size"])

            duration = time.time() - start
            result["duration_seconds"] = duration

            # --- blocked run: still log, mark FAILED ---
            if report["blocked"]:
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_artifact(os.path.join(ROOT, "reports", "quality_report.json"))
                mlflow.log_artifact(os.path.join(ROOT, "reports", "quality_report.md"))
                mlflow.end_run(status="FAILED")
                result["status"] = "BLOCKED"
                result["error_message"] = "; ".join(report["block_reasons"])
                print(f"RUN {model_version}: BLOCKED by data quality, logged as FAILED")
                return result

            # --- train ---
            clean = clean.dropna(subset=FEATURES + [TARGET])
            X = clean[FEATURES]
            y = clean[TARGET]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config["test_size"],
                random_state=config["random_state"],
                stratify=y if config["stratify"] else None,
            )

            preprocess = ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["subscription_type"]),
                    (
                        "num",
                        SimpleImputer(strategy="median"),
                        [f for f in FEATURES if f != "subscription_type"],
                    ),
                ]
            )
            clf = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                class_weight=config["class_weight"],
                random_state=config["random_state"],
            )
            model = Pipeline([("preprocess", preprocess), ("clf", clf)])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "duration_seconds": duration,
            }
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            result["metrics"] = metrics

            # --- artifacts ---
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(os.path.join(ROOT, "reports", "quality_report.json"))
            mlflow.log_artifact(os.path.join(ROOT, "reports", "quality_report.md"))

            mlflow.end_run(status="FINISHED")
            result["status"] = "SUCCESS"
            print(
                f"RUN {model_version}: accuracy={metrics['accuracy']:.3f} "
                f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
                f"duration={duration:.2f}s"
            )
    except Exception as exc:  # noqa: BLE001 - record crash in MLflow + result
        try:
            mlflow.end_run(status="FAILED")
        except Exception:  # noqa: BLE001
            pass
        result["status"] = "FAILED"
        result["error_message"] = str(exc)
        result["duration_seconds"] = time.time() - start
        print(f"RUN {model_version}: CRASHED - {exc}")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train churn model with MLflow tracking")
    parser.add_argument("--model-version", required=True, help="e.g. v1, v2")
    parser.add_argument("--config", default=None, help="JSON config overriding defaults")
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "customers.csv"))
    parser.add_argument("--tracking-uri", default=TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument("--experiment", default=EXPERIMENT, help="MLflow experiment name")
    args = parser.parse_args()

    config = _load_config(args.config)
    result = run_training(
        args.model_version,
        config,
        args.data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )

    if result["status"] == "BLOCKED":
        return 2
    if result["status"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
