#!/usr/bin/env bash
set -euo pipefail

# One-command demo: populates everything so a reviewer can see the full platform.
# Usage: ./demo.sh   (requires docker compose)

COMPOSE="docker compose"

echo "== Starting long-running services (mlflow, dashboard, n8n) =="
$COMPOSE up -d mlflow dashboard n8n

echo "== Waiting for services to become healthy =="
$COMPOSE wait mlflow dashboard n8n 2>/dev/null || true

echo "== Generate synthetic data =="
$COMPOSE run --rm pipeline python3 -m ai_run_tracker.data.generate

echo "== Run v1 on clean data =="
$COMPOSE run --rm pipeline --model-version v1 --data /app/data/customers.csv

echo "== Run v1 on the defective data (should be BLOCKED) =="
$COMPOSE run --rm pipeline --model-version v1 --data /app/data/customers.csv || true

echo "== Force a pipeline failure (missing data file) =="
$COMPOSE run --rm pipeline --model-version v1 --data /app/nonexistent.csv || true

echo "== Generate weekly report =="
$COMPOSE run --rm pipeline python3 -m ai_run_tracker.reports.weekly_report --db /app/runs.db --send

echo "== Demo complete =="
echo "MLflow UI:      http://localhost:5000"
echo "Dashboard:      http://localhost:8501"
echo "n8n UI:         http://localhost:5678"
