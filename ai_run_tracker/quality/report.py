"""CLI: run quality checks on data/customers.csv and write JSON + markdown reports."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from ai_run_tracker.quality.check import check_quality, report_to_markdown  # noqa: E402

INPUT = os.path.join(ROOT, "data", "customers.csv")
OUT_DIR = os.path.join(ROOT, "reports")
JSON_OUT = os.path.join(OUT_DIR, "quality_report.json")
MD_OUT = os.path.join(OUT_DIR, "quality_report.md")


def main() -> int:
    if not os.path.exists(INPUT):
        print(f"Missing input: {INPUT} (run `python -m ai_run_tracker.data.generate` first)")
        return 1
    df = pd.read_csv(INPUT)
    report, clean = check_quality(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        f.write(json.dumps(report, indent=2))
    with open(MD_OUT, "w") as f:
        f.write(report_to_markdown(report))
    print(f"Blocked={report['blocked']} rows_removed={report['rows_removed']}")
    print(f"Wrote {JSON_OUT} and {MD_OUT}")
    return 0 if not report["blocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
