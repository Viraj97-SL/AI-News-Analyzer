"""Print a drift report over recorded research-pipeline runs.

Usage:
    python -m scripts.research_drift_report [--path output/metrics/research_runs.jsonl]
                                             [--baseline 8] [--recent 4]

Exit code 1 when drift is detected, so it can gate a CI/cron step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.drift import detect_drift, format_report
from app.core.run_metrics import DEFAULT_METRICS_PATH, load_run_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--baseline", type=int, default=8)
    parser.add_argument("--recent", type=int, default=4)
    args = parser.parse_args(argv)

    if args.baseline < 1 or args.recent < 1:
        parser.error("--baseline and --recent must be >= 1")

    records = load_run_metrics(args.path)
    alerts = detect_drift(records, baseline_window=args.baseline, recent_window=args.recent)
    sys.stdout.write(format_report(alerts, n_records=len(records)) + "\n")
    if len(records) < args.baseline + args.recent:
        sys.stdout.write(f"(need {args.baseline + args.recent} runs for a comparison)\n")
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
