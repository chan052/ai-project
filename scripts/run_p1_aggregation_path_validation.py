#!/usr/bin/env python3
"""Run P1 Return/Risk Aggregation & Path Validation (analysis-only)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.p1_aggregation_path_validation import (
    P1AggregationPathValidationRunner,
    format_aggregation_path_summary,
    save_p1_aggregation_path_validation_report,
)
from chartai.data.market_data import load_ohlcv_csv


def _pytest_pass_count() -> int | None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        m = re.search(r"(\d+) passed", proc.stdout + proc.stderr)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 Aggregation & Path Validation")
    parser.add_argument("--csv", type=Path, default=Path("data/market/btcusdt_3m.csv"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/p1_aggregation_path_validation_report.json"),
    )
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    btc = load_ohlcv_csv(str(args.csv), symbol="BTCUSDT")
    print(f"Loaded BTC {btc.num_bars} bars from {btc.source}")

    pass_count = None if args.skip_pytest else _pytest_pass_count()
    if pass_count is not None:
        print(f"pytest: {pass_count} passed")

    report = P1AggregationPathValidationRunner(btc).run(test_pass_count=pass_count)
    print(format_aggregation_path_summary(report))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_p1_aggregation_path_validation_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
