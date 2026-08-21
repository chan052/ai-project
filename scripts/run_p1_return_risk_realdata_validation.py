#!/usr/bin/env python3
"""Run P1 Return/Risk Target Real-Data Validation (analysis-only)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.p1_return_risk_realdata_validation import (
    P1ReturnRiskRealDataValidationRunner,
    format_realdata_summary,
    save_realdata_report,
)
from chartai.data.market_data import load_ohlcv_csv


def _run_pytest_count() -> int | None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        combined = proc.stdout + proc.stderr
        m = re.search(r"(\d+) passed", combined)
        if m:
            return int(m.group(1))
        return None
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 Return/Risk Real-Data Validation")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/p1_return_risk_realdata_validation_report.json"),
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip full pytest (test count will be null in report)",
    )
    args = parser.parse_args()
    btc = load_ohlcv_csv(str(args.csv), symbol="BTCUSDT")
    print(f"Loaded BTC {btc.num_bars} bars from {btc.source}")

    pass_count = None if args.skip_pytest else _run_pytest_count()
    if pass_count is not None:
        print(f"pytest: {pass_count} passed")

    report = P1ReturnRiskRealDataValidationRunner(btc).run(test_pass_count=pass_count)
    print(format_realdata_summary(report))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_realdata_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
