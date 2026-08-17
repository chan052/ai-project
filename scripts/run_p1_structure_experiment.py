#!/usr/bin/env python3
"""Run P1 structure comparison: Baseline P+U-MAE vs S+D+U-MAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.p1_structure_experiment import format_summary_table, run_and_print, save_report
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 structure comparison experiment")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/p1_structure_experiment_report.json"),
    )
    args = parser.parse_args()
    market_data = load_ohlcv_csv(args.csv, symbol="BTCUSDT")
    print(f"Loaded {market_data.num_bars} bars from {market_data.source}")
    report = run_and_print(market_data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
