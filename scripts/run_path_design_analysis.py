#!/usr/bin/env python3
"""Run P1 Path Design Analysis (analysis-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.path_design_analysis import (
    format_path_design_summary,
    run_and_print,
    save_path_design_report,
)
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 Path Design Analysis")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/path_design_analysis_report.json"),
    )
    args = parser.parse_args()
    md = load_ohlcv_csv(str(args.csv), symbol="BTCUSDT")
    print(f"Loaded {md.num_bars} bars from {md.source}")
    report = run_and_print(md)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_path_design_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
