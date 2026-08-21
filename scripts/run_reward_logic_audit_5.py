#!/usr/bin/env python3
"""Run Reward Logic Audit 5 — U persistence / MAE early-path (analysis-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.reward_logic_audit_5 import (
    format_audit5_summary,
    run_and_print,
    save_audit5_report,
)
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Reward Logic Audit 5")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/reward_logic_audit_5_report.json"),
    )
    args = parser.parse_args()
    md = load_ohlcv_csv(str(args.csv), symbol="BTCUSDT")
    print(f"Loaded {md.num_bars} bars from {md.source}")
    report = run_and_print(md)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_audit5_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
