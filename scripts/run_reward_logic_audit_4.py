#!/usr/bin/env python3
"""Run Reward Logic Audit 4: f_n time-profile + MAE decomposition (analysis-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.reward_logic_audit_4 import (
    format_audit4_summary,
    run_and_print,
    save_audit4_report,
)
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Reward Logic Audit 4")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/reward_logic_audit_4_report.json"),
    )
    args = parser.parse_args()
    market_data = load_ohlcv_csv(args.csv, symbol="BTCUSDT")
    print(f"Loaded {market_data.num_bars} bars from {market_data.source}")
    report = run_and_print(market_data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_audit4_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
