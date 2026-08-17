#!/usr/bin/env python3
"""Run P1 Path variant comparison on real 3m market data.

Usage:
  python scripts/run_path_experiment.py --source binance --symbol BTCUSDT
  python scripts/run_path_experiment.py --csv data/market/btcusdt_3m.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.path_experiment import format_comparison_table, run_and_print, save_report
from chartai.data.market_data import fetch_binance_klines_3m, load_ohlcv_csv, save_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 Path variant market experiment")
    parser.add_argument("--source", choices=("binance", "csv"), default="binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--csv", type=Path, help="Path to OHLCV CSV (with --source csv)")
    parser.add_argument("--max-bars", type=int, default=5000)
    parser.add_argument("--save-csv", type=Path, help="Optional path to cache fetched bars")
    parser.add_argument("--report", type=Path, default=Path("reports/path_experiment_report.json"))
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv:
            parser.error("--csv is required when --source csv")
        market_data = load_ohlcv_csv(args.csv, symbol=args.symbol)
    else:
        market_data = fetch_binance_klines_3m(args.symbol, max_bars=args.max_bars)
        if args.save_csv:
            save_ohlcv_csv(market_data.bars, args.save_csv)
            print(f"Cached bars to {args.save_csv}")

    print(f"Loaded {market_data.num_bars} bars from {market_data.source}")
    print(f"Period: {market_data.start_time} .. {market_data.end_time}")

    report = run_and_print(market_data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(args.report))
    print(f"\nFull report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
