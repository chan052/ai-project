#!/usr/bin/env python3
"""Run P1 Return/Risk/Direction Target Design Audit (analysis-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.p1_target_design_audit import (
    P1TargetDesignAuditRunner,
    format_target_design_summary,
    save_target_design_report,
)
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P1 Return/Risk/Direction Target Design Audit"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/p1_target_design_audit_report.json"),
    )
    parser.add_argument(
        "--synthetic-bars",
        type=int,
        default=3000,
        help="3m bars for SYNTH_LONG secondary dataset",
    )
    args = parser.parse_args()
    btc = load_ohlcv_csv(str(args.csv), symbol="BTCUSDT")
    print(f"Loaded BTC {btc.num_bars} bars from {btc.source}")
    runner = P1TargetDesignAuditRunner.from_btc_and_synthetic_long(
        btc, synthetic_3m_bars=args.synthetic_bars
    )
    report = runner.run()
    print(format_target_design_summary(report))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_target_design_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
