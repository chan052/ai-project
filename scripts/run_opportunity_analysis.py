#!/usr/bin/env python3
"""Run Case A/B + Opportunity Analysis experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.opportunity_analysis import OpportunityAnalysisRunner, save_report
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/market/btcusdt_3m.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/opportunity_analysis_report.json"))
    args = parser.parse_args()
    data = load_ohlcv_csv(args.csv, symbol="BTCUSDT")
    print(f"Loaded {data.num_bars} bars")
    report = OpportunityAnalysisRunner(data).run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(args.report))
    cab = report["case_ab_analysis"]
    print(f"Case A count: {cab['case_a_dip_then_rise']['count']}")
    print(f"Case B count: {cab['case_b_rise_then_fall']['count']}")
    print(f"F baseline A vs B: {cab['case_a_dip_then_rise'].get('mean_f_baseline')} vs {cab['case_b_rise_then_fall'].get('mean_f_baseline')}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
