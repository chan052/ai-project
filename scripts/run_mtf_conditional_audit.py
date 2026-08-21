#!/usr/bin/env python3
"""Run MTF Conditional Information Audit (Audit 5) — analysis-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.mtf_conditional_audit import (
    format_mtf_audit_summary,
    load_mtf_from_csv,
    run_and_print,
    save_mtf_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MTF conditional information audit")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/market/btcusdt_3m.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/mtf_conditional_audit_report.json"),
    )
    args = parser.parse_args()
    mtf_data = load_mtf_from_csv(str(args.csv))
    print(f"Loaded {mtf_data.num_bars} 3m bars (+ resampled 1H/4H) from {mtf_data.source}")
    report = run_and_print(mtf_data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_mtf_audit_report(report, str(args.report))
    print(f"\nFull report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
