#!/usr/bin/env python3
"""Run Dual-axis (Immediate vs Deferred) opportunity labeling experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chartai.analysis.dual_axis_analysis import (
    DualAxisAnalysisRunner,
    generate_plots,
    save_report,
)
from chartai.data.market_data import load_ohlcv_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/market/btcusdt_3m.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/dual_axis_analysis_report.json"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures/dual_axis"))
    args = parser.parse_args()

    data = load_ohlcv_csv(args.csv, symbol="BTCUSDT")
    print(f"Loaded {data.num_bars} bars")
    report = DualAxisAnalysisRunner(data).run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(args.report))

    plots = generate_plots(report, str(args.figures))
    id_corr = report["id_correlation"]
    q3 = report["quadrant_analysis"]["Q3_Ilow_Dhigh_fraction"]
    case_a = report["case_ab_analysis"]["case_a_dip_then_rise"]
    case_b = report["case_ab_analysis"]["case_b_rise_then_fall"]

    print(f"Eval samples: {report['eval_samples']}")
    print(f"Pearson I-D: {id_corr['pearson_I_D']:.4f}")
    print(f"Spearman I-D: {id_corr['spearman_I_D']:.4f}")
    print(f"Q3 (I-low/D-high) fraction: {q3:.1%}")
    print(f"Case A I={case_a.get('mean_I', float('nan')):.3f} D={case_a.get('mean_D', float('nan')):.6f}")
    print(f"Case B I={case_b.get('mean_I', float('nan')):.3f} D={case_b.get('mean_D', float('nan')):.6f}")
    print(f"Report: {args.report}")
    if plots:
        print(f"Figures: {', '.join(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
