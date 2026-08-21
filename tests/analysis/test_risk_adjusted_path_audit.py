"""Tests for risk-adjusted path metrics audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.path_risk_adjusted_metrics import (
    compute_path_sharpe,
    compute_risk_adjusted_path_metrics,
    compute_ulcer_index,
)
from chartai.analysis.risk_adjusted_path_audit import RiskAdjustedPathAuditRunner
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditRunner, UMaeResidualAuditConfig
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def _market_source() -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=150, reward_horizon=10)
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_risk_adjusted_audit_runs() -> None:
    report = RiskAdjustedPathAuditRunner(_market_source()).run()
    assert "archetype_ABC_analysis" in report
    assert "final_judgments" in report
    assert len([a for a in report["archetype_ABC_analysis"] if "metrics" in a]) == 3


def test_ulcer_higher_on_drawdown_path() -> None:
    calm = (0.01, 0.01, 0.01, 0.01)
    crash = (0.03, -0.01, -0.03, -0.01)
    assert compute_ulcer_index(calm) < compute_ulcer_index(crash)


def test_sharpe_distinguishes_vol() -> None:
    low_vol = (0.01, 0.01, 0.01, 0.01)
    mixed = (0.03, -0.02, 0.02, -0.01)
    assert compute_path_sharpe(low_vol) > compute_path_sharpe(mixed)


def test_archetype_C_negative_terminal() -> None:
    cfg = UMaeResidualAuditConfig(reward_horizon=10)
    runner = UMaeResidualAuditRunner(_market_source(), config=cfg)
    path = runner._path_from_cumulative(
        "arch_C", [0, 3, -1, -3], 10, adverse_wick=True
    )
    ra = compute_risk_adjusted_path_metrics(path.to_context(), Action.LONG, 10)
    assert ra.terminal_return < 0
    assert ra.ulcer_index > 0


def test_archetype_A_vs_B_sharpe_not_equal() -> None:
    report = RiskAdjustedPathAuditRunner(_market_source()).run()
    notes = next(a for a in report["archetype_ABC_analysis"] if a.get("id") == "comparison")
    ab = notes["pairwise_notes"]["A_vs_B"]
    assert ab["sharpe"]["A"] != ab["sharpe"]["B"]
