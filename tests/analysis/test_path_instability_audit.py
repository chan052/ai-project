"""Tests for path instability vs MAE audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.path_instability_audit import PathInstabilityAuditRunner
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
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


def test_path_instability_audit_runs() -> None:
    report = PathInstabilityAuditRunner(_market_source()).run()
    assert "CONFIRMED" in report
    assert "SEMANTIC_VERDICT" in report
    assert len(report["primary_ABC"]) == 3


def test_excursion_volatility_computed() -> None:
    cfg = UMaeResidualAuditConfig(reward_horizon=10)
    runner = UMaeResidualAuditRunner(_market_source(), config=cfg)
    path = runner._path_from_cumulative("vol", [0, 1, 3, 1], 10)
    obs = compute_path_residual_observables(path.to_context(), Action.LONG, 10)
    assert obs.excursion_volatility > 0.0


def test_abc_risk_order_chop_or_giveback() -> None:
    report = PathInstabilityAuditRunner(_market_source()).run()
    ro = report["risk_ordering_B_lt_A_lt_C"]
    matching = ro["metrics_matching_target_order"]
    assert len(matching) >= 1


def test_giveback_chop_not_identical_on_G() -> None:
    report = PathInstabilityAuditRunner(_market_source()).run()
    q = report["giveback_vs_chop_quadrants"]["classification"]
    assert q["Q_lg_hc"]["chop"] > q["Q_lg_lc"]["chop"]


def test_recovery_same_mae_different_outcome() -> None:
    report = PathInstabilityAuditRunner(_market_source()).run()
    rec = report["recovery_comparison"]
    assert rec["recovery_after_mae"]["recover"] != rec["recovery_after_mae"]["sustain"]
