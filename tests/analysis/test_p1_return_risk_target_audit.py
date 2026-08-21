"""Tests for P1 Return/Risk Target Validation Audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_return_risk_target_audit import P1ReturnRiskTargetAuditRunner
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


def test_p1_return_risk_target_audit_runs() -> None:
    report = P1ReturnRiskTargetAuditRunner(_market_source()).run()
    assert "CONFIRMED" in report
    assert "structure_adoption_judgment" in report
    assert len(report["synthetic_paths"]) >= 5
    assert report["candidate_structure"]["expected_return"] == ["U", "MFE"]


def test_giveback_B_lt_A_lt_C() -> None:
    report = P1ReturnRiskTargetAuditRunner(_market_source()).run()
    gb = report["risk_validation"]["ABC_ordering"]["giveback"]
    assert gb["B_lt_A"] and gb["A_lt_C"]


def test_MFE_A_gt_B() -> None:
    report = P1ReturnRiskTargetAuditRunner(_market_source()).run()
    assert report["return_validation"]["A_vs_B"]["MFE"]["A_gt_B"]


def test_C_U_lower_than_AB() -> None:
    report = P1ReturnRiskTargetAuditRunner(_market_source()).run()
    c = report["return_validation"]["C_return_not_unconditionally_high"]
    assert c["U_suppresses_unconditional_high"]


def test_G_chop_higher_than_B() -> None:
    report = P1ReturnRiskTargetAuditRunner(_market_source()).run()
    assert report["risk_validation"]["chop_catches_round_trip"]["chop_G_higher"]
