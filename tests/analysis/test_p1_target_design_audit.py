"""Tests for P1 Return/Risk/Direction Target Design Audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_target_design_audit import P1TargetDesignAuditRunner
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def _synth_market(num_3m: int = 150) -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=num_3m, reward_horizon=10)
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_p1_target_design_audit_runs() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    assert report["audit"].startswith("P1 Return/Risk/Direction")
    assert "CONFIRMED" in report
    assert "final_questions" in report
    assert len(report["dataset_reports"]) == 1
    dr = report["dataset_reports"][0]
    assert "A_expected_return" in dr
    assert "B_acceptable_risk" in dr
    assert "C_recovery_experiment" in dr
    assert "D_direction_design" in dr


def test_abc_MFE_ranks_A_above_B() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    chk = report["dataset_reports"][0]["A_expected_return"]["checks"]["U_spike_vs_sustained_AB"]
    assert chk["MFE_ranks_A_above_B"]


def test_abc_U_ranks_B_above_A() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    chk = report["dataset_reports"][0]["A_expected_return"]["checks"]["U_spike_vs_sustained_AB"]
    assert chk["U_ranks_B_above_A"]


def test_risk_giveback_B_lt_A_lt_C() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    assert report["dataset_reports"][0]["B_acceptable_risk"]["synthetic_B_lt_A_lt_C_giveback_z"]


def test_recovery_separates_same_mae_paths() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    rec = report["dataset_reports"][0]["C_recovery_experiment"]
    assert rec["same_MAE"]
    assert rec["recovery_improves_separation"]


def test_direction_prefers_directional_ev() -> None:
    report = P1TargetDesignAuditRunner([("SYNTH", _synth_market())]).run()
    d = report["dataset_reports"][0]["D_direction_design"]
    assert d["recommended"] == "directional_expected_value_per_action"
