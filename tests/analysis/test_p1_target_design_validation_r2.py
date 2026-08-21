"""Tests for P1 Target Design Validation Round 2 (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_target_design_validation_r2 import P1TargetDesignValidationR2Runner
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


def test_validation_r2_runs() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    assert report["audit"].startswith("P1 Target Design Validation Round 2")
    assert "8_final_yes_no_partial" in report
    assert "9_prior_conclusions" in report
    assert "P1_candidate_structure" in report


def test_u_scalar_b_first() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    dr = report["dataset_reports"][0]
    assert dr["U_vs_MFE"]["C_scalar_vs_separate"]["scalar_B_still_first"]


def test_mfe_facet_a_above_b() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    assert report["dataset_reports"][0]["U_vs_MFE"]["C_scalar_vs_separate"]["MFE_facet_A_above_B"]


def test_final_q1_u_only_no() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    assert report["8_final_yes_no_partial"]["Q1_U_only_sufficient_for_Expected_Return"]["answer"] == "NO"


def test_final_q3_scalar_no() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    assert report["8_final_yes_no_partial"]["Q3_U_MFE_scalar_appropriate"]["answer"] == "NO"


def test_recovery_raw_separates_same_mae() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    rec = report["dataset_reports"][0]["recovery"]
    assert rec["same_MAE_synthetic_pair"]
    assert rec["raw_recovery"]["abs_diff"] > 0.3
    assert len(rec["z_weakening_analysis"]["reasons"]) >= 2


def test_synthetic_matched_pairs_present() -> None:
    report = P1TargetDesignValidationR2Runner([("SYNTH", _synth_market())]).run()
    pairs = report["7_matched_path_synthetic"]["pairs"]
    assert len(pairs) >= 4
    assert any(p["pair_id"] == "recovery_only" for p in pairs)
