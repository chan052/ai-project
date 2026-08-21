"""Tests for P1 Return/Risk Real-Data Validation (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_return_risk_realdata_validation import (
    FIXED_CANDIDATE,
    P1ReturnRiskRealDataValidationRunner,
)
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def _synth_market(num_3m: int = 200) -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=num_3m, reward_horizon=10)
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_realdata_validation_runs() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    assert report["audit"].startswith("P1 Return/Risk Target Real-Data")
    assert report["fixed_candidate_structure"] == FIXED_CANDIDATE
    assert "9_verdicts" in report
    assert "10_final_recommendation" in report
    assert report["3_return_validation"]["designs"]["A2_Return_UMFE"] == "z(U)+z(MFE)"


def test_return_tail_bins_present() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    tail = report["3_return_validation"]["U_tail_dominance"]
    assert "abs_zU_gt_2" in tail
    assert "abs_zU_gt_3" in tail


def test_risk_designs_present() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    agg = report["5_risk_aggregation_audit"]
    assert agg["designs"]["B3_Risk_MGC"] == "z(MAE)+z(giveback)+z(chop)"


def test_matched_path_sections() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    matched = report["6_matched_path_analysis"]
    assert "pair1_MAE_similar_giveback_diff" in matched
    assert "pair4_single_facet_extreme" in matched


def test_verdicts_have_q1_q10() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    v = report["9_verdicts"]
    assert "Q1_Return_scalar_semantic_valid" in v
    assert "Q10_recovery_adds_beyond_MGC" in v


def test_recommendation_choice_is_letter() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    choice = report["10_final_recommendation"]["choice"]
    assert choice in ("A", "B", "C", "D")


def test_recovery_excluded_from_risk_sum() -> None:
    report = P1ReturnRiskRealDataValidationRunner(_synth_market()).run()
    rec = report["8_recovery_diagnostic"]
    assert rec["recovery_not_in_Risk_MGC"] is True
