"""Tests for P1 Return/Risk Aggregation & Path Validation (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_aggregation_path_validation import (
    FIXED_STRUCTURE,
    P1AggregationPathValidationRunner,
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


def test_aggregation_path_validation_runs() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    assert report["fixed_structure"] == FIXED_STRUCTURE
    assert report["1_executive_summary"]["eval_n"] > 0
    assert report["9_final_verdict"]["questions"]["Q8_composite_preserves_P1_intent"] == "FAILED"


def test_return_archetypes_present() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    ids = {a["id"] for a in report["2_return"]["archetypes"]}
    assert {"A", "B", "C", "G", "REC"}.issubset(ids)


def test_x_sigma_sign_preserved() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    for key in ("U", "MFE", "MAE", "giveback"):
        assert report["4_x_sigma_normalization"]["per_facet"][key]["sign_preserved"]


def test_path_buckets_exist() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    buckets = report["5_path_chop"]["representative_cases"]
    assert "low_MAE_low_Chop" in buckets
    assert "high_MAE_high_Chop" in buckets


def test_recovery_rec_archetype() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    rec = report["7_recovery_diagnostic"]["archetype_REC"]
    assert "recovery" in rec["raw"]
    assert rec["Risk_composite"] >= 0


def test_failure_cases_minimum() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    assert report["8_concrete_failure_cases"]["count"] >= 10


def test_chop_decomposition_correlations() -> None:
    report = P1AggregationPathValidationRunner(_synth_market()).run()
    corr = report["6_chop_frequency_magnitude"]["correlations"]
    assert "freq_vs_chop" in corr
    assert "mag_vs_chop" in corr
