"""Tests for P1 Normalization Semantic Preservation experiment (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_normalization_semantic_experiment import (
    FIXED_STRUCTURE,
    NORM_METHODS,
    NormBundle,
    P1NormalizationSemanticExperimentRunner,
    PrefixNormParams,
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


def test_prefix_norm_zero_stdscale() -> None:
    p = PrefixNormParams.fit("U", (0.1, 0.2, 0.3))
    assert abs(p.at_zero("stdscale")) < 1e-12
    assert abs(p.at_zero("rmsscale")) < 1e-12
    assert abs(p.at_zero("raw")) < 1e-12


def test_prefix_norm_zscore_zero_not_zero() -> None:
    p = PrefixNormParams.fit("U", (0.1, 0.2, 0.3))
    assert abs(p.at_zero("zscore")) > 1e-6


def test_norm_bundle_composite_return() -> None:
    rows = [
        {"U": 0.01, "MFE": 0.02, "MAE": 0.001, "giveback": 0.5, "chop": 0.1,
         "recovery": 0.0, "P_long": 0.01, "P_short": -0.01},
        {"U": 0.02, "MFE": 0.03, "MAE": 0.002, "giveback": 0.4, "chop": 0.2,
         "recovery": 0.0, "P_long": 0.02, "P_short": -0.02},
    ]
    bundle = NormBundle.fit_from_rows(rows)
    raw = rows[0]
    assert bundle.composite_return(raw, "raw") == raw["U"] + raw["MFE"]


def test_experiment_runs() -> None:
    report = P1NormalizationSemanticExperimentRunner(_synth_market()).run()
    assert report["fixed_structure"] == FIXED_STRUCTURE
    assert len(report["12_semantic_vs_scale_comparison_table"]) == len(NORM_METHODS)
    assert "final_question_answer" in report


def test_four_normalizations_in_table() -> None:
    report = P1NormalizationSemanticExperimentRunner(_synth_market()).run()
    names = {r["normalization"] for r in report["12_semantic_vs_scale_comparison_table"]}
    assert names == set(NORM_METHODS)


def test_archetype_section_has_ABCG() -> None:
    report = P1NormalizationSemanticExperimentRunner(_synth_market()).run()
    ids = {p["id"] for p in report["8_archetype_results"]["paths"]}
    assert {"A", "B", "C", "G", "REC"}.issubset(ids)


def test_stdscale_preserves_sign() -> None:
    report = P1NormalizationSemanticExperimentRunner(_synth_market()).run()
    assert report["6_stdscale_audit"]["sign_preserved"] is True
