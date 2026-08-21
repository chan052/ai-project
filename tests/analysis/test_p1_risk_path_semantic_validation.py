"""Tests for P1 Risk vs Path Semantic Validation (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_risk_path_semantic_validation import (
    STRUCTURE_A,
    STRUCTURE_B,
    P1RiskPathSemanticValidationRunner,
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


def test_risk_path_validation_runs() -> None:
    report = P1RiskPathSemanticValidationRunner(_synth_market()).run()
    assert report["structures_compared"]["A"] == STRUCTURE_A
    assert report["structures_compared"]["B"] == STRUCTURE_B
    assert report["12_final_verdict"]["choice"] in ("A", "B", "C")


def test_archetypes_present() -> None:
    report = P1RiskPathSemanticValidationRunner(_synth_market()).run()
    ids = {p["id"] for p in report["3_synthetic_archetype_analysis"]["paths"]}
    assert {"B", "A", "G", "C", "REC"}.issubset(ids)


def test_structure_b_path_is_chop() -> None:
    report = P1RiskPathSemanticValidationRunner(_synth_market()).run()
    g = next(p for p in report["3_synthetic_archetype_analysis"]["paths"] if p["id"] == "G")
    b = next(p for p in report["3_synthetic_archetype_analysis"]["paths"] if p["id"] == "B")
    assert g["Path_B"] >= b["Path_B"]


def test_risk_b_excludes_chop() -> None:
    report = P1RiskPathSemanticValidationRunner(_synth_market()).run()
    p = report["3_synthetic_archetype_analysis"]["paths"][0]
    assert p["Risk_B"] == p["scaled"]["MAE"] + p["scaled"]["giveback"]


def test_btc_boundary_categories() -> None:
    report = P1RiskPathSemanticValidationRunner(_synth_market()).run()
    cases = report["8_real_btc_boundary_cases"]
    assert "low_MAE_high_Chop" in cases
    assert "high_MAE_low_Chop" in cases
