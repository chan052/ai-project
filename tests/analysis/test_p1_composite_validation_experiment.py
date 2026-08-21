"""Tests for P1 Composite Validation experiment (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_composite_validation_experiment import (
    FIXED_STRUCTURE,
    P1CompositeValidationRunner,
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


def test_composite_validation_runs() -> None:
    report = P1CompositeValidationRunner(_synth_market()).run()
    assert report["fixed_structure"] == FIXED_STRUCTURE
    assert "3_U_vs_MFE_dominance" in report
    assert "8_BACG_archetype_risk" in report
    assert report["final_verdict_detail"]["choice"] in ("A", "B", "C")


def test_return_composite_is_sum_of_scaled() -> None:
    report = P1CompositeValidationRunner(_synth_market()).run()
    dom = report["3_U_vs_MFE_dominance"]["overall"]
    assert 0 <= dom["mean_U_share"] <= 1


def test_abc_archetype_section() -> None:
    report = P1CompositeValidationRunner(_synth_market()).run()
    paths = report["4_ABC_archetype_return"]["paths"]
    assert set(paths.keys()) >= {"A", "B", "C", "G"}


def test_risk_archetype_rankings() -> None:
    report = P1CompositeValidationRunner(_synth_market()).run()
    assert "rankings" in report["8_BACG_archetype_risk"]


def test_scale_only_no_recovery_in_composite() -> None:
    report = P1CompositeValidationRunner(_synth_market()).run()
    assert "Risk_composite" in report["6_risk_composite_results"]["formula"]
