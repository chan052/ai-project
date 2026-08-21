"""Tests for P1 Return/Risk Weighting Robustness Validation (analysis-only)."""

from __future__ import annotations

from chartai.analysis.p1_weighting_robustness_validation import (
    FIXED_STRUCTURE,
    P1WeightingRobustnessRunner,
    WEIGHT_GRID,
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


def test_weighting_robustness_runs() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    assert report["fixed_structure"] == FIXED_STRUCTURE
    assert report["1_executive_summary"]["eval_n"] > 0
    assert len(report["2_return_weighting_sweep"]["per_weight"]) == len(WEIGHT_GRID)


def test_weight_grid_complete() -> None:
    assert len(WEIGHT_GRID) == 11
    assert WEIGHT_GRID[0] == (1.0, 0.0)
    assert WEIGHT_GRID[-1] == (0.0, 1.0)


def test_return_sweep_has_archetypes() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    w = report["2_return_weighting_sweep"]["per_weight"][5]
    assert "archetype_semantics" in w
    assert "B" in w["archetype_semantics"]["notes"]


def test_risk_false_equivalence_tracked() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    w = report["3_risk_weighting_sweep"]["per_weight"][0]
    assert "false_equivalence" in w


def test_robustness_regions_present() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    assert "robustness" in report["2_return_weighting_sweep"]
    assert "robustness" in report["3_risk_weighting_sweep"]


def test_final_verdict_categories() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    ret_cat = report["6_final_verdict"]["return"]["category"]
    risk_cat = report["6_final_verdict"]["risk"]["category"]
    assert ret_cat in ("A_strong_candidate", "B_robust_region", "C_fragile_region", "D_no_valid_scalar")
    assert risk_cat in ("A_strong_candidate", "B_robust_region", "C_fragile_region", "D_no_valid_scalar")


def test_failure_cases_structure() -> None:
    report = P1WeightingRobustnessRunner(_synth_market()).run()
    cases = report["5_concrete_failure_cases"]["cases"]
    if cases:
        c = cases[0]
        assert "path_ascii" in c
        assert "why_misleading" in c
