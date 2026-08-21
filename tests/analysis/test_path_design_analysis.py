"""Tests for P1 Path Design Analysis (analysis-only)."""

from __future__ import annotations

from chartai.analysis.path_design_analysis import (
    OBSERVABLE_CATALOG,
    PathDesignAnalysisRunner,
)
from chartai.analysis.path_residual_diagnostics import (
    CANDIDATE_SPECS,
    compute_path_residual_observables,
)
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.synthetic import build_scenario, SyntheticScenario


def _market_source() -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=150, reward_horizon=10)
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_path_design_analysis_runs() -> None:
    report = PathDesignAnalysisRunner(_market_source()).run()
    assert "1_observable_catalog" in report
    assert "10_case1_spike_vs_grind" in report
    assert "14_recommended_structures" in report
    assert report["final_conclusion_category"] == "2_Path_as_P1_output_diagnostic"
    assert len(report["9_archetype_cases"]) == 5


def test_extended_observables_present() -> None:
    h = 10
    path = build_scenario(SyntheticScenario.DOWN_THEN_UP, horizon=h)
    obs = compute_path_residual_observables(path.to_context(), Action.LONG, h)
    assert obs.recovery_speed >= 0.0
    assert obs.time_under_water >= 0.0
    assert obs.path_sign_entropy >= 0.0
    keys = {s.key for s in CANDIDATE_SPECS}
    assert "recovery_speed" in keys
    assert "time_under_water" in keys


def test_catalog_covers_core_observables() -> None:
    catalog_keys = {o.key for o in OBSERVABLE_CATALOG}
    for key in (
        "giveback_ratio",
        "reversal_depth",
        "oscillation_chop",
        "recovery_shape_score",
    ):
        assert key in catalog_keys


def test_case1_design_question_present() -> None:
    report = PathDesignAnalysisRunner(_market_source()).run()
    case1 = report["10_case1_spike_vs_grind"]
    assert case1["giveback_A_much_higher"]
    assert "core_design_question" in case1
