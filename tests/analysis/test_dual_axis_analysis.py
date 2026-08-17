"""Tests for dual-axis opportunity labeling analysis."""

from __future__ import annotations

from chartai.analysis.dual_axis_scores import (
    compute_dual_axis_scores,
    truncate_reward_context,
)
from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def test_truncate_reward_context() -> None:
    ds = SyntheticMTFDataset.build_standard()
    ctx = ds.future_context_builder().build(50)
    short = truncate_reward_context(ctx, 5)
    assert short.reward_horizon == 5
    assert len(short.future_closes) == 5


def test_dual_axis_scores_runs() -> None:
    ds = SyntheticMTFDataset.build_standard()
    builder = ds.future_context_builder()
    immediate, deferred, _ = compute_dual_axis_scores(
        builder,
        t_index=50,
        horizon=10,
        action=Action.LONG,
        max_bar_index=len(ds.bars_3m) - 1,
    )
    assert 0.0 <= immediate.captureability <= 1.0 or immediate.captureability != immediate.captureability
    assert deferred.consensus_best_delay >= 0


def test_dual_axis_analysis_smoke() -> None:
    from chartai.analysis.dual_axis_analysis import DualAxisAnalysisRunner
    from chartai.data.market_data import MarketDataSource

    ds = SyntheticMTFDataset.build_standard()
    source = MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )
    report = DualAxisAnalysisRunner(source).run()
    assert "quadrant_analysis" in report
    assert "id_correlation" in report
    assert "case_ab_analysis" in report
