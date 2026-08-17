"""Tests for execution proxy and opportunity analysis."""

from __future__ import annotations

from chartai.analysis.execution_proxy import simulate_target_stop
from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def test_target_stop_runs() -> None:
    ds = SyntheticMTFDataset.build_standard()
    ctx = ds.future_context_builder().build(50)
    r = simulate_target_stop(ctx, Action.LONG, target_pct=0.01, stop_pct=0.005)
    assert r.target_pct == 0.01


def test_opportunity_analysis_smoke() -> None:
    from chartai.analysis.opportunity_analysis import OpportunityAnalysisRunner
    from chartai.data.market_data import MarketDataSource

    ds = SyntheticMTFDataset.build_standard()
    source = MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )
    report = OpportunityAnalysisRunner(source).run()
    assert "case_ab_analysis" in report
