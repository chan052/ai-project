"""Smoke tests for market path experiment runner."""

from __future__ import annotations

from chartai.analysis.path_experiment import PathExperimentRunner
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def test_path_experiment_runs_on_synthetic_bars() -> None:
    ds = SyntheticMTFDataset.build_standard()
    source = MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic:test",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )
    report = PathExperimentRunner(source).run()
    assert "variants" in report
    assert set(report["variants"]) == {
        "raw_return",
        "sign_based",
        "vol_normalized",
        "bounded_tanh",
    }
    assert report["causality"]["sigma_unchanged_by_future_bar"] is True
