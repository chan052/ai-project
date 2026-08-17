"""Smoke test for P1 structure experiment."""

from __future__ import annotations

from chartai.analysis.p1_structure_experiment import P1StructureExperimentRunner
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset


def test_p1_structure_experiment_runs() -> None:
    ds = SyntheticMTFDataset.build_standard()
    source = MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )
    report = P1StructureExperimentRunner(source).run()
    assert "baseline" in report
    assert "sd_pairs" in report
    assert len(report["sd_pairs"]) == 4
