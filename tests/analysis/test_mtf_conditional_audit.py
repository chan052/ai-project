"""Tests for MTF conditional information audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.mtf_conditional_audit import MtfConditionalAuditRunner
from chartai.analysis.mtf_context_encoding import TrendRegime, encode_htf_slice
from chartai.analysis.mtf_market_data import MTFMarketDataSource
from chartai.core.types import OHLCVBar, Timeframe
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.state import TimeframeStateSlice
from chartai.core.types import TimeframeWindow, DecisionTime
import pandas as pd


def _synthetic_mtf_source() -> MTFMarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=200, reward_horizon=15)
    return MTFMarketDataSource(
        symbol="SYNTH",
        bars_3m=tuple(ds.bars_3m),
        bars_1h=tuple(ds.bars_1h),
        bars_4h=tuple(ds.bars_4h),
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_mtf_conditional_audit_runs() -> None:
    report = MtfConditionalAuditRunner(_synthetic_mtf_source()).run()
    assert "3_leakage_check" in report
    assert report["3_leakage_check"]["all_passed"] is True
    assert "MTF_AUDIT_VERDICT" in report
    assert "final_questions" in report
    assert len(report["15_confirmed"]) >= 1


def test_htf_regime_encoding() -> None:
    bars = (
        OHLCVBar(
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1.0,
        ),
        OHLCVBar(
            start=pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            end=pd.Timestamp("2024-01-01 02:00", tz="UTC"),
            open=100.5,
            high=102.0,
            low=100.0,
            close=101.5,
            volume=1.0,
        ),
    )
    dt = DecisionTime(timestamp=bars[-1].end)
    w = TimeframeWindow(
        timeframe=Timeframe.H1,
        start_index=0,
        end_index=1,
        decision_time=dt,
    )
    sl = TimeframeStateSlice(timeframe=Timeframe.H1, window=w, bars=bars)
    obs = encode_htf_slice(sl)
    assert obs.regime is TrendRegime.BULLISH


def test_report_sections() -> None:
    report = MtfConditionalAuditRunner(_synthetic_mtf_source()).run()
    for key in (
        "6_same_3m_different_mtf",
        "13_negative_control",
        "14_representative_pairs",
    ):
        assert key in report
