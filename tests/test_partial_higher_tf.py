"""Partial higher-TF bar causality tests (P1 MTF alignment)."""

from __future__ import annotations

import pandas as pd
import pytest

from chartai.core.types import OHLCVBar, Timeframe
from chartai.data.mtf_aligner import (
    BarSeriesBuilder,
    HigherTfBarKind,
    MultiTimeframeAligner,
)
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.state import StateBuilder


def _aligner_14_27_with_varying_3m() -> tuple[MultiTimeframeAligner, int]:
    """3m bars through 14:27 with distinct OHLCV for aggregation tests."""
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        pd.date_range("2024-01-02 09:00", periods=8, freq="h")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        pd.date_range("2024-01-02 08:00", periods=4, freq="4h")
    )
    m3_starts = pd.date_range("2024-01-02 13:54", periods=11, freq="3min")
    m3: list[OHLCVBar] = []
    for i, ts in enumerate(m3_starts):
        close = 100.0 + i * 0.1
        m3.append(
            OHLCVBar(
                start=ts,
                end=ts + pd.Timedelta(minutes=3),
                open=close - 0.05,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1.0 + i,
            )
        )
    aligner = MultiTimeframeAligner(bars_3m=m3, bars_1h=h1, bars_4h=h4)
    return aligner, len(m3) - 1


def test_partial_bar_unchanged_when_future_3m_mutates() -> None:
    """Test 1: future 3m mutation must not change partial 1H/4H at t."""
    aligner, t_index = _aligner_14_27_with_varying_3m()
    decision = aligner.decision_time_at_3m_index(t_index)
    before_1h = aligner.build_partial_bar(Timeframe.H1, decision)
    before_4h = aligner.build_partial_bar(Timeframe.H4, decision)
    assert before_1h is not None and before_4h is not None

    bars_3m = list(aligner._bars(Timeframe.M3))
    extra = [
        OHLCVBar(
            start=ts,
            end=ts + pd.Timedelta(minutes=3),
            open=999.0,
            high=999.0,
            low=999.0,
            close=999.0,
            volume=999.0,
        )
        for ts in pd.date_range(bars_3m[-1].end, periods=5, freq="3min")
    ]
    bars_3m.extend(extra)
    bars_3m[t_index + 3] = OHLCVBar(
        start=bars_3m[t_index + 3].start,
        end=bars_3m[t_index + 3].end,
        open=999.0,
        high=999.0,
        low=999.0,
        close=999.0,
        volume=999.0,
    )
    aligner2 = MultiTimeframeAligner(
        bars_3m=bars_3m,
        bars_1h=aligner._bars(Timeframe.H1),
        bars_4h=aligner._bars(Timeframe.H4),
    )
    assert aligner2.build_partial_bar(Timeframe.H1, decision) == before_1h
    assert aligner2.build_partial_bar(Timeframe.H4, decision) == before_4h


def test_future_mutation_does_not_change_mtf_state() -> None:
    """Test 2: large future 3m changes leave State(t) unchanged across TFs."""
    ds = SyntheticMTFDataset.build_standard()
    t_index = 50
    before = ds.state_builder().build(t_index)
    fp = before.fingerprint()

    for idx in range(t_index + 1, t_index + 15):
        if idx < len(ds.bars_3m):
            ds.set_3m_close(idx, ds.bars_3m[idx].close * 50.0)

    after = ds.state_builder().build(t_index)
    assert after.fingerprint() == fp


def test_partial_bar_updates_when_past_3m_changes() -> None:
    """Test 3: changing 3m data at/before t updates partial bar."""
    aligner, t_index = _aligner_14_27_with_varying_3m()
    decision = aligner.decision_time_at_3m_index(t_index)
    before = aligner.build_partial_bar(Timeframe.H1, decision)
    assert before is not None

    bars_3m = list(aligner._bars(Timeframe.M3))
    bars_3m[t_index] = OHLCVBar(
        start=bars_3m[t_index].start,
        end=bars_3m[t_index].end,
        open=200.0,
        high=210.0,
        low=190.0,
        close=205.0,
        volume=50.0,
    )
    aligner2 = MultiTimeframeAligner(
        bars_3m=bars_3m,
        bars_1h=aligner._bars(Timeframe.H1),
        bars_4h=aligner._bars(Timeframe.H4),
    )
    after = aligner2.build_partial_bar(Timeframe.H1, decision)
    assert after is not None
    assert after.close == 205.0
    assert after != before


def test_completed_bars_match_native_series() -> None:
    """Test 4: completed HTF bars identical to native series bars."""
    aligner, t_index = _aligner_14_27_with_varying_3m()
    decision = aligner.decision_time_at_3m_index(t_index)
    native = aligner._bars(Timeframe.H1)
    for idx in aligner.completed_higher_tf_bar_indices(Timeframe.H1, decision):
        sb = next(
            sb for sb in aligner.state_bars(Timeframe.H1, decision, lookback_bars=10)
            if sb.native_index == idx
        )
        assert sb.bar == native[idx]


def test_partial_bar_no_future_ohlcv_contamination() -> None:
    """Test 5: partial OHLCV uses only 3m bars with end <= t."""
    aligner, t_index = _aligner_14_27_with_varying_3m()
    decision = aligner.decision_time_at_3m_index(t_index)
    interval = aligner.in_progress_higher_tf_interval(Timeframe.H1, decision)
    assert interval is not None
    m3_used = aligner.contributing_3m_bars_for_interval(
        interval.start, interval.end, decision
    )
    partial = aligner.build_partial_bar(Timeframe.H1, decision)
    assert partial is not None
    assert all(b.end <= decision.timestamp for b in m3_used)
    assert partial.open == m3_used[0].open
    assert partial.close == m3_used[-1].close
    assert partial.high == max(b.high for b in m3_used)
    assert partial.low == min(b.low for b in m3_used)
    assert partial.volume == sum(b.volume for b in m3_used)


def test_boundary_exact_hour_close() -> None:
    """Test 6: t=15:00 uses completed 14:00–15:00 bar, no 15:00–16:00 partial."""
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        pd.date_range("2024-01-02 13:00", periods=4, freq="h")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        [pd.Timestamp("2024-01-02 08:00"), pd.Timestamp("2024-01-02 12:00")]
    )
    m3 = BarSeriesBuilder(Timeframe.M3).build_3m_bars(
        pd.date_range("2024-01-02 14:45", periods=5, freq="3min")
    )
    from chartai.core.config import StateConfig, TimeframeStateConfig

    cfg = StateConfig(
        timeframes={
            "3m": TimeframeStateConfig(lookback_bars=3),
            "1h": TimeframeStateConfig(lookback_bars=2),
            "4h": TimeframeStateConfig(lookback_bars=1),
        }
    )
    aligner = MultiTimeframeAligner(bars_3m=m3, bars_1h=h1, bars_4h=h4, state_config=cfg)
    t_index = len(m3) - 1
    state = StateBuilder(aligner).build(t_index)

    assert state.decision_time.timestamp == pd.Timestamp("2024-01-02 15:00")
    assert state.slice_1h.has_partial_bar is False
    assert state.slice_1h.bars[-1].end == pd.Timestamp("2024-01-02 15:00")
    assert all(b.start < pd.Timestamp("2024-01-02 15:00") for b in state.slice_1h.bars)
