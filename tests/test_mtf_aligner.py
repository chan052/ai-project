"""Tests for MultiTimeframeAligner — P1 MTF partial bar rules."""

from __future__ import annotations

import pandas as pd
import pytest

from chartai.core.config import StateConfig
from chartai.core.types import DecisionTime, OHLCVBar, Timeframe
from chartai.data.mtf_aligner import (
    BarSeriesBuilder,
    HigherTfBarKind,
    MultiTimeframeAligner,
)


def test_14_27_includes_partial_1h_bar(aligner_14_27: MultiTimeframeAligner) -> None:
    """At t=14:27, partial 1H bar for 14:00–15:00 exists with data through 14:27 only."""
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1
    decision_time = aligner_14_27.decision_time_at_3m_index(t_index)
    assert decision_time.timestamp == pd.Timestamp("2024-01-02 14:27")

    state_bars = aligner_14_27.state_bars(Timeframe.H1, decision_time, lookback_bars=2)
    assert len(state_bars) == 2
    assert state_bars[0].kind is HigherTfBarKind.COMPLETED
    assert state_bars[0].bar.start == pd.Timestamp("2024-01-02 13:00")
    assert state_bars[0].bar.end == pd.Timestamp("2024-01-02 14:00")

    partial = state_bars[1]
    assert partial.kind is HigherTfBarKind.PARTIAL
    assert partial.bar.start == pd.Timestamp("2024-01-02 14:00")
    assert partial.bar.end == pd.Timestamp("2024-01-02 15:00")
    assert partial.bar.close == bars_3m[t_index].close

    m3_contrib = aligner_14_27.contributing_3m_bars_for_interval(
        pd.Timestamp("2024-01-02 14:00"),
        pd.Timestamp("2024-01-02 15:00"),
        decision_time,
    )
    assert partial.bar.open == m3_contrib[0].open
    assert partial.bar.high == max(b.high for b in m3_contrib)
    assert partial.bar.low == min(b.low for b in m3_contrib)
    assert partial.bar.volume == sum(b.volume for b in m3_contrib)


def test_higher_tf_state_includes_completed_and_partial(
    aligner_14_27: MultiTimeframeAligner,
) -> None:
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1
    decision_time = aligner_14_27.decision_time_at_3m_index(t_index)

    for tf in (Timeframe.H1, Timeframe.H4):
        state_bars = aligner_14_27.state_bars(tf, decision_time, lookback_bars=2)
        assert state_bars[-1].kind is HigherTfBarKind.PARTIAL
        assert all(sb.bar.is_completed_at(decision_time.timestamp) or sb.kind is HigherTfBarKind.PARTIAL for sb in state_bars)


def test_changing_future_3m_bars_does_not_change_state_at_t() -> None:
    h1_starts = pd.date_range("2024-01-02 09:00", periods=8, freq="h")
    bars_1h = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(h1_starts)
    h4_starts = pd.date_range("2024-01-02 08:00", periods=4, freq="4h")
    bars_4h = BarSeriesBuilder(Timeframe.H4).build_4h_bars(h4_starts)

    m3_starts = pd.date_range("2024-01-02 13:54", periods=11, freq="3min")
    bars_3m_a = list(BarSeriesBuilder(Timeframe.M3).build_3m_bars(m3_starts))

    aligner_a = MultiTimeframeAligner(bars_3m=bars_3m_a, bars_1h=bars_1h, bars_4h=bars_4h)
    t_index = len(bars_3m_a) - 1
    decision = aligner_a.decision_time_at_3m_index(t_index)
    before_1h = aligner_a.state_bars(Timeframe.H1, decision, lookback_bars=2)
    before_4h = aligner_a.state_bars(Timeframe.H4, decision, lookback_bars=1)

    extra_starts = pd.date_range(m3_starts[-1] + pd.Timedelta(minutes=3), periods=5, freq="3min")
    extra_bars = [
        OHLCVBar(
            start=ts,
            end=ts + pd.Timedelta(minutes=3),
            open=999.0,
            high=999.0,
            low=999.0,
            close=999.0,
            volume=999.0,
        )
        for ts in extra_starts
    ]
    aligner_b = MultiTimeframeAligner(
        bars_3m=bars_3m_a + extra_bars, bars_1h=bars_1h, bars_4h=bars_4h
    )
    after_1h = aligner_b.state_bars(Timeframe.H1, decision, lookback_bars=2)
    after_4h = aligner_b.state_bars(Timeframe.H4, decision, lookback_bars=1)

    assert before_1h == after_1h
    assert before_4h == after_4h


def test_completed_higher_tf_bars_unchanged_by_partial_logic(
    aligner_14_27: MultiTimeframeAligner,
) -> None:
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1
    decision_time = aligner_14_27.decision_time_at_3m_index(t_index)
    native_1h = aligner_14_27._bars(Timeframe.H1)

    state_bars = aligner_14_27.state_bars(Timeframe.H1, decision_time, lookback_bars=3)
    completed = [sb for sb in state_bars if sb.kind is HigherTfBarKind.COMPLETED]
    for sb in completed:
        assert sb.native_index is not None
        assert sb.bar == native_1h[sb.native_index]


def test_boundary_t_equals_1h_close_uses_completed_not_partial() -> None:
    """t=15:00 → 14:00–15:00 is completed; 15:00–16:00 must not appear."""
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        pd.date_range("2024-01-02 13:00", periods=4, freq="h")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        [pd.Timestamp("2024-01-02 08:00"), pd.Timestamp("2024-01-02 12:00")]
    )
    m3 = BarSeriesBuilder(Timeframe.M3).build_3m_bars(
        pd.date_range("2024-01-02 14:45", periods=5, freq="3min")
    )
    aligner = MultiTimeframeAligner(bars_3m=m3, bars_1h=h1, bars_4h=h4)
    t_index = len(m3) - 1
    assert aligner.decision_time_at_3m_index(t_index).timestamp == pd.Timestamp("2024-01-02 15:00")

    decision = aligner.decision_time_at_3m_index(t_index)
    assert aligner.in_progress_higher_tf_interval(Timeframe.H1, decision) is None
    state_bars = aligner.state_bars(Timeframe.H1, decision, lookback_bars=1)
    assert len(state_bars) == 1
    assert state_bars[0].kind is HigherTfBarKind.COMPLETED
    assert state_bars[0].bar.start == pd.Timestamp("2024-01-02 14:00")
    assert state_bars[0].bar.end == pd.Timestamp("2024-01-02 15:00")


def test_completed_only_deprecated_flag() -> None:
    cfg = StateConfig(use_completed_higher_tf_bars_only=True)
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        pd.date_range("2024-01-02 13:00", periods=3, freq="h")
    )
    m3 = BarSeriesBuilder(Timeframe.M3).build_3m_bars(
        pd.date_range("2024-01-02 14:18", periods=4, freq="3min")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        [pd.Timestamp("2024-01-02 08:00")]
    )
    aligner = MultiTimeframeAligner(
        bars_3m=m3, bars_1h=h1, bars_4h=h4, state_config=cfg
    )
    decision = DecisionTime(timestamp=pd.Timestamp("2024-01-02 14:27"))
    state_bars = aligner.state_bars(Timeframe.H1, decision, lookback_bars=2)
    assert all(sb.kind is HigherTfBarKind.COMPLETED for sb in state_bars)
    assert state_bars[-1].bar.end <= decision.timestamp


def test_config_default_uses_partial_higher_tf_bars() -> None:
    cfg = StateConfig()
    assert cfg.use_completed_higher_tf_bars_only is False
    assert cfg.use_incomplete_higher_tf_bars is True
