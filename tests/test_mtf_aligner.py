"""Tests for MultiTimeframeAligner — P1 MTF causality rules."""

from __future__ import annotations

import pandas as pd
import pytest

from chartai.core.config import StateConfig
from chartai.core.types import DecisionTime, OHLCVBar, Timeframe
from chartai.data.mtf_aligner import (
    BarSeriesBuilder,
    IncompleteHigherTimeframeBarError,
    MultiTimeframeAligner,
)


def test_14_27_excludes_in_progress_1h_bar(aligner_14_27: MultiTimeframeAligner) -> None:
    """At t=14:27, 14:00-14:59 1H bar is excluded; 13:00-13:59 bar is last."""
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1  # bar closing at 14:27
    decision_time = aligner_14_27.decision_time_at_3m_index(t_index)
    assert decision_time.timestamp == pd.Timestamp("2024-01-02 14:27")

    last_1h_idx = aligner_14_27.last_available_bar_index(Timeframe.H1, decision_time)
    bars_1h = aligner_14_27._bars(Timeframe.H1)
    last_bar = bars_1h[last_1h_idx]

    assert last_bar.start == pd.Timestamp("2024-01-02 13:00")
    assert last_bar.end == pd.Timestamp("2024-01-02 14:00")

    in_progress = OHLCVBar(
        start=pd.Timestamp("2024-01-02 14:00"),
        end=pd.Timestamp("2024-01-02 15:00"),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=0.0,
    )
    assert not in_progress.is_completed_at(decision_time.timestamp)
    assert last_bar.end <= decision_time.timestamp
    assert in_progress.end > decision_time.timestamp


def test_higher_tf_window_contains_only_completed_bars(
    aligner_14_27: MultiTimeframeAligner,
) -> None:
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1
    windows = aligner_14_27.align_at_3m_index(
        t_index, lookback_3m=4, lookback_1h=2, lookback_4h=1
    )
    t = windows.decision_time.timestamp

    for tf in (Timeframe.H1, Timeframe.H4):
        window = windows.by_timeframe(tf)
        bars = aligner_14_27._bars(tf)
        for idx in range(window.start_index, window.end_index + 1):
            assert bars[idx].end <= t, f"{tf.value} bar {idx} incomplete at {t}"


def test_changing_future_3m_bars_does_not_change_higher_tf_alignment() -> None:
    """MTF-2: future 3m bar OHLCV changes must not affect past alignment indices."""
    h1_starts = pd.date_range("2024-01-02 09:00", periods=8, freq="h")
    bars_1h = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(h1_starts)
    h4_starts = pd.date_range("2024-01-02 08:00", periods=4, freq="4h")
    bars_4h = BarSeriesBuilder(Timeframe.H4).build_4h_bars(h4_starts)

    m3_starts = pd.date_range("2024-01-02 13:54", periods=10, freq="3min")
    bars_3m_a = BarSeriesBuilder(Timeframe.M3).build_3m_bars(m3_starts)

    aligner_a = MultiTimeframeAligner(bars_3m=bars_3m_a, bars_1h=bars_1h, bars_4h=bars_4h)
    t_index = len(bars_3m_a) - 1
    win_a = aligner_a.align_at_3m_index(t_index, lookback_3m=4, lookback_1h=2, lookback_4h=1)

    # Append future 3m bars with different OHLCV (same timestamps extended).
    extra_starts = pd.date_range(m3_starts[-1] + pd.Timedelta(minutes=3), periods=5, freq="3min")
    extra_bars = tuple(
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
    )
    bars_3m_b = bars_3m_a + extra_bars
    aligner_b = MultiTimeframeAligner(bars_3m=bars_3m_b, bars_1h=bars_1h, bars_4h=bars_4h)
    win_b = aligner_b.align_at_3m_index(t_index, lookback_3m=4, lookback_1h=2, lookback_4h=1)

    assert win_a.window_1h == win_b.window_1h
    assert win_a.window_4h == win_b.window_4h
    assert win_a.window_3m == win_b.window_3m


def test_last_available_higher_tf_satisfies_end_lte_decision_time(
    aligner_14_27: MultiTimeframeAligner,
) -> None:
    bars_3m = aligner_14_27._bars(Timeframe.M3)
    t_index = len(bars_3m) - 1
    decision_time = aligner_14_27.decision_time_at_3m_index(t_index)

    for tf in (Timeframe.H1, Timeframe.H4):
        idx = aligner_14_27.last_available_bar_index(tf, decision_time)
        bar = aligner_14_27._bars(tf)[idx]
        assert bar.end <= decision_time.timestamp


def test_incomplete_higher_tf_flag_raises_not_implemented() -> None:
    cfg = StateConfig(use_incomplete_higher_tf_bars=True)
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        pd.date_range("2024-01-02 09:00", periods=3, freq="h")
    )
    m3 = BarSeriesBuilder(Timeframe.M3).build_3m_bars(
        pd.date_range("2024-01-02 09:00", periods=3, freq="3min")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        pd.date_range("2024-01-02 08:00", periods=2, freq="4h")
    )
    aligner = MultiTimeframeAligner(
        bars_3m=m3, bars_1h=h1, bars_4h=h4, state_config=cfg
    )
    decision = DecisionTime(timestamp=pd.Timestamp("2024-01-02 09:10"))
    with pytest.raises(NotImplementedError, match="future ablation"):
        aligner.last_available_bar_index(Timeframe.H1, decision)


def test_validate_window_rejects_incomplete_higher_tf_bar() -> None:
    h1 = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(
        [pd.Timestamp("2024-01-02 13:00"), pd.Timestamp("2024-01-02 14:00")]
    )
    m3 = BarSeriesBuilder(Timeframe.M3).build_3m_bars(
        pd.date_range("2024-01-02 14:18", periods=4, freq="3min")
    )
    h4 = BarSeriesBuilder(Timeframe.H4).build_4h_bars(
        [pd.Timestamp("2024-01-02 08:00")]
    )
    aligner = MultiTimeframeAligner(bars_3m=m3, bars_1h=h1, bars_4h=h4)
    decision = DecisionTime(timestamp=pd.Timestamp("2024-01-02 14:27"))

    # Force an invalid window that includes the in-progress 14:00-15:00 1H bar.
    from chartai.core.types import TimeframeWindow

    bad_window = TimeframeWindow(
        timeframe=Timeframe.H1,
        start_index=0,
        end_index=1,
        decision_time=decision,
    )
    with pytest.raises(IncompleteHigherTimeframeBarError):
        aligner.validate_window(bad_window)


def test_config_default_disallows_incomplete_higher_tf_bars() -> None:
    cfg = StateConfig()
    assert cfg.use_incomplete_higher_tf_bars is False
