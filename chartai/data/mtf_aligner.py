"""Multi-timeframe temporal alignment for P1 state construction.

P1 MTF alignment (confirmed):
    At 3m decision time ``t`` (3m bar close):

    **3m:** bars through the decision bar (``end <= t``).

    **1H / 4H:** completed higher-TF bars with ``end <= t`` **plus** a partial bar
    for the in-progress interval when ``t`` falls strictly inside ``[start, end)``.
    Partial OHLCV is aggregated from 3m bars with ``end <= t`` only.

    When ``t`` equals a higher-TF bar boundary (``t == bar.end``), that bar is
    treated as **completed** — no partial bar for the next interval.

Example (``t = 14:27``):
    - 1H completed: 13:00–14:00
    - 1H partial:   14:00–15:00 bucket, OHLCV from 14:00..14:27 3m data only
    - 4H completed: 08:00–12:00
    - 4H partial:   12:00–16:00 bucket, OHLCV from 12:00..14:27 3m data only
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import pandas as pd

from chartai.core.config import StateConfig
from chartai.core.types import (
    DecisionTime,
    MultiTimeframeWindows,
    OHLCVBar,
    Timeframe,
    TimeframeWindow,
)


class HigherTfBarKind(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"


@dataclass(frozen=True)
class StateBar:
    """OHLCV bar in MTF state with completed vs partial distinction."""

    bar: OHLCVBar
    kind: HigherTfBarKind
    native_index: int | None = None  # index in native HTF series when completed


class IncompleteHigherTimeframeBarError(ValueError):
    """Raised when native HTF bar data would leak future information."""


class PartialBarBuildError(ValueError):
    """Raised when partial bar cannot be built from available 3m data."""


_HTF_DURATIONS = {
    Timeframe.H1: pd.Timedelta(hours=1),
    Timeframe.H4: pd.Timedelta(hours=4),
}


def _duration_for(timeframe: Timeframe) -> pd.Timedelta:
    if timeframe is Timeframe.M3:
        return pd.Timedelta(minutes=3)
    return _HTF_DURATIONS[timeframe]


class MultiTimeframeAligner:
    """Align past-only state windows across 3m, 1H, and 4H at decision time."""

    def __init__(
        self,
        *,
        bars_3m: Sequence[OHLCVBar],
        bars_1h: Sequence[OHLCVBar],
        bars_4h: Sequence[OHLCVBar],
        state_config: Optional[StateConfig] = None,
    ) -> None:
        self._series = {
            Timeframe.M3: tuple(bars_3m),
            Timeframe.H1: tuple(bars_1h),
            Timeframe.H4: tuple(bars_4h),
        }
        self._state_config = state_config or StateConfig()
        self._validate_series_order()

    @property
    def state_config(self) -> StateConfig:
        return self._state_config

    def _validate_series_order(self) -> None:
        for timeframe, bars in self._series.items():
            for prev, nxt in zip(bars, bars[1:]):
                if nxt.start < prev.start:
                    raise ValueError(f"{timeframe.value} bars must be sorted by start time")
                if nxt.start < prev.end:
                    raise ValueError(
                        f"{timeframe.value} bars overlap: {prev.end} > {nxt.start}"
                    )

    def _bars(self, timeframe: Timeframe) -> tuple[OHLCVBar, ...]:
        return self._series[timeframe]

    def decision_time_at_3m_index(self, t_index: int) -> DecisionTime:
        bars_3m = self._bars(Timeframe.M3)
        if t_index < 0 or t_index >= len(bars_3m):
            raise IndexError(f"3m t_index={t_index} out of range [0, {len(bars_3m) - 1}]")
        return DecisionTime(timestamp=bars_3m[t_index].end)

    def completed_higher_tf_bar_indices(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
    ) -> tuple[int, ...]:
        """Native indices of fully completed higher-TF bars (``end <= t``)."""
        if timeframe is Timeframe.M3:
            raise ValueError("completed_higher_tf_bar_indices applies to 1H/4H only")
        bars = self._bars(timeframe)
        t = decision_time.timestamp
        return tuple(i for i, bar in enumerate(bars) if bar.is_completed_at(t))

    def in_progress_higher_tf_interval(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
    ) -> OHLCVBar | None:
        """Return the native HTF interval ``[start, end)`` containing ``t`` strictly inside.

        Returns ``None`` when ``t`` sits on a boundary (``t == bar.end``) — the bar
        is completed instead and no partial bar is needed for the next interval.
        """
        if timeframe is Timeframe.M3:
            return None
        t = decision_time.timestamp
        for bar in self._bars(timeframe):
            # Strict interior: t == bar.start is the new interval boundary (no partial yet).
            if bar.start < t < bar.end:
                return bar
        return None

    def contributing_3m_bars_for_interval(
        self,
        interval_start: pd.Timestamp,
        interval_end: pd.Timestamp,
        decision_time: DecisionTime,
    ) -> tuple[OHLCVBar, ...]:
        """3m bars in ``[interval_start, interval_end)`` with ``end <= decision_time``."""
        t = decision_time.timestamp
        if t > interval_end:
            raise ValueError("decision_time exceeds interval_end")
        selected = [
            b
            for b in self._bars(Timeframe.M3)
            if b.start >= interval_start
            and b.end <= interval_end
            and b.end <= t
        ]
        return tuple(selected)

    def build_partial_bar(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        *,
        interval: OHLCVBar | None = None,
    ) -> OHLCVBar | None:
        """Aggregate partial higher-TF OHLCV from 3m bars through ``decision_time``."""
        if timeframe is Timeframe.M3:
            raise ValueError("build_partial_bar applies to 1H/4H only")

        bucket = interval or self.in_progress_higher_tf_interval(timeframe, decision_time)
        if bucket is None:
            return None

        m3_bars = self.contributing_3m_bars_for_interval(
            bucket.start, bucket.end, decision_time
        )
        if not m3_bars:
            raise PartialBarBuildError(
                f"No 3m bars available for partial {timeframe.value} "
                f"[{bucket.start}, {bucket.end}) at decision_time={decision_time.timestamp}"
            )

        return OHLCVBar(
            start=bucket.start,
            end=bucket.end,
            open=m3_bars[0].open,
            high=max(b.high for b in m3_bars),
            low=min(b.low for b in m3_bars),
            close=m3_bars[-1].close,
            volume=sum(b.volume for b in m3_bars),
        )

    def last_available_bar_index(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
    ) -> int:
        """Last usable bar index at ``decision_time`` (completed or partial bucket)."""
        if timeframe is Timeframe.M3:
            bars = self._bars(Timeframe.M3)
            t = decision_time.timestamp
            candidates = [i for i, bar in enumerate(bars) if bar.end <= t]
            if not candidates:
                raise ValueError(f"No completed 3m bar at or before decision_time={t}")
            return candidates[-1]

        completed = self.completed_higher_tf_bar_indices(timeframe, decision_time)
        if self.in_progress_higher_tf_interval(timeframe, decision_time) is not None:
            if not completed and self.build_partial_bar(timeframe, decision_time) is None:
                raise ValueError(
                    f"No completed or partial {timeframe.value} bar at "
                    f"decision_time={decision_time.timestamp}"
                )
            return len(self._bars(timeframe))  # sentinel: partial extends beyond native
        if not completed:
            raise ValueError(
                f"No completed {timeframe.value} bar at decision_time={decision_time.timestamp}"
            )
        return completed[-1]

    def _lookback_for(self, timeframe: Timeframe) -> Optional[int]:
        key = timeframe.value
        tf_cfg = self._state_config.timeframes.get(key)
        if tf_cfg is None:
            return None
        return tf_cfg.lookback_bars

    def state_bars(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        *,
        lookback_bars: Optional[int] = None,
    ) -> tuple[StateBar, ...]:
        """Build state bars for one timeframe — completed history + optional partial."""
        lookback = lookback_bars if lookback_bars is not None else self._lookback_for(timeframe)
        if lookback is None:
            raise ValueError(
                f"lookback_bars for {timeframe.value} is not set; "
                "pass explicitly or configure state.timeframes"
            )
        if lookback <= 0:
            raise ValueError("lookback_bars must be positive")

        if timeframe is Timeframe.M3:
            end_index = self.last_available_bar_index(timeframe, decision_time)
            start_index = max(0, end_index - lookback + 1)
            native = self._bars(timeframe)
            return tuple(
                StateBar(bar=native[i], kind=HigherTfBarKind.COMPLETED, native_index=i)
                for i in range(start_index, end_index + 1)
            )

        if self._state_config.use_completed_higher_tf_bars_only:
            return self._state_bars_completed_only(timeframe, decision_time, lookback)

        return self._state_bars_with_partial(timeframe, decision_time, lookback)

    def _state_bars_completed_only(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        lookback: int,
    ) -> tuple[StateBar, ...]:
        """Deprecated Phase-0 mode — completed native HTF bars only."""
        completed = self.completed_higher_tf_bar_indices(timeframe, decision_time)
        if not completed:
            raise ValueError(
                f"No completed {timeframe.value} bars at decision_time={decision_time.timestamp}"
            )
        window_indices = completed[-lookback:]
        native = self._bars(timeframe)
        return tuple(
            StateBar(bar=native[i], kind=HigherTfBarKind.COMPLETED, native_index=i)
            for i in window_indices
        )

    def _state_bars_with_partial(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        lookback: int,
    ) -> tuple[StateBar, ...]:
        native = self._bars(timeframe)
        completed_indices = list(self.completed_higher_tf_bar_indices(timeframe, decision_time))
        partial = self.build_partial_bar(timeframe, decision_time)

        items: list[StateBar] = [
            StateBar(bar=native[i], kind=HigherTfBarKind.COMPLETED, native_index=i)
            for i in completed_indices
        ]
        if partial is not None:
            items.append(StateBar(bar=partial, kind=HigherTfBarKind.PARTIAL, native_index=None))

        if not items:
            raise ValueError(
                f"No {timeframe.value} state bars at decision_time={decision_time.timestamp}"
            )
        selected = items[-lookback:]
        self._validate_state_bars(timeframe, decision_time, selected)
        return tuple(selected)

    def _validate_state_bars(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        bars: Sequence[StateBar],
    ) -> None:
        t = decision_time.timestamp
        for item in bars:
            if item.kind is HigherTfBarKind.COMPLETED:
                if not item.bar.is_completed_at(t):
                    raise IncompleteHigherTimeframeBarError(
                        f"Completed {timeframe.value} bar [{item.bar.start}, {item.bar.end}) "
                        f"not closed at decision_time={t}"
                    )
            elif item.kind is HigherTfBarKind.PARTIAL:
                m3_used = self.contributing_3m_bars_for_interval(
                    item.bar.start, item.bar.end, decision_time
                )
                if not m3_used:
                    raise PartialBarBuildError("Partial bar has no contributing 3m bars")
                if any(b.end > t for b in m3_used):
                    raise IncompleteHigherTimeframeBarError(
                        "Partial bar uses 3m data after decision_time"
                    )
                if item.bar.close != m3_used[-1].close:
                    raise PartialBarBuildError(
                        "Partial bar close must equal last contributing 3m close at t"
                    )

    def state_window(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        *,
        lookback_bars: Optional[int] = None,
    ) -> TimeframeWindow:
        """Inclusive window metadata — indices refer to native series when applicable."""
        state_bars = self.state_bars(
            timeframe, decision_time, lookback_bars=lookback_bars
        )
        if timeframe is Timeframe.M3:
            end_index = state_bars[-1].native_index
            assert end_index is not None
            start_index = state_bars[0].native_index
            assert start_index is not None
        else:
            completed_in_window = [
                sb.native_index for sb in state_bars if sb.kind is HigherTfBarKind.COMPLETED
            ]
            start_index = completed_in_window[0] if completed_in_window else 0
            end_index = (
                completed_in_window[-1]
                if completed_in_window
                else self.last_available_bar_index(timeframe, decision_time)
            )

        return TimeframeWindow(
            timeframe=timeframe,
            start_index=start_index,
            end_index=end_index,
            decision_time=decision_time,
        )

    def align_at_3m_index(
        self,
        t_index: int,
        *,
        lookback_3m: Optional[int] = None,
        lookback_1h: Optional[int] = None,
        lookback_4h: Optional[int] = None,
    ) -> MultiTimeframeWindows:
        decision_time = self.decision_time_at_3m_index(t_index)
        return MultiTimeframeWindows(
            decision_time=decision_time,
            window_3m=self.state_window(
                Timeframe.M3, decision_time, lookback_bars=lookback_3m
            ),
            window_1h=self.state_window(
                Timeframe.H1, decision_time, lookback_bars=lookback_1h
            ),
            window_4h=self.state_window(
                Timeframe.H4, decision_time, lookback_bars=lookback_4h
            ),
        )

    def bars_in_window(self, window: TimeframeWindow) -> tuple[OHLCVBar, ...]:
        """Return OHLCV bars for an aligned window (includes partial when applicable)."""
        lookback = self._lookback_for(window.timeframe)
        if lookback is None:
            lookback = window.end_index - window.start_index + 1
        state_bars = self.state_bars(
            window.timeframe, window.decision_time, lookback_bars=lookback
        )
        return tuple(sb.bar for sb in state_bars)

    def state_bars_in_window(self, window: TimeframeWindow) -> tuple[StateBar, ...]:
        lookback = self._lookback_for(window.timeframe)
        if lookback is None:
            lookback = window.end_index - window.start_index + 1
        return self.state_bars(window.timeframe, window.decision_time, lookback_bars=lookback)

    def validate_window(self, window: TimeframeWindow) -> None:
        self.state_bars(
            window.timeframe,
            window.decision_time,
            lookback_bars=max(1, window.end_index - window.start_index + 1),
        )

    def bar_end_times_in_window(self, window: TimeframeWindow) -> tuple[pd.Timestamp, ...]:
        return tuple(b.end for b in self.bars_in_window(window))


@dataclass(frozen=True)
class BarSeriesBuilder:
    """Test/helper utility to build regularly spaced OHLCV bars."""

    timeframe: Timeframe

    def build_hourly_bars(self, hour_starts: Sequence[pd.Timestamp]) -> tuple[OHLCVBar, ...]:
        delta = pd.Timedelta(hours=1)
        return tuple(
            OHLCVBar(
                start=ts,
                end=ts + delta,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0,
            )
            for ts in hour_starts
        )

    def build_3m_bars(self, starts: Sequence[pd.Timestamp]) -> tuple[OHLCVBar, ...]:
        delta = pd.Timedelta(minutes=3)
        return tuple(
            OHLCVBar(
                start=ts,
                end=ts + delta,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0,
            )
            for ts in starts
        )

    def build_4h_bars(self, starts: Sequence[pd.Timestamp]) -> tuple[OHLCVBar, ...]:
        delta = pd.Timedelta(hours=4)
        return tuple(
            OHLCVBar(
                start=ts,
                end=ts + delta,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=0.0,
            )
            for ts in starts
        )
