"""Multi-timeframe temporal alignment for P1 state construction.

P1 causality rule (Phase 0 — confirmed):
    At 3m decision time ``t``:
    - 3m state may include bars through the decision bar (index ``t``).
    - 1H and 4H state includes **only fully completed** bars whose ``end``
      timestamp is ``<= decision_time``.
    - Any 1H/4H bar still in progress at ``t`` is excluded entirely; its
      OHLCV must not appear in state in any form.

Example (user-specified):
    ``decision_time = 14:27``
    - The 14:00–14:59 1H bar is **excluded** (incomplete at 14:27).
    - The last usable 1H bar is 13:00–13:59 (closes at 14:00).

Future ablation:
    ``use_incomplete_higher_tf_bars=True`` is reserved but **not implemented**
    in Phase 0; requesting it raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class IncompleteHigherTimeframeBarError(ValueError):
    """Raised when alignment would include an incomplete higher-TF bar."""


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
        # Decision occurs at the close of the 3m bar at ``t_index``.
        return DecisionTime(timestamp=bars_3m[t_index].end)

    def last_available_bar_index(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
    ) -> int:
        """Return the index of the last bar usable in state at ``decision_time``."""
        if self._state_config.use_incomplete_higher_tf_bars:
            raise NotImplementedError(
                "use_incomplete_higher_tf_bars=True is a future ablation candidate; "
                "Phase 0 only supports completed higher-TF bars."
            )

        bars = self._bars(timeframe)
        if not bars:
            raise ValueError(f"No bars available for timeframe {timeframe.value}")

        t = decision_time.timestamp

        if timeframe is Timeframe.M3:
            candidates = [i for i, bar in enumerate(bars) if bar.end <= t]
            if not candidates:
                raise ValueError(
                    f"No completed 3m bar at or before decision_time={t}"
                )
            return candidates[-1]

        # Higher TF: strictly completed bars only (end <= decision_time).
        candidates = [i for i, bar in enumerate(bars) if bar.is_completed_at(t)]
        if not candidates:
            raise ValueError(
                f"No completed {timeframe.value} bar at decision_time={t}"
            )
        return candidates[-1]

    def _lookback_for(self, timeframe: Timeframe) -> Optional[int]:
        key = timeframe.value
        tf_cfg = self._state_config.timeframes.get(key)
        if tf_cfg is None:
            return None
        return tf_cfg.lookback_bars

    def state_window(
        self,
        timeframe: Timeframe,
        decision_time: DecisionTime,
        *,
        lookback_bars: Optional[int] = None,
    ) -> TimeframeWindow:
        """Build inclusive ``[start_index, end_index]`` past-only state window."""
        end_index = self.last_available_bar_index(timeframe, decision_time)
        lookback = lookback_bars if lookback_bars is not None else self._lookback_for(timeframe)

        if lookback is None:
            raise ValueError(
                f"lookback_bars for {timeframe.value} is not set; "
                "pass explicitly or configure state.timeframes"
            )
        if lookback <= 0:
            raise ValueError("lookback_bars must be positive")

        start_index = max(0, end_index - lookback + 1)
        window = TimeframeWindow(
            timeframe=timeframe,
            start_index=start_index,
            end_index=end_index,
            decision_time=decision_time,
        )
        self.validate_window(window)
        return window

    def align_at_3m_index(
        self,
        t_index: int,
        *,
        lookback_3m: Optional[int] = None,
        lookback_1h: Optional[int] = None,
        lookback_4h: Optional[int] = None,
    ) -> MultiTimeframeWindows:
        """Compute all timeframe state windows for 3m decision index ``t_index``."""
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

    def validate_window(self, window: TimeframeWindow) -> None:
        """Assert window obeys P1 causality rules."""
        bars = self._bars(window.timeframe)
        t = window.decision_time.timestamp

        if window.end_index >= len(bars):
            raise IndexError(
                f"{window.timeframe.value} end_index={window.end_index} "
                f"out of range (len={len(bars)})"
            )

        for idx in range(window.start_index, window.end_index + 1):
            bar = bars[idx]
            if window.timeframe.is_higher_timeframe:
                if not bar.is_completed_at(t):
                    raise IncompleteHigherTimeframeBarError(
                        f"Incomplete {window.timeframe.value} bar at index {idx} "
                        f"([{bar.start}, {bar.end})) included in state at "
                        f"decision_time={t}. Higher-TF bars must satisfy end <= t."
                    )
            else:
                if bar.end > t:
                    raise ValueError(
                        f"3m bar at index {idx} ends at {bar.end}, "
                        f"after decision_time={t}"
                    )

    def bar_end_times_in_window(self, window: TimeframeWindow) -> tuple[pd.Timestamp, ...]:
        bars = self._bars(window.timeframe)
        return tuple(bars[i].end for i in range(window.start_index, window.end_index + 1))


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
