"""Shared types for temporal alignment and RL actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional

import pandas as pd


class Timeframe(str, Enum):
    """Supported timeframes for P1 multi-timeframe state."""

    M3 = "3m"
    H1 = "1h"
    H4 = "4h"

    @property
    def is_decision_timeframe(self) -> bool:
        return self is Timeframe.M3

    @property
    def is_higher_timeframe(self) -> bool:
        return self in (Timeframe.H1, Timeframe.H4)


class Action(IntEnum):
    """P1 action space — direction judgment at decision time t."""

    LONG = 0
    HOLD = 1
    SHORT = 2


@dataclass(frozen=True)
class BarIndex:
    """Index of a single bar within a timeframe-specific series."""

    timeframe: Timeframe
    index: int


@dataclass(frozen=True)
class OHLCVBar:
    """Single OHLCV bar with explicit time boundaries.

    Convention: ``[start, end)`` — ``end`` is the bar close / first moment
    after the bar period. A bar is *completed* at decision time ``t`` when
    ``end <= t``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def is_completed_at(self, decision_time: pd.Timestamp) -> bool:
        """Return True iff this bar has fully closed by ``decision_time``."""
        return self.end <= decision_time


@dataclass(frozen=True)
class DecisionTime:
    """3m decision timestamp — all MTF alignment is relative to this instant."""

    timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, pd.Timestamp):
            object.__setattr__(self, "timestamp", pd.Timestamp(self.timestamp))


@dataclass(frozen=True)
class TimeframeWindow:
    """Inclusive index range ``[start_index, end_index]`` for one timeframe."""

    timeframe: Timeframe
    start_index: int
    end_index: int
    decision_time: DecisionTime

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.end_index < self.start_index:
            raise ValueError("end_index must be >= start_index")

    @property
    def length(self) -> int:
        return self.end_index - self.start_index + 1

    def as_slice(self) -> slice:
        return slice(self.start_index, self.end_index + 1)


@dataclass(frozen=True)
class MultiTimeframeWindows:
    """State windows for all P1 timeframes at a single 3m decision time."""

    decision_time: DecisionTime
    window_3m: TimeframeWindow
    window_1h: TimeframeWindow
    window_4h: TimeframeWindow

    def by_timeframe(self, timeframe: Timeframe) -> TimeframeWindow:
        mapping = {
            Timeframe.M3: self.window_3m,
            Timeframe.H1: self.window_1h,
            Timeframe.H4: self.window_4h,
        }
        return mapping[timeframe]


def bars_from_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    start_col: str = "start",
    end_col: str = "end",
) -> tuple[OHLCVBar, ...]:
    """Convert an OHLCV DataFrame into immutable ``OHLCVBar`` tuples."""
    required = {start_col, end_col, "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    bars: list[OHLCVBar] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        bars.append(
            OHLCVBar(
                start=pd.Timestamp(row_dict[start_col]),
                end=pd.Timestamp(row_dict[end_col]),
                open=float(row_dict["open"]),
                high=float(row_dict["high"]),
                low=float(row_dict["low"]),
                close=float(row_dict["close"]),
                volume=float(row_dict["volume"]),
            )
        )
    return tuple(bars)
