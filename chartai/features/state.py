"""Multi-timeframe state representation at 3m decision time t."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.config import StateConfig
from chartai.core.types import (
    DecisionTime,
    MultiTimeframeWindows,
    OHLCVBar,
    Timeframe,
    TimeframeWindow,
)
from chartai.data.mtf_aligner import MultiTimeframeAligner, StateBar


@dataclass(frozen=True)
class TimeframeStateSlice:
    """Past-only OHLCV bars for one timeframe branch."""

    timeframe: Timeframe
    window: TimeframeWindow
    bars: tuple[OHLCVBar, ...]
    state_bars: tuple[StateBar, ...] = ()

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(b.close for b in self.bars)

    @property
    def bar_end_times(self) -> tuple:
        return tuple(b.end for b in self.bars)

    @property
    def has_partial_bar(self) -> bool:
        from chartai.data.mtf_aligner import HigherTfBarKind

        return any(sb.kind is HigherTfBarKind.PARTIAL for sb in self.state_bars)


@dataclass(frozen=True)
class MultiTimeframeState:
    """P1 state at decision time t — past-only across 3m, 1H, 4H."""

    t_index: int
    decision_time: DecisionTime
    windows: MultiTimeframeWindows
    slice_3m: TimeframeStateSlice
    slice_1h: TimeframeStateSlice
    slice_4h: TimeframeStateSlice

    def by_timeframe(self, timeframe: Timeframe) -> TimeframeStateSlice:
        mapping = {
            Timeframe.M3: self.slice_3m,
            Timeframe.H1: self.slice_1h,
            Timeframe.H4: self.slice_4h,
        }
        return mapping[timeframe]

    def fingerprint(self) -> tuple:
        """Deterministic tuple for causality tests — OHLCV closes and bar ends."""
        return (
            self.t_index,
            self.decision_time.timestamp,
            self.slice_3m.closes,
            tuple(b.end for b in self.slice_3m.bars),
            self.slice_1h.closes,
            tuple(b.end for b in self.slice_1h.bars),
            self.slice_4h.closes,
            tuple(b.end for b in self.slice_4h.bars),
        )


class StateBuilder:
    """Build past-only MTF state at 3m decision index ``t_index``.

    Delegates all timestamp alignment to :class:`MultiTimeframeAligner`.
    Does not implement separate alignment logic.
    """

    def __init__(
        self,
        aligner: MultiTimeframeAligner,
        *,
        state_config: StateConfig | None = None,
    ) -> None:
        self._aligner = aligner
        self._state_config = state_config or aligner.state_config

    @property
    def aligner(self) -> MultiTimeframeAligner:
        return self._aligner

    def build(self, t_index: int) -> MultiTimeframeState:
        windows = self._aligner.align_at_3m_index(t_index)
        return MultiTimeframeState(
            t_index=t_index,
            decision_time=windows.decision_time,
            windows=windows,
            slice_3m=self._slice_for(Timeframe.M3, windows.window_3m),
            slice_1h=self._slice_for(Timeframe.H1, windows.window_1h),
            slice_4h=self._slice_for(Timeframe.H4, windows.window_4h),
        )

    def _slice_for(self, timeframe: Timeframe, window: TimeframeWindow) -> TimeframeStateSlice:
        lookback = self._aligner._lookback_for(timeframe)
        if lookback is None:
            lookback = window.end_index - window.start_index + 1
        state_bars = self._aligner.state_bars(
            timeframe, window.decision_time, lookback_bars=lookback
        )
        return TimeframeStateSlice(
            timeframe=timeframe,
            window=window,
            bars=tuple(sb.bar for sb in state_bars),
            state_bars=state_bars,
        )
