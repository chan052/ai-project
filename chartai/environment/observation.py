"""Observation adapter — bridge MultiTimeframeState to Gymnasium observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chartai.core.config import StateConfig
from chartai.core.types import OHLCVBar
from chartai.data.mtf_aligner import HigherTfBarKind
from chartai.features.state import MultiTimeframeState, TimeframeStateSlice


def action_from_env_int(action: int):
    """Map Gymnasium action integer to P1 :class:`Action`."""
    from chartai.core.types import Action

    try:
        return Action(action)
    except ValueError as exc:
        raise ValueError(f"Invalid action {action}; expected 0=LONG, 1=SHORT") from exc


def _bar_array(bars: tuple[OHLCVBar, ...]) -> np.ndarray:
    """Shape ``(n, 5)`` — open, high, low, close, volume."""
    if not bars:
        return np.zeros((0, 5), dtype=np.float64)
    return np.array(
        [[b.open, b.high, b.low, b.close, b.volume] for b in bars],
        dtype=np.float64,
    )


def _pad_ohlcv(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Right-align bars to fixed lookback length (most recent at end)."""
    out = np.zeros((target_n, 5), dtype=np.float64)
    n = min(target_n, arr.shape[0])
    if n > 0:
        out[-n:] = arr[-n:]
    return out


def _pad_flags(arr: np.ndarray, target_n: int) -> np.ndarray:
    out = np.zeros(target_n, dtype=np.float64)
    n = min(target_n, arr.shape[0])
    if n > 0:
        out[-n:] = arr[-n:]
    return out


def _bar_kind_flags(slice_: TimeframeStateSlice) -> np.ndarray:
    """Shape ``(n,)`` — 1.0 for partial, 0.0 for completed."""
    if not slice_.state_bars:
        return np.zeros(len(slice_.bars), dtype=np.float64)
    return np.array(
        [1.0 if sb.kind is HigherTfBarKind.PARTIAL else 0.0 for sb in slice_.state_bars],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class P1Observation:
    """Structured P1 observation — not a finalized policy-network tensor."""

    t_index: int
    decision_time_iso: str
    ohlcv_3m: np.ndarray
    ohlcv_1h: np.ndarray
    ohlcv_4h: np.ndarray
    partial_flag_1h: np.ndarray
    partial_flag_4h: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_index": np.int64(self.t_index),
            "3m": self.ohlcv_3m,
            "1h": self.ohlcv_1h,
            "4h": self.ohlcv_4h,
            "1h_partial_flags": self.partial_flag_1h,
            "4h_partial_flags": self.partial_flag_4h,
        }


class P1ObservationAdapter:
    """Convert :class:`MultiTimeframeState` to Gymnasium-friendly observations."""

    def __init__(self, state_config: StateConfig) -> None:
        self._state_config = state_config
        tf = state_config.timeframes
        self._lookback_3m = tf["3m"].lookback_bars
        self._lookback_1h = tf["1h"].lookback_bars
        self._lookback_4h = tf["4h"].lookback_bars
        if None in (self._lookback_3m, self._lookback_1h, self._lookback_4h):
            raise ValueError(
                "ObservationAdapter requires explicit lookback_bars for all timeframes"
            )

    @property
    def lookbacks(self) -> tuple[int, int, int]:
        return self._lookback_3m, self._lookback_1h, self._lookback_4h

    def from_state(self, state: MultiTimeframeState) -> P1Observation:
        raw_3m = _bar_array(state.slice_3m.bars)
        raw_1h = _bar_array(state.slice_1h.bars)
        raw_4h = _bar_array(state.slice_4h.bars)
        flags_1h = _bar_kind_flags(state.slice_1h)
        flags_4h = _bar_kind_flags(state.slice_4h)
        return P1Observation(
            t_index=state.t_index,
            decision_time_iso=state.decision_time.timestamp.isoformat(),
            ohlcv_3m=_pad_ohlcv(raw_3m, self._lookback_3m),
            ohlcv_1h=_pad_ohlcv(raw_1h, self._lookback_1h),
            ohlcv_4h=_pad_ohlcv(raw_4h, self._lookback_4h),
            partial_flag_1h=_pad_flags(flags_1h, self._lookback_1h),
            partial_flag_4h=_pad_flags(flags_4h, self._lookback_4h),
        )

    def to_observation(self, state: MultiTimeframeState) -> dict[str, Any]:
        return self.from_state(state).to_dict()

    def gymnasium_space(self):
        """Build ``Dict`` observation space from configured lookbacks."""
        from gymnasium import spaces

        def ohlcv_box(n: int):
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n, 5),
                dtype=np.float64,
            )

        def flag_box(n: int):
            return spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float64)

        return spaces.Dict(
            {
                "t_index": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(),
                    dtype=np.int64,
                ),
                "3m": ohlcv_box(self._lookback_3m),
                "1h": ohlcv_box(self._lookback_1h),
                "4h": ohlcv_box(self._lookback_4h),
                "1h_partial_flags": flag_box(self._lookback_1h),
                "4h_partial_flags": flag_box(self._lookback_4h),
            }
        )
