"""MTF context encoding for conditional information audit (analysis-only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from chartai.core.types import Action
from chartai.features.state import MultiTimeframeState, TimeframeStateSlice
from chartai.reward.base import directional_sign


class TrendRegime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class HtfObservables:
    recent_return: float
    slope: float
    volatility: float
    favorable_occupancy: float
    momentum: float
    dist_from_low: float
    dist_from_high: float
    regime: TrendRegime


@dataclass(frozen=True)
class MtfContextSnapshot:
    pattern_key: tuple[int, ...]
    pattern_returns_norm: tuple[float, ...]
    h1: HtfObservables
    h4: HtfObservables
    h1_regime: TrendRegime
    h4_regime: TrendRegime
    interaction: str


def _simple_returns(closes: tuple[float, ...]) -> tuple[float, ...]:
    if len(closes) < 2:
        return ()
    return tuple((closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)))


def _regime_from_momentum(momentum: float, *, thr: float = 0.001) -> TrendRegime:
    if momentum > thr:
        return TrendRegime.BULLISH
    if momentum < -thr:
        return TrendRegime.BEARISH
    return TrendRegime.NEUTRAL


def encode_htf_slice(slice_tf: TimeframeStateSlice) -> HtfObservables:
    closes = slice_tf.closes
    rets = _simple_returns(closes)
    if not closes:
        return HtfObservables(0, 0, 0, 0, 0, 0.5, 0.5, TrendRegime.NEUTRAL)
    recent = rets[-1] if rets else 0.0
    momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] else 0.0
    vol = float(np.std(rets)) if len(rets) > 1 else 0.0
    fav = sum(1 for r in rets if r > 0) / len(rets) if rets else 0.0
    hi = max(closes)
    lo = min(closes)
    span = hi - lo
    dist_lo = (closes[-1] - lo) / span if span > 1e-15 else 0.5
    dist_hi = (hi - closes[-1]) / span if span > 1e-15 else 0.5
    xs = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(xs, closes, 1)[0] / closes[-1]) if len(closes) >= 2 else 0.0
    regime = _regime_from_momentum(momentum)
    return HtfObservables(
        recent_return=recent,
        slope=slope,
        volatility=vol,
        favorable_occupancy=fav,
        momentum=momentum,
        dist_from_low=dist_lo,
        dist_from_high=dist_hi,
        regime=regime,
    )


def compute_3m_pattern_key(
    past_returns: tuple[float, ...],
    *,
    n_levels: int = 5,
) -> tuple[int, ...]:
    """Discretize normalized return shape for matching (level indices 0..n_levels-1)."""
    if not past_returns:
        return ()
    arr = np.asarray(past_returns, dtype=float)
    scale = float(np.std(arr))
    if scale < 1e-12:
        scale = 1e-12
    z = arr / scale
    # map z to bins centered at 0
    half = n_levels // 2
    bins = []
    for v in z:
        idx = int(np.clip(np.round(v) + half, 0, n_levels - 1))
        bins.append(idx)
    return tuple(bins)


def past_3m_returns(bars_3m: tuple, t_index: int, lookback: int) -> tuple[float, ...]:
    start = max(1, t_index - lookback + 1)
    rets: list[float] = []
    for i in range(start, t_index + 1):
        prev = bars_3m[i - 1].close
        cur = bars_3m[i].close
        rets.append((cur - prev) / prev if prev else 0.0)
    return tuple(rets)


def encode_mtf_context(
    state: MultiTimeframeState,
    *,
    pattern_lookback: int = 8,
    n_pattern_levels: int = 5,
) -> MtfContextSnapshot:
    rets = past_3m_returns(state.slice_3m.bars, len(state.slice_3m.bars) - 1, pattern_lookback)
    # use full 3m native series via t_index — pattern from aligner window may be shorter
    return encode_mtf_context_from_series(
        state,
        pattern_returns=rets,
        n_pattern_levels=n_pattern_levels,
    )


def encode_mtf_context_at(
    bars_3m: tuple,
    state: MultiTimeframeState,
    t_index: int,
    *,
    pattern_lookback: int = 8,
    n_pattern_levels: int = 5,
) -> MtfContextSnapshot:
    rets = past_3m_returns(bars_3m, t_index, pattern_lookback)
    return encode_mtf_context_from_series(
        state,
        pattern_returns=rets,
        n_pattern_levels=n_pattern_levels,
    )


def encode_mtf_context_from_series(
    state: MultiTimeframeState,
    *,
    pattern_returns: tuple[float, ...],
    n_pattern_levels: int = 5,
) -> MtfContextSnapshot:
    h1 = encode_htf_slice(state.slice_1h)
    h4 = encode_htf_slice(state.slice_4h)
    scale = float(np.std(pattern_returns)) if pattern_returns else 1.0
    norm = tuple(r / max(scale, 1e-12) for r in pattern_returns)
    key = compute_3m_pattern_key(pattern_returns, n_levels=n_pattern_levels)
    interaction = f"{h1.regime.value}__{h4.regime.value}"
    return MtfContextSnapshot(
        pattern_key=key,
        pattern_returns_norm=norm,
        h1=h1,
        h4=h4,
        h1_regime=h1.regime,
        h4_regime=h4.regime,
        interaction=interaction,
    )


def interaction_label(h1: TrendRegime, h4: TrendRegime) -> str:
    return f"{h1.value}__{h4.value}"
