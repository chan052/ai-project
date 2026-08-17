"""Build RewardContext from 3m bars at decision time t."""

from __future__ import annotations

from typing import Sequence

from chartai.core.temporal import TemporalSplit, WindowSpec
from chartai.core.types import OHLCVBar
from chartai.reward.config import RewardConfig
from chartai.reward.context import RewardContext
from chartai.reward.move_surprise import compute_sigma


class FutureContextBuilder:
    """Construct :class:`RewardContext` for F-target computation at ``t_index``.

    Uses future 3m OHLC ``t+1 .. t+reward_horizon`` and past-only sigma inputs.
    Does not reference MTF state features.
    """

    def __init__(
        self,
        bars_3m: Sequence[OHLCVBar],
        *,
        reward_horizon: int = 10,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._bars_3m = tuple(bars_3m)
        self._reward_horizon = reward_horizon
        self._reward_config = reward_config or RewardConfig(reward_horizon=reward_horizon)

    @property
    def reward_horizon(self) -> int:
        return self._reward_horizon

    def build(self, t_index: int) -> RewardContext:
        self._validate_t_index(t_index)
        bar_t = self._bars_3m[t_index]
        price_at_t = bar_t.close
        future_start = t_index + 1
        future_end = t_index + self._reward_horizon
        future_bars = tuple(self._bars_3m[i] for i in range(future_start, future_end + 1))
        past_closes = self._past_closes_for_sigma(t_index)

        ctx = RewardContext(
            t_index=t_index,
            price_at_t=price_at_t,
            future_closes=tuple(b.close for b in future_bars),
            future_highs=tuple(b.high for b in future_bars),
            future_lows=tuple(b.low for b in future_bars),
            past_closes_for_sigma=past_closes,
            reward_horizon=self._reward_horizon,
        )
        ctx.validate_temporal_causality()
        return ctx

    def sigma_at_t(self, t_index: int) -> float:
        """Expose sigma_market_t for causality tests — past-only."""
        past = self._past_closes_for_sigma(t_index)
        surprise_cfg = self._reward_config.surprise
        return compute_sigma(
            past,
            surprise_cfg.sigma_method,
            surprise_cfg.sigma_window,
        )

    def fingerprint(self, t_index: int) -> tuple:
        ctx = self.build(t_index)
        return (
            ctx.t_index,
            ctx.price_at_t,
            ctx.future_closes,
            ctx.future_highs,
            ctx.future_lows,
            ctx.past_closes_for_sigma,
        )

    def _past_closes_for_sigma(self, t_index: int) -> tuple[float, ...]:
        """Closes at indices ``0 .. t_index`` inclusive — no future bars."""
        return tuple(self._bars_3m[i].close for i in range(0, t_index + 1))

    def _validate_t_index(self, t_index: int) -> None:
        if t_index < 0 or t_index >= len(self._bars_3m):
            raise IndexError(
                f"t_index={t_index} out of range for 3m series length {len(self._bars_3m)}"
            )
        split = TemporalSplit(
            t_index=t_index,
            spec=WindowSpec(reward_horizon=self._reward_horizon),
        )
        split.assert_valid_series_length(len(self._bars_3m))

    def reward_indices_used(self, t_index: int) -> range:
        split = TemporalSplit(
            t_index=t_index,
            spec=WindowSpec(reward_horizon=self._reward_horizon),
        )
        indices = split.reward_indices()
        assert indices is not None
        return indices
