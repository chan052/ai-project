"""Reward computation context — future path and past sigma inputs only.

State features (MTF windows, indicators, etc.) must **not** be passed here.
Only the minimal scalars/series required for reward components are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.temporal import TemporalSplit, WindowSpec


@dataclass(frozen=True)
class RewardContext:
    """Inputs for reward at 3m decision index ``t_index``.

    ``future_closes``:
        Close prices at t+1 .. t+reward_horizon (inclusive).

    ``past_closes_for_sigma``:
        Historical closes at or before ``t`` for sigma_market_t only.
        Must not include any future bar.

    ``price_at_t``:
        Close at decision time ``t`` (reference for returns / excursions).
    """

    t_index: int
    price_at_t: float
    future_closes: tuple[float, ...]
    past_closes_for_sigma: tuple[float, ...]
    reward_horizon: int = 10

    def __post_init__(self) -> None:
        if self.price_at_t <= 0:
            raise ValueError("price_at_t must be positive")
        if self.t_index < 0:
            raise ValueError("t_index must be non-negative")
        if len(self.future_closes) != self.reward_horizon:
            raise ValueError(
                f"future_closes length must be {self.reward_horizon}, "
                f"got {len(self.future_closes)}"
            )
        if not self.past_closes_for_sigma:
            raise ValueError("past_closes_for_sigma must be non-empty")

    def validate_temporal_causality(self) -> None:
        """Ensure reward indices align with ``t_index`` on the 3m timeline."""
        split = TemporalSplit(
            t_index=self.t_index,
            spec=WindowSpec(reward_horizon=self.reward_horizon),
        )
        series_length = self.t_index + 1 + self.reward_horizon
        split.assert_valid_series_length(series_length)
        reward_idx = split.reward_indices()
        assert reward_idx is not None
        split.assert_no_past_in_reward(reward_idx)

    @property
    def future_prices(self) -> tuple[float, ...]:
        """Price path including anchor at t: [price_at_t, future_closes...]."""
        return (self.price_at_t,) + self.future_closes

    def per_step_simple_returns(self) -> tuple[float, ...]:
        """Per-step simple returns r_k for k=1..H (candidate — TODO finalize)."""
        prices = self.future_prices
        returns: list[float] = []
        for k in range(1, len(prices)):
            prev = prices[k - 1]
            if prev == 0:
                raise ValueError("zero price in path")
            returns.append((prices[k] - prev) / prev)
        return tuple(returns)

    def per_step_log_returns(self) -> tuple[float, ...]:
        """Log returns per step (candidate — TODO finalize)."""
        import math

        prices = self.future_prices
        returns: list[float] = []
        for k in range(1, len(prices)):
            prev = prices[k - 1]
            if prev <= 0 or prices[k] <= 0:
                raise ValueError("non-positive price in log return path")
            returns.append(math.log(prices[k] / prev))
        return tuple(returns)

    def horizon_simple_return(self) -> float:
        """Cumulative simple return over reward horizon."""
        last = self.future_closes[-1]
        return (last - self.price_at_t) / self.price_at_t
