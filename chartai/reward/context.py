"""Reward computation context — future path and past sigma inputs only.

State features (MTF windows, indicators, etc.) must **not** be passed here.
Only the minimal scalars/series required for reward components are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.temporal import TemporalSplit, WindowSpec


@dataclass(frozen=True)
class RewardContext:
    """Inputs for F-target computation at 3m decision index ``t_index``.

    ``future_closes`` / ``future_highs`` / ``future_lows``:
        OHLC at t+1 .. t+reward_horizon (inclusive).

    ``past_closes_for_sigma``:
        Historical closes at or before ``t`` (legacy / optional diagnostics).

    ``price_at_t``:
        Close at decision time ``t`` (reference for returns / excursions).
    """

    t_index: int
    price_at_t: float
    future_closes: tuple[float, ...]
    future_highs: tuple[float, ...]
    future_lows: tuple[float, ...]
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
        if len(self.future_highs) != self.reward_horizon:
            raise ValueError(
                f"future_highs length must be {self.reward_horizon}, "
                f"got {len(self.future_highs)}"
            )
        if len(self.future_lows) != self.reward_horizon:
            raise ValueError(
                f"future_lows length must be {self.reward_horizon}, "
                f"got {len(self.future_lows)}"
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
        """Close path including anchor at t: [price_at_t, future_closes...]."""
        return (self.price_at_t,) + self.future_closes

    def return_from_t(self, k: int) -> float:
        """Simple return R_k = (C_{t+k} - C_t) / C_t for k in 1..reward_horizon."""
        if k < 1 or k > self.reward_horizon:
            raise ValueError(f"k must be in 1..{self.reward_horizon}, got {k}")
        return (self.future_closes[k - 1] - self.price_at_t) / self.price_at_t

    def position_return_at_n(self, action_sign: float, n: int) -> float:
        """Direction-aligned return at t+n: sign * R_n."""
        return action_sign * self.return_from_t(n)

    def per_step_simple_returns(self) -> tuple[float, ...]:
        """Bar-to-bar simple returns (legacy helper — Path uses :meth:`return_from_t`)."""
        prices = self.future_prices
        returns: list[float] = []
        for k in range(1, len(prices)):
            prev = prices[k - 1]
            if prev == 0:
                raise ValueError("zero price in path")
            returns.append((prices[k] - prev) / prev)
        return tuple(returns)

    def per_step_log_returns(self) -> tuple[float, ...]:
        """Log returns per step (legacy helper)."""
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
        """Cumulative simple return over full reward horizon."""
        return self.return_from_t(self.reward_horizon)
