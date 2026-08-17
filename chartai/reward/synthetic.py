"""Deterministic synthetic price paths for reward unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chartai.reward.context import RewardContext


class SyntheticScenario(str, Enum):
    STEADY_UP = "steady_up"
    UP_THEN_DOWN = "up_then_down"
    DOWN_THEN_UP = "down_then_up"
    STEADY_DOWN = "steady_down"
    FLAT = "flat"
    FLAT_THEN_UP = "flat_then_up"
    UP_THEN_FLAT = "up_then_flat"
    FLAT_THEN_DOWN = "flat_then_down"
    DOWN_THEN_FLAT = "down_then_flat"
    QUIET_THEN_BIG_UP = "quiet_then_big_up"
    QUIET_THEN_BIG_DOWN = "quiet_then_big_down"
    OFFSETTING_SWING = "offsetting_swing"
    QUIET_FLAT = "quiet_flat"


@dataclass(frozen=True)
class SyntheticPath:
    name: str
    price_at_t: float
    future_closes: tuple[float, ...]
    past_closes_for_sigma: tuple[float, ...]
    future_highs: tuple[float, ...] | None = None
    future_lows: tuple[float, ...] | None = None
    t_index: int = 100
    reward_horizon: int = 10

    def __post_init__(self) -> None:
        n = len(self.future_closes)
        if self.future_highs is None:
            object.__setattr__(self, "future_highs", self.future_closes)
        if self.future_lows is None:
            object.__setattr__(self, "future_lows", self.future_closes)
        if len(self.future_highs) != n or len(self.future_lows) != n:
            raise ValueError("future OHLC lengths must match future_closes")

    def to_context(self) -> RewardContext:
        return RewardContext(
            t_index=self.t_index,
            price_at_t=self.price_at_t,
            future_closes=self.future_closes,
            future_highs=self.future_highs,  # type: ignore[arg-type]
            future_lows=self.future_lows,  # type: ignore[arg-type]
            past_closes_for_sigma=self.past_closes_for_sigma,
            reward_horizon=self.reward_horizon,
        )


def _quiet_past(base: float = 100.0, n: int = 30) -> tuple[float, ...]:
    return tuple(base + 0.01 * (i % 3 - 1) for i in range(n))


def _future_from_relative_moves(anchor: float, moves: list[float]) -> tuple[float, ...]:
    prices = [anchor]
    for m in moves:
        prices.append(prices[-1] * (1.0 + m))
    return tuple(prices[1:])


def _ohlc_from_closes(
    closes: tuple[float, ...],
    *,
    wick_up: float = 0.002,
    wick_down: float = 0.002,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    highs = tuple(c * (1.0 + wick_up) for c in closes)
    lows = tuple(c * (1.0 - wick_down) for c in closes)
    return highs, lows


def build_scenario(scenario: SyntheticScenario, *, horizon: int = 10) -> SyntheticPath:
    anchor = 100.0

    if scenario is SyntheticScenario.STEADY_UP:
        moves = [0.01] * horizon
        past = _quiet_past()
    elif scenario is SyntheticScenario.STEADY_DOWN:
        moves = [-0.01] * horizon
        past = _quiet_past()
    elif scenario is SyntheticScenario.UP_THEN_DOWN:
        moves = [0.02] * (horizon // 2) + [-0.02] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.DOWN_THEN_UP:
        moves = [-0.02] * (horizon // 2) + [0.02] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.FLAT:
        moves = [0.0] * horizon
        past = _quiet_past()
    elif scenario is SyntheticScenario.FLAT_THEN_UP:
        moves = [0.0] * (horizon // 2) + [0.03] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.UP_THEN_FLAT:
        moves = [0.03] * (horizon // 2) + [0.0] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.FLAT_THEN_DOWN:
        moves = [0.0] * (horizon // 2) + [-0.03] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.DOWN_THEN_FLAT:
        moves = [-0.03] * (horizon // 2) + [0.0] * (horizon - horizon // 2)
        past = _quiet_past()
    elif scenario is SyntheticScenario.QUIET_THEN_BIG_UP:
        moves = [0.0] * (horizon - 2) + [0.08, 0.08]
        past = _quiet_past()
    elif scenario is SyntheticScenario.QUIET_THEN_BIG_DOWN:
        moves = [0.0] * (horizon - 2) + [-0.08, -0.08]
        past = _quiet_past()
    elif scenario is SyntheticScenario.OFFSETTING_SWING:
        half = horizon // 2
        moves = [0.05] * half + [-0.05] * (horizon - half)
        past = _quiet_past()
    elif scenario is SyntheticScenario.QUIET_FLAT:
        moves = [0.0] * horizon
        past = _quiet_past(100.0, 50)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    closes = _future_from_relative_moves(anchor, moves)
    highs, lows = _ohlc_from_closes(closes)

    return SyntheticPath(
        name=scenario.value,
        price_at_t=anchor,
        future_closes=closes,
        future_highs=highs,
        future_lows=lows,
        past_closes_for_sigma=past,
        reward_horizon=horizon,
    )


def mae_adverse_long_path(*, horizon: int = 10) -> SyntheticPath:
    """LONG adverse path — dip in lows before recovery in close."""
    anchor = 100.0
    closes = tuple(100.0 + 0.5 * i for i in range(1, horizon + 1))
    lows = (95.0,) + tuple(99.0 + 0.5 * i for i in range(horizon - 1))
    highs = tuple(c * 1.001 for c in closes)
    return SyntheticPath(
        name="mae_adverse_long",
        price_at_t=anchor,
        future_closes=closes,
        future_highs=highs,
        future_lows=lows,
        past_closes_for_sigma=_quiet_past(),
        reward_horizon=horizon,
    )


def mae_adverse_short_path(*, horizon: int = 10) -> SyntheticPath:
    """SHORT adverse path — spike in highs before decline in close."""
    anchor = 100.0
    closes = tuple(100.0 - 0.5 * i for i in range(1, horizon + 1))
    highs = (105.0,) + tuple(100.5 - 0.5 * i for i in range(horizon - 1))
    lows = tuple(c * 0.999 for c in closes)
    return SyntheticPath(
        name="mae_adverse_short",
        price_at_t=anchor,
        future_closes=closes,
        future_highs=highs,
        future_lows=lows,
        past_closes_for_sigma=_quiet_past(),
        reward_horizon=horizon,
    )
