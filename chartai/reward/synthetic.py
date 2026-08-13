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
    HOLD_QUIET_VS_VOLATILE = "hold_quiet_vs_volatile"


@dataclass(frozen=True)
class SyntheticPath:
    name: str
    price_at_t: float
    future_closes: tuple[float, ...]
    past_closes_for_sigma: tuple[float, ...]
    t_index: int = 100
    reward_horizon: int = 10

    def to_context(self) -> RewardContext:
        return RewardContext(
            t_index=self.t_index,
            price_at_t=self.price_at_t,
            future_closes=self.future_closes,
            past_closes_for_sigma=self.past_closes_for_sigma,
            reward_horizon=self.reward_horizon,
        )


def _quiet_past(base: float = 100.0, n: int = 30) -> tuple[float, ...]:
    return tuple(base + 0.01 * (i % 3 - 1) for i in range(n))


def _volatile_past(base: float = 100.0, n: int = 30) -> tuple[float, ...]:
    return tuple(base * (1.0 + 0.02 * ((i % 5) - 2)) for i in range(n))


def _future_from_relative_moves(anchor: float, moves: list[float]) -> tuple[float, ...]:
    prices = [anchor]
    for m in moves:
        prices.append(prices[-1] * (1.0 + m))
    return tuple(prices[1:])


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
    elif scenario is SyntheticScenario.HOLD_QUIET_VS_VOLATILE:
        moves = [0.0] * horizon
        past = _quiet_past()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return SyntheticPath(
        name=scenario.value,
        price_at_t=anchor,
        future_closes=_future_from_relative_moves(anchor, moves),
        past_closes_for_sigma=past,
        reward_horizon=horizon,
    )


def hold_quatile_volatile_path(*, horizon: int = 10) -> SyntheticPath:
    """Path B: large mid-path swings with similar terminal price."""
    anchor = 100.0
    moves = [0.05, -0.095, 0.047, -0.047, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005]
    moves = moves[:horizon]
    return SyntheticPath(
        name="hold_volatile_mid",
        price_at_t=anchor,
        future_closes=_future_from_relative_moves(anchor, moves),
        past_closes_for_sigma=_quiet_past(),
        reward_horizon=horizon,
    )


def hold_quiet_path(*, horizon: int = 10) -> SyntheticPath:
    """Path A: small oscillations around anchor."""
    anchor = 100.0
    moves = [0.001, -0.0008, 0.0005, -0.0005, 0.001, -0.001, 0.0008, -0.0008, 0.001, -0.001]
    moves = moves[:horizon]
    return SyntheticPath(
        name="hold_quiet_mid",
        price_at_t=anchor,
        future_closes=_future_from_relative_moves(anchor, moves),
        past_closes_for_sigma=_quiet_past(),
        reward_horizon=horizon,
    )
