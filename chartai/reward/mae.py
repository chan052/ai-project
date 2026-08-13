"""MAE / adverse movement for LONG and SHORT."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardComponent
from chartai.reward.config import MaeConfig
from chartai.reward.context import RewardContext


def long_downward_excursion(ctx: RewardContext) -> float:
    """Maximum adverse downward move relative to price_at_t during future path."""
    anchor = ctx.price_at_t
    running_max = anchor
    max_adverse = 0.0
    for price in ctx.future_prices[1:]:
        running_max = max(running_max, price)
        drawdown = (running_max - price) / anchor
        max_adverse = max(max_adverse, drawdown)
    return max_adverse


def short_upward_excursion(ctx: RewardContext) -> float:
    """Maximum adverse upward move relative to price_at_t during future path."""
    anchor = ctx.price_at_t
    running_min = anchor
    max_adverse = 0.0
    for price in ctx.future_prices[1:]:
        running_min = min(running_min, price)
        adverse = (price - running_min) / anchor
        max_adverse = max(max_adverse, adverse)
    return max_adverse


class MaeComponent(RewardComponent):
    """Directional adverse excursion — returned as **positive magnitude**.

    Composer applies configured (typically negative) weight as penalty.
    Exact normalization remains TODO.
    """

    name = "mae"

    def __init__(self, config: MaeConfig) -> None:
        self._config = config

    def compute(self, ctx: RewardContext, action: Action) -> float:
        if action is Action.LONG:
            return long_downward_excursion(ctx)
        if action is Action.SHORT:
            return short_upward_excursion(ctx)
        raise ValueError(f"MAE requires LONG or SHORT, got {action!r}")
