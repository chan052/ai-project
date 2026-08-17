"""MAE / adverse excursion for LONG and SHORT."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardComponent
from chartai.reward.config import MaeConfig
from chartai.reward.context import RewardContext


def compute_mae_n(ctx: RewardContext, action: Action, n: int) -> float:
    """Cumulative adverse excursion from t through t+n (positive magnitude).

    LONG:  (C_t - min(L_{t+1}, ..., L_{t+n})) / C_t
    SHORT: (max(H_{t+1}, ..., H_{t+n}) - C_t) / C_t
    """
    if n < 1 or n > ctx.reward_horizon:
        raise ValueError(f"n must be in 1..{ctx.reward_horizon}, got {n}")
    anchor = ctx.price_at_t
    if action is Action.LONG:
        min_low = min(ctx.future_lows[:n])
        return (anchor - min_low) / anchor
    if action is Action.SHORT:
        max_high = max(ctx.future_highs[:n])
        return (max_high - anchor) / anchor
    raise ValueError(f"MAE requires LONG or SHORT, got {action!r}")


def long_downward_excursion(ctx: RewardContext, *, n: int | None = None) -> float:
    """LONG MAE over t+1..t+n (default full horizon)."""
    steps = n if n is not None else ctx.reward_horizon
    return compute_mae_n(ctx, Action.LONG, steps)


def short_upward_excursion(ctx: RewardContext, *, n: int | None = None) -> float:
    """SHORT MAE over t+1..t+n (default full horizon)."""
    steps = n if n is not None else ctx.reward_horizon
    return compute_mae_n(ctx, Action.SHORT, steps)


class MaeComponent(RewardComponent):
    """Full-horizon MAE — prefer :func:`compute_mae_n` for F-target steps."""

    name = "mae"

    def __init__(self, config: MaeConfig) -> None:
        self._config = config

    def compute(self, ctx: RewardContext, action: Action, *, n: int | None = None) -> float:
        steps = n if n is not None else ctx.reward_horizon
        return compute_mae_n(ctx, action, steps)
