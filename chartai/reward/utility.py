"""Utility component — asymmetric gain/loss at t+n."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardComponent, directional_sign
from chartai.reward.config import UtilityConfig
from chartai.reward.context import RewardContext


def _require_utility_params(config: UtilityConfig) -> tuple[float, float, float]:
    if config.alpha is None or config.beta is None or config.lambda_ is None:
        raise ValueError("utility alpha, beta, and lambda must be set")
    return config.alpha, config.beta, config.lambda_


def utility_u(x: float, *, alpha: float, beta: float, lambda_: float) -> float:
    """U(x) = x^alpha if x >= 0 else -lambda * |x|^beta."""
    if x >= 0:
        return x**alpha
    return -lambda_ * (abs(x) ** beta)


def compute_utility_n(
    ctx: RewardContext,
    action: Action,
    n: int,
    config: UtilityConfig,
) -> float:
    """U_n = U(x_n) where x_n is position-aligned return at t+n."""
    alpha, beta, lambda_ = _require_utility_params(config)
    sign = directional_sign(action)
    x_n = sign * ctx.return_from_t(n)
    return utility_u(x_n, alpha=alpha, beta=beta, lambda_=lambda_)


class UtilityComponent(RewardComponent):
    """Full-horizon utility at t+H — prefer :func:`compute_utility_n` for F steps."""

    name = "utility"

    def __init__(self, config: UtilityConfig) -> None:
        self._config = config

    def compute(self, ctx: RewardContext, action: Action, *, n: int | None = None) -> float:
        steps = n if n is not None else ctx.reward_horizon
        return compute_utility_n(ctx, action, steps, self._config)
