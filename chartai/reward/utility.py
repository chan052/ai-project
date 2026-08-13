"""Utility reward component — LONG / SHORT only."""

from __future__ import annotations

import math

from chartai.core.types import Action
from chartai.reward.base import RewardComponent, directional_sign
from chartai.reward.config import UtilityConfig, UtilityInputSource
from chartai.reward.context import RewardContext
from chartai.reward.path import _require_gamma, gamma_weights, per_step_returns


def _require_utility_params(config: UtilityConfig) -> tuple[float, float, float]:
    if config.alpha is None or config.beta is None or config.lambda_ is None:
        raise ValueError("utility alpha, beta, and lambda must be set")
    return config.alpha, config.beta, config.lambda_


def utility_u(x: float, *, alpha: float, beta: float, lambda_: float) -> float:
    """U(x) = x^alpha if x >= 0 else -lambda * |x|^beta."""
    if x >= 0:
        return x**alpha
    return -lambda_ * (abs(x) ** beta)


def utility_input_with_path_gamma(
    ctx: RewardContext,
    action: Action,
    config: UtilityConfig,
    *,
    path_gamma: float | None,
) -> float:
    sign = directional_sign(action)
    if config.input_source is UtilityInputSource.HORIZON_RETURN:
        return sign * ctx.horizon_simple_return()
    if config.input_source is UtilityInputSource.PATH_WEIGHTED_RETURN:
        if path_gamma is None:
            raise ValueError("path_gamma required for PATH_WEIGHTED_RETURN")
        from chartai.reward.config import PathConfig, PerStepReturnMode

        returns = per_step_returns(ctx, PerStepReturnMode.SIMPLE)
        aligned = tuple(sign * r for r in returns)
        weights = gamma_weights(len(aligned), path_gamma)
        return sum(w * r for w, r in zip(weights, aligned))
    raise ValueError(f"Unsupported utility input_source: {config.input_source}")


class UtilityComponent(RewardComponent):
    """Asymmetric utility on direction-aligned return magnitude."""

    name = "utility"

    def __init__(self, config: UtilityConfig, *, path_gamma: float | None = None) -> None:
        self._config = config
        self._path_gamma = path_gamma

    def compute(self, ctx: RewardContext, action: Action) -> float:
        alpha, beta, lambda_ = _require_utility_params(self._config)
        x = utility_input_with_path_gamma(
            ctx, action, self._config, path_gamma=self._path_gamma
        )
        return utility_u(x, alpha=alpha, beta=beta, lambda_=lambda_)
