"""Directional Path reward component for LONG / SHORT."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardComponent, directional_sign
from chartai.reward.config import PathConfig, PerStepReturnMode
from chartai.reward.context import RewardContext


def _require_gamma(config: PathConfig) -> float:
    if config.gamma is None:
        raise ValueError("path.gamma must be set to compute Path reward")
    return config.gamma


def per_step_returns(ctx: RewardContext, mode: PerStepReturnMode) -> tuple[float, ...]:
    if mode is PerStepReturnMode.SIMPLE:
        return ctx.per_step_simple_returns()
    if mode is PerStepReturnMode.LOG:
        return ctx.per_step_log_returns()
    raise ValueError(f"Unsupported per_step_return_mode: {mode}")


def gamma_weights(num_steps: int, gamma: float) -> tuple[float, ...]:
    """Weights gamma^(k-1) for k=1..num_steps."""
    return tuple(gamma ** (k - 1) for k in range(1, num_steps + 1))


class DirectionalPathComponent(RewardComponent):
    """Gamma-weighted directional path: sum_k gamma^(k-1) * aligned_r_k.

    LONG: positive return -> favorable (positive contribution).
    SHORT: negative return -> favorable (via sign flip).

    Exact ``r_k`` definition remains configurable (TODO finalize).
    """

    name = "path"

    def __init__(self, config: PathConfig) -> None:
        self._config = config

    @property
    def config(self) -> PathConfig:
        return self._config

    def compute(self, ctx: RewardContext, action: Action) -> float:
        gamma = _require_gamma(self._config)
        returns = per_step_returns(ctx, self._config.per_step_return_mode)
        sign = directional_sign(action)
        aligned = tuple(sign * r for r in returns)
        weights = gamma_weights(len(aligned), gamma)
        return sum(w * r for w, r in zip(weights, aligned))
