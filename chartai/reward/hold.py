"""HOLD-specific reward components."""

from __future__ import annotations

from chartai.reward.base import RewardComponent
from chartai.reward.config import HoldMovementConfig, HoldNeutralPathConfig, MovementMetric, PathConfig
from chartai.reward.context import RewardContext
from chartai.reward.path import gamma_weights, per_step_returns
from chartai.reward.config import PerStepReturnMode


def _resolve_gamma(hold_config: HoldNeutralPathConfig, path_config: PathConfig) -> float:
    gamma = hold_config.gamma if hold_config.gamma is not None else path_config.gamma
    if gamma is None:
        raise ValueError("hold neutral path gamma must be set (hold_neutral_path.gamma or path.gamma)")
    return gamma


def _resolve_scale(hold_config: HoldNeutralPathConfig) -> float:
    return hold_config.scale if hold_config.scale is not None else 1.0


def compute_neutral_path(
    ctx: RewardContext,
    hold_config: HoldNeutralPathConfig,
    path_config: PathConfig,
) -> float:
    """HOLD neutral path score — NOT the negation of directional Path.

    Reference candidate (TODO finalize formula):
        score_k = 1 / (1 + |r_k| / scale)
        R = sum_k gamma^(k-1) * score_k
    """
    gamma = _resolve_gamma(hold_config, path_config)
    scale_val = _resolve_scale(hold_config)
    returns = per_step_returns(ctx, path_config.per_step_return_mode)
    weights = gamma_weights(len(returns), gamma)
    scores = tuple(1.0 / (1.0 + abs(r) / scale_val) for r in returns)
    return sum(w * s for w, s in zip(weights, scores))


def compute_hold_movement(ctx: RewardContext, config: HoldMovementConfig) -> float:
    """Movement / excursion magnitude for HOLD — large swing -> higher value (penalized).

    Reference candidate (TODO finalize metric):
        MAX_ABS_DEVIATION: max |price - price_at_t| / price_at_t
    """
    if config.metric is MovementMetric.MAX_ABS_DEVIATION:
        anchor = ctx.price_at_t
        return max(abs(p - anchor) / anchor for p in ctx.future_prices[1:])
    if config.metric is MovementMetric.MAX_ABS_RETURN:
        returns = ctx.per_step_simple_returns()
        return max(abs(r) for r in returns) if returns else 0.0
    raise ValueError(f"Unsupported movement metric: {config.metric}")


class HoldNeutralPathComponent(RewardComponent):
    name = "hold_neutral_path"

    def __init__(
        self,
        hold_config: HoldNeutralPathConfig,
        path_config: PathConfig,
    ) -> None:
        self._hold_config = hold_config
        self._path_config = path_config

    def compute(self, ctx: RewardContext) -> float:
        return compute_neutral_path(ctx, self._hold_config, self._path_config)


class HoldMovementComponent(RewardComponent):
    name = "hold_movement"

    def __init__(self, config: HoldMovementConfig) -> None:
        self._config = config

    def compute(self, ctx: RewardContext) -> float:
        return compute_hold_movement(ctx, self._config)
