"""Future Move Surprise — magnitude-only S_move >= 0."""

from __future__ import annotations

import math
from statistics import pstdev

from chartai.reward.config import (
    MFutureMode,
    SigmaMethod,
    SurpriseConfig,
    SurpriseTransform,
)
from chartai.reward.context import RewardContext


def apply_transform(value: float, transform: SurpriseTransform, *, cap: float | None) -> float:
    """Apply non-negative transform; result is always >= 0 for non-negative input."""
    x = max(0.0, value)
    if transform is SurpriseTransform.IDENTITY:
        out = x
    elif transform is SurpriseTransform.LOG1P:
        out = math.log1p(x)
    elif transform is SurpriseTransform.SQRT:
        out = math.sqrt(x)
    else:
        raise ValueError(f"Unsupported surprise transform: {transform}")
    if cap is not None:
        out = min(out, cap)
    return max(0.0, out)


def compute_m_future(ctx: RewardContext, mode: MFutureMode) -> float:
    """Future directional movement magnitude — always non-negative (TODO finalize)."""
    if mode is MFutureMode.ABS_CUMULATIVE_RETURN:
        return abs(ctx.horizon_simple_return())
    if mode is MFutureMode.ABS_PATH_SUM:
        return sum(abs(r) for r in ctx.per_step_simple_returns())
    raise ValueError(f"Unsupported m_future_mode: {mode}")


def compute_sigma(past_closes: tuple[float, ...], method: SigmaMethod, window: int | None) -> float:
    """Past-only volatility estimate at t (TODO finalize methods)."""
    if len(past_closes) < 2:
        return 0.0

    returns: list[float] = []
    series = past_closes if window is None else past_closes[-(window + 1) :]
    for prev, nxt in zip(series, series[1:]):
        if prev == 0:
            continue
        returns.append((nxt - prev) / prev)

    if not returns:
        return 0.0

    if method is SigmaMethod.ROLLING_STD:
        return pstdev(returns)
    if method is SigmaMethod.REALIZED_VOL:
        return math.sqrt(sum(r * r for r in returns) / len(returns))
    raise ValueError(f"Unsupported sigma_method: {method}")


def compute_s_move(ctx: RewardContext, config: SurpriseConfig) -> float:
    """S_move = f(|M_future| / (sigma + epsilon)) — magnitude only, always >= 0."""
    m_future = compute_m_future(ctx, config.m_future_mode)
    sigma = compute_sigma(
        ctx.past_closes_for_sigma,
        config.sigma_method,
        config.sigma_window,
    )
    raw = m_future / (sigma + config.epsilon)
    return apply_transform(raw, config.transform, cap=config.cap)


class MoveSurpriseComponent:
    """Computes S_move independently — does NOT encode direction."""

    name = "surprise"

    def __init__(self, config: SurpriseConfig) -> None:
        self._config = config

    def compute_s_move(self, ctx: RewardContext) -> float:
        return compute_s_move(ctx, self._config)

    def compute(self, ctx: RewardContext) -> float:
        return self.compute_s_move(ctx)
