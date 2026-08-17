"""Swappable Path formulations for P1 F-target evaluation.

The canonical default remains :func:`chartai.reward.path.compute_path_n`
(``raw_return``). Other variants are used for comparative experiments only.
"""

from __future__ import annotations

import math
from enum import Enum

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.move_surprise import compute_sigma
from chartai.reward.path import normalized_decay_weights
from chartai.reward.config import SigmaMethod


class PathVariant(str, Enum):
    """Path candidate definitions for P1 design evaluation."""

    RAW_RETURN = "raw_return"
    SIGN_BASED = "sign_based"
    VOL_NORMALIZED = "vol_normalized"
    BOUNDED_TANH = "bounded_tanh"


DEFAULT_SIGMA_WINDOW = 20


def sigma_at_t(
    past_closes: tuple[float, ...],
    *,
    window: int = DEFAULT_SIGMA_WINDOW,
) -> float:
    """Past-only volatility at decision time t — uses at most ``window`` bars.

    Returns the rolling standard deviation of simple returns over the last
    ``window`` completed bars ending at t (inclusive). Never reads future data.
    """
    if len(past_closes) < 2:
        return 1e-8
    return max(
        compute_sigma(
            past_closes,
            SigmaMethod.ROLLING_STD,
            window,
        ),
        1e-8,
    )


def _aligned_return(ctx: RewardContext, action: Action, k: int) -> float:
    sign = directional_sign(action)
    return sign * ctx.return_from_t(k)


def path_step_d(
    variant: PathVariant,
    aligned_r: float,
    *,
    sigma_t: float,
) -> float:
    """Transform a single aligned return into the Path summand D_k."""
    if variant is PathVariant.RAW_RETURN:
        return aligned_r
    if variant is PathVariant.SIGN_BASED:
        if aligned_r > 0:
            return 1.0
        if aligned_r < 0:
            return -1.0
        return 0.0
    if variant is PathVariant.VOL_NORMALIZED:
        return aligned_r / sigma_t
    if variant is PathVariant.BOUNDED_TANH:
        return math.tanh(aligned_r / sigma_t)
    raise ValueError(f"Unknown PathVariant: {variant!r}")


def compute_path_n_variant(
    ctx: RewardContext,
    action: Action,
    n: int,
    *,
    variant: PathVariant,
    decay_rate: float,
    sigma_window: int = DEFAULT_SIGMA_WINDOW,
) -> float:
    """P_n for a given Path variant with normalized decay weights."""
    if n < 1 or n > ctx.reward_horizon:
        raise ValueError(f"n must be in 1..{ctx.reward_horizon}, got {n}")

    sigma_t = sigma_at_t(ctx.past_closes_for_sigma, window=sigma_window)
    weights = normalized_decay_weights(n, decay_rate)
    total = 0.0
    for k in range(1, n + 1):
        aligned_r = _aligned_return(ctx, action, k)
        d_k = path_step_d(variant, aligned_r, sigma_t=sigma_t)
        total += weights[k - 1] * d_k
    return total
