"""Directional Path component — t-anchored returns with exponential decay."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardComponent, directional_sign
from chartai.reward.config import PathConfig
from chartai.reward.context import RewardContext


def _require_decay_rate(config: PathConfig) -> float:
    if config.gamma is None:
        raise ValueError("path.gamma (decay rate r) must be set to compute Path")
    return config.gamma


def normalized_decay_weights(num_steps: int, decay_rate: float) -> tuple[float, ...]:
    """Normalized weights w_k ∝ r^(k-1) for k=1..num_steps."""
    if num_steps <= 0:
        return ()
    raw = tuple(decay_rate ** (k - 1) for k in range(1, num_steps + 1))
    total = sum(raw)
    if total == 0:
        raise ValueError("decay weights sum to zero")
    return tuple(w / total for w in raw)


def gamma_weights(num_steps: int, gamma: float) -> tuple[float, ...]:
    """Unnormalized weights gamma^(k-1) — legacy alias for tests."""
    return tuple(gamma ** (k - 1) for k in range(1, num_steps + 1))


def compute_path_n(
    ctx: RewardContext,
    action: Action,
    n: int,
    *,
    decay_rate: float,
) -> float:
    """P_n = sum_{k=1..n} w_k * aligned_R_k with t-anchored returns.

    aligned_R_k = sign * (C_{t+k} - C_t) / C_t
    """
    if n < 1 or n > ctx.reward_horizon:
        raise ValueError(f"n must be in 1..{ctx.reward_horizon}, got {n}")
    sign = directional_sign(action)
    weights = normalized_decay_weights(n, decay_rate)
    return sum(weights[k - 1] * sign * ctx.return_from_t(k) for k in range(1, n + 1))


class DirectionalPathComponent(RewardComponent):
    """Full-horizon path (P_H) — prefer :func:`compute_path_n` for F-target steps."""

    name = "path"

    def __init__(self, config: PathConfig) -> None:
        self._config = config

    @property
    def config(self) -> PathConfig:
        return self._config

    def compute(self, ctx: RewardContext, action: Action, *, n: int | None = None) -> float:
        decay_rate = _require_decay_rate(self._config)
        steps = n if n is not None else ctx.reward_horizon
        return compute_path_n(ctx, action, steps, decay_rate=decay_rate)
