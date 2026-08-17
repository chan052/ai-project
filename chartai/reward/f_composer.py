"""P1 F-target composition — f_n and F_position from Path / Utility / MAE."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import FnBreakdown, FTargetBreakdown, directional_sign
from chartai.reward.config import RewardConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import ComponentNormalizer, IdentityNormalizer
from chartai.reward.path import _require_decay_rate, compute_path_n
from chartai.reward.utility import _require_utility_params, compute_utility_n


class FTargetComposer:
    """Compose ``f_n`` and ``F_position`` for one action at decision time t.

    For each n in 1..reward_horizon::

        f_n = norm(P_n) + alpha * norm(U_n) - lambda * norm(MAE_n)

    ``F_position = mean(f_1, ..., f_H)`` — no additional temporal weighting.
    """

    def __init__(
        self,
        config: RewardConfig,
        normalizer: ComponentNormalizer | None = None,
    ) -> None:
        self._config = config
        self._normalizer = normalizer or IdentityNormalizer()

    @property
    def config(self) -> RewardConfig:
        return self._config

    @property
    def normalizer(self) -> ComponentNormalizer:
        return self._normalizer

    def compose_fn(self, ctx: RewardContext, action: Action, n: int) -> FnBreakdown:
        cfg = self._config
        decay_rate = _require_decay_rate(cfg.path)
        alpha, beta, lambda_ = _require_utility_params(cfg.utility)

        path_raw = compute_path_n(ctx, action, n, decay_rate=decay_rate) if cfg.use_path else 0.0
        utility_raw = (
            compute_utility_n(ctx, action, n, cfg.utility) if cfg.use_utility else 0.0
        )
        mae_raw = compute_mae_n(ctx, action, n) if cfg.use_mae else 0.0

        path_norm = self._normalizer.normalize_path(path_raw)
        utility_norm = self._normalizer.normalize_utility(utility_raw)
        mae_norm = self._normalizer.normalize_mae(mae_raw)

        f_n = path_norm + alpha * utility_norm - lambda_ * mae_norm

        return FnBreakdown(
            n=n,
            path_raw=path_raw,
            utility_raw=utility_raw,
            mae_raw=mae_raw,
            path_normalized=path_norm,
            utility_normalized=utility_norm,
            mae_normalized=mae_norm,
            f_n=f_n,
        )

    def compose(self, ctx: RewardContext, action: Action) -> FTargetBreakdown:
        if action not in (Action.LONG, Action.SHORT):
            raise ValueError(f"P1 F-target requires LONG or SHORT, got {action!r}")

        breakdowns: list[FnBreakdown] = []
        for n in range(1, ctx.reward_horizon + 1):
            breakdowns.append(self.compose_fn(ctx, action, n))

        fn_values = tuple(bd.f_n for bd in breakdowns)
        f_position = sum(fn_values) / len(fn_values)

        return FTargetBreakdown(
            action=action,
            f_position=f_position,
            fn_values=fn_values,
            fn_breakdowns=tuple(breakdowns),
            metadata={
                "decay_rate": self._config.path.gamma,
                "action_sign": directional_sign(action),
            },
        )
