"""P1 Reward Engine — routes actions to directional or HOLD composers."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardBreakdown
from chartai.reward.composer import DirectionalRewardComposer, HoldRewardComposer
from chartai.reward.config import RewardConfig
from chartai.reward.context import RewardContext


class RewardEngine:
    """Compute decomposed reward for LONG / HOLD / SHORT at decision time t.

    Uses only :class:`RewardContext` (future path + past sigma series).
    Does **not** accept MTF state features — no D_ret / market-relative component.

    **Role in P1 regression (interim):** :meth:`compute` returns a
    :class:`RewardBreakdown` whose ``total`` is used as an **F target candidate**.
    ``RewardEngine.total ≠ finalized F definition``. Reward components
    (Path, Utility, MAE, Surprise, HOLD, …) are one way to *express* F;
    F itself is not yet finalized.

    See :mod:`chartai.features.target`.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._directional = DirectionalRewardComposer(config)
        self._hold = HoldRewardComposer(config)

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compute(self, action: Action, ctx: RewardContext) -> RewardBreakdown:
        ctx.validate_temporal_causality()
        if len(ctx.future_closes) != self._config.reward_horizon:
            raise ValueError(
                f"RewardContext horizon {len(ctx.future_closes)} != "
                f"config.reward_horizon {self._config.reward_horizon}"
            )

        if action is Action.HOLD:
            return self._hold.compose(ctx)

        return self._directional.compose(ctx, action)

    def enabled_component_names(self) -> list[str]:
        cfg = self._config
        names: list[str] = []
        if cfg.use_path:
            names.append("path")
        if cfg.use_utility:
            names.append("utility")
        if cfg.use_mae:
            names.append("mae")
        if cfg.use_surprise:
            names.append("surprise")
        if cfg.use_hold_neutral_path:
            names.append("hold_neutral_path")
        if cfg.use_hold_movement:
            names.append("hold_movement")
        if cfg.use_hold_surprise:
            names.append("hold_surprise")
        return names
