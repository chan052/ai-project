"""P1 Reward Engine — F-target computation for LONG / SHORT."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import FTargetBreakdown
from chartai.reward.config import RewardConfig
from chartai.reward.context import RewardContext
from chartai.reward.f_composer import FTargetComposer
from chartai.reward.normalization import ComponentNormalizer, IdentityNormalizer


class RewardEngine:
    """Compute P1 F-position targets for LONG / SHORT at decision time t.

    Uses only :class:`RewardContext` (future OHLC path). Does **not** accept
    MTF state features.

    Output :class:`FTargetBreakdown` with::

        F_position = mean(f_1, ..., f_10)

    See :mod:`chartai.features.target` and :mod:`chartai.reward.f_composer`.
    """

    def __init__(
        self,
        config: RewardConfig,
        normalizer: ComponentNormalizer | None = None,
    ) -> None:
        self._config = config
        self._normalizer = normalizer or IdentityNormalizer()
        self._composer = FTargetComposer(config, normalizer=self._normalizer)

    @property
    def config(self) -> RewardConfig:
        return self._config

    @property
    def normalizer(self) -> ComponentNormalizer:
        return self._normalizer

    def compute(self, action: Action, ctx: RewardContext) -> FTargetBreakdown:
        ctx.validate_temporal_causality()
        if len(ctx.future_closes) != self._config.reward_horizon:
            raise ValueError(
                f"RewardContext horizon {len(ctx.future_closes)} != "
                f"config.reward_horizon {self._config.reward_horizon}"
            )
        return self._composer.compose(ctx, action)

    def compute_both(self, ctx: RewardContext) -> dict[Action, FTargetBreakdown]:
        return {
            Action.LONG: self.compute(Action.LONG, ctx),
            Action.SHORT: self.compute(Action.SHORT, ctx),
        }

    def enabled_component_names(self) -> list[str]:
        cfg = self._config
        names: list[str] = []
        if cfg.use_path:
            names.append("path")
        if cfg.use_utility:
            names.append("utility")
        if cfg.use_mae:
            names.append("mae")
        return names
