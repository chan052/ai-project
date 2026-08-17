"""Legacy reward composition — superseded by :mod:`chartai.reward.f_composer`.

HOLD and S_Move composers are retained for reference but are **not** used by
the P1 :class:`~chartai.reward.engine.RewardEngine`.
"""

from __future__ import annotations

from chartai.core.types import Action
from chartai.reward.base import RewardBreakdown
from chartai.reward.config import HoldSurpriseApplyMode, RewardConfig, SurpriseApplyMode
from chartai.reward.context import RewardContext
from chartai.reward.hold import HoldMovementComponent, HoldNeutralPathComponent
from chartai.reward.mae import MaeComponent
from chartai.reward.move_surprise import MoveSurpriseComponent, apply_transform
from chartai.reward.path import DirectionalPathComponent
from chartai.reward.utility import UtilityComponent


class DirectionalRewardComposer:
    """Legacy full-horizon composer — use :class:`~chartai.reward.f_composer.FTargetComposer`."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._path = DirectionalPathComponent(config.path)
        self._utility = UtilityComponent(config.utility)
        self._mae = MaeComponent(config.mae)
        self._surprise = MoveSurpriseComponent(config.surprise)

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compose(self, ctx: RewardContext, action: Action) -> RewardBreakdown:
        if action not in (Action.LONG, Action.SHORT):
            raise ValueError("DirectionalRewardComposer requires LONG or SHORT")

        cfg = self._config
        components: dict[str, float] = {}
        weighted: dict[str, float] = {}
        base_total = 0.0

        if cfg.use_path:
            val = self._path.compute(ctx, action)
            components["path"] = val
            weighted["path"] = val
            base_total += val

        if cfg.use_utility:
            val = self._utility.compute(ctx, action)
            components["utility"] = val
            weighted["utility"] = val
            base_total += val

        if cfg.use_mae:
            val = self._mae.compute(ctx, action)
            components["mae"] = val
            weighted["mae"] = -val
            base_total -= val

        multipliers: dict[str, float] = {}
        total = base_total

        if cfg.use_surprise:
            s_move = self._surprise.compute_s_move(ctx)
            multipliers["s_move"] = s_move
            if cfg.surprise.apply_mode is SurpriseApplyMode.MULTIPLY_BASE:
                transformed = apply_transform(
                    s_move,
                    cfg.surprise.transform,
                    cap=cfg.surprise.cap,
                )
                surprise_multiplier = 1.0 + transformed
                multipliers["surprise_multiplier"] = surprise_multiplier
                total = base_total * surprise_multiplier
            else:
                raise NotImplementedError(
                    f"Surprise apply_mode {cfg.surprise.apply_mode!r} not implemented"
                )

        return RewardBreakdown(
            action=action,
            components=components,
            multipliers=multipliers,
            weighted_components=weighted,
            base_total=base_total,
            total=total,
        )


class HoldRewardComposer:
    """Legacy HOLD composer — not used in P1 (HOLD removed from action space)."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._neutral = HoldNeutralPathComponent(config.hold_neutral_path, config.path)
        self._movement = HoldMovementComponent(config.hold_movement)
        self._surprise = MoveSurpriseComponent(config.surprise)

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compose(self, ctx: RewardContext) -> RewardBreakdown:
        raise NotImplementedError("HOLD is not part of P1 action space")
