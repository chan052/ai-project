"""Reward composition — enabled components and configurable weights only."""

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
    """Compose LONG / SHORT rewards from independent directional components."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._path = DirectionalPathComponent(config.path)
        self._utility = UtilityComponent(config.utility, path_gamma=config.path.gamma)
        self._mae = MaeComponent(config.mae)
        self._surprise = MoveSurpriseComponent(config.surprise)

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compose(self, ctx: RewardContext, action: Action) -> RewardBreakdown:
        if action is Action.HOLD:
            raise ValueError("DirectionalRewardComposer does not support HOLD")

        cfg = self._config
        components: dict[str, float] = {}
        weighted: dict[str, float] = {}
        base_total = 0.0

        if cfg.use_path:
            val = self._path.compute(ctx, action)
            weight = cfg.weight_for("path")
            components["path"] = val
            weighted["path"] = weight * val
            base_total += weighted["path"]

        if cfg.use_utility:
            val = self._utility.compute(ctx, action)
            weight = cfg.weight_for("utility")
            components["utility"] = val
            weighted["utility"] = weight * val
            base_total += weighted["utility"]

        if cfg.use_mae:
            val = self._mae.compute(ctx, action)
            weight = cfg.weight_for("mae")
            components["mae"] = val
            weighted["mae"] = weight * val
            base_total += weighted["mae"]

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
                surprise_weight = cfg.weight_for("surprise")
                # Candidate: base × (1 + weight × S_move) — TODO finalize multiplier form.
                surprise_multiplier = 1.0 + surprise_weight * transformed
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
    """Compose HOLD reward — separate from directional (not a negation)."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._neutral = HoldNeutralPathComponent(config.hold_neutral_path, config.path)
        self._movement = HoldMovementComponent(config.hold_movement)
        self._surprise = MoveSurpriseComponent(config.surprise)

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compose(self, ctx: RewardContext) -> RewardBreakdown:
        cfg = self._config
        components: dict[str, float] = {}
        weighted: dict[str, float] = {}
        base_total = 0.0

        if cfg.use_hold_neutral_path:
            val = self._neutral.compute(ctx)
            weight = cfg.weight_for("hold_neutral_path")
            components["hold_neutral_path"] = val
            weighted["hold_neutral_path"] = weight * val
            base_total += weighted["hold_neutral_path"]

        if cfg.use_hold_movement:
            val = self._movement.compute(ctx)
            weight = cfg.weight_for("hold_movement")
            components["hold_movement"] = val
            weighted["hold_movement"] = weight * val
            base_total += weighted["hold_movement"]

        multipliers: dict[str, float] = {}
        total = base_total

        if cfg.use_hold_surprise:
            s_move = self._surprise.compute_s_move(ctx)
            multipliers["s_move"] = s_move
            penalty = apply_transform(
                s_move,
                cfg.hold_surprise.transform,
                cap=cfg.hold_surprise.cap,
            )
            multipliers["hold_surprise_penalty"] = penalty
            if cfg.hold_surprise.apply_mode is HoldSurpriseApplyMode.SUBTRACT_WEIGHTED:
                weight = cfg.weight_for("hold_surprise")
                total = base_total - weight * penalty
            else:
                raise NotImplementedError(
                    f"Hold surprise apply_mode {cfg.hold_surprise.apply_mode!r} not implemented"
                )

        return RewardBreakdown(
            action=Action.HOLD,
            components=components,
            multipliers=multipliers,
            weighted_components=weighted,
            base_total=base_total,
            total=total,
        )
