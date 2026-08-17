"""Reward engine configuration — P1 F-target parameters."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PerStepReturnMode(str, Enum):
    """Legacy — Path now uses t-anchored returns (:meth:`RewardContext.return_from_t`)."""

    SIMPLE = "simple"
    LOG = "log"


class PathConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gamma: Optional[float] = Field(
        default=0.75,
        description="Temporal decay rate r for Path weights w_k ∝ r^(k-1)",
    )
    per_step_return_mode: PerStepReturnMode = PerStepReturnMode.SIMPLE
    normalization: Optional[str] = Field(default=None, description="TBD — not implemented")


class UtilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    alpha: Optional[float] = Field(default=1.0, description="Gain exponent in U(x) and f_n weight")
    beta: Optional[float] = Field(default=2.0, description="Loss exponent in U(x)")
    lambda_: Optional[float] = Field(
        default=1.5,
        alias="lambda",
        description="Loss asymmetry in U(x) and MAE penalty in f_n",
    )


class MaeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalization: Optional[str] = Field(default=None, description="TBD — not implemented")


# ---------------------------------------------------------------------------
# Legacy / P2 configs — retained on disk but not used in P1 F-target engine.
# ---------------------------------------------------------------------------


class UtilityInputSource(str, Enum):
    HORIZON_RETURN = "horizon_return"
    PATH_WEIGHTED_RETURN = "path_weighted_return"


class MFutureMode(str, Enum):
    ABS_CUMULATIVE_RETURN = "abs_cumulative_return"
    ABS_PATH_SUM = "abs_path_sum"


class SigmaMethod(str, Enum):
    ROLLING_STD = "rolling_std"
    REALIZED_VOL = "realized_vol"


class SurpriseTransform(str, Enum):
    IDENTITY = "identity"
    LOG1P = "log1p"
    SQRT = "sqrt"


class SurpriseApplyMode(str, Enum):
    MULTIPLY_BASE = "multiply_base"


class HoldSurpriseApplyMode(str, Enum):
    SUBTRACT_WEIGHTED = "subtract_weighted"


class MovementMetric(str, Enum):
    MAX_ABS_DEVIATION = "max_abs_deviation"
    MAX_ABS_RETURN = "max_abs_return"


class NeutralPathMode(str, Enum):
    INVERSE_ABS_RETURN = "inverse_abs_return"


class SurpriseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    m_future_mode: MFutureMode = MFutureMode.ABS_CUMULATIVE_RETURN
    sigma_method: SigmaMethod = SigmaMethod.ROLLING_STD
    sigma_window: Optional[int] = None
    transform: SurpriseTransform = SurpriseTransform.IDENTITY
    cap: Optional[float] = None
    epsilon: float = 1e-8
    apply_mode: SurpriseApplyMode = SurpriseApplyMode.MULTIPLY_BASE


class HoldNeutralPathConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: NeutralPathMode = NeutralPathMode.INVERSE_ABS_RETURN
    gamma: Optional[float] = None
    scale: Optional[float] = None
    normalization: Optional[str] = None


class HoldMovementConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: MovementMetric = MovementMetric.MAX_ABS_DEVIATION
    normalization: Optional[str] = None


class HoldSurpriseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    apply_mode: HoldSurpriseApplyMode = HoldSurpriseApplyMode.SUBTRACT_WEIGHTED
    transform: SurpriseTransform = SurpriseTransform.IDENTITY
    cap: Optional[float] = None
    epsilon: float = 1e-8


class RewardConfig(BaseModel):
    """P1 F-target component toggles and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    reward_horizon: int = Field(default=10, description="Future 3m bars t+1..t+10")

    use_path: bool = True
    use_utility: bool = True
    use_mae: bool = True

    path: PathConfig = Field(default_factory=PathConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    mae: MaeConfig = Field(default_factory=MaeConfig)

    # Legacy fields — ignored by P1 F-target engine (HOLD / S_Move removed from P1).
    use_surprise: bool = False
    use_hold_neutral_path: bool = False
    use_hold_movement: bool = False
    use_hold_surprise: bool = False
    surprise: SurpriseConfig = Field(default_factory=SurpriseConfig)
    hold_neutral_path: HoldNeutralPathConfig = Field(default_factory=HoldNeutralPathConfig)
    hold_movement: HoldMovementConfig = Field(default_factory=HoldMovementConfig)
    hold_surprise: HoldSurpriseConfig = Field(default_factory=HoldSurpriseConfig)
