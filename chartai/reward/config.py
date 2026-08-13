"""Reward engine configuration — unset research parameters remain optional."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PerStepReturnMode(str, Enum):
    """TODO: finalize per-step return representation for Path."""

    SIMPLE = "simple"
    LOG = "log"


class UtilityInputSource(str, Enum):
    """TODO: finalize input ``x`` for Utility U(x)."""

    HORIZON_RETURN = "horizon_return"
    PATH_WEIGHTED_RETURN = "path_weighted_return"


class MFutureMode(str, Enum):
    """TODO: finalize M_future definition for Future Move Surprise."""

    ABS_CUMULATIVE_RETURN = "abs_cumulative_return"
    ABS_PATH_SUM = "abs_path_sum"


class SigmaMethod(str, Enum):
    """TODO: finalize sigma_market_t estimation."""

    ROLLING_STD = "rolling_std"
    REALIZED_VOL = "realized_vol"


class SurpriseTransform(str, Enum):
    """TODO: finalize S_move transform."""

    IDENTITY = "identity"
    LOG1P = "log1p"
    SQRT = "sqrt"


class SurpriseApplyMode(str, Enum):
    """How S_move combines with directional base reward."""

    MULTIPLY_BASE = "multiply_base"
    # TODO: additive, cap, separate weight — future candidates


class HoldSurpriseApplyMode(str, Enum):
    """How S_move reduces HOLD reward — exact penalty TODO."""

    SUBTRACT_WEIGHTED = "subtract_weighted"
    # TODO: multiply_factor, custom penalty function


class MovementMetric(str, Enum):
    """TODO: finalize HOLD movement / excursion metric."""

    MAX_ABS_DEVIATION = "max_abs_deviation"
    MAX_ABS_RETURN = "max_abs_return"


class NeutralPathMode(str, Enum):
    """TODO: finalize HOLD neutral path scoring function."""

    INVERSE_ABS_RETURN = "inverse_abs_return"


class PathConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gamma: Optional[float] = Field(default=None, description="TODO: temporal discount")
    per_step_return_mode: PerStepReturnMode = PerStepReturnMode.SIMPLE
    normalization: Optional[str] = Field(default=None, description="TODO")


class UtilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    alpha: Optional[float] = Field(default=None, description="TODO: ablation candidate")
    beta: Optional[float] = Field(default=None, description="TODO: ablation candidate")
    lambda_: Optional[float] = Field(
        default=None,
        alias="lambda",
        description="TODO: loss asymmetry weight",
    )
    input_source: UtilityInputSource = UtilityInputSource.HORIZON_RETURN


class MaeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalization: Optional[str] = Field(default=None, description="TODO")


class SurpriseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    m_future_mode: MFutureMode = MFutureMode.ABS_CUMULATIVE_RETURN
    sigma_method: SigmaMethod = SigmaMethod.ROLLING_STD
    sigma_window: Optional[int] = Field(default=None, description="TODO")
    transform: SurpriseTransform = SurpriseTransform.IDENTITY
    cap: Optional[float] = Field(default=None, description="TODO: upper cap on S_move")
    epsilon: float = 1e-8
    apply_mode: SurpriseApplyMode = SurpriseApplyMode.MULTIPLY_BASE


class HoldNeutralPathConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: NeutralPathMode = NeutralPathMode.INVERSE_ABS_RETURN
    gamma: Optional[float] = Field(default=None, description="TODO: may mirror path gamma")
    scale: Optional[float] = Field(
        default=None,
        description="TODO: scaling for neutral path non-linearity",
    )
    normalization: Optional[str] = Field(default=None, description="TODO")


class HoldMovementConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: MovementMetric = MovementMetric.MAX_ABS_DEVIATION
    normalization: Optional[str] = Field(default=None, description="TODO")


class HoldSurpriseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    apply_mode: HoldSurpriseApplyMode = HoldSurpriseApplyMode.SUBTRACT_WEIGHTED
    transform: SurpriseTransform = SurpriseTransform.IDENTITY
    cap: Optional[float] = Field(default=None, description="TODO")
    epsilon: float = 1e-8


class ComponentWeights(BaseModel):
    """Per-component weights — no hardcoded aggregation in code."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    path: Optional[float] = None
    utility: Optional[float] = None
    mae: Optional[float] = None
    surprise: Optional[float] = None
    hold_neutral_path: Optional[float] = None
    hold_movement: Optional[float] = None
    hold_surprise: Optional[float] = None


class RewardConfig(BaseModel):
    """P1 reward component toggles and parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    reward_horizon: int = Field(default=10, description="Future 3m bars t+1..t+10")

    use_path: bool = True
    use_utility: bool = True
    use_mae: bool = True
    use_surprise: bool = False

    use_hold_neutral_path: bool = True
    use_hold_movement: bool = True
    use_hold_surprise: bool = False

    weights: ComponentWeights = Field(default_factory=ComponentWeights)

    path: PathConfig = Field(default_factory=PathConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    mae: MaeConfig = Field(default_factory=MaeConfig)
    surprise: SurpriseConfig = Field(default_factory=SurpriseConfig)
    hold_neutral_path: HoldNeutralPathConfig = Field(default_factory=HoldNeutralPathConfig)
    hold_movement: HoldMovementConfig = Field(default_factory=HoldMovementConfig)
    hold_surprise: HoldSurpriseConfig = Field(default_factory=HoldSurpriseConfig)

    def weight_for(self, component: str) -> float:
        """Return configured weight; enabled components require explicit weight."""
        value = getattr(self.weights, component, None)
        if value is None:
            raise ValueError(
                f"Component '{component}' is enabled but weights.{component} is not set"
            )
        return value
