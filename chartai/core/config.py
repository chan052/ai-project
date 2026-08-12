"""Configuration models — unset research parameters remain ``None`` / TODO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from chartai.core.types import Timeframe


class TimeframeStateConfig(BaseModel):
    """Per-timeframe state settings — lookback/features/normalization TBD."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lookback_bars: Optional[int] = Field(
        default=None,
        description="TODO: number of past bars to include in state",
    )
    features: Optional[list[str]] = Field(
        default=None,
        description="TODO: feature names for this timeframe branch",
    )
    normalization: Optional[str] = Field(
        default=None,
        description="TODO: normalization strategy identifier",
    )


class StateConfig(BaseModel):
    """Multi-timeframe P1 state configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_timeframe: Timeframe = Timeframe.M3
    timeframes: dict[str, TimeframeStateConfig] = Field(
        default_factory=lambda: {
            "3m": TimeframeStateConfig(),
            "1h": TimeframeStateConfig(),
            "4h": TimeframeStateConfig(),
        }
    )
    fusion: Optional[str] = Field(
        default=None,
        description="TODO: policy input fusion strategy (concat, multi_encoder, ...)",
    )
    use_incomplete_higher_tf_bars: bool = Field(
        default=False,
        description=(
            "P1 Phase 0: False — only completed 1H/4H bars at decision time. "
            "Future ablation candidate; not implemented in Phase 0."
        ),
    )


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state_window: Optional[int] = Field(default=None, description="TODO: 3m state window")
    reward_horizon: Optional[int] = Field(default=None, description="TODO: 3m reward horizon")


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_symbol: Optional[str] = Field(default=None, description="TODO: SPY, QQQ, ...")
    sigma_method: Optional[str] = Field(
        default=None,
        description="TODO: rolling_std | realized_vol | atr",
    )
    sigma_timeframe: Optional[str] = Field(
        default=None,
        description="TODO: timeframe used to compute sigma_market_t",
    )
    sigma_window: Optional[int] = None
    epsilon: float = 1e-8


class ExperimentConfig(BaseModel):
    """Top-level experiment config skeleton for Phase 0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    seed: Optional[int] = None
    state: StateConfig = Field(default_factory=StateConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml_config(path)
    return ExperimentConfig.model_validate(raw)
