"""Shared reward primitives and F-target output types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chartai.core.types import Action


@dataclass(frozen=True)
class FnBreakdown:
    """Per-horizon-step ``f_n`` decomposition at decision time t."""

    n: int
    path_raw: float
    utility_raw: float
    mae_raw: float
    path_normalized: float
    utility_normalized: float
    mae_normalized: float
    f_n: float


@dataclass(frozen=True)
class FTargetBreakdown:
    """P1 F-position target for one action at decision time t.

    ``F_position = mean(f_1, ..., f_10)`` with no additional temporal weighting.
    """

    action: Action
    f_position: float
    fn_values: tuple[float, ...]
    fn_breakdowns: tuple[FnBreakdown, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Alias for env / legacy callers expecting a scalar reward."""
        return self.f_position


@dataclass(frozen=True)
class RewardBreakdown:
    """Legacy decomposed reward output — retained for backward compatibility.

    New P1 code should prefer :class:`FTargetBreakdown`.
    """

    action: Action
    components: dict[str, float] = field(default_factory=dict)
    multipliers: dict[str, float] = field(default_factory=dict)
    weighted_components: dict[str, float] = field(default_factory=dict)
    base_total: float = 0.0
    total: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardComponent(ABC):
    """Independent reward component."""

    name: str

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


def directional_sign(action: Action) -> float:
    """+1 for LONG, -1 for SHORT."""
    if action is Action.LONG:
        return 1.0
    if action is Action.SHORT:
        return -1.0
    raise ValueError(f"directional_sign requires LONG or SHORT, got {action!r}")
