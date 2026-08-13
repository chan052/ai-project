"""Shared reward primitives and component protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chartai.core.types import Action


@dataclass(frozen=True)
class RewardBreakdown:
    """Decomposed reward output — components vs multipliers kept separate.

    ``total`` is the composed reward scalar for one action at time t.
    When used by :mod:`chartai.features.target`, ``total`` serves as an
    **F target candidate** only — not the finalized P1 F definition.
    """

    action: Action
    components: dict[str, float] = field(default_factory=dict)
    multipliers: dict[str, float] = field(default_factory=dict)
    weighted_components: dict[str, float] = field(default_factory=dict)
    base_total: float = 0.0
    total: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardComponent(ABC):
    """Independent, toggleable reward component."""

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
