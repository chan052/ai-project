"""Valid decision index range for P1 episodes."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.features.future_context import FutureContextBuilder
from chartai.features.state import StateBuilder


@dataclass(frozen=True)
class EpisodeBounds:
    """Inclusive ``[first_t, last_t]`` of valid 3m decision indices."""

    first_t: int
    last_t: int
    reward_horizon: int

    def __post_init__(self) -> None:
        if self.first_t < 0:
            raise ValueError("first_t must be non-negative")
        if self.last_t < self.first_t:
            raise ValueError("last_t must be >= first_t")

    def contains(self, t_index: int) -> bool:
        return self.first_t <= t_index <= self.last_t

    @property
    def num_steps(self) -> int:
        return self.last_t - self.first_t + 1


def compute_episode_bounds(
    *,
    num_3m_bars: int,
    reward_horizon: int,
    state_builder: StateBuilder,
    future_context_builder: FutureContextBuilder,
) -> EpisodeBounds:
    """Determine decision indices with valid state **and** reward future window."""
    if num_3m_bars <= reward_horizon:
        raise ValueError(
            f"num_3m_bars={num_3m_bars} must exceed reward_horizon={reward_horizon}"
        )

    last_t = num_3m_bars - 1 - reward_horizon
    first_t: int | None = None
    for t in range(0, last_t + 1):
        try:
            state_builder.build(t)
            future_context_builder.build(t)
            first_t = t
            break
        except (ValueError, IndexError):
            continue

    if first_t is None:
        raise ValueError("No valid decision index found for episode bounds")

    # Verify last_t is buildable.
    state_builder.build(last_t)
    future_context_builder.build(last_t)

    return EpisodeBounds(
        first_t=first_t,
        last_t=last_t,
        reward_horizon=reward_horizon,
    )
