"""Temporal split between state (past) and reward (future) zones on 3m."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import DecisionTime, Timeframe


@dataclass(frozen=True)
class WindowSpec:
    """Configurable window sizes — defaults are placeholders for P1."""

    state_window: int | None = None  # TODO: 3m lookback bars (e.g. 100)
    reward_horizon: int | None = None  # TODO: future 3m bars (e.g. 10)


@dataclass(frozen=True)
class TemporalSplit:
    """Index boundaries enforcing causality at 3m decision index ``t``.

    STATE zone (3m):  ``[state_start, t]`` inclusive — past only through t.
    REWARD zone (3m): ``[t + 1, t + reward_horizon]`` inclusive — future only.

    Higher timeframes (1H, 4H) use :class:`MultiTimeframeAligner`; this split
    applies to the decision timeframe and reward computation horizon.
    """

    t_index: int
    spec: WindowSpec

    def __post_init__(self) -> None:
        if self.t_index < 0:
            raise ValueError("t_index must be non-negative")
        if self.spec.state_window is not None and self.spec.state_window <= 0:
            raise ValueError("state_window must be positive when set")
        if self.spec.reward_horizon is not None and self.spec.reward_horizon <= 0:
            raise ValueError("reward_horizon must be positive when set")

    @property
    def decision_timeframe(self) -> Timeframe:
        return Timeframe.M3

    @property
    def state_start(self) -> int | None:
        if self.spec.state_window is None:
            return None
        return max(0, self.t_index - self.spec.state_window + 1)

    @property
    def state_end(self) -> int:
        return self.t_index

    @property
    def reward_start(self) -> int | None:
        if self.spec.reward_horizon is None:
            return None
        return self.t_index + 1

    @property
    def reward_end(self) -> int | None:
        if self.spec.reward_horizon is None:
            return None
        return self.t_index + self.spec.reward_horizon

    def state_indices(self) -> range | None:
        if self.spec.state_window is None:
            return None
        return range(self.state_start, self.state_end + 1)

    def reward_indices(self) -> range | None:
        if self.spec.reward_horizon is None:
            return None
        return range(self.reward_start, self.reward_end + 1)

    def assert_valid_series_length(self, series_length: int) -> None:
        if series_length <= 0:
            raise ValueError("series_length must be positive")
        if self.t_index >= series_length:
            raise ValueError(
                f"t_index={self.t_index} out of range for series_length={series_length}"
            )
        if self.spec.reward_horizon is not None:
            if self.reward_end >= series_length:
                raise ValueError(
                    "reward horizon extends beyond available 3m series: "
                    f"need index <= {series_length - 1}, got reward_end={self.reward_end}"
                )

    def assert_no_future_in_state(self, used_indices: range, *, label: str = "state") -> None:
        """Raise if any index in ``used_indices`` exceeds ``t_index``."""
        for idx in used_indices:
            if idx > self.t_index:
                raise ValueError(
                    f"{label} uses future index {idx} at t_index={self.t_index}"
                )

    def assert_no_past_in_reward(self, used_indices: range, *, label: str = "reward") -> None:
        """Raise if any index in ``used_indices`` is at or before ``t_index``."""
        for idx in used_indices:
            if idx <= self.t_index:
                raise ValueError(
                    f"{label} uses non-future index {idx} at t_index={self.t_index}"
                )

    @classmethod
    def from_decision(
        cls,
        decision: DecisionTime,
        t_index: int,
        spec: WindowSpec,
    ) -> TemporalSplit:
        """Factory retaining ``DecisionTime`` association via ``t_index`` only.

        ``DecisionTime`` is carried for downstream MTF alignment; index split
        is on the 3m integer timeline.
        """
        _ = decision  # reserved for future validation against bar timestamps
        return cls(t_index=t_index, spec=spec)
