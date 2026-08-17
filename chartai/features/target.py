"""P1 supervised regression targets — F_LONG / F_SHORT at decision time t."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from chartai.core.types import Action
from chartai.features.sample import P1DecisionSample, P1SampleAssembler
from chartai.features.state import MultiTimeframeState
from chartai.reward.base import FTargetBreakdown
from chartai.reward.engine import RewardEngine

P1_ACTION_TARGET_ORDER: Final[tuple[Action, ...]] = (
    Action.LONG,
    Action.SHORT,
)


@dataclass(frozen=True)
class ActionTargetVector:
    """F_LONG and F_SHORT targets at decision time t.

    Each value is ``FTargetBreakdown.f_position`` for the corresponding action.
    Both are computed from the **same** actual future market path at t.
    """

    f_long: float
    f_short: float

    @classmethod
    def from_breakdowns(
        cls,
        breakdowns: dict[Action, FTargetBreakdown],
    ) -> ActionTargetVector:
        missing = [a for a in P1_ACTION_TARGET_ORDER if a not in breakdowns]
        if missing:
            raise ValueError(f"Missing action breakdowns: {missing}")
        return cls(
            f_long=breakdowns[Action.LONG].f_position,
            f_short=breakdowns[Action.SHORT].f_position,
        )

    def for_action(self, action: Action) -> float:
        return {
            Action.LONG: self.f_long,
            Action.SHORT: self.f_short,
        }[action]

    def as_tuple(self) -> tuple[float, float]:
        return (self.f_long, self.f_short)

    def as_list(self) -> list[float]:
        return list(self.as_tuple())

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        return self.as_tuple()[index]

    @staticmethod
    def action_at_index(index: int) -> Action:
        return P1_ACTION_TARGET_ORDER[index]


@dataclass(frozen=True)
class P1RegressionSample:
    """P1 supervised unit: ``State(t)`` + ``[F_LONG, F_SHORT]`` targets."""

    t_index: int
    state: MultiTimeframeState
    targets: ActionTargetVector

    @property
    def decision_time(self):
        return self.state.decision_time

    @classmethod
    def from_decision_sample(
        cls,
        sample: P1DecisionSample,
        engine: RewardEngine,
    ) -> P1RegressionSample:
        return cls(
            t_index=sample.t_index,
            state=sample.state,
            targets=sample.compute_target_vector(engine),
        )


class P1RegressionSampleBuilder:
    """Assemble :class:`P1RegressionSample` with F targets at ``t``."""

    def __init__(
        self,
        sample_assembler: P1SampleAssembler,
        reward_engine: RewardEngine,
    ) -> None:
        self._sample_assembler = sample_assembler
        self._reward_engine = reward_engine

    @property
    def sample_assembler(self) -> P1SampleAssembler:
        return self._sample_assembler

    @property
    def reward_engine(self) -> RewardEngine:
        return self._reward_engine

    def build(self, t_index: int) -> P1RegressionSample:
        sample = self._sample_assembler.assemble(t_index)
        return P1RegressionSample.from_decision_sample(sample, self._reward_engine)

    def build_batch(self, t_indices: Sequence[int]) -> tuple[P1RegressionSample, ...]:
        return tuple(self.build(t) for t in t_indices)
