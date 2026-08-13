"""P1 supervised regression targets — F target candidates at time t.

Conceptual boundary (not final design):

**F** — P1 prediction target concept: how favorable the *same* future market
path is from each action's perspective (LONG / HOLD / SHORT). F is **not**
finalized as a formula yet.

**Reward** — compositional expression of F via Path, Utility, MAE, Surprise,
HOLD components, etc. Reward formulas and weights remain research candidates.

**Current wiring** — :class:`ActionTargetVector` values are populated from
:class:`RewardBreakdown`.total. That ``total`` is an **F target candidate**,
not the finalized definition of F::

    RewardEngine.total  ≠  finalized F definition

Future training flow (conceptual)::

    actual future data  →  F_target
    State(t)            →  Neural Network  →  F_predicted

F is not defined as simple final return; path shape, adverse movement, and
other properties may matter when F is eventually specified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from chartai.core.types import Action
from chartai.features.sample import P1DecisionSample, P1SampleAssembler
from chartai.features.state import MultiTimeframeState
from chartai.reward.base import RewardBreakdown
from chartai.reward.engine import RewardEngine

# Fixed action order for multi-output regression (Architecture B candidate).
P1_ACTION_TARGET_ORDER: Final[tuple[Action, ...]] = (
    Action.LONG,
    Action.HOLD,
    Action.SHORT,
)


@dataclass(frozen=True)
class ActionTargetVector:
    """F_long, F_hold, F_short target **candidates** at decision time t.

    Each value is sourced from :attr:`RewardBreakdown.total` for the
    corresponding action. That reward total is a **candidate** expression of F
    under the current reward-component design — not the finalized F definition.

    All three values are computed from the **same** actual future market path
    (``RewardContext`` at t). Actions do not alter the underlying future;
    they change the action-conditioned scoring perspective only.
    """

    f_long: float
    f_hold: float
    f_short: float

    @classmethod
    def from_breakdowns(cls, breakdowns: dict[Action, RewardBreakdown]) -> ActionTargetVector:
        """Map reward totals to F target candidates — temporary, not final F."""
        missing = [a for a in P1_ACTION_TARGET_ORDER if a not in breakdowns]
        if missing:
            raise ValueError(f"Missing action breakdowns: {missing}")
        return cls(
            f_long=breakdowns[Action.LONG].total,
            f_hold=breakdowns[Action.HOLD].total,
            f_short=breakdowns[Action.SHORT].total,
        )

    def for_action(self, action: Action) -> float:
        """Scalar target for action-conditioned models (Architecture A candidate)."""
        return {
            Action.LONG: self.f_long,
            Action.HOLD: self.f_hold,
            Action.SHORT: self.f_short,
        }[action]

    def as_tuple(self) -> tuple[float, float, float]:
        """Multi-output vector in fixed ``[LONG, HOLD, SHORT]`` order."""
        return (self.f_long, self.f_hold, self.f_short)

    def as_list(self) -> list[float]:
        return list(self.as_tuple())

    def __iter__(self):
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        return self.as_tuple()[index]

    @staticmethod
    def action_at_index(index: int) -> Action:
        return P1_ACTION_TARGET_ORDER[index]


@dataclass(frozen=True)
class P1RegressionSample:
    """P1 supervised unit: ``State(t)`` + F target **candidates**.

    ``targets`` holds ``[F_long, F_hold, F_short]`` candidates currently
    sourced from :class:`RewardEngine` totals. These are **not** finalized F
    labels; they approximate F under the present reward design.

    Primary training path (future)::

        actual future  →  F_target
        State(t)       →  Network        →  F_predicted

    Does **not** require Gymnasium.
    """

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
    """Assemble :class:`P1RegressionSample` with F target candidates at ``t``."""

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
