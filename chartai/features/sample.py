"""P1 decision sample — connects state and reward context at time t."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.features.future_context import FutureContextBuilder
from chartai.features.state import MultiTimeframeState, StateBuilder
from chartai.reward.base import RewardBreakdown
from chartai.reward.context import RewardContext
from chartai.reward.engine import RewardEngine


@dataclass(frozen=True)
class P1DecisionSample:
    """Single P1 unit at decision time t — shared by regression targets and env steps.

    ``state`` uses past-only MTF data through t.
    ``reward_context`` uses future 3m bars t+1..t+10 and past-only sigma inputs.

    For supervised P1 learning, :meth:`compute_target_vector` produces
    ``[F_long, F_hold, F_short]`` **candidates** via :class:`RewardEngine`.
    Those values come from ``RewardBreakdown.total`` and are **not** the
    finalized F definition. All three share the same ``reward_context``
    (identical future market path).
    """

    t_index: int
    state: MultiTimeframeState
    reward_context: RewardContext

    @property
    def decision_time(self):
        return self.state.decision_time

    def compute_reward(self, action: Action, engine: RewardEngine) -> RewardBreakdown:
        return engine.compute(action, self.reward_context)

    def compute_all_action_rewards(self, engine: RewardEngine) -> dict[Action, RewardBreakdown]:
        return {
            action: self.compute_reward(action, engine)
            for action in (Action.LONG, Action.HOLD, Action.SHORT)
        }

    def compute_target_vector(self, engine: RewardEngine):
        """Build F target **candidates** ``[F_long, F_hold, F_short]`` at this t.

        Sourced from ``RewardEngine.compute(...).total`` — a temporary bridge
        to the current reward-component design, not finalized F.
        """
        from chartai.features.target import ActionTargetVector

        return ActionTargetVector.from_breakdowns(self.compute_all_action_rewards(engine))


class P1SampleAssembler:
    """Assemble :class:`P1DecisionSample` from state and future context builders."""

    def __init__(
        self,
        state_builder: StateBuilder,
        future_context_builder: FutureContextBuilder,
    ) -> None:
        self._state_builder = state_builder
        self._future_context_builder = future_context_builder

    @property
    def state_builder(self) -> StateBuilder:
        return self._state_builder

    @property
    def future_context_builder(self) -> FutureContextBuilder:
        return self._future_context_builder

    def assemble(self, t_index: int) -> P1DecisionSample:
        state = self._state_builder.build(t_index)
        reward_context = self._future_context_builder.build(t_index)
        if state.t_index != reward_context.t_index:
            raise ValueError(
                f"State t_index={state.t_index} != reward_context t_index={reward_context.t_index}"
            )
        return P1DecisionSample(
            t_index=t_index,
            state=state,
            reward_context=reward_context,
        )
