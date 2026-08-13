"""P1 Gymnasium environment — 3-action direction judgment at each 3m bar."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete

from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.environment.episode import EpisodeBounds, compute_episode_bounds
from chartai.environment.observation import P1ObservationAdapter, action_from_env_int
from chartai.features.sample import P1SampleAssembler
from chartai.features.state import MultiTimeframeState
from chartai.reward.engine import RewardEngine


class P1TradingEnv(Env):
    """Optional sequential interface for P1 — not the primary supervised training path.

    P1 learning is defined as supervised regression:
    ``State(t) -> [F_long, F_hold, F_short]`` (see :mod:`chartai.features.target`).
    Use :class:`P1RegressionSampleBuilder` for dataset/target generation.

    This environment remains for validation, causality checks, and sequential
    data iteration. It is **not** a PPO / policy-gradient training requirement.

    Each step:
        1. Reward at current ``t`` for ``action``
        2. Advance decision index by **one** 3m bar (``t -> t+1``)

    Not a trade execution simulator (no entry/exit/holding — P2 scope).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: SyntheticMTFDataset,
        reward_engine: RewardEngine,
        *,
        start_t: int | None = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._reward_engine = reward_engine
        self._assembler = dataset.sample_assembler()
        self._state_builder = self._assembler.state_builder
        self._future_builder = self._assembler.future_context_builder
        self._obs_adapter = P1ObservationAdapter(dataset.state_config)
        self._bounds = compute_episode_bounds(
            num_3m_bars=len(dataset.bars_3m),
            reward_horizon=dataset.reward_horizon,
            state_builder=self._state_builder,
            future_context_builder=self._future_builder,
        )
        self._start_t = start_t if start_t is not None else self._bounds.first_t
        if not self._bounds.contains(self._start_t):
            raise ValueError(
                f"start_t={self._start_t} outside episode bounds "
                f"[{self._bounds.first_t}, {self._bounds.last_t}]"
            )

        self._t_index: int | None = None
        self._current_state: MultiTimeframeState | None = None

        self.action_space = Discrete(3)
        self.observation_space = self._obs_adapter.gymnasium_space()

    @property
    def bounds(self) -> EpisodeBounds:
        return self._bounds

    @property
    def t_index(self) -> int | None:
        return self._t_index

    @property
    def sample_assembler(self) -> P1SampleAssembler:
        return self._assembler

    @property
    def reward_engine(self) -> RewardEngine:
        return self._reward_engine

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        t_index = opts.get("start_t", self._start_t)
        if not self._bounds.contains(t_index):
            raise ValueError(f"start_t={t_index} outside episode bounds")

        self._t_index = int(t_index)
        self._current_state = self._state_builder.build(self._t_index)
        obs = self._obs_adapter.to_observation(self._current_state)
        return obs, self._info(reward_breakdown=None)

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._t_index is None or self._current_state is None:
            raise RuntimeError("step() called before reset()")
        if not self._bounds.contains(self._t_index):
            raise RuntimeError(f"Invalid t_index={self._t_index} at step start")

        p1_action = action_from_env_int(action)
        sample = self._assembler.assemble(self._t_index)
        breakdown = self._reward_engine.compute(p1_action, sample.reward_context)
        reward = float(breakdown.total)

        next_t = self._t_index + 1
        terminated = next_t > self._bounds.last_t
        truncated = False

        info = self._info(reward_breakdown=breakdown, action=p1_action)

        if terminated:
            self._t_index = next_t
            obs = self._obs_adapter.to_observation(self._current_state)
            return obs, reward, terminated, truncated, info

        self._t_index = next_t
        self._current_state = self._state_builder.build(self._t_index)
        obs = self._obs_adapter.to_observation(self._current_state)
        return obs, reward, terminated, truncated, info

    def _info(self, reward_breakdown, action: Action | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "t_index": self._t_index,
            "bounds": {
                "first_t": self._bounds.first_t,
                "last_t": self._bounds.last_t,
                "reward_horizon": self._bounds.reward_horizon,
            },
        }
        if self._current_state is not None:
            info["decision_time"] = self._current_state.decision_time.timestamp.isoformat()
            info["has_partial_1h"] = self._current_state.slice_1h.has_partial_bar
            info["has_partial_4h"] = self._current_state.slice_4h.has_partial_bar
        if action is not None:
            info["action"] = action.name
        if reward_breakdown is not None:
            info["reward_breakdown"] = {
                "total": reward_breakdown.total,
                "components": dict(reward_breakdown.components),
                "multipliers": dict(reward_breakdown.multipliers),
            }
        return info

    def compute_reward_at_current_t(self, action: int) -> float:
        """Expose Reward(t, action) without stepping — for tests."""
        if self._t_index is None:
            raise RuntimeError("compute_reward_at_current_t called before reset()")
        sample = self._assembler.assemble(self._t_index)
        p1_action = action_from_env_int(action)
        return float(self._reward_engine.compute(p1_action, sample.reward_context).total)
