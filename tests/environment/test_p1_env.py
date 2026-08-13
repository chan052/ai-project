"""Phase 2-B P1 Gymnasium environment tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from chartai.core.types import Action
from chartai.data.mtf_aligner import HigherTfBarKind
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.environment.observation import action_from_env_int
from chartai.environment.p1_env import P1TradingEnv
from chartai.reward.config import ComponentWeights, RewardConfig
from chartai.reward.engine import RewardEngine


@pytest.fixture
def env_setup() -> tuple[P1TradingEnv, SyntheticMTFDataset]:
    ds = SyntheticMTFDataset.build_standard()
    reward_cfg = RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        use_surprise=False,
        use_hold_neutral_path=True,
        use_hold_movement=False,
        weights=ComponentWeights(path=1.0, hold_neutral_path=1.0),
        path={"gamma": 0.9},
        hold_neutral_path={"scale": 0.01},
    )
    env = P1TradingEnv(ds, RewardEngine(reward_cfg))
    return env, ds


def test_reset_returns_valid_mtf_observation(env_setup) -> None:
    env, _ = env_setup
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert obs["3m"].shape == (8, 5)
    assert obs["1h"].shape == (4, 5)
    assert obs["4h"].shape == (3, 5)
    assert info["t_index"] == env.bounds.first_t


def test_observation_has_all_timeframes(env_setup) -> None:
    env, _ = env_setup
    obs, _ = env.reset()
    assert "3m" in obs and "1h" in obs and "4h" in obs


def test_observation_can_include_partial_higher_tf_bars(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    assert env.bounds.contains(t_index)
    obs, info = env.reset(options={"start_t": t_index})
    state = ds.state_builder().build(t_index)
    if state.slice_1h.has_partial_bar:
        assert info["has_partial_1h"] is True
        assert obs["1h_partial_flags"][-1] == 1.0


def test_action_mapping() -> None:
    assert action_from_env_int(0) is Action.LONG
    assert action_from_env_int(1) is Action.HOLD
    assert action_from_env_int(2) is Action.SHORT
    with pytest.raises(ValueError):
        action_from_env_int(3)


def test_different_actions_give_different_rewards(env_setup) -> None:
    env, _ = env_setup
    env.reset(options={"start_t": 50})
    r_long = env.compute_reward_at_current_t(0)
    r_hold = env.compute_reward_at_current_t(1)
    r_short = env.compute_reward_at_current_t(2)
    assert r_long != r_short


def test_step_advances_one_3m_bar(env_setup) -> None:
    env, _ = env_setup
    env.reset(options={"start_t": 50})
    assert env.t_index == 50
    env.step(1)
    assert env.t_index == 51


def test_reward_uses_t_plus_1_to_t_plus_10(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    env.reset(options={"start_t": t_index})
    ctx = ds.future_context_builder().build(t_index)
    assert len(ctx.future_closes) == 10
    expected = tuple(ds.bars_3m[i].close for i in range(t_index + 1, t_index + 11))
    assert ctx.future_closes == expected


def test_t_plus_11_mutation_does_not_change_reward(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    env.reset(options={"start_t": t_index})
    before = env.compute_reward_at_current_t(0)
    ds.set_3m_close(t_index + 11, 99999.0)
    after = env.compute_reward_at_current_t(0)
    assert before == pytest.approx(after)


def test_future_mutation_does_not_change_current_state(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    obs_before, _ = env.reset(options={"start_t": t_index})
    fp = ds.state_builder().build(t_index).fingerprint()
    ds.set_3m_close(t_index + 5, 5000.0)
    obs_after, _ = env.reset(options={"start_t": t_index})
    assert ds.state_builder().build(t_index).fingerprint() == fp
    np.testing.assert_array_equal(obs_before["3m"], obs_after["3m"])


def test_next_state_uses_observable_data_only(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    env.reset(options={"start_t": t_index})
    env.step(0)
    state = ds.state_builder().build(env.t_index)
    decision = ds.aligner().decision_time_at_3m_index(env.t_index)
    for bar in state.slice_3m.bars:
        assert bar.end <= decision.timestamp


def test_partial_bars_exclude_future_data(env_setup) -> None:
    env, ds = env_setup
    t_index = 50
    env.reset(options={"start_t": t_index})
    state = ds.state_builder().build(t_index)
    decision = ds.aligner().decision_time_at_3m_index(t_index)
    for tf_name, slice_ in (("1h", state.slice_1h), ("4h", state.slice_4h)):
        for sb in slice_.state_bars:
            if sb.kind is HigherTfBarKind.PARTIAL:
                m3 = ds.aligner().contributing_3m_bars_for_interval(
                    sb.bar.start, sb.bar.end, decision
                )
                assert all(b.end <= decision.timestamp for b in m3)


def test_episode_terminates_after_last_valid_t(env_setup) -> None:
    env, _ = env_setup
    env.reset(options={"start_t": env.bounds.last_t})
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True


def test_no_reward_calculation_beyond_last_valid_t(env_setup) -> None:
    env, ds = env_setup
    last_t = env.bounds.last_t
    with pytest.raises((ValueError, IndexError)):
        ds.sample_assembler().assemble(last_t + 1)


def test_reset_is_deterministic_with_seed(env_setup) -> None:
    env, _ = env_setup
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    for key in ("3m", "1h", "4h"):
        np.testing.assert_array_equal(obs_a[key], obs_b[key])
