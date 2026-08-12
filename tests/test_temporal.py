"""Tests for 3m TemporalSplit (state vs reward zones)."""

from __future__ import annotations

import pytest

from chartai.core.temporal import TemporalSplit, WindowSpec


def test_state_and_reward_indices_are_disjoint() -> None:
    split = TemporalSplit(t_index=50, spec=WindowSpec(state_window=100, reward_horizon=10))

    state = list(split.state_indices())
    reward = list(split.reward_indices())

    assert state[-1] == 50
    assert state[0] == 0  # clamped at series start
    assert reward[0] == 51
    assert reward[-1] == 60
    assert set(state).isdisjoint(reward)


def test_assert_no_future_in_state_raises() -> None:
    split = TemporalSplit(t_index=10, spec=WindowSpec(state_window=5, reward_horizon=3))
    with pytest.raises(ValueError, match="future index"):
        split.assert_no_future_in_state(range(8, 12))


def test_assert_no_past_in_reward_raises() -> None:
    split = TemporalSplit(t_index=10, spec=WindowSpec(state_window=5, reward_horizon=3))
    with pytest.raises(ValueError, match="non-future index"):
        split.assert_no_past_in_reward(range(9, 13))


def test_reward_horizon_beyond_series_raises() -> None:
    split = TemporalSplit(t_index=98, spec=WindowSpec(state_window=10, reward_horizon=10))
    with pytest.raises(ValueError, match="reward horizon extends beyond"):
        split.assert_valid_series_length(100)


def test_unset_window_spec_returns_none_ranges() -> None:
    split = TemporalSplit(t_index=5, spec=WindowSpec())
    assert split.state_indices() is None
    assert split.reward_indices() is None
