"""Shared reward test fixtures."""

from __future__ import annotations

import pytest

from chartai.reward.config import ComponentWeights, RewardConfig, UtilityConfig


@pytest.fixture
def base_reward_config() -> RewardConfig:
    """Explicit test weights — not research defaults."""
    return RewardConfig(
        reward_horizon=10,
        use_path=True,
        use_utility=True,
        use_mae=True,
        use_surprise=False,
        use_hold_neutral_path=True,
        use_hold_movement=True,
        use_hold_surprise=False,
        weights=ComponentWeights(
            path=1.0,
            utility=1.0,
            mae=-1.0,
            surprise=0.5,
            hold_neutral_path=1.0,
            hold_movement=-1.0,
            hold_surprise=1.0,
        ),
        path={"gamma": 0.9},
        utility=UtilityConfig(alpha=1.0, beta=1.0, lambda_=2.0),
        hold_neutral_path={"scale": 0.01},
    )


@pytest.fixture
def path_only_config() -> RewardConfig:
    return RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        use_surprise=False,
        weights=ComponentWeights(path=1.0),
        path={"gamma": 0.9},
    )
