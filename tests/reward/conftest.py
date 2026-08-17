"""Shared reward test fixtures."""

from __future__ import annotations

import pytest

from chartai.reward.config import RewardConfig


@pytest.fixture
def base_reward_config() -> RewardConfig:
    """P1 baseline — alpha=1, beta=2, lambda=1.5, decay r=0.75."""
    return RewardConfig(
        reward_horizon=10,
        use_path=True,
        use_utility=True,
        use_mae=True,
        path={"gamma": 0.75},
        utility={"alpha": 1.0, "beta": 2.0, "lambda": 1.5},
    )


@pytest.fixture
def path_only_config() -> RewardConfig:
    return RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        path={"gamma": 0.75},
    )
