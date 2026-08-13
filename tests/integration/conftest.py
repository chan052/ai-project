"""Phase 2-A integration test fixtures."""

from __future__ import annotations

import pytest

from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.config import ComponentWeights, RewardConfig


@pytest.fixture
def mtf_dataset() -> SyntheticMTFDataset:
    return SyntheticMTFDataset.build_standard()


@pytest.fixture
def t_index(mtf_dataset: SyntheticMTFDataset) -> int:
    # Valid t with full reward horizon and sufficient lookback history.
    return 50


@pytest.fixture
def reward_engine_config() -> RewardConfig:
    return RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        use_surprise=True,
        use_hold_neutral_path=True,
        use_hold_movement=False,
        use_hold_surprise=True,
        weights=ComponentWeights(
            path=1.0,
            surprise=0.1,
            hold_neutral_path=1.0,
            hold_surprise=1.0,
        ),
        path={"gamma": 0.9},
        hold_neutral_path={"scale": 0.01},
    )
