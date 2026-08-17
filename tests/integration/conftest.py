"""Phase 2-A integration test fixtures."""

from __future__ import annotations

import pytest

from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.config import RewardConfig


@pytest.fixture
def mtf_dataset() -> SyntheticMTFDataset:
    return SyntheticMTFDataset.build_standard()


@pytest.fixture
def t_index(mtf_dataset: SyntheticMTFDataset) -> int:
    return 50


@pytest.fixture
def reward_engine_config() -> RewardConfig:
    return RewardConfig(
        use_path=True,
        use_utility=True,
        use_mae=True,
        path={"gamma": 0.75},
        utility={"alpha": 1.0, "beta": 2.0, "lambda": 1.5},
    )
