"""Tests for Speed / Persistence candidates."""

from __future__ import annotations

import pytest

from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.core.types import Action
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SDPair,
    SpeedCandidate,
    compute_persistence_n,
    compute_sd_pair_n,
    compute_speed_n,
)


@pytest.fixture
def ctx():
    ds = SyntheticMTFDataset.build_standard()
    return ds.future_context_builder().build(50)


def test_speed_persistence_bounded_or_finite(ctx) -> None:
    for sc in SpeedCandidate:
        for n in range(1, 11):
            val = compute_speed_n(ctx, Action.LONG, n, sc, decay_rate=0.75)
            assert val == pytest.approx(val)
    for pc in PersistenceCandidate:
        for n in range(1, 11):
            val = compute_persistence_n(ctx, Action.LONG, n, pc, decay_rate=0.75)
            assert val == pytest.approx(val)


def test_sd_pair_runs(ctx) -> None:
    for pair in SDPair:
        s, d = compute_sd_pair_n(ctx, Action.LONG, 5, pair, decay_rate=0.75)
        assert s == pytest.approx(s)
        assert d == pytest.approx(d)


def test_time_to_favorable_range(ctx) -> None:
    val = compute_speed_n(ctx, Action.LONG, 10, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=0.75)
    assert 0.0 <= val <= 1.0
