"""Tests for future-path observables."""

from __future__ import annotations

from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.path_observables import compute_mfe_n, compute_path_observables


def test_mfe_non_negative() -> None:
    ds = SyntheticMTFDataset.build_standard()
    ctx = ds.future_context_builder().build(50)
    mfe = compute_mfe_n(ctx, Action.LONG, 10)
    assert mfe >= 0


def test_path_observables_fields() -> None:
    ds = SyntheticMTFDataset.build_standard()
    ctx = ds.future_context_builder().build(50)
    obs = compute_path_observables(ctx, Action.LONG, 10)
    assert obs.n == 10
    assert obs.favorable_duration + obs.adverse_duration <= 10
    assert 0 <= obs.favorable_occupancy <= 1
