"""Tests for swappable Path variant formulations."""

from __future__ import annotations

import math

import pytest

from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.path import compute_path_n
from chartai.reward.path_variants import (
    PathVariant,
    compute_path_n_variant,
    sigma_at_t,
)


@pytest.fixture
def ctx_builder() -> FutureContextBuilder:
    ds = SyntheticMTFDataset.build_standard()
    return ds.future_context_builder()


def test_raw_return_matches_canonical(ctx_builder: FutureContextBuilder) -> None:
    t_index = 50
    ctx = ctx_builder.build(t_index)
    decay = 0.75
    canonical = compute_path_n(ctx, Action.LONG, 5, decay_rate=decay)
    variant = compute_path_n_variant(
        ctx, Action.LONG, 5, variant=PathVariant.RAW_RETURN, decay_rate=decay
    )
    assert variant == pytest.approx(canonical)


def test_sign_based_bounded_in_minus_one_one(ctx_builder: FutureContextBuilder) -> None:
    t_index = 50
    ctx = ctx_builder.build(t_index)
    for n in range(1, 11):
        p = compute_path_n_variant(
            ctx, Action.LONG, n, variant=PathVariant.SIGN_BASED, decay_rate=0.75
        )
        assert -1.0 - 1e-9 <= p <= 1.0 + 1e-9


def test_sigma_uses_past_only(ctx_builder: FutureContextBuilder) -> None:
    ds = SyntheticMTFDataset.build_standard()
    t_index = 50
    past = ds.future_context_builder().build(t_index).past_closes_for_sigma
    sigma_before = sigma_at_t(past, window=20)

    ds.set_3m_close(t_index + 2, ds.bars_3m[t_index + 2].close * 10.0)
    past_after = ds.future_context_builder().build(t_index).past_closes_for_sigma
    sigma_after = sigma_at_t(past_after, window=20)
    assert sigma_after == pytest.approx(sigma_before)

    ds.set_3m_close(t_index - 3, ds.bars_3m[t_index - 3].close * 0.1)
    past_mut = ds.future_context_builder().build(t_index).past_closes_for_sigma
    sigma_past = sigma_at_t(past_mut, window=20)
    assert sigma_past != pytest.approx(sigma_before)


def test_vol_and_tanh_finite(ctx_builder: FutureContextBuilder) -> None:
    ctx = ctx_builder.build(50)
    for variant in (PathVariant.VOL_NORMALIZED, PathVariant.BOUNDED_TANH):
        for n in range(1, 11):
            val = compute_path_n_variant(ctx, Action.LONG, n, variant=variant, decay_rate=0.75)
            assert math.isfinite(val)
