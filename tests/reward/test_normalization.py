"""Tests for component normalization."""

from __future__ import annotations

from chartai.reward.normalization import FittedRobustZScoreNormalizer, FittedZScoreNormalizer


def test_zscore_standardizes_prefix() -> None:
    path = (0.0, 1.0, 2.0, 3.0, 4.0)
    utility = (0.0, 0.5, 1.0, 1.5, 2.0)
    mae = (1.0, 1.0, 1.0, 1.0, 1.0)
    norm = FittedZScoreNormalizer.fit(path, utility, mae)
    z_path = [norm.normalize_path(v) for v in path]
    assert abs(sum(z_path) / len(z_path)) < 0.2


def test_robust_z_handles_zero_mad_with_fallback() -> None:
    utility = (0.0, 0.0, 0.0, 0.0, 0.0)
    norm = FittedRobustZScoreNormalizer.fit((1.0, 2.0), utility, (0.5, 0.5))
    assert norm.utility_stats.scale >= 1e-12
