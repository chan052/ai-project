"""Tests for Path vs S+D role separation audit (analysis-only)."""

from __future__ import annotations

import pytest

from chartai.analysis.path_sd_role_audit import PathSDRoleAuditRunner, StructureTag
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.synthetic import SyntheticScenario, build_scenario
from chartai.reward.speed_persistence import SpeedCandidate, PersistenceCandidate, compute_speed_n, compute_persistence_n
from chartai.core.types import Action
from chartai.reward.path import compute_path_n


def _market_source() -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard()
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_path_sd_role_audit_runs_on_synthetic() -> None:
    report = PathSDRoleAuditRunner(_market_source()).run()
    assert "1_experiment_purpose" in report
    assert "structures_A_through_F" in report
    assert "10_controlled_magnitude_experiment" in report
    assert "should_replace_raw_p_with_sd" in report
    assert len(report["13_CONFIRMED"]) >= 1


def test_controlled_magnitude_s_d_invariant() -> None:
    cfg_h = 10
    decay = 0.75
    small_path = build_scenario(SyntheticScenario.STEADY_UP, horizon=cfg_h)
    large_moves = [0.10] * cfg_h
    from chartai.reward.synthetic import SyntheticPath, _quiet_past, _future_from_relative_moves

    anchor = 100.0
    closes = _future_from_relative_moves(anchor, large_moves)
    large_path = SyntheticPath(
        name="steady_up_large",
        price_at_t=anchor,
        future_closes=closes,
        future_highs=tuple(c * 1.001 for c in closes),
        future_lows=tuple(c * 0.999 for c in closes),
        past_closes_for_sigma=_quiet_past(),
        reward_horizon=cfg_h,
    )
    ctx_s = small_path.to_context()
    ctx_l = large_path.to_context()
    p_s = compute_path_n(ctx_s, Action.LONG, cfg_h, decay_rate=decay)
    p_l = compute_path_n(ctx_l, Action.LONG, cfg_h, decay_rate=decay)
    s_s = compute_speed_n(ctx_s, Action.LONG, cfg_h, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=decay)
    s_l = compute_speed_n(ctx_l, Action.LONG, cfg_h, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=decay)
    d_s = compute_persistence_n(
        ctx_s, Action.LONG, cfg_h, PersistenceCandidate.FAVORABLE_OCCUPANCY, decay_rate=decay
    )
    d_l = compute_persistence_n(
        ctx_l, Action.LONG, cfg_h, PersistenceCandidate.FAVORABLE_OCCUPANCY, decay_rate=decay
    )
    assert p_l > p_s * 5
    assert abs(s_l - s_s) < 0.02
    assert abs(d_l - d_s) < 0.02


def test_structure_tags_present() -> None:
    report = PathSDRoleAuditRunner(_market_source()).run()
    keys = set(report["structures_A_through_F"].keys())
    assert StructureTag.A_RAW_P.value in keys
    assert StructureTag.F_S_PLUS_D_PLUS_U_MINUS_MAE.value in keys
