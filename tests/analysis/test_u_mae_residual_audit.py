"""Tests for U/MAE Residual Path Information Audit (analysis-only)."""

from __future__ import annotations

from chartai.analysis.path_residual_diagnostics import (
    CANDIDATE_SPECS,
    compute_path_residual_observables,
    get_candidate_value,
)
from chartai.analysis.u_mae_residual_audit import (
    UMaeResidualAuditRunner,
    UMaeResidualAuditConfig,
)
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.synthetic import build_scenario, SyntheticScenario


def _market_source() -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard(num_3m=150, reward_horizon=10)
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_u_mae_residual_audit_runs() -> None:
    report = UMaeResidualAuditRunner(_market_source()).run()
    assert "double_counting_table" in report
    assert "CONFIRMED" in report
    assert "RESIDUAL_PATH_CANDIDATES" in report
    assert "synthetic_archetype_pairs" in report
    assert len(report["synthetic_archetype_pairs"]) == 5


def test_path_residual_observables_keys() -> None:
    h = 10
    path = build_scenario(SyntheticScenario.UP_THEN_DOWN, horizon=h)
    obs = compute_path_residual_observables(path.to_context(), Action.LONG, h)
    for spec in CANDIDATE_SPECS:
        val = get_candidate_value(obs, spec)
        assert isinstance(val, float)


def test_case1_giveback_discriminates() -> None:
    cfg = UMaeResidualAuditConfig(reward_horizon=10)
    runner = UMaeResidualAuditRunner(_market_source(), config=cfg)
    pairs = runner._synthetic_case_pairs(cfg)
    case1 = next(c for c in pairs if c["case_id"] == 1)
    gb_a = case1["path_a"]["candidates"]["giveback_ratio"]
    gb_b = case1["path_b"]["candidates"]["giveback_ratio"]
    assert gb_a > gb_b


def test_case4_oscillation_higher_on_round_trip() -> None:
    cfg = UMaeResidualAuditConfig(reward_horizon=10)
    runner = UMaeResidualAuditRunner(_market_source(), config=cfg)
    pairs = runner._synthetic_case_pairs(cfg)
    case4 = next(c for c in pairs if c["case_id"] == 4)
    chop_a = case4["path_a"]["candidates"]["oscillation_chop"]
    chop_b = case4["path_b"]["candidates"]["oscillation_chop"]
    assert chop_a > chop_b
