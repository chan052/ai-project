"""Tests for Reward Logic Audit 4 (analysis-only)."""

from __future__ import annotations

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.reward_logic_audit_4 import RewardLogicAudit4Runner
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.synthetic import SyntheticScenario, build_scenario


def _market_source() -> MarketDataSource:
    ds = SyntheticMTFDataset.build_standard()
    return MarketDataSource(
        symbol="SYNTH",
        bars=ds.bars_3m,
        source="synthetic",
        start_time=ds.bars_3m[0].start,
        end_time=ds.bars_3m[-1].end,
    )


def test_reward_logic_audit_4_runs() -> None:
    report = RewardLogicAudit4Runner(_market_source()).run()
    assert "1_fn_time_profile_audit" in report
    assert "2_mae_role_decomposition_audit" in report
    assert "CONFIRMED" in report
    assert "DO_NOT_CONCLUDE" in report
    assert len(report["CONFIRMED"]) >= 1


def test_mae_diagnostics_distinguish_case_a_vs_sustained() -> None:
    h = 10
    ctx_a = build_scenario(SyntheticScenario.DOWN_THEN_UP, horizon=h).to_context()
    ctx_s = build_scenario(SyntheticScenario.STEADY_DOWN, horizon=h).to_context()
    da = compute_mae_diagnostics(ctx_a, Action.LONG, h, early_bars=3)
    ds = compute_mae_diagnostics(ctx_s, Action.LONG, h, early_bars=3)
    assert da.recovery_after_mae > ds.recovery_after_mae


def test_fn_aggregation_sections_present() -> None:
    report = RewardLogicAudit4Runner(_market_source()).run()
    agg = report["1_fn_time_profile_audit"]["aggregation_comparison"]
    assert "A_mean_f1_f10" in agg
    assert "D_late_only_f8_f10" in agg
    profile = report["1_fn_time_profile_audit"]["individual_fn_profile_eval_mean"]
    assert "f_1" in profile
    assert "f_10" in profile
