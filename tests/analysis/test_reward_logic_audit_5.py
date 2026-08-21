"""Tests for Reward Logic Audit 5 (analysis-only)."""

from __future__ import annotations

from chartai.analysis.mae_recovery_diagnostics import MaeCase, compute_mae_case_diagnostics
from chartai.analysis.reward_logic_audit_5 import RewardLogicAudit5Runner
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.config import UtilityConfig
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


def test_reward_logic_audit_5_runs() -> None:
    report = RewardLogicAudit5Runner(_market_source()).run()
    assert "semantic_role_table" in report
    assert "CONFIRMED" in report
    assert "final_questions_Q1_Q10" in report
    assert report["final_questions_Q1_Q10"]["Q4_MAE_core_blind_spot"]


def test_u_distinguishes_spike_vs_hold_synthetic() -> None:
    h = 10
    spike = build_scenario(SyntheticScenario.UP_THEN_DOWN, horizon=h)
    hold = build_scenario(SyntheticScenario.STEADY_UP, horizon=h)
    cfg = UtilityConfig()
    us = compute_u_diagnostics(spike.to_context(), Action.LONG, horizon=h, utility_config=cfg)
    uh = compute_u_diagnostics(hold.to_context(), Action.LONG, horizon=h, utility_config=cfg)
    assert us.favorable_occupancy != uh.favorable_occupancy or us.u_mean != uh.u_mean


def test_mae_recovery_vs_sustained() -> None:
    h = 10
    rec = build_scenario(SyntheticScenario.DOWN_THEN_UP, horizon=h)
    sus = build_scenario(SyntheticScenario.STEADY_DOWN, horizon=h)
    dr = compute_mae_case_diagnostics(
        rec.to_context(), Action.LONG, MaeCase.EARLY_ADVERS_RECOVERY, horizon=h
    )
    ds = compute_mae_case_diagnostics(
        sus.to_context(), Action.LONG, MaeCase.EARLY_ADVERS_SUSTAINED, horizon=h
    )
    assert dr.recovery_after_mae > ds.recovery_after_mae
