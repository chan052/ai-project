"""Phase 1 reward engine tests — structure and causality, not optimal numeric values."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chartai.core.types import Action
from chartai.reward.composer import DirectionalRewardComposer, HoldRewardComposer
from chartai.reward.config import ComponentWeights, RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.engine import RewardEngine
from chartai.reward.mae import MaeComponent, long_downward_excursion, short_upward_excursion
from chartai.reward.move_surprise import MoveSurpriseComponent, compute_s_move
from chartai.reward.path import DirectionalPathComponent, gamma_weights
from chartai.reward.synthetic import (
    SyntheticScenario,
    build_scenario,
    hold_quiet_path,
    hold_quatile_volatile_path,
)
from chartai.reward.utility import UtilityComponent, utility_u


def test_long_short_directional_sign_inverts(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    engine = RewardEngine(base_reward_config)
    long_bd = engine.compute(Action.LONG, ctx)
    short_bd = engine.compute(Action.SHORT, ctx)
    assert long_bd.components["path"] > 0
    assert short_bd.components["path"] < 0
    assert long_bd.components["path"] == pytest.approx(-short_bd.components["path"])


def test_wrong_direction_can_be_negative(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    engine = RewardEngine(path_only_config)
    short_bd = engine.compute(Action.SHORT, ctx)
    assert short_bd.components["path"] < 0


def test_s_move_is_non_negative(base_reward_config: RewardConfig) -> None:
    quiet = build_scenario(SyntheticScenario.QUIET_FLAT).to_context()
    big_up = build_scenario(SyntheticScenario.QUIET_THEN_BIG_UP).to_context()
    surprise = MoveSurpriseComponent(base_reward_config.surprise)
    assert surprise.compute_s_move(quiet) >= 0
    assert surprise.compute_s_move(big_up) >= 0


def test_s_move_does_not_encode_direction(base_reward_config: RewardConfig) -> None:
    anchor = 100.0
    past = tuple(100.0 + 0.01 * (i % 3 - 1) for i in range(30))
    up_ctx = RewardContext(
        t_index=100,
        price_at_t=anchor,
        future_closes=(110.0,) + (110.0,) * 9,
        past_closes_for_sigma=past,
    )
    down_ctx = RewardContext(
        t_index=100,
        price_at_t=anchor,
        future_closes=(90.0,) + (90.0,) * 9,
        past_closes_for_sigma=past,
    )
    surprise = MoveSurpriseComponent(base_reward_config.surprise)
    assert surprise.compute_s_move(up_ctx) == pytest.approx(surprise.compute_s_move(down_ctx))


def test_gamma_temporal_weighting(path_only_config: RewardConfig) -> None:
    early = build_scenario(SyntheticScenario.UP_THEN_FLAT).to_context()
    late = build_scenario(SyntheticScenario.FLAT_THEN_UP).to_context()
    low_gamma_cfg = RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        weights=ComponentWeights(path=1.0),
        path={"gamma": 0.5},
    )
    high_gamma_cfg = RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        weights=ComponentWeights(path=1.0),
        path={"gamma": 0.95},
    )
    path_low = DirectionalPathComponent(low_gamma_cfg.path)
    path_high = DirectionalPathComponent(high_gamma_cfg.path)
    early_low = path_low.compute(early, Action.LONG)
    late_low = path_low.compute(late, Action.LONG)
    early_high = path_high.compute(early, Action.LONG)
    late_high = path_high.compute(late, Action.LONG)
    # Lower gamma -> near future weighted more -> early-up beats late-up by wider margin.
    assert early_low - late_low > early_high - late_high
    weights = gamma_weights(10, 0.5)
    assert weights[0] > weights[-1]


def test_utility_branches() -> None:
    assert utility_u(2.0, alpha=2.0, beta=2.0, lambda_=3.0) == pytest.approx(4.0)
    assert utility_u(-2.0, alpha=2.0, beta=2.0, lambda_=3.0) == pytest.approx(-12.0)


def test_long_mae_downward_excursion() -> None:
    ctx = build_scenario(SyntheticScenario.UP_THEN_DOWN).to_context()
    mae = MaeComponent(RewardConfig().mae)
    val = mae.compute(ctx, Action.LONG)
    assert val == pytest.approx(long_downward_excursion(ctx))
    assert val > 0


def test_short_mae_upward_excursion() -> None:
    ctx = build_scenario(SyntheticScenario.DOWN_THEN_UP).to_context()
    mae = MaeComponent(RewardConfig().mae)
    val = mae.compute(ctx, Action.SHORT)
    assert val == pytest.approx(short_upward_excursion(ctx))
    assert val > 0


def test_hold_neutral_path_not_negation_of_directional(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.FLAT_THEN_UP).to_context()
    hold_cfg = RewardConfig(
        use_hold_neutral_path=True,
        use_hold_movement=False,
        weights=ComponentWeights(hold_neutral_path=1.0),
        path={"gamma": 0.9},
        hold_neutral_path={"scale": 0.01},
    )
    hold_bd = RewardEngine(hold_cfg).compute(Action.HOLD, ctx)
    long_bd = RewardEngine(path_only_config).compute(Action.LONG, ctx)
    assert hold_bd.components["hold_neutral_path"] > 0
    assert long_bd.components["path"] > 0
    assert hold_bd.components["hold_neutral_path"] != pytest.approx(-long_bd.components["path"])


def test_hold_neutral_flat_then_up_beats_up_then_flat() -> None:
    cfg = RewardConfig(
        use_hold_neutral_path=True,
        use_hold_movement=False,
        weights=ComponentWeights(hold_neutral_path=1.0),
        path={"gamma": 0.8},
        hold_neutral_path={"scale": 0.02},
    )
    engine = RewardEngine(cfg)
    flat_then_up = build_scenario(SyntheticScenario.FLAT_THEN_UP).to_context()
    up_then_flat = build_scenario(SyntheticScenario.UP_THEN_FLAT).to_context()
    assert engine.compute(Action.HOLD, flat_then_up).total > engine.compute(
        Action.HOLD, up_then_flat
    ).total


def test_hold_movement_detects_large_mid_swing() -> None:
    cfg = RewardConfig(
        use_hold_neutral_path=False,
        use_hold_movement=True,
        weights=ComponentWeights(hold_movement=-1.0),
    )
    engine = RewardEngine(cfg)
    quiet = hold_quiet_path().to_context()
    volatile = hold_quatile_volatile_path().to_context()
    quiet_total = engine.compute(Action.HOLD, quiet).total
    volatile_total = engine.compute(Action.HOLD, volatile).total
    assert volatile_total < quiet_total


def test_hold_surprise_penalizes_large_move() -> None:
    cfg = RewardConfig(
        use_hold_neutral_path=False,
        use_hold_movement=False,
        use_hold_surprise=True,
        weights=ComponentWeights(hold_surprise=1.0),
    )
    engine = RewardEngine(cfg)
    quiet = build_scenario(SyntheticScenario.QUIET_FLAT).to_context()
    big = build_scenario(SyntheticScenario.QUIET_THEN_BIG_UP).to_context()
    assert engine.compute(Action.HOLD, big).total < engine.compute(Action.HOLD, quiet).total


def test_hold_does_not_include_utility(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.FLAT).to_context()
    bd = RewardEngine(base_reward_config).compute(Action.HOLD, ctx)
    assert "utility" not in bd.components
    assert "utility" not in bd.weighted_components


def test_component_off_removes_influence(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    full = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    no_mae_cfg = base_reward_config.model_copy(update={"use_mae": False})
    no_mae = RewardEngine(no_mae_cfg).compute(Action.LONG, ctx)
    assert "mae" not in no_mae.components
    assert no_mae.total != full.total or full.components.get("mae", 0) == 0


def test_reward_window_stays_within_horizon() -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    assert len(ctx.future_closes) == 10
    ctx.validate_temporal_causality()


def test_reward_context_has_no_state_fields() -> None:
    fields = {f.name for f in dataclasses.fields(RewardContext)}
    assert "state" not in fields
    assert "window_3m" not in fields
    assert "features" not in fields


def test_no_d_ret_component_in_engine() -> None:
    from pathlib import Path

    import chartai.reward

    reward_dir = Path(chartai.reward.__file__).parent
    assert not (reward_dir / "market_relative.py").exists()
    cfg = RewardConfig()
    assert "dret" not in cfg.model_dump().keys()
    assert not hasattr(cfg, "use_dret")
    engine = RewardEngine(
        RewardConfig(
            use_path=True,
            use_utility=False,
            use_mae=False,
            weights=ComponentWeights(path=1.0),
            path={"gamma": 0.9},
        )
    )
    assert "d_ret" not in engine.enabled_component_names()


def test_directional_path_ordering_steady_up_vs_up_down(path_only_config: RewardConfig) -> None:
    engine = RewardEngine(path_only_config)
    steady = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    up_down = build_scenario(SyntheticScenario.UP_THEN_DOWN).to_context()
    assert engine.compute(Action.LONG, steady).total > engine.compute(Action.LONG, up_down).total


def test_surprise_multiplier_only_when_enabled(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.QUIET_THEN_BIG_UP).to_context()
    off = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    on_cfg = base_reward_config.model_copy(update={"use_surprise": True})
    on = RewardEngine(on_cfg).compute(Action.LONG, ctx)
    assert "s_move" not in off.multipliers
    assert "s_move" in on.multipliers
