"""Phase 1 F-target engine tests — structure and causality, not optimal numeric values."""

from __future__ import annotations

import dataclasses

import pytest

from chartai.core.types import Action
from chartai.reward.config import RewardConfig
from chartai.reward.context import RewardContext
from chartai.reward.engine import RewardEngine
from chartai.reward.f_composer import FTargetComposer
from chartai.reward.mae import compute_mae_n, long_downward_excursion, short_upward_excursion
from chartai.reward.normalization import IdentityNormalizer
from chartai.reward.path import compute_path_n, gamma_weights, normalized_decay_weights
from chartai.reward.synthetic import (
    SyntheticScenario,
    build_scenario,
    mae_adverse_long_path,
    mae_adverse_short_path,
)
from chartai.reward.utility import compute_utility_n, utility_u


def test_long_short_directional_sign_inverts(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    engine = RewardEngine(base_reward_config)
    long_bd = engine.compute(Action.LONG, ctx)
    short_bd = engine.compute(Action.SHORT, ctx)
    assert long_bd.f_position > short_bd.f_position
    # Path at each n should invert sign.
    for long_fn, short_fn in zip(long_bd.fn_breakdowns, short_bd.fn_breakdowns):
        assert long_fn.path_raw == pytest.approx(-short_fn.path_raw)


def test_wrong_direction_can_be_negative(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    engine = RewardEngine(path_only_config)
    short_bd = engine.compute(Action.SHORT, ctx)
    assert short_bd.f_position < 0


def test_return_from_t_uses_anchor_price() -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    anchor = ctx.price_at_t
    for k in range(1, ctx.reward_horizon + 1):
        expected = (ctx.future_closes[k - 1] - anchor) / anchor
        assert ctx.return_from_t(k) == pytest.approx(expected)


def test_path_n_uses_cumulative_window(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.UP_THEN_FLAT).to_context()
    decay = path_only_config.path.gamma
    p3 = compute_path_n(ctx, Action.LONG, 3, decay_rate=decay)
    p5 = compute_path_n(ctx, Action.LONG, 5, decay_rate=decay)
    # More steps included -> different path score on upward-then-flat path.
    assert p5 != pytest.approx(p3)


def test_path_decay_rate_baseline_075(path_only_config: RewardConfig) -> None:
    assert path_only_config.path.gamma == pytest.approx(0.75)
    weights = normalized_decay_weights(10, 0.75)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[-1]
    legacy = gamma_weights(10, 0.75)
    assert legacy[0] > legacy[-1]


def test_path_decay_rate_changes_weighting(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.UP_THEN_FLAT).to_context()
    n = ctx.reward_horizon
    p_low = compute_path_n(ctx, Action.LONG, n, decay_rate=0.5)
    p_high = compute_path_n(ctx, Action.LONG, n, decay_rate=0.95)
    assert p_low != pytest.approx(p_high)
    weights_low = normalized_decay_weights(n, 0.5)
    weights_high = normalized_decay_weights(n, 0.95)
    assert weights_low[0] / weights_low[-1] > weights_high[0] / weights_high[-1]


def test_utility_branches() -> None:
    assert utility_u(2.0, alpha=1.0, beta=2.0, lambda_=1.5) == pytest.approx(2.0)
    assert utility_u(-2.0, alpha=1.0, beta=2.0, lambda_=1.5) == pytest.approx(-6.0)


def test_utility_config_baseline_applied(base_reward_config: RewardConfig) -> None:
    assert base_reward_config.utility.alpha == pytest.approx(1.0)
    assert base_reward_config.utility.beta == pytest.approx(2.0)
    assert base_reward_config.utility.lambda_ == pytest.approx(1.5)
    ctx = build_scenario(SyntheticScenario.FLAT).to_context()
    x = ctx.return_from_t(5)
    expected = utility_u(x, alpha=1.0, beta=2.0, lambda_=1.5)
    assert compute_utility_n(ctx, Action.LONG, 5, base_reward_config.utility) == pytest.approx(
        expected
    )


def test_long_mae_uses_minimum_low() -> None:
    ctx = mae_adverse_long_path().to_context()
    mae3 = compute_mae_n(ctx, Action.LONG, 3)
    expected = (ctx.price_at_t - min(ctx.future_lows[:3])) / ctx.price_at_t
    assert mae3 == pytest.approx(expected)
    assert mae3 == pytest.approx(long_downward_excursion(ctx, n=3))
    assert mae3 > 0


def test_short_mae_uses_maximum_high() -> None:
    ctx = mae_adverse_short_path().to_context()
    mae3 = compute_mae_n(ctx, Action.SHORT, 3)
    expected = (max(ctx.future_highs[:3]) - ctx.price_at_t) / ctx.price_at_t
    assert mae3 == pytest.approx(expected)
    assert mae3 == pytest.approx(short_upward_excursion(ctx, n=3))
    assert mae3 > 0


def test_mae_accumulates_over_n() -> None:
    ctx = mae_adverse_long_path().to_context()
    mae1 = compute_mae_n(ctx, Action.LONG, 1)
    mae5 = compute_mae_n(ctx, Action.LONG, 5)
    assert mae5 >= mae1


def test_one_fn_per_horizon_step(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    bd = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    assert len(bd.fn_values) == 10
    assert len(bd.fn_breakdowns) == 10
    assert [fb.n for fb in bd.fn_breakdowns] == list(range(1, 11))


def test_f_position_is_simple_mean_of_fn(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    bd = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    assert bd.f_position == pytest.approx(sum(bd.fn_values) / len(bd.fn_values))


def test_f_position_has_no_extra_temporal_weighting(path_only_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.UP_THEN_FLAT).to_context()
    composer = FTargetComposer(path_only_config)
    bd = composer.compose(ctx, Action.LONG)
    manual_mean = sum(bd.fn_values) / len(bd.fn_values)
    assert bd.f_position == pytest.approx(manual_mean)


def test_hold_not_in_p1_action_space() -> None:
    assert list(Action) == [Action.LONG, Action.SHORT]
    assert Action.LONG.value == 0
    assert Action.SHORT.value == 1


def test_s_move_not_in_f_computation(base_reward_config: RewardConfig) -> None:
    engine = RewardEngine(base_reward_config)
    assert "surprise" not in engine.enabled_component_names()
    ctx = build_scenario(SyntheticScenario.QUIET_THEN_BIG_UP).to_context()
    bd = engine.compute(Action.LONG, ctx)
    assert "s_move" not in bd.metadata


def test_no_excess_loss_component_in_f(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    bd = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    for fb in bd.fn_breakdowns:
        assert set(fb.__dataclass_fields__) == {
            "n",
            "path_raw",
            "utility_raw",
            "mae_raw",
            "path_normalized",
            "utility_normalized",
            "mae_normalized",
            "f_n",
        }


def test_normalization_is_identity_placeholder(base_reward_config: RewardConfig) -> None:
    engine = RewardEngine(base_reward_config, normalizer=IdentityNormalizer())
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    bd = engine.compute(Action.LONG, ctx)
    for fb in bd.fn_breakdowns:
        assert fb.path_normalized == pytest.approx(fb.path_raw)
        assert fb.utility_normalized == pytest.approx(fb.utility_raw)
        assert fb.mae_normalized == pytest.approx(fb.mae_raw)


def test_component_off_removes_influence(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    full = RewardEngine(base_reward_config).compute(Action.LONG, ctx)
    no_mae_cfg = base_reward_config.model_copy(update={"use_mae": False})
    no_mae = RewardEngine(no_mae_cfg).compute(Action.LONG, ctx)
    assert no_mae.f_position != pytest.approx(full.f_position)


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
    engine = RewardEngine(RewardConfig(use_path=True, use_utility=False, use_mae=False))
    assert "d_ret" not in engine.enabled_component_names()


def test_directional_path_ordering_steady_up_vs_steady_down(path_only_config: RewardConfig) -> None:
    engine = RewardEngine(path_only_config)
    up = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    down = build_scenario(SyntheticScenario.STEADY_DOWN).to_context()
    assert engine.compute(Action.LONG, up).f_position > engine.compute(Action.LONG, down).f_position
    assert engine.compute(Action.SHORT, down).f_position > engine.compute(Action.SHORT, up).f_position


def test_fn_formula_structure(base_reward_config: RewardConfig) -> None:
    ctx = build_scenario(SyntheticScenario.STEADY_UP).to_context()
    composer = FTargetComposer(base_reward_config)
    fb = composer.compose_fn(ctx, Action.LONG, 5)
    alpha = base_reward_config.utility.alpha
    lambda_ = base_reward_config.utility.lambda_
    expected = fb.path_normalized + alpha * fb.utility_normalized - lambda_ * fb.mae_normalized
    assert fb.f_n == pytest.approx(expected)
