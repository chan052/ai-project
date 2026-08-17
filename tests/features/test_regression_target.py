"""P1 supervised regression target generation tests."""

from __future__ import annotations

import pytest

from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.target import (
    P1_ACTION_TARGET_ORDER,
    ActionTargetVector,
    P1RegressionSample,
    P1RegressionSampleBuilder,
)
from chartai.reward.config import RewardConfig
from chartai.reward.engine import RewardEngine


@pytest.fixture
def regression_setup() -> tuple[SyntheticMTFDataset, RewardEngine, P1RegressionSampleBuilder]:
    ds = SyntheticMTFDataset.build_standard()
    reward_cfg = RewardConfig(
        use_path=True,
        use_utility=True,
        use_mae=True,
        path={"gamma": 0.75},
        utility={"alpha": 1.0, "beta": 2.0, "lambda": 1.5},
    )
    engine = RewardEngine(reward_cfg)
    builder = P1RegressionSampleBuilder(ds.sample_assembler(), engine)
    return ds, engine, builder


def test_all_action_targets_computed_at_same_t(regression_setup) -> None:
    _, engine, builder = regression_setup
    t_index = 50
    sample = builder.build(t_index)
    breakdowns = builder.sample_assembler.assemble(t_index).compute_all_action_targets(engine)
    assert set(breakdowns) == {Action.LONG, Action.SHORT}
    assert sample.targets.f_long == pytest.approx(breakdowns[Action.LONG].f_position)
    assert sample.targets.f_short == pytest.approx(breakdowns[Action.SHORT].f_position)


def test_target_vector_action_order_is_fixed(regression_setup) -> None:
    _, _, builder = regression_setup
    sample = builder.build(50)
    assert P1_ACTION_TARGET_ORDER == (Action.LONG, Action.SHORT)
    assert list(sample.targets) == [
        sample.targets.for_action(Action.LONG),
        sample.targets.for_action(Action.SHORT),
    ]
    assert ActionTargetVector.action_at_index(0) is Action.LONG
    assert ActionTargetVector.action_at_index(1) is Action.SHORT


def test_t_plus_11_mutation_does_not_change_targets(regression_setup) -> None:
    ds, _, builder = regression_setup
    t_index = 50
    before = builder.build(t_index).targets.as_tuple()
    ds.set_3m_close(t_index + 11, 99999.0)
    after = builder.build(t_index).targets.as_tuple()
    assert before == pytest.approx(after)


def test_state_has_no_future_contamination(regression_setup) -> None:
    ds, _, builder = regression_setup
    t_index = 50
    fp_before = builder.build(t_index).state.fingerprint()
    ds.set_3m_close(t_index + 5, 5000.0)
    fp_after = builder.build(t_index).state.fingerprint()
    assert fp_before == fp_after


def test_target_generation_is_deterministic(regression_setup) -> None:
    _, engine, builder = regression_setup
    t_index = 50
    sample_a = builder.build(t_index)
    sample_b = builder.build(t_index)
    decision = builder.sample_assembler.assemble(t_index)
    direct = P1RegressionSample.from_decision_sample(decision, engine)
    assert sample_a.targets.as_tuple() == pytest.approx(sample_b.targets.as_tuple())
    assert direct.targets.as_tuple() == pytest.approx(sample_a.targets.as_tuple())


def test_targets_equal_f_position(regression_setup) -> None:
    _, engine, builder = regression_setup
    decision = builder.sample_assembler.assemble(50)
    breakdowns = decision.compute_all_action_targets(engine)
    targets = decision.compute_target_vector(engine)
    assert targets.f_long == pytest.approx(breakdowns[Action.LONG].f_position)
    assert targets.f_short == pytest.approx(breakdowns[Action.SHORT].f_position)


def test_all_action_targets_share_same_future_market_path(regression_setup) -> None:
    _, engine, builder = regression_setup
    decision = builder.sample_assembler.assemble(50)
    ctx = decision.reward_context
    for action in P1_ACTION_TARGET_ORDER:
        assert decision.compute_f_target(action, engine).f_position == pytest.approx(
            engine.compute(action, ctx).f_position
        )
    targets = decision.compute_target_vector(engine)
    assert targets.as_tuple() == pytest.approx(
        tuple(engine.compute(a, ctx).f_position for a in P1_ACTION_TARGET_ORDER)
    )


def test_no_hold_in_targets(regression_setup) -> None:
    _, _, builder = regression_setup
    sample = builder.build(50)
    assert len(sample.targets) == 2
    assert not hasattr(sample.targets, "f_hold")
