"""P1 supervised regression target generation tests."""

from __future__ import annotations

import pytest

from chartai.core.types import Action
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.sample import P1DecisionSample
from chartai.features.target import (
    P1_ACTION_TARGET_ORDER,
    ActionTargetVector,
    P1RegressionSample,
    P1RegressionSampleBuilder,
)
from chartai.reward.config import ComponentWeights, RewardConfig
from chartai.reward.engine import RewardEngine


@pytest.fixture
def regression_setup() -> tuple[SyntheticMTFDataset, RewardEngine, P1RegressionSampleBuilder]:
    ds = SyntheticMTFDataset.build_standard()
    reward_cfg = RewardConfig(
        use_path=True,
        use_utility=False,
        use_mae=False,
        use_surprise=False,
        use_hold_neutral_path=True,
        use_hold_movement=False,
        weights=ComponentWeights(path=1.0, hold_neutral_path=1.0),
        path={"gamma": 0.9},
        hold_neutral_path={"scale": 0.01},
    )
    engine = RewardEngine(reward_cfg)
    builder = P1RegressionSampleBuilder(ds.sample_assembler(), engine)
    return ds, engine, builder


def test_all_action_targets_computed_at_same_t(regression_setup) -> None:
    _, engine, builder = regression_setup
    t_index = 50
    sample = builder.build(t_index)
    breakdowns = builder.sample_assembler.assemble(t_index).compute_all_action_rewards(engine)
    assert set(breakdowns) == {Action.LONG, Action.HOLD, Action.SHORT}
    assert sample.targets.f_long == pytest.approx(breakdowns[Action.LONG].total)
    assert sample.targets.f_hold == pytest.approx(breakdowns[Action.HOLD].total)
    assert sample.targets.f_short == pytest.approx(breakdowns[Action.SHORT].total)


def test_target_vector_action_order_is_fixed(regression_setup) -> None:
    _, engine, builder = regression_setup
    sample = builder.build(50)
    assert P1_ACTION_TARGET_ORDER == (Action.LONG, Action.HOLD, Action.SHORT)
    assert list(sample.targets) == [
        sample.targets.for_action(Action.LONG),
        sample.targets.for_action(Action.HOLD),
        sample.targets.for_action(Action.SHORT),
    ]
    assert ActionTargetVector.action_at_index(0) is Action.LONG
    assert ActionTargetVector.action_at_index(1) is Action.HOLD
    assert ActionTargetVector.action_at_index(2) is Action.SHORT


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


def test_targets_are_reward_total_candidates(regression_setup) -> None:
    """Document: F candidates currently equal RewardBreakdown.total, not final F."""
    _, engine, builder = regression_setup
    decision = builder.sample_assembler.assemble(50)
    breakdowns = decision.compute_all_action_rewards(engine)
    targets = decision.compute_target_vector(engine)
    assert targets.f_long == pytest.approx(breakdowns[Action.LONG].total)
    assert targets.f_hold == pytest.approx(breakdowns[Action.HOLD].total)
    assert targets.f_short == pytest.approx(breakdowns[Action.SHORT].total)


def test_all_action_targets_share_same_future_market_path(regression_setup) -> None:
    """LONG/HOLD/SHORT F candidates score one RewardContext — same actual future path."""
    _, engine, builder = regression_setup
    decision = builder.sample_assembler.assemble(50)
    ctx = decision.reward_context
    for action in P1_ACTION_TARGET_ORDER:
        assert decision.compute_reward(action, engine).total == pytest.approx(
            engine.compute(action, ctx).total
        )
    targets = decision.compute_target_vector(engine)
    assert targets.as_tuple() == pytest.approx(
        tuple(engine.compute(a, ctx).total for a in P1_ACTION_TARGET_ORDER)
    )
