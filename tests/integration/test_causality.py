"""Phase 2-A end-to-end causality integration tests."""

from __future__ import annotations

import pytest

from chartai.core.types import Action, Timeframe
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.reward.config import RewardConfig
from chartai.reward.engine import RewardEngine


def _assemble(ds: SyntheticMTFDataset, t_index: int):
    return ds.sample_assembler().assemble(t_index)


def test_state_future_separation(mtf_dataset: SyntheticMTFDataset, t_index: int) -> None:
    sample = _assemble(mtf_dataset, t_index)
    assert sample.state.t_index == sample.reward_context.t_index == t_index
    assert sample.state.decision_time.timestamp == mtf_dataset.aligner().decision_time_at_3m_index(
        t_index
    ).timestamp
    future_builder = mtf_dataset.future_context_builder()
    used = future_builder.reward_indices_used(t_index)
    assert list(used) == list(range(t_index + 1, t_index + 1 + mtf_dataset.reward_horizon))


def test_a_future_mutation_does_not_change_state(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
    reward_engine_config: RewardConfig,
) -> None:
    sample_before = _assemble(mtf_dataset, t_index)
    fp_before = sample_before.state.fingerprint()
    ctx_fp_before = mtf_dataset.future_context_builder().fingerprint(t_index)

    mutate_idx = t_index + 5
    mtf_dataset.set_3m_close(mutate_idx, sample_before.reward_context.future_closes[4] * 10.0)

    sample_after = _assemble(mtf_dataset, t_index)
    ctx_fp_after = mtf_dataset.future_context_builder().fingerprint(t_index)
    engine = RewardEngine(reward_engine_config)
    reward_before = sample_before.compute_reward(Action.LONG, engine).total
    reward_after = sample_after.compute_reward(Action.LONG, engine).total

    assert sample_after.state.fingerprint() == fp_before
    assert ctx_fp_after != ctx_fp_before
    assert reward_after != reward_before


def _future_reward_slice(ctx) -> tuple:
    """Future-only portion of reward context — excludes past sigma series."""
    return (ctx.t_index, ctx.price_at_t, ctx.future_closes)


def test_b_past_state_mutation_does_not_change_future_context(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
    reward_engine_config: RewardConfig,
) -> None:
    sample_before = _assemble(mtf_dataset, t_index)
    future_before = _future_reward_slice(sample_before.reward_context)
    engine = RewardEngine(reward_engine_config)
    reward_before = sample_before.compute_all_action_rewards(engine)

    # Mutate past 1H bar in the active state window (not in 3m reward series).
    state_before = sample_before.state
    h1_end_index = state_before.slice_1h.window.end_index
    mtf_dataset.set_1h_close(h1_end_index, 888.0)

    sample_after = _assemble(mtf_dataset, t_index)
    future_after = _future_reward_slice(sample_after.reward_context)
    reward_after = sample_after.compute_all_action_rewards(engine)

    assert sample_after.state.slice_1h.closes != sample_before.state.slice_1h.closes
    assert future_after == future_before
    assert sample_after.reward_context.past_closes_for_sigma == (
        sample_before.reward_context.past_closes_for_sigma
    )
    for action in (Action.LONG, Action.HOLD, Action.SHORT):
        assert reward_after[action].total == pytest.approx(reward_before[action].total)


def test_c_partial_higher_tf_bar_included_at_t() -> None:
    """At decision inside an 1H interval, state includes partial bar through t."""
    t_index = 40
    ds = SyntheticMTFDataset.build_with_incomplete_higher_tf_at(t_index)
    aligner = ds.aligner()
    decision = aligner.decision_time_at_3m_index(t_index)
    state = ds.state_builder().build(t_index)

    assert state.slice_1h.has_partial_bar
    partial = state.slice_1h.state_bars[-1]
    assert partial.kind.value == "partial"
    assert partial.bar.start < decision.timestamp < partial.bar.end

    m3_contrib = aligner.contributing_3m_bars_for_interval(
        partial.bar.start, partial.bar.end, decision
    )
    assert partial.bar.close == m3_contrib[-1].close
    assert all(b.end <= decision.timestamp for b in m3_contrib)

    in_progress_native = [
        b for b in ds.bars_1h if b.start < decision.timestamp < b.end
    ]
    assert len(in_progress_native) == 1
    # Native full-interval bar must not appear verbatim in state when partial.
    assert in_progress_native[0] not in state.slice_1h.bars


def test_d_reward_horizon_t_plus_11_mutation(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
    reward_engine_config: RewardConfig,
) -> None:
    sample_before = _assemble(mtf_dataset, t_index)
    engine = RewardEngine(reward_engine_config)
    rewards_before = sample_before.compute_all_action_rewards(engine)

    beyond_idx = t_index + 11
    assert beyond_idx < len(mtf_dataset.bars_3m)
    mtf_dataset.set_3m_close(beyond_idx, 99999.0)

    sample_after = _assemble(mtf_dataset, t_index)
    rewards_after = sample_after.compute_all_action_rewards(engine)

    assert sample_after.reward_context.future_closes == sample_before.reward_context.future_closes
    for action in (Action.LONG, Action.HOLD, Action.SHORT):
        assert rewards_after[action].total == pytest.approx(rewards_before[action].total)


def test_e_future_mutation_leaves_all_tf_states_unchanged(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
) -> None:
    sample_before = _assemble(mtf_dataset, t_index)
    fp = sample_before.state.fingerprint()

    for idx in (t_index + 1, t_index + 5, t_index + 10):
        mtf_dataset.set_3m_close(idx, mtf_dataset.bars_3m[idx].close * 5.0)

    sample_after = _assemble(mtf_dataset, t_index)
    assert sample_after.state.fingerprint() == fp
    assert sample_after.state.slice_3m.closes == sample_before.state.slice_3m.closes
    assert sample_after.state.slice_1h.closes == sample_before.state.slice_1h.closes
    assert sample_after.state.slice_4h.closes == sample_before.state.slice_4h.closes


def test_sigma_uses_past_only(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
) -> None:
    future_builder = mtf_dataset.future_context_builder()
    sigma_before = future_builder.sigma_at_t(t_index)

    mtf_dataset.set_3m_close(t_index + 3, 500.0)
    sigma_after_future_mut = mtf_dataset.future_context_builder().sigma_at_t(t_index)
    assert sigma_after_future_mut == pytest.approx(sigma_before)

    mtf_dataset.set_3m_close(t_index - 2, 50.0)
    sigma_after_past_mut = mtf_dataset.future_context_builder().sigma_at_t(t_index)
    assert sigma_after_past_mut != pytest.approx(sigma_before)


def test_all_actions_rewards_from_same_sample(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
    reward_engine_config: RewardConfig,
) -> None:
    sample = _assemble(mtf_dataset, t_index)
    engine = RewardEngine(reward_engine_config)
    rewards = sample.compute_all_action_rewards(engine)
    assert set(rewards.keys()) == {Action.LONG, Action.HOLD, Action.SHORT}
    for action, breakdown in rewards.items():
        assert breakdown.action is action


def test_reward_window_exactly_t_plus_1_to_t_plus_10(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
) -> None:
    ctx = mtf_dataset.future_context_builder().build(t_index)
    assert len(ctx.future_closes) == 10
    expected = tuple(mtf_dataset.bars_3m[i].close for i in range(t_index + 1, t_index + 11))
    assert ctx.future_closes == expected
    assert ctx.price_at_t == mtf_dataset.bars_3m[t_index].close


def test_lookback_none_raises_configuration_error() -> None:
    ds = SyntheticMTFDataset.build_standard()
    from chartai.core.config import StateConfig, TimeframeStateConfig

    ds.state_config = StateConfig(
        timeframes={
            "3m": TimeframeStateConfig(lookback_bars=None),
            "1h": TimeframeStateConfig(lookback_bars=3),
            "4h": TimeframeStateConfig(lookback_bars=2),
        }
    )
    with pytest.raises(ValueError, match="lookback_bars"):
        ds.state_builder().build(50)


def test_sample_assembler_wires_state_and_reward(
    mtf_dataset: SyntheticMTFDataset,
    t_index: int,
) -> None:
    assembler = mtf_dataset.sample_assembler()
    sample = assembler.assemble(t_index)
    assert sample.state.t_index == t_index
    assert sample.reward_context.t_index == t_index
    assert sample.state.slice_3m.window.end_index == t_index
