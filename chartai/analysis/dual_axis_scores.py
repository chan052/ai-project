"""Immediate vs Deferred opportunity observables — analysis only, NOT canonical P1.



These scores use hindsight for deferred-axis diagnostics. They must never be used

as P1 state features or canonical reward inputs.

"""



from __future__ import annotations



from dataclasses import dataclass

from statistics import mean



import numpy as np



from chartai.analysis.execution_proxy import (

    TargetStopResult,

    policy_robustness_score,

    simulate_target_stop,

)

from chartai.core.types import Action

from chartai.features.future_context import FutureContextBuilder

from chartai.reward.context import RewardContext

from chartai.reward.path_observables import compute_path_observables





# Extended grid — no single policy is canonical.

STANDARD_POLICY_GRID: tuple[tuple[float, float], ...] = (

    (0.0005, 0.00025),

    (0.001, 0.0005),

    (0.001, 0.001),

    (0.002, 0.001),

    (0.002, 0.002),

    (0.003, 0.0015),

    (0.005, 0.002),

)



VOL_SIGMA_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0)





def policy_key(target_pct: float, stop_pct: float) -> str:

    return f"t{target_pct}_s{stop_pct}"





def build_delayed_context_from_anchor(
    builder: FutureContextBuilder,
    *,
    anchor_t: int,
    delay: int,
    horizon: int,
    max_bar_index: int,
) -> RewardContext | None:
    """Entry at ``anchor_t + delay`` using future bars within ``[anchor_t, anchor_t+horizon]``."""
    remaining = horizon - delay
    if remaining < 1 or anchor_t + horizon > max_bar_index:
        return None
    anchor_ctx = builder.build(anchor_t)
    if delay == 0:
        return truncate_reward_context(anchor_ctx, remaining)
    if delay > len(anchor_ctx.future_closes):
        return None
    return RewardContext(
        t_index=anchor_t + delay,
        price_at_t=anchor_ctx.future_closes[delay - 1],
        future_closes=anchor_ctx.future_closes[delay : delay + remaining],
        future_highs=anchor_ctx.future_highs[delay : delay + remaining],
        future_lows=anchor_ctx.future_lows[delay : delay + remaining],
        past_closes_for_sigma=anchor_ctx.past_closes_for_sigma,
        reward_horizon=remaining,
    )


def truncate_reward_context(ctx: RewardContext, remaining: int) -> RewardContext:
    """Keep only the first ``remaining`` future bars from an entry at ``ctx.t_index``."""
    if remaining < 1 or remaining > ctx.reward_horizon:
        raise ValueError(f"remaining must be in 1..{ctx.reward_horizon}, got {remaining}")
    return RewardContext(
        t_index=ctx.t_index,
        price_at_t=ctx.price_at_t,
        future_closes=ctx.future_closes[:remaining],
        future_highs=ctx.future_highs[:remaining],
        future_lows=ctx.future_lows[:remaining],
        past_closes_for_sigma=ctx.past_closes_for_sigma,
        reward_horizon=remaining,
    )





def _sigma_pct(ctx: RewardContext) -> float:

    past = np.asarray(ctx.past_closes_for_sigma, dtype=float)

    if len(past) < 2:

        return 0.001

    rets = np.diff(past) / past[:-1]

    return float(max(np.std(rets), 1e-6))





def build_policy_grid(ctx: RewardContext) -> tuple[tuple[float, float], ...]:

    """Standard fixed thresholds plus volatility-scaled candidates."""

    sigma = _sigma_pct(ctx)

    vol_policies = tuple(

        (mult * sigma, mult * sigma * 0.5) for mult in VOL_SIGMA_MULTIPLIERS

    )

    return STANDARD_POLICY_GRID + vol_policies





@dataclass(frozen=True)

class PolicyImmediateOutcome:

    target_pct: float

    stop_pct: float

    target_first: bool | None

    proxy_pnl: float





@dataclass(frozen=True)

class ImmediateAxisScores:

    """Observables for entry at t (delay=0)."""



    outcomes: tuple[PolicyImmediateOutcome, ...]

    captureability: float

    robustness: float

    mean_proxy_pnl: float

    composite: float



    @property

    def target_first_rate(self) -> float:

        return self.captureability





@dataclass(frozen=True)

class DeferredPolicyScan:

    target_pct: float

    stop_pct: float

    best_delay: int

    immediate_pnl: float

    best_pnl: float

    improvement: float

    best_target_first: bool | None

    immediate_target_first: bool | None





@dataclass(frozen=True)

class DeferredAxisScores:

    """Hindsight scan over delays τ ∈ [t, t+H] — analysis only."""



    scans: tuple[DeferredPolicyScan, ...]

    consensus_best_delay: int

    mean_improvement: float

    deferred_capture_at_best: float

    deferred_robustness: float

    best_entry_mfe: float

    best_entry_mae: float

    best_entry_terminal: float

    composite: float

    time_to_best_entry: int





def _immediate_outcomes(ctx: RewardContext, action: Action, policies: tuple[tuple[float, float], ...]) -> ImmediateAxisScores:

    outcomes: list[PolicyImmediateOutcome] = []

    for tgt, stp in policies:

        r = simulate_target_stop(ctx, action, target_pct=tgt, stop_pct=stp)

        outcomes.append(

            PolicyImmediateOutcome(

                target_pct=tgt,

                stop_pct=stp,

                target_first=r.target_first,

                proxy_pnl=r.proxy_pnl,

            )

        )

    defined = [o for o in outcomes if o.target_first is not None]

    capture = (

        sum(1 for o in defined if o.target_first) / len(defined) if defined else float("nan")

    )

    pnls = [o.proxy_pnl for o in outcomes]

    robust = policy_robustness_score(pnls)

    composite = capture if not np.isnan(capture) else 0.0

    return ImmediateAxisScores(

        outcomes=tuple(outcomes),

        captureability=capture,

        robustness=robust,

        mean_proxy_pnl=float(mean(pnls)),

        composite=composite,

    )





def _scan_deferred(

    builder: FutureContextBuilder,

    *,

    t_index: int,

    horizon: int,

    action: Action,

    policies: tuple[tuple[float, float], ...],

    max_bar_index: int,

) -> DeferredAxisScores:

    scans: list[DeferredPolicyScan] = []

    best_delays: list[int] = []



    for tgt, stp in policies:

        immediate_ctx = builder.build(t_index)

        imm_trunc = truncate_reward_context(immediate_ctx, horizon)

        imm_r = simulate_target_stop(imm_trunc, action, target_pct=tgt, stop_pct=stp)



        best_d = 0

        best_pnl = imm_r.proxy_pnl

        best_tf = imm_r.target_first



        for delay in range(1, horizon + 1):
            ctx_d = build_delayed_context_from_anchor(
                builder,
                anchor_t=t_index,
                delay=delay,
                horizon=horizon,
                max_bar_index=max_bar_index,
            )
            if ctx_d is None:
                break
            r = simulate_target_stop(ctx_d, action, target_pct=tgt, stop_pct=stp)
            if r.proxy_pnl > best_pnl:
                best_pnl = r.proxy_pnl
                best_d = delay
                best_tf = r.target_first



        scans.append(

            DeferredPolicyScan(

                target_pct=tgt,

                stop_pct=stp,

                best_delay=best_d,

                immediate_pnl=imm_r.proxy_pnl,

                best_pnl=best_pnl,

                improvement=best_pnl - imm_r.proxy_pnl,

                best_target_first=best_tf,

                immediate_target_first=imm_r.target_first,

            )

        )

        best_delays.append(best_d)



    improvements = [s.improvement for s in scans]

    mean_imp = float(mean(improvements)) if improvements else 0.0



    consensus = int(np.median(best_delays)) if best_delays else 0

    remaining = horizon - consensus

    if remaining < 1:

        remaining = horizon

        consensus = 0

    best_ctx = build_delayed_context_from_anchor(
        builder,
        anchor_t=t_index,
        delay=consensus,
        horizon=horizon,
        max_bar_index=max_bar_index,
    )
    if best_ctx is not None:
        obs = compute_path_observables(best_ctx, action, best_ctx.reward_horizon)

        best_mfe = obs.mfe

        best_mae = obs.mae

        best_terminal = obs.terminal_return

    else:

        best_mfe = best_mae = best_terminal = float("nan")



    deferred_outcomes = [

        PolicyImmediateOutcome(

            target_pct=s.target_pct,

            stop_pct=s.stop_pct,

            target_first=s.best_target_first,

            proxy_pnl=s.best_pnl,

        )

        for s in scans

    ]

    defined = [o for o in deferred_outcomes if o.target_first is not None]

    deferred_capture = (

        sum(1 for o in defined if o.target_first) / len(defined) if defined else float("nan")

    )

    deferred_robust = policy_robustness_score([s.best_pnl for s in scans])



    composite = mean_imp



    return DeferredAxisScores(

        scans=tuple(scans),

        consensus_best_delay=consensus,

        mean_improvement=mean_imp,

        deferred_capture_at_best=deferred_capture,

        deferred_robustness=deferred_robust,

        best_entry_mfe=best_mfe,

        best_entry_mae=best_mae,

        best_entry_terminal=best_terminal,

        composite=composite,

        time_to_best_entry=consensus,

    )





def compute_dual_axis_scores(

    builder: FutureContextBuilder,

    *,

    t_index: int,

    horizon: int,

    action: Action,

    max_bar_index: int,

) -> tuple[ImmediateAxisScores, DeferredAxisScores, tuple[tuple[float, float], ...]]:

    ctx = builder.build(t_index)

    policies = build_policy_grid(ctx)

    immediate = _immediate_outcomes(truncate_reward_context(ctx, horizon), action, policies)

    deferred = _scan_deferred(

        builder,

        t_index=t_index,

        horizon=horizon,

        action=action,

        policies=policies,

        max_bar_index=max_bar_index,

    )

    return immediate, deferred, policies





def delay_captureability_curve(

    builder: FutureContextBuilder,

    *,

    t_index: int,

    horizon: int,

    action: Action,

    delays: tuple[int, ...],

    policies: tuple[tuple[float, float], ...],

    max_bar_index: int,

) -> dict[int, dict[str, float]]:

    """Captureability and mean proxy PnL by entry delay."""

    out: dict[int, dict[str, float]] = {}

    for delay in delays:
        ctx = build_delayed_context_from_anchor(
            builder,
            anchor_t=t_index,
            delay=delay,
            horizon=horizon,
            max_bar_index=max_bar_index,
        )
        if ctx is None:
            out[delay] = {"captureability": float("nan"), "mean_proxy_pnl": float("nan")}
            continue
        imm = _immediate_outcomes(ctx, action, policies)

        out[delay] = {

            "captureability": imm.captureability,

            "mean_proxy_pnl": imm.mean_proxy_pnl,

            "robustness": imm.robustness,

        }

    return out


