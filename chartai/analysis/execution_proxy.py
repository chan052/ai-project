"""Heuristic execution proxies for opportunity analysis — NOT canonical P2.

These policies explore which future-path characteristics correlate with
*achievable* outcomes under simple rules. Results are observables, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext


@dataclass(frozen=True)
class TargetStopResult:
    """Outcome of a target/stop first-hit simulation from entry at t."""

    target_pct: float
    stop_pct: float
    target_hit: bool
    stop_hit: bool
    target_first: bool | None
    time_to_target: int | None
    time_to_stop: int | None
    proxy_pnl: float
    bars_held: int


@dataclass(frozen=True)
class FixedHorizonResult:
    horizon: int
    proxy_pnl: float


@dataclass(frozen=True)
class ThresholdTiming:
    threshold_pct: float
    first_time: int | None
    reached: bool
    bars_above_after_first: int


def _long_excursions_to_k(ctx: RewardContext, k: int) -> tuple[float, float]:
    anchor = ctx.price_at_t
    fav = (max(ctx.future_highs[:k]) - anchor) / anchor
    adv = (anchor - min(ctx.future_lows[:k])) / anchor
    return fav, adv


def _short_excursions_to_k(ctx: RewardContext, k: int) -> tuple[float, float]:
    anchor = ctx.price_at_t
    fav = (anchor - min(ctx.future_lows[:k])) / anchor
    adv = (max(ctx.future_highs[:k]) - anchor) / anchor
    return fav, adv


def simulate_target_stop(
    ctx: RewardContext,
    action: Action,
    *,
    target_pct: float,
    stop_pct: float,
) -> TargetStopResult:
    """Bar-by-bar first-hit target/stop using cumulative H/L from t."""
    n = ctx.reward_horizon
    time_target: int | None = None
    time_stop: int | None = None

    for k in range(1, n + 1):
        if action is Action.LONG:
            fav, adv = _long_excursions_to_k(ctx, k)
        else:
            fav, adv = _short_excursions_to_k(ctx, k)

        if time_target is None and fav >= target_pct:
            time_target = k
        if time_stop is None and adv >= stop_pct:
            time_stop = k
        if time_target is not None and time_stop is not None:
            break

    target_hit = time_target is not None
    stop_hit = time_stop is not None
    if target_hit and stop_hit:
        target_first = time_target < time_stop
        if target_first:
            proxy_pnl = target_pct
            bars_held = time_target
        else:
            proxy_pnl = -stop_pct
            bars_held = time_stop
    elif target_hit:
        target_first = True
        proxy_pnl = target_pct
        bars_held = time_target or n
    elif stop_hit:
        target_first = False
        proxy_pnl = -stop_pct
        bars_held = time_stop or n
    else:
        target_first = None
        sign = directional_sign(action)
        proxy_pnl = sign * ctx.return_from_t(n)
        bars_held = n

    return TargetStopResult(
        target_pct=target_pct,
        stop_pct=stop_pct,
        target_hit=target_hit,
        stop_hit=stop_hit,
        target_first=target_first,
        time_to_target=time_target,
        time_to_stop=time_stop,
        proxy_pnl=proxy_pnl,
        bars_held=bars_held,
    )


def simulate_fixed_horizon(ctx: RewardContext, action: Action, *, horizon: int | None = None) -> FixedHorizonResult:
    h = horizon or ctx.reward_horizon
    sign = directional_sign(action)
    return FixedHorizonResult(
        horizon=h,
        proxy_pnl=sign * ctx.return_from_t(h),
    )


def threshold_timing(
    ctx: RewardContext,
    action: Action,
    *,
    threshold_pct: float,
    n: int | None = None,
) -> ThresholdTiming:
    """First bar k where favorable excursion >= threshold; bars maintained after."""
    steps = n or ctx.reward_horizon
    first: int | None = None
    for k in range(1, steps + 1):
        if action is Action.LONG:
            fav, _ = _long_excursions_to_k(ctx, k)
        else:
            fav, _ = _short_excursions_to_k(ctx, k)
        if fav >= threshold_pct:
            first = k
            break

    if first is None:
        return ThresholdTiming(threshold_pct, None, False, 0)

    bars_above = 0
    for k in range(first, steps + 1):
        if action is Action.LONG:
            fav, _ = _long_excursions_to_k(ctx, k)
        else:
            fav, _ = _short_excursions_to_k(ctx, k)
        if fav >= threshold_pct:
            bars_above += 1

    return ThresholdTiming(threshold_pct, first, True, bars_above)


def target_before_stop(ctx: RewardContext, action: Action, target_pct: float, stop_pct: float) -> bool | None:
    r = simulate_target_stop(ctx, action, target_pct=target_pct, stop_pct=stop_pct)
    if not r.target_hit and not r.stop_hit:
        return None
    if r.target_hit and not r.stop_hit:
        return True
    if r.stop_hit and not r.target_hit:
        return False
    return r.target_first


def policy_robustness_score(pnls: list[float]) -> float:
    """Fraction of proxy policies with positive PnL for one sample."""
    if not pnls:
        return float("nan")
    return sum(1 for p in pnls if p > 0) / len(pnls)
