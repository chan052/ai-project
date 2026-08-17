"""Future-path observables for P1 opportunity assessment analysis.

These quantities use future data as supervised *labels / analysis targets* only.
They must never appear in state features or causal normalization inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n


@dataclass(frozen=True)
class PathObservables:
    """Future-path statistics at horizon ``n`` for one action."""

    n: int
    mfe: float
    mae: float
    terminal_return: float
    time_to_mfe: int | None
    time_to_favorable: int | None
    time_to_adverse: int | None
    favorable_duration: int
    adverse_duration: int
    favorable_occupancy: float
    max_favorable_run: int
    early_mean_return: float
    late_mean_return: float


def _aligned_returns(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def compute_mfe_n(ctx: RewardContext, action: Action, n: int) -> float:
    """Maximum favorable excursion from t through t+n (positive magnitude)."""
    if n < 1 or n > ctx.reward_horizon:
        raise ValueError(f"n must be in 1..{ctx.reward_horizon}, got {n}")
    anchor = ctx.price_at_t
    if action is Action.LONG:
        return (max(ctx.future_highs[:n]) - anchor) / anchor
    if action is Action.SHORT:
        return (anchor - min(ctx.future_lows[:n])) / anchor
    raise ValueError(f"MFE requires LONG or SHORT, got {action!r}")


def time_to_mfe_n(ctx: RewardContext, action: Action, n: int) -> int | None:
    """First bar index k (1..n) at which running MFE is attained."""
    if action is Action.LONG:
        running_best = float("-inf")
        for k in range(1, n + 1):
            excursion = (max(ctx.future_highs[:k]) - ctx.price_at_t) / ctx.price_at_t
            if excursion > running_best + 1e-15:
                running_best = excursion
            if abs(excursion - (max(ctx.future_highs[:n]) - ctx.price_at_t) / ctx.price_at_t) < 1e-15:
                return k
        return n
    if action is Action.SHORT:
        running_best = float("-inf")
        for k in range(1, n + 1):
            excursion = (ctx.price_at_t - min(ctx.future_lows[:k])) / ctx.price_at_t
            if excursion > running_best + 1e-15:
                running_best = excursion
            if abs(excursion - (ctx.price_at_t - min(ctx.future_lows[:n])) / ctx.price_at_t) < 1e-15:
                return k
        return n
    raise ValueError(action)


def _first_index_where(returns: tuple[float, ...], predicate) -> int | None:
    for k, value in enumerate(returns, start=1):
        if predicate(value):
            return k
    return None


def _max_consecutive_run(returns: tuple[float, ...], *, favorable: bool) -> int:
    best = 0
    current = 0
    for value in returns:
        hit = value > 0 if favorable else value < 0
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_path_observables(ctx: RewardContext, action: Action, n: int) -> PathObservables:
    """Compute opportunity-related future-path observables at horizon n."""
    rets = _aligned_returns(ctx, action, n)
    fav = sum(1 for r in rets if r > 0)
    adv = sum(1 for r in rets if r < 0)
    half = max(1, n // 2)
    early = rets[:half]
    late = rets[half:]
    return PathObservables(
        n=n,
        mfe=compute_mfe_n(ctx, action, n),
        mae=compute_mae_n(ctx, action, n),
        terminal_return=rets[-1],
        time_to_mfe=time_to_mfe_n(ctx, action, n),
        time_to_favorable=_first_index_where(rets, lambda r: r > 0),
        time_to_adverse=_first_index_where(rets, lambda r: r < 0),
        favorable_duration=fav,
        adverse_duration=adv,
        favorable_occupancy=fav / n,
        max_favorable_run=_max_consecutive_run(rets, favorable=True),
        early_mean_return=float(sum(early) / len(early)),
        late_mean_return=float(sum(late) / len(late)) if late else 0.0,
    )
