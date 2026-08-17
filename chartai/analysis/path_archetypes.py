"""Extended path observables for opportunity analysis."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.context import RewardContext
from chartai.reward.path_observables import PathObservables, _aligned_returns, compute_path_observables


@dataclass(frozen=True)
class ExtendedPathObservables:
    base: PathObservables
    max_adverse_run: int
    path_sign_changes: int
    monotonicity_score: float
    early_min_return: float
    early_max_return: float


def _max_run(returns: tuple[float, ...], *, favorable: bool) -> int:
    best = current = 0
    for r in returns:
        hit = (r > 0) if favorable else (r < 0)
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _sign_changes(returns: tuple[float, ...]) -> int:
    signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in returns]
    changes = 0
    prev = 0
    for s in signs:
        if s != 0 and prev != 0 and s != prev:
            changes += 1
        if s != 0:
            prev = s
    return changes


def compute_extended_observables(ctx: RewardContext, action: Action, n: int) -> ExtendedPathObservables:
    base = compute_path_observables(ctx, action, n)
    rets = _aligned_returns(ctx, action, n)
    half = max(1, n // 2)
    early = rets[:half]
    mono = sum(1 for r in rets if r > 0) / n if rets else 0.0
    return ExtendedPathObservables(
        base=base,
        max_adverse_run=_max_run(rets, favorable=False),
        path_sign_changes=_sign_changes(rets),
        monotonicity_score=mono,
        early_min_return=min(early) if early else 0.0,
        early_max_return=max(early) if early else 0.0,
    )


def classify_archetype(ext: ExtendedPathObservables, *, early_thr: float = 0.0003, flat_thr: float = 0.0002) -> str:
    b = ext.base
    if b.mfe < flat_thr and b.mae < flat_thr:
        return "flat_no_opportunity"
    if b.early_mean_return < -early_thr and b.terminal_return > 0:
        return "dip_then_rise"
    if b.early_mean_return > early_thr and b.terminal_return < 0:
        return "rise_then_fall"
    if b.terminal_return > early_thr and ext.monotonicity_score >= 0.8:
        return "monotonic_rise"
    if b.terminal_return < -early_thr and ext.monotonicity_score <= 0.2:
        return "monotonic_fall"
    if ext.early_max_return > early_thr and b.terminal_return < 0:
        return "spike_reversal"
    if ext.path_sign_changes >= 4:
        return "choppy"
    if abs(b.terminal_return) < flat_thr:
        return "slow_flat"
    return "mixed"
