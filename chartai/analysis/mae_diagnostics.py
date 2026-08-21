"""MAE decomposition observables for analysis-only reward audits.

These quantities use future path data as supervised labels — never as state features.
Not part of canonical reward; diagnostic and audit candidates only.
"""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n


@dataclass(frozen=True)
class MaeDiagnostics:
    """MAE-related future-path statistics at horizon ``n`` for one action."""

    n: int
    full_mae: float
    early_mae: float
    early_bars: int
    time_to_mae: int | None
    adverse_duration: int
    adverse_occupancy: float
    recovery_after_mae: float
    early_to_full_mae_ratio: float
    terminal_aligned_return: float


def _aligned_returns(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def compute_time_to_mae_n(ctx: RewardContext, action: Action, n: int) -> int | None:
    """First bar k (1..n) at which cumulative MAE reaches its value at horizon n."""
    if n < 1:
        return None
    target = compute_mae_n(ctx, action, n)
    if target <= 1e-15:
        return None
    for k in range(1, n + 1):
        if compute_mae_n(ctx, action, k) >= target - 1e-15:
            return k
    return n


def compute_adverse_duration_n(ctx: RewardContext, action: Action, n: int) -> int:
    """Count of steps with aligned close return < 0."""
    rets = _aligned_returns(ctx, action, n)
    return sum(1 for r in rets if r < 0)


def compute_recovery_after_mae_n(ctx: RewardContext, action: Action, n: int) -> float:
    """Recovery proxy after worst adverse: terminal aligned return relative to full MAE.

    Positive when terminal move offsets entry adverse (Case A-like).
    Uses only t+1..t+n — same causal window as other reward labels.
    """
    full = compute_mae_n(ctx, action, n)
    if full <= 1e-15:
        return 0.0
    rets = _aligned_returns(ctx, action, n)
    terminal = rets[-1]
    return terminal / full


def compute_mae_diagnostics(
    ctx: RewardContext,
    action: Action,
    n: int,
    *,
    early_bars: int = 3,
) -> MaeDiagnostics:
    """Full MAE decomposition bundle at horizon n."""
    early_bars = max(1, min(early_bars, n))
    full = compute_mae_n(ctx, action, n)
    early = compute_mae_n(ctx, action, early_bars)
    rets = _aligned_returns(ctx, action, n)
    adv = compute_adverse_duration_n(ctx, action, n)
    ratio = early / full if full > 1e-15 else 0.0
    return MaeDiagnostics(
        n=n,
        full_mae=full,
        early_mae=early,
        early_bars=early_bars,
        time_to_mae=compute_time_to_mae_n(ctx, action, n),
        adverse_duration=adv,
        adverse_occupancy=adv / n,
        recovery_after_mae=compute_recovery_after_mae_n(ctx, action, n),
        early_to_full_mae_ratio=ratio,
        terminal_aligned_return=rets[-1],
    )
