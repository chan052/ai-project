"""Path-level risk-adjusted metrics — Sharpe, Sortino, Ulcer (analysis-only)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_mfe_n


def _aligned_bar_returns(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    cum: list[float] = []
    out: list[float] = []
    for k in range(1, n + 1):
        c = sign * ctx.return_from_t(k)
        if not cum:
            out.append(c)
        else:
            out.append(c - cum[-1])
        cum.append(c)
    return tuple(out)


def _aligned_cumulative(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def _safe_std(values: tuple[float, ...], *, ddof: int = 0) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / max(len(values) - ddof, 1)
    return math.sqrt(max(var, 0.0))


def _downside_std(values: tuple[float, ...], mar: float = 0.0) -> float:
    downs = tuple(min(0.0, x - mar) for x in values)
    if not downs:
        return 0.0
    sq = sum(d * d for d in downs) / len(downs)
    return math.sqrt(sq)


def _wealth_from_cumulative(cum: tuple[float, ...]) -> tuple[float, ...]:
    w = [1.0]
    for c in cum:
        w.append(w[-1] * (1.0 + c))
    return tuple(w[1:])


def compute_ulcer_index(cum: tuple[float, ...]) -> float:
    """Ulcer Index from cumulative return path (analysis label)."""
    if not cum:
        return 0.0
    wealth = _wealth_from_cumulative(cum)
    peak = wealth[0]
    sq_dd: list[float] = []
    for w in wealth:
        peak = max(peak, w)
        if peak > 1e-15:
            dd = (peak - w) / peak
            sq_dd.append(dd * dd)
    if not sq_dd:
        return 0.0
    return math.sqrt(sum(sq_dd) / len(sq_dd))


def compute_max_drawdown_abs(cum: tuple[float, ...]) -> float:
    if not cum:
        return 0.0
    wealth = _wealth_from_cumulative(cum)
    peak = wealth[0]
    max_dd = 0.0
    for w in wealth:
        peak = max(peak, w)
        if peak > 1e-15:
            max_dd = max(max_dd, (peak - w) / peak)
    return max_dd


def compute_path_sharpe(bar: tuple[float, ...], *, eps: float = 1e-12) -> float:
    if not bar:
        return 0.0
    mu = sum(bar) / len(bar)
    sigma = _safe_std(bar)
    if sigma < eps:
        return 0.0 if abs(mu) < eps else math.copysign(1e4, mu)
    return mu / sigma


def compute_path_sortino(bar: tuple[float, ...], *, mar: float = 0.0, eps: float = 1e-12) -> float:
    if not bar:
        return 0.0
    mu = sum(bar) / len(bar)
    ds = _downside_std(bar, mar=mar)
    if ds < eps:
        return 0.0 if abs(mu) < eps else math.copysign(1e4, mu)
    return (mu - mar) / ds


@dataclass(frozen=True)
class RiskAdjustedPathMetrics:
    """Path-level risk-adjusted observables at horizon n."""

    n: int
    path_sharpe: float
    path_sortino: float
    ulcer_index: float
    calmar_proxy: float
    return_over_ulcer: float
    terminal_return: float
    mean_bar_return: float
    bar_volatility: float
    downside_deviation: float
    max_drawdown: float
    mfe: float
    mae: float
    differential_sharpe_vs_zero: float
    differential_sortino_vs_zero: float


@dataclass(frozen=True)
class DifferentialSharpePair:
    """Sharpe gap between actions on the same future market path."""

    t_index: int
    sharpe_long: float
    sharpe_short: float
    sharpe_hold: float
    diff_long_minus_hold: float
    diff_long_minus_short: float
    diff_short_minus_hold: float


RISK_ADJUSTED_SPECS: tuple[tuple[str, str], ...] = (
    ("path_sharpe", "Path Sharpe (mean/vol of bar returns)"),
    ("path_sortino", "Path Sortino (mean/downside dev)"),
    ("ulcer_index", "Ulcer Index (RMS drawdown from wealth peak)"),
    ("calmar_proxy", "Calmar proxy (terminal / max_drawdown)"),
    ("return_over_ulcer", "Return / Ulcer (Martin-like ratio)"),
    ("differential_sharpe_vs_zero", "Sharpe on path (flat baseline = 0)"),
    ("differential_sortino_vs_zero", "Sortino on path (flat baseline = 0)"),
)


def compute_risk_adjusted_path_metrics(
    ctx: RewardContext,
    action: Action,
    n: int,
) -> RiskAdjustedPathMetrics:
    bar = _aligned_bar_returns(ctx, action, n)
    cum = _aligned_cumulative(ctx, action, n)
    terminal = cum[-1] if cum else 0.0
    mu = sum(bar) / len(bar) if bar else 0.0
    sigma = _safe_std(bar)
    ds = _downside_std(bar)
    ulcer = compute_ulcer_index(cum)
    max_dd = compute_max_drawdown_abs(cum)
    sharpe = compute_path_sharpe(bar)
    sortino = compute_path_sortino(bar)
    mfe = compute_mfe_n(ctx, action, n)
    mae = compute_mae_n(ctx, action, n)

    if max_dd > 1e-8:
        calmar = terminal / max_dd
    elif abs(terminal) < 1e-12:
        calmar = 0.0
    else:
        calmar = math.copysign(1e4, terminal)

    if ulcer > 1e-8:
        rou = terminal / ulcer
    elif abs(terminal) < 1e-12:
        rou = 0.0
    else:
        rou = math.copysign(1e4, terminal)

    return RiskAdjustedPathMetrics(
        n=n,
        path_sharpe=sharpe,
        path_sortino=sortino,
        ulcer_index=ulcer,
        calmar_proxy=max(-1e4, min(1e4, calmar)),
        return_over_ulcer=max(-1e4, min(1e4, rou)),
        terminal_return=terminal,
        mean_bar_return=mu,
        bar_volatility=sigma,
        downside_deviation=ds,
        max_drawdown=max_dd,
        mfe=mfe,
        mae=mae,
        differential_sharpe_vs_zero=sharpe,
        differential_sortino_vs_zero=sortino,
    )


def compute_differential_sharpe_pair(ctx: RewardContext, n: int) -> DifferentialSharpePair:
    sl = compute_path_sharpe(_aligned_bar_returns(ctx, Action.LONG, n))
    ss = compute_path_sharpe(_aligned_bar_returns(ctx, Action.SHORT, n))
    sh = 0.0
    return DifferentialSharpePair(
        t_index=ctx.t_index,
        sharpe_long=sl,
        sharpe_short=ss,
        sharpe_hold=sh,
        diff_long_minus_hold=sl - sh,
        diff_long_minus_short=sl - ss,
        diff_short_minus_hold=ss - sh,
    )


def risk_adjusted_to_dict(m: RiskAdjustedPathMetrics) -> dict[str, float]:
    return {
        "path_sharpe": m.path_sharpe,
        "path_sortino": m.path_sortino,
        "ulcer_index": m.ulcer_index,
        "calmar_proxy": m.calmar_proxy,
        "return_over_ulcer": m.return_over_ulcer,
        "differential_sharpe_vs_zero": m.differential_sharpe_vs_zero,
        "differential_sortino_vs_zero": m.differential_sortino_vs_zero,
        "terminal_return": m.terminal_return,
        "mean_bar_return": m.mean_bar_return,
        "bar_volatility": m.bar_volatility,
        "downside_deviation": m.downside_deviation,
        "max_drawdown": m.max_drawdown,
    }
