"""Path residual observables — structure beyond U/MAE (analysis-only)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_mfe_n, time_to_mfe_n


def _aligned_cumulative(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def _aligned_bar_returns(cum: tuple[float, ...]) -> tuple[float, ...]:
    if not cum:
        return ()
    out = [cum[0]]
    for i in range(1, len(cum)):
        out.append(cum[i] - cum[i - 1])
    return tuple(out)


def _sign_changes(bar_rets: tuple[float, ...]) -> int:
    changes = 0
    prev = 0
    for r in bar_rets:
        s = 1 if r > 0 else (-1 if r < 0 else 0)
        if s != 0 and prev != 0 and s != prev:
            changes += 1
        if s != 0:
            prev = s
    return changes


def _peak_index(cum: tuple[float, ...]) -> int:
    if not cum:
        return 0
    best_v = max(cum)
    return next(i for i, v in enumerate(cum) if v >= best_v - 1e-15)


@dataclass(frozen=True)
class PathResidualObservables:
    """Candidate path-structure metrics at horizon n."""

    n: int
    giveback_ratio: float
    reversal_depth: float
    excursion_stability: float
    peak_timing: float
    peak_after_decay: float
    recovery_shape_score: float
    oscillation_chop: float
    mfe_terminal_ratio: float
    mae_terminal_ratio: float
    path_efficiency: float
    terminal_proximity_mfe: float
    transition_count: int
    time_near_mfe: float
    drawdown_from_mfe: float
    directional_consistency: float
    favorable_adverse_alternation: float
    post_mae_recovery_high: float
    time_under_water: float
    time_under_favorable: float
    recovery_speed: float
    drawdown_duration: float
    path_sign_entropy: float
    excursion_concentration: float
    excursion_volatility: float
    terminal_return: float
    mfe: float
    mae: float


@dataclass(frozen=True)
class ResidualCandidateSpec:
    key: str
    label: str
    category: str
    extract: str


CANDIDATE_SPECS: tuple[ResidualCandidateSpec, ...] = (
    ResidualCandidateSpec("giveback_ratio", "A Giveback", "core", "giveback_ratio"),
    ResidualCandidateSpec("reversal_depth", "B Reversal Depth", "core", "reversal_depth"),
    ResidualCandidateSpec("excursion_stability", "C Excursion Stability", "core", "excursion_stability"),
    ResidualCandidateSpec("peak_timing", "D Peak Timing", "core", "peak_timing"),
    ResidualCandidateSpec("peak_after_decay", "E Peak-after Decay", "core", "peak_after_decay"),
    ResidualCandidateSpec("recovery_shape_score", "F Recovery Shape", "core", "recovery_shape_score"),
    ResidualCandidateSpec("oscillation_chop", "G Oscillation/Chop", "core", "oscillation_chop"),
    ResidualCandidateSpec("mfe_terminal_ratio", "MFE/terminal ratio", "extra", "mfe_terminal_ratio"),
    ResidualCandidateSpec("path_efficiency", "Path efficiency", "extra", "path_efficiency"),
    ResidualCandidateSpec(
        "terminal_proximity_mfe", "Terminal proximity to MFE", "extra", "terminal_proximity_mfe"
    ),
    ResidualCandidateSpec("transition_count", "Favorable/adverse transitions", "extra", "transition_count"),
    ResidualCandidateSpec("time_near_mfe", "Time spent near MFE", "extra", "time_near_mfe"),
    ResidualCandidateSpec("drawdown_from_mfe", "Max drawdown from MFE", "extra", "drawdown_from_mfe"),
    ResidualCandidateSpec(
        "post_mae_recovery_high", "Post-MAE new favorable high", "extra", "post_mae_recovery_high"
    ),
    ResidualCandidateSpec("time_under_water", "Time under water", "extended", "time_under_water"),
    ResidualCandidateSpec("time_under_favorable", "Time under favorable", "extended", "time_under_favorable"),
    ResidualCandidateSpec("recovery_speed", "Recovery speed", "extended", "recovery_speed"),
    ResidualCandidateSpec("drawdown_duration", "Drawdown duration", "extended", "drawdown_duration"),
    ResidualCandidateSpec("path_sign_entropy", "Path sign entropy", "extended", "path_sign_entropy"),
    ResidualCandidateSpec(
        "excursion_concentration", "Excursion concentration", "extended", "excursion_concentration"
    ),
    ResidualCandidateSpec(
        "excursion_volatility", "Excursion volatility (favorable path std)", "instability", "excursion_volatility"
    ),
)


def compute_path_residual_observables(
    ctx: RewardContext,
    action: Action,
    n: int,
) -> PathResidualObservables:
    cum = _aligned_cumulative(ctx, action, n)
    bar = _aligned_bar_returns(cum)
    mfe = compute_mfe_n(ctx, action, n)
    mae = compute_mae_n(ctx, action, n)
    terminal = cum[-1] if cum else 0.0

    peak_cum = max(cum) if cum else 0.0
    peak_idx = _peak_index(cum)
    mfe_ref = max(mfe, peak_cum, 1e-12)

    giveback = max(0.0, min(1.0, (mfe_ref - terminal) / mfe_ref)) if mfe_ref > 1e-12 else 0.0

    post_peak = cum[peak_idx:] if cum else ()
    reversal = 0.0
    if post_peak and mfe_ref > 1e-12:
        min_after = min(post_peak)
        reversal = max(0.0, min(1.0, (mfe_ref - min_after) / mfe_ref))

    peak_after = 0.0
    if post_peak and len(post_peak) > 1 and mfe_ref > 1e-12:
        decays = [max(0.0, min(1.0, (mfe_ref - v) / mfe_ref)) for v in post_peak[1:]]
        peak_after = sum(decays) / len(decays)

    exc_stab = 0.0
    fav_start = next((i for i, v in enumerate(cum) if v > 0), None)
    if fav_start is not None:
        seg = cum[fav_start:]
        if len(seg) > 1:
            mu = sum(seg) / len(seg)
            var = sum((x - mu) ** 2 for x in seg) / len(seg)
            exc_stab = 1.0 / (1.0 + math.sqrt(var) * 100)

    t_mfe = time_to_mfe_n(ctx, action, n)
    peak_timing = float(t_mfe or (n + 1)) / n

    recovery_shape = 0.0
    if mae > 1e-5:
        recovery_shape = max(-5.0, min(5.0, terminal / mae))

    chop = float(_sign_changes(bar)) / max(len(bar) - 1, 1)
    path_vol = sum(abs(r) for r in bar)
    efficiency = max(-1.0, min(1.0, terminal / path_vol)) if path_vol > 1e-12 else 0.0
    prox_mfe = max(0.0, min(2.0, terminal / mfe)) if mfe > 1e-5 else 0.0

    transitions = _sign_changes(bar)
    near_mfe = 0.0
    if mfe_ref > 1e-12 and cum:
        thresh = 0.9 * mfe_ref
        near_mfe = sum(1 for v in cum if v >= thresh) / n

    max_dd = 0.0
    if mfe_ref > 1e-12 and cum:
        running_peak = cum[0]
        for v in cum:
            running_peak = max(running_peak, v)
            if running_peak > 1e-12:
                max_dd = max(max_dd, max(0.0, min(1.0, (running_peak - v) / mfe_ref)))

    post_mae_rec = 0.0
    if mae > 1e-12 and cum:
        mae_k = 1
        for k in range(1, n + 1):
            if compute_mae_n(ctx, action, k) >= mae - 1e-15:
                mae_k = k
                break
        if mae_k < n:
            post_max = max(cum[mae_k:])
            post_mae_rec = post_max / mae

    alt = transitions / max(n - 1, 1)
    consistency = 1.0 - chop

    time_under_water = sum(1 for v in cum if v < 0) / n if cum else 0.0
    time_under_fav = sum(1 for v in cum if v > 0) / n if cum else 0.0

    recovery_speed = 0.0
    if mae > 1e-5 and cum:
        mae_k = 1
        for k in range(1, n + 1):
            if compute_mae_n(ctx, action, k) >= mae - 1e-15:
                mae_k = k
                break
        recover_k = None
        for j in range(mae_k, n):
            if cum[j] >= -1e-12:
                recover_k = j - mae_k + 1
                break
        if recover_k is not None:
            recovery_speed = 1.0 - (recover_k / max(n - mae_k, 1))

    dd_duration = 0.0
    if cum:
        running_peak = cum[0]
        cur_run = 0
        max_run = 0
        for v in cum:
            running_peak = max(running_peak, v)
            if v < running_peak - 1e-12:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        dd_duration = max_run / n

    pos = sum(1 for r in bar if r > 0)
    neg = sum(1 for r in bar if r < 0)
    flat = len(bar) - pos - neg
    total = len(bar) or 1
    probs = [c / total for c in (pos, neg, flat) if c > 0]
    entropy = -sum(p * math.log(p + 1e-15) for p in probs)
    path_entropy = entropy / math.log(3) if len(probs) > 1 else 0.0

    exc_conc = 0.0
    if mfe_ref > 1e-12 and bar:
        peak_bar = max(range(len(bar)), key=lambda i: cum[i])
        bar_up = max(bar[peak_bar], 0.0)
        total_up = sum(max(r, 0.0) for r in bar)
        exc_conc = bar_up / total_up if total_up > 1e-12 else 0.0

    exc_vol = 0.0
    if fav_start is not None:
        fav_bars = bar[fav_start:]
        if len(fav_bars) > 1:
            mu_b = sum(fav_bars) / len(fav_bars)
            exc_vol = math.sqrt(sum((x - mu_b) ** 2 for x in fav_bars) / len(fav_bars))

    mae_term = max(0.0, min(10.0, mae / max(abs(terminal), 1e-5))) if abs(terminal) > 1e-5 else 0.0
    mfe_term = max(0.0, min(10.0, mfe / max(abs(terminal), 1e-5))) if abs(terminal) > 1e-5 else 0.0

    return PathResidualObservables(
        n=n,
        giveback_ratio=giveback,
        reversal_depth=reversal,
        excursion_stability=exc_stab,
        peak_timing=peak_timing,
        peak_after_decay=peak_after,
        recovery_shape_score=recovery_shape,
        oscillation_chop=chop,
        mfe_terminal_ratio=mfe_term,
        mae_terminal_ratio=mae_term,
        path_efficiency=efficiency,
        terminal_proximity_mfe=prox_mfe,
        transition_count=transitions,
        time_near_mfe=near_mfe,
        drawdown_from_mfe=max_dd,
        directional_consistency=consistency,
        favorable_adverse_alternation=alt,
        post_mae_recovery_high=post_mae_rec,
        time_under_water=time_under_water,
        time_under_favorable=time_under_fav,
        recovery_speed=recovery_speed,
        drawdown_duration=dd_duration,
        path_sign_entropy=path_entropy,
        excursion_concentration=exc_conc,
        excursion_volatility=exc_vol,
        terminal_return=terminal,
        mfe=mfe,
        mae=mae,
    )


def get_candidate_value(obs: PathResidualObservables, spec: ResidualCandidateSpec) -> float:
    return float(getattr(obs, spec.extract))


def observables_to_dict(obs: PathResidualObservables) -> dict[str, float]:
    return {spec.key: get_candidate_value(obs, spec) for spec in CANDIDATE_SPECS}
