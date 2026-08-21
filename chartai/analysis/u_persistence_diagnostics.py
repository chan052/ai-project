"""U persistence / path-shape diagnostics for Reward Logic Audit 5 (analysis-only)."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.path_observables import compute_path_observables
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_speed_n,
)
from chartai.reward.utility import compute_utility_n, UtilityConfig


@dataclass(frozen=True)
class UDiagnostics:
    """Utility component bundle at horizon H for one action."""

    horizon: int
    u_profile: tuple[float, ...]  # U_1 .. U_H
    u_mean: float
    u_terminal: float
    terminal_return: float
    mfe: float
    favorable_occupancy: float
    max_favorable_run: int
    time_to_favorable: int | None
    speed_ttf: float
    persistence_occ: float
    path_obs_terminal: float


def _aligned_returns(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def compute_u_diagnostics(
    ctx: RewardContext,
    action: Action,
    *,
    horizon: int,
    utility_config: UtilityConfig,
    decay_rate: float = 0.75,
) -> UDiagnostics:
    profile = tuple(
        compute_utility_n(ctx, action, n, utility_config) for n in range(1, horizon + 1)
    )
    obs = compute_path_observables(ctx, action, horizon)
    return UDiagnostics(
        horizon=horizon,
        u_profile=profile,
        u_mean=float(sum(profile) / len(profile)) if profile else 0.0,
        u_terminal=profile[-1] if profile else 0.0,
        terminal_return=obs.terminal_return,
        mfe=obs.mfe,
        favorable_occupancy=obs.favorable_occupancy,
        max_favorable_run=obs.max_favorable_run,
        time_to_favorable=obs.time_to_favorable,
        speed_ttf=compute_speed_n(
            ctx, action, horizon, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=decay_rate
        ),
        persistence_occ=compute_persistence_n(
            ctx, action, horizon, PersistenceCandidate.FAVORABLE_OCCUPANCY, decay_rate=decay_rate
        ),
        path_obs_terminal=obs.terminal_return,
    )


def u_profile_to_dict(d: UDiagnostics) -> dict[str, float | int | None | list[float]]:
    return {
        "u_mean": d.u_mean,
        "u_terminal": d.u_terminal,
        "u_profile": list(d.u_profile),
        "terminal_return": d.terminal_return,
        "mfe": d.mfe,
        "favorable_occupancy": d.favorable_occupancy,
        "max_favorable_run": d.max_favorable_run,
        "time_to_favorable": d.time_to_favorable,
        "speed_ttf": d.speed_ttf,
        "persistence_occ": d.persistence_occ,
    }


def persistence_information_level(
    *,
    occupancy_corr: float,
    run_corr: float,
    controlled_discriminates: bool,
) -> str:
    """Q1/Q2 coarse classification — not adoption criterion."""
    if controlled_discriminates and max(abs(occupancy_corr), abs(run_corr)) > 0.5:
        return "included_and_partially_preserves"
    if controlled_discriminates or max(abs(occupancy_corr), abs(run_corr)) > 0.4:
        return "included_but_coarse"
    if max(abs(occupancy_corr), abs(run_corr)) > 0.25:
        return "weak_persistence_signal"
    return "persistence_insufficient"
