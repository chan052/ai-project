"""Future path behavior observables for MTF conditional audit (analysis-only)."""

from __future__ import annotations

from dataclasses import dataclass

from chartai.core.types import Action
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_path_observables
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_speed_n,
)


@dataclass(frozen=True)
class FutureBehaviorObservables:
    horizon: int
    action: Action
    terminal_return: float
    mfe: float
    mae: float
    time_to_favorable: int | None
    time_to_adverse: int | None
    early_favorable_occupancy: float
    late_favorable_occupancy: float
    favorable_occupancy: float
    max_favorable_run: int
    reversal: bool
    dip_then_rise: bool
    speed_ttf: float
    persistence_occ: float


def compute_future_behavior(
    ctx: RewardContext,
    action: Action,
    horizon: int,
    *,
    decay_rate: float = 0.75,
) -> FutureBehaviorObservables:
    if horizon < 1 or horizon > ctx.reward_horizon:
        raise ValueError(f"horizon must be in 1..{ctx.reward_horizon}")
    obs = compute_path_observables(ctx, action, horizon)
    half = max(1, horizon // 2)
    sign = 1.0 if action is Action.LONG else -1.0
    rets = tuple(sign * ctx.return_from_t(k) for k in range(1, horizon + 1))
    early = rets[:half]
    late = rets[half:]
    early_fav = sum(1 for r in early if r > 0) / len(early) if early else 0.0
    late_fav = sum(1 for r in late if r > 0) / len(late) if late else 0.0
    reversal = (
        obs.early_mean_return > 0.0002 and obs.terminal_return < -0.0002
    ) or (obs.early_mean_return < -0.0002 and obs.terminal_return > 0.0002)
    dip_rise = obs.early_mean_return < -0.0002 and obs.terminal_return > 0.0002
    s = compute_speed_n(
        ctx, action, horizon, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=decay_rate
    )
    d = compute_persistence_n(
        ctx, action, horizon, PersistenceCandidate.FAVORABLE_OCCUPANCY, decay_rate=decay_rate
    )
    return FutureBehaviorObservables(
        horizon=horizon,
        action=action,
        terminal_return=obs.terminal_return,
        mfe=obs.mfe,
        mae=compute_mae_n(ctx, action, horizon),
        time_to_favorable=obs.time_to_favorable,
        time_to_adverse=obs.time_to_adverse,
        early_favorable_occupancy=early_fav,
        late_favorable_occupancy=late_fav,
        favorable_occupancy=obs.favorable_occupancy,
        max_favorable_run=obs.max_favorable_run,
        reversal=reversal,
        dip_then_rise=dip_rise,
        speed_ttf=s,
        persistence_occ=d,
    )


def behavior_to_dict(fb: FutureBehaviorObservables) -> dict[str, float | int | bool | None]:
    return {
        "terminal_return": fb.terminal_return,
        "mfe": fb.mfe,
        "mae": fb.mae,
        "time_to_favorable": fb.time_to_favorable,
        "time_to_adverse": fb.time_to_adverse,
        "early_favorable_occupancy": fb.early_favorable_occupancy,
        "late_favorable_occupancy": fb.late_favorable_occupancy,
        "favorable_occupancy": fb.favorable_occupancy,
        "max_favorable_run": fb.max_favorable_run,
        "reversal": fb.reversal,
        "dip_then_rise": fb.dip_then_rise,
        "speed_ttf": fb.speed_ttf,
        "persistence_occ": fb.persistence_occ,
    }
