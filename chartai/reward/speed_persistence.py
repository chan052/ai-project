"""Speed (S) and Persistence (D) candidate formulations for P1 experiments.

These are *experimental* decompositions of Path — not canonical replacements.
All definitions are causal: only aligned returns R_k from t to t+k are used.
"""

from __future__ import annotations

from enum import Enum

from chartai.core.types import Action
from chartai.reward.base import directional_sign
from chartai.reward.context import RewardContext
from chartai.reward.path import normalized_decay_weights


class SpeedCandidate(str, Enum):
    """Speed component candidates — magnitude-free where noted."""

    EARLY_SIGN = "early_sign"
    TIME_TO_FAVORABLE = "time_to_favorable"
    EARLY_FAVORABLE_MASS = "early_favorable_mass"


class PersistenceCandidate(str, Enum):
    """Persistence / duration component candidates."""

    LATE_SIGN = "late_sign"
    FAVORABLE_OCCUPANCY = "favorable_occupancy"
    MAX_FAVORABLE_RUN = "max_favorable_run"
    LATE_FAVORABLE_MASS = "late_favorable_mass"


class SDPair(str, Enum):
    """Pre-defined S + D pairings for structure comparison."""

    EARLY_LATE_SIGN = "S_early_sign__D_late_sign"
    TTF_OCCUPANCY = "S_time_to_favorable__D_occupancy"
    TTF_RUN = "S_time_to_favorable__D_max_run"
    EARLY_SIGN_OCC = "S_early_sign__D_occupancy"


def _aligned_returns(ctx: RewardContext, action: Action, n: int) -> tuple[float, ...]:
    sign = directional_sign(action)
    return tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _early_decay_weights(num_steps: int, decay_rate: float) -> tuple[float, ...]:
    """Emphasize small k: w_k ∝ (1-r)^(k-1), normalized."""
    if num_steps <= 0:
        return ()
    raw = tuple((1.0 - decay_rate) ** (k - 1) for k in range(1, num_steps + 1))
    total = sum(raw)
    return tuple(w / total for w in raw)


def compute_speed_n(
    ctx: RewardContext,
    action: Action,
    n: int,
    candidate: SpeedCandidate,
    *,
    decay_rate: float = 0.75,
) -> float:
    """Speed score at horizon n — higher means faster favorable structure."""
    rets = _aligned_returns(ctx, action, n)
    signs = tuple(_sign(r) for r in rets)

    if candidate is SpeedCandidate.EARLY_SIGN:
        weights = _early_decay_weights(n, decay_rate)
        return sum(weights[k - 1] * signs[k - 1] for k in range(1, n + 1))

    if candidate is SpeedCandidate.TIME_TO_FAVORABLE:
        tau = next((k for k, r in enumerate(rets, start=1) if r > 0), None)
        if tau is None:
            return 0.0
        return (n + 1 - tau) / n

    if candidate is SpeedCandidate.EARLY_FAVORABLE_MASS:
        half = max(1, n // 2)
        weights = _early_decay_weights(half, decay_rate)
        favorable = tuple(1.0 if r > 0 else 0.0 for r in rets[:half])
        return sum(weights[k - 1] * favorable[k - 1] for k in range(1, half + 1))

    raise ValueError(candidate)


def compute_persistence_n(
    ctx: RewardContext,
    action: Action,
    n: int,
    candidate: PersistenceCandidate,
    *,
    decay_rate: float = 0.75,
) -> float:
    """Persistence score at horizon n — higher means longer favorable structure."""
    rets = _aligned_returns(ctx, action, n)
    signs = tuple(_sign(r) for r in rets)

    if candidate is PersistenceCandidate.LATE_SIGN:
        weights = normalized_decay_weights(n, decay_rate)
        return sum(weights[k - 1] * signs[k - 1] for k in range(1, n + 1))

    if candidate is PersistenceCandidate.FAVORABLE_OCCUPANCY:
        return sum(1 for r in rets if r > 0) / n

    if candidate is PersistenceCandidate.MAX_FAVORABLE_RUN:
        best = 0
        current = 0
        for r in rets:
            if r > 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best / n

    if candidate is PersistenceCandidate.LATE_FAVORABLE_MASS:
        start = n // 2
        late = rets[start:]
        if not late:
            return 0.0
        weights = normalized_decay_weights(len(late), decay_rate)
        favorable = tuple(1.0 if r > 0 else 0.0 for r in late)
        return sum(weights[k - 1] * favorable[k - 1] for k in range(1, len(late) + 1))

    raise ValueError(candidate)


def sd_pair_components(pair: SDPair) -> tuple[SpeedCandidate, PersistenceCandidate]:
    mapping = {
        SDPair.EARLY_LATE_SIGN: (SpeedCandidate.EARLY_SIGN, PersistenceCandidate.LATE_SIGN),
        SDPair.TTF_OCCUPANCY: (SpeedCandidate.TIME_TO_FAVORABLE, PersistenceCandidate.FAVORABLE_OCCUPANCY),
        SDPair.TTF_RUN: (SpeedCandidate.TIME_TO_FAVORABLE, PersistenceCandidate.MAX_FAVORABLE_RUN),
        SDPair.EARLY_SIGN_OCC: (SpeedCandidate.EARLY_SIGN, PersistenceCandidate.FAVORABLE_OCCUPANCY),
    }
    return mapping[pair]


def compute_sd_pair_n(
    ctx: RewardContext,
    action: Action,
    n: int,
    pair: SDPair,
    *,
    decay_rate: float = 0.75,
) -> tuple[float, float]:
    speed_c, persist_c = sd_pair_components(pair)
    s = compute_speed_n(ctx, action, n, speed_c, decay_rate=decay_rate)
    d = compute_persistence_n(ctx, action, n, persist_c, decay_rate=decay_rate)
    return s, d


CANDIDATE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "S_early_sign": "Early-weighted sign(aligned R_k); magnitude-free; emphasizes quick direction",
    "S_time_to_favorable": "Inverse time-to-first favorable step; pure timing",
    "S_early_favorable_mass": "Early-window favorable occupancy with early decay",
    "D_late_sign": "Late-weighted sign(aligned R_k); magnitude-free persistence",
    "D_favorable_occupancy": "Fraction of steps with aligned R_k > 0",
    "D_max_favorable_run": "Longest consecutive favorable run / n",
    "D_late_favorable_mass": "Late-window favorable occupancy with standard decay",
}
