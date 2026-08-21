"""MAE recovery / early-path diagnostics for Reward Logic Audit 5 (analysis-only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chartai.analysis.mae_diagnostics import MaeDiagnostics, compute_mae_diagnostics
from chartai.core.types import Action
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n


class MaeCase(str, Enum):
    EARLY_ADVERS_RECOVERY = "early_adverse_recovery"
    EARLY_ADVERS_SUSTAINED = "early_adverse_sustained"
    LATE_ADVERS = "late_adverse"
    SMALL_THEN_LARGE_ADVERS = "small_then_large_adverse"
    LARGE_THEN_RECOVERY = "large_then_recovery"


@dataclass(frozen=True)
class MaeCaseDiagnostics:
    case: MaeCase
    full_mae: float
    early_mae: float
    time_to_mae: int | None
    adverse_duration: int
    recovery_after_mae: float
    terminal_return: float
    mae_profile: tuple[float, ...]  # MAE_1 .. MAE_H


def compute_mae_profile(ctx: RewardContext, action: Action, horizon: int) -> tuple[float, ...]:
    return tuple(compute_mae_n(ctx, action, n) for n in range(1, horizon + 1))


def compute_mae_case_diagnostics(
    ctx: RewardContext,
    action: Action,
    case: MaeCase,
    *,
    horizon: int = 10,
    early_bars: int = 3,
) -> MaeCaseDiagnostics:
    d = compute_mae_diagnostics(ctx, action, horizon, early_bars=early_bars)
    return MaeCaseDiagnostics(
        case=case,
        full_mae=d.full_mae,
        early_mae=d.early_mae,
        time_to_mae=d.time_to_mae,
        adverse_duration=d.adverse_duration,
        recovery_after_mae=d.recovery_after_mae,
        terminal_return=d.terminal_aligned_return,
        mae_profile=compute_mae_profile(ctx, action, horizon),
    )


def mae_blind_spot_level(
    *,
    same_mae_different_recovery_pairs: int,
    recovery_corr_with_full_mae: float,
) -> str:
    if same_mae_different_recovery_pairs > 0 and abs(recovery_corr_with_full_mae) < 0.3:
        return "recovery_blind_spot_confirmed"
    if same_mae_different_recovery_pairs > 0:
        return "partial_recovery_blind_spot"
    return "unresolved"


def early_info_level(
    *,
    early_full_corr: float,
    time_to_mae_discriminates: bool,
) -> str:
    """Whether MAE structure already carries early adverse timing/magnitude."""
    if early_full_corr > 0.85 and time_to_mae_discriminates:
        return "early_info_present_via_horizon_profile"
    if early_full_corr > 0.7:
        return "early_magnitude_present_timing_coarse"
    return "early_info_weak"
