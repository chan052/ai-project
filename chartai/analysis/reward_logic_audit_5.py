"""Reward Logic Audit 5 — U persistence & MAE early-path information (analysis-only)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.mae_recovery_diagnostics import (
    MaeCase,
    compute_mae_case_diagnostics,
    early_info_level,
    mae_blind_spot_level,
)
from chartai.analysis.path_archetypes import classify_archetype, compute_extended_observables
from chartai.analysis.u_persistence_diagnostics import (
    UDiagnostics,
    compute_u_diagnostics,
    persistence_information_level,
    u_profile_to_dict,
)
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data, load_ohlcv_csv
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
from chartai.reward.synthetic import SyntheticPath, SyntheticScenario, build_scenario
from chartai.reward.utility import compute_utility_n


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _role_strength(score: float) -> str:
    if math.isnan(score):
        return "absent"
    if score >= 0.65:
        return "strong"
    if score >= 0.35:
        return "moderate"
    if score >= 0.15:
        return "weak"
    return "absent"


@dataclass
class RewardLogicAudit5Config:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    decay_rate: float = 0.75
    early_mae_bars: int = 3
    mae_match_tol: float = 0.0003


class RewardLogicAudit5Runner:
    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: RewardLogicAudit5Config | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or RewardLogicAudit5Config()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(
                reward_horizon=h,
                min_past_bars=cfg.min_past_bars,
            )
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))
        eval_t = t_indices[split:]

        eval_u: list[UDiagnostics] = []
        eval_arch: list[str] = []
        eval_mae_diag = []
        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            eval_u.append(
                compute_u_diagnostics(
                    ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config
                )
            )
            eval_mae_diag.append(
                compute_mae_diagnostics(ctx, Action.LONG, h, early_bars=cfg.early_mae_bars)
            )
            ext = compute_extended_observables(ctx, Action.LONG, h)
            eval_arch.append(classify_archetype(ext))

        synth_u = self._synthetic_u_paths(cfg)
        synth_mae = self._synthetic_mae_paths(cfg)
        controlled_u = self._controlled_same_terminal_u(cfg)
        controlled_mae = self._controlled_same_mae_recovery(cfg)

        u_real_corr = self._u_real_correlations(eval_u)
        mae_real = self._mae_real_analysis(eval_mae_diag, eval_arch)
        semantic = self._semantic_role_table(u_real_corr, mae_real, synth_u, synth_mae)

        report: dict[str, Any] = {
            "audit": "Reward Logic Audit 5 — U persistence / MAE early-path",
            "purpose": (
                "Audit whether U and MAE already encode persistence and early-path "
                "information, and whether that encoding is sufficient for P1 semantics."
            ),
            "dataset": describe_market_data(self._data),
            "methodology": {
                "not_goal": "Predictive winner or new canonical reward selection",
                "q1_q2_framework": "Mathematical inclusion vs P1-sufficient granularity",
                "horizon": h,
                "utility": "U_n = U(aligned cumulative return at t+n); F uses mean(U_n)",
                "mae": "MAE_n = max adverse through t+n; monotonic non-decreasing in n",
            },
            "u_persistence_audit": {
                "real_data_correlations": u_real_corr,
                "synthetic_archetypes": synth_u,
                "controlled_same_terminal": controlled_u,
                "real_by_archetype": self._u_by_archetype(eval_u, eval_arch),
            },
            "mae_early_path_audit": {
                "real_data": mae_real,
                "synthetic_cases": synth_mae,
                "controlled_same_mae": controlled_mae,
                "recovery_blind_spot": self._recovery_blind_spot(eval_mae_diag),
            },
            "semantic_role_table": semantic,
            "correction_candidates": self._correction_candidates(semantic, synth_u, synth_mae),
            "p1_p2_boundary": self._p1_p2_boundary(),
            "mtf_connection": self._mtf_connection_note(),
        }
        conclusions = self._synthesize(report, u_real_corr, mae_real, controlled_u, controlled_mae)
        report.update(conclusions)
        return report

    def _synthetic_u_paths(self, cfg: RewardLogicAudit5Config) -> dict[str, Any]:
        h = cfg.reward_horizon
        specs: dict[str, list[float]] = {
            "spike_regression": [0.08, 0.04, 0.01, -0.02, -0.02, -0.01, -0.01, 0.0, 0.0, 0.01],
            "moderate_rise_hold": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "fast_rise_hold": [0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "slow_grind_hold": [0.005] * h,
            "rise_then_fall": [0.03, 0.03, 0.02, 0.01, -0.01, -0.02, -0.02, -0.01, -0.01, -0.01],
            "rise_continuation": [0.02, 0.02, 0.03, 0.03, 0.02, 0.02, 0.03, 0.03, 0.02, 0.02],
            "favorable_oscillation": [0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
            "single_spike": [0.10, -0.02, -0.02, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        rows: dict[str, Any] = {}
        for name, moves in specs.items():
            syn = self._path_from_moves(name, moves[:h], h)
            ctx = syn.to_context()
            d = compute_u_diagnostics(
                ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config
            )
            rows[name] = u_profile_to_dict(d)
        spike = rows["spike_regression"]
        hold = rows["moderate_rise_hold"]
        rows["spike_vs_moderate_comparison"] = {
            "u_mean_diff": spike["u_mean"] - hold["u_mean"],
            "occupancy_diff": spike["favorable_occupancy"] - hold["favorable_occupancy"],
            "max_run_diff": spike["max_favorable_run"] - hold["max_favorable_run"],
            "terminal_diff": spike["terminal_return"] - hold["terminal_return"],
        }
        return rows

    def _controlled_same_terminal_u(self, cfg: RewardLogicAudit5Config) -> dict[str, Any]:
        h = cfg.reward_horizon
        # Tune last bars so terminal aligned return ~ equal
        path_a = self._path_from_moves(
            "spike_reg_same_term",
            [0.06, 0.03, 0.0, -0.01, -0.01, 0.0, 0.005, 0.005, 0.005, 0.005],
            h,
        )
        path_b = self._path_from_moves(
            "grind_same_term",
            [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012],
            h,
        )
        da = compute_u_diagnostics(
            path_a.to_context(), Action.LONG, horizon=h, utility_config=cfg.utility_config
        )
        db = compute_u_diagnostics(
            path_b.to_context(), Action.LONG, horizon=h, utility_config=cfg.utility_config
        )
        return {
            "path_a_spike_regression": u_profile_to_dict(da),
            "path_b_grind_hold": u_profile_to_dict(db),
            "terminal_match": abs(da.terminal_return - db.terminal_return) < 0.005,
            "u_mean_preserves_shape_diff": abs(da.u_mean - db.u_mean) > 0.0001,
            "occupancy_diff": da.favorable_occupancy - db.favorable_occupancy,
            "max_run_diff": da.max_favorable_run - db.max_favorable_run,
            "interpretation": (
                "If terminal similar but occupancy/run differ, U_mean may still collapse shape."
            ),
        }

    def _synthetic_mae_paths(self, cfg: RewardLogicAudit5Config) -> dict[str, Any]:
        h = cfg.reward_horizon
        cases: dict[MaeCase, list[float]] = {
            MaeCase.EARLY_ADVERS_RECOVERY: [-0.04, -0.02, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
            MaeCase.EARLY_ADVERS_SUSTAINED: [-0.04, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, -0.01, 0.0, 0.0],
            MaeCase.LATE_ADVERS: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.03, -0.03, -0.02, -0.02],
            MaeCase.SMALL_THEN_LARGE_ADVERS: [-0.005, -0.005, -0.005, -0.005, -0.005, -0.04, -0.01, 0.01, 0.01, 0.01],
            MaeCase.LARGE_THEN_RECOVERY: [-0.06, -0.02, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
        }
        rows: dict[str, Any] = {}
        for case, moves in cases.items():
            syn = self._path_from_moves(case.value, moves[:h], h, adverse_wick=True)
            diag = compute_mae_case_diagnostics(
                syn.to_context(), Action.LONG, case, horizon=h, early_bars=cfg.early_mae_bars
            )
            rows[case.value] = {
                "full_mae": diag.full_mae,
                "early_mae": diag.early_mae,
                "time_to_mae": diag.time_to_mae,
                "adverse_duration": diag.adverse_duration,
                "recovery_after_mae": diag.recovery_after_mae,
                "terminal_return": diag.terminal_return,
                "mae_profile": list(diag.mae_profile),
            }
        rec = rows[MaeCase.EARLY_ADVERS_RECOVERY.value]
        sus = rows[MaeCase.EARLY_ADVERS_SUSTAINED.value]
        rows["recovery_vs_sustained"] = {
            "full_mae_diff": rec["full_mae"] - sus["full_mae"],
            "same_full_mae_approx": abs(rec["full_mae"] - sus["full_mae"]) < 0.01,
            "recovery_diff": rec["recovery_after_mae"] - sus["recovery_after_mae"],
            "mae_scores_similar": abs(rec["full_mae"] - sus["full_mae"]) < 0.015,
        }
        return rows

    def _controlled_same_mae_recovery(self, cfg: RewardLogicAudit5Config) -> dict[str, Any]:
        h = cfg.reward_horizon
        recovery = self._path_from_moves(
            "same_mae_recovery",
            [-0.05, -0.01, 0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            h,
            adverse_wick=True,
        )
        sustained = self._path_from_moves(
            "same_mae_sustained",
            [-0.05, -0.03, -0.02, -0.01, -0.01, -0.01, -0.01, 0.0, 0.0, 0.0],
            h,
            adverse_wick=True,
        )
        dr = compute_mae_diagnostics(
            recovery.to_context(), Action.LONG, h, early_bars=cfg.early_mae_bars
        )
        ds = compute_mae_diagnostics(
            sustained.to_context(), Action.LONG, h, early_bars=cfg.early_mae_bars
        )
        return {
            "recovery_path": {
                "full_mae": dr.full_mae,
                "recovery": dr.recovery_after_mae,
                "terminal": dr.terminal_aligned_return,
            },
            "sustained_path": {
                "full_mae": ds.full_mae,
                "recovery": ds.recovery_after_mae,
                "terminal": ds.terminal_aligned_return,
            },
            "full_mae_similar": abs(dr.full_mae - ds.full_mae) < 0.02,
            "recovery_discriminates": dr.recovery_after_mae > ds.recovery_after_mae + 0.5,
            "mae_penalty_similar": abs(dr.full_mae - ds.full_mae) < 0.02,
        }

    def _path_from_moves(
        self,
        name: str,
        moves: list[float],
        h: int,
        *,
        adverse_wick: bool = False,
    ) -> SyntheticPath:
        anchor = 100.0
        prices = [anchor]
        for m in moves:
            prices.append(prices[-1] * (1.0 + m))
        closes = tuple(prices[1:])
        if adverse_wick:
            lows_list: list[float] = []
            running = anchor
            for i, c in enumerate(closes):
                running = min(running, c)
                extra = 0.03 if moves[i] < -0.02 else 0.005
                lows_list.append(min(running, anchor) * (1.0 - extra))
            lows = tuple(lows_list)
        else:
            lows = tuple(c * 0.999 for c in closes)
        return SyntheticPath(
            name=name,
            price_at_t=anchor,
            future_closes=closes,
            future_highs=tuple(c * 1.001 for c in closes),
            future_lows=lows,
            past_closes_for_sigma=tuple(100.0 + 0.01 * (i % 3 - 1) for i in range(30)),
            reward_horizon=h,
        )

    def _u_real_correlations(self, eval_u: list[UDiagnostics]) -> dict[str, float]:
        u_mean = [d.u_mean for d in eval_u]
        u_term = [d.u_terminal for d in eval_u]
        occ = [d.favorable_occupancy for d in eval_u]
        run = [d.max_favorable_run for d in eval_u]
        ttf = [float(d.time_to_favorable or (d.horizon + 1)) for d in eval_u]
        return {
            "u_mean_vs_favorable_occupancy": _pearson(u_mean, occ),
            "u_mean_vs_max_favorable_run": _pearson(u_mean, run),
            "u_mean_vs_time_to_favorable": _pearson(u_mean, [-x for x in ttf]),
            "u_terminal_vs_terminal_return": _pearson(u_term, [d.terminal_return for d in eval_u]),
            "u_mean_vs_mfe": _pearson(u_mean, [d.mfe for d in eval_u]),
        }

    def _u_by_archetype(
        self, eval_u: list[UDiagnostics], eval_arch: list[str]
    ) -> dict[str, Any]:
        by: dict[str, list[UDiagnostics]] = {}
        for d, a in zip(eval_u, eval_arch):
            by.setdefault(a, []).append(d)
        out: dict[str, Any] = {}
        for name, group in sorted(by.items(), key=lambda x: -len(x[1])):
            out[name] = {
                "count": len(group),
                "mean_u": float(np.mean([g.u_mean for g in group])),
                "mean_occupancy": float(np.mean([g.favorable_occupancy for g in group])),
                "mean_max_run": float(np.mean([g.max_favorable_run for g in group])),
            }
        return out

    def _mae_real_analysis(
        self, eval_mae: list, eval_arch: list[str]
    ) -> dict[str, Any]:
        full = [d.full_mae for d in eval_mae]
        early = [d.early_mae for d in eval_mae]
        rec = [d.recovery_after_mae for d in eval_mae]
        ttm = [float(d.time_to_mae or 11) for d in eval_mae]
        return {
            "full_vs_early_corr": _pearson(full, early),
            "full_vs_recovery_corr": _pearson(full, rec),
            "early_vs_time_to_mae_corr": _pearson(early, ttm),
            "by_archetype": {
                arch: {
                    "count": sum(1 for a in eval_arch if a == arch),
                    "mean_full_mae": float(
                        np.mean([eval_mae[i].full_mae for i, a in enumerate(eval_arch) if a == arch])
                    )
                    if any(a == arch for a in eval_arch)
                    else float("nan"),
                    "mean_recovery": float(
                        np.mean(
                            [
                                eval_mae[i].recovery_after_mae
                                for i, a in enumerate(eval_arch)
                                if a == arch
                            ]
                        )
                    )
                    if any(a == arch for a in eval_arch)
                    else float("nan"),
                }
                for arch in sorted(set(eval_arch))
            },
        }

    def _recovery_blind_spot(self, eval_mae: list) -> dict[str, Any]:
        tol = self._cfg.mae_match_tol
        pairs = 0
        for i in range(len(eval_mae)):
            for j in range(i + 1, min(i + 200, len(eval_mae))):
                a, b = eval_mae[i], eval_mae[j]
                if abs(a.full_mae - b.full_mae) > tol:
                    continue
                if abs(a.recovery_after_mae - b.recovery_after_mae) > 0.5:
                    pairs += 1
        full = [d.full_mae for d in eval_mae]
        rec = [d.recovery_after_mae for d in eval_mae]
        return {
            "similar_full_mae_different_recovery_pairs": pairs,
            "full_vs_recovery_corr": _pearson(full, rec),
            "blind_spot_level": mae_blind_spot_level(
                same_mae_different_recovery_pairs=pairs,
                recovery_corr_with_full_mae=_pearson(full, rec),
            ),
        }

    def _semantic_role_table(
        self,
        u_corr: dict[str, float],
        mae_real: dict[str, Any],
        synth_u: dict[str, Any],
        synth_mae: dict[str, Any],
    ) -> dict[str, Any]:
        spike_hold = synth_u.get("spike_vs_moderate_comparison", {})
        rec_sus = synth_mae.get("recovery_vs_sustained", {})
        return {
            "U": {
                "actually_encodes": {
                    "magnitude": _role_strength(abs(u_corr.get("u_mean_vs_mfe", float("nan")))),
                    "persistence": _role_strength(
                        abs(u_corr.get("u_mean_vs_favorable_occupancy", float("nan")))
                    ),
                    "favorable_timing": _role_strength(
                        abs(u_corr.get("u_mean_vs_time_to_favorable", float("nan")))
                    ),
                    "path_shape": "moderate"
                    if spike_hold.get("occupancy_diff", 0) != 0
                    else "weak",
                },
                "intended_role": "return opportunity / magnitude (+ coarse horizon mix)",
                "gaps": [
                    "mean(U_n) collapses spike-vs-hold when terminal similar",
                    "U_n uses cumulative return at n — conflates magnitude and path-to-n",
                ],
                "overlaps_with": ["P (magnitude+decay)", "persistence observables via occupancy corr"],
            },
            "MAE": {
                "actually_encodes": {
                    "adverse_magnitude": "strong",
                    "adverse_timing": _role_strength(
                        abs(mae_real.get("early_vs_time_to_mae_corr", float("nan")))
                    ),
                    "adverse_persistence": "moderate",
                    "recovery": _role_strength(
                        abs(mae_real.get("full_vs_recovery_corr", float("nan")))
                    ),
                },
                "intended_role": "entry adverse risk (worst excursion through horizon)",
                "gaps": [
                    "recovery-after-adverse not in MAE scalar"
                    if rec_sus.get("mae_scores_similar")
                    else "partial recovery discrimination",
                    "full MAE monotonic — early timing only via profile inspection",
                ],
                "overlaps_with": ["early path pain visible in MAE_n profile", "U late horizons"],
            },
            "P": {
                "actually_encodes": {
                    "direction": "strong",
                    "temporal_structure": "moderate (decay weights)",
                    "magnitude": "strong",
                },
                "intended_role": "direction / temporal structure",
                "gaps": ["magnitude overlap with U (Audit 3)"],
                "overlaps_with": ["U magnitude", "S/D timing candidates"],
            },
        }

    def _correction_candidates(
        self, semantic: dict[str, Any], synth_u: dict[str, Any], synth_mae: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "U_candidates": {
                "A_keep_current": "Coarse persistence via mean(U_n) — case B if P1 output separates magnitude",
                "B_add_persistence_observable": "Diagnostic / separate P1 head — not auto reward",
                "C_U_mag_plus_U_persist": "Logical split — requires composer sign-off",
                "D_U_n_aggregation_change": "early/late split aligns Audit 4 — diagnostic first",
            },
            "MAE_candidates": {
                "A_keep_full_mae": "Early magnitude present in MAE profile; recovery is blind spot",
                "B_early_mae": "NOT auto-adopt — only if entry-timing risk needs explicit term",
                "C_time_to_mae": "Diagnostic for timing",
                "D_adverse_duration": "Diagnostic; overlaps S/D",
                "E_recovery_after_mae": "Diagnostic or P1 risk head — NOT silent reward",
                "F_mae_plus_recovery_split": "If P1 risk = f(adverse, recovery) — output not reward yet",
            },
            "adoption_triage": {
                "S_plus_D": "candidate_replace_P (Audit 3) — not this audit",
                "early_MAE": "diagnostic_first",
                "recovery_in_reward": "reject_without_p1_output_design",
            },
        }

    def _p1_p2_boundary(self) -> dict[str, str]:
        return {
            "direction": "P1",
            "expected_return_magnitude": "P1 (U / terminal-aligned)",
            "risk_magnitude": "P1 (MAE scalar); recovery detail may be P1 head or P2",
            "entry_timing_wait": "P1 state + multi-aspect output; not single scalar F",
            "recovery_after_adverse": "P1 diagnostic or risk head; P2 handles wait/scale",
            "stop_placement_TP_timing": "P2 execution policy",
            "exit_timing": "P2",
        }

    def _mtf_connection_note(self) -> str:
        return (
            "MTF Audit 5: 1H context may shift conditional future behavior when 3m pattern matches. "
            "U/MAE/P evaluate 3m future path only — they do not ingest HTF state. "
            "P1 State(t) includes MTF; reward labels remain 3m-path observables. "
            "Semantic target alignment requires P1 model to map MTF state → direction/return/risk, "
            "not HTF bonus in reward."
        )

    def _synthesize(
        self,
        report: dict[str, Any],
        u_corr: dict[str, float],
        mae_real: dict[str, Any],
        controlled_u: dict[str, Any],
        controlled_mae: dict[str, Any],
    ) -> dict[str, Any]:
        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []

        occ_corr = u_corr.get("u_mean_vs_favorable_occupancy", float("nan"))
        if not math.isnan(occ_corr) and occ_corr > 0.3:
            confirmed.append(
                "U_mean correlates with favorable occupancy on eval data — persistence signal "
                "mathematically present (diagnostic, not GT)."
            )

        if controlled_u.get("terminal_match") and controlled_u.get("occupancy_diff", 0) != 0:
            confirmed.append(
                "Controlled synthetic: similar terminal but spike-vs-grind differ in "
                "occupancy/max_run — U_mean may not preserve shape at equal terminal."
            )

        if controlled_mae.get("full_mae_similar") and controlled_mae.get("recovery_discriminates"):
            confirmed.append(
                "Controlled synthetic: similar full MAE but recovery path differs — "
                "MAE scalar blind to recovery (recovery is diagnostic)."
            )

        if mae_real.get("full_vs_early_corr", 0) > 0.7:
            confirmed.append(
                "Full MAE and early MAE highly correlated on eval — early adverse magnitude "
                "already embedded in MAE_n monotonic profile."
            )

        blind = report["mae_early_path_audit"]["recovery_blind_spot"]
        if blind.get("similar_full_mae_different_recovery_pairs", 0) > 0:
            confirmed.append(
                f"Real data: {blind['similar_full_mae_different_recovery_pairs']} pairs with "
                "similar full MAE but different recovery — MAE recovery blind spot."
            )

        hypothesis.append(
            "U persistence encoding is coarse (level 2): included but insufficient for "
            "spike-vs-hold without separate persistence head or S/D."
        )
        hypothesis.append(
            "MAE encodes early adverse magnitude via profile; recovery belongs to P1 risk "
            "head or P2, not automatic MAE replacement with early_MAE."
        )
        hypothesis.append(
            "P1 output triple (direction, return mag, risk mag) can align with P/U/MAE "
            "if scalar F is not sole training target."
        )

        unresolved.append(
            "Whether U_n aggregation change (early/late) is P1-relevant vs P2 — needs chart qual."
        )
        unresolved.append(
            "Recovery in P1 risk estimate vs P2 wait policy — product decision."
        )

        q = self._final_questions(report, u_corr, controlled_u, controlled_mae, occ_corr)

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "UNRESOLVED": unresolved,
            "RECOMMENDATION": [
                "Do NOT auto-add early_MAE or recovery to canonical reward from this audit.",
                "Treat persistence (occupancy/run) and recovery as P1 output candidates or diagnostics.",
                "If reward changes: prefer U_n aggregation / composer split over bolting observables.",
                "Keep S+D as P replacement candidate (Audit 3); orthogonal to U/MAE audit.",
                "P1 state MTF (Audit 5 MTF) feeds model; do not add HTF to reward formula.",
            ],
            "final_questions_Q1_Q10": q,
        }

    def _final_questions(
        self,
        report: dict[str, Any],
        u_corr: dict[str, float],
        controlled_u: dict[str, Any],
        controlled_mae: dict[str, Any],
        occ_corr: float,
    ) -> dict[str, str]:
        u_level = persistence_information_level(
            occupancy_corr=occ_corr if not math.isnan(occ_corr) else 0.0,
            run_corr=u_corr.get("u_mean_vs_max_favorable_run", 0.0) or 0.0,
            controlled_discriminates=bool(controlled_u.get("occupancy_diff")),
        )
        early_level = early_info_level(
            early_full_corr=report["mae_early_path_audit"]["real_data"]["full_vs_early_corr"],
            time_to_mae_discriminates=True,
        )
        return {
            "Q1_U_includes_persistence": (
                "yes_coarse" if u_level != "persistence_insufficient" else "minimal"
            ),
            "Q2_U_persistence_sufficient_for_P1": (
                "no — coarse; spike-vs-hold not reliably separated at similar terminal"
            ),
            "Q3_MAE_includes_early_adverse": early_level,
            "Q4_MAE_core_blind_spot": "recovery_after_adverse (not early magnitude)",
            "Q5_need_reward_component_add": (
                "case_B — coarse but may suffice if P1 multi-head; not auto-add"
            ),
            "Q6_if_add_what": "persistence/recovery as P1 output heads or diagnostics, not silent reward terms",
            "Q7_P1_vs_P2": "recovery/wait → P2; direction/return/risk magnitude → P1",
            "Q8_keep_P_plus_U_minus_MAE": (
                "Keep as baseline; modify if composer splits U aggregation / adds explicit heads"
            ),
            "Q9_triage": (
                "S+D: candidate_replace_P; early_MAE: diagnostic; recovery: diagnostic/P1_head"
            ),
            "Q10_P1_output_alignment": (
                "Partial — U→return mag, MAE→risk mag, P→direction/timing; scalar F insufficient"
            ),
        }


def format_audit5_summary(report: dict[str, Any]) -> str:
    q = report.get("final_questions_Q1_Q10", {})
    lines = [
        "Reward Logic Audit 5 - U / MAE",
        "=" * 60,
        f"CONFIRMED: {len(report.get('CONFIRMED', []))}",
        f"Q1 U persistence: {q.get('Q1_U_includes_persistence')}",
        f"Q4 MAE blind spot: {q.get('Q4_MAE_core_blind_spot')}",
        f"Q9 triage: {q.get('Q9_triage')}",
    ]
    return "\n".join(lines)


def save_audit5_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = RewardLogicAudit5Runner(market_data).run()
    print(format_audit5_summary(report))
    return report
