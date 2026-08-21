"""Reward Logic Audit — U/MAE residual path information (analysis-only)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.path_residual_diagnostics import (
    CANDIDATE_SPECS,
    ResidualCandidateSpec,
    compute_path_residual_observables,
    get_candidate_value,
    observables_to_dict,
)
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_speed_n,
)
from chartai.reward.synthetic import SyntheticPath
from chartai.reward.utility import compute_utility_n


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ols_r2_residual_std(
    y: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> tuple[float, float]:
    """R² of y ~ x1 + x2 and residual std."""
    if len(y) < 3:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(y)), x1, x2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    resid_std = float(np.sqrt(ss_res / max(len(y) - 3, 1)))
    return r2, resid_std


def _overlap_label(corr: float) -> str:
    if math.isnan(corr):
        return "unknown"
    a = abs(corr)
    if a >= 0.65:
        return "strong"
    if a >= 0.35:
        return "moderate"
    if a >= 0.15:
        return "weak"
    return "absent"


def _new_info_label(r2: float, corr_u: float, corr_mae: float) -> str:
    if math.isnan(r2):
        return "unknown"
    max_corr = max(abs(corr_u) if not math.isnan(corr_u) else 0.0,
                   abs(corr_mae) if not math.isnan(corr_mae) else 0.0)
    unexplained = 1.0 - r2
    if unexplained >= 0.25 and max_corr < 0.65:
        return "likely_residual"
    if unexplained >= 0.12:
        return "partial_residual"
    if max_corr >= 0.65:
        return "mostly_explained_by_U_or_MAE"
    return "weak_or_redundant"


@dataclass
class UMaeResidualAuditConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    u_mae_match_tol: float = 0.0005
    same_bucket_max_pairs: int = 300


SYNTHETIC_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": 1,
        "label": "spike_giveback_vs_grind_hold",
        "path_a_levels": [0, 1, 3, 1],
        "path_b_levels": [0, 2, 2, 2],
        "question": "Similar U but different giveback / stability?",
    },
    {
        "case_id": 2,
        "label": "late_peak_hold_vs_early_peak",
        "path_a_levels": [0, 2, 3, 2.8],
        "path_b_levels": [0, 2, 2, 2],
        "question": "Near terminal but different peak timing / decay?",
    },
    {
        "case_id": 3,
        "label": "adverse_then_rise_fast_vs_slow",
        "path_a_levels": [0, -1, 2, 2],
        "path_b_levels": [0, -1, 0, 2],
        "question": "Same terminal, different recovery speed?",
    },
    {
        "case_id": 4,
        "label": "round_trip_vs_monotone",
        "path_a_levels": [0, 2, 0, 2],
        "path_b_levels": [0, 2, 2, 2],
        "question": "Same terminal, oscillation vs hold?",
    },
    {
        "case_id": 5,
        "label": "gradual_peak_vs_early_spike_late_hold",
        "path_a_levels": [0, 1, 2, 3],
        "path_b_levels": [0, 3, 2, 3],
        "question": "Same terminal, different peak shape / chop?",
    },
)


class UMaeResidualAuditRunner:
    """Audit: path information beyond U and MAE for Expected Return / Risk."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: UMaeResidualAuditConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or UMaeResidualAuditConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))
        eval_t = t_indices[split:]

        eval_rows = self._collect_eval_rows(eval_t, h)
        corr_table = self._correlation_table(eval_rows)
        residual_table = self._residual_regression_table(eval_rows)
        bucket_disc = self._same_u_mae_bucket_discrimination(eval_rows)
        synth_pairs = self._synthetic_case_pairs(cfg)
        sd_residual = self._sd_vs_u_mae_residual(eval_rows)
        double_counting = self._double_counting_table(corr_table, residual_table)
        p1_mapping = self._p1_output_mapping(double_counting, synth_pairs)
        reward_design = self._reward_design_candidates(double_counting, synth_pairs)
        synthesis = self._synthesize(
            corr_table, residual_table, bucket_disc, synth_pairs, sd_residual, double_counting
        )

        return {
            "audit": "U/MAE Residual Path Information Audit",
            "market": describe_market_data(self._data),
            "config": {
                "reward_horizon": h,
                "eval_samples": len(eval_rows),
                "decay_rate": cfg.decay_rate,
            },
            "correlation_with_U_MAE_P": corr_table,
            "residual_after_U_plus_MAE": residual_table,
            "same_u_mae_bucket_discrimination": bucket_disc,
            "synthetic_archetype_pairs": synth_pairs,
            "S_D_residual_vs_U_MAE": sd_residual,
            "double_counting_table": double_counting,
            "p1_output_mapping": p1_mapping,
            "reward_design_candidates": reward_design,
            **synthesis,
        }

    def _collect_eval_rows(self, eval_t: Sequence[int], h: int) -> list[dict[str, Any]]:
        cfg = self._cfg
        rows: list[dict[str, Any]] = []
        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            ud = compute_u_diagnostics(
                ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config
            )
            obs = compute_path_residual_observables(ctx, Action.LONG, h)
            u_mean = ud.u_mean
            mae = compute_mae_n(ctx, Action.LONG, h)
            p_val = compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate)
            row: dict[str, Any] = {
                "t_index": t_index,
                "u_mean": u_mean,
                "mae": mae,
                "p": p_val,
                "terminal": obs.terminal_return,
                "mfe": obs.mfe,
                "candidates": observables_to_dict(obs),
            }
            rows.append(row)
        return rows

    def _correlation_table(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        u = [r["u_mean"] for r in rows]
        mae = [r["mae"] for r in rows]
        p_vals = [r["p"] for r in rows]
        out: dict[str, Any] = {}
        for spec in CANDIDATE_SPECS:
            cand = [r["candidates"][spec.key] for r in rows]
            out[spec.key] = {
                "label": spec.label,
                "corr_u": _pearson(cand, u),
                "corr_mae": _pearson(cand, mae),
                "corr_p": _pearson(cand, p_vals),
                "corr_terminal": _pearson(cand, [r["terminal"] for r in rows]),
                "corr_mfe": _pearson(cand, [r["mfe"] for r in rows]),
            }
        return out

    def _residual_regression_table(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        u = np.asarray([r["u_mean"] for r in rows], dtype=float)
        mae = np.asarray([r["mae"] for r in rows], dtype=float)
        out: dict[str, Any] = {}
        for spec in CANDIDATE_SPECS:
            y = np.asarray([r["candidates"][spec.key] for r in rows], dtype=float)
            if not np.all(np.isfinite(y)):
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            r2, resid_std = _ols_r2_residual_std(y, u, mae)
            out[spec.key] = {
                "r2_explained_by_u_mae": r2,
                "residual_std": resid_std,
                "unexplained_variance_frac": 1.0 - r2 if not math.isnan(r2) else float("nan"),
            }
        return out

    def _same_u_mae_bucket_discrimination(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        tol = cfg.u_mae_match_tol
        pairs = 0
        discriminating_keys: dict[str, int] = {spec.key: 0 for spec in CANDIDATE_SPECS}
        max_diff: dict[str, float] = {spec.key: 0.0 for spec in CANDIDATE_SPECS}

        for i in range(len(rows)):
            for j in range(i + 1, min(i + 80, len(rows))):
                a, b = rows[i], rows[j]
                if abs(a["u_mean"] - b["u_mean"]) > tol:
                    continue
                if abs(a["mae"] - b["mae"]) > tol:
                    continue
                pairs += 1
                for spec in CANDIDATE_SPECS:
                    da = a["candidates"][spec.key]
                    db = b["candidates"][spec.key]
                    diff = abs(da - db)
                    max_diff[spec.key] = max(max_diff[spec.key], diff)
                    if diff > tol:
                        discriminating_keys[spec.key] += 1
                if pairs >= cfg.same_bucket_max_pairs:
                    break
            if pairs >= cfg.same_bucket_max_pairs:
                break

        ranked = sorted(
            discriminating_keys.items(),
            key=lambda x: (-x[1], -max_diff[x[0]]),
        )
        return {
            "matched_pairs": pairs,
            "tol_u_mae": tol,
            "discriminating_pair_counts": dict(ranked),
            "max_abs_diff_in_buckets": max_diff,
            "top_discriminators": [k for k, v in ranked[:5] if v > 0],
        }

    def _path_from_cumulative(
        self,
        name: str,
        levels: list[float],
        h: int,
        *,
        scale: float = 0.01,
        adverse_wick: bool = False,
    ) -> SyntheticPath:
        scaled = [x * scale for x in levels]
        moves: list[float] = []
        for i in range(1, len(scaled)):
            moves.append(scaled[i] - scaled[i - 1])
        while len(moves) < h:
            moves.append(0.0)
        return self._path_from_moves(name, moves[:h], h, adverse_wick=adverse_wick)

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
                extra = 0.03 if moves[i] < -0.015 else 0.008
                lows_list.append(min(running, anchor) * (1.0 - extra))
            lows = tuple(lows_list)
        else:
            lows = tuple(c * 0.998 for c in closes)
        highs = tuple(c * 1.002 for c in closes)
        return SyntheticPath(
            name=name,
            price_at_t=anchor,
            future_closes=closes,
            future_highs=highs,
            future_lows=lows,
            past_closes_for_sigma=tuple(100.0 + 0.01 * (i % 3 - 1) for i in range(30)),
            reward_horizon=h,
        )

    def _path_metrics(self, path: SyntheticPath, cfg: UMaeResidualAuditConfig) -> dict[str, Any]:
        h = cfg.reward_horizon
        ctx = path.to_context()
        ud = compute_u_diagnostics(
            ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config
        )
        obs = compute_path_residual_observables(ctx, Action.LONG, h)
        mae_d = compute_mae_diagnostics(ctx, Action.LONG, h)
        return {
            "name": path.name,
            "u_mean": ud.u_mean,
            "u_terminal": ud.u_terminal,
            "mae": compute_mae_n(ctx, Action.LONG, h),
            "terminal": obs.terminal_return,
            "mfe": obs.mfe,
            "p": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
            "recovery_after_mae": mae_d.recovery_after_mae,
            "favorable_occupancy": ud.favorable_occupancy,
            "candidates": observables_to_dict(obs),
        }

    def _synthetic_case_pairs(self, cfg: UMaeResidualAuditConfig) -> list[dict[str, Any]]:
        h = cfg.reward_horizon
        results: list[dict[str, Any]] = []
        for case in SYNTHETIC_CASES:
            adverse = case["case_id"] in (3, 4)
            pa = self._path_from_cumulative(
                f"case{case['case_id']}_a",
                case["path_a_levels"],
                h,
                adverse_wick=adverse,
            )
            pb = self._path_from_cumulative(
                f"case{case['case_id']}_b",
                case["path_b_levels"],
                h,
                adverse_wick=adverse,
            )
            ma = self._path_metrics(pa, cfg)
            mb = self._path_metrics(pb, cfg)
            cand_diff: dict[str, dict[str, float]] = {}
            for spec in CANDIDATE_SPECS:
                va = ma["candidates"][spec.key]
                vb = mb["candidates"][spec.key]
                cand_diff[spec.key] = {"a": va, "b": vb, "abs_diff": abs(va - vb)}
            ranked = sorted(
                cand_diff.items(),
                key=lambda x: -x[1]["abs_diff"],
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "question": case["question"],
                    "path_a": ma,
                    "path_b": mb,
                    "u_diff": abs(ma["u_mean"] - mb["u_mean"]),
                    "mae_diff": abs(ma["mae"] - mb["mae"]),
                    "terminal_diff": abs(ma["terminal"] - mb["terminal"]),
                    "p_diff": abs(ma["p"] - mb["p"]),
                    "candidate_diffs": cand_diff,
                    "top_discriminating_candidates": [
                        {"key": k, **v} for k, v in ranked[:5] if v["abs_diff"] > 1e-6
                    ],
                    "interpretation_notes": self._case_interpretation(case["case_id"], ma, mb),
                }
            )
        return results

    def _case_interpretation(
        self, case_id: int, ma: dict[str, Any], mb: dict[str, Any]
    ) -> list[str]:
        notes: list[str] = []
        u_close = abs(ma["u_mean"] - mb["u_mean"]) < 0.002
        mae_close = abs(ma["mae"] - mb["mae"]) < 0.003
        term_close = abs(ma["terminal"] - mb["terminal"]) < 0.003
        if u_close and mae_close:
            notes.append("U and MAE approximately matched - residual candidates should discriminate.")
        if term_close:
            notes.append("Terminal return similar - magnitude ranking alone cannot separate paths.")
        if case_id == 1:
            notes.append(
                "Case 1: spike-then-giveback vs grind - giveback/reversal/chop expected to differ; "
                "expected return vs risk split is design-dependent (not pre-judged)."
            )
        if case_id == 4:
            notes.append(
                "Case 4: round-trip vs monotone - oscillation/chop and path_efficiency should rise on A; "
                "may affect risk more than expected return."
            )
        if case_id == 3:
            notes.append(
                "Case 3: recovery speed differs - recovery_shape may overlap MAE blind spot (Audit 5)."
            )
        return notes

    def _sd_vs_u_mae_residual(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        speed_vals: list[float] = []
        persist_vals: list[float] = []
        u_vals: list[float] = []
        mae_vals: list[float] = []
        for r in rows:
            ctx = self._builder.build(r["t_index"])
            speed_vals.append(
                compute_speed_n(
                    ctx,
                    Action.LONG,
                    h,
                    SpeedCandidate.TIME_TO_FAVORABLE,
                    decay_rate=cfg.decay_rate,
                )
            )
            persist_vals.append(
                compute_persistence_n(
                    ctx,
                    Action.LONG,
                    h,
                    PersistenceCandidate.FAVORABLE_OCCUPANCY,
                    decay_rate=cfg.decay_rate,
                )
            )
            u_vals.append(r["u_mean"])
            mae_vals.append(r["mae"])

        u_arr = np.asarray(u_vals, dtype=float)
        mae_arr = np.asarray(mae_vals, dtype=float)
        for label, vals in [("speed_ttf", speed_vals), ("persistence_occ", persist_vals)]:
            y = np.asarray(vals, dtype=float)
            r2, _ = _ols_r2_residual_std(y, u_arr, mae_arr)
        speed_r2, _ = _ols_r2_residual_std(np.asarray(speed_vals), u_arr, mae_arr)
        persist_r2, _ = _ols_r2_residual_std(np.asarray(persist_vals), u_arr, mae_arr)

        return {
            "speed_corr_u": _pearson(speed_vals, u_vals),
            "speed_corr_mae": _pearson(speed_vals, mae_vals),
            "persist_corr_u": _pearson(persist_vals, u_vals),
            "persist_corr_mae": _pearson(persist_vals, mae_vals),
            "speed_r2_after_u_mae": speed_r2,
            "persist_r2_after_u_mae": persist_r2,
            "note": (
                "If S/D variance is mostly explained by U+MAE, P->S+D may re-express magnitude/timing "
                "already in U/MAE rather than new path structure."
            ),
        }

    def _double_counting_table(
        self,
        corr_table: dict[str, Any],
        residual_table: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in CANDIDATE_SPECS:
            c = corr_table[spec.key]
            r = residual_table[spec.key]
            cu, cm, cp = c["corr_u"], c["corr_mae"], c["corr_p"]
            r2 = r["r2_explained_by_u_mae"]
            new_info = _new_info_label(r2, cu, cm)
            classification = "redundant"
            if new_info in ("likely_residual", "partial_residual"):
                if abs(cu) < 0.65 and abs(cm) < 0.65:
                    classification = "residual_candidate"
                elif new_info == "likely_residual":
                    classification = "partial_residual"
            if abs(cu) >= 0.65 and abs(cm) < 0.35:
                classification = "mostly_U"
            if abs(cm) >= 0.65 and abs(cu) < 0.35:
                classification = "mostly_MAE"
            if abs(c.get("corr_terminal", 0.0) or 0.0) >= 0.7 and spec.key in (
                "path_efficiency",
                "terminal_proximity_mfe",
                "mfe_terminal_ratio",
            ):
                classification = "mostly_terminal_proxy"
            if spec.key in ("transition_count",) and abs(cu) < 0.2:
                # identical to oscillation_chop by construction
                classification = "duplicate_of_chop"

            rows.append(
                {
                    "candidate": spec.key,
                    "label": spec.label,
                    "U_overlap": _overlap_label(cu),
                    "MAE_overlap": _overlap_label(cm),
                    "P_overlap": _overlap_label(cp),
                    "corr_u": cu,
                    "corr_mae": cm,
                    "corr_p": cp,
                    "r2_explained_by_u_mae": r2,
                    "unexplained_frac": r["unexplained_variance_frac"],
                    "new_information": new_info,
                    "classification": classification,
                }
            )
        return rows

    def _p1_output_mapping(
        self,
        double_counting: list[dict[str, Any]],
        synth_pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "expected_return_candidates": [
                r["candidate"]
                for r in double_counting
                if r["classification"] == "residual_candidate"
                and r["candidate"]
                in (
                    "giveback_ratio",
                    "terminal_proximity_mfe",
                    "path_efficiency",
                    "peak_timing",
                    "time_near_mfe",
                )
            ],
            "expected_risk_candidates": [
                r["candidate"]
                for r in double_counting
                if r["classification"] in ("residual_candidate", "partial_residual")
                and r["candidate"]
                in (
                    "oscillation_chop",
                    "reversal_depth",
                    "drawdown_from_mfe",
                    "peak_after_decay",
                    "transition_count",
                    "recovery_shape_score",
                )
            ],
            "dual_axis_candidates": [
                r["candidate"]
                for r in double_counting
                if r["candidate"] in ("giveback_ratio", "peak_after_decay", "excursion_stability")
            ],
            "p2_or_deferred": ["recovery_shape_score", "post_mae_recovery_high"],
            "synthetic_case_semantics": {
                str(c["case_id"]): c.get("interpretation_notes", []) for c in synth_pairs
            },
        }

    def _reward_design_candidates(
        self,
        double_counting: list[dict[str, Any]],
        synth_pairs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        residual_keys = {
            r["candidate"] for r in double_counting if r["classification"] == "residual_candidate"
        }
        designs: dict[str, Any] = {}
        if "giveback_ratio" in residual_keys:
            designs["giveback_ratio"] = {
                "form": "nonlinear_penalty",
                "sketch": "penalty = 0 if giveback < tau else -f((giveback - tau) / (1 - tau))",
                "rationale": "Low giveback may be benign; penalty activates after material MFE retracement.",
                "linear_additive": "weak - threshold preferred",
            }
        if "oscillation_chop" in residual_keys:
            designs["oscillation_chop"] = {
                "form": "risk_adjustment",
                "sketch": "expected_risk += g(chop) * mae or interaction chop * mae",
                "rationale": "Chop increases execution variance; may not reduce expected return magnitude.",
                "linear_additive": "moderate if scoped to risk head only",
            }
        if "peak_timing" in residual_keys:
            designs["peak_timing"] = {
                "form": "conditional_on_mfe",
                "sketch": "bonus/penalty only when mfe > mfe_min",
                "rationale": "Early vs late peak matters when opportunity exists; flat paths irrelevant.",
            }
        if "recovery_shape_score" in residual_keys:
            designs["recovery_shape_score"] = {
                "form": "diagnostic_or_p1_risk_head",
                "sketch": "NOT linear reward - P1 risk head or P2 wait signal",
                "rationale": "Hindsight recovery overlaps deferred opportunity (Audit 5).",
            }
        designs["_general"] = {
            "avoid": "reward += w * candidate without U/MAE conditioning",
            "prefer": "residual = candidate - E[candidate | U, MAE] or interaction terms",
            "note": "No weights finalized; canonical reward unchanged.",
        }
        case1 = next((c for c in synth_pairs if c["case_id"] == 1), None)
        if case1:
            designs["case1_design_question"] = {
                "paths": "0->1->3->1 vs 0->2->2->2",
                "open": "Higher giveback may lower expected return capture OR raise risk - semantics TBD",
                "candidates_that_separate": case1.get("top_discriminating_candidates", []),
            }
        return designs

    def _synthesize(
        self,
        corr_table: dict[str, Any],
        residual_table: dict[str, Any],
        bucket_disc: dict[str, Any],
        synth_pairs: list[dict[str, Any]],
        sd_residual: dict[str, Any],
        double_counting: list[dict[str, Any]],
    ) -> dict[str, Any]:
        confirmed: list[str] = []
        hypothesis: list[str] = []
        redundant: list[str] = []
        residual_candidates: list[str] = []
        do_not_adopt: list[str] = []

        for row in double_counting:
            if row["classification"] == "residual_candidate":
                residual_candidates.append(row["candidate"])
            elif row["classification"] in (
                "mostly_U",
                "mostly_MAE",
                "redundant",
                "mostly_terminal_proxy",
                "duplicate_of_chop",
            ):
                redundant.append(row["candidate"])

        if bucket_disc.get("matched_pairs", 0) > 0:
            confirmed.append(
                f"Real eval: {bucket_disc['matched_pairs']} pairs with similar U and MAE exist; "
                f"residual candidates still vary (top: {bucket_disc.get('top_discriminators', [])})."
            )

        case1 = next((c for c in synth_pairs if c["case_id"] == 1), None)
        if case1 and case1["u_diff"] < 0.01:
            confirmed.append(
                "Case 1 synthetic: similar U_mean between spike-giveback and grind-hold paths; "
                "giveback/reversal/chop discriminate structure."
            )

        gb_corr_u = corr_table.get("giveback_ratio", {}).get("corr_u", float("nan"))
        if not math.isnan(gb_corr_u) and abs(gb_corr_u) < 0.5:
            confirmed.append(
                "Giveback ratio has moderate/low direct correlation with U on eval data - "
                "not fully redundant with opportunity magnitude."
            )

        rec_r2 = residual_table.get("recovery_shape_score", {}).get("r2_explained_by_u_mae", 0)
        if rec_r2 < 0.85:
            hypothesis.append(
                "Recovery shape retains variance after U+MAE - aligns with MAE recovery blind spot (Audit 5)."
            )

        if sd_residual.get("persist_r2_after_u_mae", 0) > 0.7:
            hypothesis.append(
                "Persistence (occupancy) largely explained by U+MAE - S/D may not add independent path info."
            )
        else:
            hypothesis.append(
                "Persistence retains residual after U+MAE - possible timing/hold structure beyond U mean."
            )

        hypothesis.append(
            "Peak timing vs U: early peak paths may score differently on capture timing without "
            "changing MAE - Expected Return relevance plausible, not confirmed as GT."
        )

        do_not_adopt.extend(
            [
                "recovery_shape_score as silent canonical reward term",
                "linear additive chop without risk-head semantics",
                "terminal_proximity_mfe as standalone reward (overlaps terminal/MFE)",
                "any candidate without U/MAE conditioning (double counting risk)",
            ]
        )

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "REDUNDANT": redundant,
            "RESIDUAL_PATH_CANDIDATES": residual_candidates,
            "EXPECTED_RETURN_RELEVANCE": [
                c
                for c in residual_candidates
                if c
                in (
                    "giveback_ratio",
                    "terminal_proximity_mfe",
                    "path_efficiency",
                    "peak_timing",
                    "time_near_mfe",
                    "excursion_stability",
                )
            ],
            "EXPECTED_RISK_RELEVANCE": [
                c
                for c in residual_candidates
                if c
                in (
                    "oscillation_chop",
                    "reversal_depth",
                    "drawdown_from_mfe",
                    "peak_after_decay",
                    "transition_count",
                )
            ],
            "REWARD_DESIGN_CANDIDATES": [
                c
                for c in residual_candidates
                if c not in ("recovery_shape_score", "post_mae_recovery_high", "mfe_terminal_ratio")
            ],
            "DO_NOT_ADOPT_YET": do_not_adopt,
            "RECOMMENDED_NEXT_STEP": [
                "Case A/B real-data pairs: match U+MAE buckets on BTC and chart giveback vs grind.",
                "Quantile residual analysis: E[candidate | U, MAE] bins vs realized P1 proxy (not GT).",
                "Giveback vs peak_after_decay collinearity study - adopt at most one in reward.",
                "Recovery: keep as P1 risk head diagnostic; do not merge into MAE scalar.",
                "S+D decision deferred until residual table stable across assets.",
            ],
        }


def format_u_mae_residual_summary(report: dict[str, Any]) -> str:
    lines = [
        "U/MAE Residual Path Information Audit",
        "=" * 60,
        f"eval samples: {report.get('config', {}).get('eval_samples', '?')}",
        f"CONFIRMED: {len(report.get('CONFIRMED', []))}",
        f"RESIDUAL candidates: {report.get('RESIDUAL_PATH_CANDIDATES', [])}",
        f"REDUNDANT: {len(report.get('REDUNDANT', []))}",
    ]
    for item in report.get("CONFIRMED", [])[:3]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def save_u_mae_residual_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = UMaeResidualAuditRunner(market_data).run()
    print(format_u_mae_residual_summary(report))
    return report
