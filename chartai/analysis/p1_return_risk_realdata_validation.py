"""P1 Return/Risk Target Real-Data Validation (analysis-only).

Validates fixed design candidate on real BTCUSDT data with prefix-fit Standard Z-score:
  Expected Return facets: U, MFE (scalar sum z(U)+z(MFE) tested for semantic loss)
  Acceptable Risk facets: MAE, Giveback, Chop (scalar sums B1/B2/B3 tested)
  Recovery: diagnostic only

Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_zscore_utils import P1ObservableZScoreBundle
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
from chartai.reward.path_observables import compute_mfe_n

FIXED_CANDIDATE = {
    "Expected_Return_facets": ["U", "MFE"],
    "Acceptable_Risk_facets": ["MAE", "Giveback", "Chop"],
    "Recovery": "diagnostic_only_not_in_canonical_sum",
}


def _ols_fit_predict(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    if len(y) < 3:
        return np.zeros_like(y), float("nan")
    X = np.column_stack([np.ones(len(y)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return pred, r2


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra, rb) or float("nan")


def _rank_list(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


@dataclass
class P1RealDataValidationConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    u_match_tol: float = 0.0003
    mfe_match_tol: float = 0.0003
    mae_match_tol: float = 0.0003
    giveback_match_tol: float = 0.08
    chop_match_tol: float = 0.04
    terminal_match_tol: float = 0.0005
    risk_match_tol: float = 0.25
    max_matched_pairs_per_type: int = 5
    max_pair_search_window: int = 80


class P1ReturnRiskRealDataValidationRunner:
    """Real-data validation of fixed P1 Return/Risk candidate structure."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1RealDataValidationConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1RealDataValidationConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self, *, test_pass_count: int | None = None) -> dict[str, Any]:
        cfg = self._cfg
        rows, t_indices = self._collect_eval_rows()
        split = max(1, int(len(t_indices) * cfg.prefix_fraction))
        prefix_rows = rows[:split]
        eval_rows = rows[split:]
        z_model = P1ObservableZScoreBundle.fit_from_rows(prefix_rows)

        enriched = [self._enrich_row(r, z_model) for r in eval_rows]

        exp_a = self._experiment_return(enriched, z_model)
        exp_b = self._experiment_risk_aggregation(enriched)
        z_audit = self._risk_zscore_semantic_audit(enriched, prefix_rows, z_model)
        risk_sum = self._risk_scalar_audit(enriched)
        matched = self._matched_path_analysis(enriched)
        joint = self._joint_return_risk(enriched)
        recovery = self._recovery_diagnostic(enriched, matched)
        verdicts = self._final_verdicts(exp_a, exp_b, z_audit, risk_sum, matched, recovery, joint)
        recommendation = self._final_recommendation(verdicts)

        report = {
            "audit": "P1 Return/Risk Target Real-Data Validation",
            "fixed_candidate_structure": FIXED_CANDIDATE,
            "primary_evidence": "BTCUSDT real eval (synthetic diagnostic only where noted)",
            "1_executive_summary": self._executive_summary(verdicts, recommendation, enriched),
            "2_data_normalization_protocol": {
                "market": describe_market_data(self._data),
                "prefix_n": split,
                "eval_n": len(eval_rows),
                "normalization": "z_X = (X - mu_prefix) / sigma_prefix",
                "causal_protocol": "prefix-fit on first 50% t-indices; apply to eval only",
                "prefix_stats": self._prefix_stats(z_model),
            },
            "3_return_validation": exp_a,
            "4_risk_normalization_audit": z_audit,
            "5_risk_aggregation_audit": {**exp_b, **risk_sum},
            "6_matched_path_analysis": matched,
            "7_joint_return_risk_analysis": joint,
            "8_recovery_diagnostic": recovery,
            "9_verdicts": verdicts,
            "10_final_recommendation": recommendation,
            "11_test_result": {
                "pytest_pass_count": test_pass_count,
                "note": "Set by run script after full pytest",
            },
        }
        return report

    def _raw_obs(self, ctx, action: Action, h: int) -> dict[str, float]:
        cfg = self._cfg
        ud = compute_u_diagnostics(ctx, action, horizon=h, utility_config=cfg.utility_config)
        obs = compute_path_residual_observables(ctx, action, h)
        mae_d = compute_mae_diagnostics(ctx, action, h)
        return {
            "U": ud.u_mean,
            "MFE": compute_mfe_n(ctx, action, h),
            "MAE": compute_mae_n(ctx, action, h),
            "giveback": obs.giveback_ratio,
            "chop": obs.oscillation_chop,
            "recovery": mae_d.recovery_after_mae,
            "terminal": obs.terminal_return,
            "path_efficiency": obs.path_efficiency,
            "P_long": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
            "P_short": compute_path_n(ctx, Action.SHORT, h, decay_rate=cfg.decay_rate),
        }

    def _collect_eval_rows(self) -> tuple[list[dict[str, float]], list[int]]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        rows: list[dict[str, float]] = []
        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            row = self._raw_obs(ctx, Action.LONG, h)
            row["t_index"] = float(t_index)
            rows.append(row)
        return rows, t_indices

    def _enrich_row(self, raw: dict[str, float], z_model: P1ObservableZScoreBundle) -> dict[str, Any]:
        z = z_model.transform(raw)
        return {
            "t_index": int(raw["t_index"]),
            "raw": raw,
            "z": z,
            "Return_U": z["U"],
            "Return_UMFE": z["U"] + z["MFE"],
            "separate_U": z["U"],
            "separate_MFE": z["MFE"],
            "Risk_MAE": abs(z["MAE"]),
            "Risk_MG": abs(z["MAE"]) + z["giveback"],
            "Risk_MGC": abs(z["MAE"]) + z["giveback"] + z["chop"],
            "archetype_proxy": self._classify_path_proxy(raw),
            "risk_case": self._classify_risk_case(raw),
        }

    def _classify_path_proxy(self, raw: dict[str, float]) -> str:
        u, mfe, gb, term = raw["U"], raw["MFE"], raw["giveback"], raw["terminal"]
        if mfe > 0 and u < mfe * 0.6 and gb > 0.4:
            return "spike_giveback"
        if u > mfe * 0.85 and gb < 0.25 and term > 0:
            return "grind"
        if mfe > 0 and u < mfe * 0.5 and term < 0:
            return "spike_crash"
        if u > 0 and mfe > 0 and gb < 0.3 and term > 0:
            return "strong_both"
        return "mixed"

    def _classify_risk_case(self, raw: dict[str, float]) -> str:
        mae = abs(raw["MAE"])
        gb = raw["giveback"]
        chop = raw["chop"]
        mae_q = mae > 0.001
        gb_q = gb > 0.35
        chop_q = chop > 0.15
        if not mae_q and gb_q and not chop_q:
            return "CASE_A_capture_erosion"
        if mae_q and not gb_q and chop_q:
            return "CASE_B_whip_chop"
        if mae_q and gb_q and chop_q:
            return "CASE_C_compound"
        if mae_q and not gb_q and not chop_q:
            return "CASE_D_adverse_magnitude"
        return "unclassified"

    def _prefix_stats(self, z_model: P1ObservableZScoreBundle) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": m.stats.center, "scale": m.stats.scale}
            for name, m in [
                ("U", z_model.u),
                ("MFE", z_model.mfe),
                ("MAE", z_model.mae),
                ("giveback", z_model.giveback),
                ("chop", z_model.chop),
                ("recovery", z_model.recovery),
            ]
        }

    def _experiment_return(
        self, enriched: list[dict[str, Any]], z_model: P1ObservableZScoreBundle
    ) -> dict[str, Any]:
        cfg = self._cfg
        raw_u = np.asarray([e["raw"]["U"] for e in enriched])
        raw_mfe = np.asarray([e["raw"]["MFE"] for e in enriched])
        zu = np.asarray([e["Return_U"] for e in enriched])
        zmfe = np.asarray([e["separate_MFE"] for e in enriched])
        rumfe = np.asarray([e["Return_UMFE"] for e in enriched])
        terminal = np.asarray([e["raw"]["terminal"] for e in enriched])

        mfe_pred, r2_u_mfe = _ols_fit_predict(raw_mfe, raw_u)
        u_pred, r2_mfe_u = _ols_fit_predict(raw_u, raw_mfe)
        mfe_resid = raw_mfe - mfe_pred
        u_resid = raw_u - u_pred

        tail_analysis = {}
        for thr in (2, 3, 4):
            mask = np.abs(zu) > thr
            n = int(np.sum(mask))
            if n == 0:
                tail_analysis[f"abs_zU_gt_{thr}"] = {"count": 0}
                continue
            mfe_share = float(np.mean(np.abs(zmfe[mask]) / (np.abs(zu[mask]) + np.abs(zmfe[mask]) + 1e-12)))
            tail_analysis[f"abs_zU_gt_{thr}"] = {
                "count": n,
                "frac_of_eval": n / len(enriched),
                "mean_zU": float(np.mean(zu[mask])),
                "mean_zMFE": float(np.mean(zmfe[mask])),
                "mean_Return_UMFE": float(np.mean(rumfe[mask])),
                "mean_terminal": float(np.mean(terminal[mask])),
                "mean_raw_MFE": float(np.mean(raw_mfe[mask])),
                "mean_raw_U": float(np.mean(raw_u[mask])),
                "MFE_share_of_abs_z_sum": mfe_share,
                "corr_Return_UMFE_with_zU_in_tail": _pearson(rumfe[mask], zu[mask]),
                "corr_Return_UMFE_with_zMFE_in_tail": _pearson(rumfe[mask], zmfe[mask]),
                "U_dominates_scalar": mfe_share < 0.35,
            }

        u_mfe_pairs = self._find_matched_pairs(
            enriched, match=("U",), differ="MFE", match_tol=cfg.u_match_tol, differ_min=cfg.mfe_match_tol * 3
        )
        mfe_u_pairs = self._find_matched_pairs(
            enriched, match=("MFE",), differ="U", match_tol=cfg.mfe_match_tol, differ_min=cfg.u_match_tol * 3
        )

        semantic_failures = self._return_semantic_failures(enriched)

        proxy_stats = {}
        for label in ("grind", "spike_giveback", "spike_crash", "strong_both"):
            subset = [e for e in enriched if e["archetype_proxy"] == label]
            if not subset:
                continue
            proxy_stats[label] = {
                "n": len(subset),
                "mean_Return_U": float(mean(e["Return_U"] for e in subset)),
                "mean_Return_UMFE": float(mean(e["Return_UMFE"] for e in subset)),
                "mean_separate_MFE": float(mean(e["separate_MFE"] for e in subset)),
                "mean_separate_U": float(mean(e["separate_U"] for e in subset)),
                "mean_terminal": float(mean(e["raw"]["terminal"] for e in subset)),
            }

        return {
            "designs": {
                "A1_Return_U": "z(U)",
                "A2_Return_UMFE": "z(U)+z(MFE)",
                "A3_separate": ["z(U)", "z(MFE)"],
            },
            "U_tail_dominance": tail_analysis,
            "overlap_stats": {
                "corr_raw_U_MFE": _pearson(raw_u, raw_mfe),
                "corr_zU_zMFE": _pearson(zu, zmfe),
                "r2_MFE_from_U": r2_u_mfe,
                "r2_U_from_MFE": r2_mfe_u,
                "mfe_residual_std": float(np.std(mfe_resid)),
                "u_residual_std": float(np.std(u_resid)),
                "corr_mfe_residual_terminal": _pearson(mfe_resid, terminal),
                "corr_u_residual_terminal": _pearson(u_resid, terminal),
            },
            "matched_U_similar_MFE_diff": u_mfe_pairs,
            "matched_MFE_similar_U_diff": mfe_u_pairs,
            "path_type_proxy_stats": proxy_stats,
            "semantic_failure_cases": semantic_failures,
        }

    def _return_semantic_failures(self, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        grinds = [e for e in enriched if e["archetype_proxy"] == "grind"]
        spikes = [e for e in enriched if e["archetype_proxy"] == "spike_giveback"]
        if grinds and spikes:
            g = max(grinds, key=lambda e: e["Return_UMFE"])
            s = max(spikes, key=lambda e: e["separate_MFE"])
            if g["Return_UMFE"] > s["Return_UMFE"] and s["separate_MFE"] > g["separate_MFE"]:
                failures.append(
                    {
                        "type": "grind_beats_spike_on_scalar_despite_higher_MFE_facet",
                        "grind_t_index": g["t_index"],
                        "spike_t_index": s["t_index"],
                        "grind_Return_UMFE": g["Return_UMFE"],
                        "spike_Return_UMFE": s["Return_UMFE"],
                        "grind_separate_MFE": g["separate_MFE"],
                        "spike_separate_MFE": s["separate_MFE"],
                        "interpretation": (
                            "Return scalar favors sustained U (grind) over higher MFE facet (spike) — "
                            "MFE facet information suppressed in A2 sum"
                        ),
                    }
                )
        crashes = [e for e in enriched if e["archetype_proxy"] == "spike_crash"]
        if crashes:
            c = max(crashes, key=lambda e: e["separate_MFE"])
            if c["Return_UMFE"] > 0 and c["raw"]["terminal"] < 0:
                failures.append(
                    {
                        "type": "positive_Return_UMFE_with_negative_terminal",
                        "t_index": c["t_index"],
                        "Return_UMFE": c["Return_UMFE"],
                        "terminal": c["raw"]["terminal"],
                        "separate_U": c["separate_U"],
                        "separate_MFE": c["separate_MFE"],
                        "interpretation": "High MFE with low U still yields positive scalar despite crash terminal",
                    }
                )
        return failures

    def _experiment_risk_aggregation(self, enriched: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "designs": {
                "B1_Risk_MAE": "z(MAE)",
                "B2_Risk_MG": "z(MAE)+z(giveback)",
                "B3_Risk_MGC": "z(MAE)+z(giveback)+z(chop)",
                "facets": ["z(MAE)", "z(giveback)", "z(chop)"],
            },
            "eval_variance_share": self._risk_variance_share(enriched),
        }

    def _risk_variance_share(self, enriched: list[dict[str, Any]]) -> dict[str, float]:
        keys = ("Risk_MAE", "Risk_MG", "Risk_MGC")
        out = {}
        for k in keys:
            vals = np.asarray([e[k] for e in enriched], dtype=float)
            out[k] = float(np.var(vals))
        total = sum(out.values()) or 1.0
        return {k: v / total for k, v in out.items()}

    def _risk_zscore_semantic_audit(
        self,
        enriched: list[dict[str, Any]],
        prefix_rows: list[dict[str, float]],
        z_model: P1ObservableZScoreBundle,
    ) -> dict[str, Any]:
        facets = ("MAE", "giveback", "chop")
        semantic_intent = {
            "MAE": "how far price moved adversely",
            "giveback": "fraction of favorable excursion returned",
            "chop": "path oscillation / whip",
        }
        audit: dict[str, Any] = {}
        for key in facets:
            raw_vals = [e["raw"][key if key != "MAE" else "MAE"] for e in enriched]
            z_vals = [e["z"][key if key != "MAE" else "MAE"] for e in enriched]
            raw_ranks = _rank_list(raw_vals)
            z_ranks = _rank_list(z_vals)
            rank_corr = _pearson(raw_ranks, z_ranks)
            model = getattr(z_model, key if key != "MAE" else "mae")
            p99 = sorted(abs(v) for v in z_vals)[int(0.99 * (len(z_vals) - 1))] if z_vals else 0.0
            audit[key] = {
                "semantic_intent": semantic_intent[key],
                "prefix_mu": model.stats.center,
                "prefix_sigma": model.stats.scale,
                "raw_vs_z_spearman": rank_corr,
                "ranking_preserved": (rank_corr or 0) > 0.95,
                "eval_z_p99_abs": p99,
                "tail_amplification": p99 > 4.0,
                "sigma_small_explosion_risk": model.stats.scale < 1e-4,
                "eval_z_std": float(pstdev(z_vals)) if len(z_vals) > 1 else 0.0,
            }
        return {"per_facet": audit, "note": "ranking_preserved on eval; tail |z|>4 possible on extremes"}

    def _risk_scalar_audit(self, enriched: list[dict[str, Any]]) -> dict[str, Any]:
        cases: dict[str, list[dict[str, Any]]] = {c: [] for c in (
            "CASE_A_capture_erosion", "CASE_B_whip_chop", "CASE_C_compound", "CASE_D_adverse_magnitude"
        )}
        for e in enriched:
            rc = e["risk_case"]
            if rc in cases and len(cases[rc]) < 3:
                cases[rc].append(self._path_record(e))

        collapse_examples = []
        for e in enriched:
            facets = {
                "MAE": abs(e["z"]["MAE"]),
                "giveback": e["z"]["giveback"],
                "chop": e["z"]["chop"],
            }
            dominant = max(facets, key=facets.get)
            share = facets[dominant] / (e["Risk_MGC"] + 1e-12)
            if e["Risk_MGC"] > 2.0 and share > 0.65:
                collapse_examples.append(
                    {
                        "t_index": e["t_index"],
                        "Risk_MGC": e["Risk_MGC"],
                        "dominant_facet": dominant,
                        "dominant_share": share,
                        "facets": facets,
                        "risk_case": e["risk_case"],
                    }
                )
            if len(collapse_examples) >= 5:
                break

        b_vs_g_like = self._find_b_vs_g_like(enriched)

        return {
            "risk_case_exemplars": cases,
            "scalar_dominance_examples": collapse_examples,
            "B_vs_G_like_real_pairs": b_vs_g_like,
            "mechanism_loss_verdict": (
                "Risk_MGC single scalar hides which facet drives risk when one facet dominates (>65% share) "
                "or when B vs G-like pairs differ mainly in chop"
            ),
        }

    def _path_record(self, e: dict[str, Any]) -> dict[str, Any]:
        return {
            "t_index": e["t_index"],
            "raw": e["raw"],
            "z": {"MAE": e["z"]["MAE"], "giveback": e["z"]["giveback"], "chop": e["z"]["chop"]},
            "Risk_MAE": e["Risk_MAE"],
            "Risk_MG": e["Risk_MG"],
            "Risk_MGC": e["Risk_MGC"],
            "Return_UMFE": e["Return_UMFE"],
        }

    def _find_b_vs_g_like(self, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self._cfg
        out = []
        for i in range(len(enriched)):
            for j in range(i + 1, min(i + cfg.max_pair_search_window, len(enriched))):
                a, b = enriched[i], enriched[j]
                if abs(a["raw"]["terminal"] - b["raw"]["terminal"]) > cfg.terminal_match_tol * 4:
                    continue
                if abs(a["raw"]["giveback"] - b["raw"]["giveback"]) > cfg.giveback_match_tol:
                    continue
                if abs(a["z"]["chop"] - b["z"]["chop"]) < 0.5:
                    continue
                out.append(
                    {
                        "pair": [a["t_index"], b["t_index"]],
                        "terminal_diff": abs(a["raw"]["terminal"] - b["raw"]["terminal"]),
                        "giveback_diff": abs(a["raw"]["giveback"] - b["raw"]["giveback"]),
                        "chop_z_diff": abs(a["z"]["chop"] - b["z"]["chop"]),
                        "Risk_MGC_diff": abs(a["Risk_MGC"] - b["Risk_MGC"]),
                        "a": self._path_record(a),
                        "b": self._path_record(b),
                    }
                )
                if len(out) >= cfg.max_matched_pairs_per_type:
                    return out
        return out

    def _matched_path_analysis(self, enriched: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        p1 = self._find_matched_pairs(
            enriched,
            match=("MAE",),
            differ="giveback",
            match_tol=cfg.mae_match_tol,
            differ_min=cfg.giveback_match_tol,
        )
        p2 = self._find_matched_pairs(
            enriched,
            match=("MAE", "giveback"),
            differ="chop",
            match_tol=cfg.mae_match_tol,
            differ_min=cfg.chop_match_tol,
            extra_match={"giveback": cfg.giveback_match_tol},
        )
        p3 = self._find_risk_similar_terminal_diff(enriched)
        p4 = self._find_single_facet_extreme(enriched)

        return {
            "pair1_MAE_similar_giveback_diff": p1,
            "pair2_MAE_giveback_similar_chop_diff": p2,
            "pair3_risk_similar_terminal_diff": p3,
            "pair4_single_facet_extreme": p4,
        }

    def _find_matched_pairs(
        self,
        enriched: list[dict[str, Any]],
        *,
        match: tuple[str, ...],
        differ: str,
        match_tol: float,
        differ_min: float,
        extra_match: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self._cfg
        out: list[dict[str, Any]] = []
        for i in range(len(enriched)):
            for j in range(i + 1, min(i + cfg.max_pair_search_window, len(enriched))):
                a, b = enriched[i], enriched[j]
                ok = all(abs(a["raw"][k] - b["raw"][k]) <= match_tol for k in match)
                if extra_match:
                    ok = ok and all(
                        abs(a["raw"][k] - b["raw"][k]) <= tol for k, tol in extra_match.items()
                    )
                if not ok:
                    continue
                if abs(a["raw"][differ] - b["raw"][differ]) < differ_min:
                    continue
                out.append(self._pair_table(a, b, f"{match}_similar_{differ}_diff"))
                if len(out) >= cfg.max_matched_pairs_per_type:
                    return out
        return out

    def _find_risk_similar_terminal_diff(self, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self._cfg
        out = []
        for i in range(len(enriched)):
            for j in range(i + 1, min(i + cfg.max_pair_search_window, len(enriched))):
                a, b = enriched[i], enriched[j]
                if abs(a["Risk_MGC"] - b["Risk_MGC"]) > cfg.risk_match_tol:
                    continue
                if abs(a["raw"]["terminal"] - b["raw"]["terminal"]) < cfg.terminal_match_tol * 5:
                    continue
                out.append(self._pair_table(a, b, "risk_similar_terminal_diff"))
                if len(out) >= cfg.max_matched_pairs_per_type:
                    return out
        return out

    def _find_single_facet_extreme(self, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(enriched) < 20:
            return []
        for key in ("MAE", "giveback", "chop"):
            zs = np.asarray([abs(e["z"][key if key != "MAE" else "MAE"]) for e in enriched])
            thr = float(np.quantile(zs, 0.95))
            extreme = [e for e in enriched if abs(e["z"][key if key != "MAE" else "MAE"]) >= thr]
            moderate = [
                e for e in enriched
                if abs(e["z"][key if key != "MAE" else "MAE"]) < float(np.quantile(zs, 0.5))
            ]
            if not extreme or not moderate:
                continue
            ex = extreme[0]
            mod = min(moderate, key=lambda e: abs(e["Risk_MGC"] - ex["Risk_MGC"]))
            return [self._pair_table(mod, ex, f"{key}_extreme_vs_moderate")]
        return []

    def _pair_table(self, a: dict[str, Any], b: dict[str, Any], pair_type: str) -> dict[str, Any]:
        return {
            "pair_type": pair_type,
            "t_indices": [a["t_index"], b["t_index"]],
            "rows": [self._pair_row(a), self._pair_row(b)],
        }

    def _pair_row(self, e: dict[str, Any]) -> dict[str, Any]:
        r, z = e["raw"], e["z"]
        return {
            "t_index": e["t_index"],
            "U": r["U"],
            "MFE": r["MFE"],
            "MAE": r["MAE"],
            "giveback": r["giveback"],
            "chop": r["chop"],
            "terminal": r["terminal"],
            "recovery": r["recovery"],
            "z_U": z["U"],
            "z_MFE": z["MFE"],
            "z_MAE": z["MAE"],
            "z_giveback": z["giveback"],
            "z_chop": z["chop"],
            "Return_U": e["Return_U"],
            "Return_UMFE": e["Return_UMFE"],
            "Risk_MAE": e["Risk_MAE"],
            "Risk_MG": e["Risk_MG"],
            "Risk_MGC": e["Risk_MGC"],
            "archetype_proxy": e["archetype_proxy"],
            "risk_case": e["risk_case"],
        }

    def _joint_return_risk(self, enriched: list[dict[str, Any]]) -> dict[str, Any]:
        quadrants = {
            "high_return_high_risk": [],
            "high_return_low_risk": [],
            "low_return_high_risk": [],
            "low_return_low_risk": [],
        }
        ret_med = float(np.median([e["Return_UMFE"] for e in enriched]))
        risk_med = float(np.median([e["Risk_MGC"] for e in enriched]))

        archetype_map = {"A_like": [], "B_like": [], "C_like": [], "G_like": []}
        for e in enriched:
            hr = e["Return_UMFE"] >= ret_med
            hk = e["Risk_MGC"] >= risk_med
            q = (
                "high_return_high_risk" if hr and hk else
                "high_return_low_risk" if hr else
                "low_return_high_risk" if hk else
                "low_return_low_risk"
            )
            if len(quadrants[q]) < 5:
                quadrants[q].append(self._path_record(e))

            proxy = e["archetype_proxy"]
            if proxy == "spike_giveback" and len(archetype_map["A_like"]) < 5:
                archetype_map["A_like"].append(self._path_record(e))
            elif proxy == "grind" and len(archetype_map["B_like"]) < 5:
                archetype_map["B_like"].append(self._path_record(e))
            elif proxy == "spike_crash" and len(archetype_map["C_like"]) < 5:
                archetype_map["C_like"].append(self._path_record(e))
            elif proxy == "mixed" and e["z"]["chop"] > 0.5 and len(archetype_map["G_like"]) < 5:
                archetype_map["G_like"].append(self._path_record(e))

        semantic_check = {}
        if archetype_map["A_like"] and archetype_map["B_like"]:
            a_risk = mean(x["Risk_MGC"] for x in archetype_map["A_like"])
            b_risk = mean(x["Risk_MGC"] for x in archetype_map["B_like"])
            semantic_check["A_like_risk_gt_B_like"] = a_risk > b_risk
        if archetype_map["B_like"] and archetype_map["G_like"]:
            b_risk = mean(x["Risk_MGC"] for x in archetype_map["B_like"])
            g_risk = mean(x["Risk_MGC"] for x in archetype_map["G_like"])
            semantic_check["G_like_risk_gt_B_like_chop"] = g_risk > b_risk

        return {
            "medians": {"Return_UMFE": ret_med, "Risk_MGC": risk_med},
            "quadrant_samples": quadrants,
            "archetype_proxy_samples": archetype_map,
            "semantic_checks": semantic_check,
            "note": "Real paths classified by proxy heuristics; not synthetic ABC ground truth",
        }

    def _recovery_diagnostic(
        self, enriched: list[dict[str, Any]], matched: dict[str, Any]
    ) -> dict[str, Any]:
        cfg = self._cfg
        blind = []
        for i in range(len(enriched)):
            for j in range(i + 1, min(i + cfg.max_pair_search_window, len(enriched))):
                a, b = enriched[i], enriched[j]
                if abs(a["raw"]["MAE"] - b["raw"]["MAE"]) > cfg.mae_match_tol:
                    continue
                if abs(a["raw"]["giveback"] - b["raw"]["giveback"]) > cfg.giveback_match_tol:
                    continue
                if abs(a["raw"]["chop"] - b["raw"]["chop"]) > cfg.chop_match_tol:
                    continue
                if abs(a["raw"]["recovery"] - b["raw"]["recovery"]) < cfg.u_match_tol:
                    continue
                blind.append(self._pair_table(a, b, "same_risk_facets_recovery_diff"))
                if len(blind) >= cfg.max_matched_pairs_per_type:
                    break
            if len(blind) >= cfg.max_matched_pairs_per_type:
                break

        return {
            "recovery_not_in_Risk_MGC": True,
            "same_MAE_giveback_chop_recovery_diff_pairs": blind,
            "pair_count": len(blind),
            "verdict": (
                "Recovery explains outcome variation not captured by Risk_MGC when pairs exist; "
                "P1 diagnostic / P2 timing — not canonical Risk facet"
            ),
        }

    def _executive_summary(
        self, verdicts: dict[str, str], recommendation: dict[str, Any], enriched: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "eval_samples": len(enriched),
            "fixed_structure_tested": FIXED_CANDIDATE,
            "headline": recommendation.get("choice_label"),
            "key_findings": [
                f"Q1 Return scalar semantic: {verdicts.get('Q1_Return_scalar_semantic_valid')}",
                f"Q6 Risk scalar sum: {verdicts.get('Q6_risk_scalar_sum_valid')}",
                f"Q2 U tail dominance: {verdicts.get('Q2_U_tail_dominates_MFE')}",
            ],
            "no_auto_adoption": "Evidence only; canonical structure unchanged by this experiment",
        }

    def _final_verdicts(
        self,
        exp_a: dict[str, Any],
        exp_b: dict[str, Any],
        z_audit: dict[str, Any],
        risk_sum: dict[str, Any],
        matched: dict[str, Any],
        recovery: dict[str, Any],
        joint: dict[str, Any],
    ) -> dict[str, str]:
        tail = exp_a["U_tail_dominance"]
        u_dom_t3 = tail.get("abs_zU_gt_3", {}).get("U_dominates_scalar", False)
        failures = exp_a.get("semantic_failure_cases", [])
        r2 = exp_a["overlap_stats"]["r2_MFE_from_U"] or 1.0
        u_mfe_pairs = len(exp_a.get("matched_U_similar_MFE_diff", []))
        mfe_u_pairs = len(exp_a.get("matched_MFE_similar_U_diff", []))

        facet_audit = z_audit["per_facet"]
        ranking_ok = all(facet_audit[k]["ranking_preserved"] for k in ("MAE", "giveback", "chop"))

        p1 = len(matched.get("pair1_MAE_similar_giveback_diff", []))
        p2 = len(matched.get("pair2_MAE_giveback_similar_chop_diff", []))
        p3 = len(matched.get("pair3_risk_similar_terminal_diff", []))
        rec_pairs = recovery.get("pair_count", 0)

        dominance = len(risk_sum.get("scalar_dominance_examples", []))
        b_g = len(risk_sum.get("B_vs_G_like_real_pairs", []))

        return {
            "Q1_Return_scalar_semantic_valid": (
                "FAILED" if failures else "PARTIAL"
            ),
            "Q2_U_tail_dominates_MFE": (
                "SUPPORTED" if u_dom_t3 else "PARTIAL"
            ),
            "Q3_MFE_independent_facet_despite_corr": (
                "CONFIRMED" if (u_mfe_pairs > 0 and mfe_u_pairs > 0) else "PARTIAL"
            ),
            "Q4_scalar_separates_spike_grind_crash": (
                "PARTIAL" if failures else "SUPPORTED"
            ),
            "Q5_zscore_preserves_risk_semantics": (
                "SUPPORTED" if ranking_ok else "PARTIAL"
            ),
            "Q6_risk_scalar_sum_valid": "FAILED" if (dominance > 0 or b_g > 0) else "PARTIAL",
            "Q7_scalar_dominance_or_dilution": (
                "CONFIRMED" if dominance > 0 else "PARTIAL"
            ),
            "Q8_giveback_chop_independent_on_real_data": (
                "CONFIRMED" if (p1 > 0 and p2 > 0) else "SUPPORTED" if p1 > 0 else "PARTIAL"
            ),
            "Q9_risk_blind_spot": (
                "SUPPORTED" if p3 > 0 else "UNRESOLVED"
            ),
            "Q10_recovery_adds_beyond_MGC": (
                "CONFIRMED" if rec_pairs > 0 else "UNRESOLVED"
            ),
            "_meta": {
                "r2_MFE_from_U": r2,
                "semantic_failure_count": len(failures),
                "matched_pair_counts": {"p1": p1, "p2": p2, "p3": p3, "recovery": rec_pairs},
            },
        }

    def _final_recommendation(self, verdicts: dict[str, str]) -> dict[str, Any]:
        q1 = verdicts["Q1_Return_scalar_semantic_valid"]
        q6 = verdicts["Q6_risk_scalar_sum_valid"]
        q5 = verdicts["Q5_zscore_preserves_risk_semantics"]

        if q1 in ("FAILED", "PARTIAL") or q6 in ("FAILED", "PARTIAL"):
            choice = "B"
            label = "Structure maintained; normalization/aggregation protocol needs revision"
            detail = (
                "Keep U/MFE and MAE/Giveback/Chop as facet targets, but do NOT treat z(U)+z(MFE) "
                "or z(MAE)+z(giveback)+z(chop) as canonical scalar labels. Use separate heads; "
                "monitor prefix z-score tails and recovery diagnostic."
            )
        elif q5 == "PARTIAL":
            choice = "B"
            label = "Structure maintained; z-score protocol refinement needed"
            detail = "Facet semantics mostly preserved; address tail/regime z-score behavior."
        else:
            choice = "A"
            label = "Current facet structure can proceed to next validation stage"
            detail = "Real BTC eval supports facet structure; scalar sums still diagnostic-only."

        return {
            "choice": choice,
            "choice_label": label,
            "detail": detail,
            "evidence_summary": {
                "Q1": q1,
                "Q5": q5,
                "Q6": q6,
                "Q8": verdicts["Q8_giveback_chop_independent_on_real_data"],
            },
            "not_auto_modified": True,
        }


def format_realdata_summary(report: dict[str, Any]) -> str:
    rec = report.get("10_final_recommendation", {})
    verdicts = report.get("9_verdicts", {})
    lines = [
        "P1 Return/Risk Real-Data Validation",
        "=" * 60,
        f"eval_n: {report.get('2_data_normalization_protocol', {}).get('eval_n')}",
        f"recommendation: {rec.get('choice')} - {rec.get('choice_label')}",
    ]
    for k in sorted(verdicts):
        if not k.startswith("_"):
            lines.append(f"  {k}: {verdicts[k]}")
    return "\n".join(lines)


def save_realdata_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(market_data: MarketDataSource, *, test_pass_count: int | None = None) -> dict[str, Any]:
    report = P1ReturnRiskRealDataValidationRunner(market_data).run(test_pass_count=test_pass_count)
    print(format_realdata_summary(report))
    return report
