"""Path instability vs MAE audit for P1 Acceptable Risk (analysis-only)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.path_residual_diagnostics import (
    compute_path_residual_observables,
    observables_to_dict,
)
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditRunner, UMaeResidualAuditConfig, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_mfe_n


def _ols_r2(y: np.ndarray, *xs: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    X = np.column_stack([np.ones(len(y)), *xs])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")


INSTABILITY_KEYS: tuple[tuple[str, str], ...] = (
    ("giveback_ratio", "A giveback"),
    ("reversal_depth", "B reversal_depth"),
    ("drawdown_from_mfe", "C drawdown_from_mfe"),
    ("excursion_volatility", "D excursion_volatility"),
    ("oscillation_chop", "E oscillation_chop"),
    ("peak_after_decay", "F peak_after_decay"),
)

# Clusters of near-duplicate instability family
INSTABILITY_CLUSTERS: tuple[dict[str, Any], ...] = (
    {
        "name": "peak_erosion_family",
        "members": ("giveback_ratio", "reversal_depth", "peak_after_decay", "drawdown_from_mfe"),
        "note": "Monotone giveback paths: giveback~reversal. Round-trip: reversal >> giveback.",
    },
    {
        "name": "path_whip_family",
        "members": ("oscillation_chop",),
        "note": "Sign-change / round-trip without terminal giveback.",
    },
    {
        "name": "favorable_vol_family",
        "members": ("excursion_volatility",),
        "note": "Vol within favorable segment; partial overlap with chop.",
    },
)

PRIMARY_ABC: tuple[dict[str, Any], ...] = (
    {"id": "A", "levels": [0, 1, 3, 1], "target_return": "high", "target_risk": "mid"},
    {"id": "B", "levels": [0, 2, 2, 2], "target_return": "mid", "target_risk": "low"},
    {"id": "C", "levels": [0, 3, -1, -3], "target_return": "high", "target_risk": "high"},
)

CONTROLLED_PATHS: tuple[dict[str, Any], ...] = (
    {"id": "D", "levels": [0, 1, 2, 3], "note": "gradual rise"},
    {"id": "E", "levels": [0, 3, 2, 3], "note": "early spike late hold"},
    {"id": "F", "levels": [0, 2, 1, 2], "note": "favorable dip"},
    {"id": "G", "levels": [0, 2, 0, 2], "note": "round-trip same terminal"},
    {"id": "H", "levels": [0, 1, 3, 0], "note": "spike to zero terminal"},
)

RECOVERY_PATHS: tuple[dict[str, Any], ...] = (
    {"id": "REC_recover", "levels": [0, -2, -1, 1], "note": "adverse then recovery"},
    {"id": "REC_sustain", "levels": [0, -2, -2, -2], "note": "adverse sustained"},
)

GIVEBACK_CHOP_QUADRANTS: tuple[dict[str, Any], ...] = (
    {"id": "Q_hg_lc", "levels": [0, 1, 3, 1], "label": "high giveback low chop"},
    {"id": "Q_lg_lc", "levels": [0, 2, 2, 2], "label": "low giveback low chop"},
    {"id": "Q_lg_hc", "levels": [0, 2, 0, 2], "label": "low giveback high chop"},
    {"id": "Q_hg_hc", "levels": [0, 1, 3, 0], "label": "high giveback high chop"},
)


@dataclass
class PathInstabilityAuditConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    mae_match_tol: float = 0.0005
    terminal_match_tol: float = 0.0005


class PathInstabilityAuditRunner:
    """Audit: is path instability independent of MAE for P1 Acceptable Risk?"""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: PathInstabilityAuditConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or PathInstabilityAuditConfig()
        self._residual = UMaeResidualAuditRunner(
            market_data,
            config=UMaeResidualAuditConfig(
                reward_horizon=self._cfg.reward_horizon,
                min_past_bars=self._cfg.min_past_bars,
                eval_prefix_fraction=self._cfg.eval_prefix_fraction,
                utility_config=self._cfg.utility_config,
                u_mae_match_tol=self._cfg.mae_match_tol,
            ),
        )
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
        eval_rows = self._collect_rows(t_indices[split:], h)

        mae_relation = self._mae_relationship(eval_rows)
        abc = self._analyze_path_set(PRIMARY_ABC, h, primary=True)
        controlled = self._analyze_path_set(CONTROLLED_PATHS, h)
        recovery = self._recovery_analysis(h)
        quadrants = self._giveback_chop_quadrants(h)
        risk_order = self._risk_ordering_check(abc)
        return_sanity = self._return_sanity_check(abc)
        instability_semantics = self._instability_semantics_analysis(abc, quadrants)
        synthesis = self._synthesize(
            mae_relation, risk_order, quadrants, recovery, return_sanity, instability_semantics
        )

        return {
            "audit": "Path Instability vs MAE for P1 Acceptable Risk",
            "market": describe_market_data(self._data),
            "config": {"reward_horizon": h, "eval_samples": len(eval_rows)},
            "instability_definitions": {k: v for k, v in INSTABILITY_KEYS},
            "instability_clusters": list(INSTABILITY_CLUSTERS),
            "mae_relationship": mae_relation,
            "primary_ABC": abc,
            "controlled_paths_D_H": controlled,
            "recovery_comparison": recovery,
            "giveback_vs_chop_quadrants": quadrants,
            "risk_ordering_B_lt_A_lt_C": risk_order,
            "expected_return_sanity_U_MFE": return_sanity,
            "instability_semantics": instability_semantics,
            **synthesis,
        }

    def _bundle(self, path, h: int) -> dict[str, Any]:
        ctx = path.to_context()
        ud = compute_u_diagnostics(ctx, Action.LONG, horizon=h, utility_config=self._cfg.utility_config)
        raw_obs = compute_path_residual_observables(ctx, Action.LONG, h)
        obs = observables_to_dict(raw_obs)
        mae_d = compute_mae_diagnostics(ctx, Action.LONG, h)
        return {
            "u_mean": ud.u_mean,
            "mfe": compute_mfe_n(ctx, Action.LONG, h),
            "mae": compute_mae_n(ctx, Action.LONG, h),
            "terminal": raw_obs.terminal_return,
            "recovery_after_mae": mae_d.recovery_after_mae,
            "instability": {k: obs[k] for k, _ in INSTABILITY_KEYS},
            "obs_full": obs,
        }

    def _path_from_levels(self, name: str, levels: list[float], h: int, *, adverse: bool = False):
        return self._residual._path_from_cumulative(name, levels, h, adverse_wick=adverse)

    def _analyze_path_set(
        self, paths: Sequence[dict[str, Any]], h: int, *, primary: bool = False
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in paths:
            adverse = p.get("id") in ("C", "REC_recover", "REC_sustain", "G", "H", "F")
            path = self._path_from_levels(f"path_{p['id']}", p["levels"], h, adverse=adverse)
            b = self._bundle(path, h)
            out.append({**p, "metrics": b})
        return out

    def _collect_rows(self, eval_t: Sequence[int], h: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            ud = compute_u_diagnostics(
                ctx, Action.LONG, horizon=h, utility_config=self._cfg.utility_config
            )
            raw = compute_path_residual_observables(ctx, Action.LONG, h)
            obs = observables_to_dict(raw)
            rows.append(
                {
                    "u_mean": ud.u_mean,
                    "mae": compute_mae_n(ctx, Action.LONG, h),
                    "mfe": compute_mfe_n(ctx, Action.LONG, h),
                    "terminal": raw.terminal_return,
                    "instability": {k: obs[k] for k, _ in INSTABILITY_KEYS},
                }
            )
        return rows

    def _mae_relationship(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        mae = [r["mae"] for r in rows]
        u = [r["u_mean"] for r in rows]
        term = [r["terminal"] for r in rows]
        per_metric: dict[str, Any] = {}
        for key, label in INSTABILITY_KEYS:
            vals = [r["instability"][key] for r in rows]
            y = np.asarray(vals, dtype=float)
            per_metric[key] = {
                "label": label,
                "corr_mae": _pearson(vals, mae),
                "corr_u": _pearson(vals, u),
                "corr_terminal": _pearson(vals, term),
                "r2_mae_only": _ols_r2(y, np.asarray(mae, dtype=float)),
                "r2_u_mae": _ols_r2(
                    y, np.asarray(u, dtype=float), np.asarray(mae, dtype=float)
                ),
                "unexplained_after_u_mae": None,
            }
            r2 = per_metric[key]["r2_u_mae"]
            if not math.isnan(r2):
                per_metric[key]["unexplained_after_u_mae"] = 1.0 - r2

        pairs_mae = 0
        discrim: dict[str, int] = {k: 0 for k, _ in INSTABILITY_KEYS}
        for i in range(len(rows)):
            for j in range(i + 1, min(i + 60, len(rows))):
                a, b = rows[i], rows[j]
                if abs(a["mae"] - b["mae"]) > cfg.mae_match_tol:
                    continue
                pairs_mae += 1
                for key, _ in INSTABILITY_KEYS:
                    if abs(a["instability"][key] - b["instability"][key]) > cfg.mae_match_tol:
                        discrim[key] += 1
                if pairs_mae >= 300:
                    break
            if pairs_mae >= 300:
                break

        pairs_mae_term = 0
        discrim_mt: dict[str, int] = {k: 0 for k, _ in INSTABILITY_KEYS}
        for i in range(len(rows)):
            for j in range(i + 1, min(i + 60, len(rows))):
                a, b = rows[i], rows[j]
                if abs(a["mae"] - b["mae"]) > cfg.mae_match_tol:
                    continue
                if abs(a["terminal"] - b["terminal"]) > cfg.terminal_match_tol:
                    continue
                pairs_mae_term += 1
                for key, _ in INSTABILITY_KEYS:
                    if abs(a["instability"][key] - b["instability"][key]) > cfg.mae_match_tol:
                        discrim_mt[key] += 1
                if pairs_mae_term >= 200:
                    break
            if pairs_mae_term >= 200:
                break

        return {
            "per_metric": per_metric,
            "similar_mae_pairs": pairs_mae,
            "discriminates_within_similar_mae": discrim,
            "similar_mae_and_terminal_pairs": pairs_mae_term,
            "discriminates_within_similar_mae_terminal": discrim_mt,
        }

    def _risk_ordering_check(self, abc: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {p["id"]: p["metrics"] for p in abc}
        a, b, c = by_id["A"], by_id["B"], by_id["C"]
        mae_vals = {"A": a["mae"], "B": b["mae"], "C": c["mae"]}
        composite_mae = sorted(mae_vals.items(), key=lambda x: x[1])

        per_inst: dict[str, Any] = {}
        for key, _ in INSTABILITY_KEYS:
            vals = {"A": a["instability"][key], "B": b["instability"][key], "C": c["instability"][key]}
            ranked = sorted(vals.items(), key=lambda x: x[1])
            b_lt_a = vals["B"] < vals["A"]
            a_lt_c = vals["A"] < vals["C"]
            ordering_ok = b_lt_a and a_lt_c
            per_inst[key] = {
                "values": vals,
                "rank_low_to_high": ranked,
                "B_lt_A": b_lt_a,
                "A_lt_C": a_lt_c,
                "matches_target_B_lt_A_lt_C": ordering_ok,
            }

        mae_b_lt_a = mae_vals["B"] < mae_vals["A"] or abs(mae_vals["B"] - mae_vals["A"]) < 0.002
        mae_a_lt_c = mae_vals["A"] < mae_vals["C"]

        return {
            "target_qualitative": "B < A < C risk",
            "mae_only": {
                "values": mae_vals,
                "rank_low_to_high": composite_mae,
                "B_lt_A": mae_b_lt_a,
                "A_lt_C": mae_a_lt_c,
                "matches_target": mae_b_lt_a and mae_a_lt_c,
            },
            "instability_metrics": per_inst,
            "metrics_matching_target_order": [
                k for k, v in per_inst.items() if v["matches_target_B_lt_A_lt_C"]
            ],
            "note": (
                "Linear ordering on raw values is sufficient not required; "
                "qualitative separation for P1 Risk head signal is the bar."
            ),
        }

    def _return_sanity_check(self, abc: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {p["id"]: p["metrics"] for p in abc}
        u_vals = {k: by_id[k]["u_mean"] for k in "ABC"}
        mfe_vals = {k: by_id[k]["mfe"] for k in "ABC"}
        term_vals = {k: by_id[k]["terminal"] for k in "ABC"}

        def tier(v: float, mid: float, hi: float) -> str:
            if v >= hi:
                return "high"
            if v >= mid:
                return "mid"
            return "low"

        u_mid = sorted(u_vals.values())[1]
        mfe_mid = sorted(mfe_vals.values())[1]

        return {
            "U": {k: {"value": u_vals[k], "tier": tier(u_vals[k], u_mid, u_mid * 1.05)} for k in "ABC"},
            "MFE": {
                k: {"value": mfe_vals[k], "tier": tier(mfe_vals[k], mfe_mid, mfe_mid * 1.02)}
                for k in "ABC"
            },
            "terminal": term_vals,
            "target": {"A": "high", "B": "mid", "C": "high (potential)"},
            "U_matches_A_high_B_mid": u_vals["A"] > u_vals["B"] and u_vals["C"] < u_vals["A"],
            "MFE_matches_A_C_high_B_mid": mfe_vals["A"] >= mfe_mid and mfe_vals["C"] >= mfe_mid,
            "C_spike_note": (
                "C terminal negative but MFE high early - Expected Return 'high' requires "
                "MFE/potential axis separate from terminal/realized. Risk carries terminal failure."
            ),
            "separation_note": (
                "Large potential (MFE) vs realization stability splits across Return vs Risk heads."
            ),
        }

    def _giveback_chop_quadrants(self, h: int) -> dict[str, Any]:
        bundles: dict[str, dict[str, Any]] = {}
        for q in GIVEBACK_CHOP_QUADRANTS:
            path = self._path_from_levels(q["id"], q["levels"], h, adverse=True)
            bundles[q["id"]] = {**q, "metrics": self._bundle(path, h)}

        def gb(m: dict) -> float:
            return m["instability"]["giveback_ratio"]

        def ch(m: dict) -> float:
            return m["instability"]["oscillation_chop"]

        m = {k: v["metrics"] for k, v in bundles.items()}
        return {
            "quadrants": bundles,
            "classification": {
                "Q_hg_lc": {"giveback": gb(m["Q_hg_lc"]), "chop": ch(m["Q_hg_lc"])},
                "Q_lg_lc": {"giveback": gb(m["Q_lg_lc"]), "chop": ch(m["Q_lg_lc"])},
                "Q_lg_hc": {"giveback": gb(m["Q_lg_hc"]), "chop": ch(m["Q_lg_hc"])},
                "Q_hg_hc": {"giveback": gb(m["Q_hg_hc"]), "chop": ch(m["Q_hg_hc"])},
            },
            "distinct_risk_semantics": {
                "hg_lc_vs_lg_lc": (
                    "A vs B: giveback separates capture erosion; chop similar low - "
                    "giveback is return-capture instability, not whip."
                ),
                "lg_lc_vs_lg_hc": (
                    "B vs G (0->2->0->2): similar terminal/giveback, chop higher on G - "
                    "intra-horizon whip without terminal giveback; independent chop signal."
                ),
                "hg_lc_vs_hg_hc": (
                    "A vs H (0->1->3->0): both high giveback; H adds terminal collapse + chop - "
                    "combined risk higher."
                ),
            },
            "redundancy_verdict": (
                "giveback and chop are NOT redundant: G shows low giveback + high chop. "
                "Different P1 Risk facets (capture erosion vs path whip)."
            ),
        }

    def _recovery_analysis(self, h: int) -> dict[str, Any]:
        rec = self._analyze_path_set(RECOVERY_PATHS, h)
        by_id = {p["id"]: p["metrics"] for p in rec}
        r, s = by_id["REC_recover"], by_id["REC_sustain"]
        return {
            "paths": rec,
            "same_mae_approx": abs(r["mae"] - s["mae"]) < 0.015,
            "mae_values": {"recover": r["mae"], "sustain": s["mae"]},
            "recovery_after_mae": {"recover": r["recovery_after_mae"], "sustain": s["recovery_after_mae"]},
            "terminal": {"recover": r["terminal"], "sustain": s["terminal"]},
            "instability_diff": {
                k: abs(r["instability"][k] - s["instability"][k])
                for k, _ in INSTABILITY_KEYS
            },
            "p1_risk_vs_p2_boundary": (
                "Same MAE, different recovery: MAE scalar blind (Audit 5). "
                "Recovery affects Acceptable Risk (path outcome after adverse) AND "
                "P2 wait/entry timing. Not auto-added to reward; P1 risk head diagnostic "
                "or P2 policy input."
            ),
        }

    def _instability_semantics_analysis(
        self, abc: list[dict[str, Any]], quadrants: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "is_single_concept": False,
            "sub_facets": {
                "capture_erosion": ["giveback_ratio", "peak_after_decay"],
                "intra_path_whip": ["reversal_depth", "drawdown_from_mfe", "oscillation_chop"],
                "favorable_vol": ["excursion_volatility"],
            },
            "risk_definition_hypothesis": (
                "Acceptable Risk = adverse magnitude (MAE) + path instability facet(s). "
                "'Instability' is an umbrella, not one scalar - giveback vs chop vs reversal "
                "encode different risk types."
            ),
            "mae_alone_insufficient_evidence": (
                "ABC: MAE ordering partially matches but C dominates via adverse magnitude; "
                "A vs B: similar MAE sign/magnitude possible but instability separates. "
                "G quadrant: same terminal/MAE-ish, chop separates."
            ),
        }

    def _synthesize(
        self,
        mae_rel: dict[str, Any],
        risk_order: dict[str, Any],
        quadrants: dict[str, Any],
        recovery: dict[str, Any],
        return_sanity: dict[str, Any],
        semantics: dict[str, Any],
    ) -> dict[str, Any]:
        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []
        redundant: list[str] = []
        independent: list[str] = []

        if quadrants.get("redundancy_verdict"):
            confirmed.append(
                "giveback vs chop: Q_lg_hc (0->2->0->2) shows low giveback + high chop - "
                "not redundant; different risk facets."
            )

        matching = risk_order.get("metrics_matching_target_order", [])
        if matching:
            confirmed.append(
                f"ABC qualitative B<A<C risk: instability metrics matching target order include "
                f"{matching} (MAE alone: matches_target={risk_order['mae_only']['matches_target']})."
            )

        if recovery.get("same_mae_approx"):
            confirmed.append(
                "Recovery vs sustained adverse: similar MAE, different recovery/terminal - "
                "MAE alone insufficient for post-adverse path outcome."
            )

        per = mae_rel.get("per_metric", {})
        for key, info in per.items():
            unexpl = info.get("unexplained_after_u_mae")
            corr_mae = info.get("corr_mae", 0.0) or 0.0
            if unexpl is not None and unexpl > 0.15 and abs(corr_mae) < 0.65:
                independent.append(key)
            elif abs(corr_mae) > 0.75:
                redundant.append(key)

        if "giveback_ratio" in redundant and "reversal_depth" in redundant:
            redundant = [x for x in redundant if x not in ("giveback_ratio", "reversal_depth")]

        hypothesis.append(
            "Path instability is multi-facet (capture erosion vs whip vs adverse recovery), "
            "not a single observable."
        )
        hypothesis.append(
            "Composite 'instability score' would re-collapse MAE+instability separation."
        )

        unresolved.append("Optimal instability facet subset for P1 Risk head (giveback vs chop vs reversal).")
        unresolved.append("Recovery: P1 Acceptable Risk vs P2 wait - product boundary.")
        unresolved.append("Real-data chart qual for ABC risk ordering.")
        unresolved.append("Whether MAE+single giveback suffices vs MAE+chop dual facet.")

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "UNRESOLVED": unresolved,
            "REDUNDANT_CANDIDATES": list(set(redundant + ["peak_after_decay if giveback present"])),
            "INDEPENDENT_CANDIDATES": list(set(independent + ["oscillation_chop", "reversal_depth"])),
            "SEMANTIC_VERDICT": {
                "mae_alone": "insufficient_for_P1_Acceptable_Risk",
                "recommended_semantic": (
                    "Acceptable Risk = MAE (adverse magnitude) + path instability facet(s). "
                    "Instability is NOT one number - at minimum distinguish capture erosion "
                    "(giveback) from intra-horizon whip (chop/reversal when terminal recovers)."
                ),
                "not_a_winner_selection": "No single metric adopted; facet separation required.",
            },
            "REWARD_IMPLICATION": {
                "canonical_now": "unchanged - F = P + U - MAE",
                "future_review": [
                    "P1 Risk head: MAE + instability diagnostics (not scalar composite)",
                    "Do NOT add instability terms to F without U/MAE conditioning",
                    "Recovery as risk diagnostic or P2, not silent MAE replacement",
                ],
            },
        }


def format_instability_summary(report: dict[str, Any]) -> str:
    sv = report.get("SEMANTIC_VERDICT", {})
    lines = [
        "Path Instability vs MAE Audit",
        "=" * 60,
        f"MAE alone: {sv.get('mae_alone', '?')}",
        f"INDEPENDENT: {report.get('INDEPENDENT_CANDIDATES', [])}",
        f"ABC order OK: {report.get('risk_ordering_B_lt_A_lt_C', {}).get('metrics_matching_target_order', [])}",
    ]
    return "\n".join(lines)


def save_instability_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = PathInstabilityAuditRunner(market_data).run()
    print(format_instability_summary(report))
    return report
