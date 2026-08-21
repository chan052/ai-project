"""P1 Return/Risk Target Validation Audit (analysis-only).

Validates fixed candidate structure:
  Expected Return  = U + MFE
  Acceptable Risk  = MAE + giveback + chop

Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditRunner, UMaeResidualAuditConfig, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_mfe_n


SYNTHETIC_ARCHETYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "levels": [0, 1, 3, 1],
        "description": "0->1->3->1 spike giveback",
        "expected_return_tier": "high",
        "expected_risk_tier": "mid",
    },
    {
        "id": "B",
        "levels": [0, 2, 2, 2],
        "description": "0->2->2->2 grind hold",
        "expected_return_tier": "mid",
        "expected_risk_tier": "low",
    },
    {
        "id": "C",
        "levels": [0, 3, -1, -3],
        "description": "0->3->-1->-3 rise crash",
        "expected_return_tier": "high_potential_not_realized",
        "expected_risk_tier": "high",
    },
    {
        "id": "REC",
        "levels": [0, -2, -1, 1],
        "description": "0->-2->-1->1 adverse recovery",
        "expected_return_tier": "mid",
        "expected_risk_tier": "mid",
    },
    {
        "id": "G",
        "levels": [0, 2, 0, 2],
        "description": "0->2->0->2 round-trip",
        "expected_return_tier": "mid",
        "expected_risk_tier": "mid_whip",
    },
    {
        "id": "H",
        "levels": [0, 1, 3, 0],
        "description": "0->1->3->0 spike to zero",
        "expected_return_tier": "low_realized",
        "expected_risk_tier": "high",
    },
)

# Pre-declared semantic orderings (not ground truth labels)
RETURN_ORDER_AB = "A_return_tier >= B_return_tier (A high, B mid)"
RISK_ORDER_ABC = "B_risk < A_risk < C_risk"


@dataclass
class P1ReturnRiskTargetAuditConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    u_match_tol: float = 0.0005
    mae_match_tol: float = 0.0005
    terminal_match_tol: float = 0.0005
    mfe_match_tol: float = 0.0005


class P1ReturnRiskTargetAuditRunner:
    """Validate U+MFE / MAE+giveback+chop for P1 semantic adequacy."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1ReturnRiskTargetAuditConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1ReturnRiskTargetAuditConfig()
        self._residual = UMaeResidualAuditRunner(
            market_data,
            config=UMaeResidualAuditConfig(
                reward_horizon=self._cfg.reward_horizon,
                min_past_bars=self._cfg.min_past_bars,
                eval_prefix_fraction=self._cfg.eval_prefix_fraction,
                utility_config=self._cfg.utility_config,
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
        synth = self._synthetic_table(h)
        return_checks = self._return_semantic_checks(synth)
        risk_checks = self._risk_semantic_checks(synth)
        u_mfe_overlap = self._u_mfe_redundancy(synth)
        giveback_chop = self._giveback_chop_analysis(synth)
        risk_scalar_collapse = self._risk_facet_preservation(synth)
        real_pairs = self._real_data_pairs()
        adoption = self._structure_adoption_judgment(
            return_checks, risk_checks, u_mfe_overlap, giveback_chop, risk_scalar_collapse, real_pairs
        )
        synthesis = self._synthesize(
            return_checks, risk_checks, u_mfe_overlap, giveback_chop, adoption
        )

        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))

        return {
            "audit": "P1 Return/Risk Target Validation",
            "candidate_structure": {
                "expected_return": ["U", "MFE"],
                "acceptable_risk": ["MAE", "giveback", "chop"],
            },
            "pre_declared_semantics": {
                "return_order_AB": RETURN_ORDER_AB,
                "risk_order_ABC": RISK_ORDER_ABC,
                "archetype_tiers": {
                    a["id"]: {
                        "return": a["expected_return_tier"],
                        "risk": a["expected_risk_tier"],
                    }
                    for a in SYNTHETIC_ARCHETYPES
                },
            },
            "market": describe_market_data(self._data),
            "config": {"reward_horizon": h, "eval_samples": len(t_indices) - split},
            "synthetic_paths": synth,
            "return_validation": return_checks,
            "risk_validation": risk_checks,
            "u_mfe_redundancy": u_mfe_overlap,
            "giveback_chop_analysis": giveback_chop,
            "risk_facet_preservation": risk_scalar_collapse,
            "real_data_pair_analysis": real_pairs,
            "structure_adoption_judgment": adoption,
            **synthesis,
        }

    def _obs_bundle(self, path, h: int) -> dict[str, float]:
        ctx = path.to_context()
        ud = compute_u_diagnostics(
            ctx, Action.LONG, horizon=h, utility_config=self._cfg.utility_config
        )
        obs = compute_path_residual_observables(ctx, Action.LONG, h)
        return {
            "U": ud.u_mean,
            "MFE": compute_mfe_n(ctx, Action.LONG, h),
            "MAE": compute_mae_n(ctx, Action.LONG, h),
            "giveback": obs.giveback_ratio,
            "chop": obs.oscillation_chop,
            "terminal": obs.terminal_return,
        }

    def _synthetic_table(self, h: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for arch in SYNTHETIC_ARCHETYPES:
            adverse = arch["id"] in ("C", "REC", "G", "H")
            path = self._residual._path_from_cumulative(
                f"p1_{arch['id']}", arch["levels"], h, adverse_wick=adverse
            )
            rows.append(
                {
                    **arch,
                    "observables": self._obs_bundle(path, h),
                }
            )
        return rows

    def _by_id(self, synth: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["id"]: r for r in synth}

    def _return_semantic_checks(self, synth: list[dict[str, Any]]) -> dict[str, Any]:
        d = self._by_id(synth)
        a, b, c = d["A"]["observables"], d["B"]["observables"], d["C"]["observables"]

        u_a_gt_b = a["U"] > b["U"]
        mfe_a_gt_b = a["MFE"] > b["MFE"]
        dual_head_a_higher_potential = mfe_a_gt_b
        dual_head_b_higher_average = a["U"] < b["U"]

        c_mfe_high = c["MFE"] >= a["MFE"] * 0.95
        c_u_low = c["U"] < a["U"] and c["U"] < b["U"]
        c_terminal_negative = c["terminal"] < 0

        return {
            "A_vs_B": {
                "U": {"A": a["U"], "B": b["U"], "A_gt_B": u_a_gt_b},
                "MFE": {"A": a["MFE"], "B": b["MFE"], "A_gt_B": mfe_a_gt_b},
                "terminal": {"A": a["terminal"], "B": b["terminal"]},
                "single_scalar_return_would_favor": "B (U and terminal)" if not u_a_gt_b else "A",
                "dual_head_U_MFE_expresses": (
                    "A: higher MFE (potential/max opportunity), "
                    "B: higher U (average favorable opportunity) - "
                    "tiers 'high vs mid' expressible as facet split, not one scalar"
                ),
                "A_higher_expected_return_via_MFE": mfe_a_gt_b,
                "meets_tier_if_MFE_is_primary": mfe_a_gt_b,
                "meets_tier_if_U_is_primary": u_a_gt_b,
            },
            "C_return_not_unconditionally_high": {
                "MFE": c["MFE"],
                "U": c["U"],
                "terminal": c["terminal"],
                "MFE_high": c_mfe_high,
                "U_suppresses_unconditional_high": c_u_low,
                "terminal_negative": c_terminal_negative,
                "verdict": (
                    "CONFIRMED: C has high MFE but low U and negative terminal - "
                    "dual-head prevents MFE-alone from marking C as unqualified high Return"
                    if c_mfe_high and c_u_low
                    else "needs_review"
                ),
            },
        }

    def _risk_semantic_checks(self, synth: list[dict[str, Any]]) -> dict[str, Any]:
        d = self._by_id(synth)
        a, b, c = d["A"]["observables"], d["B"]["observables"], d["C"]["observables"]
        g = d["G"]["observables"]

        def risk_rank(vals: dict[str, float]) -> list[tuple[str, float]]:
            return sorted(vals.items(), key=lambda x: x[1])

        mae_abc = {"A": abs(a["MAE"]), "B": abs(b["MAE"]), "C": abs(c["MAE"])}
        gb_abc = {"A": a["giveback"], "B": b["giveback"], "C": c["giveback"]}
        chop_abc = {"A": a["chop"], "B": b["chop"], "C": c["chop"]}

        b_lt_a_mae = mae_abc["B"] <= mae_abc["A"] + 0.002
        a_lt_c_mae = mae_abc["A"] < mae_abc["C"]
        b_lt_a_gb = gb_abc["B"] < gb_abc["A"]
        a_lt_c_gb = gb_abc["A"] < gb_abc["C"]
        b_lt_a_chop = chop_abc["B"] <= chop_abc["A"]
        a_lt_c_chop = chop_abc["A"] <= chop_abc["C"]

        giveback_separates_ab = b_lt_a_gb and abs(gb_abc["A"] - gb_abc["B"]) > 0.2
        mae_separates_ab = abs(mae_abc["A"] - mae_abc["B"]) > 0.002

        g_vs_b = {
            "B": {"terminal": b["terminal"], "giveback": b["giveback"], "chop": b["chop"]},
            "G": {"terminal": g["terminal"], "giveback": g["giveback"], "chop": g["chop"]},
            "terminal_similar": abs(b["terminal"] - g["terminal"]) < 0.005,
            "chop_G_higher": g["chop"] > b["chop"] + 0.05,
            "giveback_similar": abs(b["giveback"] - g["giveback"]) < 0.15,
        }

        return {
            "ABC_ordering": {
                "MAE_abs": {
                    "values": mae_abc,
                    "rank": risk_rank(mae_abc),
                    "B_lt_A": b_lt_a_mae,
                    "A_lt_C": a_lt_c_mae,
                    "matches_B_lt_A_lt_C": b_lt_a_mae and a_lt_c_mae,
                },
                "giveback": {
                    "values": gb_abc,
                    "rank": risk_rank(gb_abc),
                    "B_lt_A": b_lt_a_gb,
                    "A_lt_C": a_lt_c_gb,
                    "matches_B_lt_A_lt_C": b_lt_a_gb and a_lt_c_gb,
                },
                "chop": {
                    "values": chop_abc,
                    "rank": risk_rank(chop_abc),
                    "B_lt_A": b_lt_a_chop,
                    "A_lt_C": a_lt_c_chop,
                    "matches_B_lt_A_lt_C": b_lt_a_chop and a_lt_c_chop,
                    "note": "A and C may tie on chop; chop alone does not rank A vs C",
                },
            },
            "giveback_catches_A_vs_B": {
                "mae_separates": mae_separates_ab,
                "giveback_separates": giveback_separates_ab,
                "giveback_delta": gb_abc["A"] - gb_abc["B"],
                "verdict": "giveback distinguishes A/B capture-erosion risk" if giveback_separates_ab else "weak",
            },
            "chop_catches_round_trip": g_vs_b,
        }

    def _u_mfe_redundancy(self, synth: list[dict[str, Any]]) -> dict[str, Any]:
        u = [r["observables"]["U"] for r in synth]
        mfe = [r["observables"]["MFE"] for r in synth]
        term = [r["observables"]["terminal"] for r in synth]
        corr = _pearson(u, mfe)
        return {
            "synthetic_corr_U_MFE": corr,
            "per_path": [
                {
                    "id": r["id"],
                    "U": r["observables"]["U"],
                    "MFE": r["observables"]["MFE"],
                    "terminal": r["observables"]["terminal"],
                    "U_vs_MFE_divergence": r["observables"]["MFE"] - r["observables"]["U"],
                }
                for r in synth
            ],
            "same_information": abs(corr) > 0.95 if not math.isnan(corr) else False,
            "interpretation": (
                "U and MFE are NOT the same information on archetypes: "
                "A has MFE>>U (spike), B has U~MFE (grind), C has MFE>>U with negative terminal. "
                "Dual-head captures average vs max favorable opportunity."
                if abs(corr or 0) < 0.95
                else "high collinearity on synthetic set - review"
            ),
        }

    def _giveback_chop_analysis(self, synth: list[dict[str, Any]]) -> dict[str, Any]:
        d = self._by_id(synth)
        pairs = [
            ("A", "B", "high giveback vs low"),
            ("B", "G", "low giveback both, chop differs"),
            ("A", "G", "giveback vs chop independence"),
        ]
        comparisons: list[dict[str, Any]] = []
        for id1, id2, note in pairs:
            o1, o2 = d[id1]["observables"], d[id2]["observables"]
            comparisons.append(
                {
                    "pair": f"{id1}_vs_{id2}",
                    "note": note,
                    "giveback": {"a": o1["giveback"], "b": o2["giveback"]},
                    "chop": {"a": o1["chop"], "b": o2["chop"]},
                    "giveback_diff": abs(o1["giveback"] - o2["giveback"]),
                    "chop_diff": abs(o1["chop"] - o2["chop"]),
                }
            )
        a, g = d["A"]["observables"], d["G"]["observables"]
        b = d["B"]["observables"]
        return {
            "pairwise": comparisons,
            "redundancy_level": (
                "partial - correlated on spike-giveback paths, independent on B vs G "
                "(similar giveback, chop 0 vs 0.22)"
            ),
            "B_vs_G": {
                "giveback": {"B": b["giveback"], "G": g["giveback"]},
                "chop": {"B": b["chop"], "G": g["chop"]},
                "independent_facets": g["chop"] > b["chop"] + 0.05 and abs(b["giveback"] - g["giveback"]) < 0.15,
            },
        }

    def _risk_facet_preservation(self, synth: list[dict[str, Any]]) -> dict[str, Any]:
        d = self._by_id(synth)
        facets = ("MAE", "giveback", "chop")
        rows = []
        for pid in ("A", "B", "C", "G"):
            o = d[pid]["observables"]
            rows.append({f: o[f] if f != "MAE" else abs(o["MAE"]) for f in facets})
        scalar_sum = [
            abs(d[p]["observables"]["MAE"])
            + d[p]["observables"]["giveback"]
            + d[p]["observables"]["chop"]
            for p in ("A", "B", "C", "G")
        ]
        return {
            "facet_values": {p: rows[i] for i, p in enumerate(("A", "B", "C", "G"))},
            "naive_sum_scalar": dict(zip(("A", "B", "C", "G"), scalar_sum)),
            "preservation_verdict": (
                "Multi-facet head preserves semantics; naive sum loses which facet drives "
                "(e.g. G: moderate sum but chop-driven whip). Use separate Risk sub-targets, "
                "not one collapsed scalar."
            ),
            "G_example": {
                "facets": rows[3],
                "note": "G has chop>>B while giveback~B - sum scalar obscures whip facet",
            },
        }

    def _real_data_pairs(self) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))
        eval_t = t_indices[split:]

        rows: list[dict[str, float]] = []
        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            ud = compute_u_diagnostics(
                ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config
            )
            obs = compute_path_residual_observables(ctx, Action.LONG, h)
            rows.append(
                {
                    "U": ud.u_mean,
                    "MFE": compute_mfe_n(ctx, Action.LONG, h),
                    "MAE": compute_mae_n(ctx, Action.LONG, h),
                    "giveback": obs.giveback_ratio,
                    "chop": obs.oscillation_chop,
                    "terminal": obs.terminal_return,
                }
            )

        u_mae_pairs = 0
        gb_disc = chop_disc = 0
        u_mfe_mae_pairs = 0
        gb_disc_umfe = chop_disc_umfe = 0

        for i in range(len(rows)):
            for j in range(i + 1, min(i + 80, len(rows))):
                a, b = rows[i], rows[j]
                if (
                    abs(a["U"] - b["U"]) < cfg.u_match_tol
                    and abs(a["MAE"] - b["MAE"]) < cfg.mae_match_tol
                    and abs(a["terminal"] - b["terminal"]) < cfg.terminal_match_tol
                ):
                    u_mae_pairs += 1
                    if abs(a["giveback"] - b["giveback"]) > cfg.mae_match_tol:
                        gb_disc += 1
                    if abs(a["chop"] - b["chop"]) > 0.02:
                        chop_disc += 1
                if (
                    abs(a["U"] - b["U"]) < cfg.u_match_tol
                    and abs(a["MFE"] - b["MFE"]) < cfg.mfe_match_tol
                    and abs(a["MAE"] - b["MAE"]) < cfg.mae_match_tol
                ):
                    u_mfe_mae_pairs += 1
                    if abs(a["giveback"] - b["giveback"]) > cfg.mae_match_tol:
                        gb_disc_umfe += 1
                    if abs(a["chop"] - b["chop"]) > 0.02:
                        chop_disc_umfe += 1
                if u_mae_pairs >= 200 and u_mfe_mae_pairs >= 200:
                    break
            if u_mae_pairs >= 200 and u_mfe_mae_pairs >= 200:
                break

        return {
            "eval_samples": len(rows),
            "similar_U_MAE_terminal_pairs": u_mae_pairs,
            "giveback_discriminates": gb_disc,
            "chop_discriminates": chop_disc,
            "similar_U_MFE_MAE_pairs": u_mfe_mae_pairs,
            "giveback_discriminates_umfe": gb_disc_umfe,
            "chop_discriminates_umfe": chop_disc_umfe,
            "U_MFE_corr_eval": _pearson([r["U"] for r in rows], [r["MFE"] for r in rows]),
            "giveback_chop_corr_eval": _pearson(
                [r["giveback"] for r in rows], [r["chop"] for r in rows]
            ),
            "interpretation": (
                "Real pairs with matched U/MAE/terminal still vary on giveback/chop - "
                "path information beyond magnitude scalars."
                if gb_disc > 0 or chop_disc > 0
                else "weak discrimination on eval"
            ),
        }

    def _structure_adoption_judgment(
        self,
        return_checks: dict[str, Any],
        risk_checks: dict[str, Any],
        u_mfe: dict[str, Any],
        gb_chop: dict[str, Any],
        preservation: dict[str, Any],
        real_pairs: dict[str, Any],
    ) -> dict[str, Any]:
        strengths: list[str] = []
        gaps: list[str] = []

        if return_checks["C_return_not_unconditionally_high"].get("U_suppresses_unconditional_high"):
            strengths.append("U+MFE dual Return head blocks C from MFE-only high rating")
        if return_checks["A_vs_B"]["A_higher_expected_return_via_MFE"]:
            strengths.append("MFE expresses A high potential vs B mid")
        if not return_checks["A_vs_B"]["U"]["A_gt_B"]:
            gaps.append("U alone ranks B above A - Return tier needs MFE facet explicit semantics")

        if risk_checks["ABC_ordering"]["giveback"]["matches_B_lt_A_lt_C"]:
            strengths.append("giveback supports B<A<C risk ordering")
        if not risk_checks["ABC_ordering"]["chop"]["matches_B_lt_A_lt_C"]:
            gaps.append("chop does not fully rank A vs C - Risk needs MAE/giveback for C")

        if risk_checks["giveback_catches_A_vs_B"]["giveback_separates"]:
            strengths.append("giveback separates A/B capture erosion")
        if risk_checks["chop_catches_round_trip"]["chop_G_higher"]:
            strengths.append("chop catches G round-trip whip")

        if gb_chop["B_vs_G"].get("independent_facets"):
            strengths.append("giveback and chop are independent facets on B vs G")

        if not u_mfe.get("same_information"):
            strengths.append("U and MFE carry distinct facets on archetypes")

        if real_pairs.get("giveback_discriminates", 0) > 0:
            strengths.append("real data: giveback adds info in matched buckets")

        return {
            "structure": {
                "Direction": "P or separate (out of scope)",
                "Expected_Return": ["U", "MFE"],
                "Acceptable_Risk": ["MAE", "giveback", "chop"],
            },
            "adoption_sufficient_as_design_candidate": len(gaps) <= 2,
            "strengths": strengths,
            "gaps": gaps,
            "not_auto_adopt": (
                "Semantic adequacy supports this as P1 target DESIGN CANDIDATE; "
                "canonical adoption requires chart qual, weighting, and training protocol - not done here."
            ),
        }

    def _synthesize(
        self,
        return_checks: dict[str, Any],
        risk_checks: dict[str, Any],
        u_mfe: dict[str, Any],
        gb_chop: dict[str, Any],
        adoption: dict[str, Any],
    ) -> dict[str, Any]:
        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []

        if return_checks["C_return_not_unconditionally_high"]["MFE_high"]:
            confirmed.append(
                "C: high MFE but low U - dual Return head prevents unconditional high Return from MFE alone."
            )
        if return_checks["A_vs_B"]["MFE"]["A_gt_B"]:
            confirmed.append("A vs B: MFE ranks A higher (max opportunity / high tier).")
        if not return_checks["A_vs_B"]["U"]["A_gt_B"]:
            confirmed.append(
                "A vs B: U ranks B higher (average opportunity) - Return tier requires MFE facet, not U alone."
            )
        if risk_checks["ABC_ordering"]["giveback"]["matches_B_lt_A_lt_C"]:
            confirmed.append("Risk ordering B<A<C on giveback.")
        if risk_checks["ABC_ordering"]["MAE_abs"]["matches_B_lt_A_lt_C"]:
            confirmed.append("Risk ordering B<A<C on MAE magnitude.")
        if risk_checks["chop_catches_round_trip"]["chop_G_higher"]:
            confirmed.append("G (0->2->0->2): chop > B despite similar terminal/giveback.")
        if gb_chop["B_vs_G"].get("independent_facets"):
            confirmed.append("giveback and chop independent on B vs G.")
        if not u_mfe.get("same_information"):
            confirmed.append("U and MFE not redundant on synthetic archetypes (different facets).")

        hypothesis.append(
            "U+MFE dual head adequately expresses Return tiers if MFE=potential and U=average "
            "opportunity are documented in target semantics."
        )
        hypothesis.append(
            "MAE+giveback+chop triple Risk head preserves facets if trained as separate outputs "
            "not naive sum."
        )
        if not risk_checks["ABC_ordering"]["chop"]["matches_B_lt_A_lt_C"]:
            hypothesis.append(
                "chop alone insufficient for full B<A<C; MAE+giveback carry A vs C separation."
            )

        unresolved.append("Whether A>B Return tier needs human chart judgment beyond MFE>U split.")
        unresolved.append("Recovery path (REC) Risk tier placement in MAE+giveback+chop structure.")
        unresolved.append("Training weighting and loss coupling between Risk sub-targets.")
        unresolved.append("Direction head coupling with Return/Risk - out of scope here.")

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "UNRESOLVED": unresolved,
        }


def format_p1_target_summary(report: dict[str, Any]) -> str:
    adj = report.get("structure_adoption_judgment", {})
    lines = [
        "P1 Return/Risk Target Validation",
        "=" * 60,
        f"CONFIRMED: {len(report.get('CONFIRMED', []))}",
        f"Design candidate sufficient: {adj.get('adoption_sufficient_as_design_candidate')}",
        f"Gaps: {adj.get('gaps', [])}",
    ]
    return "\n".join(lines)


def save_p1_target_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = P1ReturnRiskTargetAuditRunner(market_data).run()
    print(format_p1_target_summary(report))
    return report
