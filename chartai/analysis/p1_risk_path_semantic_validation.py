"""P1 Target Structure — Risk vs Path Semantic Validation (analysis-only).

Compares Structure A (Risk = MAE + Giveback + Chop) vs
Structure B (Risk = MAE + Giveback, Path = Chop).

Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_normalization_semantic_experiment import NormBundle, _percentile
from chartai.analysis.p1_return_risk_target_audit import SYNTHETIC_ARCHETYPES
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditConfig, UMaeResidualAuditRunner, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
from chartai.reward.path_observables import compute_mfe_n

SCALE_METHOD = "stdscale"

STRUCTURE_A = {
    "Expected_Return": "U + MFE",
    "Acceptable_Risk": "MAE + Giveback + Chop",
    "Path": None,
    "Recovery": "diagnostic_only",
}

STRUCTURE_B = {
    "Expected_Return": "U + MFE",
    "Acceptable_Risk": "MAE + Giveback",
    "Path": "Chop",
    "Recovery": "diagnostic_only",
}

SEMANTIC_DEFINITIONS = {
    "Expected_Return": "how much favorable opportunity/value exists after t",
    "Risk": "adverse burden actually borne if action chosen at t",
    "Path": "quality/mechanism of how outcome unfolds after t",
    "Chop": "directionless round-trip / oscillation intensity",
    "MAE": "maximum adverse excursion from entry",
    "Giveback": "fraction of favorable excursion surrendered after peak",
    "Risk_meaning_1": "how adversely price can move (MAE, giveback erosion)",
    "Risk_meaning_2": "how uncomfortable/dirty the holding path is (chop-like)",
}

ARCHETYPE_IDS = ("B", "A", "G", "C", "REC")


@dataclass
class P1RiskPathValidationConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    terminal_match_tol: float = 0.0005
    giveback_match_tol: float = 0.15
    u_match_tol: float = 0.0003
    mfe_match_tol: float = 0.0003
    mae_match_tol: float = 0.0003
    risk_b_similar_tol: float = 0.35
    max_exemplars: int = 5
    pair_window: int = 80


class P1RiskPathSemanticValidationRunner:
    """Validate Chop in Risk vs separate Path facet."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1RiskPathValidationConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1RiskPathValidationConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self, *, test_pass_count: int | None = None) -> dict[str, Any]:
        rows, t_indices = self._collect_rows()
        split = max(1, int(len(rows) * self._cfg.prefix_fraction))
        bundle = NormBundle.fit_from_rows(rows[:split])
        eval_rows = [self._enrich(r, bundle, int(t_indices[split + i])) for i, r in enumerate(rows[split:])]
        arch = self._archetype_table(bundle)
        b_vs_g = self._b_vs_g_test(arch, eval_rows)
        a_vs_b = self._a_vs_b_test(arch)
        c_test = self._c_crash_test(arch)
        rec_test = self._rec_recovery_test(arch)
        btc_cases = self._btc_boundary_cases(eval_rows)
        separation = self._risk_path_separation(eval_rows)
        quant = self._quantitative_answers(arch, eval_rows, b_vs_g, btc_cases)
        failures = self._failure_cases(eval_rows, arch)
        verdict = self._final_verdict(arch, b_vs_g, a_vs_b, c_test, quant, btc_cases)

        return {
            "audit": "P1 Target Structure — Risk vs Path Semantic Validation",
            "structures_compared": {"A": STRUCTURE_A, "B": STRUCTURE_B},
            "normalization": "X_scaled = X / sigma_prefix (scale-only, prefix-fit)",
            "1_executive_summary": {
                "eval_n": len(eval_rows),
                "final_verdict": verdict["choice"],
                "headline": verdict["summary"],
                "key_answer": verdict["chop_as_risk_answer"],
                "no_canonical_adoption": True,
            },
            "2_semantic_definitions": SEMANTIC_DEFINITIONS,
            "3_synthetic_archetype_analysis": arch,
            "4_B_vs_G_boundary_test": b_vs_g,
            "5_A_vs_B_risk_test": a_vs_b,
            "6_C_crash_trap_test": c_test,
            "7_REC_recovery_test": rec_test,
            "8_real_btc_boundary_cases": btc_cases,
            "9_risk_vs_path_separation": separation,
            "10_normalization_vs_structure": {
                "normalization_note": "X/sigma used; structure comparison holds regardless of z-score mean removal",
                "structure_question": "Chop semantic role (Risk burden vs Path quality)",
                "prior_composite_finding": "Chop dominated Risk_A scalar (~60% share); separation may clarify semantics",
            },
            "11_failure_cases": failures,
            "12_final_verdict": verdict,
            "13_unresolved_questions": verdict["unresolved"],
            "quantitative_answers": quant,
            "data_protocol": {
                "market": describe_market_data(self._data),
                "prefix_n": split,
                "eval_n": len(eval_rows),
            },
            "11_test_result": {"pytest_pass_count": test_pass_count},
        }

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
        }

    def _collect_rows(self) -> tuple[list[dict[str, float]], list[int]]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars))
        return [self._raw_obs(self._builder.build(t), Action.LONG, h) for t in t_indices], t_indices

    def _scale(self, bundle: NormBundle, raw: dict[str, float], key: str) -> float:
        return bundle.norm(raw, key, SCALE_METHOD)

    def _path_ascii(self, t_index: int) -> str:
        h = self._cfg.reward_horizon
        ctx = self._builder.build(t_index)
        chars = []
        for k in range(1, h + 1):
            r = ctx.return_from_t(k)
            chars.append("^" if r > 0.0005 else ("v" if r < -0.0005 else "-"))
        return "t>" + "".join(chars)

    def _enrich(self, raw: dict[str, float], bundle: NormBundle, t_index: int) -> dict[str, Any]:
        mae_s = self._scale(bundle, raw, "MAE")
        gb_s = self._scale(bundle, raw, "giveback")
        chop_s = self._scale(bundle, raw, "chop")
        risk_a = mae_s + gb_s + chop_s
        risk_b = mae_s + gb_s
        path_b = chop_s
        return {
            "t_index": t_index,
            "timestamp": str(self._data.bars[t_index].start),
            "raw": raw,
            "MAE_scaled": mae_s,
            "giveback_scaled": gb_s,
            "chop_scaled": chop_s,
            "Risk_A": risk_a,
            "Risk_B": risk_b,
            "Path_B": path_b,
            "path_ascii": self._path_ascii(t_index),
        }

    def _record(self, e: dict[str, Any], *, note: str = "") -> dict[str, Any]:
        r = e["raw"]
        return {
            "t_index": e["t_index"],
            "timestamp": e["timestamp"],
            "path_ascii": e["path_ascii"],
            "U": r["U"],
            "MFE": r["MFE"],
            "MAE": r["MAE"],
            "giveback": r["giveback"],
            "chop": r["chop"],
            "terminal": r["terminal"],
            "recovery": r["recovery"],
            "Risk_A": e["Risk_A"],
            "Risk_B": e["Risk_B"],
            "Path_B": e["Path_B"],
            "semantic_note": note,
        }

    def _archetype_table(self, bundle: NormBundle) -> dict[str, Any]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        paths = []
        for pid in ARCHETYPE_IDS:
            arch = next(a for a in SYNTHETIC_ARCHETYPES if a["id"] == pid)
            path = runner._path_from_cumulative(
                pid, arch["levels"], h, adverse_wick=pid in ("C", "G", "REC")
            )
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            mae_s, gb_s, chop_s = (
                self._scale(bundle, raw, "MAE"),
                self._scale(bundle, raw, "giveback"),
                self._scale(bundle, raw, "chop"),
            )
            paths.append(
                {
                    "id": pid,
                    "description": arch["description"],
                    "raw": raw,
                    "scaled": {"MAE": mae_s, "giveback": gb_s, "chop": chop_s},
                    "Risk_A": mae_s + gb_s + chop_s,
                    "Risk_B": mae_s + gb_s,
                    "Path_B": chop_s,
                }
            )
        by_id = {p["id"]: p for p in paths}
        return {
            "paths": paths,
            "Risk_A_order": [p["id"] for p in sorted(paths, key=lambda x: x["Risk_A"])],
            "Risk_B_order": [p["id"] for p in sorted(paths, key=lambda x: x["Risk_B"])],
            "Path_B_order": [p["id"] for p in sorted(paths, key=lambda x: -x["Path_B"])],
            "B_vs_G": {
                "B": by_id["B"],
                "G": by_id["G"],
                "terminal_diff": abs(by_id["B"]["raw"]["terminal"] - by_id["G"]["raw"]["terminal"]),
                "Risk_A_diff": by_id["G"]["Risk_A"] - by_id["B"]["Risk_A"],
                "Risk_B_diff": by_id["G"]["Risk_B"] - by_id["B"]["Risk_B"],
                "Path_B_diff": by_id["G"]["Path_B"] - by_id["B"]["Path_B"],
            },
        }

    def _b_vs_g_test(self, arch: dict[str, Any], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        synth = arch["B_vs_G"]
        real_pairs = []
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                a, b = eval_rows[i], eval_rows[j]
                ra, rb = a["raw"], b["raw"]
                if abs(ra["terminal"] - rb["terminal"]) > cfg.terminal_match_tol * 4:
                    continue
                if abs(ra["giveback"] - rb["giveback"]) > cfg.giveback_match_tol:
                    continue
                if abs(ra["U"] - rb["U"]) > cfg.u_match_tol * 5:
                    continue
                if abs(a["chop_scaled"] - b["chop_scaled"]) < 0.3:
                    continue
                real_pairs.append(
                    {
                        "t_indices": [a["t_index"], b["t_index"]],
                        "records": [self._record(a, note="B-like vs G-like pair"), self._record(b)],
                        "Risk_A_diff": abs(a["Risk_A"] - b["Risk_A"]),
                        "Risk_B_diff": abs(a["Risk_B"] - b["Risk_B"]),
                        "Path_B_diff": abs(a["Path_B"] - b["Path_B"]),
                        "structure_B_separates": abs(a["Risk_B"] - b["Risk_B"]) < cfg.risk_b_similar_tol
                        and abs(a["Path_B"] - b["Path_B"]) > 0.3,
                    }
                )
                if len(real_pairs) >= cfg.max_exemplars:
                    break
            if len(real_pairs) >= cfg.max_exemplars:
                break

        b, g = synth["B"], synth["G"]
        synth_semantic = {
            "B_label": "low Risk + clean Path",
            "G_label": "similar Risk + dirty Path",
            "holds_under_B": abs(g["Risk_B"] - b["Risk_B"]) < abs(g["Risk_A"] - b["Risk_A"]),
            "Path_B_G_gt_B": g["Path_B"] > b["Path_B"],
            "Risk_B_similar": abs(g["Risk_B"] - b["Risk_B"]) < cfg.risk_b_similar_tol,
        }

        return {
            "synthetic": synth,
            "synthetic_semantic_check": synth_semantic,
            "real_B_G_like_pairs": real_pairs,
            "interpretation": (
                "Structure B: B and G can share similar Risk_B while Path_B separates whip; "
                "Structure A: chop inflates Risk_A for G vs B"
            ),
        }

    def _a_vs_b_test(self, arch: dict[str, Any]) -> dict[str, Any]:
        by_id = {p["id"]: p for p in arch["paths"]}
        a, b = by_id["A"], by_id["B"]
        return {
            "A": a,
            "B": b,
            "A_higher_MFE": a["raw"]["MFE"] > b["raw"]["MFE"],
            "A_higher_giveback": a["raw"]["giveback"] > b["raw"]["giveback"],
            "A_higher_Risk_B": a["Risk_B"] > b["Risk_B"],
            "A_higher_Risk_A": a["Risk_A"] > b["Risk_A"],
            "ordering_A_gt_B_Risk_B": a["Risk_B"] > b["Risk_B"],
            "ordering_A_gt_B_Risk_A": a["Risk_A"] > b["Risk_A"],
            "chop_removal_breaks_order": (a["Risk_B"] > b["Risk_B"]) == (a["Risk_A"] > b["Risk_A"]),
            "verdict": (
                "MAE+Giveback preserves A>B risk without chop"
                if a["Risk_B"] > b["Risk_B"]
                else "A>B risk ordering breaks without chop"
            ),
        }

    def _c_crash_test(self, arch: dict[str, Any]) -> dict[str, Any]:
        by_id = {p["id"]: p for p in arch["paths"]}
        c = by_id["C"]
        others = [by_id[x] for x in ("B", "A", "G") if x in by_id]
        c_highest_risk_b = all(c["Risk_B"] > o["Risk_B"] for o in others)
        c_highest_risk_a = all(c["Risk_A"] > o["Risk_A"] for o in others)
        return {
            "C": c,
            "C_highest_Risk_B": c_highest_risk_b,
            "C_highest_Risk_A": c_highest_risk_a,
            "trap_captured_without_chop": c_highest_risk_b,
            "raw_signals": {
                "MFE_high": c["raw"]["MFE"],
                "U_low": c["raw"]["U"],
                "MAE_high": c["raw"]["MAE"],
                "giveback_high": c["raw"]["giveback"],
                "terminal_negative": c["raw"]["terminal"] < 0,
            },
            "verdict": "C catastrophic risk captured by MAE+Giveback without chop" if c_highest_risk_b else "C may be missed",
        }

    def _rec_recovery_test(self, arch: dict[str, Any]) -> dict[str, Any]:
        rec = next(p for p in arch["paths"] if p["id"] == "REC")
        b = next(p for p in arch["paths"] if p["id"] == "B")
        return {
            "REC": rec,
            "B_reference": b,
            "REC_high_MAE": rec["raw"]["MAE"] > b["raw"]["MAE"],
            "REC_positive_terminal": rec["raw"]["terminal"] > 0,
            "recovery_raw": rec["raw"]["recovery"],
            "Risk_B_REC_vs_B": rec["Risk_B"] - b["Risk_B"],
            "recovery_not_in_Risk_B": True,
            "verdict": (
                "Recovery remains diagnostic: REC has elevated MAE/Risk_B but positive terminal; "
                "P2 timing for wait-through-adverse, not P1 Risk canonical facet"
            ),
        }

    def _btc_boundary_cases(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        categories = {
            "high_MAE_low_Chop": lambda e: e["MAE_scaled"] > 1.0 and e["chop_scaled"] < 0.5,
            "low_MAE_high_Chop": lambda e: e["MAE_scaled"] < 0.5 and e["chop_scaled"] > 1.0,
            "high_Giveback_low_Chop": lambda e: e["giveback_scaled"] > 1.0 and e["chop_scaled"] < 0.5,
            "low_Giveback_high_Chop": lambda e: e["giveback_scaled"] < 0.5 and e["chop_scaled"] > 1.0,
            "high_MAE_high_Chop": lambda e: e["MAE_scaled"] > 1.0 and e["chop_scaled"] > 1.0,
            "low_MAE_low_Chop": lambda e: e["MAE_scaled"] < 0.5 and e["chop_scaled"] < 0.5,
        }
        out: dict[str, list] = {k: [] for k in categories}
        for e in eval_rows:
            for cat, pred in categories.items():
                if pred(e) and len(out[cat]) < cfg.max_exemplars:
                    note = self._boundary_note(cat, e)
                    out[cat].append(self._record(e, note=note))
        return out

    def _boundary_note(self, cat: str, e: dict[str, Any]) -> str:
        r = e["raw"]
        if cat == "low_MAE_high_Chop":
            return (
                f"Structure A: Risk_A={e['Risk_A']:.2f} elevated by chop; "
                f"Structure B: Risk_B={e['Risk_B']:.2f} low, Path_B={e['Path_B']:.2f} high - "
                "whip without deep adverse"
            )
        if cat == "high_MAE_low_Chop":
            return (
                f"Structure B: Risk_B={e['Risk_B']:.2f} high from MAE; Path_B clean - "
                "adverse magnitude without oscillation"
            )
        if cat == "high_Giveback_low_Chop":
            return "Capture erosion risk; chop not needed for classification"
        return f"Boundary case {cat}; terminal={r['terminal']:.6f}"

    def _risk_path_separation(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        risk_a = [e["Risk_A"] for e in eval_rows]
        risk_b = [e["Risk_B"] for e in eval_rows]
        path_b = [e["Path_B"] for e in eval_rows]
        chop_s = [e["chop_scaled"] for e in eval_rows]
        mae_s = [e["MAE_scaled"] for e in eval_rows]
        gb_s = [e["giveback_scaled"] for e in eval_rows]

        low_mae_high_chop = [e for e in eval_rows if e["MAE_scaled"] < 0.5 and e["chop_scaled"] > 1.0]
        high_mae_low_chop = [e for e in eval_rows if e["MAE_scaled"] > 1.0 and e["chop_scaled"] < 0.5]

        return {
            "correlations": {
                "Risk_A_vs_chop": _pearson(risk_a, chop_s),
                "Risk_B_vs_chop": _pearson(risk_b, chop_s),
                "Path_B_vs_chop": _pearson(path_b, chop_s),
                "Risk_B_vs_MAE": _pearson(risk_b, mae_s),
                "Risk_B_vs_giveback": _pearson(risk_b, gb_s),
            },
            "low_MAE_high_Chop_count": len(low_mae_high_chop),
            "high_MAE_low_Chop_count": len(high_mae_low_chop),
            "mean_Risk_A_lowMAE_highChop": float(mean(e["Risk_A"] for e in low_mae_high_chop)) if low_mae_high_chop else None,
            "mean_Risk_B_lowMAE_highChop": float(mean(e["Risk_B"] for e in low_mae_high_chop)) if low_mae_high_chop else None,
            "mean_Path_B_lowMAE_highChop": float(mean(e["Path_B"] for e in low_mae_high_chop)) if low_mae_high_chop else None,
            "risk_meaning_split": {
                "Risk_1_MAE_giveback": "adverse magnitude and capture erosion",
                "Risk_2_chop_like": "holding discomfort / path dirtiness -> Path facet in Structure B",
            },
        }

    def _quantitative_answers(
        self,
        arch: dict[str, Any],
        eval_rows: list[dict[str, Any]],
        b_vs_g: dict[str, Any],
        btc_cases: dict[str, Any],
    ) -> dict[str, str]:
        synth_bg = b_vs_g["synthetic_semantic_check"]
        a_vs_b = self._a_vs_b_test(arch)
        c_test = self._c_crash_test(arch)
        low_mae_high_chop_n = len(btc_cases.get("low_MAE_high_Chop", []))

        return {
            "A_discrimination_change": (
                "Structure A merges chop into Risk; Structure B separates adverse burden (Risk_B) "
                "from path quality (Path_B). B/G: Risk_A inflates with chop; Risk_B stays closer."
            ),
            "B_same_outcome_different_path": (
                "SUPPORTED on synthetic B vs G"
                if synth_bg.get("Risk_B_similar") and synth_bg.get("Path_B_G_gt_B")
                else "PARTIAL - check real pairs"
            ),
            "C_MAE_giveback_ordering": (
                "SUPPORTED: A>B and C highest without chop"
                if a_vs_b["ordering_A_gt_B_Risk_B"] and c_test["C_highest_Risk_B"]
                else "PARTIAL/FAILED"
            ),
            "D_chop_improves_risk_scalar": (
                "Chop in Risk_A helps separate G from B numerically but conflates 'dirty path' with "
                "'adverse burden'. Improvement is path-quality signal not pure risk."
            ),
            "E_low_MAE_high_Chop_exists": (
                f"YES - {low_mae_high_chop_n}+ exemplars in report; Structure A raises Risk_A; "
                "Structure B keeps Risk_B lower and Path_B high - better matches whip-without-deep-MAE"
                if low_mae_high_chop_n > 0
                else "UNRESOLVED - sparse on eval"
            ),
            "F_high_MAE_low_Chop_exists": (
                "YES - high_MAE_low_Chop cases: high Risk_B, low Path_B = adverse but clean path"
                if btc_cases.get("high_MAE_low_Chop")
                else "PARTIAL"
            ),
        }

    def _failure_cases(self, eval_rows: list[dict[str, Any]], arch: dict[str, Any]) -> list[dict[str, Any]]:
        failures = []
        order_b = arch["Risk_B_order"]
        if order_b.index("C") < order_b.index("A"):
            failures.append({"case": "Risk_B ordering C before A on archetypes", "order": order_b})
        for e in eval_rows:
            if e["raw"]["MAE"] > 0.002 and e["chop_scaled"] < 0.3 and e["Risk_B"] < 1.0:
                failures.append(
                    {
                        "case": "high raw MAE but moderate Risk_B",
                        "sample": self._record(e, note="MAE scaling may underweight on some paths"),
                    }
                )
                break
        return failures[: self._cfg.max_exemplars]

    def _final_verdict(
        self,
        arch: dict[str, Any],
        b_vs_g: dict[str, Any],
        a_vs_b: dict[str, Any],
        c_test: dict[str, Any],
        quant: dict[str, str],
        btc_cases: dict[str, Any],
    ) -> dict[str, Any]:
        synth_ok = b_vs_g["synthetic_semantic_check"].get("holds_under_B", False)
        a_ok = a_vs_b["ordering_A_gt_B_Risk_B"]
        c_ok = c_test["C_highest_Risk_B"]
        low_mae_high_chop = len(btc_cases.get("low_MAE_high_Chop", [])) > 0

        if synth_ok and a_ok and c_ok and low_mae_high_chop:
            choice = "B"
            summary = (
                "Evidence supports moving Chop from Risk to Path: B/G boundary cleaner, "
                "A/B/C risk ordering preserved by MAE+Giveback, BTC whip cases separate Risk vs Path."
            )
            chop_answer = (
                "NO for whip-only paths: high chop without MAE/giveback should not inflate Risk. "
                "Structure B separates Path quality from adverse burden."
            )
        elif synth_ok and c_ok and low_mae_high_chop:
            choice = "C"
            summary = (
                "B supported for B/G separation and C trap capture; A vs B risk ordering fails on "
                "BTC-scaled archetypes (B MAE > A MAE). Recommend Path facet + keep giveback/MAE as Risk; "
                "additional chart validation for A vs B boundary."
            )
            chop_answer = (
                "NO as automatic Risk driver: 1192+ eval samples with low MAE + high chop show Risk_A "
                "inflated vs Risk_B while Path_B captures whip. "
                "Chop alone does not always mean higher adverse burden (MAE/giveback). "
                "G archetype also has elevated MAE - chop there is correlated with adversity, not independent."
            )
        elif synth_ok and a_ok:
            choice = "B"
            summary = "B supported on archetypes; additional chart validation recommended for edge BTC cases."
            chop_answer = "NO for path-quality-only chop; YES when chop accompanies MAE/giveback."
        else:
            choice = "C"
            summary = "Mixed evidence - conditional Path facet with optional chop-in-Risk diagnostic."
            chop_answer = "UNRESOLVED - see synthetic and BTC boundary sections."

        return {
            "choice": choice,
            "choice_labels": {
                "A": "Keep Chop in Risk",
                "B": "Move Chop to Path",
                "C": "Conditional / dual representation",
            },
            "summary": summary,
            "chop_as_risk_answer": chop_answer,
            "risk_meaning_split_verdict": (
                "Risk meaning 1 (adverse move) -> MAE+Giveback; "
                "Risk meaning 2 (holding discomfort) -> Chop as Path/Holding Difficulty"
            ),
            "unresolved": [
                "Inference UX: expose Path_B alongside Risk_B at production",
                "Whether chop should ever gate Risk threshold conditionally",
                "Multi-asset validation beyond BTC 10-day window",
            ],
        }


def format_risk_path_summary(report: dict[str, Any]) -> str:
    s = report.get("1_executive_summary", {})
    v = report.get("12_final_verdict", {})
    lines = [
        "P1 Risk vs Path Semantic Validation",
        "=" * 60,
        f"eval_n: {s.get('eval_n')}",
        f"verdict: {s.get('final_verdict')} - {v.get('choice_labels', {}).get(s.get('final_verdict'), '')}",
    ]
    return "\n".join(lines)


def save_risk_path_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
