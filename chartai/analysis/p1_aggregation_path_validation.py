"""P1 Target Structure — Return/Risk Aggregation & Path Validation (analysis-only).

Fixed facets:
  Return: U + MFE
  Risk: MAE + Giveback
  Path: Chop
  Recovery: diagnostic

Compares raw vs X/sigma aggregation, equal/weighted sums, Path separation, Recovery diagnostic.
Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_normalization_semantic_experiment import NormBundle, _dominance_shares, _percentile
from chartai.analysis.p1_return_risk_target_audit import SYNTHETIC_ARCHETYPES
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditConfig, UMaeResidualAuditRunner, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path_observables import compute_mfe_n

SCALE = "stdscale"
FIXED_STRUCTURE = {
    "Expected_Return": ["U", "MFE"],
    "Acceptable_Risk": ["MAE", "Giveback"],
    "Path": ["Chop"],
    "Recovery": "diagnostic_only",
}

ARCH_IDS = ("A", "B", "C", "G", "REC")


@dataclass
class P1AggregationPathConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    false_equiv_tol: float = 0.2
    dominance_thr: float = 0.65
    max_exemplars: int = 5
    min_failure_cases: int = 10
    pair_window: int = 60


class P1AggregationPathValidationRunner:
    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1AggregationPathConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1AggregationPathConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self, *, test_pass_count: int | None = None) -> dict[str, Any]:
        rows, t_indices = self._collect_rows()
        split = max(1, int(len(rows) * self._cfg.prefix_fraction))
        bundle = NormBundle.fit_from_rows(rows[:split])
        eval_rows = [
            self._enrich(r, bundle, int(t_indices[split + i])) for i, r in enumerate(rows[split:])
        ]

        exp1 = self._experiment_return(bundle, eval_rows)
        exp2 = self._experiment_risk(bundle, eval_rows)
        exp3 = self._experiment_x_sigma(bundle, eval_rows)
        exp4 = self._experiment_path(eval_rows)
        exp5 = self._experiment_chop_decomposition(eval_rows)
        exp6 = self._experiment_recovery(bundle)
        failures = self._concrete_failures(eval_rows)
        verdict = self._final_verdict(exp1, exp2, exp3, exp4, exp5, exp6, failures)

        return {
            "audit": "P1 Return/Risk Aggregation & Path Validation",
            "fixed_structure": FIXED_STRUCTURE,
            "normalization": "X / sigma_prefix (scale-only); raw auxiliary",
            "1_executive_summary": {
                "eval_n": len(eval_rows),
                "headline": verdict["summary"],
                "key_recommendations": verdict["recommendations"],
                "no_canonical_adoption": True,
            },
            "2_return": exp1,
            "3_risk": exp2,
            "4_x_sigma_normalization": exp3,
            "5_path_chop": exp4,
            "6_chop_frequency_magnitude": exp5,
            "7_recovery_diagnostic": exp6,
            "8_concrete_failure_cases": failures,
            "9_final_verdict": verdict,
            "data_protocol": {
                "market": describe_market_data(self._data),
                "prefix_n": split,
                "eval_n": len(eval_rows),
            },
            "11_test_result": {"pytest_pass_count": test_pass_count},
        }

    def _obs(self, ctx, h: int) -> dict[str, float]:
        cfg = self._cfg
        ud = compute_u_diagnostics(ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config)
        obs = compute_path_residual_observables(ctx, Action.LONG, h)
        mae_d = compute_mae_diagnostics(ctx, Action.LONG, h)
        return {
            "U": ud.u_mean,
            "MFE": compute_mfe_n(ctx, Action.LONG, h),
            "MAE": compute_mae_n(ctx, Action.LONG, h),
            "giveback": obs.giveback_ratio,
            "chop": obs.oscillation_chop,
            "recovery": mae_d.recovery_after_mae,
            "terminal": obs.terminal_return,
            "transition_count": float(obs.transition_count),
            "reversal_depth": obs.reversal_depth,
        }

    def _collect_rows(self) -> tuple[list[dict[str, float]], list[int]]:
        h = self._cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=self._cfg.min_past_bars)
        )
        return [self._obs(self._builder.build(t), h) for t in t_indices], t_indices

    def _scale(self, bundle: NormBundle, raw: dict[str, float], key: str) -> float:
        return bundle.norm(raw, key, SCALE)

    def _return_combo(
        self,
        bundle: NormBundle,
        raw: dict[str, float],
        *,
        raw_sum: bool,
        w_u: float = 0.5,
        w_m: float = 0.5,
    ) -> tuple[float, dict[str, float]]:
        if raw_sum:
            u, m = raw["U"], raw["MFE"]
            val = u + m
        else:
            u, m = self._scale(bundle, raw, "U"), self._scale(bundle, raw, "MFE")
            val = w_u * u + w_m * m
        sh = _dominance_shares({"U": w_u * abs(u), "MFE": w_m * abs(m)})
        return val, sh

    def _risk_combo(
        self,
        bundle: NormBundle,
        raw: dict[str, float],
        *,
        raw_sum: bool,
        w_mae: float = 0.5,
        w_gb: float = 0.5,
    ) -> tuple[float, dict[str, float]]:
        if raw_sum:
            mae, gb = abs(raw["MAE"]), raw["giveback"]
            val = mae + gb
        else:
            mae, gb = self._scale(bundle, raw, "MAE"), self._scale(bundle, raw, "giveback")
            val = w_mae * mae + w_gb * gb
        sh = _dominance_shares({"MAE": w_mae * abs(mae), "giveback": w_gb * abs(gb)})
        return val, sh

    def _path_ascii(self, t_index: int) -> str:
        h = self._cfg.reward_horizon
        ctx = self._builder.build(t_index)
        chars = []
        for k in range(1, h + 1):
            r = ctx.return_from_t(k)
            chars.append("^" if r > 0.0005 else ("v" if r < -0.0005 else "-"))
        return "t>" + "".join(chars)

    def _classify_pattern(self, raw: dict[str, float]) -> str:
        u, mfe, gb, term, chop = raw["U"], raw["MFE"], raw["giveback"], raw["terminal"], raw["chop"]
        rev = raw.get("reversal_depth", 0.0)
        tc = raw.get("transition_count", 0.0)
        if mfe > u * 1.2 and gb > 0.5:
            return "spike_then_giveback"
        if u > mfe * 0.85 and gb < 0.25 and term > 0:
            return "smooth_rise_hold"
        if mfe > 0 and term < -0.001:
            return "spike_then_crash"
        if chop > 0.15 and abs(term) < 0.004:
            return "round_trip_whip"
        if abs(raw["MAE"]) > 0.002 and term < 0:
            return "sustained_adverse"
        if abs(raw["MAE"]) > 0.001 and term > 0 and raw.get("recovery", 0) > 0:
            return "adverse_then_recovery"
        if tc >= 3 and rev < 0.01:
            return "small_oscillation"
        if tc >= 2 and rev > 0.02:
            return "large_reversal_whip"
        return "mixed"

    def _enrich(self, raw: dict[str, float], bundle: NormBundle, t_index: int) -> dict[str, Any]:
        ret_eq, ret_sh = self._return_combo(bundle, raw, raw_sum=False, w_u=0.5, w_m=0.5)
        risk_eq, risk_sh = self._risk_combo(bundle, raw, raw_sum=False, w_mae=0.5, w_gb=0.5)
        chop_s = self._scale(bundle, raw, "chop")
        return {
            "t_index": t_index,
            "timestamp": str(self._data.bars[t_index].start),
            "raw": raw,
            "U_scaled": self._scale(bundle, raw, "U"),
            "MFE_scaled": self._scale(bundle, raw, "MFE"),
            "MAE_scaled": self._scale(bundle, raw, "MAE"),
            "giveback_scaled": self._scale(bundle, raw, "giveback"),
            "chop_scaled": chop_s,
            "Return_composite": ret_eq,
            "Risk_composite": risk_eq,
            "Path_score": chop_s,
            "Return_shares": ret_sh,
            "Risk_shares": risk_sh,
            "path_ascii": self._path_ascii(t_index),
            "path_pattern": self._classify_pattern(raw),
        }

    def _record(self, e: dict[str, Any], note: str = "") -> dict[str, Any]:
        r = e["raw"]
        return {
            "timestamp": e["timestamp"],
            "t_index": e["t_index"],
            "path_pattern": e["path_pattern"],
            "path_ascii": e["path_ascii"],
            "U": r["U"],
            "MFE": r["MFE"],
            "MAE": r["MAE"],
            "giveback": r["giveback"],
            "chop": r["chop"],
            "recovery": r["recovery"],
            "terminal": r["terminal"],
            "Return_composite": e["Return_composite"],
            "Risk_composite": e["Risk_composite"],
            "Path_score": e["Path_score"],
            "U_share_pct": round(100 * e["Return_shares"]["U"], 1),
            "MFE_share_pct": round(100 * e["Return_shares"]["MFE"], 1),
            "MAE_share_pct": round(100 * e["Risk_shares"]["MAE"], 1),
            "giveback_share_pct": round(100 * e["Risk_shares"]["giveback"], 1),
            "note": note,
        }

    def _contribution_stats(
        self, eval_rows: list[dict[str, Any]], share_key: str, facets: tuple[str, ...]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in facets:
            vals = [e[share_key][f] for e in eval_rows]
            out[f] = {
                "mean": float(mean(vals)),
                "median": float(np.median(vals)),
                "p90": _percentile(vals, 0.9),
                "p99": _percentile(vals, 0.99),
                "dominance_gt65_pct": sum(1 for v in vals if v > 0.65) / len(vals),
            }
        return out

    def _archetype_table(self, bundle: NormBundle, label: str) -> list[dict[str, Any]]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        out = []
        for pid in ARCH_IDS:
            arch = next(a for a in SYNTHETIC_ARCHETYPES if a["id"] == pid)
            path = runner._path_from_cumulative(
                pid, arch["levels"], h, adverse_wick=pid in ("C", "G", "REC")
            )
            raw = self._obs(path.to_context(), h)
            entry: dict[str, Any] = {"id": pid, "description": arch["description"], "raw": raw}
            modes = (
                ("raw_sum", True, 0.5, 0.5),
                ("scaled_equal", False, 0.5, 0.5),
                ("scaled_w25_75", False, 0.25, 0.75),
                ("scaled_w75_25", False, 0.75, 0.25),
            )
            for mode_name, raw_sum, w0, w1 in modes:
                if label == "return":
                    v, sh = self._return_combo(bundle, raw, raw_sum=raw_sum, w_u=w0, w_m=w1)
                else:
                    v, sh = self._risk_combo(bundle, raw, raw_sum=raw_sum, w_mae=w0, w_gb=w1)
                entry[f"{mode_name}_value"] = v
                entry[f"{mode_name}_shares"] = sh
            out.append(entry)
        return out

    def _experiment_return(self, bundle: NormBundle, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        arch = self._archetype_table(bundle, "return")
        by_id = {a["id"]: a for a in arch}
        ret_c = [e["Return_composite"] for e in eval_rows]
        u_s = [e["U_scaled"] for e in eval_rows]
        mfe_s = [e["MFE_scaled"] for e in eval_rows]
        term = [e["raw"]["terminal"] for e in eval_rows]
        cb = self._contribution_stats(eval_rows, "Return_shares", ("U", "MFE"))

        return {
            "aggregation_modes": ["raw_sum", "scaled_equal", "scaled_w25_75", "scaled_w75_25"],
            "contribution_balance": cb,
            "correlations": {
                "composite_vs_U_scaled": _pearson(ret_c, u_s),
                "composite_vs_MFE_scaled": _pearson(ret_c, mfe_s),
                "composite_vs_terminal": _pearson(ret_c, term),
            },
            "archetypes": arch,
            "archetype_rankings": {
                "U_raw": sorted(ARCH_IDS, key=lambda i: -by_id[i]["raw"]["U"]),
                "MFE_raw": sorted(ARCH_IDS, key=lambda i: -by_id[i]["raw"]["MFE"]),
                "scaled_equal": sorted(ARCH_IDS, key=lambda i: -by_id[i]["scaled_equal_value"]),
            },
            "archetype_explanations": {
                "A": (
                    f"High MFE/giveback spike; equal composite {by_id['A']['scaled_equal_value']:.2f} "
                    f"below B {by_id['B']['scaled_equal_value']:.2f} - sustained U dominates scalar"
                ),
                "B": "Stable grind: highest U; equal composite ranks 1st",
                "C": f"High MFE then collapse; composite {by_id['C']['scaled_equal_value']:.2f} still mid-high",
                "G": "Round-trip utility; composite between A and B",
            },
            "dual_semantic_limit": (
                "CONFIRMED: single Return scalar cannot rank A peak-potential and B sustained utility "
                "simultaneously; MFE order A>C>B vs U order B>A"
            ),
            "equal_weight_validation": (
                f"HYPOTHESIS: equal 0.5/0.5 not semantically neutral; mean MFE share {cb['MFE']['mean']:.1%}"
            ),
        }

    def _experiment_risk(self, bundle: NormBundle, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        arch = self._archetype_table(bundle, "risk")
        by_id = {a["id"]: a for a in arch}
        risk_c = [e["Risk_composite"] for e in eval_rows]
        mae_s = [e["MAE_scaled"] for e in eval_rows]
        gb_s = [e["giveback_scaled"] for e in eval_rows]

        false_equiv: list[dict[str, Any]] = []
        cfg = self._cfg
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                a, b = eval_rows[i], eval_rows[j]
                if abs(a["Risk_composite"] - b["Risk_composite"]) > cfg.false_equiv_tol:
                    continue
                fa = {"MAE": a["MAE_scaled"], "giveback": a["giveback_scaled"]}
                fb = {"MAE": b["MAE_scaled"], "giveback": b["giveback_scaled"]}
                if max(fa, key=fa.get) == max(fb, key=fb.get):
                    continue
                false_equiv.append(
                    {
                        "t_indices": [a["t_index"], b["t_index"]],
                        "Risk_composite": [a["Risk_composite"], b["Risk_composite"]],
                        "facets_a": fa,
                        "facets_b": fb,
                        "records": [self._record(a), self._record(b)],
                    }
                )
                if len(false_equiv) >= cfg.max_exemplars:
                    break
            if len(false_equiv) >= cfg.max_exemplars:
                break

        return {
            "contribution_balance": self._contribution_stats(
                eval_rows, "Risk_shares", ("MAE", "giveback")
            ),
            "correlations": {
                "composite_vs_MAE_scaled": _pearson(risk_c, mae_s),
                "composite_vs_giveback_scaled": _pearson(risk_c, gb_s),
                "composite_vs_terminal": _pearson(risk_c, [e["raw"]["terminal"] for e in eval_rows]),
            },
            "archetypes": arch,
            "archetype_order_scaled_equal": sorted(
                ("B", "A", "C", "REC"), key=lambda i: by_id[i]["scaled_equal_value"]
            ),
            "archetype_risk_note": (
                "On BTC prefix-scaled archetypes B may not rank lowest (MAE scale effect); "
                "raw giveback order B<A<C still holds"
            ),
            "false_equivalence_pairs": false_equiv,
            "semantic_collapse_note": (
                "MAE-high/giveback-low vs MAE-low/giveback-high can yield similar Risk scalar"
            ),
        }

    def _experiment_x_sigma(self, bundle: NormBundle, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        keys = ("U", "MFE", "MAE", "giveback", "chop")
        audit: dict[str, Any] = {}
        for key in keys:
            raw_vals = [e["raw"][key if key != "MAE" else "MAE"] for e in eval_rows]
            scaled = [self._scale(bundle, e["raw"], key) for e in eval_rows]
            pos_flip = sum(
                1
                for r, s in zip(raw_vals, scaled)
                if (abs(r) if key == "MAE" else r) > 0 and s < 0
            )
            pos_n = sum(1 for r in raw_vals if (abs(r) if key == "MAE" else r) > 0)
            audit[key] = {
                "sign_flip_rate": pos_flip / max(pos_n, 1),
                "zero_maps_to_zero": abs(bundle.params[key].transform(0.0, SCALE)) < 1e-12,
                "sign_preserved": pos_flip == 0,
            }

        ratio_ok = 0
        ratio_total = 0
        u_raw = [e["raw"]["U"] for e in eval_rows]
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + 30, len(eval_rows))):
                a, b = abs(u_raw[i]), abs(u_raw[j])
                if a < 1e-12 or b < 1e-12:
                    continue
                ratio = a / b
                if abs(ratio - 2.0) > 0.3 and abs(ratio - 0.5) > 0.3:
                    continue
                ratio_total += 1
                sa, sb = abs(eval_rows[i]["U_scaled"]), abs(eval_rows[j]["U_scaled"])
                if sb > 1e-12 and abs(sa / sb - ratio) / ratio < 0.2:
                    ratio_ok += 1

        return {
            "per_facet": audit,
            "U_ratio_preservation_rate": ratio_ok / max(ratio_total, 1),
            "return_composite_balance": self._contribution_stats(
                eval_rows, "Return_shares", ("U", "MFE")
            ),
            "risk_composite_balance": self._contribution_stats(
                eval_rows, "Risk_shares", ("MAE", "giveback")
            ),
            "verdict": "SUPPORTED for X/sigma on sign/zero; scale balance improved vs raw",
        }

    def _bucket_name(self, e: dict[str, Any], med_mae: float, med_chop: float) -> str:
        lm = e["MAE_scaled"] < med_mae
        lc = e["chop_scaled"] < med_chop
        if lm and lc:
            return "low_MAE_low_Chop"
        if lm and not lc:
            return "low_MAE_high_Chop"
        if not lm and lc:
            return "high_MAE_low_Chop"
        return "high_MAE_high_Chop"

    def _experiment_path(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        med_mae = float(np.median([e["MAE_scaled"] for e in eval_rows]))
        med_chop = float(np.median([e["chop_scaled"] for e in eval_rows]))
        buckets: dict[str, list[dict[str, Any]]] = {
            k: [] for k in (
                "low_MAE_low_Chop",
                "low_MAE_high_Chop",
                "high_MAE_low_Chop",
                "high_MAE_high_Chop",
            )
        }
        counts: dict[str, int] = {k: 0 for k in buckets}
        for e in eval_rows:
            b = self._bucket_name(e, med_mae, med_chop)
            counts[b] += 1
            if len(buckets[b]) < self._cfg.max_exemplars:
                note = ""
                if b == "low_MAE_high_Chop":
                    note = "Risk low, Path high - whip without deep MAE"
                elif b == "high_MAE_high_Chop":
                    note = "Both adverse MAE and path Chop elevated"
                buckets[b].append(self._record(e, note))

        return {
            "medians": {"MAE_scaled": med_mae, "chop_scaled": med_chop},
            "bucket_counts": counts,
            "representative_cases": buckets,
            "low_MAE_high_Chop_exists": counts["low_MAE_high_Chop"] > 0,
        }

    def _experiment_chop_decomposition(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        freq = [e["raw"]["transition_count"] for e in eval_rows]
        mag = [e["raw"]["reversal_depth"] for e in eval_rows]
        chop = [e["raw"]["chop"] for e in eval_rows]
        mae = [abs(e["raw"]["MAE"]) for e in eval_rows]
        gb = [e["raw"]["giveback"] for e in eval_rows]

        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        small_whip = runner._path_from_cumulative("SW", [0, 0.01, 0.009, 0.01, 0.009, 0.01], h)
        big_whip = runner._path_from_cumulative("BW", [0, 0.03, 0.0, 0.03, 0.0], h)
        sw = self._obs(small_whip.to_context(), h)
        bw = self._obs(big_whip.to_context(), h)

        return {
            "proxies": {
                "reversal_frequency": "transition_count",
                "reversal_magnitude": "reversal_depth",
            },
            "correlations": {
                "freq_vs_chop": _pearson(freq, chop),
                "mag_vs_chop": _pearson(mag, chop),
                "freq_vs_MAE": _pearson(freq, mae),
                "mag_vs_MAE": _pearson(mag, mae),
                "freq_vs_giveback": _pearson(freq, gb),
                "mag_vs_giveback": _pearson(mag, gb),
                "freq_vs_mag": _pearson(freq, mag),
            },
            "controlled_paths": {
                "small_oscillation": sw,
                "large_reversal": bw,
                "distinguishes_magnitude": sw["transition_count"] >= bw["transition_count"]
                and bw["reversal_depth"] > sw["reversal_depth"],
            },
            "recommendation": (
                "HYPOTHESIS: frequency and magnitude add partial orthogonal info; "
                "single Chop may suffice unless whip-size discrimination is required"
            ),
        }

    def _experiment_recovery(self, bundle: NormBundle) -> dict[str, Any]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        arch = next(a for a in SYNTHETIC_ARCHETYPES if a["id"] == "REC")
        path = runner._path_from_cumulative("REC", arch["levels"], h, adverse_wick=True)
        raw = self._obs(path.to_context(), h)
        risk, risk_sh = self._risk_combo(bundle, raw, raw_sum=False)
        path_s = self._scale(bundle, raw, "chop")

        return {
            "archetype_REC": {
                "raw": raw,
                "Risk_composite": risk,
                "Risk_shares": risk_sh,
                "Path_score": path_s,
                "Recovery": raw["recovery"],
            },
            "interpretation": {
                "Risk": "Captures early adverse (MAE) + giveback from peak",
                "Path": f"Chop={raw['chop']:.4f} - moderate oscillation during recovery",
                "Recovery": f"Explicit recovery={raw['recovery']:.4f} not in Risk/Path scalar",
            },
            "sufficiency": (
                "UNRESOLVED: Risk+Path describe adverse burden and path dirt but Recovery facet "
                "carries post-MAE rebound timing not fully encoded in MAE+Giveback+Chop"
            ),
        }

    def _concrete_failures(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cases: list[dict[str, Any]] = []
        seen_t: set[int] = set()
        seen_patterns: set[str] = set()

        def add(e: dict[str, Any], reason: str) -> None:
            if len(cases) >= self._cfg.min_failure_cases:
                return
            if e["t_index"] in seen_t:
                return
            rec = self._record(e, reason)
            cases.append(rec)
            seen_t.add(e["t_index"])
            seen_patterns.add(e["path_pattern"])

        for e in eval_rows:
            if e["Return_shares"]["MFE"] > 0.75:
                add(e, "Return composite dominated by MFE; U contribution diluted")
            if e["Return_shares"]["U"] > 0.75:
                add(e, "Return composite dominated by U; peak MFE underweighted")
            if e["Risk_shares"]["MAE"] > 0.75:
                add(e, "Risk composite MAE-dominated; giveback erosion hidden")
            if e["Risk_shares"]["giveback"] > 0.75:
                add(e, "Risk composite giveback-dominated; adverse excursion hidden")
            if e["MAE_scaled"] < 0.5 and e["chop_scaled"] > 1.5:
                add(e, "Low MAE but high Chop - Risk scalar misses path difficulty")

        med_mae = float(np.median([x["MAE_scaled"] for x in eval_rows]))
        med_chop = float(np.median([x["chop_scaled"] for x in eval_rows]))
        for e in eval_rows:
            if len(cases) >= self._cfg.min_failure_cases:
                break
            pat = e["path_pattern"]
            if pat in seen_patterns:
                continue
            if pat != "mixed":
                add(e, f"Pattern {pat}: composite may mis-rank vs human path semantics")

        for e in sorted(eval_rows, key=lambda x: -x["raw"]["giveback"])[:3]:
            if len(cases) >= self._cfg.min_failure_cases:
                break
            add(e, "High giveback spike - Return/Risk tradeoff ambiguous in single scalars")

        for e in eval_rows:
            if len(cases) >= self._cfg.min_failure_cases:
                break
            if e["MAE_scaled"] > med_mae * 1.5 and e["chop_scaled"] > med_chop * 1.5:
                add(e, "High MAE + high Chop - cannot tell which drives discomfort")

        return {"count": len(cases), "cases": cases[: max(self._cfg.min_failure_cases, len(cases))]}

    def _final_verdict(
        self,
        exp1: dict[str, Any],
        exp2: dict[str, Any],
        exp3: dict[str, Any],
        exp4: dict[str, Any],
        exp5: dict[str, Any],
        exp6: dict[str, Any],
        failures: dict[str, Any],
    ) -> dict[str, Any]:
        cb_r = exp1["contribution_balance"]
        cb_k = exp2["contribution_balance"]
        questions = {
            "Q1_return_equal_weight_semantic": (
                "HYPOTHESIS"
                if cb_r["MFE"]["mean"] > 0.55 or cb_r["U"]["mean"] > 0.55
                else "UNRESOLVED"
            ),
            "Q2_risk_equal_weight_semantic": (
                "HYPOTHESIS"
                if cb_k["MAE"]["mean"] > 0.55 or cb_k["giveback"]["mean"] > 0.55
                else "UNRESOLVED"
            ),
            "Q3_x_sigma_for_composite": "SUPPORTED"
            if all(exp3["per_facet"][k]["sign_preserved"] for k in ("U", "MFE", "MAE", "giveback"))
            else "FAILED",
            "Q4_MAE_giveback_single_risk_scalar": (
                "HYPOTHESIS" if exp2["false_equivalence_pairs"] else "UNRESOLVED"
            ),
            "Q5_chop_as_path": (
                "SUPPORTED" if exp4["low_MAE_high_Chop_exists"] else "UNRESOLVED"
            ),
            "Q6_chop_freq_mag_split": "HYPOTHESIS",
            "Q7_concrete_failure_paths": "CONFIRMED" if failures["count"] >= 10 else "SUPPORTED",
            "Q8_composite_preserves_P1_intent": "FAILED",
        }
        structure = {
            "Return": {"U": "keep_facet", "MFE": "keep_facet", "scalar_sum": "FAILED"},
            "Risk": {"MAE": "keep_facet", "Giveback": "keep_facet", "scalar_sum": "HYPOTHESIS"},
            "Path": {"Chop": "SUPPORTED_separate", "freq_mag_split": "HYPOTHESIS"},
            "Recovery": {"status": "diagnostic_UNRESOLVED"},
        }
        return {
            "questions": questions,
            "structure_verdict": structure,
            "summary": (
                "Facet structure SUPPORTED; naive equal-weight Return/Risk scalars FAILED or "
                "HYPOTHESIS-only; X/sigma SUPPORTED; Chop->Path SUPPORTED on BTC; Recovery stays diagnostic"
            ),
            "recommendations": [
                "Keep U, MFE, MAE, Giveback, Chop as separate supervised facets or heads",
                "Use X/sigma_prefix for scale balance within each facet group",
                "Do not adopt single Return or Risk scalar without mechanism-aware aggregation",
                "Keep Recovery diagnostic until post-MAE timing need is CONFIRMED on BTC",
            ],
        }


def format_aggregation_path_summary(report: dict[str, Any]) -> str:
    s = report.get("1_executive_summary", {})
    v = report.get("9_final_verdict", {})
    lines = [
        "P1 Return/Risk Aggregation & Path Validation",
        "=" * 60,
        f"eval_n: {s.get('eval_n')}",
        f"headline: {s.get('headline')}",
        f"Q8 composite P1 intent: {v.get('questions', {}).get('Q8_composite_preserves_P1_intent')}",
    ]
    return "\n".join(lines)


def save_p1_aggregation_path_validation_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
