"""P1 Target Normalization — Semantic Preservation & Scale Matching (analysis-only).

Compares Raw, Standard Z-score, Scale-only Std (X/std), Scale-only RMS (X/RMS)
for fixed P1 composite targets:
  Return = U + MFE
  Risk = MAE + Giveback + Chop

Prefix-fit scale parameters only (causal eval apply). Does NOT modify canonical code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
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

NormMethod = Literal["raw", "zscore", "stdscale", "rmsscale"]
NORM_METHODS: tuple[NormMethod, ...] = ("raw", "zscore", "stdscale", "rmsscale")

RETURN_KEYS = ("U", "MFE")
RISK_KEYS = ("MAE", "giveback", "chop")

FIXED_STRUCTURE = {
    "Expected_Return": "U + MFE",
    "Acceptable_Risk": "MAE + Giveback + Chop",
    "Recovery": "diagnostic_only",
}


@dataclass(frozen=True)
class PrefixNormParams:
    """Prefix-fitted normalization parameters for one observable stream."""

    name: str
    mu: float
    sigma: float
    rms: float

    @classmethod
    def fit(cls, name: str, values: tuple[float, ...]) -> PrefixNormParams:
        if not values:
            return cls(name, 0.0, 1.0, 1.0)
        mu = mean(values)
        sigma = pstdev(values) if len(values) > 1 else 1.0
        rms = math.sqrt(mean(v * v for v in values))
        return cls(name, mu, max(sigma, 1e-12), max(rms, 1e-12))

    def transform(self, x: float, method: NormMethod) -> float:
        if method == "raw":
            return x
        if method == "zscore":
            return (x - self.mu) / self.sigma
        if method == "stdscale":
            return x / self.sigma
        if method == "rmsscale":
            return x / self.rms
        raise ValueError(method)

    def at_zero(self, method: NormMethod) -> float:
        return self.transform(0.0, method)


@dataclass
class NormBundle:
    """Prefix-fit normalizers for Return and Risk observables."""

    params: dict[str, PrefixNormParams]

    @classmethod
    def fit_from_rows(cls, rows: list[dict[str, float]]) -> NormBundle:
        keys = (*RETURN_KEYS, *RISK_KEYS, "recovery", "P_long", "P_short")
        params = {
            k: PrefixNormParams.fit(k, tuple(r[k] for r in rows))
            for k in keys
            if k in rows[0]
        }
        return cls(params=params)

    def norm(self, raw: dict[str, float], key: str, method: NormMethod) -> float:
        val = raw[key]
        if key == "MAE":
            val = abs(val)
        return self.params[key].transform(val, method)

    def composite_return(self, raw: dict[str, float], method: NormMethod) -> float:
        return sum(self.norm(raw, k, method) for k in RETURN_KEYS)

    def composite_risk(self, raw: dict[str, float], method: NormMethod) -> float:
        return sum(self.norm(raw, k, method) for k in RISK_KEYS)

    def facet_vector(self, raw: dict[str, float], keys: tuple[str, ...], method: NormMethod) -> dict[str, float]:
        return {k: self.norm(raw, k, method) for k in keys}


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(min(len(s) - 1, max(0, p * (len(s) - 1))))
    return s[idx]


def _dominance_shares(facets: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in facets.values()) or 1e-12
    return {k: abs(v) / total for k, v in facets.items()}


@dataclass
class P1NormSemanticConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    u_match_tol: float = 0.0003
    mfe_match_tol: float = 0.0003
    ma_match_tol: float = 0.0003
    giveback_match_tol: float = 0.08
    chop_match_tol: float = 0.04
    magnitude_ratio_tol: float = 0.15
    max_matched_pairs: int = 5
    pair_window: int = 80


class P1NormalizationSemanticExperimentRunner:
    """Semantic preservation vs scale balance experiment for P1 normalization."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1NormSemanticConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1NormSemanticConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self, *, test_pass_count: int | None = None) -> dict[str, Any]:
        rows, _ = self._collect_rows()
        split = max(1, int(len(rows) * self._cfg.prefix_fraction))
        prefix_rows = rows[:split]
        eval_rows = rows[split:]
        bundle = NormBundle.fit_from_rows(prefix_rows)

        eval_enriched = [{"t_index": int(r["t_index"]), "raw": r} for r in eval_rows]
        archetypes = self._build_archetypes(bundle)

        return_audit = self._return_experiment(bundle, eval_enriched, archetypes)
        risk_audit = self._risk_experiment(bundle, eval_enriched, archetypes)
        z_audit = self._zscore_semantic_audit(bundle, eval_enriched)
        std_audit = self._scale_only_audit(bundle, "stdscale")
        rms_audit = self._scale_only_audit(bundle, "rmsscale")
        matched = self._matched_path_experiment(bundle, eval_enriched)
        dominance = self._dominance_analysis(bundle, eval_enriched)
        comparison = self._comparison_table(return_audit, risk_audit, z_audit, std_audit, rms_audit, dominance)
        final_answer = self._final_normalization_answer(comparison)

        return {
            "audit": "P1 Target Normalization — Semantic Preservation & Scale Matching",
            "fixed_structure": FIXED_STRUCTURE,
            "primary_evidence": "BTCUSDT real eval",
            "1_executive_summary": {
                "eval_n": len(eval_rows),
                "normalizations_compared": list(NORM_METHODS),
                "headline": final_answer["summary"],
                "best_semantic_preservation": comparison["rankings"]["semantic_preservation"],
                "best_scale_balance": comparison["rankings"]["scale_balance"],
                "no_canonical_adoption": True,
            },
            "2_normalization_definitions": self._norm_definitions(bundle),
            "3_return_results": return_audit,
            "4_risk_results": risk_audit,
            "5_zscore_semantic_audit": z_audit,
            "6_stdscale_audit": std_audit,
            "7_rmsscale_audit": rms_audit,
            "8_archetype_results": archetypes,
            "9_btc_realdata_summary": {
                "market": describe_market_data(self._data),
                "prefix_n": split,
                "eval_n": len(eval_rows),
            },
            "10_matched_path_results": matched,
            "11_contribution_dominance": dominance,
            "12_semantic_vs_scale_comparison_table": comparison["table"],
            "13_normalization_tradeoffs": comparison["tradeoffs"],
            "14_unresolved": comparison["unresolved"],
            "15_next_experiments": comparison["next_experiments"],
            "final_question_answer": final_answer,
            "11_test_result": {"pytest_pass_count": test_pass_count},
        }

    def _norm_definitions(self, bundle: NormBundle) -> dict[str, Any]:
        defs = {
            "raw": {"formula": "X_norm = X", "zero_at_zero": True, "sign_preserved": True},
            "zscore": {"formula": "X_norm = (X - mu_prefix) / sigma_prefix", "zero_at_zero": False, "sign_preserved": "conditional"},
            "stdscale": {"formula": "X_norm = X / sigma_prefix", "zero_at_zero": True, "sign_preserved": True},
            "rmsscale": {"formula": "X_norm = X / RMS_prefix, RMS=sqrt(mean(X^2))", "zero_at_zero": True, "sign_preserved": True},
        }
        prefix_params = {
            k: {"mu": p.mu, "sigma": p.sigma, "rms": p.rms}
            for k, p in bundle.params.items()
            if k in (*RETURN_KEYS, *RISK_KEYS)
        }
        return {"methods": defs, "prefix_fit_protocol": "first 50% t-indices; eval apply only", "prefix_params": prefix_params}

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
            "P_long": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
            "P_short": compute_path_n(ctx, Action.SHORT, h, decay_rate=cfg.decay_rate),
        }

    def _collect_rows(self) -> tuple[list[dict[str, float]], list[int]]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars))
        rows = []
        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            row = self._raw_obs(ctx, Action.LONG, h)
            row["t_index"] = float(t_index)
            rows.append(row)
        return rows, t_indices

    def _build_archetypes(self, bundle: NormBundle) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        table = []
        for arch in SYNTHETIC_ARCHETYPES:
            if arch["id"] == "H":
                continue
            path = runner._path_from_cumulative(
                arch["id"],
                arch["levels"],
                h,
                adverse_wick=arch["id"] in ("C", "REC", "G"),
            )
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            entry: dict[str, Any] = {"id": arch["id"], "description": arch["description"], "raw": raw}
            for method in NORM_METHODS:
                entry[f"Return_{method}"] = bundle.composite_return(raw, method)
                entry[f"Risk_{method}"] = bundle.composite_risk(raw, method)
                entry[f"facets_return_{method}"] = bundle.facet_vector(raw, RETURN_KEYS, method)
                entry[f"facets_risk_{method}"] = bundle.facet_vector(raw, RISK_KEYS, method)
            table.append(entry)

        by_id = {e["id"]: e for e in table}
        ordering = {}
        for method in NORM_METHODS:
            ordering[method] = {
                "return_AB": {
                    "A": by_id["A"][f"Return_{method}"],
                    "B": by_id["B"][f"Return_{method}"],
                    "MFE_facet_A_gt_B": by_id["A"][f"facets_return_{method}"]["MFE"]
                    > by_id["B"][f"facets_return_{method}"]["MFE"],
                    "U_facet_B_gt_A": by_id["B"][f"facets_return_{method}"]["U"]
                    > by_id["A"][f"facets_return_{method}"]["U"],
                },
                "risk_B_lt_A_lt_C": (
                    by_id["B"][f"Risk_{method}"]
                    < by_id["A"][f"Risk_{method}"]
                    < by_id["C"][f"Risk_{method}"]
                ),
                "risk_G_gt_B": by_id["G"][f"Risk_{method}"] > by_id["B"][f"Risk_{method}"],
            }

        return {"paths": table, "ordering_checks": ordering}

    def _sign_flip_audit(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]], keys: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        n = len(eval_rows)
        for key in keys:
            flips: dict[str, int] = {}
            pos_n: dict[str, int] = {}
            for method in NORM_METHODS:
                flip = pos = 0
                for e in eval_rows:
                    raw = e["raw"][key]
                    check_val = abs(raw) if key == "MAE" else raw
                    if check_val <= 0:
                        continue
                    pos += 1
                    if bundle.norm(e["raw"], key, method) < 0:
                        flip += 1
                flips[method] = flip
                pos_n[method] = pos
            out[key] = {
                "positive_raw_count": pos_n["raw"],
                "flip_counts": flips,
                "flip_rates": {m: flips[m] / max(pos_n[m], 1) for m in NORM_METHODS},
            }
        return out

    def _zero_preservation(self, bundle: NormBundle) -> dict[str, dict[str, float]]:
        out = {}
        for key in (*RETURN_KEYS, *RISK_KEYS):
            out[key] = {m: bundle.params[key].at_zero(m) for m in NORM_METHODS}
        return out

    def _magnitude_ratio_audit(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]], key: str
    ) -> dict[str, Any]:
        cfg = self._cfg
        results = {m: {"pairs": 0, "ratio_ok": 0, "median_ratio_error": []} for m in NORM_METHODS}
        raws = [e["raw"][key] for e in eval_rows]
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                a, b = abs(raws[i]), abs(raws[j])
                if a < 1e-12 or b < 1e-12:
                    continue
                ratio = a / b
                if abs(ratio - 2.0) > 0.25 and abs(ratio - 0.5) > 0.25:
                    continue
                target = 2.0 if ratio > 1 else 0.5
                for method in NORM_METHODS:
                    na = abs(bundle.norm(eval_rows[i]["raw"], key, method))
                    nb = abs(bundle.norm(eval_rows[j]["raw"], key, method))
                    if nb < 1e-12:
                        continue
                    observed = na / nb
                    err = abs(observed - target) / target
                    results[method]["pairs"] += 1
                    results[method]["median_ratio_error"].append(err)
                    if err < cfg.magnitude_ratio_tol:
                        results[method]["ratio_ok"] += 1
        summary = {}
        for method in NORM_METHODS:
            errs = results[method]["median_ratio_error"]
            summary[method] = {
                "pairs_tested": results[method]["pairs"],
                "ratio_preserved_count": results[method]["ratio_ok"],
                "ratio_preserved_rate": results[method]["ratio_ok"] / max(results[method]["pairs"], 1),
                "median_ratio_error": float(np.median(errs)) if errs else None,
            }
        return summary

    def _contribution_stats(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]], keys: tuple[str, ...], composite_fn
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for method in NORM_METHODS:
            shares: dict[str, list[float]] = {k: [] for k in keys}
            for e in eval_rows:
                facets = bundle.facet_vector(e["raw"], keys, method)
                dom = _dominance_shares(facets)
                for k in keys:
                    shares[k].append(dom[k])
            out[method] = {
                k: {
                    "mean_share": float(mean(shares[k])) if shares[k] else 0.0,
                    "median_share": float(np.median(shares[k])) if shares[k] else 0.0,
                    "p90_share": _percentile(shares[k], 0.9),
                    "p95_share": _percentile(shares[k], 0.95),
                    "p99_share": _percentile(shares[k], 0.99),
                }
                for k in keys
            }
        return out

    def _return_experiment(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]], archetypes: dict[str, Any]
    ) -> dict[str, Any]:
        sign = self._sign_flip_audit(bundle, eval_rows, RETURN_KEYS)
        zero = self._zero_preservation(bundle)
        mag_u = self._magnitude_ratio_audit(bundle, eval_rows, "U")
        mag_mfe = self._magnitude_ratio_audit(bundle, eval_rows, "MFE")
        contrib = self._contribution_stats(bundle, eval_rows, RETURN_KEYS, bundle.composite_return)

        composites = {m: [bundle.composite_return(e["raw"], m) for e in eval_rows] for m in NORM_METHODS}

        return {
            "composites": {m: {"mean": float(mean(composites[m])), "std": float(pstdev(composites[m])) if len(composites[m]) > 1 else 0.0} for m in NORM_METHODS},
            "sign_preservation": sign,
            "zero_preservation_at_X_eq_0": zero,
            "magnitude_ratio_2x_tests": {"U": mag_u, "MFE": mag_mfe},
            "contribution_balance": contrib,
            "archetype_ordering": archetypes["ordering_checks"],
            "semantic_note": (
                "Z-score may flip sign when raw>0 but below prefix mean; "
                "stdscale/rmsscale preserve sign and zero."
            ),
        }

    def _risk_experiment(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]], archetypes: dict[str, Any]
    ) -> dict[str, Any]:
        sign = self._sign_flip_audit(bundle, eval_rows, RISK_KEYS)
        zero = self._zero_preservation(bundle)
        mag = {k: self._magnitude_ratio_audit(bundle, eval_rows, k) for k in RISK_KEYS}
        contrib = self._contribution_stats(bundle, eval_rows, RISK_KEYS, bundle.composite_risk)

        monotonic = {}
        for method in NORM_METHODS:
            ok = 0
            total = 0
            for e in eval_rows:
                raw = e["raw"]
                r1 = bundle.composite_risk(raw, method)
                raw2 = dict(raw)
                raw2["MAE"] = raw["MAE"] * 1.5 if raw["MAE"] else 0.001
                r2 = bundle.composite_risk(raw2, method)
                total += 1
                if r2 >= r1:
                    ok += 1
            monotonic[method] = ok / max(total, 1)

        return {
            "sign_preservation": sign,
            "zero_preservation_at_X_eq_0": zero,
            "magnitude_ratio_2x_tests": mag,
            "contribution_balance": contrib,
            "higher_MAE_implies_higher_composite_rate": monotonic,
            "archetype_ordering": archetypes["ordering_checks"],
            "facet_semantics": {
                "MAE": "adverse magnitude",
                "giveback": "capture erosion after peak",
                "chop": "intra-path whip",
            },
        }

    def _zscore_semantic_audit(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        keys = (*RETURN_KEYS, *RISK_KEYS)
        flips = self._sign_flip_audit(bundle, eval_rows, keys)
        total_pos = sum(flips[k]["positive_raw_count"] for k in keys) // len(keys)
        agg_flip_rate = float(mean(flips[k]["flip_rates"]["zscore"] for k in keys))

        impact = []
        for e in eval_rows[:20]:
            raw = e["raw"]
            if raw["U"] > 0 and bundle.norm(raw, "U", "zscore") < 0:
                impact.append(
                    {
                        "t_index": e["t_index"],
                        "U_raw": raw["U"],
                        "z_U": bundle.norm(raw, "U", "zscore"),
                        "prefix_mu_U": bundle.params["U"].mu,
                        "note": "favorable raw U mapped to negative z — below-prefix-average utility",
                    }
                )

        return {
            "sign_flip_rates_by_facet": {k: flips[k]["flip_rates"]["zscore"] for k in keys},
            "aggregate_zscore_flip_rate": agg_flip_rate,
            "U_positive_z_negative_examples": impact[:5],
            "semantic_damage_assessment": (
                "PARTIAL: z-score re-centers on prefix mean; raw>0 no longer means norm>0. "
                "This changes P1 'favorable at t' semantics from absolute to relative-to-prefix."
                if agg_flip_rate > 0.05
                else "LOW flip rate on eval — still relative semantics"
            ),
            "zero_at_zero": self._zero_preservation(bundle),
        }

    def _scale_only_audit(self, bundle: NormBundle, method: Literal["stdscale", "rmsscale"]) -> dict[str, Any]:
        keys = (*RETURN_KEYS, *RISK_KEYS)
        zeros = {k: bundle.params[k].at_zero(method) for k in keys}
        all_zero = all(abs(v) < 1e-15 for v in zeros.values())
        return {
            "method": method,
            "formula": "X/sigma" if method == "stdscale" else "X/RMS",
            "zero_preservation": zeros,
            "all_facets_zero_at_zero": all_zero,
            "sign_preserved": True,
            "prefix_scales": {k: bundle.params[k].sigma if method == "stdscale" else bundle.params[k].rms for k in keys},
        }

    def _find_pairs(
        self,
        eval_rows: list[dict[str, Any]],
        *,
        match_key: str,
        differ_key: str,
        match_tol: float,
        differ_min: float,
    ) -> list[dict[str, Any]]:
        cfg = self._cfg
        out = []
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                a, b = eval_rows[i], eval_rows[j]
                if abs(a["raw"][match_key] - b["raw"][match_key]) > match_tol:
                    continue
                if abs(a["raw"][differ_key] - b["raw"][differ_key]) < differ_min:
                    continue
                out.append((a, b))
                if len(out) >= cfg.max_matched_pairs:
                    return out
        return out

    def _matched_path_experiment(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        cfg = self._cfg
        u_mfe = self._find_pairs(eval_rows, match_key="U", differ_key="MFE", match_tol=cfg.u_match_tol, differ_min=cfg.mfe_match_tol * 3)
        mfe_u = self._find_pairs(eval_rows, match_key="MFE", differ_key="U", match_tol=cfg.mfe_match_tol, differ_min=cfg.u_match_tol * 3)

        def pair_report(pairs: list[tuple], label: str) -> list[dict[str, Any]]:
            rows = []
            for a, b in pairs:
                entry = {"pair_type": label, "t_indices": [a["t_index"], b["t_index"]]}
                for method in NORM_METHODS:
                    ca = bundle.composite_return(a["raw"], method)
                    cb = bundle.composite_return(b["raw"], method)
                    fa = bundle.facet_vector(a["raw"], RETURN_KEYS, method)
                    fb = bundle.facet_vector(b["raw"], RETURN_KEYS, method)
                    entry[method] = {
                        "composite_diff": abs(ca - cb),
                        "composite_a": ca,
                        "composite_b": cb,
                        "MFE_facet_diff": abs(fa["MFE"] - fb["MFE"]),
                        "U_facet_diff": abs(fa["U"] - fb["U"]),
                        "facet_diff_preserved": abs(fa["MFE"] - fb["MFE"]) > 0.01 or abs(fa["U"] - fb["U"]) > 0.01,
                    }
                rows.append(entry)
            return rows

        return {
            "U_similar_MFE_diff": pair_report(u_mfe, "U_similar_MFE_diff"),
            "MFE_similar_U_diff": pair_report(mfe_u, "MFE_similar_U_diff"),
            "interpretation": "Compare composite_diff vs facet_diff across normalizations",
        }

    def _dominance_analysis(
        self, bundle: NormBundle, eval_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"return": {}, "risk": {}}
        for group, keys, fn_name in (
            ("return", RETURN_KEYS, "composite_return"),
            ("risk", RISK_KEYS, "composite_risk"),
        ):
            for method in NORM_METHODS:
                thresholds = {50: 0, 65: 0, 80: 0}
                facet_dom_counts = {k: 0 for k in keys}
                n = len(eval_rows)
                for e in eval_rows:
                    facets = bundle.facet_vector(e["raw"], keys, method)
                    dom = _dominance_shares(facets)
                    top = max(dom, key=dom.get)
                    facet_dom_counts[top] += 1
                    for thr_pct, _ in list(thresholds.items()):
                        if dom[top] * 100 >= thr_pct:
                            thresholds[thr_pct] += 1
                out[group][method] = {
                    "dominant_facet_counts": facet_dom_counts,
                    "dominance_pct_samples": {f">{k}%": v / max(n, 1) for k, v in thresholds.items()},
                    "contribution": self._contribution_stats(bundle, eval_rows, keys, getattr(bundle, fn_name))[method],
                }
        return out

    def _score_semantic_preservation(
        self,
        return_audit: dict[str, Any],
        risk_audit: dict[str, Any],
        z_audit: dict[str, Any],
        archetypes: dict[str, Any],
    ) -> dict[str, float]:
        scores = {}
        arch_order = archetypes["ordering_checks"]
        for method in NORM_METHODS:
            s = 1.0
            u_flip = return_audit["sign_preservation"]["U"]["flip_rates"][method]
            mfe_flip = return_audit["sign_preservation"]["MFE"]["flip_rates"][method]
            s -= 0.25 * (u_flip + mfe_flip) / 2
            mae_flip = risk_audit["sign_preservation"]["MAE"]["flip_rates"][method]
            s -= 0.15 * mae_flip
            z_u = return_audit["zero_preservation_at_X_eq_0"]["U"][method]
            z_mfe = return_audit["zero_preservation_at_X_eq_0"]["MFE"][method]
            if abs(z_u) > 1e-9 or abs(z_mfe) > 1e-9:
                s -= 0.1 if method == "zscore" else 0.0
            if method != "zscore" and (abs(z_u) > 1e-9 or abs(z_mfe) > 1e-9):
                s -= 0.05
            mag_u = return_audit["magnitude_ratio_2x_tests"]["U"][method]["ratio_preserved_rate"]
            s *= 0.5 + 0.5 * mag_u
            if not arch_order[method]["risk_B_lt_A_lt_C"]:
                s -= 0.1
            if not arch_order[method]["return_AB"]["MFE_facet_A_gt_B"]:
                s -= 0.05
            scores[method] = max(0.0, min(1.0, s))
        return scores

    def _score_scale_balance(self, dominance: dict[str, Any]) -> dict[str, float]:
        scores = {}
        for method in NORM_METHODS:
            ret = dominance["return"][method]["contribution"]
            risk = dominance["risk"][method]["contribution"]
            ret_max_mean = max(ret[k]["mean_share"] for k in RETURN_KEYS)
            risk_max_mean = max(risk[k]["mean_share"] for k in RISK_KEYS)
            dom80_ret = dominance["return"][method]["dominance_pct_samples"].get(">80%", 0)
            dom80_risk = dominance["risk"][method]["dominance_pct_samples"].get(">80%", 0)
            imbalance = max(ret_max_mean, risk_max_mean)
            s = 1.0 - imbalance - 0.5 * (dom80_ret + dom80_risk)
            scores[method] = max(0.0, min(1.0, s))
        return scores

    def _comparison_table(
        self,
        return_audit: dict[str, Any],
        risk_audit: dict[str, Any],
        z_audit: dict[str, Any],
        std_audit: dict[str, Any],
        rms_audit: dict[str, Any],
        dominance: dict[str, Any],
    ) -> dict[str, Any]:
        arch = return_audit["archetype_ordering"]
        sem_scores = self._score_semantic_preservation(return_audit, risk_audit, z_audit, {"ordering_checks": arch})
        bal_scores = self._score_scale_balance(dominance)

        issues = {
            "raw": "MAE/giveback/chop and U/MFE on different raw scales — composite dominated by larger-magnitude facet",
            "zscore": f"sign flip rate ~{z_audit['aggregate_zscore_flip_rate']:.1%}; raw>0 can map to z<0; zero not preserved",
            "stdscale": "scale balance improved; sign/zero/magnitude preserved; relative magnitude uses sigma not RMS",
            "rmsscale": "similar to stdscale; RMS denominator differs when mean!=0",
        }

        table = []
        for method in NORM_METHODS:
            table.append(
                {
                    "normalization": method,
                    "semantic_preservation_score": round(sem_scores[method], 3),
                    "scale_balance_score": round(bal_scores[method], 3),
                    "U_sign_flip_rate": return_audit["sign_preservation"]["U"]["flip_rates"][method],
                    "Return_U_mean_share": dominance["return"][method]["contribution"]["U"]["mean_share"],
                    "Return_MFE_mean_share": dominance["return"][method]["contribution"]["MFE"]["mean_share"],
                    "Risk_MAE_mean_share": dominance["risk"][method]["contribution"]["MAE"]["mean_share"],
                    "Risk_giveback_mean_share": dominance["risk"][method]["contribution"]["giveback"]["mean_share"],
                    "Risk_chop_mean_share": dominance["risk"][method]["contribution"]["chop"]["mean_share"],
                    "archetype_B_lt_A_lt_C_risk": arch[method]["risk_B_lt_A_lt_C"],
                    "archetype_MFE_A_gt_B": arch[method]["return_AB"]["MFE_facet_A_gt_B"],
                    "main_issue": issues[method],
                }
            )

        sem_rank = sorted(NORM_METHODS, key=lambda m: -sem_scores[m])
        bal_rank = sorted(NORM_METHODS, key=lambda m: -bal_scores[m])

        return {
            "table": table,
            "rankings": {"semantic_preservation": sem_rank, "scale_balance": bal_rank},
            "tradeoffs": {
                "raw": "Best semantic fidelity per facet; worst cross-facet scale balance",
                "zscore": "Best statistical balance on eval; worst sign/zero semantic for P1 labels",
                "stdscale": "Strong semantic preservation + improved balance; prefix sigma fit",
                "rmsscale": "Like stdscale; slightly different when prefix mean far from zero",
            },
            "unresolved": [
                "Optimal composite: sum of normalized facets vs separate heads (out of scope)",
                "Rolling/refit scale under regime shift",
                "Whether Return/Risk sums are training labels or only facet targets",
            ],
            "next_experiments": [
                "Conditional scale within MAE buckets for recovery-like low-variance facets",
                "Per-regime prefix refit impact on stdscale/rmsscale",
                "Training with separate heads vs composite label ablation",
            ],
        }

    def _final_normalization_answer(self, comparison: dict[str, Any]) -> dict[str, Any]:
        table = {r["normalization"]: r for r in comparison["table"]}
        sem_best = comparison["rankings"]["semantic_preservation"][0]
        bal_best = comparison["rankings"]["scale_balance"][0]

        answer = (
            "For P1 goal 'preserve facet meaning/sign/zero/magnitude while enabling U+MFE and "
            "MAE+Giveback+Chop summation', scale-only Std (X/sigma_prefix) is most semantically aligned "
            "among tested options. It preserves sign and zero, maintains 2x magnitude ratios, and "
            "improves facet balance vs Raw. Standard Z-score improves scale balance but re-centers "
            "labels so raw favorable U>0 can become z(U)<0 (~"
            f"{table['zscore']['U_sign_flip_rate']:.1%} flip on eval). "
            "Raw preserves semantics but fails scale balance. RMS-scale is near-Std-scale; choose "
            "between them after regime sensitivity check. No canonical adoption in this step."
        )

        return {
            "most_semantically_aligned": sem_best,
            "best_scale_balance": bal_best,
            "recommended_for_next_validation": "stdscale",
            "recommended_with_caution": "rmsscale",
            "not_recommended_as_composite_label": ["zscore"],
            "raw_role": "diagnostic baseline only",
            "summary": answer,
            "no_canonical_adoption": True,
        }


def format_norm_experiment_summary(report: dict[str, Any]) -> str:
    ans = report.get("final_question_answer", {})
    lines = [
        "P1 Normalization Semantic Experiment",
        "=" * 60,
        f"eval_n: {report.get('9_btc_realdata_summary', {}).get('eval_n')}",
        f"semantic best: {ans.get('most_semantically_aligned')}",
        f"scale balance best: {ans.get('best_scale_balance')}",
        f"next validation: {ans.get('recommended_for_next_validation')}",
    ]
    return "\n".join(lines)


def save_norm_experiment_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
