"""P1 Return/Risk Composite Validation — scale-only X/sigma, equal-weight sum (analysis-only).

Validates whether fixed facets U+MFE and MAE+Giveback+Chop form meaningful composites
after prefix-fit scale-only normalization. Does NOT modify canonical code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_normalization_semantic_experiment import (
    FIXED_STRUCTURE,
    RETURN_KEYS,
    RISK_KEYS,
    NormBundle,
    _dominance_shares,
    _percentile,
)
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


@dataclass
class P1CompositeValidationConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    dominance_threshold: float = 0.65
    false_equiv_risk_tol: float = 0.15
    max_exemplars: int = 5
    pair_window: int = 80


INTENDED_RETURN_SEMANTICS = {
    "A": "high max opportunity; spike then giveback",
    "B": "stable sustained favorable movement (grind)",
    "C": "high initial opportunity then catastrophic failure",
    "G": "round-trip whip; return may resemble B",
}

INTENDED_RISK_SEMANTICS = {
    "B": "most stable / lowest acceptable risk",
    "A": "spike/giveback erosion risk",
    "G": "whip/chop risk above B",
    "C": "catastrophic adverse risk highest",
}


class P1CompositeValidationRunner:
    """Validate scale-only equal-weight Return/Risk composites on real BTC."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1CompositeValidationConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1CompositeValidationConfig()
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

        ret_dom = self._return_dominance(eval_rows)
        abc = self._return_archetype_analysis(bundle)
        ret_failures = self._return_path_exemplars(eval_rows)
        risk_contrib = self._risk_contribution(eval_rows)
        risk_arch = self._risk_archetype_analysis(bundle)
        risk_scalar = self._risk_scalar_adequacy(eval_rows)
        risk_failures = self._risk_path_exemplars(eval_rows)
        ranking = self._ranking_consistency(eval_rows)
        verdict = self._final_verdict(ret_dom, abc, risk_contrib, risk_arch, risk_scalar, ret_failures, risk_failures)

        return {
            "audit": "P1 Return/Risk Composite Validation (scale-only X/sigma)",
            "fixed_structure": FIXED_STRUCTURE,
            "normalization": "X_scaled = X / sigma_prefix (no mean subtraction)",
            "aggregation": "equal-weight simple sum per composite",
            "primary_evidence": "BTCUSDT real eval",
            "1_executive_summary": {
                "eval_n": len(eval_rows),
                "final_verdict": verdict["choice"],
                "headline": verdict["summary"],
                "no_canonical_adoption": True,
            },
            "2_return_composite_results": {
                "formula": "Return_composite = U/sigma_U + MFE/sigma_MFE",
                "ranking_consistency": ranking["return"],
                "dominance": ret_dom,
            },
            "3_U_vs_MFE_dominance": ret_dom,
            "4_ABC_archetype_return": abc,
            "5_btc_return_failure_exemplars": ret_failures,
            "6_risk_composite_results": {
                "formula": "Risk_composite = MAE/sigma + Giveback/sigma + Chop/sigma",
                "ranking_consistency": ranking["risk"],
                "contribution": risk_contrib,
            },
            "7_MAE_giveback_chop_contribution": risk_contrib,
            "8_BACG_archetype_risk": risk_arch,
            "9_btc_risk_failure_exemplars": risk_failures,
            "10_false_equivalence_cases": risk_scalar["false_equivalence"],
            "11_composite_semantic_adequacy": {
                "return": verdict["return_adequacy"],
                "risk": verdict["risk_adequacy"],
            },
            "12_equal_weight_judgment": verdict["equal_weight"],
            "13_normalization_vs_aggregation": verdict["norm_vs_agg"],
            "14_next_steps": verdict["next_steps"],
            "final_verdict_detail": verdict,
            "data_protocol": {
                "market": describe_market_data(self._data),
                "prefix_n": split,
                "eval_n": len(eval_rows),
                "sigma_prefix": {k: bundle.params[k].sigma for k in (*RETURN_KEYS, *RISK_KEYS)},
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
            "path_efficiency": obs.path_efficiency,
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
            rows.append(self._raw_obs(ctx, Action.LONG, h))
        return rows, t_indices

    def _scale(self, bundle: NormBundle, raw: dict[str, float], key: str) -> float:
        return bundle.norm(raw, key, SCALE_METHOD)

    def _enrich(self, raw: dict[str, float], bundle: NormBundle, t_index: int) -> dict[str, Any]:
        u_s = self._scale(bundle, raw, "U")
        mfe_s = self._scale(bundle, raw, "MFE")
        mae_s = self._scale(bundle, raw, "MAE")
        gb_s = self._scale(bundle, raw, "giveback")
        chop_s = self._scale(bundle, raw, "chop")
        ret_c = u_s + mfe_s
        risk_c = mae_s + gb_s + chop_s
        ret_sh = _dominance_shares({"U": u_s, "MFE": mfe_s})
        risk_sh = _dominance_shares({"MAE": mae_s, "giveback": gb_s, "chop": chop_s})
        return {
            "t_index": t_index,
            "timestamp": str(self._data.bars[t_index].start),
            "raw": raw,
            "U_scaled": u_s,
            "MFE_scaled": mfe_s,
            "MAE_scaled": mae_s,
            "giveback_scaled": gb_s,
            "chop_scaled": chop_s,
            "Return_composite": ret_c,
            "Risk_composite": risk_c,
            "U_share": ret_sh["U"],
            "MFE_share": ret_sh["MFE"],
            "MAE_share": risk_sh["MAE"],
            "giveback_share": risk_sh["giveback"],
            "chop_share": risk_sh["chop"],
            "path_pattern": self._classify_path_pattern(raw),
            "path_ascii": self._path_ascii(t_index),
            "regime_vol": self._regime_vol(t_index),
        }

    def _path_ascii(self, t_index: int) -> str:
        h = self._cfg.reward_horizon
        ctx = self._builder.build(t_index)
        rets = []
        for k in range(1, h + 1):
            rets.append(ctx.return_from_t(k))
        if not rets:
            return "flat"
        chars = []
        for r in rets:
            if r > 0.0005:
                chars.append("^")
            elif r < -0.0005:
                chars.append("v")
            else:
                chars.append("-")
        return "t>" + "".join(chars)

    def _regime_vol(self, t_index: int) -> float:
        lookback = 20
        start = max(0, t_index - lookback)
        closes = [self._data.bars[i].close for i in range(start, t_index + 1)]
        if len(closes) < 2:
            return 0.0
        rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
        return float(pstdev(rets)) if len(rets) > 1 else 0.0

    def _classify_path_pattern(self, raw: dict[str, float]) -> str:
        u, mfe, gb, term, chop = raw["U"], raw["MFE"], raw["giveback"], raw["terminal"], raw["chop"]
        if mfe > u * 1.3 and gb > 0.4:
            return "spike_then_giveback"
        if u > mfe * 0.85 and gb < 0.25 and term > 0:
            return "smooth_rise_hold"
        if mfe > 0 and term < -0.001 and u < mfe * 0.6:
            return "spike_then_crash"
        if chop > 0.2 and abs(term) < 0.003:
            return "round_trip_whip"
        if raw["MAE"] > 0.002 and term < 0:
            return "sustained_adverse"
        if raw["MAE"] > 0.001 and term > 0:
            return "adverse_then_recovery"
        return "mixed"

    def _sample_record(self, e: dict[str, Any]) -> dict[str, Any]:
        r = e["raw"]
        return {
            "t_index": e["t_index"],
            "timestamp": e["timestamp"],
            "path_ascii": e["path_ascii"],
            "path_pattern": e["path_pattern"],
            "U": r["U"],
            "MFE": r["MFE"],
            "MAE": r["MAE"],
            "giveback": r["giveback"],
            "chop": r["chop"],
            "terminal": r["terminal"],
            "U_scaled": e["U_scaled"],
            "MFE_scaled": e["MFE_scaled"],
            "MAE_scaled": e["MAE_scaled"],
            "giveback_scaled": e["giveback_scaled"],
            "chop_scaled": e["chop_scaled"],
            "Return_composite": e["Return_composite"],
            "Risk_composite": e["Risk_composite"],
            "U_share_pct": round(100 * e["U_share"], 1),
            "MFE_share_pct": round(100 * e["MFE_share"], 1),
            "MAE_share_pct": round(100 * e["MAE_share"], 1),
            "giveback_share_pct": round(100 * e["giveback_share"], 1),
            "chop_share_pct": round(100 * e["chop_share"], 1),
            "path_explanation": self._explain_sample(e),
        }

    def _explain_sample(self, e: dict[str, Any]) -> str:
        pat = e["path_pattern"]
        if pat == "spike_then_giveback":
            return "Rise-spike then partial giveback; MFE high, U moderate"
        if pat == "smooth_rise_hold":
            return "Gradual rise and hold; U dominates Return composite"
        if pat == "spike_then_crash":
            return "Early spike then crash; MFE may inflate Return despite bad terminal"
        if pat == "round_trip_whip":
            return "Up-down-up whip; chop drives Risk"
        if pat == "sustained_adverse":
            return "Sustained drop; MAE drives Risk"
        if pat == "adverse_then_recovery":
            return "Dip then recovery; MAE high but terminal may be OK"
        return "Mixed path pattern"

    def _return_dominance(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        u_sh = [e["U_share"] for e in eval_rows]
        mfe_sh = [e["MFE_share"] for e in eval_rows]
        u_dom = sum(1 for s in u_sh if s > 0.5) / len(eval_rows)
        mfe_dom = sum(1 for s in mfe_sh if s > 0.5) / len(eval_rows)
        u_extreme = sum(1 for s in u_sh if s > 0.65) / len(eval_rows)

        def bucket(mask: list[bool]) -> dict[str, float]:
            subset = [eval_rows[i] for i in range(len(eval_rows)) if mask[i]]
            if not subset:
                return {}
            return {
                "n": len(subset),
                "mean_U_share": float(mean(e["U_share"] for e in subset)),
                "mean_MFE_share": float(mean(e["MFE_share"] for e in subset)),
            }

        u_abs = [abs(e["U_scaled"]) for e in eval_rows]
        mfe_abs = [abs(e["MFE_scaled"]) for e in eval_rows]
        u_p99 = _percentile(u_abs, 0.99)
        mfe_p99 = _percentile(mfe_abs, 0.99)
        u_p95 = _percentile(u_abs, 0.95)
        mfe_p95 = _percentile(mfe_abs, 0.95)

        ret_c = [e["Return_composite"] for e in eval_rows]
        u_only = [e["U_scaled"] for e in eval_rows]
        mfe_only = [e["MFE_scaled"] for e in eval_rows]

        high_vol = _percentile([e["regime_vol"] for e in eval_rows], 0.75)
        regime_high = bucket([e["regime_vol"] >= high_vol for e in eval_rows])
        regime_low = bucket([e["regime_vol"] < _percentile([x["regime_vol"] for x in eval_rows], 0.25) for e in eval_rows])

        u_dominates_ex = sorted(
            [self._sample_record(e) for e in eval_rows if e["U_share"] > 0.65],
            key=lambda x: -x["U_share_pct"],
        )[: self._cfg.max_exemplars]
        mfe_dominates_ex = sorted(
            [self._sample_record(e) for e in eval_rows if e["MFE_share"] > 0.65],
            key=lambda x: -x["MFE_share_pct"],
        )[: self._cfg.max_exemplars]

        return {
            "overall": {
                "mean_U_share": float(mean(u_sh)),
                "median_U_share": float(np.median(u_sh)),
                "mean_MFE_share": float(mean(mfe_sh)),
                "median_MFE_share": float(np.median(mfe_sh)),
                "U_dominates_gt50_pct": u_dom,
                "MFE_dominates_gt50_pct": mfe_dom,
                "U_dominates_gt65_pct": u_extreme,
            },
            "tail_buckets": {
                "U_scaled_top1pct": bucket([v >= u_p99 for v in u_abs]),
                "U_scaled_top5pct": bucket([v >= u_p95 for v in u_abs]),
                "MFE_scaled_top1pct": bucket([v >= mfe_p99 for v in mfe_abs]),
                "MFE_scaled_top5pct": bucket([v >= mfe_p95 for v in mfe_abs]),
                "positive_U": bucket([e["raw"]["U"] > 0 for e in eval_rows]),
                "negative_U": bucket([e["raw"]["U"] <= 0 for e in eval_rows]),
                "positive_MFE": bucket([e["raw"]["MFE"] > 0 for e in eval_rows]),
                "high_regime_vol": regime_high,
                "low_regime_vol": regime_low,
            },
            "correlations": {
                "Return_composite_vs_U_scaled": _pearson(ret_c, u_only),
                "Return_composite_vs_MFE_scaled": _pearson(ret_c, mfe_only),
                "Return_composite_vs_U_plus_MFE_linear": "exact sum by construction",
            },
            "U_dominance_exemplars": u_dominates_ex,
            "MFE_dominance_exemplars": mfe_dominates_ex,
            "finding": (
                f"U share mean {mean(u_sh):.1%} vs MFE {mean(mfe_sh):.1%}; "
                f"U>65% dominance in {u_extreme:.1%} of eval; "
                f"corr(composite,U)={_pearson(ret_c, u_only):.3f}, "
                f"corr(composite,MFE)={_pearson(ret_c, mfe_only):.3f}"
            ),
        }

    def _return_archetype_analysis(self, bundle: NormBundle) -> dict[str, Any]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        paths = {}
        for arch in SYNTHETIC_ARCHETYPES:
            if arch["id"] not in ("A", "B", "C", "G"):
                continue
            path = runner._path_from_cumulative(
                arch["id"], arch["levels"], h, adverse_wick=arch["id"] in ("C", "G")
            )
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            u_s, mfe_s = self._scale(bundle, raw, "U"), self._scale(bundle, raw, "MFE")
            paths[arch["id"]] = {
                "intended_semantics": INTENDED_RETURN_SEMANTICS[arch["id"]],
                "raw": raw,
                "U_scaled": u_s,
                "MFE_scaled": mfe_s,
                "Return_composite": u_s + mfe_s,
                "U_share": abs(u_s) / (abs(u_s) + abs(mfe_s) + 1e-12),
                "MFE_share": abs(mfe_s) / (abs(u_s) + abs(mfe_s) + 1e-12),
                "explanation": self._archetype_return_explain(arch["id"], raw, u_s, mfe_s),
            }

        rankings = {
            "U_raw": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["raw"]["U"])],
            "MFE_raw": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["raw"]["MFE"])],
            "U_scaled": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["U_scaled"])],
            "MFE_scaled": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["MFE_scaled"])],
            "Return_composite": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["Return_composite"])],
        }

        conflicts = []
        if rankings["U_raw"][0] == "B" and rankings["MFE_raw"][0] == "A":
            conflicts.append("U ranks B first (grind), MFE ranks A first (spike) — facet split preserved")
        if rankings["Return_composite"][0] == "B":
            conflicts.append("Composite ranks B first — grind/sustained U dominates over A spike potential")
        if paths["C"]["Return_composite"] > paths["B"]["Return_composite"]:
            conflicts.append("C composite exceeds B — spike MFE may overrate failed path")

        return {
            "paths": paths,
            "rankings": rankings,
            "intended_vs_actual": conflicts,
            "semantic_conflict": len(conflicts) > 0,
        }

    def _archetype_return_explain(self, pid: str, raw: dict, u_s: float, mfe_s: float) -> str:
        if pid == "A":
            return f"Spike to peak (MFE={raw['MFE']:.4f}) then giveback; U={raw['U']:.4f}; scaled MFE share {abs(mfe_s)/(abs(u_s)+abs(mfe_s)):.0%}"
        if pid == "B":
            return f"Grind hold; U={raw['U']:.4f} > A U; composite favors sustained utility"
        if pid == "C":
            return f"Spike MFE={raw['MFE']:.4f} but terminal={raw['terminal']:.4f}; composite may still be elevated"
        if pid == "G":
            return f"Round-trip; U and MFE moderate; chop affects Risk not Return"
        return ""

    def _return_path_exemplars(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        patterns = {
            "spike_then_giveback": [],
            "smooth_rise_hold": [],
            "spike_then_crash": [],
        }
        for e in eval_rows:
            p = e["path_pattern"]
            if p in patterns and len(patterns[p]) < cfg.max_exemplars:
                patterns[p].append(self._sample_record(e))

        anomalies = []
        for e in eval_rows:
            if e["path_pattern"] == "spike_then_crash" and e["Return_composite"] > 2.0:
                anomalies.append(self._sample_record(e))
            if e["path_pattern"] == "smooth_rise_hold" and e["MFE_share"] < 0.2:
                anomalies.append(self._sample_record(e))
            if len(anomalies) >= cfg.max_exemplars:
                break

        return {
            "by_path_pattern": patterns,
            "anomalies": anomalies[: cfg.max_exemplars],
            "note": "Each record includes path_ascii (t> bar directions) and facet shares",
        }

    def _risk_contribution(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        keys = ("MAE_scaled", "giveback_scaled", "chop_scaled")
        share_keys = ("MAE_share", "giveback_share", "chop_share")
        stats = {}
        for k, sk in zip(("MAE", "giveback", "chop"), share_keys):
            vals = [e[f"{k.lower()}_scaled" if k != "MAE" else "MAE_scaled"] for e in eval_rows]
            shares = [e[sk] for e in eval_rows]
            stats[k] = {
                "mean": float(mean(vals)),
                "median": float(np.median(vals)),
                "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                "mean_abs_contribution": float(mean(abs(v) for v in vals)),
                "mean_share": float(mean(shares)),
                "median_share": float(np.median(shares)),
                "p90_share": _percentile(shares, 0.9),
                "p99_share": _percentile(shares, 0.99),
            }

        dom_counts = {"MAE": 0, "giveback": 0, "chop": 0}
        for e in eval_rows:
            sh = {"MAE": e["MAE_share"], "giveback": e["giveback_share"], "chop": e["chop_share"]}
            dom_counts[max(sh, key=sh.get)] += 1

        by_pattern: dict[str, dict[str, float]] = {}
        for e in eval_rows:
            pat = e["path_pattern"]
            if pat not in by_pattern:
                by_pattern[pat] = {"n": 0, "MAE": 0.0, "giveback": 0.0, "chop": 0.0}
            by_pattern[pat]["n"] += 1
            by_pattern[pat]["MAE"] += e["MAE_share"]
            by_pattern[pat]["giveback"] += e["giveback_share"]
            by_pattern[pat]["chop"] += e["chop_share"]
        for pat in by_pattern:
            n = by_pattern[pat]["n"]
            for k in ("MAE", "giveback", "chop"):
                by_pattern[pat][k] /= n

        return {
            "facet_stats": stats,
            "dominant_facet_sample_counts": dom_counts,
            "dominant_facet_pct": {k: v / len(eval_rows) for k, v in dom_counts.items()},
            "by_path_pattern_mean_shares": by_pattern,
            "MAE_ignored_concern": stats["MAE"]["mean_share"] < 0.15,
            "giveback_dominates": stats["giveback"]["mean_share"] > 0.4,
            "chop_tail": stats["chop"]["p99_share"],
        }

    def _risk_archetype_analysis(self, bundle: NormBundle) -> dict[str, Any]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        ids = ("B", "A", "G", "C")
        paths = {}
        for pid in ids:
            arch = next(a for a in SYNTHETIC_ARCHETYPES if a["id"] == pid)
            path = runner._path_from_cumulative(
                pid, arch["levels"], h, adverse_wick=pid in ("C", "G")
            )
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            mae_s = self._scale(bundle, raw, "MAE")
            gb_s = self._scale(bundle, raw, "giveback")
            chop_s = self._scale(bundle, raw, "chop")
            paths[pid] = {
                "intended": INTENDED_RISK_SEMANTICS[pid],
                "MAE_scaled": mae_s,
                "giveback_scaled": gb_s,
                "chop_scaled": chop_s,
                "Risk_composite": mae_s + gb_s + chop_s,
                "raw": raw,
                "pairwise_analysis": {},
            }

        rankings = {
            "MAE_scaled": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["MAE_scaled"])],
            "giveback_scaled": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["giveback_scaled"])],
            "chop_scaled": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["chop_scaled"])],
            "Risk_composite": [k for k, _ in sorted(paths.items(), key=lambda x: -x[1]["Risk_composite"])],
        }

        b, a, g, c = paths["B"], paths["A"], paths["G"], paths["C"]
        pairwise = {
            "B_vs_A": {
                "giveback_diff": a["giveback_scaled"] - b["giveback_scaled"],
                "composite_diff": a["Risk_composite"] - b["Risk_composite"],
                "driver": "giveback" if a["giveback_scaled"] > b["giveback_scaled"] else "other",
            },
            "B_vs_G": {
                "chop_diff": g["chop_scaled"] - b["chop_scaled"],
                "composite_diff": g["Risk_composite"] - b["Risk_composite"],
                "driver": "chop" if g["chop_scaled"] > b["chop_scaled"] else "other",
            },
            "A_vs_C": {
                "mae_diff": c["MAE_scaled"] - a["MAE_scaled"],
                "giveback_diff": c["giveback_scaled"] - a["giveback_scaled"],
                "composite_diff": c["Risk_composite"] - a["Risk_composite"],
                "driver": "MAE+giveback catastrophic" if c["Risk_composite"] > a["Risk_composite"] else "broken",
            },
        }

        expected_order = ["B", "A", "G", "C"]
        comp_order = rankings["Risk_composite"]
        order_ok = comp_order.index("B") < comp_order.index("A") < comp_order.index("G") < comp_order.index("C")

        return {
            "paths": paths,
            "rankings": rankings,
            "pairwise": pairwise,
            "expected_B_lt_A_lt_G_lt_C": order_ok,
            "order_break_reason": None if order_ok else f"actual composite order: {comp_order}",
        }

    def _risk_scalar_adequacy(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        cfg = self._cfg
        false_equiv = []
        cancellation = []
        dominance = []
        mechanism_collapse = []

        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                a, b = eval_rows[i], eval_rows[j]
                if abs(a["Risk_composite"] - b["Risk_composite"]) > cfg.false_equiv_risk_tol:
                    continue
                fa = {"MAE": a["MAE_scaled"], "giveback": a["giveback_scaled"], "chop": a["chop_scaled"]}
                fb = {"MAE": b["MAE_scaled"], "giveback": b["giveback_scaled"], "chop": b["chop_scaled"]}
                dom_a = max(fa, key=lambda k: abs(fa[k]))
                dom_b = max(fb, key=lambda k: abs(fb[k]))
                if dom_a != dom_b:
                    false_equiv.append(
                        {
                            "case": "false_equivalence",
                            "t_indices": [a["t_index"], b["t_index"]],
                            "Risk_composite": [a["Risk_composite"], b["Risk_composite"]],
                            "facets_a": fa,
                            "facets_b": fb,
                            "dominant_a": dom_a,
                            "dominant_b": dom_b,
                            "paths": [self._sample_record(a), self._sample_record(b)],
                        }
                    )
                if len(false_equiv) >= cfg.max_exemplars:
                    break
            if len(false_equiv) >= cfg.max_exemplars:
                break

        for e in eval_rows:
            sh = e["MAE_share"], e["giveback_share"], e["chop_share"]
            if max(sh) > cfg.dominance_threshold:
                dominance.append(self._sample_record(e))
            if len(dominance) >= cfg.max_exemplars:
                break

        for e in eval_rows:
            facets = [e["MAE_scaled"], e["giveback_scaled"], e["chop_scaled"]]
            if max(facets) > 1.5 and min(facets) < 0.3 and abs(e["Risk_composite"] - sum(facets) / 3) < 0.5:
                cancellation.append(self._sample_record(e))
            if len(cancellation) >= cfg.max_exemplars:
                break

        return {
            "false_equivalence": false_equiv,
            "facet_dominance_exemplars": dominance[: cfg.max_exemplars],
            "cancellation_like": cancellation[: cfg.max_exemplars],
            "mechanism_collapse_note": (
                "Equal-weight sum can assign similar Risk_composite to MAE-driven vs chop-driven paths"
            ),
        }

    def _risk_path_exemplars(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        patterns = ["round_trip_whip", "sustained_adverse", "adverse_then_recovery", "spike_then_giveback"]
        out = {}
        for pat in patterns:
            out[pat] = [
                self._sample_record(e)
                for e in eval_rows
                if e["path_pattern"] == pat
            ][: self._cfg.max_exemplars]
        return out

    def _ranking_consistency(self, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
        ret_c = [e["Return_composite"] for e in eval_rows]
        risk_c = [e["Risk_composite"] for e in eval_rows]
        raw = [e["raw"] for e in eval_rows]
        return {
            "return": {
                "corr_composite_U": _pearson(ret_c, [r["U"] for r in raw]),
                "corr_composite_MFE": _pearson(ret_c, [r["MFE"] for r in raw]),
                "corr_composite_terminal": _pearson(ret_c, [r["terminal"] for r in raw]),
                "corr_composite_max_favorable_MFE": _pearson(ret_c, [r["MFE"] for r in raw]),
            },
            "risk": {
                "corr_composite_MAE": _pearson(risk_c, [abs(r["MAE"]) for r in raw]),
                "corr_composite_giveback": _pearson(risk_c, [r["giveback"] for r in raw]),
                "corr_composite_chop": _pearson(risk_c, [r["chop"] for r in raw]),
                "corr_composite_terminal": _pearson(risk_c, [r["terminal"] for r in raw]),
                "corr_composite_adverse_MAE": _pearson(risk_c, [abs(r["MAE"]) for r in raw]),
            },
        }

    def _final_verdict(
        self,
        ret_dom: dict[str, Any],
        abc: dict[str, Any],
        risk_contrib: dict[str, Any],
        risk_arch: dict[str, Any],
        risk_scalar: dict[str, Any],
        ret_failures: dict[str, Any],
        risk_failures: dict[str, Any],
    ) -> dict[str, Any]:
        u_mean_share = ret_dom["overall"]["mean_U_share"]
        composite_b_first = abc["rankings"]["Return_composite"][0] == "B"
        risk_order_ok = risk_arch["expected_B_lt_A_lt_G_lt_C"]
        false_eq_n = len(risk_scalar["false_equivalence"])
        mae_share = risk_contrib["facet_stats"]["MAE"]["mean_share"]

        return_adequacy = (
            "PARTIAL: composite correlates with both U and MFE but grind (B) can outrank spike potential (A) on sum; "
            "C may retain elevated composite from MFE despite failure"
            if composite_b_first or abc["semantic_conflict"]
            else "SUPPORTED on archetype ordering"
        )
        risk_adequacy = (
            "PARTIAL: facet shares balanced on mean but false equivalence and dominance cases exist; "
            f"MAE mean share only {mae_share:.1%}"
            if false_eq_n > 0 or not risk_order_ok or mae_share < 0.15
            else "SUPPORTED"
        )

        if composite_b_first or not risk_order_ok or false_eq_n >= 3:
            choice = "B"
            summary = (
                "Facet structure (U+MFE, MAE+GB+Chop) retained; equal-weight X/sigma composite shows semantic conflicts "
                "(Return: B-first grind bias; Risk: false equivalence / ordering breaks). "
                "Recommend separate facet heads or non-equal weights - not canonical change in this step."
            )
        elif mae_share < 0.12:
            choice = "B"
            summary = "MAE contribution diluted in composite; aggregation/weighting revision needed."
        else:
            choice = "A"
            summary = "Composites broadly adequate on BTC eval with noted edge cases."

        return {
            "choice": choice,
            "summary": summary,
            "return_adequacy": return_adequacy,
            "risk_adequacy": risk_adequacy,
            "equal_weight": (
                "PARTIAL: equal weight assumes facets equally important; U dominates Return in "
                f"{ret_dom['overall']['U_dominates_gt50_pct']:.0%} samples; chop/MAE shares vary by path type"
            ),
            "norm_vs_agg": {
                "normalization": "X/sigma preserves sign/zero; adequate for facet scaling",
                "aggregation": "simple sum loses mechanism identity (false equivalence, dominance)",
            },
            "next_steps": [
                "Train separate heads vs composite label ablation",
                "Test non-equal weights tied to facet semantics (not correlation-optimal)",
                "Report facet vectors at inference even if composite used for ranking",
            ],
        }


def format_composite_validation_summary(report: dict[str, Any]) -> str:
    s = report.get("1_executive_summary", {})
    lines = [
        "P1 Composite Validation (X/sigma)",
        "=" * 60,
        f"eval_n: {s.get('eval_n')}",
        f"verdict: {s.get('final_verdict')} - {(s.get('headline') or '')[:80]}",
    ]
    return "\n".join(lines)


def save_composite_validation_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
