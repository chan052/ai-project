"""P1 Return/Risk Weighting Robustness Validation (analysis-only).

Sweeps U:MFE and MAE:Giveback weights on X/sigma_prefix composites.
Goal: find robust semantic regions, NOT equal 50:50 contribution.

Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

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
RISK_ARCH_ORDER = ("B", "A", "REC", "C")  # intended clean < giveback < severe (hypothesis)
WEIGHT_GRID = tuple((round(i / 10, 1), round(1 - i / 10, 1)) for i in range(10, -1, -1))


def _spearman(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank_list(ids: tuple[str, ...], values: dict[str, float], *, reverse: bool = True) -> list[str]:
    return sorted(ids, key=lambda i: values[i], reverse=reverse)


def _kendall_distance(rank_a: list[str], rank_b: list[str]) -> float:
    if rank_a == rank_b:
        return 0.0
    pairs = 0
    discord = 0
    for i in range(len(rank_a)):
        for j in range(i + 1, len(rank_a)):
            pairs += 1
            ai, aj = rank_a[i], rank_a[j]
            bi = rank_b.index(ai)
            bj = rank_b.index(aj)
            if (i < j) != (bi < bj):
                discord += 1
    return discord / max(pairs, 1)


@dataclass
class P1WeightingRobustnessConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    false_equiv_tol: float = 0.2
    u_match_tol: float = 0.15
    max_false_pairs: int = 200
    pair_window: int = 80
    min_failure_cases: int = 10
    max_exemplars: int = 3
    dominance_levels: tuple[float, ...] = (0.65, 0.80, 0.90)
    quantile_buckets: int = 5


class P1WeightingRobustnessRunner:
    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1WeightingRobustnessConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or P1WeightingRobustnessConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self, *, test_pass_count: int | None = None) -> dict[str, Any]:
        rows, t_indices = self._collect_rows()
        split = max(1, int(len(rows) * self._cfg.prefix_fraction))
        bundle = NormBundle.fit_from_rows(rows[:split])
        eval_base = rows[split:]
        eval_t = t_indices[split:]
        eval_rows = [self._base_row(r, bundle, int(eval_t[i])) for i, r in enumerate(eval_base)]

        arch_raw = self._archetype_raw(bundle)
        ret_sweep = self._return_weight_sweep(bundle, eval_rows, arch_raw)
        risk_sweep = self._risk_weight_sweep(bundle, eval_rows, arch_raw)
        scalar_test = self._scalar_usefulness(ret_sweep, risk_sweep, eval_rows, bundle)
        failures = self._concrete_failures(eval_rows, bundle, ret_sweep, risk_sweep)
        verdict = self._final_verdict(ret_sweep, risk_sweep, scalar_test, failures)

        return {
            "audit": "P1 Return/Risk Weighting Robustness Validation",
            "fixed_structure": FIXED_STRUCTURE,
            "normalization": "X / sigma_prefix (scale-only, prefix-fit)",
            "weight_grid": [{"w_first": w[0], "w_second": w[1]} for w in WEIGHT_GRID],
            "note_equal_weight_not_goal": (
                "Unequal weights (e.g. 7:3) may be semantically superior; "
                "50:50 contribution share is NOT a success criterion"
            ),
            "1_executive_summary": {
                "eval_n": len(eval_rows),
                "headline": verdict["summary"],
                "return_conclusion": verdict["return"],
                "risk_conclusion": verdict["risk"],
                "no_canonical_adoption": True,
            },
            "2_return_weighting_sweep": ret_sweep,
            "3_risk_weighting_sweep": risk_sweep,
            "4_scalar_usefulness": scalar_test,
            "5_concrete_failure_cases": failures,
            "6_final_verdict": verdict,
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
        }

    def _collect_rows(self) -> tuple[list[dict[str, float]], list[int]]:
        h = self._cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=self._cfg.min_past_bars)
        )
        return [self._obs(self._builder.build(t), h) for t in t_indices], t_indices

    def _scale(self, bundle: NormBundle, raw: dict[str, float], key: str) -> float:
        return bundle.norm(raw, key, SCALE)

    def _base_row(self, raw: dict[str, float], bundle: NormBundle, t_index: int) -> dict[str, Any]:
        return {
            "t_index": t_index,
            "timestamp": str(self._data.bars[t_index].start),
            "raw": raw,
            "U_scaled": self._scale(bundle, raw, "U"),
            "MFE_scaled": self._scale(bundle, raw, "MFE"),
            "MAE_scaled": self._scale(bundle, raw, "MAE"),
            "giveback_scaled": self._scale(bundle, raw, "giveback"),
            "chop_scaled": self._scale(bundle, raw, "chop"),
            "path_ascii": self._path_ascii(t_index),
            "path_pattern": self._classify_pattern(raw),
        }

    def _path_ascii(self, t_index: int) -> str:
        h = self._cfg.reward_horizon
        ctx = self._builder.build(t_index)
        chars = []
        for k in range(1, h + 1):
            r = ctx.return_from_t(k)
            chars.append("^" if r > 0.0005 else ("v" if r < -0.0005 else "-"))
        return "t>" + "".join(chars)

    def _classify_pattern(self, raw: dict[str, float]) -> str:
        u, mfe, gb, term = raw["U"], raw["MFE"], raw["giveback"], raw["terminal"]
        if mfe > u * 1.2 and gb > 0.5:
            return "spike_high_mfe_low_u"
        if u > mfe * 0.85 and gb < 0.25 and term > 0:
            return "smooth_high_u"
        if mfe > 0 and term < -0.001:
            return "mfe_then_crash"
        if abs(raw["MAE"]) < 0.0005 and gb > 0.6:
            return "low_mae_high_giveback"
        if abs(raw["MAE"]) > 0.001 and gb < 0.2:
            return "high_mae_low_giveback"
        if abs(raw["MAE"]) > 0.001 and gb > 0.5:
            return "high_mae_high_giveback"
        return "mixed"

    def _archetype_raw(self, bundle: NormBundle) -> dict[str, dict[str, Any]]:
        h = self._cfg.reward_horizon
        runner = UMaeResidualAuditRunner(
            self._data, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        out: dict[str, dict[str, Any]] = {}
        for pid in ARCH_IDS:
            arch = next(a for a in SYNTHETIC_ARCHETYPES if a["id"] == pid)
            path = runner._path_from_cumulative(
                pid, arch["levels"], h, adverse_wick=pid in ("C", "G", "REC")
            )
            raw = self._obs(path.to_context(), h)
            out[pid] = {
                "description": arch["description"],
                "raw": raw,
                "U_scaled": self._scale(bundle, raw, "U"),
                "MFE_scaled": self._scale(bundle, raw, "MFE"),
                "MAE_scaled": self._scale(bundle, raw, "MAE"),
                "giveback_scaled": self._scale(bundle, raw, "giveback"),
            }
        return out

    def _weighted_return(
        self, row: dict[str, Any], w_u: float, w_m: float
    ) -> tuple[float, dict[str, float]]:
        u, m = row["U_scaled"], row["MFE_scaled"]
        val = w_u * u + w_m * m
        sh = _dominance_shares({"U": w_u * abs(u), "MFE": w_m * abs(m)})
        return val, sh

    def _weighted_risk(
        self, row: dict[str, Any], w_mae: float, w_gb: float
    ) -> tuple[float, dict[str, float]]:
        mae, gb = row["MAE_scaled"], row["giveback_scaled"]
        val = w_mae * mae + w_gb * gb
        sh = _dominance_shares({"MAE": w_mae * abs(mae), "giveback": w_gb * abs(gb)})
        return val, sh

    def _dominance_report(
        self, shares_a: list[dict[str, float]], shares_b: list[dict[str, float]], keys: tuple[str, str]
    ) -> dict[str, Any]:
        cfg = self._cfg
        ka, kb = keys
        va = [s[ka] for s in shares_a]
        vb = [s[kb] for s in shares_b]
        out: dict[str, Any] = {}
        for name, vals in ((ka, va), (kb, vb)):
            out[name] = {
                "mean": float(mean(vals)),
                "median": float(np.median(vals)),
                "p90": _percentile(vals, 0.9),
                "p95": _percentile(vals, 0.95),
            }
            for thr in cfg.dominance_levels:
                out[name][f"pct_gt_{int(thr * 100)}"] = sum(1 for v in vals if v > thr) / len(vals)
        out["effective_single_facet"] = {
            ka: out[ka][f"pct_gt_{90}"],
            kb: out[kb][f"pct_gt_{90}"],
        }
        return out

    def _outcome_relationship(
        self, composite: list[float], terminal: list[float], n_buckets: int
    ) -> dict[str, Any]:
        pear = _pearson(composite, terminal)
        spear = _spearman(composite, terminal)
        order = np.argsort(composite)
        n = len(composite)
        q_size = max(1, n // n_buckets)
        buckets = []
        for b in range(n_buckets):
            start = b * q_size
            end = n if b == n_buckets - 1 else (b + 1) * q_size
            idx = order[start:end]
            if not len(idx):
                continue
            vals = [terminal[i] for i in idx]
            buckets.append(
                {
                    "bucket": b,
                    "composite_range": [composite[order[start]], composite[order[end - 1]]],
                    "mean_terminal": float(mean(vals)),
                    "n": len(idx),
                }
            )
        top_n = max(1, n // 10)
        bot_n = max(1, n // 10)
        top_idx = order[-top_n:]
        bot_idx = order[:bot_n]
        return {
            "pearson": pear,
            "spearman": spear,
            "quantile_buckets": buckets,
            "top10pct_mean_terminal": float(mean(terminal[i] for i in top_idx)),
            "bottom10pct_mean_terminal": float(mean(terminal[i] for i in bot_idx)),
            "top_bottom_separation": float(
                mean(terminal[i] for i in top_idx) - mean(terminal[i] for i in bot_idx)
            ),
            "monotonic_buckets": all(
                buckets[i]["mean_terminal"] <= buckets[i + 1]["mean_terminal"]
                for i in range(len(buckets) - 1)
            ),
        }

    def _false_equivalence_count(
        self,
        eval_rows: list[dict[str, Any]],
        composite: list[float],
        facet_a: str,
        facet_b: str,
    ) -> dict[str, Any]:
        cfg = self._cfg
        pairs: list[dict[str, Any]] = []
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + cfg.pair_window, len(eval_rows))):
                if abs(composite[i] - composite[j]) > cfg.false_equiv_tol:
                    continue
                fa = eval_rows[i][facet_a]
                fb = eval_rows[i][facet_b]
                ga = eval_rows[j][facet_a]
                gb = eval_rows[j][facet_b]
                dom_i = facet_a if fa >= fb else facet_b
                dom_j = facet_a if ga >= gb else facet_b
                if dom_i == dom_j:
                    continue
                pairs.append({"i": eval_rows[i]["t_index"], "j": eval_rows[j]["t_index"]})
                if len(pairs) >= cfg.max_exemplars:
                    break
            if len(pairs) >= cfg.max_exemplars:
                break
        return {"count_sampled": len(pairs), "exemplars": pairs[: cfg.max_exemplars]}

    def _marginal_mfe(self, eval_rows: list[dict[str, Any]], w_u: float, w_m: float) -> dict[str, Any]:
        cfg = self._cfg
        u_vals = [e["U_scaled"] for e in eval_rows]
        mfe_vals = [e["MFE_scaled"] for e in eval_rows]
        term = [e["raw"]["terminal"] for e in eval_rows]
        u_only = u_vals
        composite = [w_u * u + w_m * m for u, m in zip(u_vals, mfe_vals)]

        matched: list[dict[str, Any]] = []
        for i in range(len(eval_rows)):
            for j in range(i + 1, min(i + 40, len(eval_rows))):
                if abs(u_vals[i] - u_vals[j]) > cfg.u_match_tol:
                    continue
                if abs(mfe_vals[i] - mfe_vals[j]) < 0.05:
                    continue
                matched.append(
                    {
                        "pair": [eval_rows[i]["t_index"], eval_rows[j]["t_index"]],
                        "U_scaled": [u_vals[i], u_vals[j]],
                        "MFE_scaled": [mfe_vals[i], mfe_vals[j]],
                        "terminal": [term[i], term[j]],
                        "composite": [composite[i], composite[j]],
                        "u_only": [u_only[i], u_only[j]],
                    }
                )
                if len(matched) >= 5:
                    break
            if len(matched) >= 5:
                break

        mfe_sep = 0
        for p in matched:
            if abs(p["MFE_scaled"][0] - p["MFE_scaled"][1]) > 0.1:
                if abs(p["composite"][0] - p["composite"][1]) > 0.05:
                    mfe_sep += 1
        u_only_sep = sum(
            1 for p in matched if abs(p["u_only"][0] - p["u_only"][1]) > 0.05
        )

        u_buckets: dict[int, list[int]] = {}
        for idx, u in enumerate(u_vals):
            b = int(min(9, max(0, u / (max(u_vals) + 1e-12) * 10)))
            u_buckets.setdefault(b, []).append(idx)
        within_u_mfe_term = []
        for _, idxs in u_buckets.items():
            if len(idxs) < 20:
                continue
            sub_mfe = [mfe_vals[i] for i in idxs]
            sub_term = [term[i] for i in idxs]
            c = _pearson(sub_mfe, sub_term)
            if not np.isnan(c):
                within_u_mfe_term.append(c)

        return {
            "matched_pairs": matched,
            "mfe_discriminates_composite": mfe_sep,
            "u_only_discriminates": u_only_sep,
            "within_u_bucket_mfe_vs_terminal_pearson_mean": float(mean(within_u_mfe_term))
            if within_u_mfe_term
            else float("nan"),
            "marginal_mfe_adds_info": mfe_sep > u_only_sep or (
                within_u_mfe_term and mean(within_u_mfe_term) > 0.05
            ),
        }

    def _archetype_at_weight(
        self,
        arch: dict[str, dict[str, Any]],
        combo_fn: Callable[[dict[str, Any], float, float], tuple[float, dict[str, float]]],
        w0: float,
        w1: float,
    ) -> dict[str, Any]:
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        for pid, data in arch.items():
            row = {
                "U_scaled": data["U_scaled"],
                "MFE_scaled": data["MFE_scaled"],
                "MAE_scaled": data["MAE_scaled"],
                "giveback_scaled": data["giveback_scaled"],
            }
            val, sh = combo_fn(row, w0, w1)
            scores[pid] = val
            details[pid] = {
                "composite": val,
                "shares": sh,
                "U_scaled": data["U_scaled"],
                "MFE_scaled": data["MFE_scaled"],
                "MAE_scaled": data.get("MAE_scaled"),
                "giveback_scaled": data.get("giveback_scaled"),
                "raw": data["raw"],
            }
        rank = _rank_list(ARCH_IDS, scores)
        return {"scores": scores, "ranking": rank, "details": details}

    def _return_semantic_checks(self, arch_detail: dict[str, Any]) -> dict[str, Any]:
        d = arch_detail["details"]
        scores = arch_detail["scores"]
        rank = arch_detail["ranking"]
        b_above_a = scores["B"] > scores["A"]
        c_overrated = scores["C"] > scores["A"] and d["C"]["raw"]["terminal"] < 0
        ab_gap = abs(scores["A"] - scores["B"])
        return {
            "ranking": rank,
            "B_above_A": b_above_a,
            "C_overrated_vs_spike_crash": c_overrated,
            "A_B_scalar_gap": ab_gap,
            "A_B_simultaneously_meaningful": ab_gap > 0.5 * max(abs(scores["A"]), abs(scores["B"]), 1e-9),
            "notes": {
                "A": (
                    f"U share {d['A']['shares']['U']:.1%}, MFE {d['A']['shares']['MFE']:.1%}; "
                    f"composite={scores['A']:.2f}"
                ),
                "B": (
                    f"U share {d['B']['shares']['U']:.1%}, MFE {d['B']['shares']['MFE']:.1%}; "
                    f"composite={scores['B']:.2f}"
                ),
                "C": f"terminal={d['C']['raw']['terminal']:.4f}; composite={scores['C']:.2f}",
                "G": f"composite={scores['G']:.2f}",
            },
        }

    def _risk_semantic_checks(self, arch_detail: dict[str, Any]) -> dict[str, Any]:
        scores = arch_detail["scores"]
        d = arch_detail["details"]
        order_ok = all(
            scores[RISK_ARCH_ORDER[i]] <= scores[RISK_ARCH_ORDER[i + 1]]
            for i in range(len(RISK_ARCH_ORDER) - 1)
        )
        return {
            "ranking": arch_detail["ranking"],
            "intended_B_A_REC_C_order": order_ok,
            "scores": {k: scores[k] for k in RISK_ARCH_ORDER if k in scores},
            "mechanism": {
                pid: {
                    "MAE_share": d[pid]["shares"]["MAE"],
                    "giveback_share": d[pid]["shares"]["giveback"],
                }
                for pid in ("B", "A", "C", "REC")
                if pid in d
            },
        }

    def _robustness_analysis(
        self,
        per_weight: list[dict[str, Any]],
        rank_key: str = "archetype_ranking",
    ) -> dict[str, Any]:
        if len(per_weight) < 2:
            return {"robust_regions": [], "fragile_weights": []}
        ranks = [w[rank_key] for w in per_weight]
        weights = [w["weight_label"] for w in per_weight]
        transitions = []
        for i in range(len(ranks) - 1):
            dist = _kendall_distance(ranks[i], ranks[i + 1])
            transitions.append(
                {
                    "from": weights[i],
                    "to": weights[i + 1],
                    "kendall_distance": dist,
                    "stable": dist == 0,
                }
            )
        stable_runs: list[dict[str, Any]] = []
        run_start = 0
        for i, tr in enumerate(transitions):
            if not tr["stable"]:
                if i > run_start:
                    stable_runs.append(
                        {
                            "weights": weights[run_start : i + 1],
                            "length": i + 1 - run_start,
                            "ranking": ranks[run_start],
                        }
                    )
                run_start = i + 1
        if len(transitions) > run_start:
            stable_runs.append(
                {
                    "weights": weights[run_start:],
                    "length": len(weights) - run_start,
                    "ranking": ranks[run_start],
                }
            )
        stable_runs.sort(key=lambda x: -x["length"])
        fragile = [
            weights[i]
            for i in range(1, len(weights) - 1)
            if transitions[i - 1]["kendall_distance"] > 0.3
            and (i >= len(transitions) or transitions[i]["kendall_distance"] > 0.3)
        ]
        return {
            "adjacent_transitions": transitions,
            "robust_regions": stable_runs[:3],
            "fragile_weights": fragile[:5],
            "longest_stable_run": stable_runs[0] if stable_runs else None,
        }

    def _return_weight_sweep(
        self,
        bundle: NormBundle,
        eval_rows: list[dict[str, Any]],
        arch: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        per_weight: list[dict[str, Any]] = []
        terminal = [e["raw"]["terminal"] for e in eval_rows]

        for w_u, w_m in WEIGHT_GRID:
            composites = []
            shares = []
            for e in eval_rows:
                v, sh = self._weighted_return(e, w_u, w_m)
                composites.append(v)
                shares.append(sh)
            arch_detail = self._archetype_at_weight(arch, self._weighted_return, w_u, w_m)
            sem = self._return_semantic_checks(arch_detail)
            outcome = self._outcome_relationship(composites, terminal, self._cfg.quantile_buckets)
            dom = self._dominance_report(shares, shares, ("U", "MFE"))
            marginal = self._marginal_mfe(eval_rows, w_u, w_m) if w_m > 0 else {"marginal_mfe_adds_info": False}

            per_weight.append(
                {
                    "weight_label": f"{w_u}:{w_m}",
                    "w_U": w_u,
                    "w_MFE": w_m,
                    "archetype_ranking": sem["ranking"],
                    "archetype_semantics": sem,
                    "outcome_relationship": outcome,
                    "dominance": dom,
                    "marginal_mfe": marginal,
                }
            )

        robust = self._robustness_analysis(
            [{"weight_label": w["weight_label"], "archetype_ranking": w["archetype_ranking"]} for w in per_weight]
        )
        return {
            "formula": "R = w_U * U/sigma + w_MFE * MFE/sigma",
            "per_weight": per_weight,
            "robustness": robust,
            "u_only_outcome": per_weight[0]["outcome_relationship"],
            "mfe_only_outcome": per_weight[-1]["outcome_relationship"],
        }

    def _risk_weight_sweep(
        self,
        bundle: NormBundle,
        eval_rows: list[dict[str, Any]],
        arch: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        per_weight: list[dict[str, Any]] = []
        terminal = [e["raw"]["terminal"] for e in eval_rows]

        for w_mae, w_gb in WEIGHT_GRID:
            composites = []
            shares = []
            for e in eval_rows:
                v, sh = self._weighted_risk(e, w_mae, w_gb)
                composites.append(v)
                shares.append(sh)
            arch_detail = self._archetype_at_weight(arch, self._weighted_risk, w_mae, w_gb)
            sem = self._risk_semantic_checks(arch_detail)
            false_eq = self._false_equivalence_count(
                eval_rows, composites, "MAE_scaled", "giveback_scaled"
            )
            dom = self._dominance_report(shares, shares, ("MAE", "giveback"))
            outcome = self._outcome_relationship(composites, terminal, self._cfg.quantile_buckets)

            per_weight.append(
                {
                    "weight_label": f"{w_mae}:{w_gb}",
                    "w_MAE": w_mae,
                    "w_GB": w_gb,
                    "archetype_ranking": sem["ranking"],
                    "archetype_semantics": sem,
                    "false_equivalence": false_eq,
                    "dominance": dom,
                    "outcome_relationship": outcome,
                }
            )

        robust = self._robustness_analysis(
            [{"weight_label": w["weight_label"], "archetype_ranking": w["archetype_ranking"]} for w in per_weight]
        )
        return {
            "formula": "Risk = w_MAE * MAE/sigma + w_GB * Giveback/sigma",
            "per_weight": per_weight,
            "robustness": robust,
        }

    def _scalar_usefulness(
        self,
        ret_sweep: dict[str, Any],
        risk_sweep: dict[str, Any],
        eval_rows: list[dict[str, Any]],
        bundle: NormBundle,
    ) -> dict[str, Any]:
        def _compress_loss(composites: list[float], facets: list[tuple[float, float]]) -> float:
            if len(composites) < 10:
                return float("nan")
            x = np.asarray(composites, dtype=float)
            f1 = np.asarray([f[0] for f in facets], dtype=float)
            f2 = np.asarray([f[1] for f in facets], dtype=float)
            r2_1 = _pearson(x, f1) ** 2 if not np.isnan(_pearson(x, f1)) else 0
            r2_2 = _pearson(x, f2) ** 2 if not np.isnan(_pearson(x, f2)) else 0
            return 1.0 - max(r2_1, r2_2)

        ret_loss = []
        for w in ret_sweep["per_weight"]:
            w_u, w_m = w["w_U"], w["w_MFE"]
            comp = []
            facets = []
            for e in eval_rows:
                v, _ = self._weighted_return(e, w_u, w_m)
                comp.append(v)
                facets.append((e["U_scaled"], e["MFE_scaled"]))
            ret_loss.append(
                {
                    "weight": w["weight_label"],
                    "information_loss": _compress_loss(comp, facets),
                    "monotonic_buckets": w["outcome_relationship"]["monotonic_buckets"],
                }
            )

        risk_loss = []
        for w in risk_sweep["per_weight"]:
            w_mae, w_gb = w["w_MAE"], w["w_GB"]
            comp = []
            facets = []
            for e in eval_rows:
                v, _ = self._weighted_risk(e, w_mae, w_gb)
                comp.append(v)
                facets.append((e["MAE_scaled"], e["giveback_scaled"]))
            risk_loss.append(
                {
                    "weight": w["weight_label"],
                    "information_loss": _compress_loss(comp, facets),
                    "false_equiv_sampled": w["false_equivalence"]["count_sampled"],
                }
            )

        return {
            "return_scalar": {
                "per_weight_loss": ret_loss,
                "min_loss_weight": min(ret_loss, key=lambda x: x["information_loss"])["weight"],
                "all_weights_high_loss": all(x["information_loss"] > 0.3 for x in ret_loss if not np.isnan(x["information_loss"])),
            },
            "risk_scalar": {
                "per_weight_loss": risk_loss,
                "false_equiv_present_all_weights": all(x["false_equiv_sampled"] > 0 for x in risk_loss),
            },
            "interpretation": (
                "High information_loss or persistent false-equivalence implies "
                "facet vector should be kept separate from scalar"
            ),
        }

    def _concrete_failures(
        self,
        eval_rows: list[dict[str, Any]],
        bundle: NormBundle,
        ret_sweep: dict[str, Any],
        risk_sweep: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self._cfg
        cases: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add(e: dict[str, Any], *, w_label: str, composite: float, shares: dict[str, float], reason: str, kind: str) -> None:
            if e["t_index"] in seen or len(cases) >= cfg.min_failure_cases + 5:
                return
            r = e["raw"]
            cases.append(
                {
                    "kind": kind,
                    "timestamp": e["timestamp"],
                    "t_index": e["t_index"],
                    "path_pattern": e["path_pattern"],
                    "path_ascii": e["path_ascii"],
                    "U": r["U"],
                    "MFE": r["MFE"],
                    "MAE": r["MAE"],
                    "giveback": r["giveback"],
                    "chop": r["chop"],
                    "U_scaled": e["U_scaled"],
                    "MFE_scaled": e["MFE_scaled"],
                    "MAE_scaled": e["MAE_scaled"],
                    "giveback_scaled": e["giveback_scaled"],
                    "weight": w_label,
                    "composite": composite,
                    "facet_contribution_pct": {k: round(100 * v, 1) for k, v in shares.items()},
                    "terminal": r["terminal"],
                    "why_misleading": reason,
                }
            )
            seen.add(e["t_index"])

        w_mid = ret_sweep["per_weight"][5]
        w_u, w_m = w_mid["w_U"], w_mid["w_MFE"]
        for e in eval_rows:
            if e["path_pattern"] == "spike_high_mfe_low_u":
                v, sh = self._weighted_return(e, w_u, w_m)
                add(e, w_label=w_mid["weight_label"], composite=v, shares=sh, reason="High MFE spike but low U; scalar blends opportunity vs grind", kind="return_spike")
            if e["path_pattern"] == "smooth_high_u":
                v, sh = self._weighted_return(e, w_u, w_m)
                add(e, w_label=w_mid["weight_label"], composite=v, shares=sh, reason="Smooth grind: U~MFE; weight-sensitive ranking", kind="return_smooth")
            if e["path_pattern"] == "mfe_then_crash":
                v, sh = self._weighted_return(e, 0.2, 0.8)
                add(e, w_label="0.2:0.8", composite=v, shares=sh, reason="MFE-heavy weight overstates pre-crash opportunity", kind="return_crash")

        w_risk = risk_sweep["per_weight"][5]
        for e in eval_rows:
            if e["path_pattern"] == "low_mae_high_giveback":
                v, sh = self._weighted_risk(e, w_risk["w_MAE"], w_risk["w_GB"])
                add(e, w_label=w_risk["weight_label"], composite=v, shares=sh, reason="Low MAE masks giveback erosion in scalar", kind="risk_giveback")
            if e["path_pattern"] == "high_mae_low_giveback":
                v, sh = self._weighted_risk(e, w_risk["w_MAE"], w_risk["w_GB"])
                add(e, w_label=w_risk["weight_label"], composite=v, shares=sh, reason="High MAE with low giveback; mechanism MAE-driven", kind="risk_mae")

        for fe in risk_sweep["per_weight"][5]["false_equivalence"]["exemplars"][:2]:
            i = next(x for x in eval_rows if x["t_index"] == fe["i"])
            j = next(x for x in eval_rows if x["t_index"] == fe["j"])
            v_i, sh_i = self._weighted_risk(i, 0.5, 0.5)
            v_j, sh_j = self._weighted_risk(j, 0.5, 0.5)
            add(i, w_label="0.5:0.5", composite=v_i, shares=sh_i, reason="False equiv pair: similar Risk, MAE-driven vs GB-driven", kind="risk_false_equiv")
            add(j, w_label="0.5:0.5", composite=v_j, shares=sh_j, reason="False equiv pair: similar Risk, different mechanism", kind="risk_false_equiv")

        per_w = ret_sweep["per_weight"]
        for e in eval_rows[:500]:
            if len(cases) >= cfg.min_failure_cases:
                break
            ranks = []
            for w in (per_w[3], per_w[5], per_w[7]):
                v, sh = self._weighted_return(e, w["w_U"], w["w_MFE"])
                ranks.append(v)
            if max(ranks) - min(ranks) > 2.0 * np.std([x["outcome_relationship"]["pearson"] for x in per_w]):
                v, sh = self._weighted_return(e, 0.7, 0.3)
                add(e, w_label="0.7:0.3", composite=v, shares=sh, reason="Return composite shifts sharply across adjacent weights", kind="weight_fragile")

        return {"count": len(cases), "cases": cases[: max(cfg.min_failure_cases, len(cases))]}

    def _classify_return_verdict(self, ret_sweep: dict[str, Any]) -> dict[str, Any]:
        per = ret_sweep["per_weight"]
        robust = ret_sweep["robustness"]["longest_stable_run"]
        ab_sep = [w["archetype_semantics"]["A_B_simultaneously_meaningful"] for w in per]
        any_ab_sep = any(ab_sep)
        c_over = sum(1 for w in per if w["archetype_semantics"]["C_overrated_vs_spike_crash"])
        marginal_ok = sum(1 for w in per if w.get("marginal_mfe", {}).get("marginal_mfe_adds_info"))

        if not any_ab_sep:
            category = "D_no_valid_scalar" if not robust else "B_robust_region"
            scalar_possible = "partial_ranking_only"
            recommend_facets = True
        elif robust and robust.get("length", 0) >= 3:
            category = "B_robust_region"
            scalar_possible = True
            recommend_facets = False
        elif max(ab_sep) and sum(ab_sep) <= 2:
            category = "C_fragile_region"
            scalar_possible = True
            recommend_facets = True
        else:
            category = "B_robust_region" if robust else "C_fragile_region"
            scalar_possible = "partial_ranking_only" if not any_ab_sep else True
            recommend_facets = not any_ab_sep

        return {
            "category": category,
            "robust_region": robust,
            "scalar_possible": scalar_possible,
            "recommend_separate_facets": recommend_facets,
            "A_B_never_jointly_meaningful": not any_ab_sep,
            "marginal_mfe_weights": marginal_ok,
            "recommended_interval": robust["weights"] if robust else None,
            "note": (
                "No weight chosen by max correlation. "
                "U-only may correlate highest with terminal but MFE still adds path-level discrimination."
            ),
        }

    def _classify_risk_verdict(self, risk_sweep: dict[str, Any]) -> dict[str, Any]:
        per = risk_sweep["per_weight"]
        robust = risk_sweep["robustness"]["longest_stable_run"]
        order_ok = sum(1 for w in per if w["archetype_semantics"]["intended_B_A_REC_C_order"])
        false_all = all(w["false_equivalence"]["count_sampled"] > 0 for w in per)

        if false_all and order_ok < len(per) // 3:
            category = "D_no_valid_scalar"
        elif robust and robust.get("length", 0) >= 2:
            category = "B_robust_region"
        elif order_ok >= len(per) // 2:
            category = "C_fragile_region"
        else:
            category = "D_no_valid_scalar"

        return {
            "category": category,
            "robust_region": robust,
            "scalar_possible": category != "D_no_valid_scalar",
            "recommend_separate_facets": category == "D_no_valid_scalar" or false_all,
            "recommended_interval": robust["weights"] if robust else None,
            "false_equivalence_all_weights": false_all,
        }

    def _final_verdict(
        self,
        ret_sweep: dict[str, Any],
        risk_sweep: dict[str, Any],
        scalar_test: dict[str, Any],
        failures: dict[str, Any],
    ) -> dict[str, Any]:
        ret_v = self._classify_return_verdict(ret_sweep)
        risk_v = self._classify_risk_verdict(risk_sweep)
        return {
            "return": ret_v,
            "risk": risk_v,
            "summary": (
                f"Return: {ret_v['category']}; Risk: {risk_v['category']}. "
                "Unequal weights may outperform 50:50; equal contribution is NOT success. "
                "Separate facets recommended if scalar collapses mechanisms."
            ),
            "return_questions": {
                "recommended_weight_exists": ret_v["category"] == "A_strong_candidate",
                "robust_interval_exists": ret_v["category"] == "B_robust_region",
                "scalar_possible": ret_v["scalar_possible"] not in (False, "partial_ranking_only")
                if isinstance(ret_v["scalar_possible"], bool)
                else ret_v["scalar_possible"] == True,
                "scalar_partial_ranking_only": ret_v["scalar_possible"] == "partial_ranking_only",
                "keep_separate_facets": ret_v["recommend_separate_facets"],
            },
            "risk_questions": {
                "recommended_weight_exists": risk_v["category"] == "A_strong_candidate",
                "robust_interval_exists": risk_v["category"] == "B_robust_region",
                "scalar_possible": risk_v["scalar_possible"],
                "keep_separate_facets": risk_v["recommend_separate_facets"],
            },
            "evidence_not_hypothesis": {
                "return_dual_semantic_limit": "CONFIRMED from prior + sweep: A peak vs B grind not jointly rankable",
                "risk_false_equivalence": "CONFIRMED at all sampled weights" if risk_v["false_equivalence_all_weights"] else "HYPOTHESIS",
            },
        }


def format_weighting_robustness_summary(report: dict[str, Any]) -> str:
    s = report.get("1_executive_summary", {})
    v = report.get("6_final_verdict", {})
    lines = [
        "P1 Return/Risk Weighting Robustness Validation",
        "=" * 60,
        f"eval_n: {s.get('eval_n')}",
        f"headline: {s.get('headline')}",
        f"return: {v.get('return', {}).get('category')}",
        f"risk: {v.get('risk', {}).get('category')}",
    ]
    return "\n".join(lines)


def save_p1_weighting_robustness_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
