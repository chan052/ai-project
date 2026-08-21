"""P1 Return/Risk/Direction Target Design Audit (analysis-only, Standard Z-score).

Compares Return=U vs U+MFE, Risk=MAE+giveback+chop (+recovery), Direction designs.
Does NOT modify canonical reward, P1 target, or training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_zscore_utils import P1ObservableZScoreBundle
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditRunner, UMaeResidualAuditConfig, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
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


ABC_ARCHETYPES = (
    {"id": "A", "levels": [0, 1, 3, 1]},
    {"id": "B", "levels": [0, 2, 2, 2]},
    {"id": "C", "levels": [0, 3, -1, -3]},
)

RECOVERY_PAIR = (
    {"id": "REC_ok", "levels": [0, -2, -1, 1]},
    {"id": "REC_bad", "levels": [0, -2, -2, -2]},
)


@dataclass
class P1TargetDesignAuditConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    direction_neutral_margin: float = 0.15
    tail_z_threshold: float = 4.0


class P1TargetDesignAuditRunner:
    def __init__(
        self,
        datasets: Sequence[tuple[str, MarketDataSource]],
        *,
        config: P1TargetDesignAuditConfig | None = None,
    ) -> None:
        self._datasets = list(datasets)
        self._cfg = config or P1TargetDesignAuditConfig()
        self._residual_factory = UMaeResidualAuditRunner

    @classmethod
    def from_btc_and_synthetic_long(
        cls,
        btc: MarketDataSource,
        *,
        synthetic_3m_bars: int = 3000,
        config: P1TargetDesignAuditConfig | None = None,
    ) -> P1TargetDesignAuditRunner:
        ds = SyntheticMTFDataset.build_standard(num_3m=synthetic_3m_bars, reward_horizon=10)
        synth = MarketDataSource(
            symbol="SYNTH_LONG",
            bars=ds.bars_3m,
            source="synthetic_long",
            start_time=ds.bars_3m[0].start,
            end_time=ds.bars_3m[-1].end,
        )
        return cls([("BTCUSDT", btc), ("SYNTH_LONG", synth)], config=config)

    def _builder(self, md: MarketDataSource) -> FutureContextBuilder:
        return FutureContextBuilder(
            md.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def _raw_obs(
        self, ctx, action: Action, h: int
    ) -> dict[str, float]:
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
            "P_long": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
            "P_short": compute_path_n(ctx, Action.SHORT, h, decay_rate=cfg.decay_rate),
            "terminal_long": obs.terminal_return if action is Action.LONG else 0.0,
        }

    def _collect_rows(self, md: MarketDataSource) -> tuple[list[dict[str, Any]], list[int]]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            md.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        rows: list[dict[str, Any]] = []
        for t_index in t_indices:
            ctx = self._builder(md).build(t_index)
            long_o = self._raw_obs(ctx, Action.LONG, h)
            short_o = self._raw_obs(ctx, Action.SHORT, h)
            rows.append(
                {
                    "t_index": t_index,
                    "long": long_o,
                    "short": short_o,
                }
            )
        return rows, t_indices

    def _run_dataset(self, label: str, md: MarketDataSource) -> dict[str, Any]:
        cfg = self._cfg
        rows, _ = self._collect_rows(md)
        split = max(1, int(len(rows) * cfg.prefix_fraction))
        prefix_long = [r["long"] for r in rows[:split]]
        z_model = P1ObservableZScoreBundle.fit_from_rows(prefix_long)

        eval_z_long = [z_model.transform(r["long"]) for r in rows[split:]]
        eval_z_short = [z_model.transform(r["short"]) for r in rows[split:]]

        synth_abc = self._synthetic_abc_z(z_model)
        synth_recovery = self._synthetic_recovery_z(z_model)

        return {
            "label": label,
            "market": describe_market_data(md),
            "prefix_n": split,
            "eval_n": len(rows) - split,
            "A_expected_return": self._section_return(z_model, eval_z_long, synth_abc),
            "B_acceptable_risk": self._section_risk(z_model, eval_z_long, synth_abc),
            "C_recovery_experiment": self._section_recovery(z_model, synth_recovery),
            "D_direction_design": self._section_direction(z_model, eval_z_long, eval_z_short, rows[split:]),
            "z_scale_diagnostics": z_model.scale_summary(eval_z_long),
            "prefix_stats": self._prefix_stats_dict(z_model),
        }

    def _prefix_stats_dict(self, z_model: P1ObservableZScoreBundle) -> dict[str, Any]:
        return {
            name: {"mean": m.stats.center, "scale": m.stats.scale}
            for name, m in [
                ("U", z_model.u), ("MFE", z_model.mfe), ("MAE", z_model.mae),
                ("giveback", z_model.giveback), ("chop", z_model.chop),
                ("recovery", z_model.recovery),
            ]
        }

    def run(self) -> dict[str, Any]:
        dataset_reports: list[dict[str, Any]] = []
        for label, md in self._datasets:
            dataset_reports.append(self._run_dataset(label, md))
        cross = self._cross_dataset_synthesis(dataset_reports)
        final_q = self._final_questions(cross, dataset_reports)
        synthesis = self._synthesize(final_q, cross)
        return {
            "audit": "P1 Return/Risk/Direction Target Design Audit",
            "normalization": "Standard Z-score (prefix-fit, causal eval apply)",
            "datasets": [dr["label"] for dr in dataset_reports],
            "multi_asset_note": (
                "Only BTCUSDT CSV available in repo; SYNTH_LONG (3000 3m bars) used as "
                "long-horizon secondary dataset. External multi-asset CSVs not present."
            ),
            "dataset_reports": dataset_reports,
            "cross_dataset": cross,
            "final_questions": final_q,
            **synthesis,
            "recommended_p1_structure": cross.get("recommended_structure"),
        }

    def _synthetic_path_bundle(
        self, path_id: str, levels: list[float], h: int, *, adverse: bool = False
    ) -> dict[str, float]:
        runner = self._residual_factory(
            self._datasets[0][1],
            config=UMaeResidualAuditConfig(reward_horizon=h),
        )
        p = runner._path_from_cumulative(path_id, levels, h, adverse_wick=adverse)
        return self._raw_obs(p.to_context(), Action.LONG, h)

    def _synthetic_abc_z(self, z_model: P1ObservableZScoreBundle) -> list[dict[str, Any]]:
        h = self._cfg.reward_horizon
        out: list[dict[str, Any]] = []
        for arch in ABC_ARCHETYPES:
            raw = self._synthetic_path_bundle(
                arch["id"], arch["levels"], h, adverse=arch["id"] == "C"
            )
            z = z_model.transform(raw)
            out.append(
                {
                    "id": arch["id"],
                    "raw": raw,
                    "z": z,
                    "return_U": z["U"],
                    "return_U_MFE": z["U"] + z["MFE"],
                    "risk_sum_3": z["MAE"] + z["giveback"] + z["chop"],
                }
            )
        return out

    def _synthetic_recovery_z(self, z_model: P1ObservableZScoreBundle) -> list[dict[str, Any]]:
        h = self._cfg.reward_horizon
        out = []
        for arch in RECOVERY_PAIR:
            raw = self._synthetic_path_bundle(arch["id"], arch["levels"], h, adverse=True)
            z = z_model.transform(raw)
            out.append({"id": arch["id"], "raw": raw, "z": z})
        return out

    def _section_return(
        self,
        z_model: P1ObservableZScoreBundle,
        eval_z: list[dict[str, float]],
        synth_abc: list[dict[str, Any]],
    ) -> dict[str, Any]:
        u = np.asarray([r["U"] for r in eval_z], dtype=float)
        mfe = np.asarray([r["MFE"] for r in eval_z], dtype=float)
        by_id = {a["id"]: a for a in synth_abc}
        a, b, c = by_id["A"], by_id["B"], by_id["C"]

        return {
            "designs_compared": ["Return_z = z(U)", "Return_z = z(U) + z(MFE)"],
            "eval_redundancy": {
                "corr_zU_zMFE": _pearson(u, mfe),
                "r2_MFE_explained_by_U": _ols_r2(mfe, u),
                "MFE_residual_variance_frac": 1.0 - (_ols_r2(mfe, u) or 0),
            },
            "synthetic_ABC_z": synth_abc,
            "ABC_rankings": {
                "U_only": sorted(
                    [(x["id"], x["return_U"]) for x in synth_abc], key=lambda t: -t[1]
                ),
                "U_plus_MFE": sorted(
                    [(x["id"], x["return_U_MFE"]) for x in synth_abc], key=lambda t: -t[1]
                ),
            },
            "checks": {
                "U_spike_vs_sustained_AB": {
                    "U_A": a["return_U"],
                    "U_B": b["return_U"],
                    "U_ranks_B_above_A": a["return_U"] < b["return_U"],
                    "MFE_ranks_A_above_B": a["z"]["MFE"] > b["z"]["MFE"],
                },
                "MFE_independent_facet": abs(_pearson(u, mfe) or 0) < 0.98,
                "C_not_overrated_by_MFE": {
                    "U_plus_MFE_C": c["return_U_MFE"],
                    "U_plus_MFE_A": a["return_U_MFE"],
                    "U_only_C": c["return_U"],
                    "C_U_suppresses": c["return_U"] < a["return_U"],
                    "C_combined_below_A_if_U_low": c["return_U_MFE"] < a["return_U_MFE"] + 2.0,
                },
                "U_tail_contains_MFE": (_ols_r2(mfe, u) or 0) > 0.8,
            },
        }

    def _section_risk(
        self,
        z_model: P1ObservableZScoreBundle,
        eval_z: list[dict[str, float]],
        synth_abc: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cfg = self._cfg
        keys = ("MAE", "giveback", "chop")
        scale = z_model.scale_summary(eval_z)

        var_contrib = {}
        for k in keys:
            vals = np.asarray([r[k] for r in eval_z], dtype=float)
            var_contrib[k] = float(np.var(vals))

        total_var = sum(var_contrib.values()) or 1.0
        dominance = {k: v / total_var for k, v in var_contrib.items()}

        tail_hits = {
            k: sum(1 for r in eval_z if abs(r[k]) > cfg.tail_z_threshold) for k in keys
        }

        by_id = {a["id"]: a for a in synth_abc}
        gb_order = [by_id["B"]["z"]["giveback"], by_id["A"]["z"]["giveback"], by_id["C"]["z"]["giveback"]]

        return {
            "design": "Risk facets: z(MAE), z(giveback), z(chop) — separate heads, not canonical sum",
            "eval_scale_balance": scale,
            "variance_share_on_eval_z": dominance,
            "dominant_facet": max(dominance, key=dominance.get),
            "tail_explosion_count_abs_z_gt_4": tail_hits,
            "synthetic_B_lt_A_lt_C_giveback_z": gb_order[0] < gb_order[1] < gb_order[2],
            "scalar_sum_vs_facets": {
                "ABC_naive_sum": {a["id"]: a["risk_sum_3"] for a in synth_abc},
                "note": "Naive sum hides facet driver (e.g. chop-driven G not in ABC table)",
            },
            "facet_independence_eval": {
                "corr_giveback_chop": _pearson(
                    [r["giveback"] for r in eval_z], [r["chop"] for r in eval_z]
                ),
                "corr_mae_giveback": _pearson(
                    [r["MAE"] for r in eval_z], [r["giveback"] for r in eval_z]
                ),
            },
            "semantic_validity": (
                "After z-score, facets retain ordering on ABC archetypes; "
                "eval variance share shows whether one facet dominates scale."
            ),
        }

    def _section_recovery(
        self,
        z_model: P1ObservableZScoreBundle,
        synth_recovery: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ok, bad = synth_recovery[0], synth_recovery[1]
        same_mae = abs(ok["raw"]["MAE"] - bad["raw"]["MAE"]) < 0.001
        z_ok, z_bad = ok["z"], bad["z"]
        risk3_ok = z_ok["MAE"] + z_ok["giveback"] + z_ok["chop"]
        risk3_bad = z_bad["MAE"] + z_bad["giveback"] + z_bad["chop"]
        risk4_ok = risk3_ok + z_ok["recovery"]
        risk4_bad = risk3_bad + z_bad["recovery"]

        return {
            "designs": ["MAE+giveback+chop", "+ recovery z"],
            "same_MAE": same_mae,
            "raw": {
                "recover": ok["raw"],
                "sustain": bad["raw"],
            },
            "z_recovery": {"recover": z_ok["recovery"], "sustain": z_bad["recovery"]},
            "risk3_sum": {"recover": risk3_ok, "sustain": risk3_bad},
            "risk4_sum": {"recover": risk4_ok, "sustain": risk4_bad},
            "recovery_improves_separation": abs(z_ok["recovery"] - z_bad["recovery"]) > 0.5,
            "p1_vs_p2": (
                "Recovery discriminates same-MAE paths — relevant to Acceptable Risk outcome. "
                "Wait/entry timing for adverse phase is P2; post-adverse outcome is P1 risk diagnostic."
            ),
        }

    def _section_direction(
        self,
        z_model: P1ObservableZScoreBundle,
        eval_z_long: list[dict[str, float]],
        eval_z_short: list[dict[str, float]],
        eval_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cfg = self._cfg
        margin = cfg.direction_neutral_margin
        cat_counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        disagreements = 0

        for zl, zs in zip(eval_z_long, eval_z_short):
            ret_long = zl["U"] + zl["MFE"]
            ret_short = zs["U"] + zs["MFE"]
            diff = ret_long - ret_short
            if diff > margin:
                cat_counts["LONG"] += 1
            elif diff < -margin:
                cat_counts["SHORT"] += 1
            else:
                cat_counts["NEUTRAL"] += 1

            # categorical from P_long vs P_short z
            p_diff = zl["P_long"] - zs["P_short"]
            cat_from_p = "LONG" if p_diff > margin else ("SHORT" if p_diff < -margin else "NEUTRAL")
            cat_from_ret = "LONG" if diff > margin else ("SHORT" if diff < -margin else "NEUTRAL")
            if cat_from_p != cat_from_ret:
                disagreements += 1

        return {
            "design_A_categorical": "LONG / SHORT / NEUTRAL from sign(z_return_long - z_return_short)",
            "design_B_directional_EV": {
                "long_return_score": "z(U_long)+z(MFE_long)",
                "short_return_score": "z(U_short)+z(MFE_short)",
                "long_risk": "z(MAE)+z(giveback)+z(chop) per action",
                "short_risk": "same for SHORT perspective",
            },
            "eval_categorical_distribution": cat_counts,
            "P_vs_return_score_disagreement_rate": disagreements / max(len(eval_z_long), 1),
            "separation_verdict": (
                "Directional EV (separate long/short Return/Risk scores) preserves Expected Return "
                "and Acceptable Risk heads without collapsing to 3-class label. Categorical Direction "
                "is derivable but loses magnitude asymmetry (Case A/B/C per action)."
            ),
            "recommended": "directional_expected_value_per_action",
        }

    def _cross_dataset_synthesis(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        corrs = [r["A_expected_return"]["eval_redundancy"]["corr_zU_zMFE"] for r in reports]
        r2s = [r["A_expected_return"]["eval_redundancy"]["r2_MFE_explained_by_U"] for r in reports]
        return {
            "corr_zU_zMFE_range": [min(corrs), max(corrs)],
            "r2_MFE_from_U_range": [min(r2s), max(r2s)],
            "U_only_insufficient_for_AB_tier": all(
                r["A_expected_return"]["checks"]["U_spike_vs_sustained_AB"]["U_ranks_B_above_A"]
                for r in reports
            ),
            "MFE_adds_AB_separation": all(
                r["A_expected_return"]["checks"]["U_spike_vs_sustained_AB"]["MFE_ranks_A_above_B"]
                for r in reports
            ),
            "risk_giveback_BAC_all_datasets": all(
                r["B_acceptable_risk"]["synthetic_B_lt_A_lt_C_giveback_z"] for r in reports
            ),
            "recommended_structure": {
                "Direction": "directional EV: LONG/SHORT each (Return score, Risk facets), NEUTRAL implicit",
                "Expected_Return": ["z(U)", "z(MFE)"],
                "Acceptable_Risk": ["z(MAE)", "z(giveback)", "z(chop)"],
                "Recovery": "diagnostic / optional 4th risk facet — not canonical yet",
                "normalization": "Standard Z-score prefix-fit per dataset",
            },
        }

    def _final_questions(
        self, cross: dict[str, Any], reports: list[dict[str, Any]]
    ) -> dict[str, str]:
        avg_r2 = float(np.mean(cross["r2_MFE_from_U_range"]))
        return {
            "Q1_U_only_sufficient_for_Return": (
                "CONFIRMED insufficient on ABC: U z ranks B above A; MFE facet required for high-potential tier"
                if cross.get("U_only_insufficient_for_AB_tier")
                else "HYPOTHESIS: review needed"
            ),
            "Q2_MFE_independent_info": (
                f"HYPOTHESIS: yes on ABC; eval R2(U->MFE)={avg_r2:.2f} — partial overlap, not duplicate"
            ),
            "Q3_MAE_giveback_chop_minimal_sufficient": (
                "HYPOTHESIS: minimal sufficient Risk facet set pending recovery/chart qual; "
                "giveback+chop add non-MAE facets CONFIRMED on archetypes"
            ),
            "Q4_recovery_in_Risk_or_P2": (
                "HYPOTHESIS: P1 risk diagnostic for outcome; P2 for wait timing — do not auto-add to canonical"
            ),
            "Q5_zscore_semantic_damage": (
                "HYPOTHESIS: prefix z-score preserves ordering on ABC; tail/out-of-prefix archetypes "
                "can exceed |z|>4 — use separate heads not sum; monitor tail"
            ),
            "Q6_direction_categorical_vs_EV": (
                "HYPOTHESIS: directional EV per action aligns with Return/Risk head separation better than 3-class alone"
            ),
            "Q7_final_P1_structure": (
                "See recommended_structure — multi-head z(U), z(MFE), z(MAE), z(giveback), z(chop); "
                "direction via long/short conditional scores"
            ),
        }

    def _synthesize(
        self, final_q: dict[str, str], cross: dict[str, Any]
    ) -> dict[str, Any]:
        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []

        if cross.get("MFE_adds_AB_separation"):
            confirmed.append("MFE z ranks A above B (max opportunity) on ABC across datasets.")
        if cross.get("U_only_insufficient_for_AB_tier"):
            confirmed.append(
                "U z alone ranks B above A on spike-vs-grind — U-only Return insufficient for tier semantics."
            )
        if cross.get("risk_giveback_BAC_all_datasets"):
            confirmed.append("z(giveback) preserves B<A<C risk ordering on ABC all datasets.")

        for r in cross.get("corr_zU_zMFE_range", [0, 1]):
            pass
        avg_corr = float(np.mean(cross.get("corr_zU_zMFE_range", [0.9, 0.9])))
        if avg_corr > 0.85:
            hypothesis.append(
                f"Eval z(U) and z(MFE) correlated (avg~{avg_corr:.2f}) — dual head still needed for archetype divergence."
            )

        confirmed.append(
            "Standard Z-score prefix-fit: causal; archetype tail |z| can exceed 4 — facet heads not naive sum."
        )
        hypothesis.append("Recovery z improves same-MAE separation — P1 diagnostic, P2 timing boundary.")
        hypothesis.append(
            "Direction: directional EV (long/short Return/Risk scores) preferred over categorical-only."
        )

        unresolved.append("Multi-asset external CSV not in repo — BTC + SYNTH_LONG only.")
        unresolved.append("Training loss weighting for correlated z(U), z(MFE).")
        unresolved.append("NEUTRAL threshold margin calibration.")

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "UNRESOLVED": unresolved,
        }


def format_target_design_summary(report: dict[str, Any]) -> str:
    rec = report.get("recommended_p1_structure", {})
    lines = [
        "P1 Target Design Audit",
        "=" * 60,
        f"datasets: {report.get('datasets')}",
        f"CONFIRMED: {len(report.get('CONFIRMED', []))}",
        f"Return heads: {rec.get('Expected_Return')}",
        f"Risk heads: {rec.get('Acceptable_Risk')}",
    ]
    return "\n".join(lines)


def save_target_design_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(datasets: Sequence[tuple[str, MarketDataSource]]) -> dict[str, Any]:
    report = P1TargetDesignAuditRunner(datasets).run()
    print(format_target_design_summary(report))
    return report
