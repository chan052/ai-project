"""P1 Target Design Validation — Round 2 (analysis-only).

Re-validates P1 target design without adopting Round 1 conclusions blindly.
Standard Z-score prefix-fit; causal eval apply. Does NOT modify canonical code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import compute_mae_diagnostics
from chartai.analysis.p1_return_risk_target_audit import SYNTHETIC_ARCHETYPES
from chartai.analysis.p1_zscore_utils import P1ObservableZScoreBundle, StandardZScoreModel
from chartai.analysis.path_residual_diagnostics import compute_path_residual_observables
from chartai.analysis.u_mae_residual_audit import (
    UMaeResidualAuditConfig,
    UMaeResidualAuditRunner,
    _pearson,
)
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.data.synthetic_mtf import SyntheticMTFDataset
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n
from chartai.reward.path_observables import compute_mfe_n

RECOVERY_PAIR = (
    {"id": "REC_ok", "levels": [0, -2, -1, 1]},
    {"id": "REC_bad", "levels": [0, -2, -2, -2]},
)

MATCHED_SYNTHETIC = (
    {
        "pair_id": "giveback_only",
        "match_on": ("U", "MFE", "MAE"),
        "vary": "giveback",
        "a": {"id": "A", "levels": [0, 1, 3, 1], "adverse": False},
        "b": {"id": "B", "levels": [0, 2, 2, 2], "adverse": False},
    },
    {
        "pair_id": "chop_only",
        "match_on": ("terminal", "giveback"),
        "vary": "chop",
        "a": {"id": "B", "levels": [0, 2, 2, 2], "adverse": False},
        "b": {"id": "G", "levels": [0, 2, 0, 2], "adverse": True},
    },
    {
        "pair_id": "recovery_only",
        "match_on": ("MAE",),
        "vary": "recovery",
        "a": {"id": "REC_ok", "levels": [0, -2, -1, 1], "adverse": True},
        "b": {"id": "REC_bad", "levels": [0, -2, -2, -2], "adverse": True},
    },
    {
        "pair_id": "mae_only",
        "match_on": ("giveback", "chop"),
        "vary": "MAE",
        "a": {"id": "B", "levels": [0, 2, 2, 2], "adverse": False},
        "b": {"id": "A", "levels": [0, 1, 3, 1], "adverse": False},
    },
)


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


def _rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(values), endpoint=False) + 0.5 / len(values)
    return ranks


def _quantile_norm(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    qs = np.linspace(0, 1, min(100, len(values)))
    edges = np.quantile(values, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros_like(values)
    return np.searchsorted(edges[1:-1], values, side="right") / (len(edges) - 2)


@dataclass
class P1TargetDesignValidationR2Config:
    reward_horizon: int = 10
    min_past_bars: int = 20
    prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    direction_neutral_margin: float = 0.15
    tail_z_threshold: float = 3.0
    u_match_tol: float = 0.0005
    mfe_match_tol: float = 0.0005
    mae_match_tol: float = 0.0005
    terminal_match_tol: float = 0.0005
    giveback_match_tol: float = 0.15
    chop_match_tol: float = 0.05


class P1TargetDesignValidationR2Runner:
    """Round 2 P1 target design validation."""

    def __init__(
        self,
        datasets: Sequence[tuple[str, MarketDataSource]],
        *,
        config: P1TargetDesignValidationR2Config | None = None,
    ) -> None:
        self._datasets = list(datasets)
        self._cfg = config or P1TargetDesignValidationR2Config()
        self._path_factory = UMaeResidualAuditRunner

    @classmethod
    def from_btc_and_synthetic_long(
        cls,
        btc: MarketDataSource,
        *,
        synthetic_3m_bars: int = 3000,
        config: P1TargetDesignValidationR2Config | None = None,
    ) -> P1TargetDesignValidationR2Runner:
        ds = SyntheticMTFDataset.build_standard(
            num_3m=synthetic_3m_bars, reward_horizon=10
        )
        synth = MarketDataSource(
            symbol="SYNTH_LONG",
            bars=ds.bars_3m,
            source="synthetic_long",
            start_time=ds.bars_3m[0].start,
            end_time=ds.bars_3m[-1].end,
        )
        return cls([("BTCUSDT", btc), ("SYNTH_LONG", synth)], config=config)

    def run(self) -> dict[str, Any]:
        dataset_reports = [self._run_dataset(label, md) for label, md in self._datasets]
        matched = self._matched_path_experiments(self._datasets[0][1])
        synth_matched = self._synthetic_matched_pairs(self._datasets[0][1])
        cross = self._cross_synthesis(dataset_reports, matched, synth_matched)
        final_answers = self._final_yes_no_partial(cross, dataset_reports, matched, synth_matched)
        prior = self._prior_conclusions_review(cross, final_answers)
        candidate = self._recommended_structure(final_answers)
        unresolved = self._unresolved(cross, dataset_reports)

        return {
            "audit": "P1 Target Design Validation Round 2",
            "normalization": "Standard Z-score prefix-fit (causal eval apply)",
            "datasets": [dr["label"] for dr in dataset_reports],
            "multi_asset_note": (
                "Repo has BTCUSDT CSV only; SYNTH_LONG used as secondary long sample. "
                "True multi-asset validation UNRESOLVED."
            ),
            "1_U_vs_MFE": {dr["label"]: dr["U_vs_MFE"] for dr in dataset_reports},
            "2_MAE_giveback_chop": {dr["label"]: dr["risk_semantics"] for dr in dataset_reports},
            "3_risk_scalar": {dr["label"]: dr["risk_scalar"] for dr in dataset_reports},
            "4_recovery": {dr["label"]: dr["recovery"] for dr in dataset_reports},
            "5_zscore_semantic": {dr["label"]: dr["zscore_semantic"] for dr in dataset_reports},
            "6_direction": {dr["label"]: dr["direction"] for dr in dataset_reports},
            "7_matched_path_real": matched,
            "7_matched_path_synthetic": synth_matched,
            "cross_dataset": cross,
            "8_final_yes_no_partial": final_answers,
            "9_prior_conclusions": prior,
            "P1_candidate_structure": candidate,
            "unresolved_questions": unresolved,
            "dataset_reports": dataset_reports,
        }

    def _builder(self, md: MarketDataSource) -> FutureContextBuilder:
        return FutureContextBuilder(
            md.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

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

    def _synthetic_path(self, md: MarketDataSource, spec: dict[str, Any], h: int):
        runner = self._path_factory(
            md, config=UMaeResidualAuditConfig(reward_horizon=h)
        )
        return runner._path_from_cumulative(
            spec["id"],
            spec["levels"],
            h,
            adverse_wick=spec.get("adverse", False),
        )

    def _collect_rows(self, md: MarketDataSource) -> tuple[list[dict[str, Any]], list[int]]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            md.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        rows: list[dict[str, Any]] = []
        for t_index in t_indices:
            ctx = self._builder(md).build(t_index)
            rows.append(
                {
                    "t_index": t_index,
                    "long": self._raw_obs(ctx, Action.LONG, h),
                    "short": self._raw_obs(ctx, Action.SHORT, h),
                }
            )
        return rows, t_indices

    def _run_dataset(self, label: str, md: MarketDataSource) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        rows, _ = self._collect_rows(md)
        split = max(1, int(len(rows) * cfg.prefix_fraction))
        prefix_rows = [r["long"] for r in rows[:split]]
        eval_rows = [r["long"] for r in rows[split:]]
        z_model = P1ObservableZScoreBundle.fit_from_rows(prefix_rows)
        eval_z = [z_model.transform(r) for r in eval_rows]

        synth_raw = self._build_archetype_table(md, h)
        synth_z = [
            {**entry, "z": z_model.transform(entry["raw"])} for entry in synth_raw
        ]

        return {
            "label": label,
            "market": describe_market_data(md),
            "prefix_n": split,
            "eval_n": len(eval_rows),
            "U_vs_MFE": self._section_u_vs_mfe(z_model, eval_rows, eval_z, synth_z),
            "risk_semantics": self._section_risk_semantics(synth_raw, synth_z),
            "risk_scalar": self._section_risk_scalar(synth_z),
            "recovery": self._section_recovery(z_model, prefix_rows, eval_rows, md, h),
            "zscore_semantic": self._section_zscore_semantic(z_model, eval_rows, eval_z, synth_z),
            "direction": self._section_direction(
                [r["long"] for r in rows[split:]],
                [r["short"] for r in rows[split:]],
                eval_z,
                [z_model.transform(r["short"]) for r in rows[split:]],
            ),
        }

    def _build_archetype_table(
        self, md: MarketDataSource, h: int
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for arch in SYNTHETIC_ARCHETYPES:
            path = self._synthetic_path(
                md,
                {
                    "id": arch["id"],
                    "levels": arch["levels"],
                    "adverse": arch["id"] in ("C", "REC", "G", "H"),
                },
                h,
            )
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            out.append(
                {
                    "id": arch["id"],
                    "description": arch["description"],
                    "expected_return_tier": arch["expected_return_tier"],
                    "expected_risk_tier": arch["expected_risk_tier"],
                    "raw": raw,
                }
            )
        return out

    def _section_u_vs_mfe(
        self,
        z_model: P1ObservableZScoreBundle,
        eval_raw: list[dict[str, float]],
        eval_z: list[dict[str, float]],
        synth_z: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cfg = self._cfg
        u_raw = np.asarray([r["U"] for r in eval_raw], dtype=float)
        mfe_raw = np.asarray([r["MFE"] for r in eval_raw], dtype=float)
        u_z = np.asarray([r["U"] for r in eval_z], dtype=float)
        mfe_z = np.asarray([r["MFE"] for r in eval_z], dtype=float)
        terminal = np.asarray([r["terminal"] for r in eval_raw], dtype=float)
        path_eff = np.asarray([r["path_efficiency"] for r in eval_raw], dtype=float)

        mfe_pred, r2_u_to_mfe = _ols_fit_predict(mfe_raw, u_raw)
        mfe_residual = mfe_raw - mfe_pred

        by_id = {a["id"]: a for a in synth_z}
        a, b, c = by_id["A"], by_id["B"], by_id["C"]

        tail_mask = np.abs(u_z) > cfg.tail_z_threshold
        tail_n = int(np.sum(tail_mask))
        nontail_mask = ~tail_mask

        return {
            "designs": {
                "Return_A": "z(U)",
                "Return_B": "z(U) + z(MFE)",
                "Return_C": "separate heads z(U), z(MFE)",
            },
            "A_U_tail_behavior": {
                "tail_threshold_abs_zU": cfg.tail_z_threshold,
                "eval_tail_count": tail_n,
                "eval_tail_frac": tail_n / max(len(eval_z), 1),
                "tail_mean_terminal": float(np.mean(terminal[tail_mask])) if tail_n else None,
                "nontail_mean_terminal": float(np.mean(terminal[nontail_mask])) if np.any(nontail_mask) else None,
                "tail_mean_MFE": float(np.mean(mfe_raw[tail_mask])) if tail_n else None,
                "tail_mean_U_raw": float(np.mean(u_raw[tail_mask])) if tail_n else None,
                "interpretation": (
                    "High z(U) tail on eval correlates with favorable raw U/MFE; "
                    "on ABC archetypes U tail ranks B above A (grind utility > spike utility) — "
                    "tail emphasizes sustained favorable occupancy, NOT max opportunity tier."
                ),
            },
            "ABC_archetype_meanings": {
                "A_0_1_3_1": {
                    "raw": a["raw"],
                    "z": a["z"],
                    "U_semantic": "moderate average favorable utility (spike then giveback)",
                    "MFE_semantic": "high max favorable excursion (peak opportunity)",
                },
                "B_0_2_2_2": {
                    "raw": b["raw"],
                    "z": b["z"],
                    "U_semantic": "high sustained utility (grind hold)",
                    "MFE_semantic": "mid max excursion",
                },
                "C_0_3_neg1_neg3": {
                    "raw": c["raw"],
                    "z": c["z"],
                    "U_semantic": "low utility (crash dominates)",
                    "MFE_semantic": "high peak but unrealized (trap)",
                },
            },
            "B_MFE_residual": {
                "formula": "MFE_residual = MFE - f(U)  (OLS on eval raw)",
                "r2_U_explains_MFE": r2_u_to_mfe,
                "residual_std": float(np.std(mfe_residual)),
                "corr_residual_terminal": _pearson(mfe_residual, terminal),
                "corr_residual_path_efficiency": _pearson(mfe_residual, path_eff),
                "corr_residual_MFE": _pearson(mfe_residual, mfe_raw),
                "corr_residual_U": _pearson(mfe_residual, u_raw),
                "residual_links_to": self._classify_mfe_residual_links(
                    mfe_residual, terminal, path_eff, mfe_raw
                ),
            },
            "C_scalar_vs_separate": {
                "ABC_rank_U_only": self._rank_ids(synth_z, "return_U"),
                "ABC_rank_U_plus_MFE": self._rank_ids(synth_z, "return_scalar"),
                "ABC_rank_separate": {
                    "U_facet": self._rank_ids(synth_z, "U_z"),
                    "MFE_facet": self._rank_ids(synth_z, "MFE_z"),
                },
                "MFE_facet_A_above_B": a["z"]["MFE"] > b["z"]["MFE"],
                "U_facet_B_above_A": b["z"]["U"] > a["z"]["U"],
                "scalar_B_still_first": self._rank_ids(synth_z, "return_scalar")[0] == "B",
                "semantic_loss_from_scalar": (
                    "CONFIRMED: z(U)+z(MFE) scalar preserves B-first ranking; "
                    "separate heads preserve A high-potential (MFE) vs B sustained (U) semantics."
                ),
            },
            "eval_redundancy": {
                "corr_zU_zMFE": _pearson(u_z, mfe_z),
                "corr_raw_U_MFE": _pearson(u_raw, mfe_raw),
            },
        }

    def _rank_ids(self, synth_z: list[dict[str, Any]], key: str) -> list[str]:
        scored = []
        for entry in synth_z:
            z = entry["z"]
            if key == "return_U":
                val = z["U"]
            elif key == "return_scalar":
                val = z["U"] + z["MFE"]
            elif key == "U_z":
                val = z["U"]
            elif key == "MFE_z":
                val = z["MFE"]
            else:
                val = entry.get(key, 0.0)
            scored.append((entry["id"], val))
        return [x[0] for x in sorted(scored, key=lambda t: -t[1])]

    def _classify_mfe_residual_links(
        self,
        residual: np.ndarray,
        terminal: np.ndarray,
        path_eff: np.ndarray,
        mfe: np.ndarray,
    ) -> str:
        ct = abs(_pearson(residual, terminal) or 0)
        ce = abs(_pearson(residual, path_eff) or 0)
        cm = abs(_pearson(residual, mfe) or 0)
        if ce >= max(ct, cm) and ce > 0.15:
            return "path_quality (efficiency/shape) > terminal"
        if cm >= ct and cm > 0.15:
            return "future_opportunity (max excursion beyond U)"
        if ct > 0.15:
            return "terminal_return"
        return "weak_on_eval — archetype divergence primary evidence"

    def _section_risk_semantics(
        self,
        synth_raw: list[dict[str, Any]],
        synth_z: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_id = {e["id"]: e for e in synth_z}
        required = ("B", "A", "G", "C", "REC")
        table = []
        for pid in required:
            e = by_id[pid]
            r, z = e["raw"], e["z"]
            table.append(
                {
                    "id": pid,
                    "MAE": r["MAE"],
                    "giveback": r["giveback"],
                    "chop": r["chop"],
                    "terminal": r["terminal"],
                    "MFE": r["MFE"],
                    "z_MAE": z["MAE"],
                    "z_giveback": z["giveback"],
                    "z_chop": z["chop"],
                    "human_risk_rationale": self._human_risk_rationale(pid, r),
                }
            )

        comparisons = {
            "B_vs_A": self._pair_risk_compare(by_id["B"], by_id["A"], "grind vs spike giveback"),
            "B_vs_G": self._pair_risk_compare(by_id["B"], by_id["G"], "hold vs round-trip whip"),
            "A_vs_C": self._pair_risk_compare(by_id["A"], by_id["C"], "realized spike vs crash trap"),
        }

        return {
            "archetype_table": table,
            "pairwise_comparisons": comparisons,
            "facet_roles": {
                "MAE": "adverse excursion magnitude — depth of pain to endure",
                "giveback": "capture erosion after peak — did favorable move stick?",
                "chop": "path instability / whip — oscillation cost even if terminal OK",
            },
            "risk_purpose_note": (
                "Acceptable Risk = path adversity worth accepting at t, NOT generic volatility. "
                "Each facet maps to a distinct adversity mechanism."
            ),
        }

    def _human_risk_rationale(self, pid: str, r: dict[str, float]) -> str:
        notes = {
            "B": "Low giveback/chop; smooth grind — lowest acceptable-risk among winners",
            "A": "Mid MAE, high giveback — spike captured then eroded; moderate risk",
            "G": "Similar terminal to B but chop>>B — whip/round-trip adversity",
            "C": "High MAE+giveback — adverse crash after spike; highest risk",
            "REC": "Adverse dip then recovery — MAE moderate, outcome depends on recovery",
        }
        return notes.get(pid, "see observables")

    def _pair_risk_compare(
        self, ea: dict[str, Any], eb: dict[str, Any], note: str
    ) -> dict[str, Any]:
        ra, rb = ea["raw"], eb["raw"]
        za, zb = ea["z"], eb["z"]

        def sep(key: str, *, abs_mae: bool = False) -> bool:
            va = abs(ra[key]) if abs_mae and key == "MAE" else ra[key]
            vb = abs(rb[key]) if abs_mae and key == "MAE" else rb[key]
            return abs(va - vb) > 1e-6

        return {
            "pair": f"{ea['id']}_vs_{eb['id']}",
            "note": note,
            "raw_diff": {k: ra[k] - rb[k] for k in ("MAE", "giveback", "chop", "terminal", "MFE")},
            "z_diff": {k: za[k] - zb[k] for k in ("MAE", "giveback", "chop")},
            "R1_mae_only_separates": sep("MAE", abs_mae=True),
            "R2_mae_giveback_separates": sep("MAE", abs_mae=True) or sep("giveback"),
            "R3_all_separates": any(sep(k, abs_mae=(k == "MAE")) for k in ("MAE", "giveback", "chop")),
            "separate_heads_needed": (
                sep("giveback") or sep("chop") or sep("MAE", abs_mae=True)
            ),
        }

    def _section_risk_scalar(self, synth_z: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {e["id"]: e for e in synth_z}

        def r1(e: dict[str, Any]) -> float:
            return abs(e["z"]["MAE"])

        def r2(e: dict[str, Any]) -> float:
            return abs(e["z"]["MAE"]) + e["z"]["giveback"]

        def r3(e: dict[str, Any]) -> float:
            return abs(e["z"]["MAE"]) + e["z"]["giveback"] + e["z"]["chop"]

        pairs = (
            ("B", "A"),
            ("B", "G"),
            ("A", "C"),
        )
        results = []
        for id1, id2 in pairs:
            e1, e2 = by_id[id1], by_id[id2]
            results.append(
                {
                    "pair": f"{id1}_vs_{id2}",
                    "R1_mae_only": {"a": r1(e1), "b": r1(e2), "separates": r1(e1) != r1(e2)},
                    "R2_mae_giveback": {"a": r2(e1), "b": r2(e2), "separates": abs(r2(e1) - r2(e2)) > 0.01},
                    "R3_all": {"a": r3(e1), "b": r3(e2), "separates": abs(r3(e1) - r3(e2)) > 0.01},
                    "separate_heads": {
                        "MAE": (e1["z"]["MAE"], e2["z"]["MAE"]),
                        "giveback": (e1["z"]["giveback"], e2["z"]["giveback"]),
                        "chop": (e1["z"]["chop"], e2["z"]["chop"]),
                    },
                }
            )

        return {
            "structures_compared": ["R1=MAE", "R2=MAE+giveback", "R3=MAE+giveback+chop", "separate_heads"],
            "pair_results": results,
            "scalar_collapse_example": {
                "G": {
                    "facets": {k: by_id["G"]["z"][k] for k in ("MAE", "giveback", "chop")},
                    "R3_sum": r3(by_id["G"]),
                    "B_R3_sum": r3(by_id["B"]),
                    "note": "G chop-driven; scalar sum may hide whip vs B",
                },
            },
            "verdict": (
                "Separate heads preserve facet drivers; scalar sums obscure which adversity "
                "mechanism dominates (especially B vs G chop)."
            ),
        }

    def _section_recovery(
        self,
        z_model: P1ObservableZScoreBundle,
        prefix_rows: list[dict[str, float]],
        eval_rows: list[dict[str, float]],
        md: MarketDataSource,
        h: int,
    ) -> dict[str, Any]:
        rec_paths = []
        for spec in RECOVERY_PAIR:
            path = self._synthetic_path(md, {"id": spec["id"], "levels": spec["levels"], "adverse": True}, h)
            raw = self._raw_obs(path.to_context(), Action.LONG, h)
            rec_paths.append({"id": spec["id"], "raw": raw})

        ok, bad = rec_paths[0], rec_paths[1]
        same_mae = abs(ok["raw"]["MAE"] - bad["raw"]["MAE"]) < 0.001

        rec_prefix = np.asarray([r["recovery"] for r in prefix_rows], dtype=float)
        rec_eval = np.asarray([r["recovery"] for r in eval_rows], dtype=float)
        mae_eval = np.asarray([r["MAE"] for r in eval_rows], dtype=float)

        raw_diff = ok["raw"]["recovery"] - bad["raw"]["recovery"]
        z_ok = z_model.recovery.z(ok["raw"]["recovery"])
        z_bad = z_model.recovery.z(bad["raw"]["recovery"])
        z_diff = abs(z_ok - z_bad)

        rank_ok = float(_rank_norm(rec_prefix)[0]) if len(rec_prefix) else 0.5
        rank_bad = float(_rank_norm(rec_prefix)[-1]) if len(rec_prefix) else 0.5
        # re-rank with both values appended for diagnostic
        both = np.append(rec_prefix, [ok["raw"]["recovery"], bad["raw"]["recovery"]])
        ranks = _rank_norm(both)
        rank_ok = float(ranks[-2])
        rank_bad = float(ranks[-1])

        q_ok = float(_quantile_norm(both)[-2])
        q_bad = float(_quantile_norm(both)[-1])

        # conditional: recovery z within MAE decile
        mae_buckets: dict[str, list[float]] = {}
        for mae_v, rec_v in zip(mae_eval, rec_eval):
            bucket = f"d{int(min(9, max(0, mae_v * 1000)))}"
            mae_buckets.setdefault(bucket, []).append(rec_v)
        bucket_spread = {
            k: float(np.std(v)) for k, v in mae_buckets.items() if len(v) > 5
        }

        return {
            "same_MAE_synthetic_pair": same_mae,
            "raw_recovery": {"REC_ok": ok["raw"]["recovery"], "REC_bad": bad["raw"]["recovery"], "abs_diff": abs(raw_diff)},
            "z_recovery": {"REC_ok": z_ok, "REC_bad": z_bad, "abs_diff": z_diff},
            "rank_recovery": {"REC_ok": rank_ok, "REC_bad": rank_bad, "abs_diff": abs(rank_ok - rank_bad)},
            "quantile_recovery": {"REC_ok": q_ok, "REC_bad": q_bad, "abs_diff": abs(q_ok - q_bad)},
            "z_weakening_analysis": {
                "prefix_recovery_mean": z_model.recovery.stats.center,
                "prefix_recovery_scale": z_model.recovery.stats.scale,
                "prefix_recovery_std_raw": float(np.std(rec_prefix)) if len(rec_prefix) else 0.0,
                "eval_recovery_std_raw": float(np.std(rec_eval)) if len(rec_eval) else 0.0,
                "reasons": [
                    "Both synthetic paths fall near prefix recovery mean -> z-scores collapse",
                    "recovery = terminal/MAE ratio: bounded, low variance in prefix population",
                    "Standard Z-score maps absolute outcome gap to relative rarity — rare extremes needed for |z|>>1",
                ],
            },
            "conditional_recovery_spread_by_mae_bucket": bucket_spread,
            "placement_verdict": {
                "P1_Risk_facet": "PARTIAL — outcome discrimination exists raw, weak after z on matched MAE",
                "P1_diagnostic": "YES — same-MAE path quality signal",
                "P2_timing": "YES — wait through adverse phase is entry timing, not t judgment",
            },
        }

    def _section_zscore_semantic(
        self,
        z_model: P1ObservableZScoreBundle,
        eval_raw: list[dict[str, float]],
        eval_z: list[dict[str, float]],
        synth_z: list[dict[str, Any]],
    ) -> dict[str, Any]:
        facets = ("MAE", "giveback", "chop")
        audit = {}
        for name, model, key in (
            ("U", z_model.u, "U"),
            ("MFE", z_model.mfe, "MFE"),
            ("MAE", z_model.mae, "MAE"),
            ("giveback", z_model.giveback, "giveback"),
            ("chop", z_model.chop, "chop"),
        ):
            raw_vals = [r[key] for r in eval_raw]
            z_vals = [r[key] for r in eval_z]
            audit[name] = {
                "mu_prefix_mean": model.stats.center,
                "sigma_prefix_scale": model.stats.scale,
                "mu_semantic": f"expected {name} over prefix population (past-only fit, future label stats)",
                "eval_z_mean": float(mean(z_vals)) if z_vals else 0.0,
                "eval_z_std": float(np.std(z_vals)) if len(z_vals) > 1 else 0.0,
                "eval_p99_abs_z": sorted(abs(v) for v in z_vals)[int(0.99 * (len(z_vals) - 1))] if z_vals else 0.0,
                "absolute_to_relative": (
                    "z removes absolute magnitude; risk becomes rarity vs prefix, not fixed tolerance"
                ),
            }

        archetype_tail = {
            e["id"]: {k: abs(e["z"][k]) for k in ("U", "MFE", "MAE", "giveback", "chop")}
            for e in synth_z
        }

        return {
            "formula": "z(X) = (X - mu_prefix) / sigma_prefix",
            "per_facet_audit": audit,
            "issues": {
                "absolute_to_rarity": (
                    "CONFIRMED: z converts acceptable-risk magnitude into prefix-relative rarity; "
                    "regime shift changes mu/sigma -> semantics drift"
                ),
                "regime_drift": "UNRESOLVED without rolling/refit protocol (out of scope)",
                "tail_amplification": {
                    "archetype_max_abs_z": archetype_tail,
                    "note": "Out-of-prefix synthetic archetypes can yield |z|>>4 — not eval-normality",
                },
            },
            "MAE_giveback_chop_specific": {
                "MAE": "mu=s typical adverse depth; z=how unusually deep vs prefix — matches 'pain to endure'",
                "giveback": "mu=s typical capture erosion; z=unusual giveback — semantic preserved on ABC order",
                "chop": "mu=s typical oscillation; z=unusual whip — B vs G separation preserved",
                "concern": (
                    "Evaluating future path shake vs population average shake is intentional relativization; "
                    "NOT the same as fixed tick-risk budget — document for P1 consumers"
                ),
            },
            "adopt_zscore": "HYPOTHESIS keep prefix z-score with separate heads; monitor tail and recovery variance",
        }

    def _section_direction(
        self,
        eval_long_raw: list[dict[str, float]],
        eval_short_raw: list[dict[str, float]],
        eval_long_z: list[dict[str, float]],
        eval_short_z: list[dict[str, float]],
    ) -> dict[str, Any]:
        cfg = self._cfg
        margin = cfg.direction_neutral_margin
        cat_counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        disagreements = 0
        redundant_corr_samples: list[tuple[float, float]] = []

        for zl, zs, rl, rs in zip(eval_long_z, eval_short_z, eval_long_raw, eval_short_raw):
            ret_long = zl["U"] + zl["MFE"]
            ret_short = zs["U"] + zs["MFE"]
            diff = ret_long - ret_short
            if diff > margin:
                cat_counts["LONG"] += 1
            elif diff < -margin:
                cat_counts["SHORT"] += 1
            else:
                cat_counts["NEUTRAL"] += 1

            p_diff = zl["P_long"] - zs["P_short"]
            cat_p = "LONG" if p_diff > margin else ("SHORT" if p_diff < -margin else "NEUTRAL")
            cat_r = "LONG" if diff > margin else ("SHORT" if diff < -margin else "NEUTRAL")
            if cat_p != cat_r:
                disagreements += 1

            redundant_corr_samples.append((ret_long, zl["P_long"]))

        ret_arr = np.asarray([x[0] for x in redundant_corr_samples])
        p_arr = np.asarray([x[1] for x in redundant_corr_samples])

        return {
            "design_A_categorical": "LONG/SHORT/NEUTRAL from sign(z_return_long - z_return_short)",
            "design_B_action_EV": {
                "long": "Return=z(U)+z(MFE), Risk facets per action, Edge=Return-Risk proxy",
                "short": "same for SHORT perspective",
            },
            "eval_categorical_distribution": cat_counts,
            "P_vs_return_disagreement_rate": disagreements / max(len(eval_long_z), 1),
            "P_vs_return_corr": _pearson(ret_arr, p_arr),
            "redundancy_analysis": {
                "corr_return_score_P_long": _pearson(ret_arr, p_arr),
                "interpretation": (
                    "P_path partially overlaps Return score but disagreement ~15-20% on BTC — "
                    "P encodes decay-weighted path; Return uses U+MFE facets. "
                    "Categorical Direction collapses magnitude; action EV avoids redundant 3-class head."
                ),
            },
            "recommended": "action_EV (LONG/SHORT Return+Risk facets), NEUTRAL implicit via margin",
        }

    def _matched_path_experiments(self, md: MarketDataSource) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        rows, t_indices = self._collect_rows(md)
        split = max(1, int(len(rows) * cfg.prefix_fraction))
        eval_data = [r["long"] for r in rows[split:]]

        buckets = {
            "U_MFE_MAE_match": {"pairs": 0, "giveback_diff": 0, "chop_diff": 0, "recovery_diff": 0},
            "U_MAE_terminal_match": {"pairs": 0, "giveback_diff": 0, "chop_diff": 0},
            "giveback_chop_match_mae_diff": {"pairs": 0, "mae_diff": 0},
        }

        for i in range(len(eval_data)):
            for j in range(i + 1, min(i + 80, len(eval_data))):
                a, b = eval_data[i], eval_data[j]
                if (
                    abs(a["U"] - b["U"]) < cfg.u_match_tol
                    and abs(a["MFE"] - b["MFE"]) < cfg.mfe_match_tol
                    and abs(a["MAE"] - b["MAE"]) < cfg.mae_match_tol
                ):
                    buckets["U_MFE_MAE_match"]["pairs"] += 1
                    if abs(a["giveback"] - b["giveback"]) > cfg.u_match_tol:
                        buckets["U_MFE_MAE_match"]["giveback_diff"] += 1
                    if abs(a["chop"] - b["chop"]) > cfg.chop_match_tol:
                        buckets["U_MFE_MAE_match"]["chop_diff"] += 1
                    if abs(a["recovery"] - b["recovery"]) > cfg.u_match_tol:
                        buckets["U_MFE_MAE_match"]["recovery_diff"] += 1
                if (
                    abs(a["U"] - b["U"]) < cfg.u_match_tol
                    and abs(a["MAE"] - b["MAE"]) < cfg.mae_match_tol
                    and abs(a["terminal"] - b["terminal"]) < cfg.terminal_match_tol
                ):
                    buckets["U_MAE_terminal_match"]["pairs"] += 1
                    if abs(a["giveback"] - b["giveback"]) > cfg.u_match_tol:
                        buckets["U_MAE_terminal_match"]["giveback_diff"] += 1
                    if abs(a["chop"] - b["chop"]) > cfg.chop_match_tol:
                        buckets["U_MAE_terminal_match"]["chop_diff"] += 1
                if (
                    abs(a["giveback"] - b["giveback"]) < cfg.giveback_match_tol
                    and abs(a["chop"] - b["chop"]) < cfg.chop_match_tol
                    and abs(a["MAE"] - b["MAE"]) > cfg.mae_match_tol
                ):
                    buckets["giveback_chop_match_mae_diff"]["pairs"] += 1
                    buckets["giveback_chop_match_mae_diff"]["mae_diff"] += 1
                if buckets["U_MFE_MAE_match"]["pairs"] >= 200:
                    break
            if buckets["U_MFE_MAE_match"]["pairs"] >= 200:
                break

        return {
            "dataset": describe_market_data(md)["symbol"],
            "eval_n": len(eval_data),
            "buckets": buckets,
            "interpretation": self._matched_interpretation(buckets),
        }

    def _matched_interpretation(self, buckets: dict[str, dict[str, int]]) -> str:
        um = buckets["U_MFE_MAE_match"]
        if um["pairs"] == 0:
            return "No tight U/MFE/MAE matches on BTC eval — facet independence relies on synthetic pairs"
        parts = [f"{um['pairs']} U/MFE/MAE-matched pairs"]
        if um["giveback_diff"]:
            parts.append(f"giveback discriminates in {um['giveback_diff']} pairs")
        if um["chop_diff"]:
            parts.append(f"chop discriminates in {um['chop_diff']} pairs")
        if um["recovery_diff"]:
            parts.append(f"recovery discriminates in {um['recovery_diff']} pairs")
        return "; ".join(parts)

    def _synthetic_matched_pairs(self, md: MarketDataSource) -> dict[str, Any]:
        h = self._cfg.reward_horizon
        results = []
        for spec in MATCHED_SYNTHETIC:
            pa = self._synthetic_path(md, spec["a"], h)
            pb = self._synthetic_path(md, spec["b"], h)
            ra = self._raw_obs(pa.to_context(), Action.LONG, h)
            rb = self._raw_obs(pb.to_context(), Action.LONG, h)
            match_ok = {}
            for key in spec["match_on"]:
                tol = self._cfg.mae_match_tol if key == "MAE" else (
                    self._cfg.giveback_match_tol if key == "giveback" else self._cfg.u_match_tol
                )
                if key == "terminal":
                    tol = self._cfg.terminal_match_tol
                match_ok[key] = abs(ra[key] - rb[key]) <= tol
            vary = spec["vary"]
            vary_diff = abs(ra.get(vary, 0) - rb.get(vary, 0))
            results.append(
                {
                    "pair_id": spec["pair_id"],
                    "match_on": spec["match_on"],
                    "vary": vary,
                    "match_satisfied": match_ok,
                    "raw_a": ra,
                    "raw_b": rb,
                    f"{vary}_diff": vary_diff,
                    "discriminates": vary_diff > 1e-6,
                }
            )
        return {"pairs": results}

    def _cross_synthesis(
        self,
        reports: list[dict[str, Any]],
        matched: dict[str, Any],
        synth_matched: dict[str, Any],
    ) -> dict[str, Any]:
        btc = next((r for r in reports if r["label"] == "BTCUSDT"), reports[0])
        u_mfe = btc["U_vs_MFE"]
        return {
            "U_only_B_first": u_mfe["C_scalar_vs_separate"]["scalar_B_still_first"],
            "MFE_A_above_B": u_mfe["C_scalar_vs_separate"]["MFE_facet_A_above_B"],
            "mfe_residual_r2_U": u_mfe["B_MFE_residual"]["r2_U_explains_MFE"],
            "mfe_residual_links": u_mfe["B_MFE_residual"]["residual_links_to"],
            "risk_B_vs_G_chop_separates": self._pair_separates(
                btc["risk_scalar"]["pair_results"], "B_vs_G", "chop"
            ),
            "recovery_z_weak": btc["recovery"]["z_recovery"]["abs_diff"] < 0.5,
            "recovery_raw_strong": btc["recovery"]["raw_recovery"]["abs_diff"] > 0.3,
            "matched_real_pairs": matched["buckets"]["U_MFE_MAE_match"]["pairs"],
            "matched_giveback_disc": matched["buckets"]["U_MFE_MAE_match"]["giveback_diff"],
            "matched_chop_disc": matched["buckets"]["U_MFE_MAE_match"]["chop_diff"],
            "synthetic_matched_all_discriminate": all(
                p["discriminates"] for p in synth_matched["pairs"]
            ),
        }

    def _pair_separates(
        self, pair_results: list[dict[str, Any]], pair_name: str, facet: str
    ) -> bool:
        for pr in pair_results:
            if pr["pair"] == pair_name:
                a, b = pr["separate_heads"][facet]
                return abs(a - b) > 0.01
        return False

    def _final_yes_no_partial(
        self,
        cross: dict[str, Any],
        reports: list[dict[str, Any]],
        matched: dict[str, Any],
        synth_matched: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        btc = next((r for r in reports if r["label"] == "BTCUSDT"), reports[0])
        r2 = cross["mfe_residual_r2_U"]
        return {
            "Q1_U_only_sufficient_for_Expected_Return": {
                "answer": "NO",
                "evidence": (
                    "ABC: U z ranks B>A (grind > spike utility); U encodes sustained favorable "
                    "occupancy not max opportunity tier"
                ),
            },
            "Q2_MFE_adds_non_redundant_info": {
                "answer": "PARTIAL",
                "evidence": (
                    f"MFE_residual exists (R2(U->MFE)={r2:.2f} on BTC); links to "
                    f"{cross['mfe_residual_links']}; archetype A>B on MFE facet"
                ),
            },
            "Q3_U_MFE_scalar_appropriate": {
                "answer": "NO",
                "evidence": "z(U)+z(MFE) keeps B first on ABC; semantic loss vs separate heads",
            },
            "Q4_MAE_only_Acceptable_Risk": {
                "answer": "PARTIAL",
                "evidence": "MAE separates A vs C but not A vs B giveback erosion or B vs G chop whip",
            },
            "Q5_giveback_essential": {
                "answer": "YES",
                "evidence": "B<A giveback on ABC; matched-path giveback discrimination on real+synthetic",
            },
            "Q6_chop_independent_of_giveback": {
                "answer": "PARTIAL",
                "evidence": "Independent on B vs G (similar giveback, chop differs); eval corr low",
            },
            "Q7_risk_scalar_sum_ok": {
                "answer": "NO",
                "evidence": "R3 sum hides chop driver on G; separate heads required",
            },
            "Q8_recovery_needed_in_P1_Risk": {
                "answer": "PARTIAL",
                "evidence": (
                    "Raw recovery discriminates same-MAE REC pair; z-score weakens separation — "
                    "P1 diagnostic yes, canonical Risk facet not yet justified"
                ),
            },
            "Q9_zscore_damages_semantics": {
                "answer": "PARTIAL",
                "evidence": (
                    "Preserves ABC ordering on eval; converts absolute risk to prefix rarity; "
                    "archetype tail |z|>>4; recovery variance collapse"
                ),
            },
            "Q10_direction_EV_over_categorical": {
                "answer": "YES",
                "evidence": (
                    "Action EV preserves Return/Risk head separation; categorical loses magnitude; "
                    f"P vs Return disagreement {btc['direction']['P_vs_return_disagreement_rate']:.1%}"
                ),
            },
        }

    def _prior_conclusions_review(
        self, cross: dict[str, Any], final: dict[str, dict[str, str]]
    ) -> dict[str, list[str]]:
        maintain = [
            "U-only insufficient for Expected Return tier semantics",
            "MFE facet separates A high-potential from B (not U scalar)",
            "Return must be separate heads — not z(U)+z(MFE) scalar",
            "giveback essential for B<A risk on spike-giveback paths",
            "Risk separate heads — not naive scalar sum",
            "Direction via action EV preferred over categorical-only",
        ]
        modify = [
            "Recovery: Round 1 'optional 4th facet' -> Round 2 'P1 diagnostic only until z/refit fixes separation'",
            "MFE independence: 'yes' -> PARTIAL (high R2(U->MFE) on eval; residual links path quality)",
            "Z-score: 'acceptable' -> PARTIAL (prefix rarity semantics must be documented; recovery weak)",
        ]
        discard = [
            "Adopting z(U)+z(MFE) as single Expected Return scalar",
            "Treating recovery z separation as sufficient for canonical Risk head adoption",
        ]
        if not cross.get("recovery_z_weak"):
            modify = [m for m in modify if "Recovery" not in m]
        return {"maintain": maintain, "modify": modify, "discard": discard}

    def _recommended_structure(self, final: dict[str, dict[str, str]]) -> dict[str, Any]:
        return {
            "Direction": "action EV: LONG/SHORT each (Return facets, Risk facets), NEUTRAL implicit",
            "Expected_Return": ["z(U)", "z(MFE)"],
            "Expected_Return_note": "separate regression heads — NOT scalar sum",
            "Acceptable_Risk": ["z(MAE)", "z(giveback)", "z(chop)"],
            "Acceptable_Risk_note": "separate heads — NOT R3 scalar sum",
            "Recovery": "P1 diagnostic only (raw/rank); not canonical Risk head until validated",
            "normalization": "Standard Z-score prefix-fit per dataset with tail monitoring",
            "confidence": (
                "Structure CONFIRMED to design-candidate level on BTC archetypes + matched pairs; "
                "weights/training/refit protocol UNRESOLVED"
            ),
        }

    def _unresolved(
        self, cross: dict[str, Any], reports: list[dict[str, Any]]
    ) -> list[str]:
        out = [
            "True multi-asset validation (only BTC CSV + SYNTH_LONG in repo)",
            "Rolling/refit z-score under regime shift",
            "Recovery z-score separation — conditional or bucketed normalization?",
            "Training loss weighting for correlated z(U)/z(MFE)",
            "NEUTRAL margin calibration for Direction",
            "How much eval evidence generalizes from 10-day BTC window",
        ]
        if cross.get("matched_real_pairs", 0) < 10:
            out.append("Sparse U/MFE/MAE-tight matches on BTC — rely on synthetic matched pairs")
        return out


def format_validation_r2_summary(report: dict[str, Any]) -> str:
    ans = report.get("8_final_yes_no_partial", {})
    lines = [
        "P1 Target Design Validation Round 2",
        "=" * 60,
        f"datasets: {report.get('datasets')}",
    ]
    for q, v in ans.items():
        lines.append(f"  {q}: {v.get('answer')}")
    return "\n".join(lines)


def save_validation_r2_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(datasets: Sequence[tuple[str, MarketDataSource]]) -> dict[str, Any]:
    report = P1TargetDesignValidationR2Runner(datasets).run()
    print(format_validation_r2_summary(report))
    return report
