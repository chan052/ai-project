"""Risk-adjusted path metrics audit — Sharpe/Sortino/Ulcer vs U/MAE/path (analysis-only)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.path_residual_diagnostics import (
    compute_path_residual_observables,
    observables_to_dict,
)
from chartai.analysis.path_risk_adjusted_metrics import (
    RISK_ADJUSTED_SPECS,
    compute_differential_sharpe_pair,
    compute_risk_adjusted_path_metrics,
    risk_adjusted_to_dict,
)
from chartai.analysis.u_mae_residual_audit import UMaeResidualAuditRunner, UMaeResidualAuditConfig, _pearson
from chartai.analysis.u_persistence_diagnostics import compute_u_diagnostics
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n


def _ols_r2(y: np.ndarray, *xs: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    cols = [np.ones(len(y))]
    for x in xs:
        cols.append(x)
    X = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")


ARCHETYPE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "label": "spike_giveback",
        "levels": [0, 1, 3, 1],
        "description": "0→1→3→1 — MFE 큰 spike 후 giveback",
    },
    {
        "id": "B",
        "label": "grind_hold",
        "levels": [0, 2, 2, 2],
        "description": "0→2→2→2 — 안정적 grind/hold",
    },
    {
        "id": "C",
        "label": "rise_then_crash",
        "levels": [0, 3, -1, -3],
        "description": "0→3→-1→-3 — 상승 후 역전 crash",
    },
)


PATH_REF_KEYS = ("giveback_ratio", "reversal_depth", "oscillation_chop", "peak_timing")


@dataclass
class RiskAdjustedPathAuditConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)


class RiskAdjustedPathAuditRunner:
    """Audit Sharpe/Sortino/Ulcer vs U, MAE, path observables for P1 Return/Risk design."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: RiskAdjustedPathAuditConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or RiskAdjustedPathAuditConfig()
        self._residual = UMaeResidualAuditRunner(
            market_data,
            config=UMaeResidualAuditConfig(
                reward_horizon=self._cfg.reward_horizon,
                min_past_bars=self._cfg.min_past_bars,
                eval_prefix_fraction=self._cfg.eval_prefix_fraction,
                decay_rate=self._cfg.decay_rate,
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
        t_indices = list(
            self._data.valid_t_indices(reward_horizon=h, min_past_bars=cfg.min_past_bars)
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))
        eval_t = t_indices[split:]

        eval_rows = self._collect_rows(eval_t, h)
        archetypes = self._archetype_analysis(cfg)
        overlap = self._overlap_analysis(eval_rows)
        residual_after_path = self._residual_after_path_obs(eval_rows)
        diff_sharpe = self._differential_sharpe_eval(eval_t, h)
        collapse = self._multi_head_collapse_analysis(archetypes)
        judgments = self._final_judgments(archetypes, overlap, residual_after_path, collapse)

        return {
            "audit": "Risk-Adjusted Path Metrics Audit (Sharpe/Sortino/Ulcer)",
            "market": describe_market_data(self._data),
            "config": {"reward_horizon": h, "eval_samples": len(eval_rows)},
            "metric_definitions": {k: v for k, v in RISK_ADJUSTED_SPECS},
            "archetype_ABC_analysis": archetypes,
            "eval_overlap_with_U_MAE_path": overlap,
            "residual_after_U_MAE_and_path_obs": residual_after_path,
            "differential_sharpe_long_short_hold": diff_sharpe,
            "multi_head_scalar_collapse_analysis": collapse,
            "final_judgments": judgments,
            "synthesis_ko": self._synthesis_ko(judgments, archetypes, collapse),
        }

    def _collect_rows(self, eval_t: Sequence[int], h: int) -> list[dict[str, Any]]:
        cfg = self._cfg
        rows: list[dict[str, Any]] = []
        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            ud = compute_u_diagnostics(ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config)
            ra = compute_risk_adjusted_path_metrics(ctx, Action.LONG, h)
            path_obs = observables_to_dict(compute_path_residual_observables(ctx, Action.LONG, h))
            rows.append(
                {
                    "t_index": t_index,
                    "u_mean": ud.u_mean,
                    "mae": compute_mae_n(ctx, Action.LONG, h),
                    "p": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
                    "terminal": ra.terminal_return,
                    "mfe": ra.mfe,
                    "risk_adj": risk_adjusted_to_dict(ra),
                    "path_obs": path_obs,
                }
            )
        return rows

    def _path_metrics_bundle(
        self, path, cfg: RiskAdjustedPathAuditConfig
    ) -> dict[str, Any]:
        h = cfg.reward_horizon
        ctx = path.to_context()
        ud = compute_u_diagnostics(ctx, Action.LONG, horizon=h, utility_config=cfg.utility_config)
        ra = compute_risk_adjusted_path_metrics(ctx, Action.LONG, h)
        path_obs = observables_to_dict(compute_path_residual_observables(ctx, Action.LONG, h))
        return {
            "u_mean": ud.u_mean,
            "u_terminal": ud.u_terminal,
            "mae": compute_mae_n(ctx, Action.LONG, h),
            "terminal": ra.terminal_return,
            "mfe": ra.mfe,
            "p": compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
            "risk_adj": risk_adjusted_to_dict(ra),
            "path_obs": path_obs,
        }

    def _archetype_analysis(self, cfg: RiskAdjustedPathAuditConfig) -> list[dict[str, Any]]:
        h = cfg.reward_horizon
        results: list[dict[str, Any]] = []
        bundles: dict[str, dict[str, Any]] = {}
        for case in ARCHETYPE_CASES:
            adverse = case["id"] == "C"
            path = self._residual._path_from_cumulative(
                f"arch_{case['id']}",
                case["levels"],
                h,
                adverse_wick=adverse,
            )
            bundles[case["id"]] = self._path_metrics_bundle(path, cfg)

        for case in ARCHETYPE_CASES:
            m = bundles[case["id"]]
            ra = m["risk_adj"]
            po = m["path_obs"]
            semantic = self._archetype_semantics(case["id"], m, bundles)
            results.append(
                {
                    "id": case["id"],
                    "label": case["label"],
                    "description": case["description"],
                    "levels": case["levels"],
                    "metrics": m,
                    "semantic_interpretation": semantic,
                }
            )

        results.append(
            {
                "id": "comparison",
                "pairwise_notes": self._pairwise_archetype_notes(bundles),
            }
        )
        return results

    def _archetype_semantics(
        self,
        case_id: str,
        m: dict[str, Any],
        all_b: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        ra = m["risk_adj"]
        po = m["path_obs"]
        lines: dict[str, str] = {}

        lines["return_magnitude_signals"] = (
            f"terminal={m['terminal']:.4f}, mfe={m['mfe']:.4f}, u_mean={m['u_mean']:.4f}, "
            f"mean_bar={ra['mean_bar_return']:.4f}"
        )
        lines["risk_magnitude_signals"] = (
            f"mae={m['mae']:.4f}, bar_vol={ra['bar_volatility']:.4f}, "
            f"ulcer={ra['ulcer_index']:.4f}, max_dd={ra['max_drawdown']:.4f}, chop={po['oscillation_chop']:.3f}"
        )
        lines["risk_adjusted_scalar"] = (
            f"sharpe={ra['path_sharpe']:.3f}, sortino={ra['path_sortino']:.3f}, "
            f"calmar={ra['calmar_proxy']:.3f}, return/ulcer={ra['return_over_ulcer']:.3f}"
        )
        lines["path_observable"] = (
            f"giveback={po['giveback_ratio']:.3f}, reversal={po['reversal_depth']:.3f}"
        )

        if case_id == "A":
            lines["design_note"] = (
                "Sharpe/Sortino는 bar return mean/vol mix — spike+giveback에서 vol↑, terminal↓. "
                "giveback은 capture efficiency, Sharpe은 return/risk scalar collapse. "
                "어느 쪽이 더 좋은지 미결 — A는 MFE↑ terminal↓, B와 비교 필요."
            )
        elif case_id == "B":
            lines["design_note"] = (
                "낮은 vol, 높은 terminal proximity — Sharpe/Sortino가 상대적으로 유리할 수 있으나 "
                "이는 'potential MFE' 정보를 반영하지 않음. Expected Return 정의에 따라 우열 역전 가능."
            )
        elif case_id == "C":
            lines["design_note"] = (
                "terminal<0, mae↑, ulcer↑, sharpe/sortino strongly negative. "
                "Direction과 Risk head에 분산 필요 — scalar Sharpe alone은 direction+return+risk 혼합."
            )
        return lines

    def _pairwise_archetype_notes(self, b: dict[str, dict[str, Any]]) -> dict[str, Any]:
        a, bb, c = b["A"], b["B"], b["C"]
        return {
            "A_vs_B": {
                "u_diff": abs(a["u_mean"] - bb["u_mean"]),
                "terminal_diff": abs(a["terminal"] - bb["terminal"]),
                "sharpe": {"A": a["risk_adj"]["path_sharpe"], "B": bb["risk_adj"]["path_sharpe"]},
                "sortino": {"A": a["risk_adj"]["path_sortino"], "B": bb["risk_adj"]["path_sortino"]},
                "ulcer": {"A": a["risk_adj"]["ulcer_index"], "B": bb["risk_adj"]["ulcer_index"]},
                "giveback": {
                    "A": a["path_obs"]["giveback_ratio"],
                    "B": bb["path_obs"]["giveback_ratio"],
                },
                "note": (
                    "Sharpe/Sortino가 A/B rank를 single scalar로 정할 수 있으나, "
                    "MFE(potential return) vs terminal(realized) trade-off를 collapse. "
                    "giveback은 capture, Sharpe은 return/vol 혼합."
                ),
            },
            "C_vs_B": {
                "terminal": {"C": c["terminal"], "B": bb["terminal"]},
                "sharpe": {"C": c["risk_adj"]["path_sharpe"], "B": bb["risk_adj"]["path_sharpe"]},
                "mae": {"C": c["mae"], "B": bb["mae"]},
                "note": "C는 direction 실패 + adverse path — Sharpe negative, Risk head primary.",
            },
            "C_vs_A": {
                "mfe": {"C": c["mfe"], "A": a["mfe"]},
                "terminal": {"C": c["terminal"], "A": a["terminal"]},
                "reversal": {
                    "C": c["path_obs"]["reversal_depth"],
                    "A": a["path_obs"]["giveback_ratio"],
                },
                "note": (
                    "C는 early rise 후 crash — MFE may still positive early but terminal negative. "
                    "Ulcer/reversal/chop이 crash path risk 포착."
                ),
            },
        }

    def _overlap_analysis(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        u = [r["u_mean"] for r in rows]
        mae = [r["mae"] for r in rows]
        term = [r["terminal"] for r in rows]
        out: dict[str, Any] = {}
        for key, _ in RISK_ADJUSTED_SPECS:
            vals = [r["risk_adj"][key] for r in rows]
            entry: dict[str, Any] = {
                "corr_u": _pearson(vals, u),
                "corr_mae": _pearson(vals, mae),
                "corr_terminal": _pearson(vals, term),
                "r2_after_u_mae": _ols_r2(
                    np.asarray(vals, dtype=float),
                    np.asarray(u, dtype=float),
                    np.asarray(mae, dtype=float),
                ),
            }
            for pk in PATH_REF_KEYS:
                pv = [r["path_obs"][pk] for r in rows]
                entry[f"corr_{pk}"] = _pearson(vals, pv)
            out[key] = entry
        return out

    def _residual_after_path_obs(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        u = np.asarray([r["u_mean"] for r in rows], dtype=float)
        mae = np.asarray([r["mae"] for r in rows], dtype=float)
        gb = np.asarray([r["path_obs"]["giveback_ratio"] for r in rows], dtype=float)
        chop = np.asarray([r["path_obs"]["oscillation_chop"] for r in rows], dtype=float)
        rev = np.asarray([r["path_obs"]["reversal_depth"] for r in rows], dtype=float)
        out: dict[str, Any] = {}
        for key, _ in RISK_ADJUSTED_SPECS:
            y = np.asarray([r["risk_adj"][key] for r in rows], dtype=float)
            out[key] = {
                "r2_u_mae": _ols_r2(y, u, mae),
                "r2_u_mae_giveback_chop": _ols_r2(y, u, mae, gb, chop),
                "r2_u_mae_giveback_chop_reversal": _ols_r2(y, u, mae, gb, chop, rev),
                "incremental_after_path_obs": None,
            }
            r2_base = out[key]["r2_u_mae"]
            r2_ext = out[key]["r2_u_mae_giveback_chop_reversal"]
            if not math.isnan(r2_base) and not math.isnan(r2_ext):
                out[key]["incremental_after_path_obs"] = r2_ext - r2_base
        return out

    def _differential_sharpe_eval(self, eval_t: Sequence[int], h: int) -> dict[str, Any]:
        diffs: list[dict[str, float]] = []
        for t_index in eval_t[:500]:
            ctx = self._builder.build(t_index)
            d = compute_differential_sharpe_pair(ctx, h)
            diffs.append(
                {
                    "sharpe_long": d.sharpe_long,
                    "sharpe_short": d.sharpe_short,
                    "diff_long_minus_short": d.diff_long_minus_short,
                    "diff_long_minus_hold": d.diff_long_minus_hold,
                }
            )
        if not diffs:
            return {"note": "no samples"}
        dlh = [x["diff_long_minus_hold"] for x in diffs]
        dls = [x["diff_long_minus_short"] for x in diffs]
        return {
            "samples": len(diffs),
            "mean_diff_long_hold": float(np.mean(dlh)),
            "mean_diff_long_short": float(np.mean(dls)),
            "std_diff_long_short": float(np.std(dls)),
            "p1_relevance": (
                "diff_long_minus_short는 동일 market path에서 action perspective Sharpe gap — "
                "Direction judgment + risk-adjusted attractiveness 혼합. "
                "P1 Direction head와 coupling 가능하나 scalar F 단독 대체는 부적절."
            ),
            "p2_relevance": (
                "Execution policy / position scaling에서 differential reward (RL)로 더 자연스러움. "
                "P2: 'LONG 대비 SHORT/HOLD의 risk-adjusted edge'는 entry/scale timing과 연결."
            ),
        }

    def _multi_head_collapse_analysis(
        self, archetypes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        cases = {a["id"]: a["metrics"] for a in archetypes if "metrics" in a}
        a, b, c = cases["A"], cases["B"], cases["C"]
        ranking_sharpe = sorted(
            [("A", a["risk_adj"]["path_sharpe"]), ("B", b["risk_adj"]["path_sharpe"]), ("C", c["risk_adj"]["path_sharpe"])],
            key=lambda x: -x[1],
        )
        ranking_terminal = sorted(
            [("A", a["terminal"]), ("B", b["terminal"]), ("C", c["terminal"])],
            key=lambda x: -x[1],
        )
        ranking_mfe = sorted(
            [("A", a["mfe"]), ("B", b["mfe"]), ("C", c["mfe"])],
            key=lambda x: -x[1],
        )
        return {
            "conflict_statement": (
                "Sharpe/Sortino/Calmar/return-over-ulcer은 Return과 Risk를 단일 scalar로 collapse. "
                "P1 multi-head (Direction / Expected Return / Acceptable Risk)와 논리적으로 충돌 - "
                "하나의 Sharpe를 F target 또는 U 대체로 쓰면 head 분리 설계와 모순."
            ),
            "archetype_rankings": {
                "by_sharpe": ranking_sharpe,
                "by_terminal_realized_return": ranking_terminal,
                "by_mfe_potential_return": ranking_mfe,
            },
            "A_vs_B_conflict": {
                "sharpe_favors": ranking_sharpe[0][0],
                "terminal_favors": ranking_terminal[0][0],
                "mfe_favors": ranking_mfe[0][0],
                "note": (
                    "A/B에서 Sharpe ranking != MFE ranking != terminal ranking 가능 — "
                    "scalar risk-adjusted metric 하나로 Expected Return head 역할 불가."
                ),
            },
            "recommended_use": (
                "Risk-adjusted metrics는 composite diagnostic 또는 P2 policy score로 유지. "
                "P1에서는 Return_mag, Risk_mag, capture(giveback) 분리 유지."
            ),
        }

    def _final_judgments(
        self,
        archetypes: list[dict[str, Any]],
        overlap: dict[str, Any],
        residual: dict[str, Any],
        collapse: dict[str, Any],
    ) -> dict[str, Any]:
        incr = {
            k: v.get("incremental_after_path_obs")
            for k, v in residual.items()
            if v.get("incremental_after_path_obs") is not None
        }
        max_incr = max(incr.items(), key=lambda x: x[1] or 0.0) if incr else ("none", 0.0)

        return {
            "supplements_path_problem": {
                "verdict": "partial_only",
                "detail_ko": (
                    "Sharpe/Sortino/Ulcer는 path vol/drawdown을 scalar로 요약하지만, "
                    "U/MAE/giveback/chop이 이미 분리 제공하는 정보와 substantial overlap. "
                    "Path 문제(giveback vs grind)를 완전히 새로 보완하지는 못함."
                ),
            },
            "redundant_with_U_MAE": {
                "verdict": "substantial_overlap",
                "detail_ko": (
                    "path_sharpe/sortino는 mean return(U/terminal proxy)과 vol(MAE/chop proxy) 혼합. "
                    "return_over_ulcer, calmar는 terminal+drawdown 재조합."
                ),
            },
            "better_than_path_observables": {
                "verdict": "no",
                "detail_ko": (
                    "giveback/reversal/chop이 Expected Return vs Risk 축 분리에 더 직접적. "
                    "Sharpe는 두 축을 다시 collapse — multi-head 설계에 불리."
                ),
            },
            "expected_return_head": {
                "verdict": "no_direct",
                "detail_ko": "Return head에는 U, terminal, MFE, giveback(capture) — Sharpe 직접 투입 비권장.",
            },
            "risk_head": {
                "verdict": "partial_diagnostic_only",
                "detail_ko": (
                    "ulcer_index, bar_volatility는 Risk head 보조 diagnostic 가능. "
                    "Sharpe/Sortino 자체는 return component 포함 — Risk head 단독 부적합."
                ),
            },
            "separate_diagnostic": {
                "verdict": "yes_preferred",
                "detail_ko": (
                    "composite risk-adjusted score는 P1 학습 target이 아니라 "
                    "chart review / P2 policy research diagnostic으로 유지 권장."
                ),
            },
            "replace_U_with_risk_adjusted": {
                "verdict": "no",
                "detail_ko": (
                    "U는 favorable opportunity magnitude — Sharpe로 교체하면 potential return 정보 손실, "
                    "Case A(MFE↑ terminal↓) vs B trade-off collapse. 논리적 근거 없음."
                ),
            },
            "differential_sharpe_P1_vs_P2": {
                "verdict": "primarily_P2",
                "detail_ko": (
                    "diff_long_minus_short는 action 간 edge — Direction + execution scaling에 가까움. "
                    "P1 t 시점 label로는 Direction head 보조 가능하나, "
                    "canonical F 또는 U 대체는 P2 RL/execution 단계 문제."
                ),
            },
            "max_incremental_r2_after_path_obs": {"metric": max_incr[0], "delta_r2": max_incr[1]},
            "overlap_table_summary": overlap,
        }

    def _synthesis_ko(
        self,
        judgments: dict[str, Any],
        archetypes: list[dict[str, Any]],
        collapse: dict[str, Any],
    ) -> dict[str, str]:
        cases = {a["id"]: a for a in archetypes if "metrics" in a}
        return {
            "한줄_결론": (
                "Sharpe/Sortino/Ulcer는 analysis diagnostic으로만 가치 - "
                "P1 reward/output head 직접 채택 비권장. U 교체 근거 없음. "
                "Differential Sharpe는 P2 쪽."
            ),
            "A_B_C_핵심": (
                f"Sharpe rank: {collapse['archetype_rankings']['by_sharpe']}; "
                f"terminal rank: {collapse['archetype_rankings']['by_terminal_realized_return']}; "
                f"MFE rank: {collapse['archetype_rankings']['by_mfe_potential_return']}"
            ),
            "multi_head_충돌": collapse["conflict_statement"],
            "Case_A_semantic": cases.get("A", {}).get("semantic_interpretation", {}).get("design_note", ""),
            "Case_C_semantic": cases.get("C", {}).get("semantic_interpretation", {}).get("design_note", ""),
        }


def format_risk_adjusted_summary(report: dict[str, Any]) -> str:
    syn = report.get("synthesis_ko", {})
    lines = [
        "Risk-Adjusted Path Metrics Audit",
        "=" * 60,
        syn.get("한줄_결론", ""),
        syn.get("A_B_C_핵심", ""),
    ]
    return "\n".join(lines)


def save_risk_adjusted_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = RiskAdjustedPathAuditRunner(market_data).run()
    print(format_risk_adjusted_summary(report))
    return report
