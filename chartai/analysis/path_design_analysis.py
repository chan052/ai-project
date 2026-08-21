"""P1 Path Design Analysis — role of path observables for Return/Risk (analysis-only)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.path_residual_diagnostics import (
    CANDIDATE_SPECS,
    ResidualCandidateSpec,
    compute_path_residual_observables,
    get_candidate_value,
    observables_to_dict,
)
from chartai.analysis.u_mae_residual_audit import (
    SYNTHETIC_CASES,
    UMaeResidualAuditConfig,
    UMaeResidualAuditRunner,
    _pearson,
)
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.mae import compute_mae_n
from chartai.reward.path import compute_path_n


class RedundancyClass(str, Enum):
    A_REDUNDANT = "A_U_MAE_redundant"
    B_PARTIAL = "B_partial_path_detail"
    C_INDEPENDENT = "C_mostly_independent"
    D_UNCLEAR = "D_unclear_or_low_relevance"


@dataclass(frozen=True)
class ObservableDefinition:
    key: str
    name_ko: str
    definition: str
    u_mae_already_contains: str
    blind_spot_addressed: str
    redundancy_class: str
    direction_relevance: str
    return_relevance: str
    risk_relevance: str
    reward_candidate: bool
    output_diagnostic_candidate: bool
    discard: bool
    notes: str = ""


OBSERVABLE_CATALOG: tuple[ObservableDefinition, ...] = (
    ObservableDefinition(
        "giveback_ratio",
        "Giveback (MFE 반납)",
        "(MFE - terminal) / MFE. Peak 대비 horizon 종료 시 favorable excursion 반납 비율.",
        "U는 favorable 크기와 decay-weight mix를 담음. terminal/MFE 비율은 U_mean에 암묵 포함될 수 있으나 giveback 자체는 분리되지 않음.",
        "U가 비슷한 spike-vs-grind path에서 '얼마나 반납했는가' 구분.",
        RedundancyClass.C_INDEPENDENT.value,
        "low",
        "high — realized capture vs potential (MFE 대비 terminal 유지)",
        "moderate — 큰 giveback은 intra-horizon risk 신호일 수 있으나 terminal이 회복하면 risk와 분리",
        False,
        True,
        False,
        "Expected Return = potential(MFE) vs realized(terminal) 분리 시 핵심.",
    ),
    ObservableDefinition(
        "reversal_depth",
        "Reversal Depth (peak 이후 역방향 깊이)",
        "(MFE - min_after_peak) / MFE. Peak 이후 최악 되돌림 깊이.",
        "MAE는 worst adverse excursion이지만 peak 이후 favorable zone 이탈 깊이와는 다름.",
        "terminal이 회복해도 peak 이후 깊은 되돌림 (round-trip) 포착.",
        RedundancyClass.C_INDEPENDENT.value,
        "low",
        "moderate — terminal giveback과 다를 때 return capture와 분리",
        "high — intra-horizon path risk, stop/whip 가능성",
        False,
        True,
        False,
        "Case 4 (0→2→0→2): reversal=1.0, giveback≈0.11.",
    ),
    ObservableDefinition(
        "excursion_stability",
        "Excursion Stability (favorable zone 안정성)",
        "favorable 진입 후 cumulative path variance의 역수 스케일.",
        "U persistence (occupancy, max_run)과 부분 중복 (Audit 5).",
        "fine-grained hold vs spike within favorable zone.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate — hold 품질, capture 안정성",
        "moderate — unstable excursion은 risk proxy",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "peak_timing",
        "Peak Timing (MFE 발생 시점)",
        "time_to_mfe / horizon. MFE가 horizon 내 언제 발생했는가.",
        "U decay weights, P timing, MAE profile timing과 부분 중복.",
        "동일 MFE/terminal에서 early vs late peak 구분.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate — conditional on MFE>0; deferred opportunity 잔여 가능성",
        "low-moderate — late peak는 아직 움직임 남음 vs early peak는 giveback 위험",
        False,
        True,
        False,
        "P2 wait/entry timing과 경계 모호 — P1 단독 scalar로 흡수 비권장.",
    ),
    ObservableDefinition(
        "peak_after_decay",
        "Peak-after Decay (peak 이후 평균 악화)",
        "Peak 이후 bar들의 평균 (MFE-v)/MFE.",
        "giveback(terminal), reversal(max min)와 삼각 관계.",
        "peak 이후 평균 erosion — giveback과 reversal 사이.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate",
        "moderate",
        False,
        True,
        False,
        "giveback/reversal과 동시 reward 채택 금지 — 하나만.",
    ),
    ObservableDefinition(
        "recovery_shape_score",
        "Recovery Shape (adverse 이후 회복)",
        "terminal / MAE when MAE>0. 회복 크기 proxy.",
        "MAE scalar는 adverse magnitude; recovery는 Audit 5 blind spot.",
        "동일 MAE에서 recovery vs sustained adverse 구분.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate — deferred opportunity 성격",
        "high — path risk / 회복 가능성",
        False,
        True,
        False,
        "P2 wait policy와 겹침 — reward 직접 투입 비권장.",
    ),
    ObservableDefinition(
        "oscillation_chop",
        "Oscillation / Chop",
        "bar return sign change 비율.",
        "U/MAE/terminal과 거의 독립.",
        "동일 terminal/U/MAE에서 중간 경로 흔들림.",
        RedundancyClass.C_INDEPENDENT.value,
        "low",
        "low — return magnitude 자체보다 path quality",
        "high — execution/holding variance, acceptable risk",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "time_near_mfe",
        "Time near MFE",
        "cumulative return >= 0.9*MFE 인 bar 비율.",
        "U occupancy, persistence와 중복.",
        "MFE 수준 유지 시간 — grind vs spike.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate",
        "low",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "drawdown_from_mfe",
        "Drawdown from running peak",
        "running peak 대비 max drawdown / MFE.",
        "reversal_depth와 Case 4에서 유사; MAE와 다른 관점.",
        "peak 대비 intra-path drawdown.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "moderate",
        "high",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "path_efficiency",
        "Path efficiency",
        "terminal / sum(|bar returns|).",
        "terminal return proxy — U/terminal과 강한 중복.",
        "minimal beyond terminal.",
        RedundancyClass.A_REDUNDANT.value,
        "low",
        "redundant with terminal",
        "low",
        False,
        False,
        True,
    ),
    ObservableDefinition(
        "recovery_speed",
        "Recovery speed",
        "MAE bar 이후 favorable 복귀까지 bar 수 기반 (빠를수록 1에 가까움).",
        "MAE profile timing 일부 포함; recovery magnitude는 MAE blind spot.",
        "Case 3 fast vs slow recovery — peak_timing만 약하게 구분, speed는 보완.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "low-moderate",
        "moderate",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "time_under_water",
        "Time under water",
        "cumulative return < 0 인 bar 비율.",
        "MAE magnitude + adverse duration과 부분 중복.",
        "adverse regime 체류 시간.",
        RedundancyClass.B_PARTIAL.value,
        "low",
        "low",
        "moderate-high",
        False,
        True,
        False,
    ),
    ObservableDefinition(
        "path_sign_entropy",
        "Path sign entropy",
        "bar return 부호 분포 entropy.",
        "oscillation_chop과 높은 collinearity.",
        "chop과 동일 family.",
        RedundancyClass.A_REDUNDANT.value,
        "low",
        "low",
        "moderate",
        False,
        False,
        True,
        "oscillation_chop 하나만 유지.",
    ),
    ObservableDefinition(
        "directional_consistency",
        "Directional consistency",
        "1 - chop. chop과 완전 중복.",
        "chop과 동일.",
        "없음 (chop과 동일).",
        RedundancyClass.A_REDUNDANT.value,
        "low",
        "low",
        "moderate",
        False,
        False,
        True,
    ),
)


CASE1_LEVELS_A = [0, 1, 3, 1]
CASE1_LEVELS_B = [0, 2, 2, 2]


@dataclass
class PathDesignAnalysisConfig:
    reward_horizon: int = 10
    min_past_bars: int = 20
    eval_prefix_fraction: float = 0.5
    decay_rate: float = 0.75
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    u_mae_match_tol: float = 0.0005


class PathDesignAnalysisRunner:
    """Comprehensive path observable design analysis for P1 Return/Risk."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: PathDesignAnalysisConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or PathDesignAnalysisConfig()
        self._residual = UMaeResidualAuditRunner(
            market_data,
            config=UMaeResidualAuditConfig(
                reward_horizon=self._cfg.reward_horizon,
                min_past_bars=self._cfg.min_past_bars,
                eval_prefix_fraction=self._cfg.eval_prefix_fraction,
                decay_rate=self._cfg.decay_rate,
                utility_config=self._cfg.utility_config,
                u_mae_match_tol=self._cfg.u_mae_match_tol,
            ),
        )
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        base = self._residual.run()
        cfg = self._cfg
        t_indices = list(
            self._data.valid_t_indices(
                reward_horizon=cfg.reward_horizon,
                min_past_bars=cfg.min_past_bars,
            )
        )
        split = max(1, int(len(t_indices) * cfg.eval_prefix_fraction))
        eval_rows = self._residual._collect_eval_rows(
            t_indices[split:], cfg.reward_horizon
        )

        inter_corr = self._inter_observable_correlation(eval_rows)
        collinearity_groups = self._collinearity_groups(inter_corr)
        case1 = self._case1_deep_dive(cfg)
        archetypes = base.get("synthetic_archetype_pairs", [])
        q_answers = self._answer_core_questions(base, eval_rows, inter_corr, case1)
        structures = self._structure_candidates()
        conclusion = self._final_conclusion(q_answers, structures)

        report: dict[str, Any] = {
            "audit": "P1 Path Design Analysis — Expected Return / Acceptable Risk",
            "market": describe_market_data(self._data),
            "config": base.get("config", {}),
            "1_observable_catalog": [self._catalog_entry(o) for o in OBSERVABLE_CATALOG],
            "2_definitions": {o.key: o.definition for o in OBSERVABLE_CATALOG},
            "3_u_mae_already_contains": {o.key: o.u_mae_already_contains for o in OBSERVABLE_CATALOG},
            "4_u_mae_blind_spots": self._blind_spot_summary(),
            "5_blind_spot_coverage": {o.key: o.blind_spot_addressed for o in OBSERVABLE_CATALOG},
            "6_redundancy_vs_u_mae": base.get("double_counting_table", []),
            "7_inter_observable_collinearity": {
                "high_corr_pairs": collinearity_groups,
                "full_matrix_keys": [s.key for s in CANDIDATE_SPECS],
            },
            "8_p1_output_mapping": {
                o.key: {
                    "direction": o.direction_relevance,
                    "expected_return": o.return_relevance,
                    "acceptable_risk": o.risk_relevance,
                    "redundancy_class": o.redundancy_class,
                }
                for o in OBSERVABLE_CATALOG
            },
            "9_archetype_cases": archetypes,
            "10_case1_spike_vs_grind": case1,
            "11_reward_direct_candidates": [
                o.key for o in OBSERVABLE_CATALOG if o.reward_candidate
            ],
            "12_output_diagnostic_candidates": [
                o.key for o in OBSERVABLE_CATALOG if o.output_diagnostic_candidate and not o.discard
            ],
            "13_discard_candidates": [o.key for o in OBSERVABLE_CATALOG if o.discard],
            "14_recommended_structures": structures,
            "15_structure_pros_cons": self._structure_pros_cons(structures),
            "16_unresolved_human_decisions": self._unresolved_questions(case1),
            "core_questions_Q1_Q7": q_answers,
            "path_role_definition": self._path_role_definition(),
            "final_conclusion_category": conclusion,
            "residual_audit_crossref": {
                "CONFIRMED": base.get("CONFIRMED", []),
                "RESIDUAL_PATH_CANDIDATES": base.get("RESIDUAL_PATH_CANDIDATES", []),
            },
        }
        report["synthesis_ko"] = self._synthesis_ko(report)
        return report

    def _catalog_entry(self, o: ObservableDefinition) -> dict[str, Any]:
        return {
            "key": o.key,
            "name_ko": o.name_ko,
            "definition": o.definition,
            "redundancy_class": o.redundancy_class,
            "reward_candidate": o.reward_candidate,
            "output_diagnostic": o.output_diagnostic_candidate,
            "discard": o.discard,
        }

    def _blind_spot_summary(self) -> dict[str, str]:
        return {
            "U_blind_spot": (
                "동일/유사 U에서 path shape 차이 (spike-giveback vs grind-hold, "
                "capture efficiency vs MFE potential)"
            ),
            "MAE_blind_spot": (
                "동일 MAE에서 recovery shape, adverse 이후 회복 vs sustained pain, "
                "round-trip intra-horizon risk invisible to terminal"
            ),
            "P_blind_spot": (
                "magnitude overlap with U; timing overlap with peak_timing/occupancy"
            ),
            "shared_blind_spot": (
                "U+MAE가 '얼마나 유리/불리'는 담지만 '어떤 경로로' 발생했는지 세밀 구분 부족"
            ),
        }

    def _inter_observable_correlation(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        keys = [s.key for s in CANDIDATE_SPECS]
        matrix: dict[str, dict[str, float]] = {}
        for k1 in keys:
            matrix[k1] = {}
            v1 = [r["candidates"][k1] for r in rows]
            for k2 in keys:
                v2 = [r["candidates"][k2] for r in rows]
                matrix[k1][k2] = _pearson(v1, v2)
        return matrix

    def _collinearity_groups(
        self, matrix: dict[str, dict[str, float]], threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        keys = [s.key for s in CANDIDATE_SPECS]
        pairs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                if (k1, k2) in seen:
                    continue
                c = matrix[k1].get(k2, float("nan"))
                if not math.isnan(c) and abs(c) >= threshold:
                    pairs.append({"a": k1, "b": k2, "corr": c})
                    seen.add((k1, k2))
        return sorted(pairs, key=lambda x: -abs(x["corr"]))

    def _case1_deep_dive(self, cfg: PathDesignAnalysisConfig) -> dict[str, Any]:
        h = cfg.reward_horizon
        pa = self._residual._path_from_cumulative("case1_a", CASE1_LEVELS_A, h)
        pb = self._residual._path_from_cumulative("case1_b", CASE1_LEVELS_B, h)
        ma = self._residual._path_metrics(pa, self._residual._cfg)
        mb = self._residual._path_metrics(pb, self._residual._cfg)

        u_ratio = ma["u_mean"] / mb["u_mean"] if mb["u_mean"] else float("inf")
        design_question = {
            "path_A": "0→1→3→1 (spike then giveback)",
            "path_B": "0→2→2→2 (grind hold)",
            "metrics": {"A": ma, "B": mb},
            "u_similar": abs(ma["u_mean"] - mb["u_mean"]) < 0.01,
            "mfe_A_higher": ma["mfe"] > mb["mfe"],
            "terminal_A_lower": ma["terminal"] < mb["terminal"],
            "giveback_A_much_higher": ma["candidates"]["giveback_ratio"]
            > mb["candidates"]["giveback_ratio"] + 0.3,
            "core_design_question": (
                "P1이 단일 scalar로 A/B 우열을 정해야 하는가? "
                "Expected Return(potential=MFE vs realized=terminal)과 "
                "Acceptable Risk(giveback/chop/reversal)을 분리하면 "
                "A는 높은 potential+낮은 capture, B는 낮은 potential+높은 capture로 "
                "동시 표현 가능 — 우열은 output semantics에 달림."
            ),
            "dimension_mapping": {
                "direction": "동일 (LONG favorable)",
                "expected_return_if_potential": "A 우위 (MFE 3.2% vs 2.2%)",
                "expected_return_if_realized": "B 우위 (terminal 2.0% vs 1.0%)",
                "expected_return_if_capture_efficiency": "B 우위 (giveback 0 vs 0.68)",
                "acceptable_risk": "A: giveback/reversal 높음; B: chop 낮음, hold 안정",
                "single_scalar_F": "A/B trade-off를 한 숫자로 collapse하면 정보 손실",
            },
            "recommended_expression": (
                "multi-head: Return_mag ≈ f(U, terminal, MFE); "
                "Risk_mag ≈ f(MAE, chop, reversal); "
                "capture_efficiency ≈ f(giveback, terminal/MFE) as diagnostic"
            ),
        }
        return design_question

    def _answer_core_questions(
        self,
        base: dict[str, Any],
        eval_rows: list[dict[str, Any]],
        inter_corr: dict[str, dict[str, float]],
        case1: dict[str, Any],
    ) -> dict[str, Any]:
        bucket = base.get("same_u_mae_bucket_discrimination", {})
        archetypes = base.get("synthetic_archetype_pairs", [])

        q1_cases = []
        for c in archetypes:
            if c.get("u_diff", 999) < 0.01 or c.get("mae_diff", 999) < 0.003:
                q1_cases.append(
                    {
                        "case_id": c["case_id"],
                        "label": c["label"],
                        "u_diff": c["u_diff"],
                        "mae_diff": c["mae_diff"],
                        "top_separators": c.get("top_discriminating_candidates", [])[:3],
                    }
                )

        q2_same_terminal = [
            {
                "case_id": c["case_id"],
                "label": c["label"],
                "terminal_diff": c.get("terminal_diff"),
                "u_diff": c.get("u_diff"),
                "mae_diff": c.get("mae_diff"),
                "top_separators": c.get("top_discriminating_candidates", [])[:3],
            }
            for c in archetypes
            if c.get("terminal_diff", 999) < 0.005
        ]

        q3 = {o.key: o.redundancy_class for o in OBSERVABLE_CATALOG}

        q4 = {
            o.key: {
                "direction": o.direction_relevance,
                "expected_return": o.return_relevance,
                "acceptable_risk": o.risk_relevance,
            }
            for o in OBSERVABLE_CATALOG
        }

        q5 = {
            "baseline": "F = P + U - MAE (canonical, unchanged)",
            "candidates": [
                {
                    "name": "A_multi_head_no_path_in_reward",
                    "formula": "F_baseline + P1 heads: Return(U,terminal,MFE), Risk(MAE,chop,reversal), Capture(giveback)",
                    "path_in_reward": False,
                },
                {
                    "name": "B_conditional_path_penalty",
                    "formula": "F = P + U - MAE - g(giveback|U,MFE) - h(chop|MAE)",
                    "path_in_reward": True,
                    "note": "U/MAE conditioning 필수 — double counting 방지",
                },
                {
                    "name": "C_split_F_return_F_risk",
                    "formula": "F_return = U + capture_diag; F_risk = MAE + path_risk_diag",
                    "path_in_reward": "partial",
                },
            ],
            "preferred_separation": (
                "Path를 reward scalar에 직접 더하기보다 P1 output/diagnostic head로 분리하면 "
                "Return vs Risk 역할이 명확해짐 (Case 1 설계 질문 해소에 유리)."
            ),
        }

        q6_groups = self._collinearity_groups(inter_corr, threshold=0.75)
        q6_notes = [
            "giveback ≈ terminal_proximity (1-giveback) when MFE>0 — 하나만",
            "reversal_depth vs drawdown_from_mfe: Case 4에서 유사, monotone path에서 giveback과 분리",
            "peak_after_decay: giveback과 reversal 사이 — 3개 중 1개",
            "oscillation_chop = 1 - directional_consistency; transition_count = chop * scale",
            "path_sign_entropy ≈ chop family",
        ]

        q7 = {
            "verdict_options": [
                "1_Path_in_reward",
                "2_Path_as_P1_output_diagnostic",
                "3_Path_unnecessary",
            ],
            "selected": "2_Path_as_P1_output_diagnostic",
            "rationale_ko": (
                "U/MAE만으로 magnitude는 충분하나 path shape 정보는 존재하고 P1 Return/Risk "
                "분리 표현에 의미 있음. 다만 reward scalar에 직접 추가하면 U/MAE와 "
                "double counting 위험 — output/diagnostic이 역할 분리에 유리."
            ),
        }

        return {
            "Q1_u_mae_similar_path_diff": q1_cases,
            "Q2_same_terminal_u_mae": q2_same_terminal,
            "Q3_redundancy_classification": q3,
            "Q4_p1_output_mapping": q4,
            "Q5_composition_candidates": q5,
            "Q6_inter_observable_collinearity": {"pairs": q6_groups, "notes": q6_notes},
            "Q7_is_path_needed": q7,
            "real_data_u_mae_buckets": bucket,
            "case1_design": case1.get("core_design_question"),
        }

    def _structure_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "S1_multi_head_diagnostic",
                "name_ko": "Multi-head P1 + Path는 diagnostic/target 보조",
                "sketch": {
                    "direction": "P 또는 별도 head",
                    "expected_return": "U + terminal + MFE (potential/realized 분리 가능)",
                    "acceptable_risk": "MAE + chop + reversal_depth",
                    "path_diagnostic": "giveback, peak_timing, recovery_shape",
                    "reward": "F = P + U - MAE (변경 없음)",
                },
                "path_role": "U/MAE residual path structure를 P1 output 학습 target/diagnostic으로",
            },
            {
                "id": "S2_conditional_reward_adjustment",
                "name_ko": "Conditional path adjustment (연구용, 비채택)",
                "sketch": {
                    "reward": "F = P + U - MAE - penalty(giveback | MFE>τ) - penalty(chop | MAE>τ)",
                    "conditioning": "반드시 U/MAE/MFE 조건부",
                },
                "path_role": "reward에 넣되 nonlinear conditional — canonical 채택 전 추가 audit 필요",
            },
            {
                "id": "S3_minimal_no_path",
                "name_ko": "Minimal — U/MAE/P만, path 없음",
                "sketch": {
                    "reward": "F = P + U - MAE",
                    "p1_output": "direction + U_proxy(return) + MAE_proxy(risk)",
                },
                "path_role": "path shape 무시 — Case 1 A/B collapse, multi-head 없으면 정보 손실",
            },
        ]

    def _structure_pros_cons(self, structures: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "S1_multi_head_diagnostic": {
                "pros": [
                    "Case 1 spike vs grind을 Return/Risk/capture로 분리 표현",
                    "reward canonical 유지, double counting 최소",
                    "P1 output semantics와 정렬",
                ],
                "cons": [
                    "학습 target 설계 복잡도 증가",
                    "output head 간 상관 관리 필요",
                ],
            },
            "S2_conditional_reward_adjustment": {
                "pros": ["단일 F scalar 유지 가능", "giveback/chop을 risk-adjusted F에 반영"],
                "cons": [
                    "U/MAE overlap 재발 위험",
                    "weight/threshold 확정 전 채택 불가",
                    "Case 1 우열을 F 하나로 여전히 collapse",
                ],
            },
            "S3_minimal_no_path": {
                "pros": ["단순", "기존 audit과 일관"],
                "cons": [
                    "path shape blind spot 잔존",
                    "Expected Return vs capture efficiency 구분 불가",
                ],
            },
        }

    def _unresolved_questions(self, case1: dict[str, Any]) -> list[str]:
        return [
            "Expected Return Magnitude 정의: MFE(potential) vs terminal(realized) vs U-aggregate 중 무엇인가?",
            "Acceptable Risk에 giveback(reversal)을 포함할 것인가, MAE만으로 충분한가?",
            "Case 1 (0→1→3→1 vs 0→2→2→2): chart qual — 트레이더가 t에서 어느 쪽을 더 attractive하게 보는가?",
            "peak_timing을 P1 risk에 넣을지, P2 wait/deferred opportunity에 넣을지",
            "recovery_shape를 P1 risk head vs P2 entry timing으로 분리하는 product boundary",
            "giveback / reversal / peak_after_decay 셋 중 diagnostic으로 어떤 하나를 primary로 둘지",
            "multi-asset에서 residual path signal robustness (현재 BTC only)",
        ]

    def _path_role_definition(self) -> dict[str, str]:
        return {
            "what_path_is_NOT": (
                "speed/persistence 재측정, S+D 분해 목적, correlation winner 선택, "
                "Enter/Wait/Avoid 직접 학습"
            ),
            "what_path_IS": (
                "U/MAE가 동일하게 평가하는 서로 다른 future path를 구분하고, "
                "그 차이가 P1 Expected Return Magnitude vs Acceptable Risk Magnitude "
                "판단에 논리적으로 기여할 때만 의미 있는 residual path structure"
            ),
            "preferred_locus": (
                "현 단계: P1 output/diagnostic head 또는 학습 보조 target. "
                "canonical reward scalar 직접 수정은 보류."
            ),
        }

    def _final_conclusion(self, q_answers: dict[str, Any], structures: list) -> str:
        return q_answers["Q7_is_path_needed"]["selected"]

    def _synthesis_ko(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "path_역할_정의": report["path_role_definition"]["what_path_IS"],
            "결론_카테고리": "2 - Path 정보는 필요, reward component보다 P1 output/diagnostic 분리 권장",
            "reward_직접_투입_후보": report["11_reward_direct_candidates"],
            "output_diagnostic_후보": report["12_output_diagnostic_candidates"],
            "불필요_후보": report["13_discard_candidates"],
            "핵심_잔여_정보": [
                "giveback / capture efficiency → Expected Return (realized vs potential)",
                "reversal_depth / chop → Acceptable Risk (intra-horizon path risk)",
                "recovery_shape → MAE blind spot, P1 risk 또는 P2 boundary",
            ],
            "Case1_핵심": report["10_case1_spike_vs_grind"].get("core_design_question"),
            "권장_구조": "S1_multi_head_diagnostic",
        }


def format_path_design_summary(report: dict[str, Any]) -> str:
    syn = report.get("synthesis_ko", {})
    lines = [
        "P1 Path Design Analysis",
        "=" * 60,
        f"결론: {syn.get('결론_카테고리', '?')}",
        f"권장: {syn.get('권장_구조', '?')}",
        f"diagnostic 후보: {syn.get('output_diagnostic_후보', [])[:6]}...",
    ]
    return "\n".join(lines)


def save_path_design_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = PathDesignAnalysisRunner(market_data).run()
    print(format_path_design_summary(report))
    return report
