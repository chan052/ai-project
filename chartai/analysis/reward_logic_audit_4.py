"""Reward Logic Audit 4 — f_n time-profile and MAE decomposition (analysis-only).

Does not modify canonical P1 reward, target, or training code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.mae_diagnostics import MaeDiagnostics, compute_mae_diagnostics
from chartai.analysis.path_archetypes import classify_archetype, compute_extended_observables
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import FittedZScoreNormalizer
from chartai.reward.path import compute_path_n, normalized_decay_weights
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_speed_n,
)
from chartai.reward.synthetic import (
    SyntheticPath,
    SyntheticScenario,
    build_scenario,
    mae_adverse_long_path,
)
from chartai.reward.utility import compute_utility_n


class FnAggregation(str, Enum):
    MEAN_ALL = "A_mean_f1_f10"
    EARLY_ONLY = "B_early_only_f1_f3"
    MIDDLE_ONLY = "C_middle_only_f4_f7"
    LATE_ONLY = "D_late_only_f8_f10"
    INDIVIDUAL_PROFILE = "E_individual_fn_profile"


EARLY_NS = (1, 2, 3)
MIDDLE_NS = (4, 5, 6, 7)
LATE_NS = (8, 9, 10)
ALL_NS = tuple(range(1, 11))

FOCUS_ARCHETYPES = (
    "dip_then_rise",
    "rise_then_fall",
    "monotonic_rise",
    "monotonic_fall",
    "spike_reversal",
)


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _aggregate_fn(values: Sequence[float], ns: Sequence[int]) -> float:
    idx = [n - 1 for n in ns]
    picked = [values[i] for i in idx if i < len(values)]
    return float(np.mean(picked)) if picked else float("nan")


@dataclass
class Audit4Sample:
    t_index: int
    archetype: str
    fn_values: tuple[float, ...]
    p_raw: tuple[float, ...]
    u_raw: tuple[float, ...]
    mae_raw: tuple[float, ...]
    s_ttf: tuple[float, ...]
    d_occ: tuple[float, ...]
    early_mean_return: float
    late_mean_return: float
    terminal_return: float
    mae_diag: MaeDiagnostics


@dataclass
class RewardLogicAudit4Config:
    reward_horizon: int = 10
    decay_rate: float = 0.75
    min_past_bars: int = 20
    norm_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    early_mae_bars: int = 3
    mae_match_tol: float = 0.0003


class RewardLogicAudit4Runner:
    """Audit 4: f_n time-profile + MAE role/decomposition."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: RewardLogicAudit4Config | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or RewardLogicAudit4Config()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        t_indices = list(
            self._data.valid_t_indices(
                reward_horizon=h,
                min_past_bars=cfg.min_past_bars,
            )
        )
        if not t_indices:
            raise ValueError("No valid samples")

        raw_samples: list[dict[str, Any]] = []
        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            raw_samples.append(self._collect_raw(t_index, ctx))

        split_idx = max(1, int(len(t_indices) * cfg.norm_prefix_fraction))
        prefix = raw_samples[:split_idx]
        eval_raw = raw_samples[split_idx:]

        norm = self._fit_normalizer(prefix, h)
        eval_samples = [self._finalize_sample(r, norm, h) for r in eval_raw]

        time_audit = self._time_profile_audit(eval_samples, cfg)
        mae_audit = self._mae_decomposition_audit(eval_samples, cfg)
        synthetic_mae = self._synthetic_mae_paths(cfg)

        report: dict[str, Any] = {
            "audit": "Reward Logic Audit 4",
            "purpose": (
                "Audit f_n temporal aggregation and MAE scalar adequacy for P1 entry-attractiveness "
                "semantics — not predictive performance or canonical reward selection."
            ),
            "meta": {
                "market_data": describe_market_data(self._data),
                "num_valid_samples": len(t_indices),
                "eval_samples": len(eval_samples),
                "norm_prefix_fraction": cfg.norm_prefix_fraction,
                "baseline_fn": "f_n = norm(P_n) + norm(U_n) - norm(MAE_n); F = mean(f_1..f_H)",
                "primary_S": "time_to_favorable",
                "primary_D": "favorable_occupancy",
            },
            "1_fn_time_profile_audit": time_audit,
            "2_mae_role_decomposition_audit": mae_audit,
            "3_synthetic_mae_path_comparison": synthetic_mae,
        }

        conclusions = self._synthesize_conclusions(report)
        report.update(conclusions)
        return report

    def _collect_raw(self, t_index: int, ctx: RewardContext) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.reward_horizon
        ext = compute_extended_observables(ctx, Action.LONG, h)
        p, u, m, s, d = [], [], [], [], []
        for n in range(1, h + 1):
            p.append(compute_path_n(ctx, Action.LONG, n, decay_rate=cfg.decay_rate))
            u.append(compute_utility_n(ctx, Action.LONG, n, cfg.utility_config))
            m.append(compute_mae_n(ctx, Action.LONG, n))
            s.append(
                compute_speed_n(
                    ctx,
                    Action.LONG,
                    n,
                    SpeedCandidate.TIME_TO_FAVORABLE,
                    decay_rate=cfg.decay_rate,
                )
            )
            d.append(
                compute_persistence_n(
                    ctx,
                    Action.LONG,
                    n,
                    PersistenceCandidate.FAVORABLE_OCCUPANCY,
                    decay_rate=cfg.decay_rate,
                )
            )
        return {
            "t_index": t_index,
            "archetype": classify_archetype(ext),
            "p_raw": tuple(p),
            "u_raw": tuple(u),
            "mae_raw": tuple(m),
            "s_ttf": tuple(s),
            "d_occ": tuple(d),
            "early_mean_return": ext.base.early_mean_return,
            "late_mean_return": ext.base.late_mean_return,
            "terminal_return": ext.base.terminal_return,
            "mae_diag": compute_mae_diagnostics(
                ctx, Action.LONG, h, early_bars=cfg.early_mae_bars
            ),
        }

    def _fit_normalizer(
        self, prefix: list[dict[str, Any]], h: int
    ) -> FittedZScoreNormalizer:
        p_all: list[float] = []
        u_all: list[float] = []
        m_all: list[float] = []
        for rec in prefix:
            p_all.extend(rec["p_raw"])
            u_all.extend(rec["u_raw"])
            m_all.extend(rec["mae_raw"])
        return FittedZScoreNormalizer.fit(tuple(p_all), tuple(u_all), tuple(m_all))

    def _finalize_sample(
        self, raw: dict[str, Any], norm: FittedZScoreNormalizer, h: int
    ) -> Audit4Sample:
        fn = tuple(
            norm.normalize_path(raw["p_raw"][n - 1])
            + norm.normalize_utility(raw["u_raw"][n - 1])
            - norm.normalize_mae(raw["mae_raw"][n - 1])
            for n in range(1, h + 1)
        )
        return Audit4Sample(
            t_index=raw["t_index"],
            archetype=raw["archetype"],
            fn_values=fn,
            p_raw=raw["p_raw"],
            u_raw=raw["u_raw"],
            mae_raw=raw["mae_raw"],
            s_ttf=raw["s_ttf"],
            d_occ=raw["d_occ"],
            early_mean_return=raw["early_mean_return"],
            late_mean_return=raw["late_mean_return"],
            terminal_return=raw["terminal_return"],
            mae_diag=raw["mae_diag"],
        )

    def _time_profile_audit(
        self, eval_samples: list[Audit4Sample], cfg: RewardLogicAudit4Config
    ) -> dict[str, Any]:
        aggregations = {
            FnAggregation.MEAN_ALL: ALL_NS,
            FnAggregation.EARLY_ONLY: EARLY_NS,
            FnAggregation.MIDDLE_ONLY: MIDDLE_NS,
            FnAggregation.LATE_ONLY: LATE_NS,
        }

        def f_agg(s: Audit4Sample, ns: Sequence[int]) -> float:
            return _aggregate_fn(s.fn_values, ns)

        agg_means = {
            key.value: float(np.mean([f_agg(s, ns) for s in eval_samples]))
            for key, ns in aggregations.items()
        }

        profile = [
            float(np.mean([s.fn_values[n - 1] for s in eval_samples]))
            for n in ALL_NS
        ]

        by_arch: dict[str, Any] = {}
        for arch in FOCUS_ARCHETYPES:
            group = [s for s in eval_samples if s.archetype == arch]
            if not group:
                by_arch[arch] = {"count": 0}
                continue
            by_arch[arch] = {
                "count": len(group),
                "F_mean": float(np.mean([f_agg(s, ALL_NS) for s in group])),
                "F_early": float(np.mean([f_agg(s, EARLY_NS) for s in group])),
                "F_middle": float(np.mean([f_agg(s, MIDDLE_NS) for s in group])),
                "F_late": float(np.mean([f_agg(s, LATE_NS) for s in group])),
                "fn_profile_mean": [
                    float(np.mean([s.fn_values[n - 1] for s in group]))
                    for n in ALL_NS
                ],
                "mean_terminal_return": float(np.mean([s.terminal_return for s in group])),
                "mean_early_return": float(np.mean([s.early_mean_return for s in group])),
                "semantic_note": _archetype_fn_note(arch),
            }

        case_a = [s for s in eval_samples if s.archetype == "dip_then_rise"]
        case_b = [s for s in eval_samples if s.archetype == "rise_then_fall"]

        decay_weights = normalized_decay_weights(cfg.reward_horizon, cfg.decay_rate)
        p_early_mass = sum(decay_weights[k - 1] for k in EARLY_NS)
        p_late_mass = sum(decay_weights[k - 1] for k in LATE_NS)

        fn_vs_sd = {
            f"f{n}_vs_S": _pearson(
                [s.fn_values[n - 1] for s in eval_samples],
                [s.s_ttf[n - 1] for s in eval_samples],
            )
            for n in ALL_NS
        }
        fn_vs_sd.update(
            {
                f"f{n}_vs_D": _pearson(
                    [s.fn_values[n - 1] for s in eval_samples],
                    [s.d_occ[n - 1] for s in eval_samples],
                )
                for n in ALL_NS
            }
        )

        f_early_vals = [f_agg(s, EARLY_NS) for s in eval_samples]
        f_late_vals = [f_agg(s, LATE_NS) for s in eval_samples]

        return {
            "aggregation_comparison": {
                key.value: {
                    "horizons": list(ns),
                    "mean_F_long": agg_means[key.value],
                }
                for key, ns in aggregations.items()
            },
            "individual_fn_profile_eval_mean": {
                f"f_{n}": profile[n - 1] for n in ALL_NS
            },
            "semantic_questions": {
                "Q1_early_terminal_mixed_in_mean": {
                    "F_early_vs_F_late_corr": _pearson(f_early_vals, f_late_vals),
                    "F_mean_vs_F_early_corr": _pearson(
                        [f_agg(s, ALL_NS) for s in eval_samples], f_early_vals
                    ),
                    "F_mean_vs_terminal_return": _pearson(
                        [f_agg(s, ALL_NS) for s in eval_samples],
                        [s.terminal_return for s in eval_samples],
                    ),
                    "F_late_vs_terminal_return": _pearson(
                        f_late_vals, [s.terminal_return for s in eval_samples]
                    ),
                    "interpretation": (
                        "High F_mean~terminal and F_late~terminal with divergent F_early "
                        "indicates horizon mixing — late outcome dominates mean F."
                    ),
                },
                "Q2_late_U_MAE_overwrites_early": {
                    "case_a_F_early_vs_F_late_gap": (
                        float(
                            np.mean([f_agg(s, EARLY_NS) for s in case_a])
                            - np.mean([f_agg(s, LATE_NS) for s in case_a])
                        )
                        if case_a
                        else float("nan")
                    ),
                    "case_b_F_early_vs_F_late_gap": (
                        float(
                            np.mean([f_agg(s, EARLY_NS) for s in case_b])
                            - np.mean([f_agg(s, LATE_NS) for s in case_b])
                        )
                        if case_b
                        else float("nan")
                    ),
                    "note": "Case A: early adverse should depress F_early; late favorable may lift F_late.",
                },
                "Q3_early_vs_late_different_meaning": {
                    "early_aligns_early_return": _pearson(
                        f_early_vals, [s.early_mean_return for s in eval_samples]
                    ),
                    "late_aligns_terminal_return": _pearson(
                        f_late_vals, [s.terminal_return for s in eval_samples]
                    ),
                    "early_vs_late_rank_inversion_rate": self._rank_inversion_rate(
                        eval_samples, EARLY_NS, LATE_NS
                    ),
                },
                "Q4_mean_vs_timing_structure": {
                    "F_mean_vs_S_at_h10": _pearson(
                        [f_agg(s, ALL_NS) for s in eval_samples],
                        [s.s_ttf[-1] for s in eval_samples],
                    ),
                    "F_early_vs_S_at_h3": _pearson(
                        f_early_vals, [s.s_ttf[2] for s in eval_samples]
                    ),
                },
                "Q5_fn_profile_vs_SD": fn_vs_sd,
                "Q6_path_decay_implicit_immediate": {
                    "P_decay_weight_early_fraction": p_early_mass,
                    "P_decay_weight_late_fraction": p_late_mass,
                    "decay_rate": cfg.decay_rate,
                    "note": (
                        "Path P_n already weights early bars via w_k ∝ r^(k-1). "
                        "Mean(f_n) adds a second layer of horizon mixing on top."
                    ),
                },
            },
            "by_archetype": by_arch,
            "case_a_vs_b": {
                "case_a": _fn_case_stats(case_a),
                "case_b": _fn_case_stats(case_b),
            },
        }

    def _rank_inversion_rate(
        self,
        samples: list[Audit4Sample],
        early_ns: Sequence[int],
        late_ns: Sequence[int],
    ) -> float:
        inversions = 0
        pairs = 0
        for i in range(len(samples)):
            for j in range(i + 1, min(i + 50, len(samples))):
                fe_i = _aggregate_fn(samples[i].fn_values, early_ns)
                fe_j = _aggregate_fn(samples[j].fn_values, early_ns)
                fl_i = _aggregate_fn(samples[i].fn_values, late_ns)
                fl_j = _aggregate_fn(samples[j].fn_values, late_ns)
                if (fe_i - fe_j) * (fl_i - fl_j) < 0:
                    inversions += 1
                pairs += 1
        return inversions / pairs if pairs else float("nan")

    def _mae_decomposition_audit(
        self, eval_samples: list[Audit4Sample], cfg: RewardLogicAudit4Config
    ) -> dict[str, Any]:
        h = cfg.reward_horizon
        tol = cfg.mae_match_tol

        def diag_field(name: str) -> list[float]:
            if name == "full_mae":
                return [s.mae_diag.full_mae for s in eval_samples]
            if name == "early_mae":
                return [s.mae_diag.early_mae for s in eval_samples]
            if name == "time_to_mae":
                return [
                    float(s.mae_diag.time_to_mae or (h + 1))
                    for s in eval_samples
                ]
            if name == "adverse_duration":
                return [float(s.mae_diag.adverse_duration) for s in eval_samples]
            if name == "recovery_after_mae":
                return [s.mae_diag.recovery_after_mae for s in eval_samples]
            if name == "early_to_full_ratio":
                return [s.mae_diag.early_to_full_mae_ratio for s in eval_samples]
            raise KeyError(name)

        candidates = {
            "A_full_mae": "full_mae",
            "B_early_mae": "early_mae",
            "C_time_to_mae": "time_to_mae",
            "D_adverse_duration": "adverse_duration",
            "E_recovery_after_mae": "recovery_after_mae",
            "F_early_to_full_ratio": "early_to_full_ratio",
        }

        full = diag_field("full_mae")
        early = diag_field("early_mae")

        by_arch: dict[str, Any] = {}
        for arch in FOCUS_ARCHETYPES + ("choppy", "mixed", "slow_flat"):
            group = [s for s in eval_samples if s.archetype == arch]
            if not group:
                continue
            by_arch[arch] = {
                "count": len(group),
                "mean_full_mae": float(np.mean([s.mae_diag.full_mae for s in group])),
                "mean_early_mae": float(np.mean([s.mae_diag.early_mae for s in group])),
                "mean_recovery": float(
                    np.mean([s.mae_diag.recovery_after_mae for s in group])
                ),
                "mean_time_to_mae": float(
                    np.nanmean(
                        [
                            float(s.mae_diag.time_to_mae or (h + 1))
                            for s in group
                        ]
                    )
                ),
            }

        case_a = [s for s in eval_samples if s.archetype == "dip_then_rise"]
        sustained = [
            s
            for s in eval_samples
            if s.archetype in ("monotonic_fall", "rise_then_fall")
            and s.mae_diag.recovery_after_mae < 0.5
        ]

        similar_mae_pairs = self._similar_mae_pairs(eval_samples, tol)

        corr_with_s = {
            k: _pearson(diag_field(v), [s.s_ttf[-1] for s in eval_samples])
            for k, v in candidates.items()
        }
        corr_with_d = {
            k: _pearson(diag_field(v), [s.d_occ[-1] for s in eval_samples])
            for k, v in candidates.items()
        }

        return {
            "candidate_summary": {
                name: {
                    "field": field,
                    "mean": float(np.mean(diag_field(field))),
                    "std": float(np.std(diag_field(field))),
                }
                for name, field in candidates.items()
            },
            "semantic_questions": {
                "Q1_full_mae_conflates_entry_and_later": {
                    "full_vs_early_corr": _pearson(full, early),
                    "case_a_mean_full": float(np.mean([s.mae_diag.full_mae for s in case_a]))
                    if case_a
                    else float("nan"),
                    "case_a_mean_early": float(np.mean([s.mae_diag.early_mae for s in case_a]))
                    if case_a
                    else float("nan"),
                    "sustained_mean_full": float(
                        np.mean([s.mae_diag.full_mae for s in sustained])
                    )
                    if sustained
                    else float("nan"),
                    "similar_mae_different_archetype_pairs": len(similar_mae_pairs),
                },
                "Q2_early_mae_entry_risk": {
                    "early_mae_case_a_vs_b": {
                        "case_a": float(np.mean([s.mae_diag.early_mae for s in case_a]))
                        if case_a
                        else float("nan"),
                        "case_b": float(
                            np.mean(
                                [
                                    s.mae_diag.early_mae
                                    for s in eval_samples
                                    if s.archetype == "rise_then_fall"
                                ]
                            )
                        ),
                    },
                    "early_mae_vs_early_return": _pearson(
                        early, [s.early_mean_return for s in eval_samples]
                    ),
                },
                "Q3_time_to_mae_immediate_vs_delayed": {
                    "case_a_mean_time_to_mae": float(
                        np.nanmean(
                            [
                                float(s.mae_diag.time_to_mae or (h + 1))
                                for s in case_a
                            ]
                        )
                    )
                    if case_a
                    else float("nan"),
                    "monotonic_fall_mean_time_to_mae": float(
                        np.nanmean(
                            [
                                float(s.mae_diag.time_to_mae or (h + 1))
                                for s in eval_samples
                                if s.archetype == "monotonic_fall"
                            ]
                        )
                    ),
                },
                "Q4_recovery_reward_vs_diagnostic": {
                    "recovery_case_a_mean": float(
                        np.mean([s.mae_diag.recovery_after_mae for s in case_a])
                    )
                    if case_a
                    else float("nan"),
                    "recovery_sustained_mean": float(
                        np.mean([s.mae_diag.recovery_after_mae for s in sustained])
                    )
                    if sustained
                    else float("nan"),
                    "interpretation": (
                        "Recovery uses terminal outcome relative to MAE — overlaps deferred "
                        "opportunity / hindsight narrative. Treat as diagnostic unless explicitly "
                        "assigned to a separate P1 output head, not silent reward injection."
                    ),
                },
                "Q5_mae_vs_SD_overlap": {
                    "corr_with_S_at_h10": corr_with_s,
                    "corr_with_D_at_h10": corr_with_d,
                },
            },
            "by_archetype": by_arch,
            "similar_full_mae_different_meaning": similar_mae_pairs[:15],
            "case_a_vs_sustained_adverse": {
                "case_a": _mae_case_stats(case_a),
                "sustained_adverse_proxy": _mae_case_stats(sustained),
            },
        }

    def _similar_mae_pairs(
        self, samples: list[Audit4Sample], tol: float
    ) -> list[dict[str, Any]]:
        """Pairs with similar full MAE but different recovery / archetype."""
        out: list[dict[str, Any]] = []
        bins: dict[int, list[Audit4Sample]] = {}
        for s in samples:
            key = int(round(s.mae_diag.full_mae / tol))
            bins.setdefault(key, []).append(s)

        for group in bins.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if abs(a.mae_diag.full_mae - b.mae_diag.full_mae) > tol:
                        continue
                    if a.archetype == b.archetype:
                        continue
                    rec_diff = abs(
                        a.mae_diag.recovery_after_mae - b.mae_diag.recovery_after_mae
                    )
                    if rec_diff < 0.3:
                        continue
                    out.append(
                        {
                            "full_mae": a.mae_diag.full_mae,
                            "archetype_a": a.archetype,
                            "archetype_b": b.archetype,
                            "early_mae_a": a.mae_diag.early_mae,
                            "early_mae_b": b.mae_diag.early_mae,
                            "recovery_a": a.mae_diag.recovery_after_mae,
                            "recovery_b": b.mae_diag.recovery_after_mae,
                            "terminal_a": a.terminal_return,
                            "terminal_b": b.terminal_return,
                        }
                    )
                    if len(out) >= 50:
                        return out
        return out

    def _synthetic_mae_paths(self, cfg: RewardLogicAudit4Config) -> dict[str, Any]:
        h = cfg.reward_horizon
        scenarios: list[tuple[str, RewardContext]] = []

        def add(name: str, ctx: RewardContext) -> None:
            scenarios.append((name, ctx))

        add("dip_then_rise_case_a", build_scenario(SyntheticScenario.DOWN_THEN_UP, horizon=h).to_context())
        add("sustained_adverse", build_scenario(SyntheticScenario.STEADY_DOWN, horizon=h).to_context())
        add("rise_then_fall_case_b", build_scenario(SyntheticScenario.UP_THEN_DOWN, horizon=h).to_context())
        add(
            "spike_reversal",
            build_scenario(SyntheticScenario.QUIET_THEN_BIG_UP, horizon=h).to_context(),
        )
        add("mae_flash_recovery", mae_adverse_long_path(horizon=h).to_context())

        small_adv = self._custom_path(
            name="small_adverse_then_rise",
            moves=[-0.005, -0.005, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            h=h,
        )
        large_adv = self._custom_path(
            name="large_adverse_then_rise",
            moves=[-0.03, -0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            h=h,
        )
        late_pullback = self._custom_path(
            name="late_pullback",
            moves=[0.01] * 7 + [-0.02, -0.02, -0.02],
            h=h,
        )
        add("small_adverse_then_rise", small_adv.to_context())
        add("large_adverse_then_rise", large_adv.to_context())
        add("late_pullback", late_pullback.to_context())

        rows: dict[str, Any] = {}
        for name, ctx in scenarios:
            d = compute_mae_diagnostics(
                ctx, Action.LONG, h, early_bars=cfg.early_mae_bars
            )
            rows[name] = {
                "full_mae": d.full_mae,
                "early_mae": d.early_mae,
                "time_to_mae": d.time_to_mae,
                "adverse_duration": d.adverse_duration,
                "recovery_after_mae": d.recovery_after_mae,
                "early_to_full_ratio": d.early_to_full_mae_ratio,
                "terminal_return": d.terminal_aligned_return,
            }

        return {
            "paths": rows,
            "narrative_checks": {
                "case_a_vs_sustained": {
                    "case_a_recovery_gt_sustained": (
                        rows.get("dip_then_rise_case_a", {}).get("recovery_after_mae", 0)
                        > rows.get("sustained_adverse", {}).get("recovery_after_mae", 0)
                    ),
                    "similar_full_mae_possible": (
                        abs(
                            rows.get("small_adverse_then_rise", {}).get("full_mae", 0)
                            - rows.get("large_adverse_then_rise", {}).get("full_mae", 0)
                        )
                        < 0.02
                    ),
                    "early_mae_distinguishes_small_vs_large": (
                        rows.get("small_adverse_then_rise", {}).get("early_mae", 0)
                        < rows.get("large_adverse_then_rise", {}).get("early_mae", 0)
                    ),
                },
            },
        }

    def _custom_path(
        self, *, name: str, moves: list[float], h: int
    ) -> SyntheticPath:
        anchor = 100.0
        prices = [anchor]
        for m in moves:
            prices.append(prices[-1] * (1.0 + m))
        closes = tuple(prices[1:])
        # Ensure lows can pierce anchor on adverse legs (avoid negative MAE artifacts).
        lows_list: list[float] = []
        running = anchor
        for c in closes:
            running = min(running, c)
            lows_list.append(min(running, anchor) * 0.998)
        return SyntheticPath(
            name=name,
            price_at_t=anchor,
            future_closes=closes,
            future_highs=tuple(c * 1.001 for c in closes),
            future_lows=tuple(lows_list),
            past_closes_for_sigma=tuple(100.0 + 0.01 * (i % 3 - 1) for i in range(30)),
            reward_horizon=h,
        )

    def _synthesize_conclusions(self, report: dict[str, Any]) -> dict[str, Any]:
        time_q = report["1_fn_time_profile_audit"]["semantic_questions"]
        mae_q = report["2_mae_role_decomposition_audit"]["semantic_questions"]
        synth = report["3_synthetic_mae_path_comparison"]["narrative_checks"]

        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []
        recommendation: list[str] = []
        do_not: list[str] = []

        q1 = time_q["Q1_early_terminal_mixed_in_mean"]
        if q1.get("F_late_vs_terminal_return", 0) > 0.5:
            confirmed.append(
                "F_late aligns more strongly with terminal return than F_early — "
                "mean(f_n) mixes immediate entry signal with terminal outcome."
            )
        if q1.get("F_mean_vs_terminal_return", 0) > 0.4:
            confirmed.append(
                "mean(f_1..f_10) correlates with terminal return — behaves partly as "
                "'average of outcome horizons' rather than pure entry-attractiveness."
            )

        q2 = time_q["Q2_late_U_MAE_overwrites_early"]
        case = report["1_fn_time_profile_audit"]["case_a_vs_b"]
        if case.get("case_a", {}).get("F_late", 0) > case.get("case_a", {}).get("F_early", 0):
            confirmed.append(
                "Case A (dip→rise): F_late > F_early on average — late favorable U/MAE "
                "can lift aggregate F despite poor immediate entry at t."
            )

        q3 = time_q["Q3_early_vs_late_different_meaning"]
        if q3.get("early_vs_late_rank_inversion_rate", 0) > 0.3:
            confirmed.append(
                "Early-only vs late-only F rankings diverge materially — they carry "
                "different semantic content, not redundant views."
            )

        decay = time_q["Q6_path_decay_implicit_immediate"]
        if decay.get("P_decay_weight_early_fraction", 0) > 0.5:
            confirmed.append(
                f"Path decay (r={decay.get('decay_rate')}) already assigns "
                f"{decay.get('P_decay_weight_early_fraction', 0):.0%} weight to early bars — "
                "immediate preference exists before mean(f_n) aggregation."
            )

        mae1 = mae_q["Q1_full_mae_conflates_entry_and_later"]
        if mae1.get("similar_mae_different_archetype_pairs", 0) > 0:
            confirmed.append(
                f"Found {mae1['similar_mae_different_archetype_pairs']} full-MAE-matched pairs "
                "with different archetype/recovery — full MAE alone under-specifies entry risk."
            )

        if synth.get("case_a_vs_sustained", {}).get("early_mae_distinguishes_small_vs_large"):
            confirmed.append(
                "Synthetic: early MAE distinguishes small vs large immediate adverse "
                "when full MAE may be similar — early MAE better targets entry-time pain."
            )

        hypothesis.append(
            "Splitting F aggregation (early vs late heads or non-mean composer) may better "
            "serve P1 'start now vs wait' without adopting I/D as reward ground truth."
        )
        hypothesis.append(
            "Early MAE (k≤3) as entry-risk term alongside full MAE diagnostic may reduce "
            "Case A false penalty if paired with magnitude-free S/D — not auto-adopted."
        )
        hypothesis.append(
            "Recovery-after-MAE belongs in diagnostic or separate P1 output axis, not silent "
            "reward injection — it encodes deferred outcome."
        )

        unresolved.append(
            "Optimal f_n aggregation (mean vs weighted vs multi-head) requires human chart "
            "qualitative review after reward logic sign-off."
        )
        unresolved.append(
            "Whether time-to-MAE adds information beyond early MAE + S/D without redundancy."
        )
        unresolved.append(
            "P→S+D replacement remains Audit 3 hypothesis — not decided by time-profile or MAE audit."
        )

        recommendation.append(
            "Review non-mean f_n composition: treat F_early and F_late as distinct semantic "
            "probes in next experiments; do not pick by correlation alone."
        )
        recommendation.append(
            "If MAE is split, prioritize early_MAE for entry-risk reward term; keep recovery "
            "and time-to-MAE as diagnostics until P1 output structure is chosen."
        )
        recommendation.append(
            "When modifying reward (future phase), address double immediate bias: Path decay "
            "AND mean(f_n) early mixing AND scalar collapse."
        )

        do_not.append("Adopt mean(f_n) winner by terminal correlation alone.")
        do_not.append("Auto-insert recovery-after-MAE into canonical reward from this audit.")
        do_not.append("Declare MAE decomposition or S+D canonical from Audit 4.")
        do_not.append("Use captureability/robustness/I/D as ground truth.")
        do_not.append("Finalize P1 output or training target.")

        return {
            "CONFIRMED": confirmed,
            "HYPOTHESIS": hypothesis,
            "UNRESOLVED": unresolved,
            "RECOMMENDATION": recommendation,
            "DO_NOT_CONCLUDE": do_not,
        }


def _fn_case_stats(group: list[Audit4Sample]) -> dict[str, float | int]:
    if not group:
        return {"count": 0}
    return {
        "count": len(group),
        "F_mean": float(np.mean([_aggregate_fn(s.fn_values, ALL_NS) for s in group])),
        "F_early": float(np.mean([_aggregate_fn(s.fn_values, EARLY_NS) for s in group])),
        "F_late": float(np.mean([_aggregate_fn(s.fn_values, LATE_NS) for s in group])),
    }


def _mae_case_stats(group: list[Audit4Sample]) -> dict[str, float | int]:
    if not group:
        return {"count": 0}
    return {
        "count": len(group),
        "mean_full_mae": float(np.mean([s.mae_diag.full_mae for s in group])),
        "mean_early_mae": float(np.mean([s.mae_diag.early_mae for s in group])),
        "mean_recovery": float(np.mean([s.mae_diag.recovery_after_mae for s in group])),
        "mean_time_to_mae": float(
            np.nanmean(
                [
                    float(s.mae_diag.time_to_mae or 11)
                    for s in group
                ]
            )
        ),
    }


def _archetype_fn_note(arch: str) -> str:
    notes = {
        "dip_then_rise": "Case A: F_early should reflect adverse entry; F_late may rise with recovery.",
        "rise_then_fall": "Case B: F_early high (immediate scalp); F_late may fall with terminal loss.",
        "monotonic_rise": "F_early ≈ F_late ≈ F_mean — consistent favorable entry.",
        "monotonic_fall": "All F aggregates negative — direction avoid for LONG.",
        "spike_reversal": "F_early elevated; F_late depressed — horizon split visible.",
    }
    return notes.get(arch, "")


def format_audit4_summary(report: dict[str, Any]) -> str:
    time_a = report["1_fn_time_profile_audit"]["aggregation_comparison"]
    meta = report["meta"]
    lines = [
        "Reward Logic Audit 4",
        "=" * 72,
        f"eval_samples={meta['eval_samples']}",
        "-" * 72,
        "F aggregation means (LONG):",
        f"  mean(f_1..10)  {time_a['A_mean_f1_f10']['mean_F_long']:>8.3f}",
        f"  early f_1..3   {time_a['B_early_only_f1_f3']['mean_F_long']:>8.3f}",
        f"  middle f_4..7  {time_a['C_middle_only_f4_f7']['mean_F_long']:>8.3f}",
        f"  late f_8..10   {time_a['D_late_only_f8_f10']['mean_F_long']:>8.3f}",
        "-" * 72,
        f"CONFIRMED items: {len(report.get('CONFIRMED', []))}",
        f"HYPOTHESIS items: {len(report.get('HYPOTHESIS', []))}",
    ]
    return "\n".join(lines)


def save_audit4_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = RewardLogicAudit4Runner(market_data).run()
    print(format_audit4_summary(report))
    return report
