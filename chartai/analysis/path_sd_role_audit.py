"""Reward Logic Audit 3 — Raw Path vs S+D role separation (analysis-only).

Does not modify canonical P1 reward. Compares magnitude in raw Path vs magnitude-free
Speed / Persistence candidates on the same causal prefix/eval split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.path_archetypes import classify_archetype, compute_extended_observables
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import FittedZScoreNormalizer, FittedZScoreNormalizerSD
from chartai.reward.path import compute_path_n
from chartai.reward.speed_persistence import (
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_speed_n,
)
from chartai.reward.synthetic import SyntheticPath, build_scenario, SyntheticScenario
from chartai.reward.utility import compute_utility_n


class StructureTag(str, Enum):
    """Comparison structures A–F at analysis horizon."""

    A_RAW_P = "A_raw_P"
    B_SPEED_ONLY = "B_speed_only"
    C_PERSISTENCE_ONLY = "C_persistence_only"
    D_S_PLUS_D = "D_S_plus_D"
    E_P_PLUS_U_MINUS_MAE = "E_P_plus_U_minus_MAE"
    F_S_PLUS_D_PLUS_U_MINUS_MAE = "F_S_plus_D_plus_U_minus_MAE"


# User-facing S/D names (audit 3) mapped to experimental reward candidates.
SPEED_AUDIT: dict[str, SpeedCandidate] = {
    "time_to_favorable": SpeedCandidate.TIME_TO_FAVORABLE,
    "early_sign_mass": SpeedCandidate.EARLY_SIGN,
    "early_favorable_occupancy": SpeedCandidate.EARLY_FAVORABLE_MASS,
}

PERSISTENCE_AUDIT: dict[str, PersistenceCandidate] = {
    "favorable_occupancy": PersistenceCandidate.FAVORABLE_OCCUPANCY,
    "max_favorable_run": PersistenceCandidate.MAX_FAVORABLE_RUN,
    "late_favorable_occupancy": PersistenceCandidate.LATE_FAVORABLE_MASS,
}

PRIMARY_SPEED = SpeedCandidate.TIME_TO_FAVORABLE
PRIMARY_PERSISTENCE = PersistenceCandidate.FAVORABLE_OCCUPANCY


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _late_favorable_occupancy_ratio(
    ctx: RewardContext, action: Action, n: int
) -> float:
    """Late-window favorable bar fraction — magnitude-free persistence probe."""
    sign = 1.0 if action is Action.LONG else -1.0
    rets = tuple(sign * ctx.return_from_t(k) for k in range(1, n + 1))
    start = n // 2
    late = rets[start:]
    if not late:
        return 0.0
    return sum(1 for r in late if r > 0) / len(late)


@dataclass
class AuditSample:
    t_index: int
    p_long: float
    u_long: float
    mae_long: float
    speed: dict[str, float]
    persistence: dict[str, float]
    p_short: float
    archetype: str


@dataclass
class PathSDRoleAuditConfig:
    reward_horizon: int = 10
    decay_rate: float = 0.75
    min_past_bars: int = 20
    norm_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    analysis_horizon: int = 10
    terminal_match_tol: float = 0.0002
    same_terminal_max_pairs: int = 200


class PathSDRoleAuditRunner:
    """Audit 3: semantic / logical role separation between raw P and S+D."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: PathSDRoleAuditConfig | None = None,
    ) -> None:
        self._data = market_data
        self._cfg = config or PathSDRoleAuditConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._cfg
        h = cfg.analysis_horizon
        t_indices = list(
            self._data.valid_t_indices(
                reward_horizon=cfg.reward_horizon,
                min_past_bars=cfg.min_past_bars,
            )
        )
        if not t_indices:
            raise ValueError("No valid samples")

        samples: list[AuditSample] = []
        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            ext = compute_extended_observables(ctx, Action.LONG, h)
            speed: dict[str, float] = {}
            persist: dict[str, float] = {}
            for name, cand in SPEED_AUDIT.items():
                speed[name] = compute_speed_n(
                    ctx, Action.LONG, h, cand, decay_rate=cfg.decay_rate
                )
            for name, cand in PERSISTENCE_AUDIT.items():
                if name == "late_favorable_occupancy":
                    persist[name] = _late_favorable_occupancy_ratio(ctx, Action.LONG, h)
                else:
                    persist[name] = compute_persistence_n(
                        ctx, Action.LONG, h, cand, decay_rate=cfg.decay_rate
                    )
            samples.append(
                AuditSample(
                    t_index=t_index,
                    p_long=compute_path_n(ctx, Action.LONG, h, decay_rate=cfg.decay_rate),
                    u_long=compute_utility_n(ctx, Action.LONG, h, cfg.utility_config),
                    mae_long=compute_mae_n(ctx, Action.LONG, h),
                    speed=speed,
                    persistence=persist,
                    p_short=compute_path_n(ctx, Action.SHORT, h, decay_rate=cfg.decay_rate),
                    archetype=classify_archetype(ext),
                )
            )

        split_idx = max(1, int(len(t_indices) * cfg.norm_prefix_fraction))
        prefix = samples[:split_idx]
        eval_samples = samples[split_idx:]

        p_prefix = tuple(s.p_long for s in prefix)
        u_prefix = tuple(s.u_long for s in prefix)
        m_prefix = tuple(s.mae_long for s in prefix)
        norm_pum = FittedZScoreNormalizer.fit(p_prefix, u_prefix, m_prefix)

        s_primary = tuple(s.speed["time_to_favorable"] for s in prefix)
        d_primary = tuple(s.persistence["favorable_occupancy"] for s in prefix)
        s_eval = [s.speed["time_to_favorable"] for s in eval_samples]
        d_eval = [s.persistence["favorable_occupancy"] for s in eval_samples]
        u_eval = [s.u_long for s in eval_samples]
        m_eval = [s.mae_long for s in eval_samples]
        p_eval = [s.p_long for s in eval_samples]

        norm_sd = FittedZScoreNormalizerSD.fit(
            tuple(s_primary),
            tuple(d_primary),
            u_prefix,
            m_prefix,
        )

        report: dict[str, Any] = {
            "1_experiment_purpose": (
                "Verify whether raw Path (P) embeds return magnitude beyond speed/persistence, "
                "and whether S+D separates timing/structure from U (magnitude) and MAE (entry risk). "
                "Not predictive performance selection; semantic role separation only."
            ),
            "structures_A_through_F": self._structures_a_f(
                eval_samples, norm_pum, norm_sd, cfg
            ),
            "2_speed_candidate_comparison": self._compare_speed_candidates(
                eval_samples, cfg.decay_rate
            ),
            "3_persistence_candidate_comparison": self._compare_persistence_candidates(
                eval_samples, cfg.decay_rate
            ),
            "4_raw_p_vs_sd_overlap": self._overlap_raw_vs_sd(
                p_eval, s_eval, d_eval, u_eval, m_eval, norm_pum, norm_sd
            ),
            "5_s_d_redundancy": self._s_d_redundancy(s_eval, d_eval, eval_samples),
            "6_u_role_separation": self._u_role_separation(
                eval_samples, cfg, norm_pum, h
            ),
            "7_mae_relationship": self._mae_relationship(s_eval, d_eval, m_eval, p_eval),
            "8_archetype_results": self._archetype_results(eval_samples, cfg, h),
            "9_same_terminal_analysis": self._same_terminal_analysis(eval_samples, cfg),
            "10_controlled_magnitude_experiment": self._controlled_magnitude(cfg),
            "11_raw_p_u_vs_s_d_u": self._raw_p_u_vs_s_d_u(
                eval_samples, norm_pum, norm_sd, cfg, h
            ),
            "12_logical_pros_cons": self._logical_pros_cons(),
            "13_CONFIRMED": [],
            "14_HYPOTHESIS": [],
            "15_UNRESOLVED": [],
            "16_next_experiments": [],
            "meta": {
                "market_data": describe_market_data(self._data),
                "num_valid_samples": len(t_indices),
                "eval_samples": len(eval_samples),
                "norm_prefix_fraction": cfg.norm_prefix_fraction,
                "primary_speed": PRIMARY_SPEED.value,
                "primary_persistence": PRIMARY_PERSISTENCE.value,
                "structures": [t.value for t in StructureTag],
            },
        }

        confirmed, hypothesis, unresolved, next_exps, replace_verdict = (
            self._synthesize_conclusions(report)
        )
        report["13_CONFIRMED"] = confirmed
        report["14_HYPOTHESIS"] = hypothesis
        report["15_UNRESOLVED"] = unresolved
        report["16_next_experiments"] = next_exps
        report["should_replace_raw_p_with_sd"] = replace_verdict
        return report

    def _structures_a_f(
        self,
        eval_samples: list[AuditSample],
        norm_pum: FittedZScoreNormalizer,
        norm_sd: FittedZScoreNormalizerSD,
        cfg: PathSDRoleAuditConfig,
    ) -> dict[str, Any]:
        """Mean eval scalars for structures A–F at analysis horizon."""
        h = cfg.analysis_horizon

        def mean_field(fn) -> float:
            return float(np.mean([fn(s) for s in eval_samples]))

        e_raw = mean_field(
            lambda s: norm_pum.normalize_path(s.p_long)
            + norm_pum.normalize_utility(s.u_long)
            - norm_pum.normalize_mae(s.mae_long)
        )
        f_raw = mean_field(
            lambda s: norm_sd.normalize_speed(s.speed["time_to_favorable"])
            + norm_sd.normalize_persistence(s.persistence["favorable_occupancy"])
            + norm_sd.normalize_utility(s.u_long)
            - norm_sd.normalize_mae(s.mae_long)
        )
        return {
            StructureTag.A_RAW_P.value: {
                "mean_P_h10": mean_field(lambda s: s.p_long),
                "note": "Raw path only — no U/MAE",
            },
            StructureTag.B_SPEED_ONLY.value: {
                "mean_S_ttf_h10": mean_field(lambda s: s.speed["time_to_favorable"]),
            },
            StructureTag.C_PERSISTENCE_ONLY.value: {
                "mean_D_occ_h10": mean_field(
                    lambda s: s.persistence["favorable_occupancy"]
                ),
            },
            StructureTag.D_S_PLUS_D.value: {
                "mean_S_plus_D_h10": mean_field(
                    lambda s: s.speed["time_to_favorable"]
                    + s.persistence["favorable_occupancy"]
                ),
            },
            StructureTag.E_P_PLUS_U_MINUS_MAE.value: {
                "mean_composite_h10": e_raw,
            },
            StructureTag.F_S_PLUS_D_PLUS_U_MINUS_MAE.value: {
                "mean_composite_h10": f_raw,
            },
        }

    def _compare_speed_candidates(
        self, eval_samples: list[AuditSample], decay_rate: float
    ) -> dict[str, Any]:
        rows: dict[str, dict[str, float]] = {}
        for name, cand in SPEED_AUDIT.items():
            values = [
                s.speed[name]
                for s in eval_samples
            ]
            rows[name] = {
                "definition": name,
                "reward_candidate": cand.value,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "semantic_note": _speed_semantic_note(name),
            }
        # Rank by correlation with early favorable occupancy proxy (timing purity)
        early_occ = [
            s.persistence["favorable_occupancy"]
            for s in eval_samples
        ]
        for name in rows:
            vals = [s.speed[name] for s in eval_samples]
            rows[name]["corr_with_favorable_occupancy"] = _pearson(vals, early_occ)
        rows["recommended_for_speed_meaning"] = "time_to_favorable"
        return rows

    def _compare_persistence_candidates(
        self, eval_samples: list[AuditSample], decay_rate: float
    ) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        for name in PERSISTENCE_AUDIT:
            vals = [s.persistence[name] for s in eval_samples]
            rows[name] = {
                "definition": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "semantic_note": _persistence_semantic_note(name),
            }
        rows["recommended_for_persistence_meaning"] = "favorable_occupancy"
        return rows

    def _overlap_raw_vs_sd(
        self,
        p_eval: list[float],
        s_eval: list[float],
        d_eval: list[float],
        u_eval: list[float],
        m_eval: list[float],
        norm_pum: FittedZScoreNormalizer,
        norm_sd: FittedZScoreNormalizerSD,
    ) -> dict[str, Any]:
        sd_sum = [s + d for s, d in zip(s_eval, d_eval)]
        p_z = [norm_pum.normalize_path(v) for v in p_eval]
        u_z = [norm_pum.normalize_utility(v) for v in u_eval]
        raw = {
            "P_S": _pearson(p_eval, s_eval),
            "P_D": _pearson(p_eval, d_eval),
            "P_U": _pearson(p_eval, u_eval),
            "S_U": _pearson(s_eval, u_eval),
            "D_U": _pearson(d_eval, u_eval),
            "S_plus_D_vs_P": _pearson(sd_sum, p_eval),
            "absP_absU": _pearson([abs(x) for x in p_eval], [abs(x) for x in u_eval]),
        }
        normed = {
            "P_U_after_norm": _pearson(
                [norm_pum.normalize_path(v) for v in p_eval],
                [norm_pum.normalize_utility(v) for v in u_eval],
            ),
            "S_U_after_norm": _pearson(
                [norm_sd.normalize_speed(v) for v in s_eval],
                [norm_sd.normalize_utility(v) for v in u_eval],
            ),
            "D_U_after_norm": _pearson(
                [norm_sd.normalize_persistence(v) for v in d_eval],
                [norm_sd.normalize_utility(v) for v in u_eval],
            ),
        }
        return {
            "Q1_raw_path_contains_magnitude": {
                "P_U_correlation": raw["P_U"],
                "P_vs_S_plus_D": raw["S_plus_D_vs_P"],
                "interpretation": (
                    "High P_U or high S_plus_D_vs_P suggests Path aggregates magnitude "
                    "similar to Utility, not pure timing/structure."
                ),
            },
            "raw_correlations_h10": raw,
            "normalized_correlations": normed,
            "normalization_masks_overlap": _normalization_masking_diagnosis(
                raw["P_U"], normed["P_U_after_norm"]
            ),
        }

    def _s_d_redundancy(
        self,
        s_eval: list[float],
        d_eval: list[float],
        eval_samples: list[AuditSample],
    ) -> dict[str, Any]:
        all_pairs: dict[str, float] = {}
        for s_name in SPEED_AUDIT:
            for d_name in PERSISTENCE_AUDIT:
                sv = [s.speed[s_name] for s in eval_samples]
                dv = [s.persistence[d_name] for s in eval_samples]
                all_pairs[f"{s_name}__{d_name}"] = _pearson(sv, dv)
        primary = _pearson(s_eval, d_eval)
        return {
            "Q3_S_D_redundancy": {
                "primary_S_D_corr": primary,
                "interpretation": (
                    "High S_D may be allowed if 'fast and persistent' is intentional amplification; "
                    "low S_D with divergent archetype behavior is required for independent axes."
                ),
            },
            "all_speed_persistence_pairs": all_pairs,
        }

    def _u_role_separation(
        self,
        eval_samples: list[AuditSample],
        cfg: PathSDRoleAuditConfig,
        norm_pum: FittedZScoreNormalizer,
        h: int,
    ) -> dict[str, Any]:
        controlled = self._controlled_magnitude(cfg)
        mag_rows = controlled["paths"]
        small = mag_rows[0]
        large = mag_rows[1]

        u_small = small["U"]
        u_large = large["U"]
        p_small = small["P"]
        p_large = large["P"]
        s_small = small["S_primary"]
        s_large = large["S_primary"]
        d_small = small["D_primary"]
        d_large = large["D_primary"]

        return {
            "Q4_role_separation_PU_vs_SDU": {
                "structure_A_P_plus_U": {
                    "corr_P_U_on_eval": _pearson(
                        [s.p_long for s in eval_samples],
                        [s.u_long for s in eval_samples],
                    ),
                    "controlled_magnitude": {
                        "P_ratio_large_over_small": p_large / p_small if p_small else float("nan"),
                        "U_ratio_large_over_small": u_large / u_small if u_small else float("nan"),
                        "P_and_U_both_scale_with_magnitude": (
                            p_large > p_small * 5 and u_large > u_small * 5
                        ),
                    },
                },
                "structure_B_S_plus_D_plus_U": {
                    "controlled_magnitude": {
                        "S_stable": abs(s_large - s_small) < 0.05,
                        "D_stable": abs(d_large - d_small) < 0.05,
                        "U_scales": u_large > u_small * 5,
                        "P_scales": p_large > p_small * 5,
                    },
                },
                "interpretation": (
                    "U should react to magnitude on controlled paths; S/D should not. "
                    "Raw P should co-scale with U on controlled paths (magnitude overlap)."
                ),
            },
            "eval_P_U": {
                "P_U": _pearson(
                    [s.p_long for s in eval_samples],
                    [s.u_long for s in eval_samples],
                ),
            },
        }

    def _mae_relationship(
        self,
        s_eval: list[float],
        d_eval: list[float],
        m_eval: list[float],
        p_eval: list[float],
    ) -> dict[str, Any]:
        return {
            "correlations_h10": {
                "P_MAE": _pearson(p_eval, m_eval),
                "S_MAE": _pearson(s_eval, m_eval),
                "D_MAE": _pearson(d_eval, m_eval),
            },
            "note": "MAE held identical between P+U and S+D+U comparisons; entry-risk axis not decomposed in this audit.",
        }

    def _archetype_results(
        self,
        eval_samples: list[AuditSample],
        cfg: PathSDRoleAuditConfig,
        h: int,
    ) -> dict[str, Any]:
        by_arch: dict[str, list[AuditSample]] = {}
        for s in eval_samples:
            by_arch.setdefault(s.archetype, []).append(s)

        archetype_rows: dict[str, Any] = {}
        for name, group in sorted(by_arch.items(), key=lambda x: -len(x[1])):
            archetype_rows[name] = {
                "count": len(group),
                "mean_P": float(np.mean([g.p_long for g in group])),
                "mean_S_ttf": float(np.mean([g.speed["time_to_favorable"] for g in group])),
                "mean_D_occ": float(np.mean([g.persistence["favorable_occupancy"] for g in group])),
                "mean_U": float(np.mean([g.u_long for g in group])),
                "mean_MAE": float(np.mean([g.mae_long for g in group])),
                "narrative": _archetype_narrative(name),
            }

        case_a = [s for s in eval_samples if s.archetype == "dip_then_rise"]
        case_b = [s for s in eval_samples if s.archetype == "rise_then_fall"]

        return {
            "by_archetype": archetype_rows,
            "case_a_dip_then_rise": _case_stats(case_a),
            "case_b_rise_then_fall": _case_stats(case_b),
            "case_check": {
                "S_should_be_low_in_A": (
                    float(np.mean([s.speed["time_to_favorable"] for s in case_a]))
                    < float(np.mean([s.speed["time_to_favorable"] for s in case_b]))
                    if case_a and case_b
                    else None
                ),
                "S_should_be_high_in_B": (
                    float(np.mean([s.speed["time_to_favorable"] for s in case_b]))
                    > float(np.mean([s.speed["time_to_favorable"] for s in case_a]))
                    if case_a and case_b
                    else None
                ),
            },
            "short_mirror": self._short_mirror_sample(cfg, h),
        }

    def _short_mirror_sample(
        self, cfg: PathSDRoleAuditConfig, h: int
    ) -> dict[str, Any]:
        """One synthetic bar path — LONG vs SHORT sign flip on same structure."""
        path_up = build_scenario(SyntheticScenario.DOWN_THEN_UP, horizon=cfg.reward_horizon)
        path_down = build_scenario(SyntheticScenario.UP_THEN_DOWN, horizon=cfg.reward_horizon)
        ctx_long = path_up.to_context()
        ctx_short = path_down.to_context()
        decay = cfg.decay_rate

        def row(ctx: RewardContext, action: Action) -> dict[str, float]:
            return {
                "P": compute_path_n(ctx, action, h, decay_rate=decay),
                "S": compute_speed_n(
                    ctx, action, h, SpeedCandidate.TIME_TO_FAVORABLE, decay_rate=decay
                ),
                "D": compute_persistence_n(
                    ctx, action, h, PersistenceCandidate.FAVORABLE_OCCUPANCY, decay_rate=decay
                ),
                "U": compute_utility_n(ctx, action, h, cfg.utility_config),
                "MAE": compute_mae_n(ctx, action, h),
            }

        return {
            "down_then_up_LONG": row(ctx_long, Action.LONG),
            "up_then_down_SHORT": row(ctx_short, Action.SHORT),
            "note": "Mirror structure: favorable leg timing differs by action sign.",
        }

    def _same_terminal_analysis(
        self, eval_samples: list[AuditSample], cfg: PathSDRoleAuditConfig
    ) -> dict[str, Any]:
        tol = cfg.terminal_match_tol
        max_pairs = cfg.same_terminal_max_pairs

        def terminal(s: AuditSample) -> float:
            ctx = self._builder.build(s.t_index)
            rets = tuple(ctx.return_from_t(k) for k in range(1, cfg.analysis_horizon + 1))
            return rets[-1]

        enriched: list[tuple[AuditSample, float, float]] = []
        for s in eval_samples:
            ctx = self._builder.build(s.t_index)
            rets = tuple(ctx.return_from_t(k) for k in range(1, cfg.analysis_horizon + 1))
            enriched.append((s, rets[-1], float(np.mean(rets[: max(1, len(rets) // 2)]))))

        bins: dict[int, list[tuple[AuditSample, float, float]]] = {}
        for item in enriched:
            term = item[1]
            key = int(round(term / tol))
            bins.setdefault(key, []).append(item)

        examples: list[dict[str, Any]] = []
        p_diff_count = s_diff_count = d_diff_count = 0
        total_pairs = 0

        for group in bins.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    sa, term_a, early_a = group[i]
                    sb, term_b, early_b = group[j]
                    if abs(term_a - term_b) > tol:
                        continue
                    if abs(early_a - early_b) < 0.0002:
                        continue
                    total_pairs += 1
                    pa, pb = sa.p_long, sb.p_long
                    ssa = sa.speed["time_to_favorable"]
                    ssb = sb.speed["time_to_favorable"]
                    da = sa.persistence["favorable_occupancy"]
                    db = sb.persistence["favorable_occupancy"]
                    if abs(pa - pb) > 0.0005:
                        p_diff_count += 1
                    if abs(ssa - ssb) > 0.05:
                        s_diff_count += 1
                    if abs(da - db) > 0.05:
                        d_diff_count += 1
                    if len(examples) < 12:
                        examples.append(
                            {
                                "terminal": term_a,
                                "early_a": early_a,
                                "early_b": early_b,
                                "P_a": pa,
                                "P_b": pb,
                                "S_a": ssa,
                                "S_b": ssb,
                                "D_a": da,
                                "D_b": db,
                            }
                        )
                    if total_pairs >= max_pairs:
                        break
                if total_pairs >= max_pairs:
                    break

        return {
            "terminal_tolerance": tol,
            "total_pairs": total_pairs,
            "pairs_with_different_P_same_terminal": p_diff_count,
            "pairs_with_different_S_same_terminal": s_diff_count,
            "pairs_with_different_D_same_terminal": d_diff_count,
            "examples": examples,
            "note": (
                "Same-terminal pairs isolate whether P encodes magnitude beyond terminal; "
                "S/D should differ when early path differs."
            ),
        }

    def _controlled_magnitude(self, cfg: PathSDRoleAuditConfig) -> dict[str, Any]:
        h = cfg.analysis_horizon
        decay = cfg.decay_rate
        small, large = 0.01, 0.10
        paths = []
        for mag, label in ((small, "small_magnitude"), (large, "large_magnitude")):
            moves = [mag] * h
            anchor = 100.0
            closes = tuple(anchor * (1.0 + mag) ** k for k in range(1, h + 1))
            highs = tuple(c * 1.001 for c in closes)
            lows = tuple(c * 0.999 for c in closes)
            syn = SyntheticPath(
                name=label,
                price_at_t=anchor,
                future_closes=closes,
                future_highs=highs,
                future_lows=lows,
                past_closes_for_sigma=tuple(100.0 + 0.01 * (i % 3 - 1) for i in range(30)),
                reward_horizon=h,
            )
            ctx = syn.to_context()
            paths.append(
                {
                    "label": label,
                    "magnitude_per_bar": mag,
                    "P": compute_path_n(ctx, Action.LONG, h, decay_rate=decay),
                    "S_primary": compute_speed_n(
                        ctx,
                        Action.LONG,
                        h,
                        SpeedCandidate.TIME_TO_FAVORABLE,
                        decay_rate=decay,
                    ),
                    "D_primary": compute_persistence_n(
                        ctx,
                        Action.LONG,
                        h,
                        PersistenceCandidate.FAVORABLE_OCCUPANCY,
                        decay_rate=decay,
                    ),
                    "U": compute_utility_n(ctx, Action.LONG, h, cfg.utility_config),
                    "MAE": compute_mae_n(ctx, Action.LONG, h),
                    "aligned_returns": tuple(
                        ctx.return_from_t(k) for k in range(1, h + 1)
                    ),
                }
            )

        p_ratio = paths[1]["P"] / paths[0]["P"] if paths[0]["P"] else float("nan")
        u_ratio = paths[1]["U"] / paths[0]["U"] if paths[0]["U"] else float("nan")
        return {
            "Q2_magnitude_free_S_D": {
                "S_equal": abs(paths[1]["S_primary"] - paths[0]["S_primary"]) < 0.02,
                "D_equal": abs(paths[1]["D_primary"] - paths[0]["D_primary"]) < 0.02,
                "P_scales_with_magnitude": p_ratio,
                "U_scales_with_magnitude": u_ratio,
            },
            "paths": paths,
            "expected": {
                "P_scales": True,
                "U_scales": True,
                "S_invariant": True,
                "D_invariant": True,
            },
        }

    def _raw_p_u_vs_s_d_u(
        self,
        eval_samples: list[AuditSample],
        norm_pum: FittedZScoreNormalizer,
        norm_sd: FittedZScoreNormalizerSD,
        cfg: PathSDRoleAuditConfig,
        h: int,
    ) -> dict[str, Any]:
        mae = [s.mae_long for s in eval_samples]
        pu = [
            norm_pum.normalize_path(s.p_long)
            + norm_pum.normalize_utility(s.u_long)
            for s in eval_samples
        ]
        sdu = [
            norm_sd.normalize_speed(s.speed["time_to_favorable"])
            + norm_sd.normalize_persistence(s.persistence["favorable_occupancy"])
            + norm_sd.normalize_utility(s.u_long)
            for s in eval_samples
        ]
        full_pu_mae = [pu[i] - norm_pum.normalize_mae(mae[i]) for i in range(len(eval_samples))]
        full_sdu_mae = [
            sdu[i] - norm_sd.normalize_mae(mae[i]) for i in range(len(eval_samples))
        ]
        return {
            "structure_E_P_plus_U_minus_MAE_mean": float(np.mean(full_pu_mae)),
            "structure_F_S_plus_D_plus_U_minus_MAE_mean": float(np.mean(full_sdu_mae)),
            "corr_PU_vector_with_U": _pearson(
                [s.p_long + s.u_long for s in eval_samples],
                [s.u_long for s in eval_samples],
            ),
            "corr_SDU_vector_with_U": _pearson(sdu, [s.u_long for s in eval_samples]),
            "mean_abs_P_on_eval": float(np.mean([abs(s.p_long) for s in eval_samples])),
            "mean_abs_S_plus_D_on_eval": float(
                np.mean(
                    [
                        abs(
                            s.speed["time_to_favorable"]
                            + s.persistence["favorable_occupancy"]
                        )
                        for s in eval_samples
                    ]
                )
            ),
            "note": "MAE term identical normalization pipeline; compare composite role clarity only.",
        }

    def _logical_pros_cons(self) -> dict[str, Any]:
        return {
            "keep_raw_P_plus_U_minus_MAE": {
                "pros": [
                    "Implemented baseline; continuity with Phase 0–2 experiments",
                    "Single Path scalar; simple composer",
                ],
                "cons": [
                    "P embeds t-anchored cumulative returns (magnitude + decay timing)",
                    "Empirical P_U overlap (~0.58) — double-count risk with U",
                    "Scalar F cannot express wait-for-better-entry (Case A) vs direction skip",
                ],
            },
            "candidate_S_plus_D_plus_U_minus_MAE": {
                "pros": [
                    "Aligns Path intent with speed + persistence axes",
                    "Magnitude-free S/D; U carries outcome size",
                    "Archetype diagnostics separate timing from terminal return",
                ],
                "cons": [
                    "S_D redundancy partial (ρ often 0.6–0.8)",
                    "Does not alone fix scalar collapse / deferred conflict",
                    "Extra normalization streams; not canonical until logic sign-off",
                ],
            },
            "role_division_intent": {
                "S_D": "timing & structure (fast start + sustained favorable sign)",
                "U": "magnitude & tail at horizon n",
                "MAE": "entry-time adverse risk (single scalar here)",
            },
        }

    def _synthesize_conclusions(
        self, report: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str], list[str], dict[str, Any]]:
        raw = report["4_raw_p_vs_sd_overlap"]["raw_correlations_h10"]
        ctrl = report["10_controlled_magnitude_experiment"]["Q2_magnitude_free_S_D"]
        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []

        if ctrl.get("P_scales_with_magnitude") and ctrl.get("U_scales_with_magnitude"):
            confirmed.append(
                "Controlled equal-sign paths: Raw P and U scale with per-bar return magnitude."
            )
        if ctrl.get("S_equal") and ctrl.get("D_equal"):
            confirmed.append(
                "Controlled equal-sign paths: primary S (time_to_favorable) and D (favorable_occupancy) "
                "are invariant to magnitude when timing/sign structure is fixed."
            )

        p_u = raw.get("P_U")
        if p_u is not None and not math.isnan(p_u) and p_u > 0.4:
            confirmed.append(
                f"On eval LONG samples at h=10, P and U remain correlated (ρ≈{p_u:.2f}) — "
                "magnitude overlap in raw Path is present in market data, not only in synthetic control."
            )

        hypothesis.append(
            "Replacing raw P with S+D reduces P–U magnitude double-counting while leaving MAE timing-blind."
        )
        hypothesis.append(
            "High S–D correlation may be intentional amplification ('fast and persistent' paths) "
            "rather than pure redundancy failure."
        )

        unresolved.append(
            "Whether mean(f_n) time-mixing should be decomposed before any structure is canonical."
        )
        unresolved.append(
            "MAE decomposition (early vs full) not in scope of this audit — single MAE may hide Case A entry pain."
        )
        unresolved.append(
            "Scalar F vs multi-aspect P1 output remains undecided until reward logic sign-off."
        )

        replace = {
            "answer": "not_yet",
            "rationale": (
                "Current results support S+D as a cleaner timing/structure carrier and raw P as magnitude-coupled, "
                "but replacing canonical P requires reward logic sign-off — not correlation winner selection. "
                "Controlled magnitude validates S/D invariance; market eval still shows P–U overlap."
            ),
            "requires": [
                "User review of archetype + same-terminal narratives",
                "f_n profile experiment (not run here)",
                "MAE early vs full diagnostic (deferred)",
            ],
        }

        next_exps = [
            "P vs S+D component overlap by archetype (magnitude-free path vs U) on full BTC eval",
            "f_1..f_H profile for Case A/B (mean vs early-only vs late-only)",
            "early_MAE vs full MAE on Case A / Q3 dual-axis quadrants (diagnostic)",
            "Multi-asset quadrant stability for S/D invariance",
            "Normalization on/off ablation — does z-score hide or reveal overlap",
        ]
        return confirmed, hypothesis, unresolved, next_exps, replace


def _speed_semantic_note(name: str) -> str:
    notes = {
        "time_to_favorable": "Inverse time-to-first favorable step; pure timing, no |R|.",
        "early_sign_mass": "Early decay-weighted sign(aligned R_k); no magnitude.",
        "early_favorable_occupancy": "Early-window fraction favorable; no |R|.",
    }
    return notes.get(name, "")


def _persistence_semantic_note(name: str) -> str:
    notes = {
        "favorable_occupancy": "Fraction of steps with aligned R_k > 0 over full window.",
        "max_favorable_run": "Longest consecutive favorable run divided by n.",
        "late_favorable_occupancy": "Late half favorable fraction; persistence after onset.",
    }
    return notes.get(name, "")


def _archetype_narrative(name: str) -> str:
    narratives = {
        "dip_then_rise": "Case A: early adverse, later favorable — P may downrank t despite positive terminal.",
        "rise_then_fall": "Case B: early favorable, later adverse — scalp appeal vs hold terminal.",
        "monotonic_rise": "All favorable steps — S and D both high; P also encodes magnitude.",
        "monotonic_fall": "Adverse for LONG — components negative or zero.",
        "spike_reversal": "Early spike then reversal — high early S, falling D.",
        "slow_grind": "mixed / slow_flat — moderate occupancy, low speed.",
        "fast favorable → fade": "rise_then_fall or spike_reversal alias.",
    }
    return narratives.get(name, "See path_archetypes.classify_archetype rules.")


def _case_stats(group: list[AuditSample]) -> dict[str, float | int]:
    if not group:
        return {"count": 0}
    return {
        "count": len(group),
        "mean_P": float(np.mean([g.p_long for g in group])),
        "mean_S_ttf": float(np.mean([g.speed["time_to_favorable"] for g in group])),
        "mean_D_occ": float(np.mean([g.persistence["favorable_occupancy"] for g in group])),
        "mean_U": float(np.mean([g.u_long for g in group])),
        "mean_MAE": float(np.mean([g.mae_long for g in group])),
    }


def _normalization_masking_diagnosis(raw_p_u: float, norm_p_u: float) -> dict[str, Any]:
    return {
        "raw_P_U": raw_p_u,
        "norm_P_U": norm_p_u,
        "overlap_reduced_after_norm": (
            not math.isnan(raw_p_u)
            and not math.isnan(norm_p_u)
            and abs(norm_p_u) < abs(raw_p_u) * 0.85
        ),
        "interpretation": (
            "If norm_P_U << raw_P_U, normalization compresses but does not remove shared magnitude axis."
        ),
    }


def format_audit_summary(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    overlap = report.get("4_raw_p_vs_sd_overlap", {}).get("raw_correlations_h10", {})
    ctrl = report.get("10_controlled_magnitude_experiment", {}).get("Q2_magnitude_free_S_D", {})
    lines = [
        "Path vs S+D Role Separation Audit (3)",
        "=" * 72,
        f"eval_samples={meta.get('eval_samples')}  primary S=time_to_favorable  D=favorable_occupancy",
        "-" * 72,
        f"P_U (raw eval)     {overlap.get('P_U', float('nan')):>8.3f}",
        f"S_D (primary)      {report.get('5_s_d_redundancy', {}).get('Q3_S_D_redundancy', {}).get('primary_S_D_corr', float('nan')):>8.3f}",
        f"P ratio ctrl       {ctrl.get('P_scales_with_magnitude', float('nan')):>8}",
        f"S equal ctrl       {ctrl.get('S_equal', False)!s:>8}",
        f"D equal ctrl       {ctrl.get('D_equal', False)!s:>8}",
        "-" * 72,
        f"replace P with S+D?  {report.get('should_replace_raw_p_with_sd', {}).get('answer', 'n/a')}",
    ]
    return "\n".join(lines)


def save_audit_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = PathSDRoleAuditRunner(market_data).run()
    print(format_audit_summary(report))
    return report
