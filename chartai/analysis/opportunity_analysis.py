"""Case A/B and opportunity analysis — validates what 'entry opportunity' means."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable

import numpy as np

from chartai.analysis.execution_proxy import (
    FixedHorizonResult,
    TargetStopResult,
    policy_robustness_score,
    simulate_fixed_horizon,
    simulate_target_stop,
    threshold_timing,
)
from chartai.analysis.path_archetypes import ExtendedPathObservables, classify_archetype, compute_extended_observables
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import FittedZScoreNormalizer, FittedZScoreNormalizerSD
from chartai.reward.path import compute_path_n
from chartai.reward.speed_persistence import SDPair, compute_sd_pair_n
from chartai.reward.utility import compute_utility_n

# Representative S+D pair from prior experiment (not canonical)
SD_REPRESENTATIVE = SDPair.TTF_OCCUPANCY

TARGET_STOP_GRID = (
    (0.001, 0.0005),
    (0.001, 0.001),
    (0.002, 0.001),
    (0.003, 0.0015),
    (0.005, 0.002),
)

THRESHOLD_GRID = (0.001, 0.002, 0.003, 0.005)


@dataclass
class EnrichedSample:
    t_index: int
    f_baseline: float
    f_sd: float
    ext: ExtendedPathObservables
    archetype: str
    target_stop_results: tuple[TargetStopResult, ...]
    fixed_horizon: FixedHorizonResult
    threshold_results: tuple
    policy_robustness: float
    captureability: float
    entry_delay_pnl: dict[int, float]


@dataclass
class OpportunityAnalysisConfig:
    reward_horizon: int = 10
    decay_rate: float = 0.75
    min_past_bars: int = 20
    norm_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    terminal_match_tol: float = 0.0002
    entry_delays: tuple[int, ...] = (0, 1, 2, 3)


def _spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


class OpportunityAnalysisRunner:
    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: OpportunityAnalysisConfig | None = None,
    ) -> None:
        self._data = market_data
        self._config = config or OpportunityAnalysisConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._config.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._config.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._config
        t_indices = list(
            self._data.valid_t_indices(
                reward_horizon=cfg.reward_horizon,
                min_past_bars=cfg.min_past_bars,
            )
        )
        split_idx = max(1, int(len(t_indices) * cfg.norm_prefix_fraction))
        prefix_t = set(t_indices[:split_idx])
        eval_t = [t for t in t_indices if t not in prefix_t]

        prefix_raw: list[tuple] = []
        sd_prefix_raw: list[tuple] = []
        enriched: list[EnrichedSample] = []

        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            if t_index in prefix_t:
                for n in range(cfg.reward_horizon):
                    prefix_raw.append((
                        compute_path_n(ctx, Action.LONG, n + 1, decay_rate=cfg.decay_rate),
                        compute_utility_n(ctx, Action.LONG, n + 1, cfg.utility_config),
                        compute_mae_n(ctx, Action.LONG, n + 1),
                    ))
                    s, d = compute_sd_pair_n(
                        ctx, Action.LONG, n + 1, SD_REPRESENTATIVE, decay_rate=cfg.decay_rate
                    )
                    sd_prefix_raw.append((s, d, prefix_raw[-1][1], prefix_raw[-1][2]))

        norm_b = FittedZScoreNormalizer.fit(
            tuple(x[0] for x in prefix_raw),
            tuple(x[1] for x in prefix_raw),
            tuple(x[2] for x in prefix_raw),
        )
        norm_sd = FittedZScoreNormalizerSD.fit(
            tuple(x[0] for x in sd_prefix_raw),
            tuple(x[1] for x in sd_prefix_raw),
            tuple(x[2] for x in sd_prefix_raw),
            tuple(x[3] for x in sd_prefix_raw),
        )

        for t_index in eval_t:
            ctx = self._builder.build(t_index)
            enriched.append(self._enrich_sample(ctx, t_index, norm_b, norm_sd))

        report: dict[str, Any] = {
            "purpose": "Validate entry opportunity vs future-path observables; Case A/B; execution proxies",
            "market_data": describe_market_data(self._data),
            "eval_samples": len(enriched),
            "sd_representative": SD_REPRESENTATIVE.value,
            "case_ab_analysis": self._case_ab(enriched),
            "archetype_summary": self._archetype_summary(enriched),
            "same_terminal_pairs": self._same_terminal_pairs(enriched),
            "mfe_limitations": self._mfe_limitations(enriched),
            "execution_proxy": self._execution_summary(enriched),
            "captureability": self._captureability_analysis(enriched),
            "timing_sensitivity": self._timing_sensitivity(eval_t),
            "opportunity_robustness": self._robustness_analysis(enriched),
            "ranking_behavior": self._ranking_behavior(enriched),
            "key_questions": self._key_questions(enriched),
        }
        return report

    def _f_baseline(self, ctx: RewardContext, norm: FittedZScoreNormalizer) -> float:
        cfg = self._config
        return mean(
            norm.normalize_path(compute_path_n(ctx, Action.LONG, n, decay_rate=cfg.decay_rate))
            + norm.normalize_utility(compute_utility_n(ctx, Action.LONG, n, cfg.utility_config))
            - norm.normalize_mae(compute_mae_n(ctx, Action.LONG, n))
            for n in range(1, cfg.reward_horizon + 1)
        )

    def _f_sd(self, ctx: RewardContext, norm: FittedZScoreNormalizerSD) -> float:
        cfg = self._config
        return mean(
            norm.normalize_speed(
                compute_sd_pair_n(ctx, Action.LONG, n, SD_REPRESENTATIVE, decay_rate=cfg.decay_rate)[0]
            )
            + norm.normalize_persistence(
                compute_sd_pair_n(ctx, Action.LONG, n, SD_REPRESENTATIVE, decay_rate=cfg.decay_rate)[1]
            )
            + norm.normalize_utility(compute_utility_n(ctx, Action.LONG, n, cfg.utility_config))
            - norm.normalize_mae(compute_mae_n(ctx, Action.LONG, n))
            for n in range(1, cfg.reward_horizon + 1)
        )

    def _enrich_sample(
        self,
        ctx: RewardContext,
        t_index: int,
        norm_b: FittedZScoreNormalizer,
        norm_sd: FittedZScoreNormalizerSD,
    ) -> EnrichedSample:
        cfg = self._config
        h = cfg.reward_horizon
        ext = compute_extended_observables(ctx, Action.LONG, h)
        ts_results = tuple(
            simulate_target_stop(ctx, Action.LONG, target_pct=t, stop_pct=s)
            for t, s in TARGET_STOP_GRID
        )
        pnls = [r.proxy_pnl for r in ts_results]
        thr_results = tuple(
            threshold_timing(ctx, Action.LONG, threshold_pct=th, n=h) for th in THRESHOLD_GRID
        )
        mfe = ext.base.mfe
        capture = sum(1 for r in ts_results if r.target_first is True) / len(ts_results)
        delay_pnl: dict[int, float] = {}
        max_t = len(self._data.bars) - cfg.reward_horizon - 1
        for d in cfg.entry_delays:
            if t_index + d <= max_t:
                delay_pnl[d] = self._delayed_entry_pnl(t_index, d)
            else:
                delay_pnl[d] = float("nan")

        return EnrichedSample(
            t_index=t_index,
            f_baseline=self._f_baseline(ctx, norm_b),
            f_sd=self._f_sd(ctx, norm_sd),
            ext=ext,
            archetype=classify_archetype(ext),
            target_stop_results=ts_results,
            fixed_horizon=simulate_fixed_horizon(ctx, Action.LONG, horizon=h),
            threshold_results=thr_results,
            policy_robustness=policy_robustness_score(pnls),
            captureability=capture,
            entry_delay_pnl=delay_pnl,
        )

    def _delayed_entry_pnl(self, t_index: int, delay: int) -> float:
        cfg = self._config
        new_t = t_index + delay
        remaining = cfg.reward_horizon - delay
        if remaining < 1:
            return float("nan")
        if new_t + remaining > len(self._data.bars) - 1:
            return float("nan")
        ctx = self._builder.build(new_t)
        return simulate_fixed_horizon(ctx, Action.LONG, horizon=remaining).proxy_pnl

    def _case_ab(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        case_a = [s for s in samples if s.archetype == "dip_then_rise"]
        case_b = [s for s in samples if s.archetype == "rise_then_fall"]

        def bucket_stats(group: list[EnrichedSample]) -> dict[str, Any]:
            if not group:
                return {"count": 0}
            return {
                "count": len(group),
                "mean_f_baseline": float(np.mean([s.f_baseline for s in group])),
                "mean_f_sd": float(np.mean([s.f_sd for s in group])),
                "mean_mfe": float(np.mean([s.ext.base.mfe for s in group])),
                "mean_mae": float(np.mean([s.ext.base.mae for s in group])),
                "mean_terminal": float(np.mean([s.ext.base.terminal_return for s in group])),
                "mean_early": float(np.mean([s.ext.base.early_mean_return for s in group])),
                "mean_policy_robustness": float(np.mean([s.policy_robustness for s in group])),
                "mean_captureability": float(np.mean([s.captureability for s in group])),
                "target_first_rate": float(
                    np.mean([
                        np.mean([1 if r.target_first else 0 for r in s.target_stop_results if r.target_first is not None])
                        for s in group
                    ])
                ) if group else float("nan"),
                "mean_proxy_pnl_grid": float(
                    np.mean([np.mean([r.proxy_pnl for r in s.target_stop_results]) for s in group])
                ),
            }

        def exec_detail(group: list[EnrichedSample], label: str) -> dict[str, Any]:
            out: dict[str, Any] = {"archetype": label, "by_policy": {}}
            for i, (tgt, stp) in enumerate(TARGET_STOP_GRID):
                key = f"target_{tgt}_stop_{stp}"
                out["by_policy"][key] = {
                    "target_hit_rate": float(np.mean([s.target_stop_results[i].target_hit for s in group])),
                    "stop_hit_rate": float(np.mean([s.target_stop_results[i].stop_hit for s in group])),
                    "target_first_rate": float(np.mean([
                        1 if s.target_stop_results[i].target_first else 0
                        for s in group
                        if s.target_stop_results[i].target_first is not None
                    ])) if group else float("nan"),
                    "mean_proxy_pnl": float(np.mean([s.target_stop_results[i].proxy_pnl for s in group])),
                    "mean_time_to_target": float(np.nanmean([
                        s.target_stop_results[i].time_to_target or float("nan") for s in group
                    ])),
                }
            return out

        return {
            "case_a_dip_then_rise": bucket_stats(case_a),
            "case_b_rise_then_fall": bucket_stats(case_b),
            "case_a_execution": exec_detail(case_a, "dip_then_rise"),
            "case_b_execution": exec_detail(case_b, "rise_then_fall"),
            "f_ranking_gap_a_minus_b": (
                float(np.mean([s.f_baseline for s in case_b]) - np.mean([s.f_baseline for s in case_a]))
                if case_a and case_b
                else float("nan")
            ),
            "interpretation_note": (
                "Positive gap means Case B ranked higher than Case A by F. "
                "If Case B is 'opportunity at t' and Case A is 'poor entry at t', "
                "positive gap may align with opportunity hypothesis — must cross-check execution proxies."
            ),
        }

    def _archetype_summary(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        by_type: dict[str, list[EnrichedSample]] = {}
        for s in samples:
            by_type.setdefault(s.archetype, []).append(s)

        out = {}
        for name, group in sorted(by_type.items(), key=lambda x: -len(x[1])):
            out[name] = {
                "count": len(group),
                "mean_f_baseline": float(np.mean([s.f_baseline for s in group])),
                "mean_f_sd": float(np.mean([s.f_sd for s in group])),
                "mean_mfe": float(np.mean([s.ext.base.mfe for s in group])),
                "mean_mae": float(np.mean([s.ext.base.mae for s in group])),
                "mean_terminal": float(np.mean([s.ext.base.terminal_return for s in group])),
                "mean_robustness": float(np.mean([s.policy_robustness for s in group])),
            }
        return out

    def _same_terminal_pairs(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        cfg = self._config
        tol = cfg.terminal_match_tol
        pairs_analyzed: list[dict[str, Any]] = []
        inversions_baseline = 0
        inversions_sd = 0
        total_pairs = 0

        bins: dict[int, list[EnrichedSample]] = {}
        for s in samples:
            key = int(round(s.ext.base.terminal_return / tol))
            bins.setdefault(key, []).append(s)

        for _key, group in bins.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if abs(a.ext.base.terminal_return - b.ext.base.terminal_return) > tol:
                        continue
                    early_diff = abs(a.ext.base.early_mean_return - b.ext.base.early_mean_return)
                    if early_diff < 0.0002:
                        continue
                    total_pairs += 1
                    f_diff_b = b.f_baseline - a.f_baseline
                    early_higher = b if b.ext.base.early_mean_return > a.ext.base.early_mean_return else a
                    early_lower = a if early_higher is b else b
                    if early_higher.f_baseline > early_lower.f_baseline:
                        inversions_baseline += 1
                    if early_higher.f_sd > early_lower.f_sd:
                        inversions_sd += 1
                    if len(pairs_analyzed) < 15:
                        pairs_analyzed.append({
                            "t_a": a.t_index,
                            "t_b": b.t_index,
                            "terminal": a.ext.base.terminal_return,
                            "early_a": a.ext.base.early_mean_return,
                            "early_b": b.ext.base.early_mean_return,
                            "f_baseline_a": a.f_baseline,
                            "f_baseline_b": b.f_baseline,
                            "f_sd_a": a.f_sd,
                            "f_sd_b": b.f_sd,
                            "mfe_a": a.ext.base.mfe,
                            "mfe_b": b.ext.base.mfe,
                            "mae_a": a.ext.base.mae,
                            "mae_b": b.ext.base.mae,
                            "robust_a": a.policy_robustness,
                            "robust_b": b.policy_robustness,
                        })

        return {
            "terminal_tolerance": tol,
            "total_pairs": total_pairs,
            "early_favor_inversion_rate_baseline": inversions_baseline / total_pairs if total_pairs else float("nan"),
            "early_favor_inversion_rate_sd": inversions_sd / total_pairs if total_pairs else float("nan"),
            "note": (
                "Inversion = pair with similar terminal but higher early return gets higher F. "
                "High rate suggests F rewards early movement over entry quality when terminal matches."
            ),
            "example_pairs": pairs_analyzed,
        }

    def _mfe_limitations(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        high_mfe = [s for s in samples if s.ext.base.mfe >= _percentile([x.ext.base.mfe for x in samples], 90)]
        hard_capture = [s for s in high_mfe if s.captureability < 0.4]
        return {
            "high_mfe_q90_count": len(high_mfe),
            "high_mfe_hard_capture_count": len(hard_capture),
            "hard_capture_fraction": len(hard_capture) / len(high_mfe) if high_mfe else float("nan"),
            "high_mfe_mean_captureability": float(np.mean([s.captureability for s in high_mfe])),
            "spearman_mfe_captureability": _spearman(
                [s.ext.base.mfe for s in samples], [s.captureability for s in samples]
            ),
            "case_b_mean_mfe": float(np.mean([
                s.ext.base.mfe for s in samples if s.archetype == "rise_then_fall"
            ])) if any(s.archetype == "rise_then_fall" for s in samples) else float("nan"),
            "case_a_mean_mfe": float(np.mean([
                s.ext.base.mfe for s in samples if s.archetype == "dip_then_rise"
            ])) if any(s.archetype == "dip_then_rise" for s in samples) else float("nan"),
        }

    def _execution_summary(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        out: dict[str, Any] = {"policies": {}}
        for i, (tgt, stp) in enumerate(TARGET_STOP_GRID):
            key = f"t{tgt}_s{stp}"
            out["policies"][key] = {
                "mean_proxy_pnl": float(np.mean([s.target_stop_results[i].proxy_pnl for s in samples])),
                "target_hit_rate": float(np.mean([s.target_stop_results[i].target_hit for s in samples])),
                "stop_hit_rate": float(np.mean([s.target_stop_results[i].stop_hit for s in samples])),
                "target_first_rate": float(np.mean([
                    1 if s.target_stop_results[i].target_first else 0
                    for s in samples
                    if s.target_stop_results[i].target_first is not None
                ])),
            }
        out["fixed_horizon_mean_pnl"] = float(np.mean([s.fixed_horizon.proxy_pnl for s in samples]))
        return out

    def _captureability_analysis(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        return {
            "mean_captureability": float(np.mean([s.captureability for s in samples])),
            "spearman_f_baseline_capture": _spearman([s.f_baseline for s in samples], [s.captureability for s in samples]),
            "spearman_f_sd_capture": _spearman([s.f_sd for s in samples], [s.captureability for s in samples]),
            "spearman_mfe_capture": _spearman([s.ext.base.mfe for s in samples], [s.captureability for s in samples]),
            "top_f_baseline_capture": float(np.mean([
                s.captureability for s in sorted(samples, key=lambda x: x.f_baseline)[-len(samples)//10:]
            ])),
            "bottom_f_baseline_capture": float(np.mean([
                s.captureability for s in sorted(samples, key=lambda x: x.f_baseline)[: len(samples)//10]
            ])),
        }

    def _timing_sensitivity(self, eval_t: list[int]) -> dict[str, Any]:
        cfg = self._config
        max_t = len(self._data.bars) - cfg.reward_horizon - 1
        rows = []
        for t in eval_t[:500]:
            if t > max_t:
                continue
            pnls: dict[int, float] = {}
            for d in cfg.entry_delays:
                new_t = t + d
                remaining = cfg.reward_horizon - d
                if remaining < 1 or new_t > max_t:
                    pnls[d] = float("nan")
                    continue
                ctx_d = self._builder.build(new_t)
                pnls[d] = simulate_fixed_horizon(ctx_d, Action.LONG, horizon=remaining).proxy_pnl
            rows.append(pnls)
        out = {}
        for d in cfg.entry_delays:
            vals = [r[d] for r in rows if not np.isnan(r[d])]
            out[f"delay_{d}_mean_pnl"] = float(np.mean(vals)) if vals else float("nan")
        out["delay_1_vs_0_retention"] = (
            out.get("delay_1_mean_pnl", float("nan")) / out.get("delay_0_mean_pnl", 1e-12)
            if out.get("delay_0_mean_pnl")
            else float("nan")
        )
        return out

    def _robustness_analysis(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        robust = [s for s in samples if s.policy_robustness >= 0.6]
        fragile = [s for s in samples if s.policy_robustness <= 0.2]
        return {
            "mean_robustness": float(np.mean([s.policy_robustness for s in samples])),
            "robust_count_ge_0.6": len(robust),
            "fragile_count_le_0.2": len(fragile),
            "robust_mean_f_baseline": float(np.mean([s.f_baseline for s in robust])) if robust else float("nan"),
            "fragile_mean_f_baseline": float(np.mean([s.f_baseline for s in fragile])) if fragile else float("nan"),
            "spearman_f_baseline_robustness": _spearman(
                [s.f_baseline for s in samples], [s.policy_robustness for s in samples]
            ),
            "spearman_f_sd_robustness": _spearman([s.f_sd for s in samples], [s.policy_robustness for s in samples]),
        }

    def _ranking_behavior(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        top = sorted(samples, key=lambda s: s.f_baseline)[-len(samples)//10:]
        bot = sorted(samples, key=lambda s: s.f_baseline)[: len(samples)//10]

        def obs_mean(group: list[EnrichedSample], attr: str) -> float:
            return float(np.mean([getattr(s.ext.base, attr) for s in group]))

        return {
            "baseline_top_decile": {
                "mean_mfe": obs_mean(top, "mfe"),
                "mean_mae": obs_mean(top, "mae"),
                "mean_terminal": obs_mean(top, "terminal_return"),
                "mean_early": obs_mean(top, "early_mean_return"),
                "mean_robustness": float(np.mean([s.policy_robustness for s in top])),
                "mean_captureability": float(np.mean([s.captureability for s in top])),
                "case_b_fraction": float(np.mean([1 if s.archetype == "rise_then_fall" else 0 for s in top])),
                "case_a_fraction": float(np.mean([1 if s.archetype == "dip_then_rise" else 0 for s in top])),
            },
            "baseline_bottom_decile": {
                "mean_mfe": obs_mean(bot, "mfe"),
                "mean_mae": obs_mean(bot, "mae"),
                "mean_terminal": obs_mean(bot, "terminal_return"),
                "mean_early": obs_mean(bot, "early_mean_return"),
                "mean_robustness": float(np.mean([s.policy_robustness for s in bot])),
                "mean_captureability": float(np.mean([s.captureability for s in bot])),
            },
            "ranking_spearman_note": {
                "F_vs_mfe": _spearman([s.f_baseline for s in samples], [s.ext.base.mfe for s in samples]),
                "F_vs_robustness": _spearman([s.f_baseline for s in samples], [s.policy_robustness for s in samples]),
                "F_vs_captureability": _spearman([s.f_baseline for s in samples], [s.captureability for s in samples]),
                "F_vs_terminal": _spearman([s.f_baseline for s in samples], [s.ext.base.terminal_return for s in samples]),
                "F_vs_early": _spearman([s.f_baseline for s in samples], [s.ext.base.early_mean_return for s in samples]),
            },
        }

    def _key_questions(self, samples: list[EnrichedSample]) -> dict[str, Any]:
        top = sorted(samples, key=lambda s: s.f_baseline)[-len(samples)//10:]
        high_mfe_low_cap = [s for s in samples if s.ext.base.mfe > 0.002 and s.captureability < 0.3]
        neg_terminal_pos_robust = [
            s for s in samples
            if s.ext.base.terminal_return < 0 and s.policy_robustness >= 0.4
        ]
        pos_terminal_low_early = [
            s for s in samples
            if s.ext.base.terminal_return > 0 and s.ext.base.early_mean_return < -0.0003
        ]
        return {
            "A_high_F_good_for_P2_proxy": {
                "top_decile_mean_robustness": float(np.mean([s.policy_robustness for s in top])),
                "top_decile_mean_captureability": float(np.mean([s.captureability for s in top])),
            },
            "C_high_mfe_hard_capture_count": len(high_mfe_low_cap),
            "C_fraction_of_samples": len(high_mfe_low_cap) / len(samples),
            "D_pos_terminal_poor_early_count": len(pos_terminal_low_early),
            "E_neg_terminal_executable_count": len(neg_terminal_pos_robust),
            "E_fraction": len(neg_terminal_pos_robust) / len(samples),
            "F_case_ranking": {
                "mean_f_case_a": float(np.mean([s.f_baseline for s in samples if s.archetype == "dip_then_rise"])),
                "mean_f_case_b": float(np.mean([s.f_baseline for s in samples if s.archetype == "rise_then_fall"])),
            },
        }


def save_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
