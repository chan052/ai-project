"""Dual-axis (Immediate vs Deferred) opportunity labeling experiment.



Analysis-only — does NOT finalize canonical P1 target or opportunity formula.

"""



from __future__ import annotations



import json

from dataclasses import dataclass, field

from statistics import mean

from typing import Any



import numpy as np



from chartai.analysis.dual_axis_scores import (

    STANDARD_POLICY_GRID,

    DeferredAxisScores,

    ImmediateAxisScores,

    compute_dual_axis_scores,

    delay_captureability_curve,

    policy_key,

)

from chartai.analysis.opportunity_analysis import (
    SD_REPRESENTATIVE,
    _spearman,
    save_report,
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

from chartai.reward.speed_persistence import compute_sd_pair_n

from chartai.reward.utility import compute_utility_n





@dataclass

class DualAxisSample:

    t_index: int

    f_baseline: float

    f_sd: float

    ext: ExtendedPathObservables

    archetype: str

    immediate: ImmediateAxisScores

    deferred: DeferredAxisScores

    I_observable: float

    D_observable: float

    delay_curve: dict[int, dict[str, float]]





@dataclass

class DualAxisAnalysisConfig:

    reward_horizon: int = 10

    decay_rate: float = 0.75

    min_past_bars: int = 20

    norm_prefix_fraction: float = 0.5

    utility_config: UtilityConfig = field(default_factory=UtilityConfig)

    terminal_match_tol: float = 0.0002

    entry_delays: tuple[int, ...] = (0, 1, 2, 3, 5, 10)

    quadrant_split: str = "median"





def _estimate_mutual_information(x: list[float], y: list[float], *, bins: int = 15) -> float:

    xa = np.asarray(x, dtype=float)

    ya = np.asarray(y, dtype=float)

    if len(xa) < 10:

        return float("nan")

    c_xy, _, _ = np.histogram2d(xa, ya, bins=bins)

    p_xy = c_xy / max(c_xy.sum(), 1e-15)

    p_x = p_xy.sum(axis=1, keepdims=True)

    p_y = p_xy.sum(axis=0, keepdims=True)

    mask = p_xy > 0

    with np.errstate(divide="ignore", invalid="ignore"):

        log_ratio = np.log(p_xy[mask] / (p_x @ p_y)[mask])

    return float(np.sum(p_xy[mask] * log_ratio))





def _pearson(a: list[float], b: list[float]) -> float:

    x = np.asarray(a, dtype=float)

    y = np.asarray(b, dtype=float)

    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:

        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])





class DualAxisAnalysisRunner:

    def __init__(

        self,

        market_data: MarketDataSource,

        *,

        config: DualAxisAnalysisConfig | None = None,

    ) -> None:

        self._data = market_data

        self._config = config or DualAxisAnalysisConfig()

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



        norm_b, norm_sd = self._fit_normalizers(prefix_t, cfg)

        max_bar = len(self._data.bars) - 1

        samples: list[DualAxisSample] = []



        for t_index in eval_t:

            ctx = self._builder.build(t_index)

            immediate, deferred, policies = compute_dual_axis_scores(

                self._builder,

                t_index=t_index,

                horizon=cfg.reward_horizon,

                action=Action.LONG,

                max_bar_index=max_bar,

            )

            ext = compute_extended_observables(ctx, Action.LONG, cfg.reward_horizon)

            delay_curve = delay_captureability_curve(

                self._builder,

                t_index=t_index,

                horizon=cfg.reward_horizon,

                action=Action.LONG,

                delays=cfg.entry_delays,

                policies=policies,

                max_bar_index=max_bar,

            )

            samples.append(

                DualAxisSample(

                    t_index=t_index,

                    f_baseline=self._f_baseline(ctx, norm_b),

                    f_sd=self._f_sd(ctx, norm_sd),

                    ext=ext,

                    archetype=classify_archetype(ext),

                    immediate=immediate,

                    deferred=deferred,

                    I_observable=immediate.composite,

                    D_observable=deferred.composite,

                    delay_curve=delay_curve,

                )

            )



        i_med = float(np.median([s.I_observable for s in samples]))

        d_med = float(np.median([s.D_observable for s in samples]))



        report: dict[str, Any] = {

            "purpose": (

                "Validate whether Immediate and Deferred opportunity exist as "

                "distinct analysis axes — NOT canonical P1 target definition"

            ),

            "market_data": describe_market_data(self._data),

            "eval_samples": len(samples),

            "config": {

                "reward_horizon": cfg.reward_horizon,

                "policy_grid_size": len(STANDARD_POLICY_GRID),

                "entry_delays": list(cfg.entry_delays),

                "I_definition": "mean target-first rate across fixed+vol-scaled policies at delay=0",

                "D_definition": (

                    "mean(proxy_pnl at best delay τ* - proxy_pnl at τ=0) across policies; "

                    "hindsight scan for analysis only"

                ),

            },

            "immediate_definitions": self._immediate_definitions(),

            "deferred_definitions": self._deferred_definitions(),

            "execution_policies": {

                "standard_grid": [list(p) for p in STANDARD_POLICY_GRID],

                "vol_scaled": "target=c*sigma_t, stop=c*sigma_t/2 for c in [1.0,1.5,2.0]",

            },

            "immediate_results": self._immediate_summary(samples),

            "deferred_results": self._deferred_summary(samples),

            "id_correlation": self._id_correlation(samples),

            "scatter_sample": [
                {"I": s.I_observable, "D": s.D_observable, "archetype": s.archetype}
                for s in samples[:: max(1, len(samples) // 500)]
            ],

            "quadrant_analysis": self._quadrant_analysis(samples, i_med, d_med),

            "case_ab_analysis": self._case_ab(samples),

            "same_terminal_pairs": self._same_terminal_pairs(samples),

            "timing_sensitivity": self._timing_sensitivity(samples),

            "policy_robustness": self._policy_robustness(samples),

            "baseline_f_vs_id": self._f_vs_id(samples, "f_baseline"),

            "sd_f_vs_id": self._f_vs_id(samples, "f_sd"),

            "representative_paths": self._representative_paths(samples, i_med, d_med),

            "confirmed": [],

            "hypothesis": [],

            "unresolved": [],

            "next_experiments": [],

        }

        report.update(self._auto_interpretation(report, samples))

        return report



    def _fit_normalizers(self, prefix_t: set[int], cfg: DualAxisAnalysisConfig):

        prefix_raw: list[tuple] = []

        sd_prefix_raw: list[tuple] = []

        for t_index in sorted(prefix_t):

            ctx = self._builder.build(t_index)

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

        return norm_b, norm_sd



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



    def _immediate_definitions(self) -> dict[str, str]:

        return {

            "meaning": "Can favorable movement be captured by entering at t immediately?",

            "captureability": "Fraction of policies where target hits before stop (delay=0)",

            "robustness": "Fraction of policies with positive proxy PnL at delay=0",

            "composite_I": "Same as captureability — analysis proxy, not canonical F",

        }



    def _deferred_definitions(self) -> dict[str, str]:

        return {

            "meaning": "Does a better entry exist within [t, t+H] even if t is poor?",

            "best_delay": "Per policy, delay maximizing proxy PnL; consensus = median across policies",

            "improvement": "best_pnl - immediate_pnl per policy; D = mean improvement",

            "hindsight_note": "Best entry uses future data — analysis label only, never P1 feature",

        }



    def _immediate_summary(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        return {

            "mean_captureability": float(np.mean([s.immediate.captureability for s in samples])),

            "mean_robustness": float(np.mean([s.immediate.robustness for s in samples])),

            "mean_proxy_pnl": float(np.mean([s.immediate.mean_proxy_pnl for s in samples])),

            "by_policy": self._policy_aggregate(samples, immediate=True),

        }



    def _deferred_summary(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        delays = [s.deferred.consensus_best_delay for s in samples]

        return {

            "mean_improvement": float(np.mean([s.deferred.mean_improvement for s in samples])),

            "mean_deferred_capture": float(np.mean([s.deferred.deferred_capture_at_best for s in samples])),

            "mean_deferred_robustness": float(np.mean([s.deferred.deferred_robustness for s in samples])),

            "best_entry_delay_distribution": {

                "mean": float(np.mean(delays)),

                "median": float(np.median(delays)),

                "pct_delay_gt_0": float(np.mean([1 if d > 0 else 0 for d in delays])),

                "histogram": {str(k): int(v) for k, v in zip(*np.unique(delays, return_counts=True))},

            },

            "mean_best_entry_mfe": float(np.nanmean([s.deferred.best_entry_mfe for s in samples])),

            "mean_best_entry_mae": float(np.nanmean([s.deferred.best_entry_mae for s in samples])),

        }



    def _policy_aggregate(self, samples: list[DualAxisSample], *, immediate: bool) -> dict[str, Any]:

        if not samples:

            return {}

        n_policies = len(samples[0].immediate.outcomes)

        out: dict[str, Any] = {}

        for i in range(n_policies):

            o0 = samples[0].immediate.outcomes[i]

            key = policy_key(o0.target_pct, o0.stop_pct)

            if immediate:

                out[key] = {

                    "mean_target_first": float(np.mean([

                        1 if s.immediate.outcomes[i].target_first else 0

                        for s in samples

                        if s.immediate.outcomes[i].target_first is not None

                    ])),

                    "mean_proxy_pnl": float(np.mean([s.immediate.outcomes[i].proxy_pnl for s in samples])),

                }

            else:

                out[key] = {

                    "mean_improvement": float(np.mean([s.deferred.scans[i].improvement for s in samples])),

                    "mean_best_delay": float(np.mean([s.deferred.scans[i].best_delay for s in samples])),

                }

        return out



    def _id_correlation(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        i_vals = [s.I_observable for s in samples]

        d_vals = [s.D_observable for s in samples]

        return {

            "pearson_I_D": _pearson(i_vals, d_vals),

            "spearman_I_D": _spearman(i_vals, d_vals),

            "mutual_information_I_D": _estimate_mutual_information(i_vals, d_vals),

            "interpretation_note": (

                "Correlation is reference only. Quadrant counts (especially I-low/D-high) "

                "matter more than correlation magnitude."

            ),

        }



    def _quadrant_analysis(

        self, samples: list[DualAxisSample], i_med: float, d_med: float

    ) -> dict[str, Any]:

        quadrants = {"Q1_Ihigh_Dlow": [], "Q2_Ihigh_Dhigh": [], "Q3_Ilow_Dhigh": [], "Q4_Ilow_Dlow": []}

        i_vals = np.array([s.I_observable for s in samples], dtype=float)
        d_vals = np.array([s.D_observable for s in samples], dtype=float)
        i_rank = np.argsort(np.argsort(i_vals))
        d_rank = np.argsort(np.argsort(d_vals))
        for s, i_r, d_r in zip(samples, i_rank, d_rank):
            i_hi = i_r >= len(samples) / 2
            d_hi = d_r >= len(samples) / 2
            if i_hi and not d_hi:
                quadrants["Q1_Ihigh_Dlow"].append(s)
            elif i_hi and d_hi:
                quadrants["Q2_Ihigh_Dhigh"].append(s)
            elif not i_hi and d_hi:
                quadrants["Q3_Ilow_Dhigh"].append(s)
            else:
                quadrants["Q4_Ilow_Dlow"].append(s)



        def stats(group: list[DualAxisSample]) -> dict[str, Any]:

            if not group:

                return {"count": 0}

            return {

                "count": len(group),

                "fraction": len(group) / len(samples),

                "mean_terminal": float(np.mean([s.ext.base.terminal_return for s in group])),

                "mean_mfe": float(np.mean([s.ext.base.mfe for s in group])),

                "mean_mae": float(np.mean([s.ext.base.mae for s in group])),

                "mean_f_baseline": float(np.mean([s.f_baseline for s in group])),

                "mean_immediate_capture": float(np.mean([s.I_observable for s in group])),

                "mean_deferred_improvement": float(np.mean([s.D_observable for s in group])),

                "mean_immediate_robustness": float(np.mean([s.immediate.robustness for s in group])),

                "mean_deferred_robustness": float(np.mean([s.deferred.deferred_robustness for s in group])),

                "mean_proxy_pnl_immediate": float(np.mean([s.immediate.mean_proxy_pnl for s in group])),

                "case_a_fraction": float(np.mean([1 if s.archetype == "dip_then_rise" else 0 for s in group])),

                "case_b_fraction": float(np.mean([1 if s.archetype == "rise_then_fall" else 0 for s in group])),

            }



        return {

            "split": {
                "I_median": i_med,
                "D_median": d_med,
                "method": "rank-based median split (top half = HIGH)",
            },

            "quadrants": {k: stats(v) for k, v in quadrants.items()},

            "Q3_Ilow_Dhigh_fraction": stats(quadrants["Q3_Ilow_Dhigh"]).get("fraction", 0),

        }



    def _case_ab(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        case_a = [s for s in samples if s.archetype == "dip_then_rise"]

        case_b = [s for s in samples if s.archetype == "rise_then_fall"]



        def bucket(group: list[DualAxisSample], label: str) -> dict[str, Any]:

            if not group:

                return {"count": 0, "archetype": label}

            return {

                "count": len(group),

                "archetype": label,

                "mean_I": float(np.mean([s.I_observable for s in group])),

                "mean_D": float(np.mean([s.D_observable for s in group])),

                "mean_terminal": float(np.mean([s.ext.base.terminal_return for s in group])),

                "mean_mfe": float(np.mean([s.ext.base.mfe for s in group])),

                "mean_mae": float(np.mean([s.ext.base.mae for s in group])),

                "mean_time_to_mfe": float(np.nanmean([

                    s.ext.base.time_to_mfe or float("nan") for s in group

                ])),

                "mean_best_entry_delay": float(np.mean([s.deferred.consensus_best_delay for s in group])),

                "mean_entry_improvement": float(np.mean([s.deferred.mean_improvement for s in group])),

                "mean_f_baseline": float(np.mean([s.f_baseline for s in group])),

            }



        return {

            "case_a_dip_then_rise": bucket(case_a, "dip_then_rise"),

            "case_b_rise_then_fall": bucket(case_b, "rise_then_fall"),

            "hypothesis_check_note": (

                "Pre-specified Case A: I↓ D↑ and Case B: I↑ D? — report actual data, not assumed."

            ),

        }



    def _same_terminal_pairs(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        tol = self._config.terminal_match_tol

        bins: dict[int, list[DualAxisSample]] = {}

        for s in samples:

            key = int(round(s.ext.base.terminal_return / tol))

            bins.setdefault(key, []).append(s)



        pairs: list[dict[str, Any]] = []

        i_diff_count = 0

        d_diff_count = 0

        total = 0



        for group in bins.values():

            if len(group) < 2:

                continue

            for i in range(len(group)):

                for j in range(i + 1, len(group)):

                    a, b = group[i], group[j]

                    if abs(a.ext.base.terminal_return - b.ext.base.terminal_return) > tol:

                        continue

                    if abs(a.ext.base.early_mean_return - b.ext.base.early_mean_return) < 0.0002:

                        continue

                    total += 1

                    if abs(a.I_observable - b.I_observable) > 0.15:

                        i_diff_count += 1

                    if abs(a.D_observable - b.D_observable) > 1e-5:

                        d_diff_count += 1

                    if len(pairs) < 12:

                        pairs.append({

                            "t_a": a.t_index,

                            "t_b": b.t_index,

                            "terminal": a.ext.base.terminal_return,

                            "early_a": a.ext.base.early_mean_return,

                            "early_b": b.ext.base.early_mean_return,

                            "I_a": a.I_observable,

                            "I_b": b.I_observable,

                            "D_a": a.D_observable,

                            "D_b": b.D_observable,

                            "f_baseline_a": a.f_baseline,

                            "f_baseline_b": b.f_baseline,

                        })



        return {

            "total_pairs": total,

            "I_diff_rate": i_diff_count / total if total else float("nan"),

            "D_diff_rate": d_diff_count / total if total else float("nan"),

            "example_pairs": pairs,

        }



    def _timing_sensitivity(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        cfg = self._config

        agg: dict[int, list[float]] = {d: [] for d in cfg.entry_delays}

        cap_agg: dict[int, list[float]] = {d: [] for d in cfg.entry_delays}

        for s in samples[:800]:

            for d, metrics in s.delay_curve.items():

                if not np.isnan(metrics.get("mean_proxy_pnl", float("nan"))):

                    agg[d].append(metrics["mean_proxy_pnl"])

                if not np.isnan(metrics.get("captureability", float("nan"))):

                    cap_agg[d].append(metrics["captureability"])



        return {

            "by_delay_mean_proxy_pnl": {

                f"delay_{d}": float(np.mean(agg[d])) if agg[d] else float("nan")

                for d in cfg.entry_delays

            },

            "by_delay_captureability": {

                f"delay_{d}": float(np.mean(cap_agg[d])) if cap_agg[d] else float("nan")

                for d in cfg.entry_delays

            },

            "deferred_best_delay_mean": float(np.mean([s.deferred.consensus_best_delay for s in samples])),

        }



    def _policy_robustness(self, samples: list[DualAxisSample]) -> dict[str, Any]:

        return {

            "immediate_mean_robustness": float(np.mean([s.immediate.robustness for s in samples])),

            "deferred_mean_robustness": float(np.mean([s.deferred.deferred_robustness for s in samples])),

            "immediate_robustness_by_policy": self._policy_aggregate(samples, immediate=True),

            "deferred_improvement_by_policy": self._policy_aggregate(samples, immediate=False),

            "spearman_I_immediate_robustness": _spearman(

                [s.I_observable for s in samples],

                [s.immediate.robustness for s in samples],

            ),

            "spearman_D_deferred_robustness": _spearman(

                [s.D_observable for s in samples],

                [s.deferred.deferred_robustness for s in samples],

            ),

        }



    def _f_vs_id(self, samples: list[DualAxisSample], f_attr: str) -> dict[str, Any]:

        f_vals = [getattr(s, f_attr) for s in samples]

        i_vals = [s.I_observable for s in samples]

        d_vals = [s.D_observable for s in samples]

        top = sorted(samples, key=lambda s: getattr(s, f_attr))[-len(samples)//10:]

        bot = sorted(samples, key=lambda s: getattr(s, f_attr))[: len(samples)//10]

        return {

            "pearson_f_I": _pearson(f_vals, i_vals),

            "pearson_f_D": _pearson(f_vals, d_vals),

            "spearman_f_I": _spearman(f_vals, i_vals),

            "spearman_f_D": _spearman(f_vals, d_vals),

            "top_decile_mean_I": float(np.mean([s.I_observable for s in top])),

            "top_decile_mean_D": float(np.mean([s.D_observable for s in top])),

            "bottom_decile_mean_I": float(np.mean([s.I_observable for s in bot])),

            "bottom_decile_mean_D": float(np.mean([s.D_observable for s in bot])),

            "note": (

                "High F vs I correlation suggests F aligns with immediate capture; "

                "compare F vs D to see deferred information inclusion."

            ),

        }



    def _representative_paths(

        self, samples: list[DualAxisSample], i_med: float, d_med: float

    ) -> dict[str, Any]:

        buckets: dict[str, list[DualAxisSample]] = {

            "Ihigh_Dlow": [],

            "Ihigh_Dhigh": [],

            "Ilow_Dhigh": [],

            "Ilow_Dlow": [],

        }

        i_vals = np.array([s.I_observable for s in samples], dtype=float)
        d_vals = np.array([s.D_observable for s in samples], dtype=float)
        i_rank = np.argsort(np.argsort(i_vals))
        d_rank = np.argsort(np.argsort(d_vals))
        for s, i_r, d_r in zip(samples, i_rank, d_rank):
            i_hi = i_r >= len(samples) / 2
            d_hi = d_r >= len(samples) / 2
            key = (
                "Ihigh_Dlow" if i_hi and not d_hi else
                "Ihigh_Dhigh" if i_hi and d_hi else
                "Ilow_Dhigh" if not i_hi and d_hi else
                "Ilow_Dlow"
            )
            buckets[key].append(s)



        examples: dict[str, list[dict[str, Any]]] = {}

        for name, group in buckets.items():

            chosen = sorted(group, key=lambda s: abs(s.f_baseline))[:3]

            examples[name] = []

            for s in chosen:

                ctx = self._builder.build(s.t_index)

                rets = tuple(ctx.return_from_t(k) for k in range(1, ctx.reward_horizon + 1))

                examples[name].append({

                    "t_index": s.t_index,

                    "I": s.I_observable,

                    "D": s.D_observable,

                    "f_baseline": s.f_baseline,

                    "terminal": s.ext.base.terminal_return,

                    "best_entry_delay": s.deferred.consensus_best_delay,

                    "aligned_returns": list(rets),

                })

        return examples



    def _auto_interpretation(self, report: dict[str, Any], samples: list[DualAxisSample]) -> dict[str, Any]:

        confirmed: list[str] = []

        hypothesis: list[str] = []

        unresolved: list[str] = []

        next_exp: list[str] = []



        id_corr = report["id_correlation"]

        q = report["quadrant_analysis"]["quadrants"]

        q3_frac = report["quadrant_analysis"]["Q3_Ilow_Dhigh_fraction"]

        case_a = report["case_ab_analysis"]["case_a_dip_then_rise"]

        case_b = report["case_ab_analysis"]["case_b_rise_then_fall"]

        f_id = report["baseline_f_vs_id"]



        if q3_frac > 0.05:

            confirmed.append(

                f"Q3 (I-low/D-high) occupies {q3_frac:.1%} of samples — deferred opportunity "

                "exists independently of immediate capture for a non-trivial subset."

            )

        if case_a.get("count", 0) > 0:

            confirmed.append(

                f"Case A (n={case_a['count']}): mean I={case_a['mean_I']:.3f}, "

                f"mean D={case_a['mean_D']:.6f}, terminal={case_a['mean_terminal']:.4%}"

            )

        if case_b.get("count", 0) > 0:

            confirmed.append(

                f"Case B (n={case_b['count']}): mean I={case_b['mean_I']:.3f}, "

                f"mean D={case_b['mean_D']:.6f}, terminal={case_b['mean_terminal']:.4%}"

            )

        if abs(f_id["spearman_f_I"]) > 0.5:

            confirmed.append(

                f"Baseline F correlates with Immediate (spearman={f_id['spearman_f_I']:.3f}) "

                "more strongly than typical random pairing."

            )

        if abs(f_id["spearman_f_D"]) < abs(f_id["spearman_f_I"]):

            hypothesis.append(

                "Baseline F may encode Immediate axis more than Deferred axis "

                f"(|ρ(F,I)|={abs(f_id['spearman_f_I']):.3f} vs |ρ(F,D)|={abs(f_id['spearman_f_D']):.3f})."

            )



        if not np.isnan(id_corr["spearman_I_D"]):

            hypothesis.append(

                f"I-D spearman={id_corr['spearman_I_D']:.3f} — partial overlap exists; "

                "quadrant structure still required for interpretation."

            )



        unresolved.extend([

            "Canonical scalar vs multi-head P1 target — NOT decided by this experiment",

            "Which execution policy family represents P2 — still unresolved",

            "Single BTC ~10d sample — generalization unknown",

            "Deferred best-entry objective varies by policy — no single hindsight ground truth",

        ])

        next_exp.extend([

            "Case A dedicated: quantify gap between F(t) and best-entry-within-window (analysis only)",

            "Expand policy grid + volatility regimes; cluster robustness profiles",

            "Multi-asset (ETH, equities) I/D quadrant stability",

            "Same-terminal pair path visualization (15+ pairs)",

            "Dual-axis label stability before any P1 target formula discussion",

        ])



        return {

            "confirmed": confirmed,

            "hypothesis": hypothesis,

            "unresolved": unresolved,

            "next_experiments": next_exp,

        }





def generate_plots(report: dict[str, Any], output_dir: str) -> list[str]:

    """Optional matplotlib plots — skipped if matplotlib unavailable."""

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        return []



    from pathlib import Path



    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []



    scatter = report.get("scatter_sample", [])
    if scatter:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter([p["I"] for p in scatter], [p["D"] for p in scatter], alpha=0.35, s=12)
        ax.set_xlabel("Immediate (I)")
        ax.set_ylabel("Deferred (D)")
        ax.set_title("I vs D scatter (subsample)")
        p = out_dir / "id_scatter.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    reps = report.get("representative_paths", {})

    all_pts: list[tuple[float, float, str]] = []

    for name, items in reps.items():

        for item in items:

            all_pts.append((item["I"], item["D"], name))



    if all_pts:

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = {"Ihigh_Dlow": "C0", "Ihigh_Dhigh": "C1", "Ilow_Dhigh": "C2", "Ilow_Dlow": "C3"}

        for i, d, name in all_pts:

            ax.scatter(i, d, c=colors.get(name, "gray"), alpha=0.7, s=30, label=name)

        handles, labels = ax.get_legend_handles_labels()

        by_label = dict(zip(labels, handles))

        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        ax.set_xlabel("Immediate (I)")

        ax.set_ylabel("Deferred (D)")

        ax.set_title("Dual-axis sample points (representative subset)")

        p = out_dir / "id_scatter_representative.png"

        fig.savefig(p, dpi=120, bbox_inches="tight")

        plt.close(fig)

        saved.append(str(p))



    qa = report.get("quadrant_analysis", {}).get("quadrants", {})

    if qa:

        fig, ax = plt.subplots(figsize=(8, 5))

        names = list(qa.keys())

        counts = [qa[n].get("count", 0) for n in names]

        ax.bar(names, counts)

        ax.set_ylabel("Count")

        ax.set_title("I/D quadrant distribution")

        plt.xticks(rotation=20, ha="right")

        p = out_dir / "quadrant_distribution.png"

        fig.savefig(p, dpi=120, bbox_inches="tight")

        plt.close(fig)

        saved.append(str(p))



    ts = report.get("timing_sensitivity", {})

    cap = ts.get("by_delay_captureability", {})

    if cap:

        fig, ax = plt.subplots(figsize=(8, 5))

        xs = [int(k.split("_")[1]) for k in cap]

        ys = [cap[k] for k in sorted(cap, key=lambda x: int(x.split("_")[1]))]

        ax.plot(xs, ys, marker="o")

        ax.set_xlabel("Entry delay (bars)")

        ax.set_ylabel("Mean captureability")

        ax.set_title("Immediate captureability vs entry delay")

        p = out_dir / "captureability_vs_delay.png"

        fig.savefig(p, dpi=120, bbox_inches="tight")

        plt.close(fig)

        saved.append(str(p))



    return saved





__all__ = ["DualAxisAnalysisRunner", "DualAxisAnalysisConfig", "save_report", "generate_plots"]


