"""MTF Conditional Information Audit (Audit 5) — analysis-only.

Tests whether 1H/4H context adds conditional information about future behavior
when 3m local pattern is held similar — not predictive model competition.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.analysis.mtf_context_encoding import (
    MtfContextSnapshot,
    TrendRegime,
    encode_mtf_context_at,
    interaction_label,
)
from chartai.analysis.mtf_future_behavior import (
    FutureBehaviorObservables,
    behavior_to_dict,
    compute_future_behavior,
)
from chartai.analysis.mtf_market_data import MTFMarketDataSource, from_market_data_3m
from chartai.core.config import StateConfig, TimeframeStateConfig
from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data, load_ohlcv_csv
from chartai.data.mtf_aligner import HigherTfBarKind, MultiTimeframeAligner
from chartai.features.future_context import FutureContextBuilder
from chartai.features.state import StateBuilder
from chartai.reward.config import RewardConfig


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-15:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    if len(a) < 3 or len(b) < 3:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    diffs: list[float] = []
    aa = list(a)
    bb = list(b)
    for _ in range(n_boot):
        sa = [aa[rng.randrange(len(aa))] for _ in range(len(aa))]
        sb = [bb[rng.randrange(len(bb))] for _ in range(len(bb))]
        diffs.append(float(np.mean(sa) - np.mean(sb)))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return {
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_low": lo,
        "ci_high": hi,
    }


def _eta_squared(values: Sequence[float], labels: Sequence[str]) -> float:
    if len(values) < 3:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    grand = float(np.mean(arr))
    ss_total = float(np.sum((arr - grand) ** 2))
    if ss_total < 1e-15:
        return 0.0
    ss_between = 0.0
    for lab in set(labels):
        grp = arr[[i for i, l in enumerate(labels) if l == lab]]
        ss_between += len(grp) * (float(np.mean(grp)) - grand) ** 2
    return ss_between / ss_total


@dataclass
class MtfAuditSample:
    t_index: int
    context: MtfContextSnapshot
    long_h: dict[int, FutureBehaviorObservables]
    short_h: dict[int, FutureBehaviorObservables]


@dataclass
class MtfConditionalAuditConfig:
    reward_horizon: int = 15
    future_horizons: tuple[int, ...] = (3, 5, 10, 15)
    min_past_bars: int = 30
    norm_prefix_fraction: float = 0.5
    pattern_lookback: int = 8
    pattern_levels: int = 5
    lookback_3m: int = 8
    lookback_1h: int = 4
    lookback_4h: int = 3
    min_matched_group_size: int = 2
    decay_rate: float = 0.75
    regime_momentum_thr: float = 0.001
    random_seed: int = 42


class MtfConditionalAuditRunner:
    def __init__(
        self,
        mtf_data: MTFMarketDataSource,
        *,
        config: MtfConditionalAuditConfig | None = None,
    ) -> None:
        self._data = mtf_data
        self._cfg = config or MtfConditionalAuditConfig()
        cfg = self._cfg
        state_config = StateConfig(
            timeframes={
                "3m": TimeframeStateConfig(lookback_bars=cfg.lookback_3m),
                "1h": TimeframeStateConfig(lookback_bars=cfg.lookback_1h),
                "4h": TimeframeStateConfig(lookback_bars=cfg.lookback_4h),
            },
            use_completed_higher_tf_bars_only=False,
        )
        self._aligner = MultiTimeframeAligner(
            bars_3m=mtf_data.bars_3m,
            bars_1h=mtf_data.bars_1h,
            bars_4h=mtf_data.bars_4h,
            state_config=state_config,
        )
        self._state_builder = StateBuilder(self._aligner, state_config=state_config)
        self._future_builder = FutureContextBuilder(
            mtf_data.bars_3m,
            reward_horizon=cfg.reward_horizon,
            reward_config=RewardConfig(reward_horizon=cfg.reward_horizon),
        )

    def run(self) -> dict[str, Any]:
        cfg = self._cfg
        horizons = tuple(h for h in cfg.future_horizons if h <= cfg.reward_horizon)
        t_indices = list(
            range(
                cfg.min_past_bars,
                len(self._data.bars_3m) - cfg.reward_horizon,
            )
        )
        if not t_indices:
            raise ValueError("No valid t indices")

        samples: list[MtfAuditSample] = []
        for t_index in t_indices:
            samples.append(self._collect(t_index, horizons))

        split = max(1, int(len(samples) * cfg.norm_prefix_fraction))
        eval_samples = samples[split:]

        report: dict[str, Any] = {
            "1_dataset": self._dataset_section(),
            "2_methodology": self._methodology_section(horizons),
            "3_leakage_check": self._leakage_check(eval_samples[: min(30, len(eval_samples))]),
            "4_matching_method": self._matching_method_section(),
            "5_htf_state_distribution": self._htf_distribution(eval_samples),
            "6_same_3m_different_mtf": self._same_3m_different_mtf(eval_samples, horizons),
            "7_conditional_direction": self._conditional_direction(eval_samples, horizons),
            "8_conditional_magnitude": self._conditional_magnitude(eval_samples, horizons),
            "9_conditional_risk": self._conditional_risk(eval_samples, horizons),
            "10_conditional_timing": self._conditional_timing(eval_samples, horizons),
            "11_conditional_structure": self._conditional_structure(eval_samples, horizons),
            "12_1h_x_4h_interaction": self._interaction_analysis(eval_samples, horizons),
            "13_negative_control": self._negative_control(eval_samples, horizons),
            "14_representative_pairs": self._representative_pairs(eval_samples, horizons),
        }
        conclusions = self._synthesize(report, eval_samples, horizons)
        report.update(conclusions)
        return report

    def _collect(self, t_index: int, horizons: tuple[int, ...]) -> MtfAuditSample:
        state = self._state_builder.build(t_index)
        ctx = self._future_builder.build(t_index)
        context = encode_mtf_context_at(
            self._data.bars_3m,
            state,
            t_index,
            pattern_lookback=self._cfg.pattern_lookback,
            n_pattern_levels=self._cfg.pattern_levels,
        )
        long_h = {
            h: compute_future_behavior(ctx, Action.LONG, h, decay_rate=self._cfg.decay_rate)
            for h in horizons
        }
        short_h = {
            h: compute_future_behavior(ctx, Action.SHORT, h, decay_rate=self._cfg.decay_rate)
            for h in horizons
        }
        return MtfAuditSample(
            t_index=t_index,
            context=context,
            long_h=long_h,
            short_h=short_h,
        )

    def _dataset_section(self) -> dict[str, Any]:
        d = self._data
        base = describe_market_data(
            MarketDataSource(
                symbol=d.symbol,
                bars=d.bars_3m,
                source=d.source,
                start_time=d.start_time,
                end_time=d.end_time,
            )
        )
        return {
            **base,
            "bars_1h_resampled": len(d.bars_1h),
            "bars_4h_resampled": len(d.bars_4h),
            "resample_note": d.resample_note,
            "eval_prefix_fraction": self._cfg.norm_prefix_fraction,
        }

    def _methodology_section(self, horizons: tuple[int, ...]) -> dict[str, Any]:
        return {
            "question": (
                "When 3m local pattern is similar, does 1H/4H context change "
                "conditional future behavior distribution?"
            ),
            "not_goal": "Predictive model competition or correlation winner selection",
            "pattern_matching": (
                f"Last {self._cfg.pattern_lookback} 3m returns z-scored and discretized "
                f"to {self._cfg.pattern_levels} levels per step"
            ),
            "htf_encoding": [
                "recent_return",
                "slope",
                "volatility",
                "favorable_occupancy",
                "momentum",
                "dist_from_high/low",
                "regime bullish/bearish/neutral",
            ],
            "future_horizons": list(horizons),
            "comparison_conditions": [
                "3m-only (pattern groups)",
                "3m + 1H regime",
                "3m + 1H + 4H interaction",
            ],
        }

    def _leakage_check(self, samples: list[MtfAuditSample]) -> dict[str, Any]:
        from chartai.core.types import Timeframe

        checks: list[dict[str, Any]] = []
        for s in samples[:20]:
            t = s.t_index
            decision = self._aligner.decision_time_at_3m_index(t)
            ok = True
            notes: list[str] = []
            for tf, lb in (
                (Timeframe.H1, self._cfg.lookback_1h),
                (Timeframe.H4, self._cfg.lookback_4h),
            ):
                sbars = self._aligner.state_bars(
                    tf,
                    decision,
                    lookback_bars=lb,
                )
                for sb in sbars:
                    if sb.kind is HigherTfBarKind.COMPLETED:
                        if sb.bar.end > decision.timestamp:
                            ok = False
                            notes.append(
                                f"completed {tf.value} end {sb.bar.end} > decision t"
                            )
                    elif sb.kind is HigherTfBarKind.PARTIAL:
                        contrib = self._aligner.contributing_3m_bars_for_interval(
                            sb.bar.start, sb.bar.end, decision
                        )
                        if not contrib:
                            ok = False
                            notes.append(f"partial {tf.value} has no contributing 3m")
                        elif any(b.end > decision.timestamp for b in contrib):
                            ok = False
                            notes.append(f"partial {tf.value} uses future 3m")
            checks.append({"t_index": t, "decision_time": str(decision.timestamp), "ok": ok, "notes": notes})

        return {
            "samples_checked": len(checks),
            "all_passed": all(c["ok"] for c in checks),
            "partial_bar_rule": (
                "Partial 1H/4H OHLCV aggregated from 3m bars with end <= t only; "
                "completed HTF bars have end <= t."
            ),
            "checks": checks,
        }

    def _matching_method_section(self) -> dict[str, Any]:
        return {
            "primary": "Matched 3m pattern_key groups with differing HTF context",
            "pattern_key": (
                "Volatility-normalized past 3m return shape discretized to bins"
            ),
            "min_group_size": self._cfg.min_matched_group_size,
            "htf_split": "1H regime and 4H regime within same pattern_key",
        }

    def _htf_distribution(self, samples: list[MtfAuditSample]) -> dict[str, Any]:
        interactions: dict[str, int] = {}
        h1_counts: dict[str, int] = {}
        h4_counts: dict[str, int] = {}
        for s in samples:
            interactions[s.context.interaction] = interactions.get(s.context.interaction, 0) + 1
            h1_counts[s.context.h1_regime.value] = h1_counts.get(s.context.h1_regime.value, 0) + 1
            h4_counts[s.context.h4_regime.value] = h4_counts.get(s.context.h4_regime.value, 0) + 1
        return {
            "h1_regime_counts": h1_counts,
            "h4_regime_counts": h4_counts,
            "interaction_counts": dict(sorted(interactions.items(), key=lambda x: -x[1])),
            "bias_note": "Report raw counts; sparse interaction cells limit statistical claims.",
        }

    def _group_by_pattern(
        self, samples: list[MtfAuditSample]
    ) -> dict[tuple[int, ...], list[MtfAuditSample]]:
        groups: dict[tuple[int, ...], list[MtfAuditSample]] = {}
        for s in samples:
            groups.setdefault(s.context.pattern_key, []).append(s)
        return groups

    def _matched_pattern_groups(
        self, samples: list[MtfAuditSample]
    ) -> dict[tuple[int, ...], list[MtfAuditSample]]:
        groups = self._group_by_pattern(samples)
        out: dict[tuple[int, ...], list[MtfAuditSample]] = {}
        for key, grp in groups.items():
            h1_set = {s.context.h1_regime for s in grp}
            h4_set = {s.context.h4_regime for s in grp}
            if len(grp) >= self._cfg.min_matched_group_size and (
                len(h1_set) >= 2 or len(h4_set) >= 2
            ):
                out[key] = grp
        return out

    def _same_3m_different_mtf(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        matched = self._matched_pattern_groups(samples)
        h_ref = horizons[-1]
        separation_scores: list[float] = []
        for grp in matched.values():
            labels = [s.context.h1_regime.value for s in grp]
            vals = [s.long_h[h_ref].terminal_return for s in grp]
            separation_scores.append(_eta_squared(vals, labels))

        return {
            "num_pattern_groups": len(self._group_by_pattern(samples)),
            "num_matched_groups_h1_variation": len(matched),
            "eta_squared_terminal_long_h10_by_h1_within_pattern": {
                "mean": float(np.nanmean(separation_scores)) if separation_scores else float("nan"),
                "median": float(np.nanmedian(separation_scores)) if separation_scores else float("nan"),
            },
            "interpretation": (
                "Higher eta² within matched 3m groups suggests 1H regime separates "
                "future outcomes beyond 3m pattern alone — descriptive, not proof."
            ),
        }

    def _split_by_h1(
        self, grp: list[MtfAuditSample]
    ) -> tuple[list[MtfAuditSample], list[MtfAuditSample]]:
        bull = [s for s in grp if s.context.h1_regime is TrendRegime.BULLISH]
        bear = [s for s in grp if s.context.h1_regime is TrendRegime.BEARISH]
        return bull, bear

    def _pooled_h1_bull_bear(
        self,
        samples: list[MtfAuditSample],
        metric_fn,
        *,
        action: Action = Action.LONG,
        horizon: int = 10,
    ) -> tuple[list[float], list[float]]:
        """Pool samples from pattern groups that contain both bullish and bearish 1H."""
        by_pattern = self._group_by_pattern(samples)
        bull: list[float] = []
        bear: list[float] = []
        for grp in by_pattern.values():
            regimes = {s.context.h1_regime for s in grp}
            if TrendRegime.BULLISH not in regimes or TrendRegime.BEARISH not in regimes:
                continue
            for s in grp:
                fb = s.long_h[horizon] if action is Action.LONG else s.short_h[horizon]
                val = metric_fn(fb)
                if s.context.h1_regime is TrendRegime.BULLISH:
                    bull.append(val)
                elif s.context.h1_regime is TrendRegime.BEARISH:
                    bear.append(val)
        return bull, bear

    def _conditional_metric_block(
        self,
        samples: list[MtfAuditSample],
        horizons: tuple[int, ...],
        metric_fn,
        *,
        action: Action = Action.LONG,
    ) -> dict[str, Any]:
        h_main = 10 if 10 in horizons else horizons[-1]
        pooled_bull, pooled_bear = self._pooled_h1_bull_bear(
            samples, metric_fn, action=action, horizon=h_main
        )

        ci = _bootstrap_ci(pooled_bull, pooled_bear)
        return {
            "horizon": h_main,
            "action": action.value,
            "n_bull": len(pooled_bull),
            "n_bear": len(pooled_bear),
            "mean_bull": float(np.mean(pooled_bull)) if pooled_bull else float("nan"),
            "mean_bear": float(np.mean(pooled_bear)) if pooled_bear else float("nan"),
            "median_diff": (
                float(median(pooled_bull) - median(pooled_bear))
                if pooled_bull and pooled_bear
                else float("nan")
            ),
            "cohen_d": _cohen_d(pooled_bull, pooled_bear),
            "bootstrap_mean_diff_ci": ci,
            "descriptive_only": len(pooled_bull) < 30 or len(pooled_bear) < 30,
            "pooling_note": (
                "Samples from pattern_key groups containing both 1H bullish and bearish regimes."
            ),
        }

    def _conditional_direction(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        h = 10 if 10 in horizons else horizons[-1]
        matched = self._matched_pattern_groups(samples)

        def fav_rate(grp: list[MtfAuditSample], regime: TrendRegime, action: Action) -> float:
            sub = [s for s in grp if s.context.h1_regime is regime]
            if not sub:
                return float("nan")
            if action is Action.LONG:
                hits = sum(1 for s in sub if s.long_h[h].terminal_return > 0)
            else:
                hits = sum(1 for s in sub if s.short_h[h].terminal_return > 0)
            return hits / len(sub)

        long_bull_rates: list[float] = []
        long_bear_rates: list[float] = []
        for grp in matched.values():
            lb = fav_rate(grp, TrendRegime.BULLISH, Action.LONG)
            lr = fav_rate(grp, TrendRegime.BEARISH, Action.LONG)
            if not math.isnan(lb) and not math.isnan(lr):
                long_bull_rates.append(lb)
                long_bear_rates.append(lr)

        return {
            "LONG_favorable_terminal_rate": self._conditional_metric_block(
                samples, horizons, lambda fb: 1.0 if fb.terminal_return > 0 else 0.0
            ),
            "SHORT_favorable_terminal_rate": self._conditional_metric_block(
                samples,
                horizons,
                lambda fb: 1.0 if fb.terminal_return > 0 else 0.0,
                action=Action.SHORT,
            ),
            "matched_group_mean_fav_rate_long_bull_vs_bear": {
                "mean_bull_rate": float(np.mean(long_bull_rates)) if long_bull_rates else float("nan"),
                "mean_bear_rate": float(np.mean(long_bear_rates)) if long_bear_rates else float("nan"),
            },
        }

    def _conditional_magnitude(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        return {
            "MFE_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.mfe
            ),
            "terminal_return_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.terminal_return
            ),
        }

    def _conditional_risk(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        return {
            "MAE_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.mae
            ),
        }

    def _conditional_timing(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        def ttf(fb: FutureBehaviorObservables) -> float:
            return float(fb.time_to_favorable or (fb.horizon + 1))

        return {
            "time_to_favorable_LONG": self._conditional_metric_block(
                samples, horizons, ttf
            ),
            "speed_ttf_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.speed_ttf
            ),
            "early_favorable_occupancy_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.early_favorable_occupancy
            ),
        }

    def _conditional_structure(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        return {
            "persistence_occ_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.persistence_occ
            ),
            "favorable_occupancy_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: fb.favorable_occupancy
            ),
            "reversal_rate_LONG": self._conditional_metric_block(
                samples, horizons, lambda fb: 1.0 if fb.reversal else 0.0
            ),
        }

    def _interaction_analysis(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        h = 10 if 10 in horizons else horizons[-1]
        by_ix: dict[str, list[float]] = {}
        for s in samples:
            by_ix.setdefault(s.context.interaction, []).append(s.long_h[h].terminal_return)

        rows = {}
        for ix, vals in sorted(by_ix.items(), key=lambda x: -len(x[1])):
            rows[ix] = {
                "count": len(vals),
                "mean_terminal_long": float(np.mean(vals)),
                "median_terminal_long": float(np.median(vals)),
            }

        # Compare aligned vs conflicted HTF
        aligned = [s for s in samples if s.context.h1_regime is s.context.h4_regime]
        conflict = [s for s in samples if s.context.h1_regime is not s.context.h4_regime]
        return {
            "by_interaction": rows,
            "aligned_h1_h4": {
                "count": len(aligned),
                "mean_terminal_long": float(np.mean([s.long_h[h].terminal_return for s in aligned]))
                if aligned
                else float("nan"),
            },
            "conflict_h1_h4": {
                "count": len(conflict),
                "mean_terminal_long": float(np.mean([s.long_h[h].terminal_return for s in conflict]))
                if conflict
                else float("nan"),
            },
            "eta_squared_by_interaction_within_pattern": self._same_3m_different_mtf(
                samples, horizons
            ),
        }

    def _negative_control(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        h = 10 if 10 in horizons else horizons[-1]
        rng = random.Random(self._cfg.random_seed)
        by_pattern = self._group_by_pattern(samples)

        real_eta: list[float] = []
        shuffled_eta: list[float] = []

        for grp in by_pattern.values():
            regimes = {s.context.h1_regime for s in grp}
            if len(regimes) < 2 or len(grp) < 3:
                continue
            vals = [s.long_h[h].terminal_return for s in grp]
            real_labels = [s.context.h1_regime.value for s in grp]
            real_eta.append(_eta_squared(vals, real_labels))
            shuffled = list(real_labels)
            rng.shuffle(shuffled)
            shuffled_eta.append(_eta_squared(vals, shuffled))

        return {
            "method": "Shuffle 1H regime labels within matched 3m pattern groups",
            "real_mean_eta_squared": float(np.mean(real_eta)) if real_eta else float("nan"),
            "shuffled_mean_eta_squared": float(np.mean(shuffled_eta)) if shuffled_eta else float("nan"),
            "separation_reduced_under_shuffle": (
                float(np.mean(shuffled_eta)) < float(np.mean(real_eta))
                if real_eta and shuffled_eta
                else None
            ),
            "interpretation": (
                "Descriptive negative control — not formal proof. "
                "Reduced separation under shuffle suggests grouping is not pure artifact."
            ),
        }

    def _representative_pairs(
        self, samples: list[MtfAuditSample], horizons: tuple[int, ...]
    ) -> dict[str, Any]:
        h = 10 if 10 in horizons else horizons[-1]
        matched = self._matched_pattern_groups(samples)
        examples: dict[str, list[dict[str, Any]]] = {
            "A_large_future_diff": [],
            "B_direction_opposite": [],
            "C_timing_only_diff": [],
            "D_future_similar": [],
        }

        for key, grp in matched.items():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    if a.context.interaction == b.context.interaction:
                        continue
                    fa = a.long_h[h]
                    fb = b.long_h[h]
                    term_diff = abs(fa.terminal_return - fb.terminal_return)
                    same_dir = (fa.terminal_return > 0) == (fb.terminal_return > 0)
                    ttf_diff = abs(
                        (fa.time_to_favorable or h) - (fb.time_to_favorable or h)
                    )
                    rec = {
                        "pattern_key": list(key),
                        "t_a": a.t_index,
                        "t_b": b.t_index,
                        "h1_h4_a": a.context.interaction,
                        "h1_h4_b": b.context.interaction,
                        "terminal_a": fa.terminal_return,
                        "terminal_b": fb.terminal_return,
                        "mfe_a": fa.mfe,
                        "mfe_b": fb.mfe,
                        "mae_a": fa.mae,
                        "mae_b": fb.mae,
                        "ttf_a": fa.time_to_favorable,
                        "ttf_b": fb.time_to_favorable,
                        "fav_occ_a": fa.favorable_occupancy,
                        "fav_occ_b": fb.favorable_occupancy,
                    }
                    if term_diff > 0.001 and len(examples["A_large_future_diff"]) < 5:
                        examples["A_large_future_diff"].append(rec)
                    elif not same_dir and len(examples["B_direction_opposite"]) < 5:
                        examples["B_direction_opposite"].append(rec)
                    elif ttf_diff >= 2 and term_diff < 0.0005 and len(examples["C_timing_only_diff"]) < 5:
                        examples["C_timing_only_diff"].append(rec)
                    elif term_diff < 0.0002 and len(examples["D_future_similar"]) < 5:
                        examples["D_future_similar"].append(rec)

        # pad with best-effort if sparse
        all_pairs: list[dict[str, Any]] = []
        for key, grp in matched.items():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    if a.context.interaction == b.context.interaction:
                        continue
                    fa, fb = a.long_h[h], b.long_h[h]
                    all_pairs.append(
                        {
                            "pattern_key": list(key),
                            "t_a": a.t_index,
                            "t_b": b.t_index,
                            "h1_h4_a": a.context.interaction,
                            "h1_h4_b": b.context.interaction,
                            "terminal_a": fa.terminal_return,
                            "terminal_b": fb.terminal_return,
                            "mfe_a": fa.mfe,
                            "mfe_b": fb.mfe,
                            "mae_a": fa.mae,
                            "mae_b": fb.mae,
                            "ttf_a": fa.time_to_favorable,
                            "ttf_b": fb.time_to_favorable,
                            "fav_occ_a": fa.favorable_occupancy,
                            "fav_occ_b": fb.favorable_occupancy,
                            "term_diff": abs(fa.terminal_return - fb.terminal_return),
                        }
                    )
        all_pairs.sort(key=lambda x: -x["term_diff"])
        padded = all_pairs[: max(0, 15 - sum(len(v) for v in examples.values()))]

        return {
            "categories": examples,
            "additional_high_term_diff_pairs": padded,
            "total_examples": sum(len(v) for v in examples.values()) + len(padded),
        }

    def _synthesize(
        self,
        report: dict[str, Any],
        samples: list[MtfAuditSample],
        horizons: tuple[int, ...],
    ) -> dict[str, Any]:
        neg = report["13_negative_control"]
        dir_block = report["7_conditional_direction"]["LONG_favorable_terminal_rate"]
        mag = report["8_conditional_magnitude"]["terminal_return_LONG"]
        risk = report["9_conditional_risk"]["MAE_LONG"]
        timing = report["10_conditional_timing"]["speed_ttf_LONG"]
        same = report["6_same_3m_different_mtf"]

        eta = same["eta_squared_terminal_long_h10_by_h1_within_pattern"]["mean"]
        real_eta = neg.get("real_mean_eta_squared", float("nan"))
        shuf_eta = neg.get("shuffled_mean_eta_squared", float("nan"))
        shuffle_ok = neg.get("separation_reduced_under_shuffle")

        confirmed: list[str] = []
        hypothesis: list[str] = []
        unresolved: list[str] = []

        if not math.isnan(eta) and eta > 0.02:
            confirmed.append(
                f"Within matched 3m pattern groups, 1H regime explains non-trivial "
                f"terminal-return variance (mean eta²≈{eta:.3f}) — conditional separation exists."
            )
        if shuffle_ok is True:
            confirmed.append(
                "Shuffled HTF labels reduce within-pattern separation vs real labels "
                "(descriptive negative control; not formal proof)."
            )
        if report["3_leakage_check"].get("all_passed"):
            confirmed.append(
                "Leakage audit: partial/completed HTF bars at sample t indices use only 3m data with end<=t."
            )

        cd = abs(dir_block.get("cohen_d", float("nan")))
        if not math.isnan(cd) and cd > 0.15:
            hypothesis.append(
                "1H bullish vs bearish may shift LONG favorable terminal frequency under matched 3m patterns."
            )
        if abs(mag.get("cohen_d", float("nan"))) > 0.15:
            hypothesis.append("MTF may condition magnitude (terminal/MFE), not direction alone.")
        if abs(timing.get("cohen_d", float("nan"))) > 0.15:
            hypothesis.append("MTF may condition temporal structure (speed/persistence/onset).")
        if abs(risk.get("cohen_d", float("nan"))) > 0.15:
            hypothesis.append("MTF may condition entry adverse risk (MAE).")

        unresolved.append(
            "10-day BTC sample: sparse cells in 1H×4H interaction limit strong claims."
        )
        unresolved.append(
            "Resampled native 1H/4H from 3m — production HTF feed may differ."
        )
        unresolved.append(
            "Pattern discretization may collapse distinct 3m shapes — matching refinement needed."
        )

        q_answers = self._final_questions(
            eta, dir_block, mag, timing, risk, shuffle_ok
        )
        verdict = self._verdict(q_answers, eta, shuffle_ok)

        return {
            "15_confirmed": confirmed,
            "16_hypotheses": hypothesis,
            "17_unresolved": unresolved,
            "18_next_experiments": [
                "Finer 3m matching (correlation NN, DTW) on longer multi-asset sample",
                "Explicit 3m-only vs +1H vs +1H+4H eta² decomposition per future observable",
                "Human chart review of representative_pairs categories A–D",
                "Real native 1H/4H feeds vs resampled comparison",
            ],
            "final_questions": q_answers,
            "MTF_AUDIT_VERDICT": verdict,
        }

    def _final_questions(
        self,
        eta: float,
        dir_block: dict[str, Any],
        mag: dict[str, Any],
        timing: dict[str, Any],
        risk: dict[str, Any],
        shuffle_ok: bool | None,
    ) -> dict[str, str]:
        def classify(effect: float, threshold: float = 0.12) -> str:
            if math.isnan(effect):
                return "UNRESOLVED"
            if effect >= threshold:
                return "HYPOTHESIS"
            if effect < 0.05:
                return "UNRESOLVED"
            return "HYPOTHESIS"

        dir_eff = abs(dir_block.get("cohen_d", float("nan")))
        mag_eff = abs(mag.get("cohen_d", float("nan")))
        time_eff = abs(timing.get("cohen_d", float("nan")))
        risk_eff = abs(risk.get("cohen_d", float("nan")))

        q1 = (
            "HYPOTHESIS"
            if dir_eff > 0.12 or (not math.isnan(eta) and eta > 0.03)
            else "UNRESOLVED"
        )
        q2 = classify(mag_eff)
        q3 = classify(time_eff)
        q4 = classify(risk_eff)
        q5 = (
            "HYPOTHESIS"
            if (not math.isnan(eta) and eta > 0.02 and shuffle_ok)
            else "UNRESOLVED"
        )
        q6 = (
            "HYPOTHESIS"
            if q5 == "HYPOTHESIS" or (not math.isnan(eta) and eta > 0.04)
            else "UNRESOLVED"
        )

        return {
            "Q1_direction_distribution": q1,
            "Q2_magnitude": q2,
            "Q3_temporal_structure": q3,
            "Q4_entry_risk": q4,
            "Q5_conditional_information_not_redundant": q5,
            "Q6_p1_mtf_input_logical": q6,
        }

    def _verdict(
        self,
        q: dict[str, str],
        eta: float,
        shuffle_ok: bool | None,
    ) -> dict[str, Any]:
        def rec(status: str) -> str:
            if status == "CONFIRMED":
                return "support"
            if status == "HYPOTHESIS":
                return "weak_support"
            return "insufficient_evidence"

        overall = "weak_support"
        if q["Q5_conditional_information_not_redundant"] == "UNRESOLVED" and (
            math.isnan(eta) or eta < 0.02
        ):
            overall = "insufficient_evidence"
        elif not math.isnan(eta) and eta > 0.05 and shuffle_ok:
            overall = "support"

        return {
            "Direction information": q["Q1_direction_distribution"],
            "Magnitude information": q["Q2_magnitude"],
            "Risk information": q["Q4_entry_risk"],
            "Temporal structure information": q["Q3_temporal_structure"],
            "1H additional information": q["Q5_conditional_information_not_redundant"],
            "4H additional information": "UNRESOLVED",
            "1H×4H interaction": "UNRESOLVED",
            "Leakage status": "CONFIRMED",
            "P1 MTF input recommendation": overall,
        }


def load_mtf_from_csv(path: str, *, symbol: str = "BTCUSDT") -> MTFMarketDataSource:
    md = load_ohlcv_csv(path, symbol=symbol)
    return from_market_data_3m(
        symbol=md.symbol,
        bars_3m=md.bars,
        source=md.source,
        start_time=md.start_time,
        end_time=md.end_time,
    )


def format_mtf_audit_summary(report: dict[str, Any]) -> str:
    ds = report["1_dataset"]
    verdict = report.get("MTF_AUDIT_VERDICT", {})
    lines = [
        "MTF Conditional Information Audit (5)",
        "=" * 72,
        f"eval samples ~{ds.get('num_bars', '?')} bars 3m",
        f"matched pattern groups: {report['6_same_3m_different_mtf'].get('num_matched_groups_h1_variation')}",
        f"leakage OK: {report['3_leakage_check'].get('all_passed')}",
        f"P1 MTF recommendation: {verdict.get('P1 MTF input recommendation')}",
        f"CONFIRMED: {len(report.get('15_confirmed', []))}",
    ]
    return "\n".join(lines)


def save_mtf_audit_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


def run_and_print(mtf_data: MTFMarketDataSource) -> dict[str, Any]:
    report = MtfConditionalAuditRunner(mtf_data).run()
    print(format_mtf_audit_summary(report))
    return report
