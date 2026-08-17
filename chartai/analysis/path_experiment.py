"""Path variant comparison on real 3m market data with causal normalization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence

import numpy as np

from chartai.core.types import Action, OHLCVBar
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import (
    CausalPrefixNormalizer,
    FittedRobustZScoreNormalizer,
    FittedZScoreNormalizer,
)
from chartai.reward.path_variants import PathVariant, compute_path_n_variant
from chartai.reward.utility import compute_utility_n


class FAblation(str, Enum):
    FULL = "P+U-MAE"
    PU = "P+U"
    UM = "U-MAE"
    PM = "P-MAE"


@dataclass
class HorizonArrays:
    """Per-horizon stacked arrays for one Path variant and one action."""

    p: np.ndarray  # shape (num_samples, horizon)
    u: np.ndarray
    mae: np.ndarray


@dataclass
class VariantSampleRecord:
    t_index: int
    f_long: float
    f_short: float
    horizon_return: float  # signed return at n=10 for LONG
    p_long: tuple[float, ...]
    u_long: tuple[float, ...]
    mae_long: tuple[float, ...]
    p_short: tuple[float, ...]
    u_short: tuple[float, ...]
    mae_short: tuple[float, ...]


def _percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _distribution_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    abs_arr = np.abs(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q95": _percentile(arr, 95),
        "q99": _percentile(arr, 99),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "extreme_freq_3sigma": float(np.mean(abs_arr > 3 * (np.std(arr) + 1e-12))),
    }


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _compose_f(
    p_n: float,
    u_n: float,
    mae_n: float,
    normalizer: FittedZScoreNormalizer | CausalPrefixNormalizer,
    ablation: FAblation,
) -> float:
    p_z = normalizer.normalize_path(p_n)
    u_z = normalizer.normalize_utility(u_n)
    m_z = normalizer.normalize_mae(mae_n)
    if ablation is FAblation.FULL:
        return p_z + u_z - m_z
    if ablation is FAblation.PU:
        return p_z + u_z
    if ablation is FAblation.UM:
        return u_z - m_z
    if ablation is FAblation.PM:
        return p_z - m_z
    raise ValueError(ablation)


def _collect_raw_components(
    ctx: RewardContext,
    *,
    variant: PathVariant,
    decay_rate: float,
    utility_config: UtilityConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    p_long: list[float] = []
    u_long: list[float] = []
    mae_long: list[float] = []
    p_short: list[float] = []
    u_short: list[float] = []
    mae_short: list[float] = []

    for n in range(1, ctx.reward_horizon + 1):
        p_long.append(
            compute_path_n_variant(ctx, Action.LONG, n, variant=variant, decay_rate=decay_rate)
        )
        u_long.append(compute_utility_n(ctx, Action.LONG, n, utility_config))
        mae_long.append(compute_mae_n(ctx, Action.LONG, n))

        p_short.append(
            compute_path_n_variant(ctx, Action.SHORT, n, variant=variant, decay_rate=decay_rate)
        )
        u_short.append(compute_utility_n(ctx, Action.SHORT, n, utility_config))
        mae_short.append(compute_mae_n(ctx, Action.SHORT, n))

    return (
        tuple(p_long),
        tuple(u_long),
        tuple(mae_long),
        tuple(p_short),
        tuple(u_short),
        tuple(mae_short),
    )


@dataclass
class PathExperimentConfig:
    reward_horizon: int = 10
    decay_rate: float = 0.75
    sigma_window: int = 20
    min_past_bars: int = 20
    norm_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)


class PathExperimentRunner:
    """Run full Path variant comparison on a :class:`MarketDataSource`."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: PathExperimentConfig | None = None,
    ) -> None:
        self._data = market_data
        self._config = config or PathExperimentConfig()
        self._builder = FutureContextBuilder(
            market_data.bars,
            reward_horizon=self._config.reward_horizon,
            reward_config=RewardConfig(reward_horizon=self._config.reward_horizon),
        )

    def valid_t_indices(self) -> list[int]:
        return list(
            self._data.valid_t_indices(
                reward_horizon=self._config.reward_horizon,
                min_past_bars=self._config.min_past_bars,
            )
        )

    def run(self) -> dict[str, Any]:
        cfg = self._config
        t_indices = self.valid_t_indices()
        if not t_indices:
            raise ValueError("No valid decision indices for experiment")

        records_by_variant: dict[PathVariant, list[VariantSampleRecord]] = {
            v: [] for v in PathVariant
        }

        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            horizon_ret = ctx.return_from_t(cfg.reward_horizon)

            for variant in PathVariant:
                pl, ul, ml, ps, us, ms = _collect_raw_components(
                    ctx,
                    variant=variant,
                    decay_rate=cfg.decay_rate,
                    utility_config=cfg.utility_config,
                )
                records_by_variant[variant].append(
                    VariantSampleRecord(
                        t_index=t_index,
                        f_long=0.0,
                        f_short=0.0,
                        horizon_return=horizon_ret,
                        p_long=pl,
                        u_long=ul,
                        mae_long=ml,
                        p_short=ps,
                        u_short=us,
                        mae_short=ms,
                    )
                )

        split_idx = max(1, int(len(t_indices) * cfg.norm_prefix_fraction))
        prefix_t = set(t_indices[:split_idx])

        report: dict[str, Any] = {
            "market_data": describe_market_data(self._data),
            "experiment": {
                "num_valid_samples": len(t_indices),
                "norm_prefix_samples": split_idx,
                "decay_rate": cfg.decay_rate,
                "utility": {
                    "alpha": cfg.utility_config.alpha,
                    "beta": cfg.utility_config.beta,
                    "lambda": cfg.utility_config.lambda_,
                },
                "f_weights": {"w_P": 1, "w_U": 1, "w_M": 1},
            },
            "variants": {},
            "synthetic_baseline": _SYNTHETIC_BASELINE,
        }

        for variant in PathVariant:
            report["variants"][variant.value] = self._analyze_variant(
                records_by_variant[variant],
                prefix_t=prefix_t,
            )

        report["causality"] = self._causality_checks(t_indices[split_idx // 2])
        return report

    def _analyze_variant(
        self,
        records: list[VariantSampleRecord],
        *,
        prefix_t: set[int],
    ) -> dict[str, Any]:
        cfg = self._config
        horizon = cfg.reward_horizon

        prefix_records = [r for r in records if r.t_index in prefix_t]
        eval_records = [r for r in records if r.t_index not in prefix_t]
        if not prefix_records or not eval_records:
            eval_records = records
            prefix_records = records[: max(1, len(records) // 2)]

        def flatten_long(recs: Iterable[VariantSampleRecord], field_name: str, n: int) -> list[float]:
            out: list[float] = []
            for rec in recs:
                out.append(getattr(rec, field_name)[n - 1])
            return out

        # Fit normalization on prefix (LONG action, all horizons pooled per component)
        p_prefix = [v for r in prefix_records for v in r.p_long]
        u_prefix = [v for r in prefix_records for v in r.u_long]
        m_prefix = [v for r in prefix_records for v in r.mae_long]
        z_norm = FittedZScoreNormalizer.fit(tuple(p_prefix), tuple(u_prefix), tuple(m_prefix))
        robust_norm = FittedRobustZScoreNormalizer.fit(tuple(p_prefix), tuple(u_prefix), tuple(m_prefix))

        # Path distribution (LONG, all n pooled)
        all_p = [v for r in eval_records for v in r.p_long]
        path_dist = _distribution_summary(all_p)

        horizon_stats: dict[str, dict[str, Any]] = {}
        pu_corr: dict[str, float] = {}
        pu_abs_corr: dict[str, float] = {}
        pm_corr: dict[str, float] = {}

        for n in range(1, horizon + 1):
            pn = flatten_long(eval_records, "p_long", n)
            un = flatten_long(eval_records, "u_long", n)
            mn = flatten_long(eval_records, "mae_long", n)
            pn_short = [r.p_short[n - 1] for r in eval_records]

            horizon_stats[str(n)] = {
                "mean_abs_p": float(np.mean(np.abs(pn))),
                "std_p": float(np.std(pn)),
                "mean_p": float(np.mean(pn)),
                "long_short_symmetry_mean_diff": float(np.mean(np.array(pn) + np.array(pn_short))),
            }
            pu_corr[str(n)] = _corr(pn, un)
            pu_abs_corr[str(n)] = _corr([abs(x) for x in pn], [abs(x) for x in un])
            pm_corr[str(n)] = _corr(pn, mn)

        # Normalized F influence (w=1)
        p_z_list: list[float] = []
        u_z_list: list[float] = []
        m_z_list: list[float] = []
        f_list: list[float] = []
        for rec in eval_records:
            for n in range(horizon):
                p_z = z_norm.normalize_path(rec.p_long[n])
                u_z = z_norm.normalize_utility(rec.u_long[n])
                m_z = z_norm.normalize_mae(rec.mae_long[n])
                p_z_list.append(p_z)
                u_z_list.append(u_z)
                m_z_list.append(m_z)
                f_list.append(p_z + u_z - m_z)

        abs_p = np.abs(p_z_list)
        abs_u = np.abs(u_z_list)
        abs_m = np.abs(m_z_list)
        total_abs = abs_p + abs_u + abs_m + 1e-12

        influence = {
            "mean_abs_P": float(np.mean(abs_p)),
            "mean_abs_U": float(np.mean(abs_u)),
            "mean_abs_MAE": float(np.mean(abs_m)),
            "share_P": float(np.mean(abs_p / total_abs)),
            "share_U": float(np.mean(abs_u / total_abs)),
            "share_MAE": float(np.mean(abs_m / total_abs)),
            "signed_mean_P": float(np.mean(p_z_list)),
            "signed_mean_U": float(np.mean(u_z_list)),
            "signed_mean_MAE": float(np.mean(m_z_list)),
            "tail_share_U_q99": _tail_component_share(p_z_list, u_z_list, m_z_list, q=99),
        }

        # LONG/SHORT discriminability
        f_long_list: list[float] = []
        f_short_list: list[float] = []
        f_gaps: list[float] = []
        correct = 0
        for rec in eval_records:
            fl = mean(
                _compose_f(rec.p_long[n], rec.u_long[n], rec.mae_long[n], z_norm, FAblation.FULL)
                for n in range(horizon)
            )
            fs = mean(
                _compose_f(rec.p_short[n], rec.u_short[n], rec.mae_short[n], z_norm, FAblation.FULL)
                for n in range(horizon)
            )
            f_long_list.append(fl)
            f_short_list.append(fs)
            gap = fl - fs
            f_gaps.append(gap)
            if (gap > 0 and rec.horizon_return > 0) or (gap < 0 and rec.horizon_return < 0):
                correct += 1
            elif gap == 0 and rec.horizon_return == 0:
                correct += 1

        disc = {
            "direction_accuracy": correct / len(eval_records),
            "mean_abs_f_gap": float(np.mean(np.abs(f_gaps))),
            "f_gap_vs_horizon_return_corr": _corr(f_gaps, [r.horizon_return for r in eval_records]),
        }

        # Utility tail analysis
        all_u_raw = [v for r in eval_records for v in r.u_long]
        all_u_z = [z_norm.normalize_utility(v) for v in all_u_raw]
        neg_u = [v for v in all_u_raw if v < 0]
        utility_tail = {
            "raw_q95": _percentile(all_u_raw, 95),
            "raw_q99": _percentile(all_u_raw, 99),
            "neg_q05": _percentile(neg_u, 5) if neg_u else float("nan"),
            "neg_q01": _percentile(neg_u, 1) if neg_u else float("nan"),
            "norm_q95": _percentile(all_u_z, 95),
            "norm_q99": _percentile(all_u_z, 99),
            "robust_u_scale": robust_norm.utility_stats.scale,
            "extreme_loss_u_dominance_freq": _u_dominance_frequency(eval_records, z_norm),
            "large_move_analysis": _large_move_utility_check(eval_records, cfg.utility_config),
        }

        # Ablation
        ablation = {}
        for ab in FAblation:
            gaps = []
            for rec in eval_records:
                fl = mean(
                    _compose_f(rec.p_long[n], rec.u_long[n], rec.mae_long[n], z_norm, ab)
                    for n in range(horizon)
                )
                fs = mean(
                    _compose_f(rec.p_short[n], rec.u_short[n], rec.mae_short[n], z_norm, ab)
                    for n in range(horizon)
                )
                gaps.append(fl - fs)
            ablation[ab.value] = {
                "mean_abs_gap": float(np.mean(np.abs(gaps))),
                "direction_accuracy": _direction_accuracy_from_gaps(gaps, eval_records),
                "gap_return_corr": _corr(gaps, [r.horizon_return for r in eval_records]),
            }

        return {
            "path_distribution": path_dist,
            "horizon": horizon_stats,
            "path_utility_corr": pu_corr,
            "path_utility_abs_corr": pu_abs_corr,
            "path_mae_corr": pm_corr,
            "normalized_f_influence": influence,
            "long_short_discriminability": disc,
            "utility_tail": utility_tail,
            "ablation": ablation,
            "normalization": {
                "method": "zscore_prefix",
                "prefix_fraction": cfg.norm_prefix_fraction,
                "path_mean": z_norm.path_stats.center,
                "path_std": z_norm.path_stats.scale,
                "utility_mean": z_norm.utility_stats.center,
                "utility_std": z_norm.utility_stats.scale,
                "mae_mean": z_norm.mae_stats.center,
                "mae_std": z_norm.mae_stats.scale,
            },
        }

    def _causality_checks(self, t_index: int) -> dict[str, Any]:
        """Verify sigma and normalization prefix do not use future bars."""
        builder = self._builder
        ctx_before = builder.build(t_index)
        sigma_before = builder.sigma_at_t(t_index)

        bars = list(self._data.bars)
        mutated = bars[t_index + 3]
        bars[t_index + 3] = type(mutated)(
            start=mutated.start,
            end=mutated.end,
            open=mutated.open,
            high=mutated.high * 5,
            low=mutated.low * 5,
            close=mutated.close * 5,
            volume=mutated.volume,
        )
        builder_mut = FutureContextBuilder(bars, reward_horizon=self._config.reward_horizon)
        sigma_after_future = builder_mut.sigma_at_t(t_index)

        past_mut = bars[t_index - 2]
        bars[t_index - 2] = type(past_mut)(
            start=past_mut.start,
            end=past_mut.end,
            open=past_mut.open,
            high=past_mut.high * 0.5,
            low=past_mut.low * 0.5,
            close=past_mut.close * 0.5,
            volume=past_mut.volume,
        )
        builder_past = FutureContextBuilder(bars, reward_horizon=self._config.reward_horizon)
        sigma_after_past = builder_past.sigma_at_t(t_index)

        variant_checks = {}
        for variant in PathVariant:
            p_before = compute_path_n_variant(
                ctx_before, Action.LONG, 5, variant=variant, decay_rate=self._config.decay_rate
            )
            ctx_after = builder_mut.build(t_index)
            p_after = compute_path_n_variant(
                ctx_after, Action.LONG, 5, variant=variant, decay_rate=self._config.decay_rate
            )
            variant_checks[variant.value] = {
                "path_changes_with_future": p_before != p_after,
            }

        return {
            "sigma_unchanged_by_future_bar": math.isclose(sigma_before, sigma_after_future, rel_tol=0, abs_tol=1e-15),
            "sigma_changes_with_past_bar": not math.isclose(sigma_before, sigma_after_past, rel_tol=1e-9),
            "path_future_sensitivity": variant_checks,
        }


def _tail_component_share(
    p_z: list[float],
    u_z: list[float],
    m_z: list[float],
    *,
    q: float,
) -> dict[str, float]:
    f_vals = [abs(p) + abs(u) + abs(m) for p, u, m in zip(p_z, u_z, m_z)]
    thresh = _percentile(f_vals, q)
    idx = [i for i, f in enumerate(f_vals) if f >= thresh]
    if not idx:
        return {"share_U": float("nan"), "share_P": float("nan"), "share_MAE": float("nan")}
    u_share = np.mean([abs(u_z[i]) for i in idx])
    p_share = np.mean([abs(p_z[i]) for i in idx])
    m_share = np.mean([abs(m_z[i]) for i in idx])
    total = u_share + p_share + m_share + 1e-12
    return {
        "share_U": float(u_share / total),
        "share_P": float(p_share / total),
        "share_MAE": float(m_share / total),
    }


def _u_dominance_frequency(
    records: list[VariantSampleRecord],
    normalizer: FittedZScoreNormalizer,
) -> float:
    dominated = 0
    total = 0
    for rec in records:
        for n in range(len(rec.p_long)):
            p_z = abs(normalizer.normalize_path(rec.p_long[n]))
            u_z = abs(normalizer.normalize_utility(rec.u_long[n]))
            m_z = abs(normalizer.normalize_mae(rec.mae_long[n]))
            f_abs = p_z + u_z + m_z
            total += 1
            if f_abs > 0 and u_z / f_abs > 0.7:
                dominated += 1
    return dominated / total if total else float("nan")


def _large_move_utility_check(
    records: list[VariantSampleRecord],
    utility_config: UtilityConfig,
) -> dict[str, Any]:
    thresholds = (0.01, 0.02, 0.03)
    out: dict[str, Any] = {}
    for thr in thresholds:
        hits = []
        for rec in records:
            if abs(rec.horizon_return) >= thr:
                hits.append(rec.u_long[-1])
        if hits:
            out[f"abs_return_ge_{int(thr*100)}pct"] = {
                "count": len(hits),
                "mean_u": float(np.mean(hits)),
                "q05_u": _percentile(hits, 5),
            }
    return out


def _direction_accuracy_from_gaps(
    gaps: list[float],
    records: list[VariantSampleRecord],
) -> float:
    correct = 0
    for gap, rec in zip(gaps, records):
        if (gap > 0 and rec.horizon_return > 0) or (gap < 0 and rec.horizon_return < 0):
            correct += 1
        elif gap == 0 and rec.horizon_return == 0:
            correct += 1
    return correct / len(records) if records else float("nan")


_SYNTHETIC_BASELINE = {
    "note": "GBM-like proxy — NOT real market data",
    "raw_return": {"p_u_corr": 0.73, "direction_accuracy": 0.767},
    "sign_based": {"p_u_corr": 0.59, "p_u_abs_corr": 0.13, "unnorm_path_share": 0.986},
    "vol_normalized": {"p_u_corr": 0.72, "note": "scale explosion"},
    "bounded_tanh": {"p_u_corr": 0.66, "p_u_abs_corr": 0.25, "unnorm_path_share": 0.98},
}


def format_comparison_table(report: dict[str, Any]) -> str:
    """Human-readable summary table for terminal / logs."""
    lines = ["Path Variant Comparison (normalized w=1)", "=" * 72]
    header = f"{'variant':<16} {'P-U corr':>10} {'|P|-|U|':>10} {'P-MAE':>10} {'dir_acc':>10} {'|F gap|':>10}"
    lines.append(header)
    lines.append("-" * 72)
    for name, data in report["variants"].items():
        n5 = "5"
        disc = data["long_short_discriminability"]
        lines.append(
            f"{name:<16} "
            f"{data['path_utility_corr'].get(n5, float('nan')):>10.3f} "
            f"{data['path_utility_abs_corr'].get(n5, float('nan')):>10.3f} "
            f"{data['path_mae_corr'].get(n5, float('nan')):>10.3f} "
            f"{disc['direction_accuracy']:>10.3f} "
            f"{disc['mean_abs_f_gap']:>10.4f}"
        )
    return "\n".join(lines)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    runner = PathExperimentRunner(market_data)
    report = runner.run()
    print(format_comparison_table(report))
    return report


def save_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
