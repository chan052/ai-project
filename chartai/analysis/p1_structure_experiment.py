"""P1 structure comparison: Baseline (P+U-MAE) vs S+D+U-MAE on real market data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any, Iterable

import numpy as np

from chartai.core.types import Action
from chartai.data.market_data import MarketDataSource, describe_market_data
from chartai.features.future_context import FutureContextBuilder
from chartai.reward.config import RewardConfig, UtilityConfig
from chartai.reward.context import RewardContext
from chartai.reward.mae import compute_mae_n
from chartai.reward.normalization import FittedZScoreNormalizer, FittedZScoreNormalizerSD
from chartai.reward.path import compute_path_n
from chartai.reward.path_observables import PathObservables, compute_path_observables
from chartai.reward.speed_persistence import (
    CANDIDATE_DESCRIPTIONS,
    SDPair,
    PersistenceCandidate,
    SpeedCandidate,
    compute_persistence_n,
    compute_sd_pair_n,
    compute_speed_n,
    sd_pair_components,
)
from chartai.reward.utility import compute_utility_n


def _percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _corr(rx, ry)


@dataclass
class SampleRecord:
    t_index: int
    p_long: tuple[float, ...]
    u_long: tuple[float, ...]
    mae_long: tuple[float, ...]
    p_short: tuple[float, ...]
    u_short: tuple[float, ...]
    mae_short: tuple[float, ...]
    sd_long: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    sd_short: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    obs_long: tuple[PathObservables, ...]
    obs_short: tuple[PathObservables, ...]
    horizon_return: float


@dataclass
class P1StructureExperimentConfig:
    reward_horizon: int = 10
    decay_rate: float = 0.75
    min_past_bars: int = 20
    norm_prefix_fraction: float = 0.5
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    analysis_horizon: int = 10


class P1StructureExperimentRunner:
    """Compare Baseline P+U-MAE vs S+D+U-MAE candidate structures."""

    def __init__(
        self,
        market_data: MarketDataSource,
        *,
        config: P1StructureExperimentConfig | None = None,
    ) -> None:
        self._data = market_data
        self._config = config or P1StructureExperimentConfig()
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
        if not t_indices:
            raise ValueError("No valid samples")

        records: list[SampleRecord] = []
        for t_index in t_indices:
            ctx = self._builder.build(t_index)
            records.append(self._collect_sample(ctx, t_index))

        split_idx = max(1, int(len(t_indices) * cfg.norm_prefix_fraction))
        prefix_t = set(t_indices[:split_idx])
        eval_records = [r for r in records if r.t_index not in prefix_t]
        prefix_records = [r for r in records if r.t_index in prefix_t]

        report: dict[str, Any] = {
            "purpose": "Compare Baseline (P+U-MAE) vs S+D+U-MAE; explore Opportunity Assessment metrics",
            "market_data": describe_market_data(self._data),
            "experiment": {
                "num_valid_samples": len(t_indices),
                "eval_samples": len(eval_records),
                "norm_prefix_fraction": cfg.norm_prefix_fraction,
                "decay_rate": cfg.decay_rate,
                "f_weights": {"w_P_or_SD": 1, "w_U": 1, "w_MAE": 1},
            },
            "candidate_definitions": self._candidate_definitions(),
            "baseline": self._analyze_baseline(prefix_records, eval_records),
            "sd_pairs": {},
            "speed_candidates": {},
            "persistence_candidates": {},
            "case_study_analysis": self._case_study_analysis(eval_records),
            "amplification_analysis": {},
        }

        for pair in SDPair:
            report["sd_pairs"][pair.value] = self._analyze_sd_pair(
                pair, prefix_records, eval_records
            )

        for sc in SpeedCandidate:
            report["speed_candidates"][sc.value] = self._analyze_single_speed(
                sc, prefix_records, eval_records
            )
        for pc in PersistenceCandidate:
            report["persistence_candidates"][pc.value] = self._analyze_single_persistence(
                pc, prefix_records, eval_records
            )

        report["amplification_analysis"] = self._amplification_analysis(
            eval_records, report["baseline"], report["sd_pairs"]
        )
        report["structure_comparison_summary"] = self._summary_table(
            report["baseline"], report["sd_pairs"]
        )
        return report

    def _collect_sample(self, ctx: RewardContext, t_index: int) -> SampleRecord:
        cfg = self._config
        horizon = cfg.reward_horizon
        p_l, u_l, m_l, p_s, u_s, m_s = [], [], [], [], [], []
        sd_l: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        sd_s: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}

        for pair in SDPair:
            sl: list[float] = []
            dl: list[float] = []
            ss: list[float] = []
            ds: list[float] = []
            for n in range(1, horizon + 1):
                s, d = compute_sd_pair_n(
                    ctx, Action.LONG, n, pair, decay_rate=cfg.decay_rate
                )
                sl.append(s)
                dl.append(d)
                s2, d2 = compute_sd_pair_n(
                    ctx, Action.SHORT, n, pair, decay_rate=cfg.decay_rate
                )
                ss.append(s2)
                ds.append(d2)
            sd_l[pair.value] = (tuple(sl), tuple(dl))
            sd_s[pair.value] = (tuple(ss), tuple(ds))

        obs_l = []
        obs_s = []
        for n in range(1, horizon + 1):
            p_l.append(compute_path_n(ctx, Action.LONG, n, decay_rate=cfg.decay_rate))
            u_l.append(compute_utility_n(ctx, Action.LONG, n, cfg.utility_config))
            m_l.append(compute_mae_n(ctx, Action.LONG, n))
            p_s.append(compute_path_n(ctx, Action.SHORT, n, decay_rate=cfg.decay_rate))
            u_s.append(compute_utility_n(ctx, Action.SHORT, n, cfg.utility_config))
            m_s.append(compute_mae_n(ctx, Action.SHORT, n))
            obs_l.append(compute_path_observables(ctx, Action.LONG, n))
            obs_s.append(compute_path_observables(ctx, Action.SHORT, n))

        return SampleRecord(
            t_index=t_index,
            p_long=tuple(p_l),
            u_long=tuple(u_l),
            mae_long=tuple(m_l),
            p_short=tuple(p_s),
            u_short=tuple(u_s),
            mae_short=tuple(m_s),
            sd_long=sd_l,
            sd_short=sd_s,
            obs_long=tuple(obs_l),
            obs_short=tuple(obs_s),
            horizon_return=ctx.return_from_t(horizon),
        )

    def _candidate_definitions(self) -> dict[str, Any]:
        pairs = {}
        for pair in SDPair:
            sc, pc = sd_pair_components(pair)
            pairs[pair.value] = {
                "speed": sc.value,
                "persistence": pc.value,
                "speed_desc": CANDIDATE_DESCRIPTIONS.get(f"S_{sc.value}", ""),
                "persistence_desc": CANDIDATE_DESCRIPTIONS.get(f"D_{pc.value}", ""),
            }
        return {
            "baseline": "F = norm(P) + norm(U) - norm(MAE), P = raw_return path",
            "candidate": "F = norm(S) + norm(D) + norm(U) - norm(MAE)",
            "sd_pairs": pairs,
            "speed_candidates": list(SpeedCandidate),
            "persistence_candidates": list(PersistenceCandidate),
        }

    def _fit_baseline_norm(
        self, prefix_records: list[SampleRecord]
    ) -> FittedZScoreNormalizer:
        p = [v for r in prefix_records for v in r.p_long]
        u = [v for r in prefix_records for v in r.u_long]
        m = [v for r in prefix_records for v in r.mae_long]
        return FittedZScoreNormalizer.fit(tuple(p), tuple(u), tuple(m))

    def _fit_sd_norm(
        self, prefix_records: list[SampleRecord], pair_key: str
    ) -> FittedZScoreNormalizerSD:
        s = [v for r in prefix_records for v in r.sd_long[pair_key][0]]
        d = [v for r in prefix_records for v in r.sd_long[pair_key][1]]
        u = [v for r in prefix_records for v in r.u_long]
        m = [v for r in prefix_records for v in r.mae_long]
        return FittedZScoreNormalizerSD.fit(tuple(s), tuple(d), tuple(u), tuple(m))

    def _f_baseline(
        self, rec: SampleRecord, norm: FittedZScoreNormalizer, *, action: Action = Action.LONG
    ) -> float:
        if action is Action.LONG:
            return mean(
                norm.normalize_path(rec.p_long[n])
                + norm.normalize_utility(rec.u_long[n])
                - norm.normalize_mae(rec.mae_long[n])
                for n in range(self._config.reward_horizon)
            )
        return mean(
            norm.normalize_path(rec.p_short[n])
            + norm.normalize_utility(rec.u_short[n])
            - norm.normalize_mae(rec.mae_short[n])
            for n in range(self._config.reward_horizon)
        )

    def _f_sd(
        self,
        rec: SampleRecord,
        norm: FittedZScoreNormalizerSD,
        pair_key: str,
        *,
        action: Action = Action.LONG,
    ) -> float:
        sd = rec.sd_long[pair_key] if action is Action.LONG else rec.sd_short[pair_key]
        sl, dl = sd
        u = rec.u_long if action is Action.LONG else rec.u_short
        m = rec.mae_long if action is Action.LONG else rec.mae_short
        return mean(
            norm.normalize_speed(sl[n])
            + norm.normalize_persistence(dl[n])
            + norm.normalize_utility(u[n])
            - norm.normalize_mae(m[n])
            for n in range(self._config.reward_horizon)
        )

    def _component_correlations(
        self,
        eval_records: list[SampleRecord],
        *,
        components: dict[str, list[float]],
        horizon_idx: int,
    ) -> dict[str, float]:
        keys = list(components.keys())
        out: dict[str, float] = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                out[f"{a}__{b}"] = _corr(components[a], components[b])
        return out

    def _layer1_terminal(self, f_gaps: list[float], records: list[SampleRecord]) -> dict[str, float]:
        rets = [r.horizon_return for r in records]
        correct = sum(
            1
            for g, ret in zip(f_gaps, rets)
            if (g > 0 and ret > 0) or (g < 0 and ret < 0) or (g == 0 and ret == 0)
        )
        return {
            "direction_accuracy": correct / len(records) if records else float("nan"),
            "f_gap_vs_terminal_return_corr": _corr(f_gaps, rets),
            "mean_abs_f_gap": float(np.mean(np.abs(f_gaps))) if f_gaps else float("nan"),
        }

    def _layer2_opportunity(
        self, f_scores: list[float], obs_list: list[PathObservables]
    ) -> dict[str, Any]:
        """Multi-observable ranking behavior — no single canonical opportunity label."""
        h = self._config.analysis_horizon - 1
        mfe = [o.mfe for o in obs_list]
        mae = [o.mae for o in obs_list]
        ttmfe = [o.time_to_mfe or (h + 1) for o in obs_list]
        fav_occ = [o.favorable_occupancy for o in obs_list]
        fav_dur = [o.favorable_duration for o in obs_list]
        early = [o.early_mean_return for o in obs_list]
        terminal = [o.terminal_return for o in obs_list]

        ranked = np.argsort(f_scores)
        n = len(f_scores)
        top = ranked[-max(1, n // 10) :]
        bot = ranked[: max(1, n // 10)]

        def bucket_mean(idx, values):
            return float(np.mean([values[i] for i in idx]))

        return {
            "ranking_correlations": {
                "spearman_F_mfe": _spearman(f_scores, mfe),
                "spearman_F_neg_mae": _spearman(f_scores, [-x for x in mae]),
                "spearman_F_neg_time_to_mfe": _spearman(f_scores, [-x for x in ttmfe]),
                "spearman_F_favorable_occupancy": _spearman(f_scores, fav_occ),
                "spearman_F_favorable_duration": _spearman(f_scores, fav_dur),
                "spearman_F_early_return": _spearman(f_scores, early),
                "spearman_F_terminal_return": _spearman(f_scores, terminal),
            },
            "top_decile_observable_means": {
                "mfe": bucket_mean(top, mfe),
                "mae": bucket_mean(top, mae),
                "terminal_return": bucket_mean(top, terminal),
                "early_mean_return": bucket_mean(top, early),
                "favorable_occupancy": bucket_mean(top, fav_occ),
            },
            "bottom_decile_observable_means": {
                "mfe": bucket_mean(bot, mfe),
                "mae": bucket_mean(bot, mae),
                "terminal_return": bucket_mean(bot, terminal),
                "early_mean_return": bucket_mean(bot, early),
                "favorable_occupancy": bucket_mean(bot, fav_occ),
            },
            "high_mae_filtering": self._high_mae_filtering(f_scores, mae),
            "poor_entry_archetype": self._archetype_scores(f_scores, obs_list),
        }

    def _high_mae_filtering(self, f_scores: list[float], mae: list[float]) -> dict[str, float]:
        thresh = _percentile(mae, 90)
        high_mae_idx = [i for i, m in enumerate(mae) if m >= thresh]
        low_mae_idx = [i for i, m in enumerate(mae) if m < _percentile(mae, 50)]
        if not high_mae_idx or not low_mae_idx:
            return {}
        return {
            "mae_q90_threshold": thresh,
            "mean_F_high_mae": float(np.mean([f_scores[i] for i in high_mae_idx])),
            "mean_F_low_mae": float(np.mean([f_scores[i] for i in low_mae_idx])),
            "high_mae_downrank_gap": float(
                np.mean([f_scores[i] for i in low_mae_idx])
                - np.mean([f_scores[i] for i in high_mae_idx])
            ),
        }

    def _archetype_scores(
        self, f_scores: list[float], obs_list: list[PathObservables]
    ) -> dict[str, Any]:
        """Case A-like vs Case B-like archetypes from real paths."""
        h = self._config.analysis_horizon - 1
        case_a: list[int] = []
        case_b: list[int] = []
        for i, ob in enumerate(obs_list):
            if ob.early_mean_return < -0.0003 and ob.terminal_return > 0:
                case_a.append(i)
            if ob.early_mean_return > 0.0003 and ob.terminal_return < 0:
                case_b.append(i)

        def stats(idxs: list[int]) -> dict[str, float]:
            if not idxs:
                return {"count": 0}
            return {
                "count": len(idxs),
                "mean_F": float(np.mean([f_scores[i] for i in idxs])),
                "mean_mfe": float(np.mean([obs_list[i].mfe for i in idxs])),
                "mean_mae": float(np.mean([obs_list[i].mae for i in idxs])),
                "mean_terminal": float(np.mean([obs_list[i].terminal_return for i in idxs])),
                "mean_early": float(np.mean([obs_list[i].early_mean_return for i in idxs])),
            }

        return {
            "case_a_dip_then_rise": stats(case_a),
            "case_b_rise_then_fall": stats(case_b),
            "case_a_vs_b_mean_F_diff": (
                float(np.mean([f_scores[i] for i in case_a]) - np.mean([f_scores[i] for i in case_b]))
                if case_a and case_b
                else float("nan")
            ),
        }

    def _normalized_influence(
        self, z_lists: dict[str, list[float]]
    ) -> dict[str, float]:
        abs_vals = {k: np.abs(v) for k, v in z_lists.items()}
        total = sum(abs_vals[k] for k in abs_vals) + 1e-12
        shares = {f"share_{k}": float(np.mean(abs_vals[k] / total)) for k in abs_vals}
        means = {f"mean_abs_{k}": float(np.mean(abs_vals[k])) for k in abs_vals}
        return {**means, **shares}

    def _analyze_baseline(
        self,
        prefix_records: list[SampleRecord],
        eval_records: list[SampleRecord],
    ) -> dict[str, Any]:
        norm = self._fit_baseline_norm(prefix_records)
        h = self._config.analysis_horizon - 1
        pn = [r.p_long[h] for r in eval_records]
        un = [r.u_long[h] for r in eval_records]
        mn = [r.mae_long[h] for r in eval_records]

        f_gaps = []
        f_long = []
        obs_h = []
        for rec in eval_records:
            fl = self._f_baseline(rec, norm, action=Action.LONG)
            fs = self._f_baseline(rec, norm, action=Action.SHORT)
            f_long.append(fl)
            f_gaps.append(fl - fs)
            obs_h.append(rec.obs_long[h])

        p_z, u_z, m_z = [], [], []
        for rec in eval_records:
            for n in range(self._config.reward_horizon):
                p_z.append(norm.normalize_path(rec.p_long[n]))
                u_z.append(norm.normalize_utility(rec.u_long[n]))
                m_z.append(norm.normalize_mae(rec.mae_long[n]))

        return {
            "structure": "P + U - MAE",
            "component_corr_h10": {
                "P_U": _corr(pn, un),
                "P_MAE": _corr(pn, mn),
                "U_MAE": _corr(un, mn),
                "absP_absU": _corr([abs(x) for x in pn], [abs(x) for x in un]),
            },
            "normalized_influence": self._normalized_influence(
                {"P": p_z, "U": u_z, "MAE": m_z}
            ),
            "layer1_terminal": self._layer1_terminal(f_gaps, eval_records),
            "layer2_opportunity": self._layer2_opportunity(f_long, obs_h),
        }

    def _analyze_sd_pair(
        self,
        pair: SDPair,
        prefix_records: list[SampleRecord],
        eval_records: list[SampleRecord],
    ) -> dict[str, Any]:
        key = pair.value
        norm = self._fit_sd_norm(prefix_records, key)
        h = self._config.analysis_horizon - 1
        sl = [r.sd_long[key][0][h] for r in eval_records]
        dl = [r.sd_long[key][1][h] for r in eval_records]
        un = [r.u_long[h] for r in eval_records]
        mn = [r.mae_long[h] for r in eval_records]
        pn = [r.p_long[h] for r in eval_records]

        f_gaps = []
        f_long = []
        obs_h = []
        for rec in eval_records:
            fl = self._f_sd(rec, norm, key, action=Action.LONG)
            fs = self._f_sd(rec, norm, key, action=Action.SHORT)
            f_long.append(fl)
            f_gaps.append(fl - fs)
            obs_h.append(rec.obs_long[h])

        s_z, d_z, u_z, m_z = [], [], [], []
        for rec in eval_records:
            sl_n, dl_n = rec.sd_long[key]
            for n in range(self._config.reward_horizon):
                s_z.append(norm.normalize_speed(sl_n[n]))
                d_z.append(norm.normalize_persistence(dl_n[n]))
                u_z.append(norm.normalize_utility(rec.u_long[n]))
                m_z.append(norm.normalize_mae(rec.mae_long[n]))

        speed_c, persist_c = sd_pair_components(pair)
        return {
            "structure": "S + D + U - MAE",
            "speed_candidate": speed_c.value,
            "persistence_candidate": persist_c.value,
            "component_corr_h10": {
                "S_D": _corr(sl, dl),
                "S_U": _corr(sl, un),
                "D_U": _corr(dl, un),
                "S_MAE": _corr(sl, mn),
                "D_MAE": _corr(dl, mn),
                "U_MAE": _corr(un, mn),
                "S_P_baseline": _corr(sl, pn),
                "D_P_baseline": _corr(dl, pn),
            },
            "normalized_influence": self._normalized_influence(
                {"S": s_z, "D": d_z, "U": u_z, "MAE": m_z}
            ),
            "layer1_terminal": self._layer1_terminal(f_gaps, eval_records),
            "layer2_opportunity": self._layer2_opportunity(f_long, obs_h),
            "vs_baseline_P_corr": {
                "S_vs_P": _corr(sl, pn),
                "D_vs_P": _corr(dl, pn),
                "S_plus_D_vs_P": _corr([s + d for s, d in zip(sl, dl)], pn),
            },
        }

    def _analyze_single_speed(
        self,
        candidate: SpeedCandidate,
        prefix_records: list[SampleRecord],
        eval_records: list[SampleRecord],
    ) -> dict[str, Any]:
        h = self._config.analysis_horizon - 1
        # Recompute from stored paths via proxy: use first pair containing this speed
        values = []
        un = [r.u_long[h] for r in eval_records]
        mn = [r.mae_long[h] for r in eval_records]
        pn = [r.p_long[h] for r in eval_records]
        for rec in eval_records:
            # approximate from any pair — compute fresh would need ctx; use stored sd
            for pair in SDPair:
                sc, _ = sd_pair_components(pair)
                if sc == candidate:
                    values.append(rec.sd_long[pair.value][0][h])
                    break
        return {
            "corr_U": _corr(values, un),
            "corr_MAE": _corr(values, mn),
            "corr_P": _corr(values, pn),
            "corr_absU": _corr([abs(x) for x in values], [abs(x) for x in un]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    def _analyze_single_persistence(
        self,
        candidate: PersistenceCandidate,
        prefix_records: list[SampleRecord],
        eval_records: list[SampleRecord],
    ) -> dict[str, Any]:
        h = self._config.analysis_horizon - 1
        values = []
        un = [r.u_long[h] for r in eval_records]
        mn = [r.mae_long[h] for r in eval_records]
        pn = [r.p_long[h] for r in eval_records]
        for rec in eval_records:
            for pair in SDPair:
                _, pc = sd_pair_components(pair)
                if pc == candidate:
                    values.append(rec.sd_long[pair.value][1][h])
                    break
        return {
            "corr_U": _corr(values, un),
            "corr_MAE": _corr(values, mn),
            "corr_P": _corr(values, pn),
            "corr_absU": _corr([abs(x) for x in values], [abs(x) for x in un]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    def _case_study_analysis(self, eval_records: list[SampleRecord]) -> dict[str, Any]:
        """Conceptual Case A / B separation on real data."""
        h = self._config.analysis_horizon - 1
        case_a, case_b, same_terminal = [], [], []
        for i, rec in enumerate(eval_records):
            o = rec.obs_long[h]
            if o.early_mean_return < -0.0003 and o.terminal_return > 0:
                case_a.append(i)
            elif o.early_mean_return > 0.0003 and o.terminal_return < 0:
                case_b.append(i)

        # Pairs with similar terminal return but different early path
        for i in range(len(eval_records)):
            for j in range(i + 1, min(i + 500, len(eval_records))):
                oi = eval_records[i].obs_long[h]
                oj = eval_records[j].obs_long[h]
                if abs(oi.terminal_return - oj.terminal_return) < 0.0002:
                    if oi.early_mean_return < -0.0002 and oj.early_mean_return > 0.0002:
                        same_terminal.append((i, j))
                    elif oj.early_mean_return < -0.0002 and oi.early_mean_return > 0.0002:
                        same_terminal.append((j, i))
                if len(same_terminal) >= 200:
                    break
            if len(same_terminal) >= 200:
                break

        return {
            "case_a_count": len(case_a),
            "case_b_count": len(case_b),
            "same_terminal_diff_early_pairs": len(same_terminal),
            "note": "Case A: early adverse then positive terminal; Case B: early favorable then negative terminal",
        }

    def _amplification_analysis(
        self,
        eval_records: list[SampleRecord],
        baseline: dict[str, Any],
        sd_pairs: dict[str, Any],
    ) -> dict[str, Any]:
        """When components align, does F amplify terminal-return preference?"""
        prefix = eval_records[: len(eval_records) // 2]
        norm_b = self._fit_baseline_norm(prefix)
        h = self._config.analysis_horizon - 1
        aligned_terminal_boost = []
        misaligned_penalty = []
        for rec in eval_records:
            p, u, m = rec.p_long[h], rec.u_long[h], rec.mae_long[h]
            fz = (
                norm_b.normalize_path(p)
                + norm_b.normalize_utility(u)
                - norm_b.normalize_mae(m)
            )
            if p > 0 and u > 0:
                aligned_terminal_boost.append(fz)
            if p > 0 and u < 0:
                misaligned_penalty.append(fz)

        return {
            "baseline_P_U_same_sign_mean_F": float(np.mean(aligned_terminal_boost))
            if aligned_terminal_boost
            else float("nan"),
            "baseline_P_U_opposite_sign_mean_F": float(np.mean(misaligned_penalty))
            if misaligned_penalty
            else float("nan"),
            "interpretation": "Large positive F when P and U agree amplifies terminal-return narrative",
            "sd_pair_S_D_corr": {k: v["component_corr_h10"]["S_D"] for k, v in sd_pairs.items()},
        }

    def _summary_table(
        self, baseline: dict[str, Any], sd_pairs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = []
        b1 = baseline["layer1_terminal"]
        b2 = baseline["layer2_opportunity"]["ranking_correlations"]
        rows.append(
            {
                "structure": "baseline_P+U-MAE",
                "dir_acc": b1["direction_accuracy"],
                "gap_terminal_corr": b1["f_gap_vs_terminal_return_corr"],
                "spearman_mfe": b2["spearman_F_mfe"],
                "spearman_neg_mae": b2["spearman_F_neg_mae"],
                "spearman_early": b2["spearman_F_early_return"],
                "P_U_corr": baseline["component_corr_h10"]["P_U"],
            }
        )
        for name, data in sd_pairs.items():
            l1 = data["layer1_terminal"]
            l2 = data["layer2_opportunity"]["ranking_correlations"]
            rows.append(
                {
                    "structure": name,
                    "dir_acc": l1["direction_accuracy"],
                    "gap_terminal_corr": l1["f_gap_vs_terminal_return_corr"],
                    "spearman_mfe": l2["spearman_F_mfe"],
                    "spearman_neg_mae": l2["spearman_F_neg_mae"],
                    "spearman_early": l2["spearman_F_early_return"],
                    "S_U_corr": data["component_corr_h10"]["S_U"],
                    "S_D_corr": data["component_corr_h10"]["S_D"],
                }
            )
        return rows


def format_summary_table(report: dict[str, Any]) -> str:
    lines = [
        "P1 Structure Comparison",
        "=" * 90,
        f"{'structure':<32} {'dir_acc':>8} {'gap~ret':>8} {'~mfe':>8} {'~(-mae)':>8} {'~early':>8}",
        "-" * 90,
    ]
    for row in report["structure_comparison_summary"]:
        lines.append(
            f"{row['structure']:<32} "
            f"{row.get('dir_acc', float('nan')):>8.3f} "
            f"{row.get('gap_terminal_corr', float('nan')):>8.3f} "
            f"{row.get('spearman_mfe', float('nan')):>8.3f} "
            f"{row.get('spearman_neg_mae', float('nan')):>8.3f} "
            f"{row.get('spearman_early', float('nan')):>8.3f}"
        )
    return "\n".join(lines)


def run_and_print(market_data: MarketDataSource) -> dict[str, Any]:
    report = P1StructureExperimentRunner(market_data).run()
    print(format_summary_table(report))
    return report


def save_report(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
