"""Component normalization layer — strategy not yet finalized for P1.

Raw Path / Utility / MAE values pass through this interface before ``f_n``
composition. The default implementation is identity (no scaling) so that no
normalization method is assumed until explicitly chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, median, pstdev


class ComponentNormalizer(ABC):
    """Normalize raw P/U/MAE scalars before combining into ``f_n``."""

    @abstractmethod
    def normalize_path(self, raw: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def normalize_utility(self, raw: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def normalize_mae(self, raw: float) -> float:
        raise NotImplementedError


class IdentityNormalizer(ComponentNormalizer):
    """Placeholder — returns raw values unchanged (normalization TBD)."""

    def normalize_path(self, raw: float) -> float:
        return raw

    def normalize_utility(self, raw: float) -> float:
        return raw

    def normalize_mae(self, raw: float) -> float:
        return raw


@dataclass(frozen=True)
class ComponentStats:
    """Location / scale for one component stream."""

    center: float
    scale: float

    def zscore(self, raw: float) -> float:
        if self.scale <= 0:
            return 0.0
        return (raw - self.center) / self.scale


def _robust_scale(values: tuple[float, ...]) -> float:
    if not values:
        return 1.0
    med = median(values)
    abs_dev = tuple(abs(v - med) for v in values)
    mad = median(abs_dev)
    if mad <= 0:
        return 1.0
    return 1.4826 * mad


class FittedZScoreNormalizer(ComponentNormalizer):
    """Z-score using pre-fit mean/std — stats must come from past-only data."""

    def __init__(
        self,
        *,
        path_stats: ComponentStats,
        utility_stats: ComponentStats,
        mae_stats: ComponentStats,
    ) -> None:
        self._path = path_stats
        self._utility = utility_stats
        self._mae = mae_stats

    @classmethod
    def fit(cls, path: tuple[float, ...], utility: tuple[float, ...], mae: tuple[float, ...]) -> FittedZScoreNormalizer:
        def stats(values: tuple[float, ...]) -> ComponentStats:
            if not values:
                return ComponentStats(0.0, 1.0)
            mu = mean(values)
            sigma = pstdev(values) if len(values) > 1 else 1.0
            return ComponentStats(mu, max(sigma, 1e-12))

        return cls(
            path_stats=stats(path),
            utility_stats=stats(utility),
            mae_stats=stats(mae),
        )

    def normalize_path(self, raw: float) -> float:
        return self._path.zscore(raw)

    def normalize_utility(self, raw: float) -> float:
        return self._utility.zscore(raw)

    def normalize_mae(self, raw: float) -> float:
        return self._mae.zscore(raw)

    @property
    def path_stats(self) -> ComponentStats:
        return self._path

    @property
    def utility_stats(self) -> ComponentStats:
        return self._utility

    @property
    def mae_stats(self) -> ComponentStats:
        return self._mae


class FittedRobustZScoreNormalizer(ComponentNormalizer):
    """Robust z-score (median / MAD) — for comparison only; can fail on U tails."""

    def __init__(
        self,
        *,
        path_stats: ComponentStats,
        utility_stats: ComponentStats,
        mae_stats: ComponentStats,
    ) -> None:
        self._path = path_stats
        self._utility = utility_stats
        self._mae = mae_stats

    @classmethod
    def fit(cls, path: tuple[float, ...], utility: tuple[float, ...], mae: tuple[float, ...]) -> FittedRobustZScoreNormalizer:
        def stats(values: tuple[float, ...]) -> ComponentStats:
            if not values:
                return ComponentStats(0.0, 1.0)
            med = median(values)
            scale = _robust_scale(values)
            return ComponentStats(med, max(scale, 1e-12))

        return cls(
            path_stats=stats(path),
            utility_stats=stats(utility),
            mae_stats=stats(mae),
        )

    def normalize_path(self, raw: float) -> float:
        return self._path.zscore(raw)

    def normalize_utility(self, raw: float) -> float:
        return self._utility.zscore(raw)

    def normalize_mae(self, raw: float) -> float:
        return self._mae.zscore(raw)

    @property
    def path_stats(self) -> ComponentStats:
        return self._path

    @property
    def utility_stats(self) -> ComponentStats:
        return self._utility

    @property
    def mae_stats(self) -> ComponentStats:
        return self._mae


class CausalPrefixNormalizer(ComponentNormalizer):
    """Z-score using statistics from a strict chronological prefix only.

    Fit stats on ``prefix_values`` (samples with t_index < current t), then
    apply to the current sample. Prevents future-sample leakage into norm stats.
    """

    def __init__(self, fitted: FittedZScoreNormalizer) -> None:
        self._inner = fitted

    @classmethod
    def from_prefix(
        cls,
        path_prefix: tuple[float, ...],
        utility_prefix: tuple[float, ...],
        mae_prefix: tuple[float, ...],
    ) -> CausalPrefixNormalizer:
        return cls(FittedZScoreNormalizer.fit(path_prefix, utility_prefix, mae_prefix))

    def normalize_path(self, raw: float) -> float:
        return self._inner.normalize_path(raw)

    def normalize_utility(self, raw: float) -> float:
        return self._inner.normalize_utility(raw)

    def normalize_mae(self, raw: float) -> float:
        return self._inner.normalize_mae(raw)


class FittedZScoreNormalizerSD:
    """Z-score for S + D + U + MAE experimental structures."""

    def __init__(
        self,
        *,
        speed_stats: ComponentStats,
        persistence_stats: ComponentStats,
        utility_stats: ComponentStats,
        mae_stats: ComponentStats,
    ) -> None:
        self._speed = speed_stats
        self._persistence = persistence_stats
        self._utility = utility_stats
        self._mae = mae_stats

    @classmethod
    def fit(
        cls,
        speed: tuple[float, ...],
        persistence: tuple[float, ...],
        utility: tuple[float, ...],
        mae: tuple[float, ...],
    ) -> FittedZScoreNormalizerSD:
        def stats(values: tuple[float, ...]) -> ComponentStats:
            if not values:
                return ComponentStats(0.0, 1.0)
            mu = mean(values)
            sigma = pstdev(values) if len(values) > 1 else 1.0
            return ComponentStats(mu, max(sigma, 1e-12))

        return cls(
            speed_stats=stats(speed),
            persistence_stats=stats(persistence),
            utility_stats=stats(utility),
            mae_stats=stats(mae),
        )

    def normalize_speed(self, raw: float) -> float:
        return self._speed.zscore(raw)

    def normalize_persistence(self, raw: float) -> float:
        return self._persistence.zscore(raw)

    def normalize_utility(self, raw: float) -> float:
        return self._utility.zscore(raw)

    def normalize_mae(self, raw: float) -> float:
        return self._mae.zscore(raw)
