"""Generic Standard Z-score fitter for analysis-only P1 target audits."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from chartai.reward.normalization import ComponentStats


@dataclass(frozen=True)
class StandardZScoreModel:
    """Prefix-fitted z-score for one observable stream."""

    name: str
    stats: ComponentStats

    @classmethod
    def fit(cls, name: str, values: tuple[float, ...]) -> StandardZScoreModel:
        if not values:
            return cls(name, ComponentStats(0.0, 1.0))
        mu = mean(values)
        sigma = pstdev(values) if len(values) > 1 else 1.0
        return cls(name, ComponentStats(mu, max(sigma, 1e-12)))

    def z(self, raw: float) -> float:
        return self.stats.zscore(raw)


@dataclass
class P1ObservableZScoreBundle:
    """Causal prefix-fit z-score models for P1 target observables."""

    u: StandardZScoreModel
    mfe: StandardZScoreModel
    mae: StandardZScoreModel
    giveback: StandardZScoreModel
    chop: StandardZScoreModel
    recovery: StandardZScoreModel
    p_long: StandardZScoreModel
    p_short: StandardZScoreModel

    @classmethod
    def fit_from_rows(cls, rows: list[dict[str, float]]) -> P1ObservableZScoreBundle:
        def col(key: str) -> tuple[float, ...]:
            return tuple(r[key] for r in rows)

        return cls(
            u=StandardZScoreModel.fit("U", col("U")),
            mfe=StandardZScoreModel.fit("MFE", col("MFE")),
            mae=StandardZScoreModel.fit("MAE", col("MAE")),
            giveback=StandardZScoreModel.fit("giveback", col("giveback")),
            chop=StandardZScoreModel.fit("chop", col("chop")),
            recovery=StandardZScoreModel.fit("recovery", col("recovery")),
            p_long=StandardZScoreModel.fit("P_long", col("P_long")),
            p_short=StandardZScoreModel.fit("P_short", col("P_short")),
        )

    def transform(self, raw: dict[str, float]) -> dict[str, float]:
        return {
            "U": self.u.z(raw["U"]),
            "MFE": self.mfe.z(raw["MFE"]),
            "MAE": self.mae.z(raw["MAE"]),
            "giveback": self.giveback.z(raw["giveback"]),
            "chop": self.chop.z(raw["chop"]),
            "recovery": self.recovery.z(raw["recovery"]),
            "P_long": self.p_long.z(raw["P_long"]),
            "P_short": self.p_short.z(raw["P_short"]),
        }

    def scale_summary(self, z_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        keys = ("U", "MFE", "MAE", "giveback", "chop", "recovery")
        out: dict[str, dict[str, float]] = {}
        for k in keys:
            vals = [r[k] for r in z_rows]
            if not vals:
                continue
            mu = mean(vals)
            sigma = pstdev(vals) if len(vals) > 1 else 0.0
            out[k] = {
                "mean": mu,
                "std": sigma,
                "min": min(vals),
                "max": max(vals),
                "p99_abs": sorted(abs(v) for v in vals)[int(0.99 * (len(vals) - 1))],
            }
        return out
