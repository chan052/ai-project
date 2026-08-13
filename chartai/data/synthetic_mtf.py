"""Synthetic multi-timeframe bar data for integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from chartai.core.config import StateConfig, TimeframeStateConfig
from chartai.core.types import OHLCVBar, Timeframe
from chartai.data.mtf_aligner import BarSeriesBuilder, MultiTimeframeAligner
from chartai.features.future_context import FutureContextBuilder
from chartai.features.sample import P1SampleAssembler
from chartai.features.state import StateBuilder


def _bar_with_close(template: OHLCVBar, close: float) -> OHLCVBar:
    return OHLCVBar(
        start=template.start,
        end=template.end,
        open=template.open,
        high=max(template.high, close),
        low=min(template.low, close),
        close=close,
        volume=template.volume,
    )


@dataclass
class SyntheticMTFDataset:
    """Mutable MTF OHLCV series for deterministic causality tests."""

    bars_3m: list[OHLCVBar] = field(default_factory=list)
    bars_1h: list[OHLCVBar] = field(default_factory=list)
    bars_4h: list[OHLCVBar] = field(default_factory=list)
    state_config: StateConfig = field(default_factory=StateConfig)
    reward_horizon: int = 10

    def set_3m_close(self, index: int, close: float) -> None:
        self.bars_3m[index] = _bar_with_close(self.bars_3m[index], close)

    def set_1h_close(self, index: int, close: float) -> None:
        self.bars_1h[index] = _bar_with_close(self.bars_1h[index], close)

    def set_4h_close(self, index: int, close: float) -> None:
        self.bars_4h[index] = _bar_with_close(self.bars_4h[index], close)

    def aligner(self) -> MultiTimeframeAligner:
        return MultiTimeframeAligner(
            bars_3m=self.bars_3m,
            bars_1h=self.bars_1h,
            bars_4h=self.bars_4h,
            state_config=self.state_config,
        )

    def state_builder(self) -> StateBuilder:
        return StateBuilder(self.aligner(), state_config=self.state_config)

    def future_context_builder(self) -> FutureContextBuilder:
        return FutureContextBuilder(
            self.bars_3m,
            reward_horizon=self.reward_horizon,
        )

    def sample_assembler(self) -> P1SampleAssembler:
        return P1SampleAssembler(self.state_builder(), self.future_context_builder())

    @classmethod
    def build_standard(
        cls,
        *,
        num_3m: int = 120,
        lookback_3m: int = 8,
        lookback_1h: int = 4,
        lookback_4h: int = 3,
        reward_horizon: int = 10,
        base_price: float = 100.0,
    ) -> SyntheticMTFDataset:
        """Build aligned synthetic MTF data with explicit test lookbacks (not research defaults)."""
        h1_starts = pd.date_range("2024-01-02 09:00", periods=24, freq="h")
        # Start early enough so decision times ~11:30 have completed 4H bars (e.g. 04:00–08:00).
        h4_starts = pd.date_range("2024-01-02 04:00", periods=12, freq="4h")
        m3_starts = pd.date_range("2024-01-02 09:00", periods=num_3m, freq="3min")

        builder = BarSeriesBuilder(Timeframe.M3)
        bars_1h = list(builder.build_hourly_bars(h1_starts))
        bars_4h = list(builder.build_4h_bars(h4_starts))
        bars_3m = list(builder.build_3m_bars(m3_starts))

        for i, bar in enumerate(bars_3m):
            close = base_price + 0.01 * i
            bars_3m[i] = _bar_with_close(bar, close)

        state_config = StateConfig(
            timeframes={
                "3m": TimeframeStateConfig(lookback_bars=lookback_3m),
                "1h": TimeframeStateConfig(lookback_bars=lookback_1h),
                "4h": TimeframeStateConfig(lookback_bars=lookback_4h),
            }
        )

        return cls(
            bars_3m=bars_3m,
            bars_1h=bars_1h,
            bars_4h=bars_4h,
            state_config=state_config,
            reward_horizon=reward_horizon,
        )

    @classmethod
    def build_with_incomplete_higher_tf_at(
        cls,
        t_index: int,
        *,
        lookback_3m: int = 6,
        lookback_1h: int = 3,
        lookback_4h: int = 2,
    ) -> SyntheticMTFDataset:
        """Dataset where decision time falls inside an in-progress 1H/4H bar."""
        ds = cls.build_standard(
            num_3m=80,
            lookback_3m=lookback_3m,
            lookback_1h=lookback_1h,
            lookback_4h=lookback_4h,
        )
        decision_time = ds.aligner().decision_time_at_3m_index(t_index).timestamp
        for bar in ds.bars_1h:
            if bar.start < decision_time < bar.end:
                assert not bar.is_completed_at(decision_time)
                break
        else:
            raise ValueError("Fixture failed to produce in-progress 1H bar at t_index")
        return ds
