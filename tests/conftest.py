"""Shared pytest fixtures for Phase 0."""

from __future__ import annotations

import pandas as pd
import pytest

from chartai.core.types import OHLCVBar, Timeframe
from chartai.data.mtf_aligner import BarSeriesBuilder, MultiTimeframeAligner


@pytest.fixture
def decision_at_14_27_bars() -> dict[str, tuple[OHLCVBar, ...]]:
    """Synthetic bars for the user-specified 14:27 decision example."""
    h1_starts = pd.date_range("2024-01-02 09:00", periods=8, freq="h")
    bars_1h = BarSeriesBuilder(Timeframe.H1).build_hourly_bars(h1_starts)

    h4_starts = pd.date_range("2024-01-02 08:00", periods=4, freq="4h")
    bars_4h = BarSeriesBuilder(Timeframe.H4).build_4h_bars(h4_starts)

    # 3m bars with last bar [14:24, 14:27) — decision at 14:27 close.
    m3_starts = pd.date_range("2024-01-02 13:54", periods=11, freq="3min")
    bars_3m = BarSeriesBuilder(Timeframe.M3).build_3m_bars(m3_starts)

    return {"3m": bars_3m, "1h": bars_1h, "4h": bars_4h}


@pytest.fixture
def aligner_14_27(decision_at_14_27_bars) -> MultiTimeframeAligner:
    b = decision_at_14_27_bars
    return MultiTimeframeAligner(
        bars_3m=b["3m"],
        bars_1h=b["1h"],
        bars_4h=b["4h"],
    )
