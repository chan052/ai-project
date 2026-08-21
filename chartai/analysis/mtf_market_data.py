"""Resample 3m OHLCV to native 1H/4H series for analysis-only MTF audits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from chartai.core.types import OHLCVBar, Timeframe


@dataclass(frozen=True)
class MTFMarketDataSource:
    """3m market data plus analysis-resampled native 1H/4H bars."""

    symbol: str
    bars_3m: tuple[OHLCVBar, ...]
    bars_1h: tuple[OHLCVBar, ...]
    bars_4h: tuple[OHLCVBar, ...]
    source: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    resample_note: str = (
        "1H/4H native bars resampled from 3m for analysis-only MTF audit; "
        "partial HTF state still built via MultiTimeframeAligner from 3m."
    )

    @property
    def num_bars(self) -> int:
        return len(self.bars_3m)


def _bucket_key(ts: pd.Timestamp, timeframe: Timeframe) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if timeframe is Timeframe.H1:
        return ts.floor("h")
    if timeframe is Timeframe.H4:
        return ts.floor("4h")
    raise ValueError(timeframe)


def _bucket_end(start: pd.Timestamp, timeframe: Timeframe) -> pd.Timestamp:
    if timeframe is Timeframe.H1:
        return start + pd.Timedelta(hours=1)
    if timeframe is Timeframe.H4:
        return start + pd.Timedelta(hours=4)
    raise ValueError(timeframe)


def resample_bars(bars_3m: tuple[OHLCVBar, ...], timeframe: Timeframe) -> tuple[OHLCVBar, ...]:
    """Aggregate 3m bars into higher-TF OHLCV buckets (analysis-only)."""
    if not bars_3m:
        return ()
    buckets: dict[pd.Timestamp, list[OHLCVBar]] = defaultdict(list)
    for bar in bars_3m:
        key = _bucket_key(bar.start, timeframe)
        buckets[key].append(bar)

    out: list[OHLCVBar] = []
    for start in sorted(buckets.keys()):
        group = buckets[start]
        out.append(
            OHLCVBar(
                start=start,
                end=_bucket_end(start, timeframe),
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            )
        )
    return tuple(out)


def from_market_data_3m(
    *,
    symbol: str,
    bars_3m: tuple[OHLCVBar, ...],
    source: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> MTFMarketDataSource:
    return MTFMarketDataSource(
        symbol=symbol,
        bars_3m=bars_3m,
        bars_1h=resample_bars(bars_3m, Timeframe.H1),
        bars_4h=resample_bars(bars_3m, Timeframe.H4),
        source=source,
        start_time=start_time,
        end_time=end_time,
    )
