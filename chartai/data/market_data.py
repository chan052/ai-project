"""Market OHLCV ingestion for P1 research — 3m decision timeframe.

Provides CSV loading and optional remote fetch helpers. Real market data must
be supplied via CSV or an explicit fetch; synthetic series must not be passed
off as market data.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from chartai.core.types import OHLCVBar, Timeframe, bars_from_ohlcv_frame


REQUIRED_OHLCV_COLUMNS = ("start", "end", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketDataSource:
    """Loaded 3m OHLCV series with provenance metadata."""

    symbol: str
    bars: tuple[OHLCVBar, ...]
    source: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    @property
    def num_bars(self) -> int:
        return len(self.bars)

    def valid_t_indices(self, reward_horizon: int = 10, min_past_bars: int = 20) -> range:
        """Decision indices t with enough past bars and future reward window."""
        start = max(min_past_bars, 0)
        end = len(self.bars) - reward_horizon - 1
        if end < start:
            return range(0)
        return range(start, end + 1)


def load_ohlcv_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
) -> MarketDataSource:
    """Load 3m OHLCV bars from a project-standard CSV file.

    Expected columns: ``start``, ``end``, ``open``, ``high``, ``low``,
    ``close``, ``volume`` (ISO timestamps for start/end).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    for col in ("start", "end"):
        frame[col] = pd.to_datetime(frame[col], utc=True)

    bars = bars_from_ohlcv_frame(frame)
    sym = symbol or csv_path.stem
    return MarketDataSource(
        symbol=sym,
        bars=bars,
        source=f"csv:{csv_path.resolve()}",
        start_time=bars[0].start,
        end_time=bars[-1].end,
    )


def bars_to_dataframe(bars: Sequence[OHLCVBar]) -> pd.DataFrame:
    """Convert bars back to a DataFrame (for export / inspection)."""
    return pd.DataFrame(
        [
            {
                "start": b.start,
                "end": b.end,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
    )


def save_ohlcv_csv(bars: Sequence[OHLCVBar], path: str | Path) -> Path:
    """Persist bars to CSV in the standard project format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = bars_to_dataframe(bars)
    frame.to_csv(out, index=False)
    return out


def _klines_to_bars(klines: list[list]) -> tuple[OHLCVBar, ...]:
    bars: list[OHLCVBar] = []
    for row in klines:
        open_ms, open_, high, low, close, volume = (
            int(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        start = pd.Timestamp(open_ms, unit="ms", tz="UTC")
        end = start + pd.Timedelta(minutes=3)
        bars.append(
            OHLCVBar(
                start=start,
                end=end,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return tuple(bars)


def fetch_binance_klines_3m(
    symbol: str = "BTCUSDT",
    *,
    max_bars: int = 5000,
    interval: str = "3m",
) -> MarketDataSource:
    """Fetch historical 3m klines from Binance public REST API (no credentials).

    Paginates backward until ``max_bars`` are collected or history ends.
    """
    if max_bars <= 0:
        raise ValueError("max_bars must be positive")

    collected: list[list] = []
    end_time_ms: int | None = None
    base_url = "https://api.binance.com/api/v3/klines"

    while len(collected) < max_bars:
        limit = min(1000, max_bars - len(collected))
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms - 1

        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChartAI/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                batch = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Binance klines fetch failed: {exc}") from exc

        if not batch:
            break

        # API returns ascending time; prepend older pages
        collected = batch + collected
        end_time_ms = int(batch[0][0])
        if len(batch) < limit:
            break

    if not collected:
        raise RuntimeError(f"No klines returned for {symbol} {interval}")

    bars = _klines_to_bars(collected[-max_bars:])
    return MarketDataSource(
        symbol=symbol,
        bars=bars,
        source=f"binance:{symbol}:{interval}",
        start_time=bars[0].start,
        end_time=bars[-1].end,
    )


def fetch_yfinance_3m(
    symbol: str = "BTC-USD",
    *,
    period: str = "60d",
) -> MarketDataSource:
    """Optional yfinance fetch — requires ``pip install yfinance``."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed. Use CSV via load_ohlcv_csv() or "
            "fetch_binance_klines_3m()."
        ) from exc

    ticker = yf.Ticker(symbol)
    frame = ticker.history(period=period, interval="3m")
    if frame.empty:
        raise RuntimeError(f"yfinance returned no 3m data for {symbol}")

    frame = frame.reset_index()
    ts_col = "Datetime" if "Datetime" in frame.columns else "Date"
    frame["start"] = pd.to_datetime(frame[ts_col], utc=True)
    frame["end"] = frame["start"] + pd.Timedelta(minutes=3)
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    bars = bars_from_ohlcv_frame(frame[["start", "end", "open", "high", "low", "close", "volume"]])
    return MarketDataSource(
        symbol=symbol,
        bars=bars,
        source=f"yfinance:{symbol}:3m:{period}",
        start_time=bars[0].start,
        end_time=bars[-1].end,
    )


def describe_market_data(source: MarketDataSource) -> dict:
    """Summary dict for experiment reporting."""
    closes = [b.close for b in source.bars]
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    return {
        "symbol": source.symbol,
        "source": source.source,
        "timeframe": Timeframe.M3.value,
        "num_bars": source.num_bars,
        "start_time": str(source.start_time),
        "end_time": str(source.end_time),
        "close_min": min(closes),
        "close_max": max(closes),
        "return_std": float(pd.Series(rets).std()) if rets else 0.0,
    }
