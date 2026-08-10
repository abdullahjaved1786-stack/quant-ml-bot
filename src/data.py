"""Data loader with OHLCV, funding rate, and open interest alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None  # type: ignore

from src.config import config


def compute_atr(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Compute Average True Range using Wilder's EMA.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback period (default: config.atr_period).

    Returns:
        ATR series aligned with df index.
    """
    window = window or config.atr_period
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's EMA = EMA with alpha = 1/window
    atr = tr.ewm(alpha=1.0 / window, adjust=False).mean()
    return atr


def fetch_ohlcv(
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int | None = None,
    exchange_id: str = "binance",
    since: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from CCXT exchange.

    Args:
        symbol: Trading pair (default: config.default_symbol).
        timeframe: Bar interval (default: config.default_timeframe).
        limit: Number of bars (default: config.default_limit).
        exchange_id: CCXT exchange ID.
        since: Unix timestamp in ms for starting point.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
        Index is timestamp (datetime, UTC).
    """
    if ccxt is None:
        raise ImportError("ccxt is required for live data fetching")

    symbol = symbol or config.default_symbol
    timeframe = timeframe or config.default_timeframe
    limit = limit or config.default_limit

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").reset_index()  # keep timestamp as column too

    return df


def fetch_funding_rate(
    symbol: str | None = None,
    limit: int = 500,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch funding rate history (8h intervals).

    Funding rates are typically published at 00:00, 08:00, 16:00 UTC.

    Args:
        symbol: Trading pair (default: config.default_symbol).
        limit: Number of funding periods.
        exchange_id: CCXT exchange ID.

    Returns:
        DataFrame with columns: timestamp, funding_rate.
        Index is timestamp (datetime, UTC).
    """
    if ccxt is None:
        raise ImportError("ccxt is required for funding rate data")

    symbol = symbol or config.default_symbol
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    # Binance uses fetch_funding_rate_history
    if hasattr(exchange, "fetch_funding_rate_history"):
        rates = exchange.fetch_funding_rate_history(symbol, limit=limit)
        df = pd.DataFrame(rates)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif "fundingTime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df = df.rename(columns={"fundingRate": "funding_rate"})
        df = df[["timestamp", "funding_rate"]].copy()
    else:
        # Fallback: generate synthetic zero rates
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    end=pd.Timestamp.now(tz="UTC").floor("8h"),
                    periods=limit,
                    freq="8h",
                ),
                "funding_rate": 0.0,
            }
        )

    df = df.set_index("timestamp").reset_index()
    return df


def fetch_open_interest(
    symbol: str | None = None,
    limit: int = 500,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch open interest history.

    Args:
        symbol: Trading pair (default: config.default_symbol).
        limit: Number of data points.
        exchange_id: CCXT exchange ID.

    Returns:
        DataFrame with columns: timestamp, open_interest.
        Index is timestamp (datetime, UTC).
    """
    if ccxt is None:
        raise ImportError("ccxt is required for open interest data")

    symbol = symbol or config.default_symbol
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    # Not all exchanges support this; fallback to synthetic data
    if hasattr(exchange, "fetch_open_interest_history"):
        try:
            oi = exchange.fetch_open_interest_history(symbol, limit=limit)
            df = pd.DataFrame(oi)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df[["timestamp", "openInterest"]].copy()
            df = df.rename(columns={"openInterest": "open_interest"})
        except Exception:
            df = _synthetic_oi(limit)
    else:
        df = _synthetic_oi(limit)

    df = df.set_index("timestamp").reset_index()
    return df


def _synthetic_oi(limit: int) -> pd.DataFrame:
    """Generate synthetic open interest for testing/fallback."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                end=pd.Timestamp.now(tz="UTC"),
                periods=limit,
                freq="5min",
            ),
            "open_interest": 1_000_000.0,  # constant placeholder
        }
    )


def align_data(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    open_interest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align OHLCV, funding rate, and open interest to common timestamp index.

    Funding rates are forward-filled from 8h publication times.

    Args:
        ohlcv: DataFrame with timestamp column.
        funding: DataFrame with timestamp, funding_rate columns.
        open_interest: DataFrame with timestamp, open_interest columns.

    Returns:
        Merged DataFrame with all columns aligned to OHLCV timestamps.
    """
    df = ohlcv.copy()

    if funding is not None and not funding.empty:
        funding = funding.set_index("timestamp")
        df = df.set_index("timestamp")
        # Reindex funding to OHLCV timestamps, forward-fill 8h rates
        df["funding_rate"] = funding["funding_rate"].reindex(df.index, method="ffill")
        df = df.reset_index()

    if open_interest is not None and not open_interest.empty:
        oi = open_interest.set_index("timestamp")
        df = df.set_index("timestamp")
        df["open_interest"] = oi["open_interest"].reindex(df.index, method="ffill")
        df = df.reset_index()

    return df


def load_cached_data(cache_path: Path, max_age_seconds: int = 60) -> pd.DataFrame | None:
    """Load cached OHLCV data if fresh enough.

    Args:
        cache_path: Path to cached parquet/CSV file.
        max_age_seconds: Maximum age of cache in seconds.

    Returns:
        DataFrame or None if cache is stale/missing.
    """
    if not cache_path.exists():
        return None

    age = pd.Timestamp.now(tz="UTC").timestamp() - cache_path.stat().st_mtime
    if age >= max_age_seconds:
        return None

    try:
        if cache_path.suffix == ".parquet":
            return pd.read_parquet(cache_path)
        # Parse timestamp columns when loading CSV
        df = pd.read_csv(cache_path)
        for col in df.columns:
            if col == "timestamp" or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], utc=True)
                except Exception:
                    pass
        return df
    except Exception:
        return None


def save_cached_data(df: pd.DataFrame, cache_path: Path) -> None:
    """Save OHLCV data to cache file.

    Args:
        df: DataFrame to cache.
        cache_path: Destination path (parquet or CSV based on suffix).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".parquet":
        df.to_parquet(cache_path, index=False)
    else:
        df.to_csv(cache_path, index=False)


def make_synthetic_ohlcv(
    n_bars: int = 500,
    seed: int = 42,
    start_price: float = 100.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n_bars: Number of bars to generate.
        seed: Random seed for reproducibility.
        start_price: Initial price.
        volatility: Price volatility per bar.

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        start="2024-01-01",
        periods=n_bars,
        freq="5min",
        tz="UTC",
    )

    # Random walk for close prices
    returns = rng.normal(0, volatility, n_bars)
    close = start_price * np.exp(np.cumsum(returns))

    # Generate OHLCV with realistic relationships
    high = close * (1 + rng.uniform(0, 0.005, n_bars))
    low = close * (1 - rng.uniform(0, 0.005, n_bars))
    open_ = close + rng.uniform(-0.002, 0.002, n_bars) * close
    volume = rng.integers(100, 10_000, n_bars).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df
