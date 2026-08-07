"""OHLCV data loading and stationary feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["log_return", "atr", "volatility_ratio", "norm_rsi"]
ATR_PERIOD = 14
RSI_PERIOD = 14


def load_ohlcv(symbol, source="yfinance", interval="1d", limit=750, exchange=None, timeframe="1h"):
    """Load OHLCV bars from yfinance or ccxt.

    source="yfinance" -> ``interval`` is a yfinance interval (1h, 1d, ...).
    source="ccxt"     -> ``exchange`` + ``timeframe`` used; ``symbol`` is the ccxt market id.
    """
    if source == "ccxt":
        if not exchange:
            raise ValueError("source=ccxt requires --exchange (e.g. binance)")
        return _load_ccxt(exchange, symbol, timeframe, limit)
    return _load_yfinance(symbol, interval, limit)


def _load_yfinance(symbol, interval, limit):
    import yfinance as yf

    period = {"1m": "7d", "5m": "1mo", "15m": "1mo", "1h": "1y", "1d": "2y"}.get(interval, "2y")
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):  # tolerate multi-ticker column shape
        raw.columns = raw.columns.get_level_values(0)
    colmap = {c.lower(): c for c in raw.columns}
    df = raw[[colmap["open"], colmap["high"], colmap["low"], colmap["close"], colmap["volume"]]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df.dropna().tail(limit)


def _load_ccxt(exchange, symbol, timeframe, limit):
    import ccxt

    ex = getattr(ccxt, exchange)()
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OHLCV into the stationary feature set (index-aligned)."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(close / close.shift(1))
    out["atr"] = _atr(df, ATR_PERIOD)
    out["volatility_ratio"] = out["atr"] / close
    out["norm_rsi"] = _norm_rsi(close, RSI_PERIOD)
    return out


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR, seeded by an EMA of the first ``period`` true ranges."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _norm_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI on a 0..1 scale (100 RSI == 1.0)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.replace([np.inf, -np.inf], 100.0)  # all-gain window -> RSI 100
    return rsi / 100.0
