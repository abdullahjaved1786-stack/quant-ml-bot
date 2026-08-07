"""Live OHLCV fetching via CCXT public API with retry/backoff for rate limits."""

from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger("quant_bot")

DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_LIMIT = 500
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRYABLE_EXCEPTIONS = ()


def fetch_live_ohlcv(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = DEFAULT_LIMIT,
    exchange_id: str = DEFAULT_EXCHANGE,
) -> pd.DataFrame:
    """Fetch public OHLCV candles from CCXT with retry/backoff.

    Returns columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    The DataFrame index is also set to timestamp so existing feature code works.
    """
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    retryable = (
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
        ccxt.RateLimitExceeded,
        TimeoutError,
        ConnectionError,
    )
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw:
                raise RuntimeError(f"{exchange_id} returned no OHLCV rows for {symbol}")

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            if df.empty:
                raise RuntimeError(f"{exchange_id} returned only empty OHLCV rows for {symbol}")
            return df.set_index("timestamp", drop=False).sort_index()

        except retryable as exc:
            last_error = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(
                "Retryable CCXT error for %s %s on %s (%d/%d): %s; retrying in %.1fs",
                symbol,
                timeframe,
                exchange_id,
                attempt,
                MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
        except ccxt.ExchangeError as exc:
            raise RuntimeError(f"Exchange error from {exchange_id} for {symbol}: {exc}") from exc

    raise RuntimeError(
        f"Failed to fetch {symbol} {timeframe} from {exchange_id} after {MAX_RETRIES} attempts: {last_error}"
    )
