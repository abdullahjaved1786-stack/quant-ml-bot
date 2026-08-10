"""Microstructure features: Order Book Imbalance (OBI).

OBI measures the imbalance between bid and ask depth at a given level of the
order book. A positive OBI means more bid-side liquidity, which can indicate
buying pressure.

OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

Ranges from -1 (all ask) to +1 (all bid). OBI near 0 indicates a balanced book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_obi(
    bid_volumes: np.ndarray | list,
    ask_volumes: np.ndarray | list,
    levels: int | None = None,
) -> float:
    """Compute Order Book Imbalance across depth levels.

    Args:
        bid_volumes: Array of bid volumes at each level (best bid first).
        ask_volumes: Array of ask volumes at each level (best ask first).
        levels: Number of levels to include (default: all provided).

    Returns:
        OBI in [-1, 1]. Returns 0.0 if both sides are zero.
    """
    bids = np.asarray(bid_volumes, dtype=float)
    asks = np.asarray(ask_volumes, dtype=float)

    if levels is not None:
        bids = bids[:levels]
        asks = asks[:levels]

    total_bid = float(bids.sum())
    total_ask = float(asks.sum())
    total = total_bid + total_ask

    if total == 0.0:
        return 0.0

    return (total_bid - total_ask) / total


def obi_series(
    bid_volume_frame: pd.DataFrame,
    ask_volume_frame: pd.DataFrame,
    levels: int | None = None,
) -> pd.Series:
    """Compute OBI for each row of a volume DataFrame pair.

    Args:
        bid_volume_frame: DataFrame where each column is a bid depth level.
        ask_volume_frame: DataFrame where each column is an ask depth level.
        levels: Number of levels to sum (default: all columns).

    Returns:
        Series of OBI values in [-1, 1], aligned with the input index.
    """
    n = len(bid_volume_frame)
    obi_vals = np.zeros(n)
    for i in range(n):
        obi_vals[i] = compute_obi(
            bid_volume_frame.iloc[i].values,
            ask_volume_frame.iloc[i].values,
            levels=levels,
        )
    return pd.Series(obi_vals, index=bid_volume_frame.index, name="obi")


def compute_trade_imbalance(
    buy_volumes: np.ndarray | list,
    sell_volumes: np.ndarray | list,
) -> float:
    """Trade flow imbalance (buy vs sell volume at the tick/bar level).

    Unlike OBI (which uses the order book), this uses executed trades.

    Args:
        buy_volumes: Cumulative buy-initiated volume per bar.
        sell_volumes: Cumulative sell-initiated volume per bar.

    Returns:
        Trade imbalance in [-1, 1].
    """
    buys = float(np.sum(np.asarray(buy_volumes, dtype=float)))
    sells = float(np.sum(np.asarray(sell_volumes, dtype=float)))
    total = buys + sells
    if total == 0.0:
        return 0.0
    return (buys - sells) / total


def make_synthetic_orderbook(
    n_rows: int = 100,
    n_levels: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic bid/ask volume DataFrames for testing.

    Args:
        n_rows: Number of order book snapshots.
        n_levels: Depth levels per side.
        seed: RNG seed.

    Returns:
        (bid_volumes, ask_volumes) DataFrames with columns level_0..level_{n-1}.
    """
    rng = np.random.default_rng(seed)
    bids = rng.integers(10, 1000, size=(n_rows, n_levels)).astype(float)
    asks = rng.integers(10, 1000, size=(n_rows, n_levels)).astype(float)
    idx = pd.RangeIndex(n_rows)
    cols = [f"level_{i}" for i in range(n_levels)]
    return pd.DataFrame(bids, index=idx, columns=cols), pd.DataFrame(asks, index=idx, columns=cols)
