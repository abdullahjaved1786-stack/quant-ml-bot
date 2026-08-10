"""Triple Barrier Method labeling with lookahead-bias prevention.

Implements Marcos López de Prado's Triple Barrier Method:
- TP barrier: price hits upper threshold (take profit)
- SL barrier: price hits lower threshold (stop loss)
- Vertical barrier: max holding period expires

Labels:
- +1: TP hit first (profitable)
- -1: SL hit first (loss)
-  0: Vertical barrier (time expiry)

Uses shared compute_atr() from src.data to prevent logic drift.
Shifts ATR by 1 bar to avoid lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config
from src.data import compute_atr


def apply_triple_barrier(
    df: pd.DataFrame,
    events: pd.DataFrame | None = None,
    vol: pd.Series | None = None,
    pt_mult: float | None = None,
    sl_mult: float | None = None,
    max_hold: int | None = None,
) -> pd.DataFrame:
    """Apply Triple Barrier labeling to OHLCV data.

    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume.
        events: Optional DataFrame with 'entry_time' column (datetime).
                If None, uses all bars as potential entries.
        vol: Optional pre-computed volatility series (ATR).
             If None, computes from df using shared compute_atr().
        pt_mult: TP barrier multiplier for ATR (default: config.pt_mult).
        sl_mult: SL barrier multiplier for ATR (default: config.sl_mult).
        max_hold: Maximum holding period in bars (default: config.max_hold).

    Returns:
        DataFrame with columns:
        - entry_time: When the trade was entered
        - exit_time (t1): When the trade was exited
        - realized_return: Percentage return from entry to exit
        - label: +1 (TP), -1 (SL), 0 (time expiry)
        - hit_time_limit: Boolean, True if vertical barrier was hit
    """
    pt_mult = pt_mult if pt_mult is not None else config.pt_mult
    sl_mult = sl_mult if sl_mult is not None else config.sl_mult
    max_hold = max_hold if max_hold is not None else config.max_hold

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Compute ATR with lookahead prevention (shift by 1)
    if vol is None:
        vol = compute_atr(df, window=config.atr_period)
    vol = vol.shift(1)  # Prevent lookahead bias

    # Set up entry events
    if events is None:
        entries = pd.DataFrame({"entry_time": df.index})
        entries["entry_idx"] = np.arange(len(df))
    else:
        entries = events[["entry_time"]].copy()
        # Map entry times to indices
        entries["entry_idx"] = entries["entry_time"].map(
            {t: i for i, t in enumerate(df.index)}
        )

    results = []

    for _, row in entries.iterrows():
        entry_idx_raw = row["entry_idx"]
        # Skip entries with missing index (e.g., event timestamp not in df.index)
        if pd.isna(entry_idx_raw):
            continue
        entry_idx = int(entry_idx_raw)
        if entry_idx >= len(df) or pd.isna(vol.iloc[entry_idx]):
            continue

        entry_price = close.iloc[entry_idx]
        atr = vol.iloc[entry_idx]
        tp_dist = pt_mult * atr
        sl_dist = sl_mult * atr

        tp_price = entry_price + tp_dist
        sl_price = entry_price - sl_dist

        # Scan forward up to max_hold bars
        exit_idx = entry_idx
        label = 0  # default: time expiry
        hit_time_limit = True

        for j in range(1, max_hold + 1):
            if entry_idx + j >= len(df):
                break

            high_j = high.iloc[entry_idx + j]
            low_j = low.iloc[entry_idx + j]
            close_j = close.iloc[entry_idx + j]

            hit_tp = high_j >= tp_price
            hit_sl = low_j <= sl_price

            if hit_tp and hit_sl:
                # Both hit in same bar — use close to determine direction
                if close_j >= entry_price:
                    label = 1
                    hit_time_limit = False
                else:
                    label = -1
                    hit_time_limit = False
                exit_idx = entry_idx + j
                break
            elif hit_tp:
                label = 1
                hit_time_limit = False
                exit_idx = entry_idx + j
                break
            elif hit_sl:
                label = -1
                hit_time_limit = False
                exit_idx = entry_idx + j
                break

        # If no barrier hit, exit at vertical barrier
        if hit_time_limit:
            final_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_idx = final_idx
            exit_price = close.iloc[exit_idx]
            realized_return = (exit_price / entry_price - 1)
            # Label by sign of realized return
            label = 1 if realized_return > 0 else (-1 if realized_return < 0 else 0)
        else:
            exit_price = close.iloc[exit_idx]
            realized_return = (exit_price / entry_price - 1)

        results.append({
            "entry_time": df.index[entry_idx],
            "exit_time": df.index[exit_idx],
            "realized_return": realized_return,
            "label": label,
            "hit_time_limit": hit_time_limit,
        })

    if not results:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "realized_return", "label", "hit_time_limit"]
        )

    return pd.DataFrame(results)


def compute_volatility_regime(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Compute volatility regime as rolling ATR percentile.

    Used by src/regime.py as a feature column.

    Args:
        df: OHLCV DataFrame.
        window: Rolling window for percentile calculation.

    Returns:
        Series with values in [0, 1] representing ATR percentile rank.
    """
    atr = compute_atr(df, window=config.atr_period)
    rolling_atr = atr.rolling(window=window)
    return rolling_atr.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
