"""Triple-Barrier labeling (Marcos López de Prado, *Advances in Financial ML*)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TP_ATR = 1.5  # upper barrier = close + 1.5 * ATR (take-profit)
SL_ATR = 1.0  # lower barrier = close - 1.0 * ATR (stop-loss)
VERTICAL_BARRIER = 5  # time expiry in bars


def triple_barrier(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    tp_atr: float = TP_ATR,
    sl_atr: float = SL_ATR,
    vertical: int = VERTICAL_BARRIER,
) -> pd.Series:
    """Label each bar t from the outcome over the next ``vertical`` bars.

    +1 if the upper barrier is touched first (TP hit),
    -1 if the lower barrier is touched first (SL hit),
     0 if neither is touched within ``vertical`` bars (time expiry).
    Bars whose full lookahead window is unavailable are NaN (excluded by dropna).

    Same-bar ties resolve to the upper barrier — intrabar order is unknowable
    from OHLCV alone.
    """
    labels = np.full(len(close), np.nan)
    for t in range(len(close) - vertical):
        upper = close.iloc[t] + tp_atr * atr.iloc[t]
        lower = close.iloc[t] - sl_atr * atr.iloc[t]
        if np.isnan(upper):
            continue  # feature warmup row — no ATR yet
        fut_high = high.iloc[t + 1 : t + 1 + vertical].to_numpy()
        fut_low = low.iloc[t + 1 : t + 1 + vertical].to_numpy()
        for i in range(vertical):
            if fut_high[i] >= upper:
                labels[t] = 1
                break
            if fut_low[i] <= lower:
                labels[t] = -1
                break
    return pd.Series(labels, index=close.index)
