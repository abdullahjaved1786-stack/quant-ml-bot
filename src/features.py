"""Feature engineering: fractional differentiation.

Implements the fractional differentiation filter from López de Prado (2018)
"Advances in Financial Machine Learning", Chapter 5.

Fractional differentiation removes only enough memory to make a time series
stationary, preserving more signal than integer differencing (d=1) which
often removes too much information.

The weight for lag k is:  w_k = w_{k-1} × (d - k + 1) / k
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def frac_diff_weights(d: float, k: int) -> np.ndarray:
    """Compute fractional differentiation weights for lags 0..k-1.

    Args:
        d: Fractional differentiation degree (0 = no diff, 1 = first diff).
        k: Number of weights (lags to include).

    Returns:
        Array of weights (length k).
    """
    if k <= 0:
        return np.array([], dtype=float)
    weights = np.zeros(k)
    weights[0] = 1.0
    for i in range(1, k):
        weights[i] = weights[i - 1] * (i - 1 - d) / i
    return weights


def frac_diff(
    series: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-4,
    max_lag: int | None = None,
) -> pd.Series:
    """Apply fractional differentiation to a pandas Series.

    Args:
        series: Input time series.
        d: Fractional differentiation degree (default config.frac_diff_d).
        threshold: Minimum weight magnitude — lags beyond this are dropped
                   to keep the window finite (López de Prado thresholding).
        max_lag: Hard cap on window length (default: as many lags as needed
                 for weights to decay below `threshold`).

    Returns:
        Fractionally differentiated series (same index, shorter by initial window).
    """
    series = series.dropna().copy()
    n = len(series)
    if n == 0:
        return series

    # Determine window length: iterate until weights decay below threshold
    if max_lag is None:
        max_lag = n
    weights = frac_diff_weights(d, max_lag)
    # Find first index where |weight| < threshold (after lag 0)
    for w_idx in range(1, len(weights)):
        if abs(weights[w_idx]) < threshold:
            max_lag = w_idx
            break
    weights = weights[:max_lag]
    w = len(weights)

    # Vectorised sliding dot product
    # Convention: w_k applied to x[t-k], so w_0 (newest) is applied to
    # the rightmost value in the window — i.e. weights are reversed in dot.
    values = series.values.astype(float)
    weights_rev = weights[::-1]
    out_len = n - w + 1
    if out_len <= 0:
        return pd.Series(dtype=float, index=series.index[:0])

    out = np.empty(out_len, dtype=float)
    for i in range(out_len):
        out[i] = np.dot(weights_rev, values[i:i + w])

    return pd.Series(out, index=series.index[w - 1:])
