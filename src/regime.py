"""Volatility regime filter: only trade in favorable volatility conditions.

Uses the ATR-percentile regime series from src.labeling to bucket each bar
into LOW / MEDIUM / HIGH volatility. Strategies can restrict entries to a
subset of regimes — typically the MEDIUM band, where barriers are meaningful
but not chaotic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config
from src.labeling import compute_volatility_regime


def classify_regime(
    regime: pd.Series,
    low_pct: float | None = None,
    high_pct: float | None = None,
) -> pd.Series:
    """Bucket a [0, 1] ATR-percentile series into LOW / MEDIUM / HIGH.

    Args:
        regime: Percentile-rank series in [0, 1] (from compute_volatility_regime).
        low_pct: Below this → LOW regime (default config.atr_percentile_low).
        high_pct: Above this → HIGH regime (default config.atr_percentile_high).

    Returns:
        Series of strings: "LOW", "MEDIUM", "HIGH" (NaN stays NaN).
    """
    low_pct = low_pct if low_pct is not None else config.atr_percentile_low
    high_pct = high_pct if high_pct is not None else config.atr_percentile_high

    out = pd.Series("MEDIUM", index=regime.index, dtype=object)
    out[regime.isna()] = np.nan
    out[regime < low_pct] = "LOW"
    out[regime > high_pct] = "HIGH"
    return out


def filter_by_regime(
    regime: pd.Series,
    allowed: tuple[str, ...] = ("MEDIUM",),
) -> pd.Series:
    """Boolean mask: True where the regime is in `allowed`.

    Args:
        regime: Classified regime series (LOW/MEDIUM/HIGH).
        allowed: Regimes that are permitted to trade.

    Returns:
        Boolean Series aligned with regime.
    """
    return regime.isin(allowed)


class RegimeFilter:
    """Compute the regime series and expose a trading mask.

    Args:
        low_pct: LOW threshold (default config.atr_percentile_low).
        high_pct: HIGH threshold (default config.atr_percentile_high).
        allowed: Regimes allowed to trade (default ("MEDIUM",)).
        window: Rolling window for the ATR percentile (default 20).
    """

    def __init__(
        self,
        low_pct: float | None = None,
        high_pct: float | None = None,
        allowed: tuple[str, ...] = ("MEDIUM",),
        window: int = 20,
    ):
        self.low_pct = low_pct if low_pct is not None else config.atr_percentile_low
        self.high_pct = high_pct if high_pct is not None else config.atr_percentile_high
        self.allowed = allowed
        self.window = window
        self.regime_: pd.Series | None = None
        self.mask_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "RegimeFilter":
        """Compute regime + trading mask for an OHLCV DataFrame."""
        percentile = compute_volatility_regime(df, window=self.window)
        self.regime_ = classify_regime(percentile, self.low_pct, self.high_pct)
        self.mask_ = filter_by_regime(self.regime_, self.allowed)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform in one call."""
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with regime/mask columns attached (allows trade or not)."""
        if self.regime_ is None:
            self.fit(df)
        out = df.copy()
        out["volatility_regime"] = self.regime_
        out["tradable"] = self.mask_
        return out
