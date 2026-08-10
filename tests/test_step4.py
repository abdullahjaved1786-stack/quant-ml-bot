"""Unit tests for Step 4: Volatility regime filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime import classify_regime, filter_by_regime, RegimeFilter


def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close + rng.normal(0, 0.05, n)
    volume = rng.integers(100, 10_000, n).astype(float)
    return pd.DataFrame({
        "timestamp": dates,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }).set_index("timestamp")


# =============================================================================
# classify_regime
# =============================================================================


def test_classify_regime_valid_labels():
    """Every non-NaN entry must be one of LOW/MEDIUM/HIGH."""
    r = pd.Series([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    labels = classify_regime(r, low_pct=0.2, high_pct=0.8)
    valid = labels.dropna()
    assert set(valid.unique()) <= {"LOW", "MEDIUM", "HIGH"}


def test_classify_regime_thresholds():
    """Values below low_pct → LOW; above high_pct → HIGH; else MEDIUM."""
    r = pd.Series([0.0, 0.15, 0.5, 0.85, 1.0])
    labels = classify_regime(r, low_pct=0.2, high_pct=0.8)
    assert labels.iloc[0] == "LOW"
    assert labels.iloc[1] == "LOW"
    assert labels.iloc[2] == "MEDIUM"
    assert labels.iloc[3] == "HIGH"
    assert labels.iloc[4] == "HIGH"


def test_classify_regime_nan_stays_nan():
    """NaN inputs → NaN output."""
    r = pd.Series([0.5, np.nan, 0.8])
    labels = classify_regime(r, low_pct=0.2, high_pct=0.9)
    assert pd.isna(labels.iloc[1])
    assert labels.iloc[0] == "MEDIUM"


# =============================================================================
# filter_by_regime
# =============================================================================


def test_filter_by_regime_only_allowed():
    """Only regimes in allowed should be True."""
    regimes = pd.Series(["LOW", "MEDIUM", "HIGH", "MEDIUM"])
    mask = filter_by_regime(regimes, allowed=("MEDIUM",))
    assert mask.tolist() == [False, True, False, True]


def test_filter_by_regime_multiple_allowed():
    """Multiple regimes in allowed → all return True."""
    regimes = pd.Series(["LOW", "MEDIUM", "HIGH"])
    mask = filter_by_regime(regimes, allowed=("LOW", "HIGH"))
    assert mask.tolist() == [True, False, True]


def test_filter_by_regime_empty_allowed():
    """Empty allowed → all False."""
    regimes = pd.Series(["LOW", "MEDIUM"])
    mask = filter_by_regime(regimes, allowed=())
    assert (mask == False).all()  # noqa: E712


# =============================================================================
# RegimeFilter class
# =============================================================================


def test_regime_filter_fit_transform():
    """fit_transform should add volatility_regime and tradable columns."""
    df = _make_ohlcv(n=200, seed=5)
    rf = RegimeFilter(low_pct=0.2, high_pct=0.8, allowed=("MEDIUM",))
    out = rf.fit_transform(df)
    assert "volatility_regime" in out.columns
    assert "tradable" in out.columns
    assert len(out) == len(df)
    assert rf.regime_ is not None


def test_regime_filter_custom_allowed():
    """Custom allowed tuple should propagate to tradable mask."""
    df = _make_ohlcv(n=200, seed=7)
    rf = RegimeFilter(low_pct=0.2, high_pct=0.8, allowed=("LOW", "MEDIUM"))
    out = rf.fit_transform(df)
    # tradable = True only for LOW or MEDIUM
    high_mask = out["volatility_regime"] == "HIGH"
    assert (out.loc[high_mask, "tradable"] == False).all()  # noqa: E712


def test_regime_filter_returns_copy():
    """transform must not mutate the original DataFrame."""
    df = _make_ohlcv(n=50, seed=9)
    original_cols = list(df.columns)
    RegimeFilter().fit_transform(df)
    assert list(df.columns) == original_cols


def test_regime_filter_fit_then_transform():
    """Calling fit() then transform() should work the same as fit_transform()."""
    df = _make_ohlcv(n=100, seed=11)
    rf1 = RegimeFilter()
    out1 = rf1.fit_transform(df)
    rf2 = RegimeFilter()
    rf2.fit(df)
    out2 = rf2.transform(df)
    pd.testing.assert_frame_equal(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
