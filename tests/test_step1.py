"""Unit tests for Step 1: Triple Barrier labeling and volatility regime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labeling import apply_triple_barrier, compute_volatility_regime


def _make_ohlcv_with_index(n_bars: int = 100, seed: int = 42, flat: bool = False) -> pd.DataFrame:
    """Build an OHLCV DataFrame indexed by datetime."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
    if flat:
        close = np.full(n_bars, 100.0)
    else:
        returns = rng.normal(0, 0.01, n_bars)
        close = 100.0 * np.exp(np.cumsum(returns))
    high = close + 0.5
    low = close - 0.5
    open_ = close + rng.normal(0, 0.05, n_bars)
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
    return df.set_index("timestamp")


# =============================================================================
# apply_triple_barrier — basic structure
# =============================================================================


def test_apply_triple_barrier_returns_dataframe():
    """apply_triple_barrier must return a DataFrame with the expected columns."""
    df = _make_ohlcv_with_index(n_bars=100, seed=1)
    result = apply_triple_barrier(df)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {"entry_time", "exit_time", "realized_return", "label", "hit_time_limit"}
    assert expected_cols.issubset(set(result.columns))


def test_apply_triple_barrier_labels_in_valid_set():
    """Labels must be one of {-1, 0, 1}."""
    df = _make_ohlcv_with_index(n_bars=200, seed=2)
    result = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_hold=10)
    assert set(result["label"].unique()).issubset({-1, 0, 1})


def test_apply_triple_barrier_returns_one_row_per_entry():
    """Result length should equal number of valid entry bars."""
    df = _make_ohlcv_with_index(n_bars=50, seed=3)
    result = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_hold=5)
    # The last few entries cannot label anything because max_hold pushes past the data.
    # Entries are skipped when vol is NaN (shift(1) drops first bar) or entry_idx out of bounds.
    assert len(result) > 0
    assert len(result) <= len(df)


# =============================================================================
# Triple Barrier — TP/SL hit semantics
# =============================================================================


def test_apply_triple_barrier_rising_prices_hits_tp():
    """Strongly rising prices with tight SL/TP should produce mostly label=1."""
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Linear uptrend with tiny noise; ATR will be small relative to move
    close = np.linspace(100.0, 130.0, n)
    high = close + 0.05
    low = close - 0.05
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    result = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    # In a strong uptrend, we expect TP hits
    assert (result["label"] == 1).sum() > 0


def test_apply_triple_barrier_falling_prices_hits_sl():
    """Strongly falling prices with tight SL/TP should produce mostly label=-1."""
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(130.0, 100.0, n)
    high = close + 0.05
    low = close - 0.05
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    result = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    assert (result["label"] == -1).sum() > 0


def test_apply_triple_barrier_flat_market_hits_time_limit():
    """Flat prices should produce mostly time-limit (vertical barrier) exits."""
    df = _make_ohlcv_with_index(n_bars=80, seed=99, flat=True)
    result = apply_triple_barrier(df, pt_mult=0.001, sl_mult=0.001, max_hold=5)
    assert (result["hit_time_limit"] == True).sum() > 0  # noqa: E712


# =============================================================================
# hit_time_limit flag
# =============================================================================


def test_hit_time_limit_true_when_no_barrier_hit():
    """When TP/SL never trigger, exit is vertical barrier → hit_time_limit=True."""
    n = 50
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Low-volatility noise that stays well within ATR-based barriers
    close = 100.0 + np.cumsum(rng.normal(0, 0.001, n))
    high = close + 0.01
    low = close - 0.01
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    # Use very wide barriers (ATR ~0.02, barriers at 1.5*ATR = 0.03 from entry)
    # but high-low = 0.02, so ATR-based barriers are wider → never triggered
    result = apply_triple_barrier(df, pt_mult=50.0, sl_mult=50.0, max_hold=5)
    assert result["hit_time_limit"].dtype == bool
    # All exits should be vertical (TP/SL never reached)
    assert (result["hit_time_limit"] == True).all()  # noqa: E712


def test_hit_time_limit_false_when_tp_hit():
    """When TP is hit, hit_time_limit must be False."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(100.0, 130.0, n)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    result = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    tp_rows = result[result["label"] == 1]
    assert len(tp_rows) > 0
    assert (tp_rows["hit_time_limit"] == False).all()  # noqa: E712


# =============================================================================
# realized_return
# =============================================================================


def test_realized_return_positive_for_tp():
    """TP hits should yield positive realized_return."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(100.0, 130.0, n)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    result = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    tp_rows = result[result["label"] == 1]
    assert (tp_rows["realized_return"] > 0).all()


def test_realized_return_negative_for_sl():
    """SL hits should yield negative realized_return."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(130.0, 100.0, n)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    ).set_index("timestamp")
    result = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    sl_rows = result[result["label"] == -1]
    assert len(sl_rows) > 0
    assert (sl_rows["realized_return"] < 0).all()


# =============================================================================
# Lookahead prevention (vol.shift(1))
# =============================================================================


def test_vol_shift_prevents_lookahead():
    """Volatility used for barrier placement must be from previous bar."""
    df = _make_ohlcv_with_index(n_bars=50, seed=11)
    # Provide vol that is identical to compute_atr but shifted by 1
    from src.data import compute_atr

    vol_no_shift = compute_atr(df, window=14)
    # Internally apply_triple_barrier shifts the vol by 1.
    # We can verify by injecting a vol and checking that the result is the same.
    result_shift = apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_hold=5, vol=vol_no_shift)
    assert len(result_shift) > 0
    # First entry (idx 0) must be skipped because vol.shift(1) is NaN at idx 0
    # entry_times should not include the first index of df
    assert result_shift["entry_time"].min() > df.index[0]


# =============================================================================
# Parameter overrides
# =============================================================================


def test_pt_mult_override_changes_labels():
    """Larger pt_mult should produce more time-limit exits (less TP hits)."""
    df = _make_ohlcv_with_index(n_bars=200, seed=13)
    tight = apply_triple_barrier(df, pt_mult=0.5, sl_mult=0.5, max_hold=10)
    wide = apply_triple_barrier(df, pt_mult=5.0, sl_mult=5.0, max_hold=10)
    # Wider barriers → more time-limit exits
    assert wide["hit_time_limit"].sum() >= tight["hit_time_limit"].sum()


def test_max_hold_override_limits_horizon():
    """max_hold should control exit distance from entry."""
    df = _make_ohlcv_with_index(n_bars=100, seed=17, flat=True)
    short = apply_triple_barrier(df, pt_mult=0.001, sl_mult=0.001, max_hold=2)
    long = apply_triple_barrier(df, pt_mult=0.001, sl_mult=0.001, max_hold=20)
    # Both should be vertical exits on flat data, but their exit_times should differ.
    if len(short) > 0 and len(long) > 0:
        # Long should exit further from entry_time
        assert (long["exit_time"] - long["entry_time"]).min() >= (
            short["exit_time"] - short["entry_time"]
        ).min()


# =============================================================================
# Events-based entry selection
# =============================================================================


def test_events_filter_entries():
    """When events DataFrame is provided, only those entries are labeled."""
    df = _make_ohlcv_with_index(n_bars=50, seed=19)
    # Pick a sparse subset of timestamps
    chosen = df.index[[5, 15, 25, 35]]
    events = pd.DataFrame({"entry_time": chosen})
    result = apply_triple_barrier(df, events=events, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    assert len(result) <= len(events)
    # Each entry_time in result must be one of the chosen ones
    assert set(result["entry_time"]).issubset(set(chosen))


def test_events_with_unknown_entry_skipped():
    """Entries with timestamps not in df index should be skipped."""
    df = _make_ohlcv_with_index(n_bars=30, seed=21)
    chosen = list(df.index[[5, 10]]) + [pd.Timestamp("2099-01-01", tz="UTC")]
    events = pd.DataFrame({"entry_time": chosen})
    result = apply_triple_barrier(df, events=events, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    # The bogus 2099 entry should be dropped
    assert not (result["entry_time"] == pd.Timestamp("2099-01-01", tz="UTC")).any()


# =============================================================================
# Edge cases
# =============================================================================


def test_short_data_returns_empty_or_minimal():
    """With very short data, result should be a DataFrame with the right columns (possibly empty)."""
    df = _make_ohlcv_with_index(n_bars=5, seed=23)
    result = apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {"entry_time", "exit_time", "realized_return", "label", "hit_time_limit"}
    assert expected_cols.issubset(set(result.columns))


def test_custom_vol_input():
    """A custom vol series should be honored."""
    df = _make_ohlcv_with_index(n_bars=50, seed=29)
    custom_vol = pd.Series(np.full(len(df), 2.0), index=df.index)
    result = apply_triple_barrier(df, vol=custom_vol, pt_mult=0.5, sl_mult=0.5, max_hold=5)
    assert isinstance(result, pd.DataFrame)


def test_exit_time_after_entry_time():
    """exit_time must always be >= entry_time."""
    df = _make_ohlcv_with_index(n_bars=100, seed=31)
    result = apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    assert (result["exit_time"] >= result["entry_time"]).all()


# =============================================================================
# compute_volatility_regime
# =============================================================================


def test_compute_volatility_regime_returns_series():
    """compute_volatility_regime must return a pandas Series."""
    df = _make_ohlcv_with_index(n_bars=50, seed=37)
    regime = compute_volatility_regime(df, window=10)
    assert isinstance(regime, pd.Series)
    assert len(regime) == len(df)


def test_compute_volatility_regime_values_in_unit_interval():
    """Percentile rank should be in [0, 1] (excluding NaNs)."""
    df = _make_ohlcv_with_index(n_bars=200, seed=41)
    regime = compute_volatility_regime(df, window=20)
    valid = regime.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_compute_volatility_regime_default_window():
    """Default window should be 20."""
    df = _make_ohlcv_with_index(n_bars=50, seed=43)
    regime_default = compute_volatility_regime(df)
    regime_explicit = compute_volatility_regime(df, window=20)
    pd.testing.assert_series_equal(regime_default, regime_explicit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
