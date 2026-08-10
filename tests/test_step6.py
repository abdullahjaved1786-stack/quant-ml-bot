"""Unit tests for Step 6: Cost-aware backtest + Deflated Sharpe Ratio."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    compute_slippage,
    backtest_cost_curve,
    deflated_sharpe_ratio,
)


# =============================================================================
# compute_slippage
# =============================================================================


def test_compute_slippage_basic():
    """Slippage = k * sqrt(size / vma)."""
    slippage = compute_slippage(1.0, 10_000.0, k=0.0001)
    assert slippage == pytest.approx(0.0001 * math.sqrt(1.0 / 10_000.0))


def test_compute_slippage_vma_zero():
    """When vma is zero, slippage should be zero."""
    assert compute_slippage(1.0, 0.0) == 0.0


def test_compute_slippage_negative_size():
    """Slippage should be non-negative even with negative size."""
    s = compute_slippage(-5.0, 1000.0)
    assert s >= 0.0


def test_compute_slippage_uses_config_k():
    """When k is None, use config.slippage_k."""
    from src.config import config
    s = compute_slippage(1.0, 10_000.0)
    expected = config.slippage_k * math.sqrt(1.0 / 10_000.0)
    assert s == pytest.approx(expected)


# =============================================================================
# backtest_cost_curve
# =============================================================================


def _make_signal_data(n=500, seed=0):
    """Synthetic price, signals, volume."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.2, n)), index=dates)
    signal = pd.Series(np.where(close > close.shift(1), 1, -1), index=dates).fillna(0)
    signal.iloc[:10] = 0  # warm-up
    vma = pd.Series(np.full(n, 50_000.0), index=dates)
    return close, signal, vma


def test_backtest_result_has_all_keys():
    """BacktestResult must contain all required metrics."""
    close, signal, vma = _make_signal_data()
    result = backtest_cost_curve(signal, close, size_series=1.0, vma=vma)
    assert result.total_return != 0.0 or result.total_trades == 0
    assert hasattr(result, "deflated_sharpe")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "cost_usd")
    assert result.max_drawdown >= 0.0


def test_backtest_positive_cost():
    """Any non-zero position flip must incur positive cost."""
    close, signal, vma = _make_signal_data()
    result = backtest_cost_curve(signal, close, size_series=1.0, vma=vma)
    assert result.cost_usd >= 0.0
    if result.total_trades > 0:
        assert result.cost_usd > 0.0


def test_backtest_no_trades_no_cost():
    """All-zero signals → 0 trades, 0 cost."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(100.0, index=dates)
    signal = pd.Series(0, index=dates, dtype=int)
    vma = pd.Series(10_000.0, index=dates)
    result = backtest_cost_curve(signal, close, 1.0, vma)
    assert result.total_trades == 0
    assert result.cost_usd == 0.0


def test_backtest_equity_curve_length():
    """Equity curve length must equal input signal length."""
    close, signal, vma = _make_signal_data()
    result = backtest_cost_curve(signal, close, 1.0, vma)
    assert len(result.net_equity_curve) == len(signal)


def test_backtest_higher_slippage_reduces_return():
    """Increasing slippage k should lower net return."""
    close, signal, vma = _make_signal_data()
    low = backtest_cost_curve(signal, close, 1.0, vma, slippage_k=0.0)
    high = backtest_cost_curve(signal, close, 1.0, vma, slippage_k=0.01)
    assert high.total_return <= low.total_return


# =============================================================================
# deflated_sharpe_ratio
# =============================================================================


def test_deflated_sharpe_high_sr_positive_dsr():
    """A high observed SR with few trials should give DSR close to 1."""
    dsr = deflated_sharpe_ratio(2.0, num_trials=10, sr_std=0.1)
    assert dsr > 0.9


def test_deflated_sharpe_low_sr_low_dsr():
    """A very low SR with many trials should give DSR close to 0."""
    dsr = deflated_sharpe_ratio(0.1, num_trials=1000, sr_std=0.3)
    assert dsr < 0.6


def test_deflated_sharpe_more_trials_reduces_dsr():
    """More trials (more testing bias) should reduce DSR for the same SR."""
    dsr_few = deflated_sharpe_ratio(1.0, num_trials=10, sr_std=0.2)
    dsr_many = deflated_sharpe_ratio(1.0, num_trials=1000, sr_std=0.2)
    assert dsr_many < dsr_few


def test_deflated_sharpe_skewness_kurtosis_adjustment():
    """Positive skewness and low kurtosis should give higher DSR."""
    base = deflated_sharpe_ratio(1.0, num_trials=50, sr_std=0.2, skewness=0.0, kurtosis=3.0)
    pos_skew = deflated_sharpe_ratio(1.0, num_trials=50, sr_std=0.2, skewness=2.0, kurtosis=3.0)
    assert pos_skew >= base  # positive skew makes the SR more credible


def test_deflated_sharpe_zero_std():
    """When sr_std is zero, DSR should still return a finite value (no crash)."""
    dsr = deflated_sharpe_ratio(1.5, num_trials=10, sr_std=0.0)
    assert math.isfinite(dsr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
