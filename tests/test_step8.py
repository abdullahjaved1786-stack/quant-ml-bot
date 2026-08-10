"""Unit tests for Step 8: Monte Carlo stationary block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    stationary_block_bootstrap,
    bootstrap_confidence_interval,
    bootstrap_sharpe,
)


# =============================================================================
# stationary_block_bootstrap
# =============================================================================


def test_bootstrap_output_shape():
    """Output must be (n_bootstrap, len(series))."""
    series = np.random.default_rng(0).normal(0, 0.01, 200)
    bs = stationary_block_bootstrap(series, n_bootstrap=50, block_len=10, seed=42)
    assert bs.shape == (50, 200)


def test_bootstrap_preserves_length():
    """Each bootstrap row must be the same length as the input."""
    series = np.random.default_rng(1).normal(0, 0.01, 150)
    bs = stationary_block_bootstrap(series, n_bootstrap=20, block_len=5, seed=7)
    for i in range(20):
        assert len(bs[i]) == 150


def test_bootstrap_blocks_reuse_original_values():
    """All values in bootstrap output must come from the original series."""
    series = np.random.default_rng(2).normal(0, 0.01, 100)
    bs = stationary_block_bootstrap(series, n_bootstrap=10, block_len=8, seed=3)
    for val in bs.flat:
        assert val in series


def test_bootstrap_different_seeds_different_output():
    """Different seeds must produce different bootstrap series."""
    series = np.random.default_rng(4).normal(0, 0.01, 100)
    bs1 = stationary_block_bootstrap(series, n_bootstrap=5, block_len=10, seed=10)
    bs2 = stationary_block_bootstrap(series, n_bootstrap=5, block_len=10, seed=20)
    assert not np.array_equal(bs1, bs2)


def test_bootstrap_same_seed_same_output():
    """Same seed must produce identical output."""
    series = np.random.default_rng(5).normal(0, 0.01, 80)
    bs1 = stationary_block_bootstrap(series, n_bootstrap=5, block_len=6, seed=99)
    bs2 = stationary_block_bootstrap(series, n_bootstrap=5, block_len=6, seed=99)
    np.testing.assert_array_equal(bs1, bs2)


def test_bootstrap_block_length_uses_config_default():
    """When block_len is None, config.block_bootstrap_block_len is used."""
    from src.config import config
    series = np.random.default_rng(6).normal(0, 0.01, 100)
    bs = stationary_block_bootstrap(series, n_bootstrap=3, block_len=None, seed=0)
    assert bs.shape[1] == 100
    # We can't directly check the internal block length, but shape must be correct.


# =============================================================================
# bootstrap_confidence_interval
# =============================================================================


def test_confidence_interval_contains_observed():
    """Observed statistic should lie within the CI (for well-behaved data)."""
    rng = np.random.default_rng(7)
    series = rng.normal(0.5, 1.0, 300)
    result = bootstrap_confidence_interval(series, np.mean, n_bootstrap=200, seed=42)
    assert result["ci_lower"] <= result["observed"] <= result["ci_upper"]


def test_confidence_interval_keys():
    """Result dict must have all required keys."""
    series = np.random.default_rng(8).normal(0, 1, 100)
    result = bootstrap_confidence_interval(series, np.mean, n_bootstrap=50, seed=0)
    assert {"observed", "ci_lower", "ci_upper", "p_value", "bootstrap_stats"}.issubset(set(result.keys()))


def test_confidence_interval_wider_for_less_data():
    """CI should be wider with fewer observations (more uncertainty)."""
    rng = np.random.default_rng(9)
    big = rng.normal(0, 1, 500)
    small = rng.normal(0, 1, 50)
    ci_big = bootstrap_confidence_interval(big, np.mean, n_bootstrap=100, seed=0)
    ci_small = bootstrap_confidence_interval(small, np.mean, n_bootstrap=100, seed=0)
    assert (ci_small["ci_upper"] - ci_small["ci_lower"]) >= (
        ci_big["ci_upper"] - ci_big["ci_lower"]
    )


def test_confidence_interval_p_value_range():
    """p_value must be in [0, 1]."""
    series = np.random.default_rng(10).normal(0, 1, 200)
    result = bootstrap_confidence_interval(series, np.mean, n_bootstrap=100, seed=0)
    assert 0.0 <= result["p_value"] <= 1.0


# =============================================================================
# bootstrap_sharpe
# =============================================================================


def test_bootstrap_sharpe_returns_dict():
    """bootstrap_sharpe must return observed Sharpe, CI, and p-value."""
    returns = np.random.default_rng(11).normal(0, 0.01, 200)
    result = bootstrap_sharpe(returns, n_bootstrap=100, seed=42)
    assert "observed" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "p_value" in result
    assert np.isfinite(result["observed"])


def test_bootstrap_sharpe_zero_vol():
    """Zero-volatility returns should still not crash."""
    returns = np.zeros(100)
    result = bootstrap_sharpe(returns, n_bootstrap=50, seed=0)
    assert result["observed"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
