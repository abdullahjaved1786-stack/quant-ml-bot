"""Unit tests for Step 12: Microstructure — Order Book Imbalance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.microstructure import (
    compute_obi,
    obi_series,
    compute_trade_imbalance,
    make_synthetic_orderbook,
)


# =============================================================================
# compute_obi
# =============================================================================


def test_obi_balanced():
    """Equal bid/ask volumes → OBI = 0."""
    obi = compute_obi([100, 100], [100, 100])
    assert obi == pytest.approx(0.0)


def test_obi_all_bid():
    """All volume on bid side → OBI = 1."""
    obi = compute_obi([200, 300], [0, 0])
    assert obi == pytest.approx(1.0)


def test_obi_all_ask():
    """All volume on ask side → OBI = -1."""
    obi = compute_obi([0, 0], [200, 300])
    assert obi == pytest.approx(-1.0)


def test_obi_asymmetric():
    """More bid than ask → positive OBI."""
    obi = compute_obi([300, 100], [50, 50])
    assert obi > 0.0


def test_obi_levels_truncation():
    """levels param must limit the number of depth levels summed."""
    bids = [100, 200, 300, 400]
    asks = [100, 200, 300, 400]
    obi_2 = compute_obi(bids, asks, levels=2)
    assert obi_2 == pytest.approx(0.0)  # bids=300, asks=300 → balanced
    obi_4 = compute_obi(bids, asks, levels=4)
    assert obi_4 == pytest.approx(0.0)


def test_obi_empty_returns_zero():
    """Both sides zero → OBI = 0."""
    assert compute_obi([0, 0], [0, 0]) == 0.0


def test_obi_numpy_input():
    """Numpy arrays should work just like lists."""
    obi = compute_obi(np.array([100.0, 50.0]), np.array([50.0, 100.0]))
    assert -1.0 <= obi <= 1.0


# =============================================================================
# obi_series
# =============================================================================


def test_obi_series_output_length():
    """Output length must match input length."""
    bids, asks = make_synthetic_orderbook(n_rows=100, n_levels=5, seed=0)
    series = obi_series(bids, asks)
    assert len(series) == 100


def test_obi_series_values_in_range():
    """All OBI values must be in [-1, 1]."""
    bids, asks = make_synthetic_orderbook(n_rows=200, n_levels=3, seed=1)
    series = obi_series(bids, asks)
    assert (series >= -1.0).all()
    assert (series <= 1.0).all()


def test_obi_series_levels_param():
    """levels param must work in obi_series."""
    bids, asks = make_synthetic_orderbook(n_rows=50, n_levels=5, seed=2)
    s1 = obi_series(bids, asks, levels=2)
    s2 = obi_series(bids, asks, levels=5)
    # Different levels → different OBI values (usually)
    assert not s1.equals(s2)


def test_obi_series_name():
    """Output Series must be named 'obi'."""
    bids, asks = make_synthetic_orderbook(n_rows=10, seed=3)
    s = obi_series(bids, asks)
    assert s.name == "obi"


# =============================================================================
# compute_trade_imbalance
# =============================================================================


def test_trade_imbalance_balanced():
    """Equal buy/sell → imbalance = 0."""
    assert compute_trade_imbalance([100, 100], [100, 100]) == pytest.approx(0.0)


def test_trade_imbalance_all_buy():
    """All buys → imbalance = 1."""
    assert compute_trade_imbalance([200, 300], [0, 0]) == pytest.approx(1.0)


def test_trade_imbalance_all_sell():
    """All sells → imbalance = -1."""
    assert compute_trade_imbalance([0, 0], [200, 300]) == pytest.approx(-1.0)


def test_trade_imbalance_empty():
    """Both zero → 0."""
    assert compute_trade_imbalance([0, 0], [0, 0]) == 0.0


def test_trade_imbalance_range():
    """Trade imbalance must always be in [-1, 1]."""
    val = compute_trade_imbalance([75, 25], [100, 50])
    assert -1.0 <= val <= 1.0


# =============================================================================
# make_synthetic_orderbook
# =============================================================================


def test_make_synthetic_orderbook_shape():
    """Synthetic orderbook must have correct dimensions."""
    bids, asks = make_synthetic_orderbook(n_rows=80, n_levels=4, seed=10)
    assert bids.shape == (80, 4)
    assert asks.shape == (80, 4)


def test_make_synthetic_orderbook_reproducible():
    """Same seed must produce identical output."""
    b1, a1 = make_synthetic_orderbook(n_rows=50, seed=99)
    b2, a2 = make_synthetic_orderbook(n_rows=50, seed=99)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(a1, a2)


def test_make_synthetic_orderbook_positive_volumes():
    """All volumes must be positive."""
    bids, asks = make_synthetic_orderbook(n_rows=100, seed=11)
    assert (bids.values > 0).all()
    assert (asks.values > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
