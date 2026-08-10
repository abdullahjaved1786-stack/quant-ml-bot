"""Unit tests for Step 5: Fractional Kelly sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sizing import fractional_kelly, kelly_bet_size, compute_position_size


# =============================================================================
# fractional_kelly
# =============================================================================


def test_fractional_kelly_basic():
    """With 60% win rate, 1:1 risk-reward, quarter-Kelly → (0.6-0.4)*0.25 = 0.05."""
    f = fractional_kelly(0.6, 1.0, kelly_fraction=0.25)
    assert f == pytest.approx(0.05)


def test_fractional_kelly_negative_clipped_to_zero():
    """When Kelly is negative (bad edge), fractional Kelly clips to 0."""
    f = fractional_kelly(0.3, 1.0, kelly_fraction=0.25)
    assert f == 0.0


def test_fractional_kelly_capped_at_config():
    """fractional_kelly must never exceed config.kelly_cap."""
    f = fractional_kelly(0.99, 10.0, kelly_fraction=1.0)
    assert f <= 0.05  # config.kelly_cap


def test_fractional_kelly_uses_config_fraction():
    """When kelly_fraction is None, default is config.kelly_fraction (0.25)."""
    # Use a small edge so we don't hit kelly_cap
    f_full = fractional_kelly(0.51, 1.0, kelly_fraction=1.0)
    f_config = fractional_kelly(0.51, 1.0)
    assert f_config == pytest.approx(0.25 * f_full)  # config fraction of full


def test_fractional_kelly_invalid_inputs():
    """Invalid win_prob or win_loss_ratio must raise."""
    with pytest.raises(ValueError):
        fractional_kelly(0.0, 1.0)
    with pytest.raises(ValueError):
        fractional_kelly(0.6, 0.0)


# =============================================================================
# kelly_bet_size
# =============================================================================


def test_kelly_bet_size_basic():
    """5% Kelly fraction, $10k equity, $100 price → 5 units."""
    size = kelly_bet_size(0.05, 10_000.0, 100.0, min_notional=1.0)
    assert size == pytest.approx(5.0)


def test_kelly_bet_size_respects_min_notional():
    """Bet below min_notional should return 0."""
    size = kelly_bet_size(0.0001, 10_000.0, 100.0, min_notional=5.0)
    assert size == 0.0


def test_kelly_bet_size_respects_equity():
    """Bet must not exceed available equity."""
    size = kelly_bet_size(1.0, 100.0, 50.0, min_notional=1.0, max_notional_frac=1.0)
    # 1.0 * 100 = 100 notional → min(100, 50) = 2 units
    assert size == pytest.approx(2.0)


def test_kelly_bet_size_zero_equity():
    """Zero equity → size 0."""
    size = kelly_bet_size(0.05, 0.0, 100.0)
    assert size == 0.0


def test_kelly_bet_size_max_cap():
    """size must not exceed kelly_cap * equity / price."""
    size = kelly_bet_size(1.0, 100_000.0, 1.0, min_notional=0.01, max_notional_frac=0.05)
    assert size == pytest.approx(5_000.0)


# =============================================================================
# compute_position_size
# =============================================================================


def test_compute_position_size_returns_dict_keys():
    """compute_position_size must return the four required keys."""
    result = compute_position_size(10_000.0, 100.0, 0.6, 1.5)
    assert {"kelly_f", "notional", "size_units", "meets_min"}.issubset(set(result.keys()))


def test_compute_position_size_kelly_f_applied():
    """kelly_f should equal fractional_kelly(output)."""
    result = compute_position_size(10_000.0, 100.0, 0.6, 1.5, kelly_fraction=0.25)
    expected_f = fractional_kelly(0.6, 1.5, kelly_fraction=0.25)
    assert result["kelly_f"] == pytest.approx(expected_f)


def test_compute_position_size_consistency():
    """size_units * price == notional."""
    result = compute_position_size(10_000.0, 50.0, 0.55, 2.0, min_notional=1.0)
    assert result["size_units"] * 50.0 == pytest.approx(result["notional"])


def test_compute_position_size_high_win_rate_bets():
    """High win rate should produce a positive, non-zero size."""
    result = compute_position_size(10_000.0, 200.0, 0.8, 2.0, min_notional=1.0)
    assert result["size_units"] > 0
    assert result["meets_min"] == True  # noqa: E712


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
