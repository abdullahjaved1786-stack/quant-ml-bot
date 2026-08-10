"""Unit tests for Step 7: Portfolio risk controls."""

from __future__ import annotations

import pytest

from src.risk import (
    daily_drawdown_gate,
    exposure_gate,
    notional_gate,
    portfolio_heat,
    risk_gate,
)


# =============================================================================
# daily_drawdown_gate
# =============================================================================


def test_daily_drawdown_gate_ok():
    """Small drawdown should pass."""
    ok, reason = daily_drawdown_gate(equity=9_950.0, start_equity=10_000.0)
    assert ok is True


def test_daily_drawdown_gate_tripped():
    """Drawdown exceeding limit must trip the circuit breaker."""
    ok, reason = daily_drawdown_gate(equity=9_600.0, start_equity=10_000.0)
    assert ok is False
    assert "daily_drawdown" in reason


def test_daily_drawdown_gate_custom_limit():
    """A tighter limit should trip earlier."""
    ok, _ = daily_drawdown_gate(equity=9_900.0, start_equity=10_000.0, limit=0.005)
    assert ok is False


def test_daily_drawdown_gate_zero_start_equity():
    """Zero start_equity is a degenerate case; should not crash."""
    ok, reason = daily_drawdown_gate(equity=0.0, start_equity=0.0)
    assert ok is True  # pass-through for degenerate case


def test_daily_drawdown_gate_exact_limit():
    """Drawdown exactly at the limit should trip the gate (>= check)."""
    ok, _ = daily_drawdown_gate(equity=9_700.0, start_equity=10_000.0, limit=0.03)
    assert ok is False


# =============================================================================
# exposure_gate
# =============================================================================


def test_exposure_gate_within_limit():
    """Total exposure below max should pass."""
    ok, _ = exposure_gate(existing_exposure=4_000.0, new_notional=1_000.0, equity=10_000.0)
    assert ok is True


def test_exposure_gate_over_limit():
    """Total exposure above max must be blocked."""
    ok, reason = exposure_gate(existing_exposure=4_000.0, new_notional=1_500.0, equity=10_000.0)
    assert ok is False
    assert "exposure" in reason


def test_exposure_gate_exact_limit():
    """Exposure exactly at the limit should be allowed (not exceeded)."""
    ok, _ = exposure_gate(existing_exposure=4_500.0, new_notional=500.0, equity=10_000.0)
    assert ok is True


def test_exposure_gate_zero_equity():
    """Zero equity should always block."""
    ok, reason = exposure_gate(existing_exposure=0, new_notional=100.0, equity=0)
    assert ok is False
    assert "equity_zero" in reason


def test_exposure_gate_custom_max():
    """Custom max_fraction should override config."""
    ok, _ = exposure_gate(
        existing_exposure=3_000.0, new_notional=2_000.0,
        equity=10_000.0, max_fraction=0.6
    )
    assert ok is True  # 5000/10000 = 0.5 <= 0.6


# =============================================================================
# notional_gate
# =============================================================================


def test_notional_gate_above_min():
    """Notional above min should pass."""
    ok, _ = notional_gate(price=100.0, size=0.1)
    assert ok is True  # 10 > 5.0


def test_notional_gate_below_min():
    """Notional below minimum must be rejected."""
    ok, reason = notional_gate(price=10.0, size=0.1, min_notional=5.0)
    assert ok is False
    assert "notional" in reason


def test_notional_gate_custom_min():
    """Custom min_notional should override config."""
    ok, _ = notional_gate(price=100.0, size=0.1, min_notional=5.0)
    assert ok is True


# =============================================================================
# portfolio_heat
# =============================================================================


def test_portfolio_heat_flat():
    """No positions → heat = 0."""
    assert portfolio_heat([], 10_000.0) == 0.0


def test_portfolio_heat_basic():
    """Heat = sum(|notional|) / equity."""
    positions = [
        {"entry_price": 100.0, "size": 1.0},  # notional = 100
        {"entry_price": 50.0, "size": 2.0},   # notional = 100
    ]
    assert portfolio_heat(positions, 10_000.0) == pytest.approx(200.0 / 10_000.0)


def test_portfolio_heat_zero_equity():
    """Zero equity → heat = 0 (avoid division by zero)."""
    positions = [{"entry_price": 100.0, "size": 1.0}]
    assert portfolio_heat(positions, 0.0) == 0.0


# =============================================================================
# risk_gate (combined)
# =============================================================================


def test_risk_gate_all_ok():
    """All constraints satisfied → open = True, reasons = ['ok']."""
    ok, reasons = risk_gate(
        equity=10_000.0,
        start_equity=10_000.0,
        existing_exposure=0.0,
        price=100.0,
        size=0.1,
    )
    assert ok is True
    assert reasons == ["ok"]


def test_risk_gate_drawdown_blocks():
    """Daily drawdown must block when breached."""
    ok, reasons = risk_gate(
        equity=9_500.0,
        start_equity=10_000.0,
        existing_exposure=0.0,
        price=100.0,
        size=0.1,
    )
    assert ok is False
    assert any("daily_drawdown" in r for r in reasons)


def test_risk_gate_exposure_blocks():
    """High existing exposure must block new trade."""
    ok, reasons = risk_gate(
        equity=10_000.0,
        start_equity=10_000.0,
        existing_exposure=4_500.0,
        price=100.0,
        size=10.0,
    )
    assert ok is False
    assert any("exposure" in r for r in reasons)


def test_risk_gate_notional_blocks():
    """Notional below minimum must block."""
    ok, reasons = risk_gate(
        equity=10_000.0,
        start_equity=10_000.0,
        existing_exposure=0.0,
        price=10.0,
        size=0.1,
    )
    assert ok is False
    assert any("notional" in r for r in reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
