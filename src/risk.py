"""Portfolio risk controls: circuit breaker, exposure limits, and risk gating.

All controls return a boolean gate and a reason string. If any gate is
`False`, the position must not be opened.
"""

from __future__ import annotations

import pandas as pd

from src.config import config


def daily_drawdown_gate(
    equity: float,
    start_equity: float,
    limit: float | None = None,
) -> tuple[bool, str]:
    """Circuit breaker: block new trades when daily drawdown exceeds limit.

    Args:
        equity: Current total equity.
        start_equity: Equity at start of the trading day.
        limit: Maximum daily drawdown fraction (default config.drawdown_limit).

    Returns:
        (is_open, reason): True = can trade, False = circuit breaker tripped.
    """
    limit = limit if limit is not None else config.drawdown_limit
    if start_equity <= 0:
        return True, "start_equity_zero"
    drawdown = (start_equity - equity) / start_equity
    if drawdown >= limit:
        return False, f"daily_drawdown {drawdown:.2%} >= limit {limit:.2%}"
    return True, "ok"


def exposure_gate(
    existing_exposure: float,
    new_notional: float,
    equity: float,
    max_fraction: float | None = None,
) -> tuple[bool, str]:
    """Block if total exposure (existing + new) would exceed max_fraction of equity.

    Args:
        existing_exposure: Sum of all open position notionals.
        new_notional: Notional of the proposed trade.
        equity: Current total equity.
        max_fraction: Max total exposure as fraction of equity (default config.max_concurrent_exposure).

    Returns:
        (is_open, reason)
    """
    max_fraction = max_fraction if max_fraction is not None else config.max_concurrent_exposure
    if equity <= 0:
        return False, "equity_zero"
    total = existing_exposure + new_notional
    frac = total / equity
    if frac > max_fraction:
        return False, f"exposure {frac:.2%} > limit {max_fraction:.2%}"
    return True, "ok"


def notional_gate(
    price: float,
    size: float,
    min_notional: float | None = None,
) -> tuple[bool, str]:
    """Reject trades below the minimum notional value.

    Args:
        price: Asset price.
        size: Position size in units.
        min_notional: Minimum USD notional (default config.min_notional).

    Returns:
        (is_open, reason)
    """
    min_notional = min_notional if min_notional is not None else config.min_notional
    notional = abs(price * size)
    if notional < min_notional:
        return False, f"notional {notional:.2f} < minimum {min_notional:.2f}"
    return True, "ok"


def portfolio_heat(
    positions: list[dict],
    equity: float,
) -> float:
    """Aggregate portfolio heat as sum of |notional| / equity.

    Args:
        positions: List of position dicts with 'entry_price' and 'size' keys.
        equity: Current total equity.

    Returns:
        Fraction in [0, inf). 0 = flat.
    """
    if equity <= 0 or not positions:
        return 0.0
    total_notional = sum(abs(p["entry_price"] * p["size"]) for p in positions)
    return total_notional / equity


def risk_gate(
    equity: float,
    start_equity: float,
    existing_exposure: float,
    price: float,
    size: float,
) -> tuple[bool, list[str]]:
    """Combined risk gate: runs all checks and returns a single boolean.

    Args:
        equity: Current total equity.
        start_equity: Equity at start of the trading day.
        existing_exposure: Sum of all open position notionals.
        price: Asset price.
        size: Position size in units.

    Returns:
        (is_open, reasons): True only if all gates pass; reasons lists any rejections.
    """
    reasons: list[str] = []

    ok, r = daily_drawdown_gate(equity, start_equity)
    if not ok:
        reasons.append(r)

    ok, r = notional_gate(price, size)
    if not ok:
        reasons.append(r)

    ok, r = exposure_gate(existing_exposure, abs(price * size), equity)
    if not ok:
        reasons.append(r)

    if len(reasons) == 0:
        reasons = ["ok"]
    return len(reasons) == 1 and reasons[0] == "ok", reasons
