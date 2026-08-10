"""Position sizing: fractional Kelly criterion with capital constraints.

Implements a conservative Kelly-fraction bet size tailored for small
capital accounts. The position is constrained by:
1. `kelly_fraction` — a fraction of the Kelly-optimal fraction (default 0.25 = quarter-Kelly)
2. `kelly_cap` — hard cap on position size as a fraction of equity (default 5%)
3. `min_notional` — minimum order size in USD (skipped if not met)
4. Available cash (never over-lever)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config


def fractional_kelly(
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float | None = None,
) -> float:
    """Fractional Kelly bet size as a proportion of capital.

    Kelly formula:  f* = p - (1-p)/b
    where p = win probability, b = win/loss ratio.

    The fractional Kelly reduces variance by applying kelly_fraction.

    Args:
        win_prob: Probability of a winning bet (0, 1).
        win_loss_ratio: Ratio of average win to average loss (b > 0).
        kelly_fraction: Fraction of the Kelly-optimal f* (default config.kelly_fraction).

    Returns:
        Fraction of equity to risk, clipped to [0, kelly_cap].
    """
    kelly_fraction = kelly_fraction if kelly_fraction is not None else config.kelly_fraction
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be > 0")
    if not (0.0 < win_prob < 1.0):
        raise ValueError("win_prob must be between 0 and 1 (exclusive)")

    # Full Kelly
    f_star = win_prob - (1.0 - win_prob) / win_loss_ratio

    # Fractional Kelly — only take a fraction of the optimal bet.
    f = kelly_fraction * f_star

    # Clip: never bet negative (that would be "never bet"), and never exceed cap.
    return float(np.clip(f, 0.0, config.kelly_cap))


def kelly_bet_size(
    kelly_fraction: float,
    equity: float,
    price: float,
    min_notional: float | None = None,
    max_notional_frac: float | None = None,
) -> float:
    """Convert a Kelly fraction into a position size (in units of the asset).

    Args:
        kelly_fraction: Fraction of equity to risk (from fractional_kelly).
        equity: Total account equity in USD.
        price: Current asset price in USD.
        min_notional: Minimum notional value (default config.min_notional).
        max_notional_frac: Max notional as fraction of equity (default config.kelly_cap).

    Returns:
        Position size in asset units. Returns 0.0 if constraints aren't met.
    """
    min_notional = min_notional if min_notional is not None else config.min_notional
    max_notional_frac = max_notional_frac if max_notional_frac is not None else config.kelly_cap

    if equity <= 0 or price <= 0:
        return 0.0

    target_notional = kelly_fraction * equity
    max_notional = max_notional_frac * equity

    # Don't exceed the hard cap
    target_notional = min(target_notional, max_notional)

    # Don't exceed available cash
    target_notional = min(target_notional, equity)

    # Check minimum notional
    if target_notional < min_notional:
        return 0.0

    return target_notional / price


def compute_position_size(
    equity: float,
    price: float,
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float | None = None,
    min_notional: float | None = None,
    max_notional_frac: float | None = None,
) -> dict:
    """One-call wrapper: probability inputs → position size.

    Args:
        equity: Total account equity in USD.
        price: Current asset price in USD.
        win_prob: Probability of a winning bet.
        win_loss_ratio: Ratio of average win to average loss.
        kelly_fraction: Fraction of the Kelly-optimal f* (default config.kelly_fraction).
        min_notional: Minimum order size in USD (default config.min_notional).
        max_notional_frac: Max notional as fraction of equity (default config.kelly_cap).

    Returns:
        Dict with 'kelly_f', 'notional', 'size_units', 'meets_min'.
    """
    kelly_fraction = kelly_fraction if kelly_fraction is not None else config.kelly_fraction
    min_notional = min_notional if min_notional is not None else config.min_notional
    max_notional_frac = max_notional_frac if max_notional_frac is not None else config.kelly_cap

    f = fractional_kelly(win_prob, win_loss_ratio, kelly_fraction)
    size = kelly_bet_size(f, equity, price, min_notional, max_notional_frac)
    notional = size * price

    return {
        "kelly_f": f,
        "notional": notional,
        "size_units": size,
        "meets_min": notional >= min_notional,
    }
