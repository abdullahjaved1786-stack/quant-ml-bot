"""Backtesting: cost-aware simulation and Deflated Sharpe Ratio.

Cost model (López de Prado):
    cost = 2 × fee  (entry + exit)
    slippage = k × √size / VMA  (square-root market impact)

Deflated Sharpe Ratio (DSR):
Adjusts the observed Sharpe for multiple-testing bias, serial correlation,
and non-Normal kurtosis/skewness. If DSR(p) > 0, the observed Sharpe is
unlikely to have been inflated by luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import config


@dataclass
class BacktestResult:
    """Container for backtest analytics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    cost_usd: float = 0.0
    net_equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    deflated_sharpe: float = 0.0


def compute_slippage(
    size: float,
    vma: float,
    k: float | None = None,
) -> float:
    """Square-root market impact slippage in USD.

    Args:
        size: Trade size in asset units.
        vma: Rolling volume moving average (asset units/bar).
        k: Market impact coefficient (default config.slippage_k).

    Returns:
        Estimated slippage in USD.
    """
    k = k if k is not None else config.slippage_k
    if vma <= 0:
        return 0.0
    return k * math.sqrt(abs(size) / vma)


def backtest_cost_curve(
    signals: pd.Series,
    close: pd.Series,
    size_series: pd.Series | float,
    vma: pd.Series,
    maker_fee: float | None = None,
    taker_fee: float | None = None,
    slippage_k: float | None = None,
) -> BacktestResult:
    """Run a cost-aware backtest over a signals Series.

    Args:
        signals: Signed signal series (+1/-1/0), aligned with close.
        close: Close prices aligned with signals.
        size_series: Position size in asset units (scalar or Series).
        vma: Volume moving average for slippage.
        maker_fee: Maker fee (default config.maker_fee).
        taker_fee: Taker fee (default config.taker_fee).
        slippage_k: Square-root impact coefficient (default config.slippage_k).

    Returns:
        BacktestResult with equity curve, Sharpe, DSR, etc.
    """
    maker_fee = maker_fee if maker_fee is not None else config.maker_fee
    taker_fee = taker_fee if taker_fee is not None else config.taker_fee
    slippage_k = slippage_k if slippage_k is not None else config.slippage_k

    signals = signals.fillna(0)
    close = close.reindex(signals.index).ffill().bfill()
    vma = vma.reindex(signals.index).ffill().bfill()

    if isinstance(size_series, (int, float)):
        size_series = pd.Series(size_series, index=signals.index)

    # Daily returns from holding the position
    position = signals.shift(1).fillna(0)  # act on yesterday's signal
    returns = close.pct_change().fillna(0) * position

    # Costs: each time position flips, pay entry + exit fee + slippage
    trades = signals.diff().fillna(0).abs()
    trade_notional = trades * size_series * close

    entry_fee = taker_fee  # taker for entries
    exit_fee = maker_fee   # maker for exits

    cost_per_trade = entry_fee + exit_fee
    slippage_cost = pd.Series(
        [compute_slippage(s, v, slippage_k) for s, v in zip(size_series, vma)],
        index=signals.index,
    )

    total_costs = trade_notional * cost_per_trade + slippage_cost * trades
    net_returns = returns - total_costs

    # Equity curve
    equity = (1.0 + net_returns).cumprod() * config.initial_capital

    # Compute metrics
    max_drawdown = _max_drawdown(equity)
    sharpe = _annualized_sharpe(net_returns)
    total_trades = int(trades.sum() / 2)  # each trade = enter + exit
    win_trades = ((net_returns[signals.shift(1) != 0]) > 0).sum()
    active = (signals.shift(1) != 0).sum()
    win_rate = win_trades / active if active > 0 else 0.0

    dsr = deflated_sharpe_ratio(
        sharpe,
        num_trials=len(signals),
        sr_std=sharpe / math.sqrt(len(signals)) if sharpe > 0 else 0.01,
        skewness=float(net_returns.skew()),
        kurtosis=float(net_returns.kurtosis()),
    )

    return BacktestResult(
        total_return=float((equity.iloc[-1] / equity.iloc[0]) - 1.0),
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        win_rate=float(win_rate),
        cost_usd=float(total_costs.sum()),
        net_equity_curve=equity,
        deflated_sharpe=dsr,
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    sr_std: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Deflated Sharpe Ratio (López de Prado, 2014).

    The DSR is the probability that the true Sharpe is > 0 given the observed
    SR, multiple testing, and non-normality.

    Args:
        observed_sr: Observed annualized Sharpe ratio.
        num_trials: Total number of strategies tested (for multiple-testing bias).
        sr_std: Standard error of the Sharpe ratio estimate.
        skewness: Sample skewness of the strategy returns.
        kurtosis: Sample excess kurtosis (3.0 = normal).
        confidence: Confidence level for DSR p-value (default 0.95).

    Returns:
        Deflated Sharpe Ratio (probability true SR > 0).
    """
    from scipy import stats

    if sr_std <= 0:
        sr_std = 1e-6

    # Expected maximum SR under null (Euler-Mascheroni + Bonferroni approximation)
    euler_mascheroni = 0.5772
    z = math.sqrt(2.0 * math.log(max(num_trials, 2))) - (
        (euler_mascheroni + math.log(2 * math.pi)) / (2.0 * math.sqrt(2.0 * math.log(max(num_trials, 2))))
    )

    # Adjusted SR for non-Normality
    _adj_var = 1.0 - skewness * observed_sr + (kurtosis - 3.0) * observed_sr**2 / 4.0
    _adj_var = max(_adj_var, 1e-6)  # clamp to avoid math.sqrt of negative
    adj_sr_std = sr_std * math.sqrt(_adj_var)
    if adj_sr_std <= 0:
        adj_sr_std = sr_std

    # Z-score: (observed SR - expected max under null) / std
    z_score = (observed_sr - z * sr_std) / adj_sr_std

    # DSR = P(SR > 0 | observed, trials, non-normality)
    return float(stats.norm.cdf(z_score))


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (positive fraction)."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(abs(drawdown.min()))


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * math.sqrt(periods_per_year))
