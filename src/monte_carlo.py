"""Monte Carlo simulation: stationary block bootstrap (Politis & Romano 1994).

Blocks of contiguous observations are drawn uniformly at random, with block
lengths drawn from a geometric distribution (mean = block_len). This preserves
serial dependence within blocks while breaking long-range structure — a
realistic simulation of return series under stationary assumptions.

Key outputs:
- Bootstrap distribution of a test statistic (e.g., Sharpe ratio).
- Confidence intervals and p-values for the observed statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config


def _geometric_block_lengths(n: int, mean_len: float, rng: np.random.Generator) -> np.ndarray:
    """Draw block start indices and lengths from a geometric distribution.

    Block lengths are drawn from Geometric(p = 1/mean_len), truncated so
    each block starts at a valid index.

    Returns:
        Array of (block_start, block_length) pairs that tile [0, n).
    """
    p = 1.0 / mean_len
    blocks = []
    pos = 0
    while pos < n:
        # Block length from geometric distribution (min 1)
        blk_len = max(1, rng.geometric(p))
        blk_len = min(blk_len, n - pos)  # clamp to remaining data
        blocks.append((pos, blk_len))
        pos += blk_len
    return np.array(blocks)


def stationary_block_bootstrap(
    series: np.ndarray,
    n_bootstrap: int = 1000,
    block_len: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate bootstrap replications of a 1-D series.

    Args:
        series: 1-D array of observations (returns, etc.).
        n_bootstrap: Number of bootstrap replications.
        block_len: Average block length (default config.block_bootstrap_block_len).
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape (n_bootstrap, len(series)) — each row is one bootstrap
        replication preserving the original series length.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    block_len = block_len if block_len is not None else config.block_bootstrap_block_len
    rng = np.random.default_rng(seed)

    out = np.empty((n_bootstrap, n))
    for i in range(n_bootstrap):
        blocks = _geometric_block_lengths(n, block_len, rng)
        bootstrap = np.empty(n)
        write_pos = 0
        for blk_start, blk_len in blocks:
            # Choose a random position in the original series
            src_start = rng.integers(0, n - blk_len + 1)
            bootstrap[write_pos:write_pos + blk_len] = series[src_start:src_start + blk_len]
            write_pos += blk_len
        out[i] = bootstrap
    return out


def bootstrap_confidence_interval(
    series: np.ndarray,
    statistic: callable,
    n_bootstrap: int = 1000,
    block_len: int | None = None,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        series: 1-D observation series.
        statistic: Function(series) -> scalar statistic.
        n_bootstrap: Number of bootstrap draws.
        block_len: Average block length (default config.block_bootstrap_block_len).
        confidence: Confidence level (default 0.95).
        seed: RNG seed.

    Returns:
        Dict with 'observed', 'ci_lower', 'ci_upper', 'p_value',
        'bootstrap_stats' (array of bootstrap statistics).
    """
    observed = statistic(series)
    bootstrap_series = stationary_block_bootstrap(
        series, n_bootstrap=n_bootstrap, block_len=block_len, seed=seed
    )
    bootstrap_stats = np.array([statistic(row) for row in bootstrap_series])

    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(bootstrap_stats, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_stats, (1.0 - alpha) * 100))

    # Two-sided p-value: fraction of bootstrap stats >= |observed|
    p_value = float(np.mean(np.abs(bootstrap_stats) >= np.abs(observed)))

    return {
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "bootstrap_stats": bootstrap_stats,
    }


def bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_len: int | None = None,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: 1-D array of periodic returns.
        n_bootstrap: Number of bootstrap draws.
        block_len: Average block length.
        confidence: Confidence level.
        seed: RNG seed.

    Returns:
        Dict with observed Sharpe, CI, and p-value.
    """
    def _sharpe(x):
        s = np.std(x, ddof=1)
        return np.mean(x) / s if s > 0 else 0.0

    return bootstrap_confidence_interval(
        returns, _sharpe, n_bootstrap=n_bootstrap,
        block_len=block_len, confidence=confidence, seed=seed
    )
