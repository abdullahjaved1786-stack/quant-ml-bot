"""Drift detection: Population Stability Index (PSI).

PSI measures the shift in a feature's distribution between two windows.
A PSI > 0.25 indicates a significant drift that warrants retraining.

PSI = Σ (p_i - q_i) × ln(p_i / q_i)
where p = current distribution, q = reference distribution (both binned).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import config


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two distributions.

    Args:
        reference: Reference distribution (training data).
        current: Current distribution (live data).
        bins: Number of bins for discretisation (default 10).
        eps: Small constant to avoid log(0).

    Returns:
        PSI value (float). Values:
            < 0.10 — no significant drift
            0.10–0.25 — moderate drift, monitor
            > 0.25 — significant drift, retrain
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Use quantile-based bins derived from the reference distribution
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Ensure edges are strictly increasing (collapse ties)
    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    p = np.histogram(reference, bins=edges)[0].astype(float) + eps
    q = np.histogram(current, bins=edges)[0].astype(float) + eps

    p /= p.sum()
    q /= q.sum()

    return float(np.sum((p - q) * np.log(p / q)))


def detect_drift(
    series: pd.Series,
    ref_window: int = 100,
    test_window: int = 50,
    threshold: float | None = None,
    bins: int = 10,
) -> pd.Series:
    """Rolling PSI drift detection on a pandas Series.

    Args:
        series: Time series to monitor for drift.
        ref_window: Number of bars for the reference distribution.
        test_window: Number of bars for the test distribution.
        threshold: PSI threshold above which drift is flagged (default config.psi_threshold).
        bins: Number of bins for PSI calculation.

    Returns:
        Boolean Series: True = drift detected at that bar.
    """
    threshold = threshold if threshold is not None else config.psi_threshold
    values = series.dropna().values.astype(float)
    n = len(values)
    psi_values = np.full(n, np.nan)

    window_end = ref_window + test_window
    for i in range(window_end, n):
        ref = values[i - ref_window - test_window:i - test_window]
        test = values[i - test_window:i]
        psi_values[i] = compute_psi(ref, test, bins=bins)

    psi_series = pd.Series(psi_values, index=series.index)
    return psi_series > threshold


class DriftDetector:
    """Stateful drift detector that monitors a rolling window.

    Args:
        ref_window: Reference window size.
        test_window: Test window size.
        threshold: PSI threshold (default config.psi_threshold).
        bins: Number of bins for PSI.
    """

    def __init__(
        self,
        ref_window: int = 100,
        test_window: int = 50,
        threshold: float | None = None,
        bins: int = 10,
    ):
        self.ref_window = ref_window
        self.test_window = test_window
        self.threshold = threshold if threshold is not None else config.psi_threshold
        self.bins = bins
        self._psi_history: list[float] = []

    def check(self, series: pd.Series) -> bool:
        """Check the latest bar for drift (returns True if drifted)."""
        values = series.dropna().values.astype(float)
        n = len(values)
        if n < self.ref_window + self.test_window:
            return False

        ref = values[n - self.ref_window - self.test_window:n - self.test_window]
        test = values[n - self.test_window:n]
        psi = compute_psi(ref, test, bins=self.bins)
        self._psi_history.append(psi)
        return psi > self.threshold

    @property
    def psi_history(self) -> list[float]:
        """List of PSI values from successive check() calls."""
        return list(self._psi_history)
