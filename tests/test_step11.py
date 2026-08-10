"""Unit tests for Step 11: fractional differentiation + PSI drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import frac_diff_weights, frac_diff
from src.drift import compute_psi, detect_drift, DriftDetector


# =============================================================================
# frac_diff_weights
# =============================================================================


def test_frac_diff_weights_d0_identity():
    """d=0 → weight[0] = 1.0, all others 0 (no differencing)."""
    w = frac_diff_weights(0.0, 5)
    assert w[0] == pytest.approx(1.0)
    assert np.allclose(w[1:], 0.0)


def test_frac_diff_weights_d1_first_diff():
    """d=1 → [1, -1] (first difference)."""
    w = frac_diff_weights(1.0, 2)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(-1.0)


def test_frac_diff_weights_length():
    """Output length must equal k."""
    w = frac_diff_weights(0.5, 10)
    assert len(w) == 10


def test_frac_diff_weights_zero_k():
    """k=0 → empty array."""
    w = frac_diff_weights(0.5, 0)
    assert len(w) == 0


# =============================================================================
# frac_diff
# =============================================================================


def test_frac_diff_preserves_index():
    """Output index must be a subset of the input index."""
    idx = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    series = pd.Series(np.random.default_rng(0).normal(size=100), index=idx)
    out = frac_diff(series, d=0.5)
    assert len(out) <= len(series)
    assert out.index[0] >= series.index[0]


def test_frac_diff_d0_no_change():
    """d=0 should return the original series (no differencing)."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = frac_diff(series, d=0.0, threshold=1e-10)
    assert len(out) == 5
    pd.testing.assert_series_equal(out, series, check_names=False)


def test_frac_diff_d1_first_difference():
    """d=1 → each output = current - previous."""
    series = pd.Series([10.0, 12.0, 15.0, 20.0])
    out = frac_diff(series, d=1.0, threshold=1e-10)
    expected = np.array([2.0, 3.0, 5.0])  # [12-10, 15-12, 20-15]
    np.testing.assert_allclose(out.values, expected)


def test_frac_diff_output_shorter_by_window():
    """Output should be shorter than input by approximately the weight window."""
    series = pd.Series(np.random.default_rng(1).normal(size=200))
    out = frac_diff(series, d=0.5)
    assert len(out) < len(series)


def test_frac_diff_nan_handling():
    """NaNs in input should be dropped before differencing."""
    series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    out = frac_diff(series, d=1.0, threshold=1e-10)
    assert out.notna().all()


def test_frac_diff_empty_series():
    """Empty input → empty output."""
    series = pd.Series(dtype=float)
    out = frac_diff(series, d=0.5)
    assert len(out) == 0


# =============================================================================
# compute_psi
# =============================================================================


def test_psi_identical_distributions():
    """Same distribution → PSI = 0."""
    data = np.random.default_rng(2).normal(0, 1, 500)
    psi = compute_psi(data, data, bins=10)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distributions():
    """Meaningfully different distributions → PSI > 0."""
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(2, 1, 500)
    psi = compute_psi(ref, cur, bins=10)
    assert psi > 0.5


def test_psi_empty_returns_zero():
    """Empty input → PSI = 0."""
    assert compute_psi(np.array([]), np.array([])) == 0.0


def test_psi_small_shift():
    """Small overlap shift → PSI between 0 and 0.5."""
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0.1, 1, 1000)
    psi = compute_psi(ref, cur, bins=10)
    assert 0.0 <= psi <= 0.5


# =============================================================================
# detect_drift
# =============================================================================


def test_detect_drift_no_drift_on_constant():
    """Constant series should produce no drift flags."""
    series = pd.Series(np.full(300, 5.0))
    drift = detect_drift(series, ref_window=100, test_window=50)
    assert drift.sum() == 0


def test_detect_drift_detects_regime_shift():
    """A sharp mean shift should be detected."""
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])
    series = pd.Series(values)
    drift = detect_drift(series, ref_window=80, test_window=40, threshold=0.1)
    # After the shift (bar ~190 onward), drift should be flagged
    assert drift.iloc[-50:].sum() > 0


def test_detect_drift_output_length():
    """Output must match input length."""
    series = pd.Series(np.random.default_rng(6).normal(size=200))
    drift = detect_drift(series, ref_window=50, test_window=25)
    assert len(drift) == len(series)


# =============================================================================
# DriftDetector class
# =============================================================================


def test_drift_detector_check():
    """DriftDetector.check() should return True when drift is present."""
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 100)
    cur = rng.normal(5, 1, 50)
    combined = np.concatenate([ref, cur])
    series = pd.Series(combined)

    det = DriftDetector(ref_window=80, test_window=40, threshold=0.1)
    result = det.check(series)
    assert result is True
    assert len(det.psi_history) == 1


def test_drift_detector_no_drift():
    """Stable series should not trigger drift."""
    series = pd.Series(np.random.default_rng(8).normal(0, 1, 300))
    det = DriftDetector(ref_window=80, test_window=40)
    result = det.check(series)
    assert result is False


def test_drift_detector_insufficient_data():
    """Too-short series should return False without crashing."""
    series = pd.Series(np.random.default_rng(9).normal(size=10))
    det = DriftDetector(ref_window=100, test_window=50)
    result = det.check(series)
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
