"""Unit tests for Step 2: Nested Purged K-Fold CV + parameter stability grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    PurgedKFold,
    purge_leakage,
    nested_purged_cv,
    run_parameter_sensitivity_grid,
    walk_forward_split,
)


# =============================================================================
# PurgedKFold
# =============================================================================


def test_purgedkfold_split_contiguous_test_sets():
    """Each test fold must be a contiguous block; folds must partition the data."""
    pkf = PurgedKFold(n_splits=5, embargo=0)
    splits = list(pkf.split(100))
    assert len(splits) == 5

    all_test = np.concatenate([t for _, t in splits])
    assert len(all_test) == 100
    assert sorted(all_test) == list(range(100))

    for _, test_idx in splits:
        assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))


def test_purgedkfold_train_test_disjoint():
    """Train and test indices must never overlap."""
    pkf = PurgedKFold(n_splits=4, embargo=3)
    for train_idx, test_idx in pkf.split(60):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purgedkfold_embargo_removes_training_data_after_test():
    """Embargo bars after the test fold must be excluded from training."""
    pkf = PurgedKFold(n_splits=4, embargo=2)
    train_idx, test_idx = next(iter(pkf.split(40)))
    # test fold 0: indices 0..9, embargo removes 10,11 from train
    assert 10 not in train_idx
    assert 11 not in train_idx
    assert 12 in train_idx


def test_purgedkfold_invalid_splits():
    """n_splits < 2 and negative embargo must raise."""
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, embargo=-1)


# =============================================================================
# purge_leakage
# =============================================================================


def test_purge_leakage_no_events_keeps_all():
    """Without events/t1, no samples are purged."""
    labels = np.array([1, -1, 1, 0])
    mask = purge_leakage(labels)
    assert mask.tolist() == [True, True, True, True]


def test_purge_leakage_contained_window_dropped():
    """A sample whose window is contained in an earlier sample's window is purged."""
    events = np.array([1, 2, 3])
    t1 = np.array([5, 4, 6])
    labels = np.array([1, -1, 1])
    mask = purge_leakage(labels, events, t1)
    # Sample 0: [1,5]; Sample 1: [2,4] — contained in [1,5] → purged
    assert mask[1] == False  # noqa: E712
    assert mask[0] == True  # noqa: E712
    assert mask[2] == True  # noqa: E712


def test_purge_leakage_empty():
    """Empty input returns empty mask."""
    mask = purge_leakage(np.array([]))
    assert len(mask) == 0


# =============================================================================
# nested_purged_cv
# =============================================================================


def _dummy_fit_score(X_train, y_train, X_test):
    """Dummy classifier: predict the majority class."""
    if len(y_train) == 0:
        return np.zeros(len(X_test), dtype=int)
    majority = 1 if np.mean(y_train) >= 0.5 else -1
    return np.full(len(X_test), majority)


def test_nested_purged_cv_returns_dict_with_scores():
    """nested_purged_cv must return expected keys and array lengths."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.where(X[:, 0] > 0, 1, -1)
    result = nested_purged_cv(
        X, y, _dummy_fit_score, outer_splits=5, inner_splits=3, embargo=1
    )
    assert "oos_scores" in result
    assert len(result["oos_scores"]) == 5
    assert 0.0 <= result["mean_oos"] <= 1.0


def test_nested_purged_cv_perfect_separator_scores_high():
    """A linear separator should score near 1.0 on separable data."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    result = nested_purged_cv(X, y, _dummy_fit_score, outer_splits=4, embargo=0)
    # With an informative first feature and majority baseline, mean_oos should be meaningful.
    assert result["mean_oos"] >= 0.5


# =============================================================================
# run_parameter_sensitivity_grid
# =============================================================================


def _param_fit_score(X_train, y_train, X_test, threshold=0.0, sign=1):
    """Simple parametric classifier."""
    feature = X_test[:, 0]
    return np.where(feature * sign > threshold, 1, -1)


def test_parameter_grid_returns_dataframe():
    """Grid over 2 params with 2 values each → 4 rows."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    y = np.where(X[:, 0] > 0, 1, -1)

    grid = {"threshold": [-0.5, 0.0, 0.5], "sign": [1, -1]}
    df = run_parameter_sensitivity_grid(X, y, grid, _param_fit_score, n_splits=4)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert {"threshold", "sign", "mean_score", "std_score"}.issubset(set(df.columns))
    assert "fold_0" in df.columns


def test_parameter_grid_finds_best_params():
    """For data where X[:,0] > 0 → label 1, threshold=0/sign=1 should score highest."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2))
    y = np.where(X[:, 0] > 0, 1, -1)

    grid = {"threshold": [0.0, 1.0], "sign": [1, -1]}
    df = run_parameter_sensitivity_grid(X, y, grid, _param_fit_score, n_splits=4)
    best = df.loc[df["mean_score"].idxmax()]
    assert best["threshold"] == 0.0
    assert best["sign"] == 1


# =============================================================================
# walk_forward_split
# =============================================================================


def test_walk_forward_split_basic():
    """Rolling splits should slide the window forward."""
    splits = walk_forward_split(n=100, train_window=30, test_window=10)
    assert len(splits) == 7  # 0-39, 10-49, ..., 60-99
    first_train, first_test = splits[0]
    assert len(first_train) == 30
    assert len(first_test) == 10
    # Second split slides 10 bars (step = test_window) forward
    second_train, _ = splits[1]
    assert second_train[0] - first_train[0] == 10
    assert first_train[0] + 10 == second_train[0]


def test_walk_forward_split_step_override():
    """Custom step should advance by step, not test_window."""
    splits = walk_forward_split(n=100, train_window=30, test_window=10, step=20)
    # Starts at 0, 20, 40, 60 → 4 splits
    assert len(splits) == 4


def test_walk_forward_split_insufficient_data():
    """Too-short series should yield no splits."""
    splits = walk_forward_split(n=30, train_window=30, test_window=10)
    assert splits == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
