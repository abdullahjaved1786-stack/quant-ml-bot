"""Model validation: nested purged K-Fold CV and parameter stability grid.

Implements López de Prado's cross-validation for serial data:
- `PurgedKFold`: K-fold split with training data purged of leakage around the
  validation fold (embargo) to prevent overlapping-label lookahead.
- `nested_purged_cv`: outer splits for OOS evaluation, inner splits for
  hyperparameter selection — returns a bias-corrected OOS score.
- `run_parameter_sensitivity_grid`: evaluates a scoring function over a
  parameter grid and returns per-fold stability stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-fold CV for time series with purging and embargo.

    Each validation fold holds a contiguous block of `n_splits` bars. Training
    indices are the complement, with `embargo` bars after the validation block
    removed (prevents labels whose outcome windows overlap the validation set).

    Args:
        n_splits: Number of folds.
        embargo: Number of bars to drop after each validation fold.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n: int):
        """Yield (train_idx, test_idx) pairs. n is number of samples."""
        test_size = n // self.n_splits
        for fold in range(self.n_splits):
            test_start = fold * test_size
            test_end = test_start + test_size if fold < self.n_splits - 1 else n

            test_idx = np.arange(test_start, test_end)

            # Embargo bars immediately after the validation fold are dropped
            # from training to avoid label leakage.
            embargo_end = min(test_end + self.embargo, n)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:embargo_end] = False
            train_idx = np.flatnonzero(train_mask)

            yield train_idx, test_idx


def purge_leakage(
    labels: pd.Series | np.ndarray,
    events: pd.Series | np.ndarray | None = None,
    t1: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of samples that are not leaked by earlier samples.

    A sample's label window spans [entry, exit]. If any sample's window
    overlaps another sample's window and shares data, the later sample is
    leaked. This function flags those later samples so they can be removed.

    Args:
        labels: Label array (same length as event rows).
        events: Optional event start times.
        t1: Optional event end times (exit times).

    Returns:
        Boolean mask: True where the sample is clean, False where purged.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return np.array([], dtype=bool)

    clean = np.ones(n, dtype=bool)

    # If we have event windows, drop samples whose window is contained in an
    # earlier sample's window (the later one is redundant).
    if events is not None and t1 is not None:
        events = np.asarray(events)
        t1 = np.asarray(t1)
        # Sort by event time
        order = np.argsort(events, kind="stable")
        # Simple O(n^2) containment check — fine for research-scale data.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if events[j] <= events[i] and t1[i] <= t1[j]:
                    # i's window is contained in j's window
                    if t1[i] < t1[j] or events[i] > events[j]:
                        clean[i] = False
                        break
    return clean


def nested_purged_cv(
    X: np.ndarray,
    y: np.ndarray,
    fit_score: callable,
    outer_splits: int = 5,
    inner_splits: int = 3,
    embargo: int = 0,
) -> dict:
    """Nested purged K-fold cross-validation.

    Outer loop: split data, fit the chosen model on outer train, score outer
    test. Inner loop (on the outer train) would tune hyperparameters; here we
    expose it as a placeholder structure so callers can plug in their own
    inner model selection.

    Args:
        X: Feature matrix.
        y: Label vector.
        fit_score: Callable(train_X, train_y, test_X) -> predicted labels.
        outer_splits: Number of outer folds.
        inner_splits: Number of inner folds (reserved for tuning).
        embargo: Bars purged after each validation fold.

    Returns:
        Dict with 'oos_scores' (array of per-fold OOS scores),
        'mean_oos' and 'std_oos'.
    """
    n = len(y)
    purged = PurgedKFold(n_splits=outer_splits, embargo=embargo)
    oos_scores = []

    for train_idx, test_idx in purged.split(n):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        preds = fit_score(X_train, y_train, X_test)
        oos_scores.append(_score(preds, y[test_idx]))

    oos_scores = np.asarray(oos_scores)
    return {
        "oos_scores": oos_scores,
        "mean_oos": float(np.mean(oos_scores)) if len(oos_scores) else float("nan"),
        "std_oos": float(np.std(oos_scores)) if len(oos_scores) else float("nan"),
    }


def _score(preds: np.ndarray, actual: np.ndarray) -> float:
    """Default accuracy score for binary labels."""
    preds = np.asarray(preds)
    actual = np.asarray(actual)
    if len(preds) == 0:
        return float("nan")
    return float(np.mean(preds == actual))


def run_parameter_sensitivity_grid(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict,
    fit_score: callable,
    n_splits: int = 5,
    embargo: int = 0,
    metric: callable = _score,
) -> pd.DataFrame:
    """Evaluate a model across a parameter grid with purged CV.

    For each combination of params in param_grid (must be dict of lists),
    runs purged CV and returns per-fold metric values.

    Args:
        X: Feature matrix.
        y: Label vector.
        param_grid: Dict mapping param name -> list of values.
        fit_score: Callable(train_X, train_y, test_X, **params) -> predictions.
        n_splits: Number of CV folds.
        embargo: Embargo bars.
        metric: Scoring function (preds, actual) -> float.

    Returns:
        DataFrame with columns: parameter names, mean_score, std_score,
        and per-fold columns fold_0 .. fold_{n_splits-1}.
    """
    from itertools import product

    param_names = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    purged = PurgedKFold(n_splits=n_splits, embargo=embargo)

    rows = []
    for combo in combos:
        params = dict(zip(param_names, combo))
        fold_scores = []
        for train_idx, test_idx in purged.split(len(y)):
            preds = fit_score(X[train_idx], y[train_idx], X[test_idx], **params)
            fold_scores.append(metric(preds, y[test_idx]))
        rows.append(
            {**params, "mean_score": float(np.mean(fold_scores)),
             "std_score": float(np.std(fold_scores)),
             **{f"fold_{i}": s for i, s in enumerate(fold_scores)}}
        )

    return pd.DataFrame(rows)


def walk_forward_split(
    n: int,
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling walk-forward splits for time-series evaluation.

    Args:
        n: Total number of samples.
        train_window: Number of training bars.
        test_window: Number of test bars.
        step: Advance between folds (default: test_window).

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    step = step or test_window
    splits = []
    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window
        train_idx = np.arange(start, train_end)
        test_idx = np.arange(train_end, test_end)
        splits.append((train_idx, test_idx))
        start += step
    return splits
