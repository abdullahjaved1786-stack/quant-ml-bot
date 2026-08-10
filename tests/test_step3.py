"""Unit tests for Step 3: Meta-labeling + feature pruning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.meta_model import MetaLabelingModel, make_meta_labels
from src.pruning import (
    compute_feature_importance,
    prune_features,
    select_features_by_importance,
)


def _make_data(n=300, seed=0, n_features=5, informative=2):
    """Synthetic features where the first `informative` features drive y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    score = X[:, :informative].sum(axis=1) + rng.normal(0, 0.5, n)
    y = np.where(score > 0, 1, -1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return df, y


# =============================================================================
# make_meta_labels
# =============================================================================


def test_make_meta_labels_basic():
    """Matching preds → 1, mismatching → 0, zero-labels → 0."""
    preds = np.array([1, 1, -1, -1, 1])
    actual = np.array([1, -1, -1, 1, 0])
    meta = make_meta_labels(preds, actual)
    assert meta.tolist() == [1, 0, 1, 0, 0]


def test_make_meta_labels_empty():
    """Empty input → empty output."""
    meta = make_meta_labels(np.array([]), np.array([]))
    assert len(meta) == 0


# =============================================================================
# MetaLabelingModel
# =============================================================================


def test_meta_labeling_fit_predict():
    """MetaLabelingModel should fit and produce gated bets in {-1, 0, 1}."""
    X, y = _make_data(seed=1)
    primary = LogisticRegression(max_iter=1000)
    meta = LogisticRegression(max_iter=1000)
    model = MetaLabelingModel(primary, meta)
    model.fit(X.values, y)
    bets = model.predict(X.values, bet_threshold=0.6)
    assert set(np.unique(bets)).issubset({-1, 0, 1})
    # Should have some bets
    assert (bets != 0).sum() > 0


def test_meta_labeling_gates_low_confidence():
    """Raising the bet threshold should produce fewer active bets."""
    X, y = _make_data(seed=2)
    model = MetaLabelingModel(
        LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
    )
    model.fit(X.values, y)
    lax = model.predict(X.values, bet_threshold=0.51)
    strict = model.predict(X.values, bet_threshold=0.95)
    assert (strict != 0).sum() <= (lax != 0).sum()


def test_meta_labeling_accuracy_on_separable_data():
    """On cleanly separable data, active bets should be mostly correct."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = np.where(X[:, 0] > 0, 1, -1)  # perfectly separable on f0
    df = pd.DataFrame(X, columns=["f0", "f1"])

    model = MetaLabelingModel(
        LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
    )
    model.fit(df.values, y)
    bets = model.predict(df.values, bet_threshold=0.6)
    active = bets != 0
    if active.sum() > 0:
        correct = (bets[active] == y[active]).mean()
        assert correct >= 0.9


def test_meta_labeling_no_active_labels_raises():
    """Fitting with all-zero labels should raise ValueError."""
    X, _ = _make_data(seed=4)
    y = np.zeros(len(X), dtype=int)
    model = MetaLabelingModel(
        LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
    )
    with pytest.raises(ValueError, match="No active"):
        model.fit(X.values, y)


def test_predict_proba_shape():
    """predict_proba should return (n, 2) with rows summing to 1."""
    X, y = _make_data(seed=5)
    model = MetaLabelingModel(
        LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
    )
    model.fit(X.values, y)
    proba = model.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# =============================================================================
# compute_feature_importance
# =============================================================================


def test_compute_feature_importance_ranks_informative_first():
    """The informative features should rank above noise features."""
    X, y = _make_data(n=300, seed=6, informative=2)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    imp = compute_feature_importance(model, X, y, n_repeats=3, seed=0)
    # f0 and f1 should be the top two
    top2 = imp.index[:2].tolist()
    assert set(top2) == {"f0", "f1"}


def test_compute_feature_importance_returns_sorted_series():
    """Return type is a Series sorted descending."""
    X, y = _make_data(seed=7)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    imp = compute_feature_importance(model, X, y, n_repeats=2, seed=0)
    assert isinstance(imp, pd.Series)
    assert len(imp) == X.shape[1]
    assert (imp.values[:-1] >= imp.values[1:]).all()  # sorted desc


# =============================================================================
# prune_features
# =============================================================================


def test_prune_features_keep_top_n():
    """keep=2 should return exactly 2 informative features."""
    X, y = _make_data(n=300, seed=8, informative=2)
    kept, imp = prune_features(
        lambda: RandomForestClassifier(n_estimators=30, random_state=0),
        X, y, keep=2, n_repeats=2, seed=0,
    )
    assert len(kept) == 2
    assert set(kept) == {"f0", "f1"}


def test_prune_features_min_importance():
    """min_importance=0.05 should drop near-zero-importance features."""
    X, y = _make_data(n=300, seed=9, informative=1)
    kept, imp = prune_features(
        lambda: RandomForestClassifier(n_estimators=30, random_state=0),
        X, y, min_importance=0.05, n_repeats=2, seed=0,
    )
    # f0 is informative; noise features get ~0 importance
    assert "f0" in kept
    assert all(k in X.columns for k in kept)


def test_prune_features_requires_threshold():
    """Calling without min_importance or keep must raise."""
    X, y = _make_data(seed=10)
    with pytest.raises(ValueError):
        prune_features(
            lambda: RandomForestClassifier(n_estimators=30, random_state=0),
            X, y,
        )


# =============================================================================
# select_features_by_importance
# =============================================================================


def test_select_features_by_importance_fraction():
    """keep_frac=0.5 on 6 features → 3 kept, informative included."""
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    score = X["f0"] + X["f1"]
    y = np.where(score > 0, 1, -1)

    kept = select_features_by_importance(X, y, keep_frac=0.5, seed=0)
    assert len(kept) == 3
    assert "f0" in kept and "f1" in kept


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
