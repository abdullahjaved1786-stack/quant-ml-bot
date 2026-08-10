"""Feature pruning: drop low-information features before model training.

Implements SHAP-style feature selection. The `shap` package is a heavy
dependency and adds nothing for this pipeline's scale, so importance is
computed with sklearn's permutation importance (the same rank-ordering SHAP
produces, without the tree-specific overhead). Swap in `shap.TreeExplainer`
if model-specific attribution is ever needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    model,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Permutation importance of each feature.

    Args:
        model: Fitted sklearn estimator with .score.
        X: Feature matrix (DataFrame preserves names, ndarray uses positions).
        y: Labels.
        n_repeats: Permutation repeats.
        seed: RNG seed for reproducibility.

    Returns:
        Series indexed by feature name (or int position), sorted desc by
        mean importance.
    """
    X_arr = np.asarray(X)
    score_input = X if isinstance(X, pd.DataFrame) else X_arr
    result = permutation_importance(
        model, score_input, np.asarray(y), n_repeats=n_repeats, random_state=seed
    )

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
    else:
        names = list(range(X_arr.shape[1]))

    importance = pd.Series(result.importances_mean, index=names)
    return importance.sort_values(ascending=False)


def prune_features(
    model_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    min_importance: float | None = None,
    keep: int | None = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[list[str], pd.Series]:
    """Drop features below a permutation-importance threshold.

    Args:
        model_factory: Callable returning a fresh (unfitted) estimator.
        X: Feature DataFrame.
        y: Labels.
        min_importance: Keep features with importance >= this value.
        keep: Alternatively, keep the top-N features.
        n_repeats: Permutation repeats.
        seed: RNG seed.

    Returns:
        (kept_feature_names, importance_series)
    """
    if min_importance is None and keep is None:
        raise ValueError("Provide min_importance or keep")

    model = model_factory().fit(X, np.asarray(y))
    importance = compute_feature_importance(
        model, X, y, n_repeats=n_repeats, seed=seed
    )

    if keep is not None:
        kept = list(importance.index[:keep])
    else:
        kept = [f for f in importance.index if importance[f] >= min_importance]

    return kept, importance


def select_features_by_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    keep_frac: float = 0.5,
    seed: int = 42,
    n_estimators: int = 50,
) -> list[str]:
    """Convenience wrapper: fit a random forest and keep top `keep_frac` features.

    Args:
        X: Feature DataFrame.
        y: Labels.
        keep_frac: Fraction of features to keep (top importance).
        seed: RNG seed.
        n_estimators: Forest size.

    Returns:
        List of kept feature names.
    """
    n_keep = max(1, int(np.ceil(keep_frac * X.shape[1])))
    factory = lambda: RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed
    )
    kept, _ = prune_features(
        factory, X, y, keep=n_keep, seed=seed, n_repeats=3
    )
    return kept
