"""Meta-labeling: second model learns whether the primary model's bet is correct.

Implements López de Prado's meta-labeling scheme:
1. Primary model fits (X, y) and predicts direction (+1 / -1).
2. Meta model fits (X, y_meta) where y_meta = 1 if the primary model's
   prediction matched the realized label, else 0.
3. Final bet size = primary prediction * meta-model probability — a weak but
   correct primary signal can still be traded when the meta model is confident.

This avoids over-betting on noisy direction calls: the meta model gates the bet.
"""

from __future__ import annotations

import numpy as np


class MetaLabelingModel:
    """Stack a gating (meta) model on top of a primary classifier.

    Args:
        primary_model: sklearn-style estimator with fit/predict/predict_proba.
        meta_model: sklearn-style estimator with fit/predict_proba.
        primary_prob_threshold: Primary label threshold (default 0.5).
    """

    def __init__(self, primary_model, meta_model, primary_prob_threshold: float = 0.5):
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.primary_prob_threshold = primary_prob_threshold
        self.primary_classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaLabelingModel":
        """Fit primary on (X, y), then meta on (X, y_meta).

        Args:
            X: Feature matrix.
            y: Labels in {-1, 0, 1} (0 treated as "no bet").
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Primary model only learns the direction of active bets.
        primary_mask = y != 0
        if not primary_mask.any():
            raise ValueError("No active (non-zero) labels to fit the primary model.")

        self.primary_model.fit(X[primary_mask], y[primary_mask])
        self.primary_classes_ = np.asarray(self.primary_model.classes_)

        # Meta model: did the primary model call the direction right?
        primary_pred = self._primary_direction(X[primary_mask])
        meta_y = (primary_pred == y[primary_mask]).astype(int)

        if len(np.unique(meta_y)) < 2:
            # Degenerate: primary is always right/wrong on train — fall back to
            # predicting the majority so the meta model remains well-defined.
            majority = int(np.bincount(meta_y).argmax())
            self._constant_meta = majority
            self._has_constant_meta = True
        else:
            self._has_constant_meta = False
            self.meta_model.fit(X[primary_mask], meta_y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return gated bet magnitude in [0, 1] and direction.

        Returns:
            Array of shape (n, 2): column 0 = P(side == -1), column 1 = P(side == +1)
            when the meta model gates the primary signal. Rows where the primary
            model is not confident are down-weighted via meta probability.
        """
        X = np.asarray(X)
        n = len(X)

        direction = self._primary_direction(X)
        meta_prob = self._meta_positive_probability(X)

        out = np.zeros((n, 2))
        for i in range(n):
            if direction[i] == 1:
                out[i, 1] = meta_prob[i]
                out[i, 0] = 1.0 - meta_prob[i]
            else:
                out[i, 0] = meta_prob[i]
                out[i, 1] = 1.0 - meta_prob[i]
        return out

    def predict(self, X: np.ndarray, bet_threshold: float = 0.5) -> np.ndarray:
        """Predict direction only where the meta model is confident enough.

        Args:
            X: Feature matrix.
            bet_threshold: Minimum meta probability to place a bet.

        Returns:
            Array of signed bets in {-1, 0, +1}: 0 = no bet (gated out).
        """
        proba = self.predict_proba(X)
        bets = np.zeros(len(proba), dtype=int)
        side = np.where(proba[:, 1] >= bet_threshold, 1, -1)
        confidence = np.maximum(proba[:, 0], proba[:, 1])
        bets[confidence >= bet_threshold] = side[confidence >= bet_threshold]
        return bets

    def _primary_direction(self, X: np.ndarray) -> np.ndarray:
        """Get signed direction from the primary model."""
        proba = self.primary_model.predict_proba(X)
        # Map model classes (e.g. -1, 1) to signed direction.
        signed = np.full(len(X), 0, dtype=int)
        for k, cls in enumerate(self.primary_classes_):
            signed[proba[:, k] >= self.primary_prob_threshold] = cls
        return signed

    def _meta_positive_probability(self, X: np.ndarray) -> np.ndarray:
        """P(primary model is correct) from the meta model."""
        if self._has_constant_meta:
            return np.full(len(X), float(self._constant_meta))
        return self.meta_model.predict_proba(X)[:, 1]


def make_meta_labels(
    primary_preds: np.ndarray,
    actual_labels: np.ndarray,
) -> np.ndarray:
    """Build meta-label targets: did the primary prediction match the outcome?

    Args:
        primary_preds: Predicted directions in {-1, 1}.
        actual_labels: Realized labels in {-1, 0, 1}.

    Returns:
        Binary array: 1 if prediction matches realized label, else 0.
        Rows with actual label 0 (no bet) return 0.
    """
    primary_preds = np.asarray(primary_preds)
    actual_labels = np.asarray(actual_labels)
    meta = np.zeros(len(primary_preds), dtype=int)
    active = actual_labels != 0
    meta[active] = (primary_preds[active] == actual_labels[active]).astype(int)
    return meta
