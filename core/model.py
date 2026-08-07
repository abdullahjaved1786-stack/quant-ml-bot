"""LightGBM signal model with the friction-filter edge computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.labeling import SL_ATR, TP_ATR

FEE_RATE = 0.00075  # per-side fee as a fraction of price (0.075%)
SLIPPAGE_BPS = 1.0  # per-side slippage in basis points of price
P_TP_THRESHOLD = 0.55  # minimum P(TP) to take a BUY


class SignalModel:
    """Trains a LightGBM (RandomForest fallback) classifier on triple-barrier
    labels and turns class probabilities into a BUY / SIT_OUT decision."""

    def __init__(self, estimator=None, **params):
        self.model = estimator if estimator is not None else self._default_estimator(**params)
        self.feature_columns = None

    @staticmethod
    def _default_estimator(**params):
        try:
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 300),
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 31),
                verbosity=-1,
            )
        except ImportError:  # lightgbm missing -> RandomForest fallback
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 300),
                min_samples_leaf=10,
                n_jobs=-1,
            )

    def train(self, X, y):
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)
        y = y.reset_index(drop=True) if isinstance(y, pd.Series) else np.asarray(y)
        self.feature_columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        if isinstance(p, list):  # some estimators return [proba]
            p = p[0]
        idx = {int(c): i for i, c in enumerate(self.model.classes_)}
        n = len(p)
        p_tp = p[:, idx[1]] if 1 in idx else np.zeros(n)
        p_sl = p[:, idx[-1]] if -1 in idx else np.zeros(n)
        return p_tp, p_sl

    def signal(self, X_row, price, atr, slippage_bps=SLIPPAGE_BPS):
        """Expected edge = (P(TP)*TP_dist) - (P(SL)*SL_dist) - round-trip costs.

        Costs = 2 * (price * FEE_RATE + slippage)  [open + close legs].
        Returns "BUY" only if edge > 0 and P(TP) > P_TP_THRESHOLD, else "SIT_OUT".

        Accepts DataFrame, list, or ndarray for X_row. Re-wraps NumPy/list inputs
        as a DataFrame using stored feature_columns so LightGBM's fit-feature
        names match and the UserWarning is silenced.
        """
        if isinstance(X_row, pd.DataFrame):
            X = X_row
        elif self.feature_columns is not None:
            X = pd.DataFrame(np.asarray(X_row, dtype=float).reshape(1, -1),
                             columns=self.feature_columns)
        else:
            X = np.asarray(X_row, dtype=float).reshape(1, -1)
        p_tp, p_sl = self.predict_proba(X)
        tp_dist, sl_dist = TP_ATR * atr, SL_ATR * atr
        slippage = price * slippage_bps / 10_000.0
        costs = 2.0 * (price * FEE_RATE + slippage)
        edge = (p_tp[0] * tp_dist) - (p_sl[0] * sl_dist) - costs
        return {
            "signal": "BUY" if edge > 0.0 and p_tp[0] > P_TP_THRESHOLD else "SIT_OUT",
            "price": float(price),
            "p_tp": float(p_tp[0]),
            "p_sl": float(p_sl[0]),
            "p_time": float(max(0.0, 1.0 - p_tp[0] - p_sl[0])),
            "edge": float(edge),
            "tp_price": float(price + tp_dist),
            "sl_price": float(price - sl_dist),
        }
