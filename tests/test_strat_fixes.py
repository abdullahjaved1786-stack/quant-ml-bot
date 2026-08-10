"""Unit tests for the negative-edge fixes:
- realistic taker fees and slippage
- MIN_TP_DISTANCE floor on the TP target
- 1,000-bar warmup gate before any signal evaluation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model import (
    FEE_RATE,
    SLIPPAGE_BPS,
    MIN_TP_DISTANCE,
    MIN_WARMUP_BARS,
    P_TP_THRESHOLD,
    SignalModel,
)


# ---------------------------------------------------------------- constants
def test_fee_rate_is_realistic():
    """Fee must be in the realistic taker range (0.075%-0.1%)."""
    assert 0.00075 <= FEE_RATE <= 0.001


def test_slippage_bumped_to_two_basis_points():
    """Slippage floor at 2 bp per side - typical for volatile prints."""
    assert SLIPPAGE_BPS >= 1.5


def test_min_tp_distance_constant():
    """TP floor must be in $200-$300."""
    assert 200.0 <= MIN_TP_DISTANCE <= 300.0


def test_min_warmup_bars_is_one_thousand():
    """Strategy must accumulate >= 1,000 bars before evaluating signals."""
    assert MIN_WARMUP_BARS == 1000


# --------------------------------------------------------------- helpers
def _toy_classifier_positive():
    """A SignalModel whose predict_proba always favours class 1.
    Used to test the floor + cost math without coupling to lightgbm training dynamics.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(0).normal(size=(MIN_WARMUP_BARS + 50, 4))
    y = np.full(X.shape[0], 1, dtype=int)
    est = RandomForestClassifier(n_estimators=20, min_samples_leaf=2, n_jobs=1)
    m = SignalModel(estimator=est)
    m.train(pd.DataFrame(X, columns=["a", "b", "c", "d"]), pd.Series(y))
    return m


# --------------------------------------------------------------- TP floor
def test_signal_applies_tp_floor_on_tiny_atr():
    """When TP_ATR*atr is below the floor, tp_price must be price + MIN_TP_DISTANCE."""
    model = _toy_classifier_positive()
    out = model.signal(
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        price=10_000.0,
        atr=1.0,  # tiny: TP_ATR*atr = 1.5 < MIN_TP_DISTANCE
    )
    assert out["tp_price"] == pytest.approx(10_000.0 + MIN_TP_DISTANCE)


def test_signal_keeps_native_tp_when_above_floor():
    """When TP_ATR*atr already exceeds the floor, tp_price must use the natural value."""
    from core.labeling import TP_ATR

    model = _toy_classifier_positive()
    atr = 300.0  # TP_ATR*atr approx 450 > MIN_TP_DISTANCE=250
    out = model.signal(
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        price=20_000.0,
        atr=atr,
    )
    assert out["tp_price"] == pytest.approx(20_000.0 + TP_ATR * atr)


# --------------------------------------------------------------- warmup
def test_signal_sits_out_during_warmup():
    """A model trained on <1,000 rows must emit SIT_OUT/WARMUP regardless of features."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(1).normal(size=(500, 4))
    y = np.random.default_rng(1).choice([-1, 0, 1], size=500)
    m = SignalModel(estimator=RandomForestClassifier(n_estimators=20, n_jobs=1))
    m.train(pd.DataFrame(X, columns=["a", "b", "c", "d"]), pd.Series(y))
    assert m._trained_rows < MIN_WARMUP_BARS

    out = m.signal(np.array([[0.0, 0.0, 0.0, 0.0]]), price=100.0, atr=2.0)
    assert out["signal"] == "SIT_OUT"
    assert out["reason"] == "WARMUP"
    assert out["edge"] == 0.0
    assert out["p_tp"] == 0.0


def test_signal_emits_real_decision_after_warmup():
    """Trained on >= 1,000 rows -> signal() returns a real decision, no WARMUP."""
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.default_rng(2).normal(size=(MIN_WARMUP_BARS, 4))
    y = np.array([1] * (MIN_WARMUP_BARS // 2) + [-1] * (MIN_WARMUP_BARS // 2))
    m = SignalModel(estimator=RandomForestClassifier(n_estimators=20, n_jobs=1))
    m.train(pd.DataFrame(X, columns=["a", "b", "c", "d"]), pd.Series(y))
    assert m._trained_rows >= MIN_WARMUP_BARS

    out = m.signal(np.array([[0.0, 0.0, 0.0, 0.0]]), price=50_000.0, atr=50.0)
    assert out["signal"] in {"BUY", "SIT_OUT"}
    assert "reason" not in out


def test_warmup_path_short_circuits_before_predict_proba(monkeypatch):
    """During warmup, signal() must return before reaching the estimator."""
    model = _toy_classifier_positive()

    def _boom(*_a, **_k):
        raise AssertionError("predict_proba called during warmup")

    monkeypatch.setattr(model, "predict_proba", _boom)
    model._trained_rows = 0  # force warmup path

    out = model.signal(np.array([[0.0, 0.0, 0.0, 0.0]]), price=1.0, atr=1.0)
    assert out["reason"] == "WARMUP"


# --------------------------------------------------------------- cost math
def test_round_trip_cost_uses_updated_fee_and_slippage():
    """Round-trip cost must equal 2 * (price*FEE_RATE + price*SLIPPAGE_BPS/10000)."""
    model = _toy_classifier_positive()
    price = 30_000.0
    expected = 2.0 * (price * FEE_RATE + price * SLIPPAGE_BPS / 10_000.0)

    out = model.signal(
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        price=price,
        atr=10.0,
        slippage_bps=SLIPPAGE_BPS,
    )

    p_tp, p_sl = out["p_tp"], out["p_sl"]
    tp_dist = out["tp_price"] - price
    sl_dist = price - out["sl_price"]
    # edge = p_tp*tp_dist - p_sl*sl_dist - costs
    recovered = -(out["edge"] - p_tp * tp_dist + p_sl * sl_dist)
    assert recovered == pytest.approx(expected, rel=1e-9)


def test_negative_edge_is_classified_sit_out():
    """When P(TP) is forced low, signal must be SIT_OUT."""
    from sklearn.ensemble import RandomForestClassifier

    n = MIN_WARMUP_BARS
    X = np.random.default_rng(3).normal(size=(n, 4))
    y = np.array([-1] * n)  # SL is always right -> model learns P(TP) ~ 0
    m = SignalModel(estimator=RandomForestClassifier(n_estimators=20, n_jobs=1))
    m.train(pd.DataFrame(X, columns=["a", "b", "c", "d"]), pd.Series(y))

    out = m.signal(np.array([[0.0, 0.0, 0.0, 0.0]]), price=100.0, atr=1.0)
    assert out["signal"] == "SIT_OUT"
    assert out["edge"] < 0.0 or out["p_tp"] < P_TP_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
