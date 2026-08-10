"""Integration test: end-to-end pipeline from data loading to backtest.

Chains together all modules (Step 0–8) using synthetic data to verify
the full research pipeline works without crashes and produces sane outputs.
No network calls — all synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import make_synthetic_ohlcv, compute_atr
from src.labeling import apply_triple_barrier, compute_volatility_regime
from src.validation import PurgedKFold, run_parameter_sensitivity_grid
from src.meta_model import MetaLabelingModel
from src.pruning import compute_feature_importance
from src.regime import RegimeFilter
from src.sizing import fractional_kelly, compute_position_size
from src.backtest import backtest_cost_curve, deflated_sharpe_ratio
from src.risk import risk_gate
from src.monte_carlo import bootstrap_sharpe

from sklearn.linear_model import LogisticRegression


def _make_ohlcv(n=500, seed=42):
    """Generate synthetic OHLCV for integration tests."""
    df = make_synthetic_ohlcv(n_bars=n, seed=seed, start_price=100.0, volatility=0.01)
    df = df.set_index("timestamp")
    return df


# =============================================================================
# Full pipeline integration
# =============================================================================


def test_full_pipeline_smoke():
    """Chain data → labeling → validation → backtest; no crashes."""
    df = _make_ohlcv(n=500, seed=0)

    # Step 1: labeling
    labels_df = apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    assert len(labels_df) > 0
    assert set(labels_df["label"].unique()).issubset({-1, 0, 1})

    # Step 4: regime filter
    rf = RegimeFilter(low_pct=0.2, high_pct=0.8, allowed=("MEDIUM",))
    df_regime = rf.fit_transform(df)
    assert "tradable" in df_regime.columns

    # Step 2: purged CV (just structure — no ML fit here)
    n_labels = len(labels_df)
    pkf = PurgedKFold(n_splits=4, embargo=1)
    splits = list(pkf.split(n_labels))
    assert len(splits) == 4

    # Step 6: backtest
    close = df["close"].iloc[:len(labels_df)]
    signals = labels_df.set_index(close.index)["label"]
    vma = pd.Series(50_000.0, index=close.index)
    result = backtest_cost_curve(signals, close, size_series=1.0, vma=vma)
    assert hasattr(result, "deflated_sharpe")
    assert result.max_drawdown >= 0.0


def test_meta_model_end_to_end():
    """Full meta-labeling: synthetic features → train → meta → predict."""
    rng = np.random.default_rng(10)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    y = np.where(X["f0"] > 0, 1, -1)

    primary = LogisticRegression(max_iter=500)
    meta = LogisticRegression(max_iter=500)
    model = MetaLabelingModel(primary, meta)
    model.fit(X.values, y)

    proba = model.predict_proba(X.values)
    assert proba.shape == (300, 2)
    bets = model.predict(X.values, bet_threshold=0.6)
    assert (bets != 0).sum() > 0


def test_pruning_in_pipeline():
    """Pruning should correctly rank informative features."""
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    y = np.where(X["f0"] + X["f1"] > 0, 1, -1)

    model = LogisticRegression(max_iter=500).fit(X, y)
    imp = compute_feature_importance(model, X, y, n_repeats=3, seed=0)
    top2 = set(imp.index[:2])
    assert top2 == {"f0", "f1"}


def test_sizing_and_risk_in_pipeline():
    """Kelly sizing → risk gate → valid position."""
    equity = 10_000.0
    price = 100.0

    # Compute Kelly-based size
    kelly = compute_position_size(equity, price, win_prob=0.6, win_loss_ratio=1.5, min_notional=1.0)
    assert kelly["size_units"] > 0

    # Check risk gate
    ok, reasons = risk_gate(
        equity=equity,
        start_equity=equity,
        existing_exposure=0.0,
        price=price,
        size=kelly["size_units"],
    )
    assert ok is True
    assert reasons == ["ok"]


def test_monte_carlo_in_pipeline():
    """Block bootstrap Sharpe CI should contain the observed value."""
    rng = np.random.default_rng(12)
    returns = rng.normal(0.001, 0.01, 300)
    result = bootstrap_sharpe(returns, n_bootstrap=100, block_len=10, seed=0)
    assert result["ci_lower"] <= result["observed"] <= result["ci_upper"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_full_pipeline_labels_in_sane_range():
    """Labels should have a reasonable distribution — not all same class."""
    df = _make_ohlcv(n=300, seed=50)
    labels_df = apply_triple_barrier(df, pt_mult=1.5, sl_mult=1.0, max_hold=5)
    counts = labels_df["label"].value_counts()
    # At least two classes should be present
    assert len(counts) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
