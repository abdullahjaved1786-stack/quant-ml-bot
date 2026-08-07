"""Phase 5 synthetic verification.

Exercises:
  * TradeLogger 3-tab header schema
  * SelfImprovementLoop.trigger logic (every 100 trades boundary)
  * Augmented feature matrix shape with synthetic trade log
  * PositionManager.snapshot + win-rate math
  * Sheets import path (gspread + google-auth) without requiring live creds
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.execution import PositionManager
from core.logger import (
    CSV_HEADERS,
    MODEL_EVOLUTION_HEADERS,
    PORTFOLIO_HEADERS,
    TAB_TRADES,
    TAB_PORTFOLIO,
    MODEL_EVOLUTION_TAB,
    TradeLogger,
)
from core.self_improve import SelfImprovementLoop, TRADE_FEATURES


def _make_trade_csv(path: Path, n: int = 100) -> None:
    rng = np.random.default_rng(seed=42)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        for i in range(n):
            entry = 100.0 + i * 0.1
            exit_p = entry * (1 + rng.normal(0.001, 0.01))
            pnl = (exit_p - entry) * 0.5
            ret_pct = (exit_p / entry - 1) * 100
            w.writerow([
                f"2025-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                f"2025-01-{(i % 28) + 1:02d}T01:00:00+00:00",
                "BTC/USDT", "BUY",
                round(entry, 4), round(exit_p, 4),
                0.5, round(pnl, 4), round(ret_pct, 4),
                "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                f"pos_{i:04d}",
            ])


def _append_trade_csv(path: Path, n: int = 100) -> None:
    """Append n synthetic rows to an existing trades.csv."""
    start = len(pd.read_csv(path))
    rng = np.random.default_rng(seed=99)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for offset in range(n):
            i = start + offset
            entry = 100.0 + i * 0.1
            exit_p = entry * (1 + rng.normal(0.001, 0.01))
            pnl = (exit_p - entry) * 0.5
            ret_pct = (exit_p / entry - 1) * 100
            w.writerow([
                f"2025-02-{(i % 28) + 1:02d}T00:00:00+00:00",
                f"2025-02-{(i % 28) + 1:02d}T01:00:00+00:00",
                "BTC/USDT", "BUY",
                round(entry, 4), round(exit_p, 4),
                0.5, round(pnl, 4), round(ret_pct, 4),
                "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
                f"pos_{i:04d}",
            ])


def _make_ohlcv(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0.3, 0.1, n))
    low = close - np.abs(rng.normal(0.3, 0.1, n))
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.integers(50, 500, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_logger_schema() -> None:
    assert CSV_HEADERS == [
        "entry_time", "exit_time", "symbol", "side", "entry_price", "exit_price",
        "size", "pnl_usd", "return_pct", "exit_reason", "position_id",
    ]
    assert "timestamp" in PORTFOLIO_HEADERS and "win_rate" in PORTFOLIO_HEADERS
    assert "trigger" in MODEL_EVOLUTION_HEADERS and "model_path" in MODEL_EVOLUTION_HEADERS
    assert {TAB_TRADES, TAB_PORTFOLIO, MODEL_EVOLUTION_TAB} == {"Trades", "Portfolio", "Model_Evolution"}
    print("OK: logger schema")


def test_logger_csv_writes(tmp_path: Path) -> None:
    csv_path = tmp_path / "trades.csv"
    logger = TradeLogger(csv_path=csv_path, sheet_name=None, creds_path=None)
    assert logger.sheet_trades is None and logger.sheet_portfolio is None and logger.sheet_model is None

    trade = {
        "entry_time": "2025-01-01T00:00:00+00:00",
        "exit_time": "2025-01-01T01:00:00+00:00",
        "symbol": "BTC/USDT", "side": "BUY",
        "entry_price": 100.0, "exit_price": 101.0, "size": 0.5,
        "pnl_usd": 0.5, "return_pct": 1.0, "exit_reason": "TAKE_PROFIT",
        "position_id": "p1",
    }
    logger.log_trade(trade)
    assert csv_path.exists() and csv_path.read_text(encoding="utf-8").count("\n") == 2
    logger.update_portfolio({"timestamp": "now", "cash_usd": 1000, "equity_usd": 1000,
                             "starting_cash_usd": 1000, "open_positions": 0,
                             "total_trades": 1, "winning_trades": 1, "win_rate": 1.0})
    logger.log_model_retrain({"timestamp": "now", "trigger": "test",
                              "n_features": 8, "n_samples": 200, "n_estimators": 300,
                              "learning_rate": 0.05, "oos_accuracy": 0.6,
                              "in_sample_accuracy": 0.8, "trade_count": 1,
                              "duration_seconds": 0.5, "model_path": "x",
                              "notes": "synthetic"})
    print("OK: logger CSV writes (no sheets)")


def test_retrain_trigger_math(tmp_path: Path) -> None:
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    trade_log = tmp_path / "trades.csv"
    loop = SelfImprovementLoop(trade_log_path=trade_log, trigger_every=100,
                                model_dir=tmp_path / "models")

    assert loop.completed_trade_count() == 0
    assert not loop.should_trigger()

    _make_trade_csv(trade_log, n=99)
    loop._last_trigger_at = 0
    assert not loop.should_trigger(), "99 trades should not trigger"

    _make_trade_csv(trade_log, n=100)
    assert loop.should_trigger(), "100 trades must trigger"
    loop._last_trigger_at = 100

    # Append 100 more -> 200 total, must trigger again
    _append_trade_csv(trade_log, n=100)
    loop._last_trigger_at = 100
    assert loop.completed_trade_count() == 200
    assert loop.should_trigger(), "200 trades must trigger second time"
    print("OK: 100-trade trigger boundary math")


def test_augmented_features(tmp_path: Path) -> None:
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    trade_log = tmp_path / "trades.csv"
    _make_trade_csv(trade_log, n=50)
    loop = SelfImprovementLoop(trade_log_path=trade_log, model_dir=tmp_path / "models")
    df = _make_ohlcv(n=300)
    feats = loop._augment_features(df)
    assert all(c in feats.columns for c in ["atr", "log_return"]), "base features missing"
    for col in TRADE_FEATURES:
        assert col in feats.columns, f"missing augmented column: {col}"
    assert feats.shape[0] == len(df)
    print(f"OK: augmented features shape={feats.shape}")


def test_retrain_pipeline(tmp_path: Path) -> None:
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    trade_log = tmp_path / "trades.csv"
    _make_trade_csv(trade_log, n=100)
    loop = SelfImprovementLoop(trade_log_path=trade_log, model_dir=tmp_path / "models")
    df = _make_ohlcv(n=400)
    assert loop.should_trigger(), "precondition: 100 trades must trigger"
    model, metrics = loop.retrain(df)
    assert metrics["trade_count"] == 100
    assert metrics["model_path"].endswith(".joblib")
    assert Path(metrics["model_path"]).exists(), "model artifact must be persisted"
    assert isinstance(metrics["oos_accuracy"], float)
    assert isinstance(metrics["in_sample_accuracy"], float)
    assert len(list(loop.model_dir.glob("model_*.joblib"))) >= 1
    print(f"OK: retrain -> {metrics['model_path']} (OOS={metrics['oos_accuracy']:.3f})")


def test_portfolio_snapshot(tmp_path: Path) -> None:
    import shutil
    p = tmp_path / "portfolio.json"
    if p.exists():
        p.unlink()
    pm = PositionManager(portfolio_path=p)
    pos = pm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0, size=1.0,
        tp_price=110.0, sl_price=95.0,
    )
    assert pos is not None and pm.has_position("BTC/USDT")
    snap = {
        "timestamp": "2025-01-01T00:00:00+00:00",
        "cash_usd": pm.portfolio.cash_usd,
        "equity_usd": pm.portfolio.cash_usd + pos.size * 100.0,
        "starting_cash_usd": pm.portfolio.starting_cash_usd,
        "open_positions": len(pm.portfolio.positions),
        "position_symbol": pos.symbol, "position_side": pos.side, "position_size": pos.size,
        "position_entry_price": pos.entry_price, "position_tp_price": pos.tp_price,
        "position_sl_price": pos.sl_price, "position_bars_held": pos.bars_held,
        "total_trades": 0, "winning_trades": 0, "win_rate": 0.0,
    }
    assert snap["open_positions"] == 1 and snap["equity_usd"] == 10_000.0
    print("OK: portfolio snapshot math")


def test_sheets_imports() -> None:
    import importlib
    for mod in ("gspread", "google.auth", "google.oauth2"):
        importlib.import_module(mod)
    print("OK: gspread + google-auth importable")


def main() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="phase5_test_") as tmp:
        base = Path(tmp)
        test_logger_schema()
        test_sheets_imports()
        test_portfolio_snapshot(base / "portfolio")
        test_retrain_trigger_math(base / "trigger")
        test_augmented_features(base / "aug")
        test_retrain_pipeline(base / "pipeline")
        test_logger_csv_writes(base / "csv")
    print("\nALL PHASE 5 CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
