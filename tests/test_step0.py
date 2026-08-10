"""Unit tests for Step 0: config, data loader, and atomic state manager."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import Step 0 modules
from src.config import Config
from src.data import (
    compute_atr,
    make_synthetic_ohlcv,
    align_data,
    load_cached_data,
    save_cached_data,
)
from src.state import StateManager, Position, Portfolio


# =============================================================================
# Config Tests
# =============================================================================

def test_config_defaults():
    """Config should have sensible defaults."""
    cfg = Config()
    assert cfg.pt_mult == 1.5
    assert cfg.sl_mult == 1.0
    assert cfg.max_hold == 5
    assert cfg.embargo_pct == 0.01
    assert cfg.kelly_fraction == 0.25
    assert cfg.kelly_cap == 0.05
    assert cfg.min_notional == 5.0
    assert cfg.drawdown_limit == 0.03
    assert cfg.atr_period == 14
    assert cfg.maker_fee == 0.0002


def test_config_singleton():
    """Global config instance should be accessible."""
    from src.config import config as global_config
    assert global_config is not None
    assert isinstance(global_config, Config)


def test_config_paths_created(tmp_path):
    """Config should create data directories on init."""
    cfg = Config(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    assert cfg.data_dir.exists()
    assert cfg.models_dir.exists()


# =============================================================================
# Data Loader Tests
# =============================================================================

def test_compute_atr_basic():
    """ATR should compute correctly on simple data."""
    df = pd.DataFrame({
        "high": [102, 103, 104, 105, 106],
        "low": [98, 99, 100, 101, 102],
        "close": [100, 101, 102, 103, 104],
    })
    atr = compute_atr(df, window=3)
    assert len(atr) == len(df)
    assert not atr.isna().all()
    # First value = TR = max(102-98, |102-100|, |98-100|) = max(4, 2, 2) = 4
    assert atr.iloc[0] == 4.0


def test_compute_atr_wilder_ema():
    """ATR should use Wilder's EMA (alpha=1/window)."""
    df = pd.DataFrame({
        "high": [110, 110, 110, 110, 110],
        "low": [90, 90, 90, 90, 90],
        "close": [100, 100, 100, 100, 100],
    })
    atr = compute_atr(df, window=14)
    assert atr.iloc[-1] > 0
    # Constant TR = 20, ATR should converge toward 20
    assert atr.iloc[-1] == pytest.approx(20.0, rel=0.1)


def test_make_synthetic_ohlcv_shape():
    """Synthetic OHLCV should have correct shape and invariants."""
    df = make_synthetic_ohlcv(n_bars=500, seed=42)
    assert len(df) == 500
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert df["volume"].gt(0).all()


def test_make_synthetic_ohlcv_reproducible():
    """Same seed should produce same data."""
    df1 = make_synthetic_ohlcv(n_bars=100, seed=123)
    df2 = make_synthetic_ohlcv(n_bars=100, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_align_data_basic():
    """Align should merge OHLCV with funding and OI."""
    ohlcv = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC"),
        "close": [100.0] * 10,
    })
    funding = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC"),
        "funding_rate": [0.0001, 0.0002],
    })
    oi = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC"),
        "open_interest": [1_000_000.0] * 10,
    })
    merged = align_data(ohlcv, funding, oi)
    assert "funding_rate" in merged.columns
    assert "open_interest" in merged.columns
    assert len(merged) == len(ohlcv)


def test_align_data_forward_fill_funding():
    """Funding rates should be forward-filled from 8h publication times."""
    ohlcv = pd.DataFrame({
        "timestamp": pd.date_range(
            "2024-01-01 00:00", periods=100, freq="5min", tz="UTC"
        ),
        "close": [100.0] * 100,
    })
    funding = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01 00:00", tz="UTC")],
        "funding_rate": [0.0005],
    })
    merged = align_data(ohlcv, funding)
    assert merged["funding_rate"].notna().sum() >= 90


def test_cache_roundtrip(tmp_path):
    """Cache save/load should preserve data."""
    df = make_synthetic_ohlcv(n_bars=100, seed=42)
    cache_path = tmp_path / "cache.csv"
    save_cached_data(df, cache_path)
    assert cache_path.exists()
    loaded = load_cached_data(cache_path, max_age_seconds=60)
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded)


def test_cache_stale(tmp_path):
    """Stale cache should return None."""
    import time
    df = make_synthetic_ohlcv(n_bars=10, seed=1)
    cache_path = tmp_path / "stale.csv"
    save_cached_data(df, cache_path)
    time.sleep(0.1)  # ensure mtime is strictly in the past
    loaded = load_cached_data(cache_path, max_age_seconds=0)
    assert loaded is None


def test_cache_missing(tmp_path):
    """Missing cache should return None."""
    loaded = load_cached_data(tmp_path / "missing.csv", max_age_seconds=60)
    assert loaded is None


# =============================================================================
# State Manager Tests
# =============================================================================

def test_position_creation():
    """Position should auto-generate ID and set defaults."""
    pos = Position(
        symbol="BTC/USDT",
        side="BUY",
        entry_price=100.0,
        size=0.5,
        tp_price=105.0,
        sl_price=95.0,
        entry_time="2024-01-01T00:00:00+00:00",
    )
    assert pos.position_id != ""
    assert pos.remaining_size == pos.size
    assert pos.bars_held == 0


def test_portfolio_defaults():
    """Portfolio should start with default cash."""
    pf = Portfolio()
    assert pf.cash_usd == 10_000.0
    assert pf.starting_cash_usd == 10_000.0
    assert pf.positions == []
    assert pf.total_trades == 0


def test_state_manager_load_fresh(tmp_path):
    """StateManager should create fresh portfolio if no state exists."""
    sm = StateManager(state_path=tmp_path / "portfolio.json")
    pf = sm.load()
    assert pf.cash_usd == 10_000.0
    assert pf.positions == []


def test_state_manager_save_load(tmp_path):
    """StateManager should persist and restore state."""
    sm = StateManager(state_path=tmp_path / "portfolio.json")
    pf = sm.load()
    pf.cash_usd = 9500.0
    pf.total_trades = 5
    sm.save(pf)

    sm2 = StateManager(state_path=tmp_path / "portfolio.json")
    pf2 = sm2.load()
    assert pf2.cash_usd == 9500.0
    assert pf2.total_trades == 5


def test_state_manager_atomic_write(tmp_path):
    """StateManager should write temp then atomic replace."""
    sm = StateManager(state_path=tmp_path / "portfolio.json")
    pf = sm.load()
    pf.cash_usd = 8000.0
    sm.save(pf)

    assert not sm.temp_path.exists()
    assert sm.state_path.exists()

    # Save again — now backup should exist from first save
    pf.cash_usd = 7000.0
    sm.save(pf)
    assert sm.backup_path.exists()


def test_state_manager_backup_recovery(tmp_path):
    """StateManager should recover from backup if main file is corrupted."""
    state_path = tmp_path / "portfolio.json"
    backup_path = tmp_path / "portfolio.bak"

    backup_path.write_text(json.dumps({
        "starting_cash_usd": 10_000.0,
        "cash_usd": 9000.0,
        "positions": [],
        "total_trades": 3,
        "winning_trades": 2,
        "daily_pnl": 0.0,
        "daily_start_equity": 10_000.0,
        "last_timestamp": "2024-01-01T00:00:00+00:00",
    }))
    state_path.write_text("{invalid json")

    sm = StateManager(state_path=state_path)
    pf = sm.load()
    assert pf.cash_usd == 9000.0
    assert pf.total_trades == 3


def test_open_position():
    """Opening position should deduct cash and add to positions."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    pos = sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    assert pos.symbol == "BTC/USDT"
    assert pos.entry_price == 100.0
    assert pos.size == 0.5
    pf = sm.load()
    assert pf.cash_usd == 10_000.0 - (100.0 * 0.5)
    assert len(pf.positions) == 1


def test_open_position_insufficient_cash():
    """Opening position should fail if insufficient cash."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    with pytest.raises(ValueError, match="Insufficient cash"):
        sm.open_position(
            symbol="BTC/USDT", side="BUY", entry_price=100_000.0,
            size=1.0, tp_price=105_000.0, sl_price=95_000.0,
        )


def test_close_position():
    """Closing position should return cash and calculate PnL."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    trade = sm.close_position("BTC/USDT", exit_price=110.0, exit_reason="TAKE_PROFIT")
    assert trade["pnl_usd"] == pytest.approx((110.0 - 100.0) * 0.5)
    assert trade["exit_reason"] == "TAKE_PROFIT"
    pf = sm.load()
    assert len(pf.positions) == 0
    assert pf.total_trades == 1
    assert pf.winning_trades == 1


def test_close_position_loss():
    """Closing position at loss should track correctly."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    trade = sm.close_position("BTC/USDT", exit_price=90.0, exit_reason="STOP_LOSS")
    assert trade["pnl_usd"] == pytest.approx((90.0 - 100.0) * 0.5)
    assert trade["pnl_usd"] < 0
    pf = sm.load()
    assert pf.winning_trades == 0


def test_get_position():
    """get_position should return position or None."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    assert sm.get_position("BTC/USDT") is None
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    pos = sm.get_position("BTC/USDT")
    assert pos is not None
    assert pos.symbol == "BTC/USDT"


def test_has_position():
    """has_position should return boolean."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    assert not sm.has_position("BTC/USDT")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    assert sm.has_position("BTC/USDT")


def test_win_rate():
    """win_rate should calculate correctly."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    assert sm.win_rate() == 0.0
    sm.open_position("BTC/USDT", "BUY", 100.0, 0.5, 105.0, 95.0)
    sm.close_position("BTC/USDT", 110.0)
    assert sm.win_rate() == 1.0
    sm.open_position("ETH/USDT", "BUY", 100.0, 0.5, 105.0, 95.0)
    sm.close_position("ETH/USDT", 90.0)
    assert sm.win_rate() == 0.5


def test_calculate_equity():
    """calculate_equity should sum cash and position values."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    assert sm.calculate_equity() == 10_000.0
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=1.0, tp_price=105.0, sl_price=95.0,
    )
    # Cash = 10000 - 100 = 9900, no position price override => use entry_price
    assert sm.calculate_equity() == 10_000.0
    # With price=110: equity = 9900 + 1.0*110 = 10010
    assert sm.calculate_equity(prices={"BTC/USDT": 110.0}) == 10_010.0


def test_reconcile_with_exchange():
    """reconcile should detect and fix size mismatches."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    discrepancies = sm.reconcile_with_exchange([
        {"symbol": "BTC/USDT", "side": "BUY", "size": 0.6, "entry_price": 100.0}
    ])
    assert len(discrepancies) == 1
    assert "Size mismatch" in discrepancies[0]
    pos = sm.get_position("BTC/USDT")
    assert pos.size == 0.6


def test_reconcile_missing_local():
    """reconcile should add exchange positions missing locally."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    discrepancies = sm.reconcile_with_exchange([
        {"symbol": "ETH/USDT", "side": "BUY", "size": 1.0, "entry_price": 200.0}
    ])
    assert len(discrepancies) == 1
    assert "missing locally" in discrepancies[0]
    pos = sm.get_position("ETH/USDT")
    assert pos is not None
    assert pos.size == 1.0


def test_reconcile_stale_local():
    """reconcile should remove local positions not on exchange."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    discrepancies = sm.reconcile_with_exchange([])
    assert len(discrepancies) == 1
    assert "not found on exchange" in discrepancies[0]
    assert not sm.has_position("BTC/USDT")


def test_increment_bars():
    """increment_bars should update bars_held counter."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position(
        symbol="BTC/USDT", side="BUY", entry_price=100.0,
        size=0.5, tp_price=105.0, sl_price=95.0,
    )
    for _ in range(5):
        sm.increment_bars("BTC/USDT")
    pos = sm.get_position("BTC/USDT")
    assert pos.bars_held == 5


def test_reset_daily():
    """reset_daily should reset daily PnL tracking."""
    sm = StateManager(state_path=Path(tempfile.mkdtemp()) / "portfolio.json")
    sm.open_position("BTC/USDT", "BUY", 100.0, 0.5, 105.0, 95.0)
    sm.close_position("BTC/USDT", 110.0)
    pf = sm.load()
    assert pf.daily_pnl > 0
    sm.reset_daily()
    pf = sm.load()
    assert pf.daily_pnl == 0.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
