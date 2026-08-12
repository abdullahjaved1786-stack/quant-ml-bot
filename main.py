"""Paper-trading execution loop configured for 5-minute intervals in GitHub Actions.

Fetches fresh 5m OHLCV data, rebuilds features + triple-barrier labels,
retrains on a rolling TRAIN_WINDOW, and logs a single BUY / SIT_OUT decision.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass  # dotenv missing or .env has encoding issues — proceed without it

from core.data_fetcher import fetch_live_ohlcv
from core.execution import PositionManager
from core.features import FEATURE_COLUMNS, build_features, load_ohlcv
from core.labeling import triple_barrier
from core.logger import TradeLogger
from core.model import SignalModel
from core.self_improve import SelfImprovementLoop
from core.telemetry import Telemetry, log, setup_logging

TRAIN_WINDOW = 1000  # rolling bars fed to each training run
RETRAIN_EVERY = 100  # retrain after this many new bars
POLL_SECONDS = 60
POSITION_SIZE_FRACTION = 0.10  # allocate ~10% of cash per entry
DEFAULT_DATA_DIR = Path("data")


def fetch_bars(cfg: dict):
    if cfg.get("source") == "ccxt":
        return fetch_live_ohlcv(
            symbol=cfg["symbol"],
            timeframe=cfg.get("timeframe", "5m"),
            limit=cfg.get("limit", TRAIN_WINDOW),
            exchange_id=cfg.get("exchange") or "binance",
        )
    return load_ohlcv(
        cfg["symbol"],
        source=cfg.get("source", "yfinance"),
        interval=cfg.get("interval", "5m"),
        limit=cfg.get("limit", TRAIN_WINDOW + 200),
        exchange=cfg.get("exchange"),
        timeframe=cfg.get("timeframe", "5m"),
    )


class PaperBot:
    def __init__(self, cfg: dict, data_dir: Path = DEFAULT_DATA_DIR):
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = SignalModel()
        self.telemetry = Telemetry(
            sheet_name=cfg.get("sheet_name"),
            creds_path=cfg.get("creds_path"),
            symbol=self.symbol,
        )
        self.positions = PositionManager(portfolio_path=self.data_dir / "portfolio.json")
        self.trade_logger = TradeLogger(csv_path=self.data_dir / "trades.csv")
        self.self_improve = SelfImprovementLoop(
            trade_log_path=self.data_dir / "trades.csv",
            model_dir=self.data_dir / "models",
        )
        self.bars_processed = 0
        self.bars_since_retrain = None  # None until first train

    def on_new_data(self, df, feats, labels) -> dict:
        data = feats.join(labels.rename("label")).dropna()

        # 1. Train / retrain the model
        if self.bars_since_retrain is None:
            self._retrain(data)
            self.bars_since_retrain = 0
        else:
            new_bars = max(0, len(data) - self.bars_processed)
            self.bars_since_retrain += new_bars
            if self.bars_since_retrain >= RETRAIN_EVERY:
                self._retrain(data)
                self.bars_since_retrain %= RETRAIN_EVERY
        self.bars_processed = len(data)

        # 2. Check existing position for TP/SL exits using the latest closed candle
        latest_candle = df.iloc[-1].to_dict()
        for trade in self.positions.evaluate_exits(self.symbol, latest_candle):
            self.trade_logger.log_trade(trade)
            self.telemetry.log_trade(
                trade["side"], trade["entry_price"], trade["exit_price"],
                trade["size"], trade["pnl_usd"],
            )

        # 3. Optional 100-trade self-improvement loop
        improved_model = self.self_improve.maybe_retrain(df, logger=self.trade_logger)
        if improved_model is not None:
            self.model = improved_model

        # 4. Generate signal for the latest bar
        sig = self._signal(data, df)

        # 5. Publish portfolio state to Sheets (if configured)
        self.trade_logger.update_portfolio(self._portfolio_snapshot(sig["price"]))

        # 6. Act on signal if not already in a position
        if sig["signal"] == "BUY" and not self.positions.has_position(self.symbol):
            cash_to_deploy = self.positions.portfolio.cash_usd * POSITION_SIZE_FRACTION
            size = cash_to_deploy / sig["price"] if sig["price"] > 0 else 0.0
            self.positions.open_position(
                symbol=self.symbol,
                side=sig["signal"],
                entry_price=sig["price"],
                size=size,
                tp_price=sig["tp_price"],
                sl_price=sig["sl_price"],
            )
        return sig

    def _retrain(self, data) -> None:
        window = data.tail(TRAIN_WINDOW)
        self.model.train(window[FEATURE_COLUMNS], window["label"].astype(int))
        log.info("Retrained on %d bars (rolling window=%d)", len(window), TRAIN_WINDOW)

    def _signal(self, data, df) -> dict:
        sig = self.model.signal(
            data[FEATURE_COLUMNS].iloc[[-1]].to_numpy(),
            price=float(df["close"].iloc[-1]),
            atr=float(data["atr"].iloc[-1]),
        )
        self.telemetry.log_signal(sig)
        return sig

    def _portfolio_snapshot(self, last_price: float) -> dict:
        pm = self.positions
        p = pm.portfolio.positions
        total_trades = self.self_improve.completed_trade_count()
        winning_trades = self.self_improve.win_count()
        pos = pm.get_position(self.symbol)
        return {
            "timestamp": pm._now_iso() if hasattr(pm, "_now_iso") else "",
            "cash_usd": pm.portfolio.cash_usd,
            "equity_usd": pm.portfolio.cash_usd + (pos.size * last_price if pos else 0.0),
            "starting_cash_usd": pm.portfolio.starting_cash_usd,
            "open_positions": len(p),
            "position_symbol": pos.symbol if pos else "",
            "position_side": pos.side if pos else "",
            "position_size": pos.size if pos else 0.0,
            "position_entry_price": pos.entry_price if pos else 0.0,
            "position_tp_price": pos.tp_price if pos else 0.0,
            "position_sl_price": pos.sl_price if pos else 0.0,
            "position_bars_held": pos.bars_held if pos else 0,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": round(self.self_improve.win_rate(), 4),
        }


def run_once(bot: PaperBot, cfg: dict) -> dict:
    df = fetch_bars(cfg)
    feats = build_features(df)
    labels = triple_barrier(df["high"], df["low"], df["close"], feats["atr"])
    return bot.on_new_data(df, feats, labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--interval", default="5m", help="yfinance interval (5m, 1h, 1d, ...)")
    ap.add_argument("--source", choices=["yfinance", "ccxt"], default="yfinance")
    ap.add_argument("--exchange", default="binance", help="ccxt exchange id when --source ccxt")
    ap.add_argument("--timeframe", default="5m", help="ccxt timeframe when --source ccxt")
    ap.add_argument("--limit", type=int, default=TRAIN_WINDOW + 200, help="bars to fetch per poll")
    ap.add_argument("--poll", type=int, default=POLL_SECONDS, help="seconds between polls")
    ap.add_argument("--once", action="store_true", help="run a single iteration and exit")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="local state directory (portfolio, trades, models)")
    ap.add_argument("--sheet-name", default=os.environ.get("GOOGLE_SHEET_NAME"))
    ap.add_argument("--creds-path", default=os.environ.get("GOOGLE_SHEET_CREDS"))
    args = ap.parse_args()

    setup_logging()
    data_dir = Path(args.data_dir)
    bot = PaperBot(vars(args), data_dir=data_dir)

    log.info("Running scheduled 5m trading cycle via GitHub Actions...")
    try:
        sig = run_once(bot, vars(args))
        log.info("Cycle complete. Signal: %s (P(TP)=%.3f edge=%.4f)", sig["signal"], sig["p_tp"], sig["edge"])
    except Exception as exc:
        log.error("Execution failed: %s", exc)


if __name__ == "__main__":
    main()