"""Trade + portfolio + model-evolution logger.

Writes three artifacts:
  * trades.csv        — one row per closed trade (Phase 3 schema).
  * portfolio.json    — current portfolio snapshot (already owned by PositionManager).
  * Google Sheet tabs — when env vars present:
        Trades            (append-only trade rows)
        Portfolio         (single live row, overwritten each poll)
        Model_Evolution   (append-only retraining metrics)
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

log = logging.getLogger("quant_bot")

DEFAULT_TRADES_CSV = Path("data/trades.csv")
CSV_HEADERS = [
    "entry_time", "exit_time", "symbol", "side", "entry_price", "exit_price",
    "size", "pnl_usd", "return_pct", "exit_reason", "position_id",
]

PORTFOLIO_HEADERS = [
    "timestamp", "cash_usd", "equity_usd", "starting_cash_usd",
    "open_positions", "position_symbol", "position_side", "position_size",
    "position_entry_price", "position_tp_price", "position_sl_price",
    "position_bars_held", "total_trades", "winning_trades", "win_rate",
]

MODEL_EVOLUTION_HEADERS = [
    "timestamp", "trigger", "n_features", "n_samples", "n_estimators",
    "learning_rate", "oos_accuracy", "in_sample_accuracy", "trade_count",
    "duration_seconds", "model_path", "notes",
]

TAB_TRADES = "Trades"
TAB_PORTFOLIO = "Portfolio"
MODEL_EVOLUTION_TAB = "Model_Evolution"


def _maybe_connect(sheet_name, creds_path, tab_name, headers):
    """Return a gspread Worksheet or None. Ensures header row exists."""
    if not (sheet_name and creds_path and os.path.exists(creds_path)):
        return None
    try:
        import gspread

        gc = gspread.service_account(creds_path)
        sh = gc.open(sheet_name)
        try:
            ws = sh.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab_name, rows="2000", cols="40")
            ws.append_row(headers, value_input_option="USER_ENTERED")
            return ws
        existing = ws.row_values(1)
        if existing[: len(headers)] != headers:
            ws.insert_row(headers, 1)
        return ws
    except Exception as exc:
        log.warning("Sheets connect failed for tab %s: %s", tab_name, exc)
        return None


class TradeLogger:
    """Writes trades to CSV + 3 Google Sheets tabs when configured."""

    def __init__(
        self,
        csv_path: Path | str = DEFAULT_TRADES_CSV,
        sheet_name: str | None = None,
        creds_path: str | None = None,
    ):
        self.csv_path = Path(csv_path)
        env_sheet = sheet_name or os.environ.get("GOOGLE_SHEET_NAME")
        env_creds = creds_path or os.environ.get("GOOGLE_SHEET_CREDS")
        self.sheet_trades = _maybe_connect(env_sheet, env_creds, TAB_TRADES, CSV_HEADERS)
        self.sheet_portfolio = _maybe_connect(env_sheet, env_creds, TAB_PORTFOLIO, PORTFOLIO_HEADERS)
        self.sheet_model = _maybe_connect(env_sheet, env_creds, MODEL_EVOLUTION_TAB, MODEL_EVOLUTION_HEADERS)
        if any((self.sheet_trades, self.sheet_portfolio, self.sheet_model)):
            log.info(
                "Sheets connected: trades=%s portfolio=%s model=%s",
                bool(self.sheet_trades), bool(self.sheet_portfolio), bool(self.sheet_model),
            )
        else:
            log.info("Google Sheets disabled (no GOOGLE_SHEET_NAME/GOOGLE_SHEET_CREDS)")

    # ------------------------------------------------------------------ trades
    def log_trade(self, trade: dict) -> None:
        row = [trade.get(h, "") for h in CSV_HEADERS]
        self._write_csv(row)
        if self.sheet_trades is not None:
            try:
                self.sheet_trades.append_row([str(v) for v in row], value_input_option="USER_ENTERED")
            except Exception as exc:
                log.warning("Sheets trade append failed: %s", exc)

    # --------------------------------------------------------------- portfolio
    def update_portfolio(self, snapshot: dict) -> None:
        """Overwrite a single live row at row 2 (row 1 is header)."""
        if self.sheet_portfolio is None:
            return
        row = [snapshot.get(h, "") for h in PORTFOLIO_HEADERS]
        try:
            end_col = chr(ord("A") + len(PORTFOLIO_HEADERS) - 1)
            self.sheet_portfolio.update(
                f"A2:{end_col}2",
                [list(map(str, row))],
                value_input_option="USER_ENTERED",
            )
        except Exception as exc:
            log.warning("Sheets portfolio update failed: %s", exc)

    # -------------------------------------------------------- model evolution
    def log_model_retrain(self, metrics: dict) -> None:
        row = [metrics.get(h, "") for h in MODEL_EVOLUTION_HEADERS]
        if self.sheet_model is not None:
            try:
                self.sheet_model.append_row([str(v) for v in row], value_input_option="USER_ENTERED")
            except Exception as exc:
                log.warning("Sheets model log failed: %s", exc)

    # --------------------------------------------------------------- private
    def _write_csv(self, row) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(CSV_HEADERS)
                writer.writerow(row)
        except Exception as exc:
            log.warning("CSV write failed: %s", exc)
