"""Console + optional Google Sheets telemetry for signals and trades."""

from __future__ import annotations

import datetime as dt
import logging
import os

log = logging.getLogger("quant_bot")

SHEET_HEADERS = [
    "timestamp", "type", "symbol", "price", "p_tp", "p_sl",
    "edge", "tp_price", "sl_price", "signal", "pnl",
]


def setup_logging(level=logging.INFO) -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False  # avoid double-emit through the root logger


class Telemetry:
    """Logs signals/trades to the console, and to a Google Sheet when creds exist."""

    def __init__(self, sheet_name=None, creds_path=None, symbol=""):
        self.symbol = symbol
        self.sheet = None
        if sheet_name and creds_path and os.path.exists(creds_path):
            self._connect_sheet(sheet_name, creds_path)
        else:
            log.info("Google Sheets disabled (set GOOGLE_SHEET_NAME + GOOGLE_SHEET_CREDS to enable)")

    def _connect_sheet(self, sheet_name, creds_path):
        try:
            import gspread

            gc = gspread.service_account(creds_path)
            self.sheet = gc.open(sheet_name).sheet1
            log.info("Google Sheets connected: %s", sheet_name)
        except Exception as exc:
            log.warning("Sheets logging unavailable: %s", exc)

    def _append(self, values):
        if self.sheet:
            try:
                self.sheet.append_row(values, value_input_option="USER_ENTERED")
            except Exception as exc:
                log.warning("Sheets write failed: %s", exc)

    def log_signal(self, sig):
        log.info(
            "SIGNAL=%-7s price=%.2f P(TP)=%.3f P(SL)=%.3f edge=%.4f | TP=%.2f SL=%.2f",
            sig["signal"], sig["price"], sig["p_tp"], sig["p_sl"], sig["edge"],
            sig["tp_price"], sig["sl_price"],
        )
        self._append([
            dt.datetime.now().isoformat(timespec="seconds"), "signal", self.symbol,
            round(sig["price"], 6), round(sig["p_tp"], 4), round(sig["p_sl"], 4),
            round(sig["edge"], 6), round(sig["tp_price"], 6), round(sig["sl_price"], 6),
            sig["signal"], "",
        ])

    def log_trade(self, side, entry_price, exit_price=None, qty=None, pnl=None):
        log.info("TRADE %s @ %.2f qty=%s pnl=%s", side, entry_price, qty, pnl)
        self._append([
            dt.datetime.now().isoformat(timespec="seconds"), "trade", self.symbol,
            round(entry_price, 6), "", "", "", "", "", side, pnl,
        ])
