"""Position manager for paper trading with state persistence.

Tracks open positions in JSON so a restart resumes them, evaluates each closed
candle against TP/SL levels, and emits close events when fills trigger.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("quant_bot")

DEFAULT_PORTFOLIO_PATH = Path("data/portfolio.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ts) -> str:
    if ts is None:
        return _now_iso()
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


@dataclass
class Position:
    """A single open position (long-only in Phase 1)."""

    symbol: str
    side: str
    entry_price: float
    size: float
    tp_price: float
    sl_price: float
    entry_time: str
    position_id: str = ""
    bars_held: int = 0
    remaining_size: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    starting_cash_usd: float = 10_000.0
    cash_usd: float = 10_000.0
    positions: list[Position] = field(default_factory=list)

    def total_position_value_usd(self, latest_prices: dict) -> float:
        return sum(
            pos.size * latest_prices.get(pos.symbol, pos.entry_price)
            for pos in self.portfolio.positions if False  # placeholder, see method below
        ) if False else sum(
            p.size * latest_prices.get(p.symbol, p.entry_price) for p in self.positions
        )


class PositionManager:
    """Owns the portfolio JSON, opens/closes positions, evaluates exits."""

    def __init__(self, portfolio_path: Path | str = DEFAULT_PORTFOLIO_PATH):
        self.portfolio_path = Path(portfolio_path)
        self.portfolio: Portfolio = self._load()

    def _load(self) -> Portfolio:
        if not self.portfolio_path.exists():
            log.info("No portfolio at %s — starting fresh (cash=$%.2f)",
                     self.portfolio_path, 10_000.0)
            return Portfolio()
        try:
            raw = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
            positions = [Position(**p) for p in raw.get("positions", [])]
            return Portfolio(
                starting_cash_usd=raw.get("starting_cash_usd", 10_000.0),
                cash_usd=raw.get("cash_usd", raw.get("starting_cash_usd", 10_000.0)),
                positions=positions,
            )
        except Exception as exc:
            log.warning("Portfolio load failed (%s); starting fresh", exc)
            return Portfolio()

    def save(self) -> None:
        self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "starting_cash_usd": self.portfolio.starting_cash_usd,
            "cash_usd": self.portfolio.cash_usd,
            "positions": [p.to_dict() for p in self.portfolio.positions],
        }
        self.portfolio_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.debug("Saved portfolio to %s", self.portfolio_path)

    def get_position(self, symbol: str) -> Position | None:
        for pos in self.portfolio.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        tp_price: float,
        sl_price: float,
        entry_time: str | None = None,
        position_id: str | None = None,
    ) -> Position | None:
        """Open a position if no existing one for the symbol and cash covers it."""
        if self.get_position(symbol) is not None:
            log.info("Skip %s: position already open", symbol)
            return None

        notional = entry_price * size
        if notional > self.portfolio.cash_usd:
            log.info("Skip %s: notional $%.2f > cash $%.2f",
                     symbol, notional, self.portfolio.cash_usd)
            return None

        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            size=float(size),
            tp_price=float(tp_price),
            sl_price=float(sl_price),
            entry_time=entry_time or _now_iso(),
            position_id=position_id or uuid.uuid4().hex[:12],
            remaining_size=float(size),
        )
        self.portfolio.positions.append(pos)
        self.portfolio.cash_usd -= notional
        self.save()
        log.info(
            "OPEN %s %s id=%s size=%.6f @ %.2f TP=%.2f SL=%.2f cash=$%.2f",
            side, symbol, pos.position_id, size, entry_price, tp_price, sl_price,
            self.portfolio.cash_usd,
        )
        return pos

    def close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str,
        exit_time: str | None = None,
        size: float | None = None,
    ) -> dict:
        close_size = size if size is not None else position.remaining_size
        pnl_usd = (exit_price - position.entry_price) * close_size
        release = exit_price * close_size
        self.portfolio.cash_usd += release
        position.remaining_size -= close_size

        if position.remaining_size <= 1e-12:
            self.portfolio.positions = [
                p for p in self.portfolio.positions if p.position_id != position.position_id
            ]
        self.save()

        return_pct = (exit_price / position.entry_price - 1.0) * 100.0
        log.info(
            "CLOSE %s id=%s @ %.2f reason=%s pnl=$%.2f (%+.2f%%) cash=$%.2f",
            position.symbol, position.position_id, exit_price, exit_reason,
            pnl_usd, return_pct, self.portfolio.cash_usd,
        )
        return {
            "symbol": position.symbol,
            "side": position.side,
            "position_id": position.position_id,
            "entry_time": position.entry_time,
            "exit_time": exit_time or _now_iso(),
            "entry_price": position.entry_price,
            "exit_price": float(exit_price),
            "size": close_size,
            "pnl_usd": float(pnl_usd),
            "return_pct": float(return_pct),
            "exit_reason": exit_reason,
        }

    def evaluate_exits(self, symbol: str, candle) -> list[dict]:
        """Check TP/SL against one closed candle for the symbol's open position.

        Same-bar ties assume worst-case for longs (SL before TP).
        """
        pos = self.get_position(symbol)
        if pos is None:
            return []

        if isinstance(candle, dict):
            high = float(candle["high"])
            low = float(candle["low"])
            ts = candle.get("timestamp")
        else:
            high = float(candle["high"])
            low = float(candle["low"])
            ts = candle["timestamp"] if "timestamp" in candle.index else None

        fills: list[dict] = []
        if low <= pos.sl_price and pos.remaining_size > 1e-12:
            fills.append(self.close_position(pos, pos.sl_price, "STOP_LOSS", exit_time=_iso(ts)))
        if high >= pos.tp_price and pos.remaining_size > 1e-12:
            fills.append(self.close_position(pos, pos.tp_price, "TAKE_PROFIT", exit_time=_iso(ts)))

        if not fills and ts is not None:
            pos.bars_held += 1
        return fills
