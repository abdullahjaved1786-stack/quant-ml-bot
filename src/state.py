"""Atomic state manager for portfolio.json with crash recovery."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import config


@dataclass
class Position:
    """Open position state."""

    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    size: float
    tp_price: float  # Take profit price
    sl_price: float  # Stop loss price
    entry_time: str  # ISO 8601 timestamp
    position_id: str = ""
    bars_held: int = 0
    remaining_size: float = 0.0

    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"pos_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if self.remaining_size == 0.0:
            self.remaining_size = self.size

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    """Portfolio state."""

    starting_cash_usd: float = 10_000.0
    cash_usd: float = 10_000.0
    positions: list[dict] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    daily_pnl: float = 0.0
    daily_start_equity: float = 10_000.0
    last_timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StateManager:
    """Atomic state persistence with temp file writing and crash recovery.

    Writes to a temporary file first, then atomically replaces the target.
    On startup, reconciles any stale state with exchange positions.
    """

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or config.data_dir / "portfolio.json"
        self.temp_path = self.state_path.with_suffix(".tmp")
        self.backup_path = self.state_path.with_suffix(".bak")
        self._portfolio: Portfolio | None = None

    def load(self) -> Portfolio:
        """Load portfolio state, with backup fallback on corruption."""
        if self._portfolio is not None:
            return self._portfolio

        # Try primary file
        data = self._read_json(self.state_path)
        if data is not None:
            self._portfolio = Portfolio.from_dict(data)
            return self._portfolio

        # Try backup on corruption
        data = self._read_json(self.backup_path)
        if data is not None:
            self._portfolio = Portfolio.from_dict(data)
            # Save restored state
            self.save(self._portfolio)
            return self._portfolio

        # Fresh state
        self._portfolio = Portfolio()
        return self._portfolio

    def _read_json(self, path: Path) -> dict | None:
        """Read JSON file, return None on error."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, portfolio: Portfolio | None = None) -> None:
        """Atomically save portfolio state.

        1. Write to temp file
        2. Sync to disk
        3. Backup existing state
        4. Atomically replace target
        """
        portfolio = portfolio or self._portfolio
        if portfolio is None:
            return

        portfolio.last_timestamp = datetime.now(timezone.utc).isoformat()
        self._portfolio = portfolio

        # Ensure directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file
        data = portfolio.to_dict()
        try:
            with open(self.temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            # If temp write fails, don't corrupt existing state
            if self.temp_path.exists():
                self.temp_path.unlink()
            raise

        # Backup existing state
        if self.state_path.exists():
            try:
                self.state_path.replace(self.backup_path)
            except OSError:
                pass  # Continue without backup if it fails

        # Atomic replace
        self.temp_path.replace(self.state_path)

    def get_position(self, symbol: str) -> Position | None:
        """Get open position for symbol, or None."""
        portfolio = self.load()
        for pos_dict in portfolio.positions:
            if pos_dict.get("symbol") == symbol:
                return Position(**pos_dict)
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return self.get_position(symbol) is not None

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        tp_price: float,
        sl_price: float,
    ) -> Position:
        """Open a new position, deducting cash from portfolio."""
        portfolio = self.load()

        # Check capital constraints
        cost = entry_price * size
        if cost > portfolio.cash_usd:
            raise ValueError(
                f"Insufficient cash: need ${cost:.2f}, have ${portfolio.cash_usd:.2f}"
            )

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            tp_price=tp_price,
            sl_price=sl_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
        )

        # Update portfolio
        portfolio.cash_usd -= cost
        portfolio.positions.append(position.to_dict())
        self.save(portfolio)

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
    ) -> dict:
        """Close position, returning trade PnL and details."""
        portfolio = self.load()
        position = self.get_position(symbol)
        if position is None:
            raise ValueError(f"No position for {symbol}")

        # Calculate PnL
        if position.side == "BUY":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        return_pct = (exit_price / position.entry_price - 1) * 100
        if position.side == "SELL":
            return_pct = -return_pct

        # Update portfolio
        portfolio.cash_usd += exit_price * position.size
        portfolio.positions = [
            p for p in portfolio.positions if p.get("symbol") != symbol
        ]
        portfolio.total_trades += 1
        if pnl > 0:
            portfolio.winning_trades += 1

        # Track daily PnL for circuit breaker
        portfolio.daily_pnl += pnl

        self.save(portfolio)

        return {
            "entry_time": position.entry_time,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl_usd": pnl,
            "return_pct": return_pct,
            "exit_reason": exit_reason,
            "position_id": position.position_id,
        }

    def increment_bars(self, symbol: str) -> None:
        """Increment bars_held counter for position."""
        portfolio = self.load()
        updated = False
        for pos in portfolio.positions:
            if pos.get("symbol") == symbol:
                pos["bars_held"] = pos.get("bars_held", 0) + 1
                updated = True
                break
        if updated:
            self.save(portfolio)

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        portfolio = self.load()
        portfolio.daily_pnl = 0.0
        portfolio.daily_start_equity = self.calculate_equity()
        self.save(portfolio)

    def calculate_equity(self, prices: dict[str, float] | None = None) -> float:
        """Calculate total equity (cash + position values).

        Args:
            prices: Optional dict mapping symbol -> current price.
                   If not provided, uses entry_price for positions.
        """
        portfolio = self.load()
        equity = portfolio.cash_usd

        for pos_dict in portfolio.positions:
            pos = Position(**pos_dict)
            price = (prices or {}).get(pos.symbol, pos.entry_price)
            equity += price * pos.size

        return equity

    def win_rate(self) -> float:
        """Calculate win rate from completed trades."""
        portfolio = self.load()
        if portfolio.total_trades == 0:
            return 0.0
        return portfolio.winning_trades / portfolio.total_trades

    def reconcile_with_exchange(
        self,
        exchange_positions: list[dict],
    ) -> list[str]:
        """Reconcile local state with exchange positions after crash.

        Args:
            exchange_positions: List of position dicts from exchange API
                with keys: symbol, side, size, entry_price.

        Returns:
            List of discrepancies found.
        """
        portfolio = self.load()
        discrepancies = []

        # Build exchange position map
        exchange_map = {(p["symbol"], p["side"]): p for p in exchange_positions}

        # Check local positions
        for pos_dict in portfolio.positions[:]:
            pos = Position(**pos_dict)
            key = (pos.symbol, pos.side)

            if key not in exchange_map:
                discrepancies.append(
                    f"Local position {pos.symbol} {pos.side} not found on exchange"
                )
                # Remove stale local position
                portfolio.positions.remove(pos_dict)
            else:
                ex_pos = exchange_map[key]
                if abs(ex_pos["size"] - pos.size) > config.step_size:
                    discrepancies.append(
                        f"Size mismatch for {pos.symbol}: local={pos.size}, "
                        f"exchange={ex_pos['size']}"
                    )
                    # Update local to match exchange
                    pos_dict["size"] = ex_pos["size"]

        # Check for missing local positions
        for (symbol, side), ex_pos in exchange_map.items():
            if not any(
                p.get("symbol") == symbol and p.get("side") == side
                for p in portfolio.positions
            ):
                discrepancies.append(
                    f"Exchange position {symbol} {side} missing locally"
                )
                # Add exchange position to local state
                portfolio.positions.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "size": ex_pos["size"],
                        "entry_price": ex_pos["entry_price"],
                        "tp_price": 0.0,  # Unknown
                        "sl_price": 0.0,  # Unknown
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "position_id": f"recovered_{symbol}_{side}",
                    }
                )

        if discrepancies:
            self.save(portfolio)

        return discrepancies


# Convenience singleton
state_manager = StateManager()
