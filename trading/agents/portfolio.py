"""Shared position / PnL state, rebuilt from the fill stream.

The broker is the source of truth for cash and positions; this is the engine's own
running view, used by the risk agent (exposure, daily loss) and the monitor. It is
reconciled against the broker at startup and after any reconnect.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from trading.brokers.paper import apply_fill_to_position
from trading.core.types import Fill, Funds, Position, ProductType, Side

log = logging.getLogger(__name__)


@dataclass
class DaySnapshot:
    day: date
    start_equity: float
    realised_pnl: float = 0.0
    fees_paid: float = 0.0
    trades: int = 0


@dataclass
class Portfolio:
    starting_equity: float = 1_000_000.0
    multiplier_for: Callable[[str], float] | None = None
    positions: dict[tuple[str, ProductType], Position] = field(default_factory=dict)
    last_price: dict[str, float] = field(default_factory=dict)
    realised_pnl: float = 0.0
    fees_paid: float = 0.0
    day: DaySnapshot | None = None
    fills_seen: int = 0

    # ------------------------------------------------------------------ helpers
    def _multiplier(self, symbol: str) -> float:
        return float(self.multiplier_for(symbol)) if self.multiplier_for else 1.0

    def position(self, symbol: str, product: ProductType) -> Position:
        key = (symbol, product)
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(symbol=symbol, product=product, multiplier=self._multiplier(symbol))
            self.positions[key] = pos
        return pos

    def net_qty(self, symbol: str, product: ProductType | None = None) -> int:
        """Signed quantity; summed across product types when ``product`` is None."""
        return sum(
            p.qty
            for (sym, prod), p in self.positions.items()
            if sym == symbol and (product is None or prod is product)
        )

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.qty != 0]

    # ------------------------------------------------------------------ updates
    def roll_day(self, ts: datetime) -> None:
        """Start a new trading day, snapshotting the equity it opened with."""
        d = ts.date()
        if self.day is None or self.day.day != d:
            self.day = DaySnapshot(day=d, start_equity=self.equity)

    def mark(self, symbol: str, price: float, ts: datetime | None = None) -> None:
        if ts is not None:
            self.roll_day(ts)
        self.last_price[symbol] = price
        for (sym, _), pos in self.positions.items():
            if sym == symbol:
                pos.last_price = price

    def apply_fill(self, fill: Fill) -> None:
        self.roll_day(fill.ts)
        pos = self.position(fill.symbol, fill.product)
        pos.multiplier = fill.multiplier or pos.multiplier
        realised = apply_fill_to_position(pos, fill.side, fill.qty, fill.price)
        fees = fill.fees.total
        pos.fees_paid += fees
        pos.last_price = fill.price
        self.last_price[fill.symbol] = fill.price
        self.realised_pnl += realised
        self.fees_paid += fees
        self.fills_seen += 1
        assert self.day is not None
        self.day.realised_pnl += realised
        self.day.fees_paid += fees
        self.day.trades += 1

    def sync_from_broker(self, funds: Funds, positions: Sequence[Position]) -> None:
        """Adopt the broker's positions after a restart or reconnect (constraint 4)."""
        self.positions = {(p.symbol, p.product): p.model_copy(deep=True) for p in positions}
        for p in self.positions.values():
            if p.last_price:
                self.last_price[p.symbol] = p.last_price
        self.realised_pnl = funds.realised_pnl
        log.info(
            "portfolio synced from broker: %d positions, cash %.2f", len(self.positions), funds.cash
        )

    # ------------------------------------------------------------------ metrics
    @property
    def unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())

    @property
    def net_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl - self.fees_paid

    @property
    def equity(self) -> float:
        return self.starting_equity + self.net_pnl

    @property
    def day_pnl(self) -> float:
        """Realised + unrealised PnL since this trading day opened, net of fees."""
        if self.day is None:
            return 0.0
        return self.equity - self.day.start_equity

    def gross_exposure(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            if pos.qty == 0:
                continue
            price = self.last_price.get(pos.symbol, pos.avg_price)
            total += abs(pos.qty) * price * pos.multiplier
        return total

    def exposure(self, symbol: str, product: ProductType | None = None) -> float:
        price = self.last_price.get(symbol)
        qty = self.net_qty(symbol, product)
        if price is None:
            pos = next((p for (s, _), p in self.positions.items() if s == symbol), None)
            price = pos.avg_price if pos else 0.0
        return abs(qty) * price * self._multiplier(symbol)

    def side_of(self, symbol: str, product: ProductType | None = None) -> Side | None:
        qty = self.net_qty(symbol, product)
        if qty == 0:
            return None
        return Side.BUY if qty > 0 else Side.SELL
