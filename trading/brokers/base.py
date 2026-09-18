"""Broker adapter interface.

The engine only ever talks to these Protocols, so any broker (Dhan, Zerodha,
Fyers, paper, sim) plugs in without engine changes.

- ``MarketData``: history, live feed, instrument master, LTP.
- ``OrderRouter``: order lifecycle, positions, funds, streaming order updates.
- ``Broker``: both.

Adapters are responsible for constraint 6: every timestamp they emit is
tz-aware IST and every symbol they emit is canonical.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from trading.core.types import (
    Bar,
    Exchange,
    Fill,
    Funds,
    InstrumentKind,
    Interval,
    OptionType,
    Order,
    OrderRequest,
    OrderType,
    Position,
    Tick,
)


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str  # canonical
    exchange: Exchange
    kind: InstrumentKind
    broker_id: str  # security id at the broker
    name: str = ""
    lot_size: int = 1
    tick_size: float = 0.05
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None
    underlying: str | None = None


class BrokerError(Exception):
    """Base class for adapter errors."""


class NotConnected(BrokerError):
    pass


class OrderRejected(BrokerError):
    pass


class UnknownOrder(BrokerError):
    pass


class RateLimited(BrokerError):
    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@runtime_checkable
class MarketData(Protocol):
    name: str

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def instruments(self) -> list[Instrument]: ...

    async def historical(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Bar]: ...

    def subscribe_live(self, symbols: Sequence[str]) -> AsyncIterator[Tick]: ...

    async def ltp(self, symbols: Sequence[str]) -> dict[str, float]: ...


@runtime_checkable
class OrderRouter(Protocol):
    name: str

    async def place_order(self, req: OrderRequest) -> Order: ...

    async def modify_order(
        self,
        order_id: str,
        *,
        price: float | None = None,
        trigger_price: float | None = None,
        qty: int | None = None,
        order_type: OrderType | None = None,
    ) -> Order: ...

    async def cancel_order(self, order_id: str) -> Order: ...

    async def order_status(self, order_id: str) -> Order: ...

    async def orders(self) -> list[Order]: ...

    async def positions(self) -> list[Position]: ...

    async def funds(self) -> Funds: ...

    def order_updates(self) -> AsyncIterator[Order | Fill]: ...


@runtime_checkable
class Broker(MarketData, OrderRouter, Protocol):
    """A full broker: market data plus order routing."""
