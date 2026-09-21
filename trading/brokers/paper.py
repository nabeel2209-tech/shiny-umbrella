"""Paper broker: fake cash and fills, real quotes.

Wraps any ``MarketData`` source for history/feed/LTP and simulates the order
side locally:

- MARKET / SLM: fill at last price +/- ``slippage_bps`` (buy pays up).
- LIMIT / SL: at placement, a marketable limit fills at last +/- slippage capped
  at the limit. A resting limit fills at its limit price once the market crosses
  it (``low <= limit`` for buys); if a bar gaps through the limit it fills at the
  bar open. ``require_trade_through`` demands a strict cross (used by the sim
  broker in backtests).
- SL / SLM wait for the trigger, then behave as LIMIT / MARKET.
- Fees from ``backtest/costs.py`` are debited on every fill.
- Cash accounting: equities and options move cash by full value; futures move
  cash only by realised PnL and block an approximate margin.
- State (cash, orders, fills, positions) persists to SQLite via ``PaperStore``.

The engine drives matching either through ``subscribe_live`` (ticks are matched
then passed through) or through ``on_bar`` during archive replay.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from trading.backtest.costs import DEFAULT_FEES, FeeSchedule, compute_fees
from trading.brokers.base import Instrument, MarketData, NotConnected, UnknownOrder
from trading.brokers.symbols import InstrumentKind, parse_symbol
from trading.core.clock import Clock, SystemClock
from trading.core.types import (
    Bar,
    Fill,
    Funds,
    Interval,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Side,
    Tick,
)

if TYPE_CHECKING:
    from trading.brokers.paper_store import PaperState

log = logging.getLogger(__name__)


class PaperStoreLike(Protocol):
    """What the broker needs from a store; ``paper_store.PaperStore`` implements it."""

    def load(self, account_id: str) -> PaperState | None: ...

    def save_account(self, account_id: str, starting_cash: float, cash: float) -> None: ...

    def upsert_order(self, account_id: str, order: Order) -> None: ...

    def add_fill(self, account_id: str, fill: Fill) -> None: ...

    def upsert_position(self, account_id: str, pos: Position) -> None: ...


@dataclass
class PaperConfig:
    starting_cash: float = 1_000_000.0
    slippage_bps: float = 2.0
    fee_schedule: FeeSchedule = field(default_factory=lambda: DEFAULT_FEES)
    futures_margin_pct: float = 0.10  # rough SPAN+exposure stand-in
    short_option_margin_pct: float = 0.15  # of strike notional, rough
    require_trade_through: bool = False
    account_id: str = "default"
    # contract multiplier lookup (MCX lots -> contract value); default 1 for everything
    multiplier_for: Callable[[str], float] | None = None
    # False: nothing fills at the moment it is placed - it waits for the next price.
    # The backtest simulator uses this so an order decided on a bar's close cannot
    # also be filled at that same close.
    fill_at_placement: bool = True
    # cap each fill at this fraction of the bar's volume (None: no cap)
    max_participation: float | None = None
    lot_size_for: Callable[[str], int] | None = None


@dataclass(frozen=True)
class PriceCtx:
    """Price information available when trying to fill an order."""

    ts: datetime
    last: float
    open: float
    high: float
    low: float
    volume: int | None = None  # bar volume; None for ticks

    @classmethod
    def from_tick(cls, t: Tick) -> PriceCtx:
        return cls(ts=t.ts, last=t.ltp, open=t.ltp, high=t.ltp, low=t.ltp)

    @classmethod
    def from_bar(cls, b: Bar) -> PriceCtx:
        return cls(ts=b.ts, last=b.close, open=b.open, high=b.high, low=b.low, volume=b.volume)

    @classmethod
    def at(cls, ts: datetime, price: float) -> PriceCtx:
        return cls(ts=ts, last=price, open=price, high=price, low=price)


def apply_fill_to_position(pos: Position, side: Side, qty: int, price: float) -> float:
    """Update ``pos`` in place with average-cost accounting; return realised PnL
    (gross of fees, scaled by the position's contract multiplier) from this fill."""
    signed = side.sign * qty
    if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
        new_qty = pos.qty + signed
        pos.avg_price = (pos.avg_price * abs(pos.qty) + price * qty) / abs(new_qty)
        pos.qty = new_qty
        return 0.0
    close_qty = min(abs(pos.qty), qty)
    direction = 1 if pos.qty > 0 else -1
    realised = (price - pos.avg_price) * close_qty * direction * pos.multiplier
    new_qty = pos.qty + signed
    if new_qty == 0:
        pos.avg_price = 0.0
    elif (new_qty > 0) != (pos.qty > 0):  # flipped through zero
        pos.avg_price = price
    pos.qty = new_qty
    pos.realised_pnl += realised
    return realised


class PaperBroker:
    name = "paper"

    def __init__(
        self,
        market_data: MarketData | None = None,
        *,
        config: PaperConfig | None = None,
        store: PaperStoreLike | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.md = market_data
        self.cfg = config or PaperConfig()
        self.store = store
        self.clock = clock or SystemClock()
        self.cash = self.cfg.starting_cash
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[tuple[str, ProductType], Position] = {}
        self._last: dict[str, float] = {}
        self._update_queues: list[asyncio.Queue[Order | Fill]] = []
        self._seq = 0
        if self.store is not None:
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        assert self.store is not None
        state = self.store.load(self.cfg.account_id)
        if state is None:
            self.store.save_account(self.cfg.account_id, self.cfg.starting_cash, self.cash)
            return
        self.cash = state.cash
        self._orders = {o.id: o for o in state.orders}
        self._fills = list(state.fills)
        self._positions = {(p.symbol, p.product): p for p in state.positions}
        self._seq = len(self._orders)
        log.info(
            "paper account %s restored: cash=%.2f orders=%d positions=%d",
            self.cfg.account_id,
            self.cash,
            len(self._orders),
            len(self._positions),
        )

    def _persist_order(self, order: Order) -> None:
        if self.store is not None:
            self.store.upsert_order(self.cfg.account_id, order)

    def _persist_fill(self, fill: Fill, pos: Position) -> None:
        if self.store is not None:
            self.store.add_fill(self.cfg.account_id, fill)
            self.store.upsert_position(self.cfg.account_id, pos)
            self.store.save_account(self.cfg.account_id, self.cfg.starting_cash, self.cash)

    # ------------------------------------------------------------------ market data pass-through
    async def connect(self) -> None:
        if self.md is not None:
            await self.md.connect()

    async def close(self) -> None:
        if self.md is not None:
            await self.md.close()

    async def instruments(self) -> list[Instrument]:
        if self.md is None:
            return []
        return await self.md.instruments()

    async def historical(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Bar]:
        if self.md is None:
            raise NotConnected("paper broker has no market data source")
        return await self.md.historical(symbol, interval, start, end)

    async def ltp(self, symbols: Sequence[str]) -> dict[str, float]:
        out = {s: self._last[s] for s in symbols if s in self._last}
        missing = [s for s in symbols if s not in out]
        if missing and self.md is not None:
            out.update(await self.md.ltp(missing))
        return out

    async def subscribe_live(self, symbols: Sequence[str]) -> AsyncIterator[Tick]:
        if self.md is None:
            raise NotConnected("paper broker has no market data source")
        async for tick in self.md.subscribe_live(symbols):
            self.on_tick(tick)
            yield tick

    # ------------------------------------------------------------------ price events
    def on_tick(self, tick: Tick) -> None:
        self._mark(tick.symbol, tick.ltp)
        self._match_all(tick.symbol, PriceCtx.from_tick(tick))

    def on_bar(self, bar: Bar) -> None:
        self._match_all(bar.symbol, PriceCtx.from_bar(bar))
        self._mark(bar.symbol, bar.close)

    def _mark(self, symbol: str, price: float) -> None:
        self._last[symbol] = price
        for (sym, _), pos in self._positions.items():
            if sym == symbol:
                pos.last_price = price

    def _match_all(self, symbol: str, px: PriceCtx) -> None:
        for order in list(self._orders.values()):
            if order.symbol == symbol and order.status.is_working:
                self._try_fill(order, px, at_placement=False)

    # ------------------------------------------------------------------ orders
    async def place_order(self, req: OrderRequest) -> Order:
        if req.tag in self._orders:  # idempotent: same tag -> same order
            return self._orders[req.tag]
        now = self.clock.now()
        self._seq += 1
        order = Order.from_request(req, now, broker_order_id=f"P{self._seq:06d}")
        order.status = (
            OrderStatus.TRIGGER_PENDING
            if req.order_type in {OrderType.SL, OrderType.SLM}
            else OrderStatus.OPEN
        )
        problem = self._validate(order)
        if problem:
            order.status = OrderStatus.REJECTED
            order.status_message = problem
            self._orders[order.id] = order
            self._persist_order(order)
            self._emit(order)
            return order
        self._orders[order.id] = order
        last = self._last.get(order.symbol)
        if last is not None:
            if self.cfg.fill_at_placement:
                self._try_fill(order, PriceCtx.at(now, last), at_placement=True)
            else:
                self._tag_marketable(order, last)
        self._persist_order(order)
        self._emit(order)
        return order

    @staticmethod
    def _tag_marketable(order: Order, last: float) -> None:
        """Remember whether a limit crossed the market when placed.

        An order that crosses is taking liquidity, so it pays slippage when it
        fills at the next open. One that merely rests gets hit at its own price.
        """
        if order.order_type is not OrderType.LIMIT or order.price is None:
            return
        crosses = order.price > last if order.side is Side.BUY else order.price < last
        order.meta["marketable"] = crosses

    def _validate(self, order: Order) -> str | None:
        parsed = parse_symbol(order.symbol)
        if parsed.kind is InstrumentKind.INDEX:
            return "indices are not tradable"
        if parsed.kind is InstrumentKind.EQUITY and order.product is ProductType.NRML:
            return "equities use MIS or CNC"
        if parsed.is_derivative and order.product is ProductType.CNC:
            return "derivatives use MIS or NRML"
        ref = order.price or self._last.get(order.symbol)
        if ref is None:
            return None  # cannot value it yet; checked again at fill time
        if order.side is Side.BUY and parsed.kind is not InstrumentKind.FUTURE:
            needed = order.qty * ref * self._multiplier(order.symbol)
            if needed > self.cash:
                return f"insufficient funds: need {needed:.2f}, have {self.cash:.2f}"
        if (
            order.side is Side.SELL
            and parsed.kind is InstrumentKind.EQUITY
            and order.product is ProductType.CNC
        ):
            held = self._positions.get((order.symbol, ProductType.CNC))
            held_qty = held.qty if held else 0
            open_sells = sum(
                o.remaining_qty
                for o in self._orders.values()
                if o.symbol == order.symbol
                and o.product is ProductType.CNC
                and o.side is Side.SELL
                and o.status.is_working
            )
            if order.qty + open_sells > held_qty:
                return f"CNC sell exceeds holdings ({held_qty})"
        return None

    async def modify_order(
        self,
        order_id: str,
        *,
        price: float | None = None,
        trigger_price: float | None = None,
        qty: int | None = None,
        order_type: OrderType | None = None,
    ) -> Order:
        order = self._get(order_id)
        if not order.status.is_working:
            raise UnknownOrder(f"order {order_id} is {order.status}, cannot modify")
        if price is not None:
            order.price = price
        if trigger_price is not None:
            order.trigger_price = trigger_price
        if qty is not None:
            if qty < order.filled_qty:
                raise ValueError("qty below filled quantity")
            order.qty = qty
        if order_type is not None:
            order.order_type = order_type
        order.updated_at = self.clock.now()
        last = self._last.get(order.symbol)
        if last is not None:
            if self.cfg.fill_at_placement:
                self._try_fill(order, PriceCtx.at(order.updated_at, last), at_placement=True)
            else:
                self._tag_marketable(order, last)
        self._persist_order(order)
        self._emit(order)
        return order

    async def cancel_order(self, order_id: str) -> Order:
        order = self._get(order_id)
        if order.status.is_working:
            order.status = OrderStatus.CANCELLED
            order.updated_at = self.clock.now()
            self._persist_order(order)
            self._emit(order)
        return order

    async def order_status(self, order_id: str) -> Order:
        return self._get(order_id)

    async def orders(self) -> list[Order]:
        return list(self._orders.values())

    async def positions(self) -> list[Position]:
        return list(self._positions.values())

    async def fills(self) -> list[Fill]:
        return list(self._fills)

    async def funds(self) -> Funds:
        return Funds(
            cash=self.cash,
            margin_used=self._margin_used(),
            realised_pnl=sum(p.realised_pnl - p.fees_paid for p in self._positions.values()),
            unrealised_pnl=sum(p.unrealised_pnl for p in self._positions.values()),
            positions_value=self.positions_value(),
        )

    def positions_value(self) -> float:
        """What open positions add to cash: market value for equities and options
        (bought with cash), unrealised PnL for futures (margined, not paid for)."""
        total = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            if parse_symbol(pos.symbol).kind is InstrumentKind.FUTURE:
                total += pos.unrealised_pnl
            else:
                total += pos.market_value
        return total

    def equity(self) -> float:
        return self.cash + self.positions_value()

    async def liquidate(
        self,
        *,
        product: ProductType | None = None,
        symbols: Sequence[str] | None = None,
        reason: str = "liquidation",
    ) -> list[Order]:
        """Close open positions at the last price, as a broker's auto square-off would.

        Fills immediately at last +/- slippage with full fees, whatever
        ``fill_at_placement`` says: it models the broker closing us out, not an
        order of ours waiting in the book.
        """
        now = self.clock.now()
        closed: list[Order] = []
        for pos in list(self._positions.values()):
            if pos.qty == 0:
                continue
            if product is not None and pos.product is not product:
                continue
            if symbols is not None and pos.symbol not in symbols:
                continue
            last = self._last.get(pos.symbol, pos.last_price or pos.avg_price)
            req = OrderRequest(
                symbol=pos.symbol,
                side=Side.SELL if pos.qty > 0 else Side.BUY,
                qty=abs(pos.qty),
                order_type=OrderType.MARKET,
                product=pos.product,
            )
            self._seq += 1
            order = Order.from_request(
                req, now, broker_order_id=f"P{self._seq:06d}", status=OrderStatus.OPEN
            )
            order.meta["liquidation"] = reason
            self._orders[order.id] = order
            self._try_fill(order, PriceCtx.at(now, last), at_placement=True)
            self._persist_order(order)
            self._emit(order)
            closed.append(order)
        return closed

    def update_queue(self) -> asyncio.Queue[Order | Fill]:
        """Register and return a queue fed with order and fill updates.

        ``order_updates()`` wraps this for streaming consumers; a replay driver
        uses the queue directly so it can drain updates between bars and stay
        deterministic.
        """
        q: asyncio.Queue[Order | Fill] = asyncio.Queue()
        self._update_queues.append(q)
        return q

    def release_queue(self, q: asyncio.Queue[Order | Fill]) -> None:
        if q in self._update_queues:
            self._update_queues.remove(q)

    async def order_updates(self) -> AsyncIterator[Order | Fill]:
        q = self.update_queue()
        try:
            while True:
                yield await q.get()
        finally:
            self.release_queue(q)

    def _get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as e:
            raise UnknownOrder(order_id) from e

    def _emit(self, item: Order | Fill) -> None:
        payload = item.model_copy(deep=True) if isinstance(item, Order) else item
        for q in self._update_queues:
            q.put_nowait(payload)

    # ------------------------------------------------------------------ matching
    def _slippage_bps(self, order: Order, qty: int, px: PriceCtx) -> float:
        """Slippage for this fill in basis points; the backtest simulator overrides it."""
        return self.cfg.slippage_bps

    def _slip(self, order: Order, price: float, px: PriceCtx, qty: int) -> float:
        bps = self._slippage_bps(order, qty, px)
        return price * (1 + order.side.sign * bps / 10_000)

    def _lot(self, symbol: str) -> int:
        return max(1, int(self.cfg.lot_size_for(symbol))) if self.cfg.lot_size_for else 1

    def _fillable_qty(self, order: Order, px: PriceCtx) -> int:
        """How much of the order this price can fill (volume participation cap)."""
        qty = order.remaining_qty
        if self.cfg.max_participation is None or px.volume is None:
            return qty
        lot = self._lot(order.symbol)
        cap = int(px.volume * self.cfg.max_participation) // lot * lot
        return min(qty, cap)

    def _multiplier(self, symbol: str) -> float:
        if self.cfg.multiplier_for is None:
            return 1.0
        return float(self.cfg.multiplier_for(symbol))

    def _try_fill(self, order: Order, px: PriceCtx, *, at_placement: bool) -> None:
        if not order.status.is_working:
            return
        buying = order.side is Side.BUY
        if order.status is OrderStatus.TRIGGER_PENDING:
            self._try_trigger(order, px, at_placement=at_placement)
            return

        qty = self._fillable_qty(order, px)
        if qty <= 0:
            return
        if order.order_type in {OrderType.MARKET, OrderType.SLM}:
            ref = px.last if at_placement else px.open
            self._fill(order, qty, self._slip(order, ref, px, qty), px.ts)
            return

        limit = order.price
        assert limit is not None
        strict = self.cfg.require_trade_through
        if buying:
            crossed = px.low < limit if strict else px.low <= limit
            if not crossed:
                return
            if at_placement:
                price = min(limit, self._slip(order, px.last, px, qty))
            elif px.open < limit:
                # through the limit at the open: a crossing order pays to take
                # liquidity, a resting one is simply hit at the opening price
                taking = order.meta.get("marketable", False)
                price = min(limit, self._slip(order, px.open, px, qty)) if taking else px.open
            else:
                price = limit  # traded down through it during the bar
        else:
            crossed = px.high > limit if strict else px.high >= limit
            if not crossed:
                return
            if at_placement:
                price = max(limit, self._slip(order, px.last, px, qty))
            elif px.open > limit:
                taking = order.meta.get("marketable", False)
                price = max(limit, self._slip(order, px.open, px, qty)) if taking else px.open
            else:
                price = limit
        self._fill(order, qty, price, px.ts)

    def _try_trigger(self, order: Order, px: PriceCtx, *, at_placement: bool) -> None:
        """A stop order: has the market reached the trigger, and where?

        If the price gapped through the trigger the stop meets the market at the
        gap price, not at the trigger - filling a gapped stop at its trigger is the
        classic way a backtest flatters a stop-loss.
        """
        buying = order.side is Side.BUY
        trig = order.trigger_price or 0.0
        hit = px.high >= trig if buying else px.low <= trig
        if not hit:
            return
        order.status = OrderStatus.OPEN
        order.updated_at = px.ts
        if at_placement:
            touch = px.last
        else:
            gapped = px.open >= trig if buying else px.open <= trig
            touch = px.open if gapped else trig
        qty = self._fillable_qty(order, px)
        if qty <= 0:
            return
        slipped = self._slip(order, touch, px, qty)
        if order.order_type is OrderType.SLM:
            self._fill(order, qty, slipped, px.ts)
            return
        # stop-limit: live as a limit from the touch price; fills only if reachable
        limit = order.price
        assert limit is not None
        if buying and touch <= limit:
            self._fill(order, qty, min(limit, slipped), px.ts)
        elif not buying and touch >= limit:
            self._fill(order, qty, max(limit, slipped), px.ts)
        # otherwise it now rests as an ordinary limit order

    def _fill(self, order: Order, qty: int, price: float, ts: datetime) -> None:
        price = round(price, 4)
        parsed = parse_symbol(order.symbol)
        mult = self._multiplier(order.symbol)
        notional = qty * price * mult
        # final affordability check for orders that were valued only now
        if (
            order.side is Side.BUY
            and parsed.kind is not InstrumentKind.FUTURE
            and notional > self.cash
        ):
            order.status = OrderStatus.REJECTED
            order.status_message = f"insufficient funds at fill: need {notional:.2f}"
            order.updated_at = ts
            self._persist_order(order)
            self._emit(order)
            return
        fees = compute_fees(
            order.symbol,
            order.side,
            qty,
            price,
            order.product,
            schedule=self.cfg.fee_schedule,
            multiplier=mult,
        )
        fill = Fill(
            order_id=order.id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            ts=ts,
            product=order.product,
            fees=fees,
            multiplier=mult,
        )
        # order state
        prev_value = (order.avg_fill_price or 0.0) * order.filled_qty
        order.filled_qty += qty
        order.avg_fill_price = (prev_value + qty * price) / order.filled_qty
        order.status = OrderStatus.FILLED if order.filled_qty >= order.qty else OrderStatus.PARTIAL
        order.updated_at = ts
        # position & cash
        key = (order.symbol, order.product)
        pos = self._positions.get(key)
        if pos is None:
            pos = Position(symbol=order.symbol, product=order.product, multiplier=mult)
            self._positions[key] = pos
        realised = apply_fill_to_position(pos, order.side, qty, price)
        pos.fees_paid += fees.total
        pos.last_price = price
        if parsed.kind is InstrumentKind.FUTURE:
            self.cash += realised
        else:
            self.cash -= order.side.sign * notional
        self.cash -= fees.total
        self._fills.append(fill)
        self._persist_fill(fill, pos)
        self._persist_order(order)
        self._emit(fill)
        self._emit(order)

    def _margin_used(self) -> float:
        total = 0.0
        for pos in self._positions.values():
            if pos.qty == 0:
                continue
            parsed = parse_symbol(pos.symbol)
            price = pos.last_price if pos.last_price is not None else pos.avg_price
            if parsed.kind is InstrumentKind.FUTURE:
                total += abs(pos.qty) * price * pos.multiplier * self.cfg.futures_margin_pct
            elif parsed.kind is InstrumentKind.OPTION and pos.qty < 0:
                strike = parsed.strike or price
                total += abs(pos.qty) * strike * pos.multiplier * self.cfg.short_option_margin_pct
        return total
