"""Execution agent: the only component that talks to a broker.

It acts on ``RiskApproval`` and nothing else. An intent without a live approval
token is refused (constraint 3), and every order carries our own UUID as the broker
tag so a retry or a reconnect can never double-fill (constraint 4).

What it does with an approved intent:

- **prices it from urgency** - PASSIVE joins the touch, NORMAL sits at the mid,
  AGGRESSIVE crosses the spread with a *marketable limit*. Never a raw market
  order on options, where a thin book can fill absurdly far from the mid.
- **slices it** when the quantity is large against recent volume or over the
  exchange freeze limit; children go out one at a time.
- **chases** an unfilled child a few ticks at a time, bounded by ``max_chase_steps``
  and ``max_chase_bps``, then cancels at TTL.
- **brackets the fill** - a stop-loss (and optional target) goes out as soon as a
  child fills, not on the next tick.
- **rate-limits** placements to stay inside the broker's orders-per-second cap.
- **reconciles** against the broker's order book on start and after a reconnect.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trading.agents.base import Agent
from trading.backtest.costs import compute_fees
from trading.brokers.base import Broker, BrokerError, Instrument, OrderRouter
from trading.brokers.lots import LotSizes
from trading.brokers.symbols import (
    InstrumentKind,
    contract_multiplier,
    parse_symbol,
)
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock
from trading.core.types import (
    AlertLevel,
    Bar,
    ControlCommand,
    Fill,
    Order,
    OrderIntent,
    OrderRequest,
    OrderStatus,
    OrderType,
    ProductType,
    RiskApproval,
    Side,
    Tick,
    Urgency,
    Validity,
    new_tag,
)


@dataclass
class ExecutionConfig:
    max_participation: float = 0.10  # child order as a fraction of recent volume
    volume_lookback_bars: int = 5
    max_chase_steps: int = 3
    max_chase_bps: float = 15.0
    chase_interval_seconds: float = 5.0
    orders_per_second: float | None = 8.0  # None: no throttling (simulated time)
    default_tick_size: float = 0.05
    place_bracket_stop: bool = True
    allow_market_orders: bool = False  # never for options, regardless
    poll_interval_seconds: float = 1.0


class OrderRateLimiter:
    """Simple token bucket so bursts of children stay inside the broker's cap."""

    def __init__(self, per_second: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_second = per_second
        self._clock = clock
        self._allowance = per_second
        self._last = clock()
        self.waits = 0

    async def acquire(self) -> None:
        while True:
            now = self._clock()
            self._allowance = min(
                self.per_second, self._allowance + (now - self._last) * self.per_second
            )
            self._last = now
            if self._allowance >= 1.0:
                self._allowance -= 1.0
                return
            self.waits += 1
            await asyncio.sleep((1.0 - self._allowance) / self.per_second)


@dataclass
class WorkingOrder:
    """One child order we have in the market, and how to manage it."""

    order: Order
    intent: OrderIntent
    approval_token: str
    deadline: datetime
    chase_steps: int = 0
    protective: bool = False

    @property
    def id(self) -> str:
        return self.order.id


@dataclass
class ExecutionPlan:
    intent: OrderIntent
    approval: RiskApproval
    slices: list[int]
    placed: int = 0
    filled_qty: int = 0
    orders: list[str] = field(default_factory=list)
    # child orders that have already released their successor; a terminal order
    # can reach us twice (our own call plus the broker's update stream) and must
    # not advance the plan twice
    advanced: set[str] = field(default_factory=set)

    @property
    def remaining_slices(self) -> list[int]:
        return self.slices[self.placed :]

    @property
    def done(self) -> bool:
        return self.placed >= len(self.slices)


class ExecutionAgent(Agent):
    name = "execution"

    def __init__(
        self,
        bus: MessageBus,
        broker: Broker | OrderRouter,
        config: ExecutionConfig | None = None,
        *,
        instruments: dict[str, Instrument] | None = None,
        lots: LotSizes | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(bus, clock=clock)
        self.broker = broker
        self.cfg = config or ExecutionConfig()
        self.instruments = instruments or {}
        self.lots = lots or LotSizes.from_instruments(self.instruments)
        # the broker's orders-per-second cap is wall-clock: it means nothing in a
        # replay, where it would only put real sleeps into simulated time
        self.limiter = (
            OrderRateLimiter(self.cfg.orders_per_second) if self.cfg.orders_per_second else None
        )
        self._intents: dict[str, OrderIntent] = {}
        self._used_tokens: set[str] = set()
        self._plans: dict[str, ExecutionPlan] = {}  # intent id -> plan
        self._working: dict[str, WorkingOrder] = {}  # order id -> working order
        self._quotes: dict[str, Tick] = {}
        self._volume: dict[str, list[int]] = {}
        self._orders_seen: dict[str, Order] = {}
        self._real_fill_orders: set[str] = set()
        self._net: dict[tuple[str, ProductType], int] = {}
        self._protective: dict[tuple[str, ProductType], list[str]] = {}
        self.halted = False
        self.placed_count = 0
        self.rejected_without_approval = 0
        self.fills_published = 0

    # ------------------------------------------------------------------ lifecycle
    async def on_start(self) -> None:
        await self.subscribe(Topics.INTENTS, self._on_intent)
        await self.subscribe(Topics.APPROVED, self._on_approval)
        await self.subscribe(Topics.TICKS_ALL, self._on_tick)
        await self.subscribe(Topics.BARS_ALL, self._on_bar)
        await self.subscribe(Topics.CONTROL, self._on_control)

    async def _on_intent(self, _topic: str, intent: OrderIntent) -> None:  # type: ignore[override]
        self._intents[intent.id] = intent

    async def _on_tick(self, _topic: str, tick: Tick) -> None:  # type: ignore[override]
        self._quotes[tick.symbol] = tick

    async def _on_bar(self, _topic: str, bar: Bar) -> None:  # type: ignore[override]
        vols = self._volume.setdefault(bar.symbol, [])
        vols.append(bar.volume)
        if len(vols) > self.cfg.volume_lookback_bars:
            del vols[0]

    async def _on_control(self, _topic: str, cmd: ControlCommand) -> None:  # type: ignore[override]
        command = cmd.command.upper()
        if command == "KILL":
            self.halted = True
            await self.cancel_all("kill switch")
        elif command == "RESUME":
            self.halted = False
        elif command == "FLATTEN":
            await self.cancel_all("flatten requested")

    # ------------------------------------------------------------------ approval gate
    async def _on_approval(self, _topic: str, approval: RiskApproval) -> None:  # type: ignore[override]
        problem = self.check_approval(approval)
        if problem is not None:
            self.rejected_without_approval += 1
            await self.alert(
                AlertLevel.ERROR, f"approval refused: {problem}", intent=approval.intent_id
            )
            return
        self._used_tokens.add(approval.token)
        await self.execute(approval.intent, approval)

    def check_approval(self, approval: RiskApproval) -> str | None:
        """Why this approval cannot be acted on, or None if it is good.

        The approved intent travels inside the approval, so this works even when
        the approval arrives before (or without) the intent itself. When we *have*
        seen the intent, it must be the same one - a mismatch means something
        tampered with the message.
        """
        if self.halted:
            return "execution is halted"
        if approval.token in self._used_tokens:
            return f"token {approval.token} already used"
        if approval.expires_at <= self.clock.now():
            return f"approval expired at {approval.expires_at:%H:%M:%S}"
        if approval.approved_qty <= 0:
            return "approved quantity is zero"
        known = self._intents.get(approval.intent_id)
        if known is not None and known != approval.intent:
            return f"approval for {approval.intent_id} does not match the intent we saw"
        if approval.approved_qty > approval.intent.qty:
            return "approved quantity exceeds the intent"
        return None

    async def submit_unapproved(self, intent: OrderIntent) -> None:
        """Explicitly refused: nothing reaches a broker without the risk agent."""
        raise PermissionError(
            f"intent {intent.id} has no risk approval; execution never places unapproved orders"
        )

    # ------------------------------------------------------------------ execution
    async def execute(self, intent: OrderIntent, approval: RiskApproval) -> ExecutionPlan:
        slices = self.plan_slices(intent.symbol, approval.approved_qty)
        plan = ExecutionPlan(intent=intent, approval=approval, slices=slices)
        self._plans[intent.id] = plan
        await self._place_next_slice(plan)
        return plan

    def plan_slices(self, symbol: str, qty: int) -> list[int]:
        """Split ``qty`` into child orders under the participation and freeze caps."""
        inst = self.instruments.get(symbol)
        lot = self.lots.get(symbol)
        caps = []
        vols = self._volume.get(symbol, [])
        if vols and self.cfg.max_participation > 0:
            recent = sum(vols) / len(vols)
            by_volume = int(recent * self.cfg.max_participation)
            if by_volume > 0:
                caps.append(by_volume)
        if inst is not None and inst.freeze_qty:
            caps.append(inst.freeze_qty)
        cap = min(caps) if caps else qty
        cap = max((cap // lot) * lot, lot)
        if cap >= qty:
            return [qty]
        out = []
        left = qty
        while left > 0:
            take = min(cap, left)
            out.append(take)
            left -= take
        return out

    async def _place_next_slice(self, plan: ExecutionPlan) -> Order | None:
        if plan.done or self.halted:
            return None
        qty = plan.slices[plan.placed]
        plan.placed += 1
        intent = plan.intent
        price, order_type = self.price_order(intent)
        req = OrderRequest(
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            order_type=order_type,
            product=intent.product,
            price=price,
            validity=Validity.DAY,
            tag=new_tag(),
        )
        order = await self._place(req, intent, plan.approval.token)
        if order is not None:
            plan.orders.append(order.id)
        return order

    async def _place(
        self,
        req: OrderRequest,
        intent: OrderIntent,
        token: str,
        *,
        protective: bool = False,
    ) -> Order | None:
        if self.limiter is not None:
            await self.limiter.acquire()
        try:
            order = await self.broker.place_order(req)
        except BrokerError as e:
            await self.alert(AlertLevel.ERROR, f"place_order failed: {e}", symbol=req.symbol)
            return None
        order.intent_id = intent.id
        order.approval_token = token
        if protective:
            order.meta["protective"] = True
        self.placed_count += 1
        self._orders_seen[order.id] = order
        if order.status.is_working:
            self._working[order.id] = WorkingOrder(
                order=order,
                intent=intent,
                approval_token=token,
                deadline=self.clock.now() + timedelta(seconds=intent.ttl_seconds),
                protective=protective,
            )
        await self.publish(Topics.ORDERS, order)
        if order.status is OrderStatus.REJECTED:
            await self.alert(
                AlertLevel.WARN, f"broker rejected order: {order.status_message}", symbol=req.symbol
            )
        elif order.status is OrderStatus.FILLED:
            await self._on_order_progress(order)
        return order

    # ------------------------------------------------------------------ pricing
    def tick_size(self, symbol: str) -> float:
        inst = self.instruments.get(symbol)
        return inst.tick_size if inst and inst.tick_size > 0 else self.cfg.default_tick_size

    def round_to_tick(self, symbol: str, price: float, side: Side) -> float:
        """Round to a valid tick, conservatively: buys down, sells up.

        The quotient is rounded to 9 places first: ``2499.0 / 0.1`` is
        24989.999999999996 in binary floating point, and flooring that would
        silently move the price a whole tick away from the market.
        """
        tick = self.tick_size(symbol)
        steps = round(price / tick, 9)
        n = math.floor(steps) if side is Side.BUY else math.ceil(steps)
        return round(max(n, 1) * tick, 4)

    def quote(self, symbol: str) -> Tick | None:
        return self._quotes.get(symbol)

    def price_order(self, intent: OrderIntent) -> tuple[float | None, OrderType]:
        """Urgency -> (limit price, order type). Bounded by ``limit_band_bps``."""
        tick = self._quotes.get(intent.symbol)
        ref = intent.reference_price
        bid = tick.bid if tick and tick.bid else None
        ask = tick.ask if tick and tick.ask else None
        mid = tick.mid if tick else ref
        buying = intent.side is Side.BUY
        parsed = parse_symbol(intent.symbol)

        if intent.urgency is Urgency.AGGRESSIVE:
            if self.cfg.allow_market_orders and parsed.kind is not InstrumentKind.OPTION:
                return None, OrderType.MARKET
            touch = (ask or mid) if buying else (bid or mid)
            band = intent.limit_band_bps / 10_000
            price = touch * (1 + band) if buying else touch * (1 - band)
        elif intent.urgency is Urgency.PASSIVE:
            price = (bid or mid) if buying else (ask or mid)
        else:  # NORMAL
            price = mid
        price = self._clamp_to_band(price, ref, intent)
        return self.round_to_tick(intent.symbol, price, intent.side), OrderType.LIMIT

    def _clamp_to_band(self, price: float, ref: float, intent: OrderIntent) -> float:
        """Never pay more (or sell lower) than the intent's band allows."""
        band = intent.limit_band_bps / 10_000
        if intent.side is Side.BUY:
            return min(price, ref * (1 + band))
        return max(price, ref * (1 - band))

    # ------------------------------------------------------------------ order updates
    async def pump_updates(self, stream: AsyncIterator[Order | Fill] | None = None) -> None:
        """Consume the broker's order/fill stream until the agent stops."""
        updates = stream if stream is not None else self.broker.order_updates()
        if hasattr(updates, "__aiter__"):
            async for update in updates:  # type: ignore[union-attr]
                await self.handle_update(update)
                if not self.running:
                    break

    async def handle_update(self, update: Order | Fill) -> None:
        if isinstance(update, Fill):
            self._real_fill_orders.add(update.order_id)
            await self._publish_fill(update)
            order = self._orders_seen.get(update.order_id)
            if order is not None:
                await self._maybe_bracket(order, update)
            return
        await self._on_order_update(update)

    async def _on_order_update(self, order: Order) -> None:
        previous = self._orders_seen.get(order.id)
        known = self._working.get(order.id)
        if known is not None:
            order.intent_id = order.intent_id or known.intent.id
            order.approval_token = order.approval_token or known.approval_token
            known.order = order
        self._orders_seen[order.id] = order
        await self.publish(Topics.ORDERS, order)
        if order.id not in self._real_fill_orders:
            delta = order.filled_qty - (previous.filled_qty if previous else 0)
            if delta > 0:
                await self._publish_fill(self._synthesise_fill(order, delta, previous))
        await self._on_order_progress(order)

    def _synthesise_fill(self, order: Order, delta: int, previous: Order | None) -> Fill:
        """Brokers that only stream order state (Dhan) get a fill derived from the delta."""
        price = order.avg_fill_price or order.price or 0.0
        if previous and previous.avg_fill_price and order.avg_fill_price:
            prior_value = previous.avg_fill_price * previous.filled_qty
            price = (order.avg_fill_price * order.filled_qty - prior_value) / delta
        inst = self.instruments.get(order.symbol)
        mult = inst.multiplier if inst else contract_multiplier(order.symbol)
        price = max(price, 1e-6)
        return Fill(
            order_id=order.id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=delta,
            price=price,
            ts=order.updated_at,
            product=order.product,
            multiplier=mult,
            # brokers that stream only order state give no charges; estimate them
            fees=compute_fees(
                order.symbol, order.side, delta, price, order.product, multiplier=mult
            ),
        )

    async def _publish_fill(self, fill: Fill) -> None:
        self.fills_published += 1
        await self.publish(Topics.FILLS, fill)
        key = (fill.symbol, fill.product)
        self._net[key] = self._net.get(key, 0) + fill.side.sign * fill.qty
        if self._net[key] == 0:
            await self._cancel_protective(key, exclude=fill.order_id)

    async def _cancel_protective(self, key: tuple[str, ProductType], *, exclude: str = "") -> int:
        """One cancels the other: a flat position has nothing left to protect.

        Without this a bracket stop outlives the position it guarded and, when it
        later triggers, opens a brand new position in the opposite direction - a
        long-only strategy would quietly end the day short.
        """
        cancelled = 0
        for order_id in list(self._protective.get(key, [])):
            self._protective[key].remove(order_id)
            if order_id == exclude or order_id not in self._working:
                continue
            await self.cancel(order_id, "position closed")
            cancelled += 1
        return cancelled

    async def _on_order_progress(self, order: Order) -> None:
        """Advance the plan when a child finishes; bracket it when it fills."""
        working = self._working.get(order.id)
        if order.status.is_terminal:
            self._working.pop(order.id, None)
        if working and working.protective:
            return
        intent_id = order.intent_id or (working.intent.id if working else None)
        plan = self._plans.get(intent_id) if intent_id else None
        if plan is None:
            return
        if not order.status.is_terminal or order.id in plan.advanced:
            return
        plan.advanced.add(order.id)
        plan.filled_qty = sum(
            o.filled_qty for oid in plan.orders if (o := self._orders_seen.get(oid))
        )
        if not plan.done:
            await self._place_next_slice(plan)

    def _intent_for(self, order: Order) -> OrderIntent | None:
        """The approved intent behind an order.

        The plan is authoritative: it holds the intent that came inside the
        approval, so this works even for an order that filled on placement and
        never entered the working set, and when the intent itself was never seen
        on the bus.
        """
        if not order.intent_id:
            return None
        plan = self._plans.get(order.intent_id)
        if plan is not None:
            return plan.intent
        return self._intents.get(order.intent_id)

    async def _maybe_bracket(self, order: Order, fill: Fill) -> None:
        if not self.cfg.place_bracket_stop:
            return
        working = self._working.get(order.id)
        if working and working.protective:
            return
        intent = working.intent if working else self._intent_for(order)
        if intent is None:
            return
        stop_pct = intent.meta.get("stop_loss_pct")
        if not stop_pct:
            return
        side = intent.side.opposite
        stop = (
            fill.price * (1 - stop_pct) if intent.side is Side.BUY else fill.price * (1 + stop_pct)
        )
        trigger = self.round_to_tick(intent.symbol, stop, side)
        limit = self.round_to_tick(
            intent.symbol, trigger * (0.998 if side is Side.SELL else 1.002), side
        )
        req = OrderRequest(
            symbol=intent.symbol,
            side=side,
            qty=fill.qty,
            order_type=OrderType.SL,
            product=intent.product,
            price=limit,
            trigger_price=trigger,
            tag=new_tag(),
        )
        placed = await self._place(
            req, intent, working.approval_token if working else "", protective=True
        )
        if placed is not None:
            placed.parent_id = order.id
            self._protective.setdefault((intent.symbol, intent.product), []).append(placed.id)
            self.log.info("bracket stop for %s at %.2f", order.id, trigger)

    # ------------------------------------------------------------------ chase & TTL
    async def manage(self, now: datetime | None = None) -> None:
        """Chase unfilled children and cancel at TTL. Call on a timer."""
        now = now or self.clock.now()
        for working in list(self._working.values()):
            if working.protective:
                continue
            if now >= working.deadline:
                await self.cancel(working.id, "TTL expired")
                continue
            await self._maybe_chase(working, now)

    async def _maybe_chase(self, working: WorkingOrder, now: datetime) -> None:
        order = working.order
        if order.order_type is not OrderType.LIMIT or not order.status.is_working:
            return
        if working.chase_steps >= self.cfg.max_chase_steps:
            return
        age = (now - order.updated_at).total_seconds()
        if age < self.cfg.chase_interval_seconds:
            return
        intent = working.intent
        tick = self.tick_size(intent.symbol)
        current = order.price or intent.reference_price
        step = tick if intent.side is Side.BUY else -tick
        new_price = current + step
        limit = intent.reference_price * (
            1 + self.cfg.max_chase_bps / 10_000
            if intent.side is Side.BUY
            else 1 - self.cfg.max_chase_bps / 10_000
        )
        if (intent.side is Side.BUY and new_price > limit) or (
            intent.side is Side.SELL and new_price < limit
        ):
            return
        try:
            updated = await self.broker.modify_order(order.id, price=round(new_price, 4))
        except BrokerError as e:
            await self.alert(AlertLevel.WARN, f"chase modify failed: {e}", order=order.id)
            return
        working.chase_steps += 1
        working.order = updated
        self._orders_seen[updated.id] = updated
        await self.publish(Topics.ORDERS, updated)
        await self._on_order_progress(updated)

    async def cancel(self, order_id: str, reason: str) -> None:
        try:
            order = await self.broker.cancel_order(order_id)
        except BrokerError as e:
            await self.alert(AlertLevel.WARN, f"cancel failed: {e}", order=order_id)
            return
        self._working.pop(order_id, None)
        self._orders_seen[order.id] = order
        self.log.info("cancelled %s: %s", order_id, reason)
        await self.publish(Topics.ORDERS, order)
        await self._on_order_progress(order)

    async def cancel_all(self, reason: str) -> int:
        return await self.cancel_where(lambda _o: True, reason)

    async def cancel_where(self, predicate: Callable[[Order], bool], reason: str) -> int:
        n = 0
        for order_id, working in list(self._working.items()):
            if predicate(working.order):
                await self.cancel(order_id, reason)
                n += 1
        return n

    # ------------------------------------------------------------------ reconciliation
    async def reconcile(self) -> dict[str, int]:
        """Match our view against the broker's book (constraint 4).

        Called at startup and after any reconnect, *before* anything new is placed.
        """
        try:
            broker_orders = await self.broker.orders()
        except BrokerError as e:
            await self.alert(AlertLevel.ERROR, f"reconciliation failed: {e}")
            return {"error": 1}
        stats = {"broker_orders": len(broker_orders), "adopted": 0, "closed": 0, "unknown": 0}
        seen = set()
        for order in broker_orders:
            seen.add(order.id)
            previous = self._orders_seen.get(order.id)
            self._orders_seen[order.id] = order
            if previous is None:
                stats["adopted"] += 1
                if order.status.is_working:
                    stats["unknown"] += 1
                    await self.alert(
                        AlertLevel.WARN,
                        f"working order {order.id} at the broker is not ours; leaving it alone",
                        symbol=order.symbol,
                    )
            if not order.status.is_working:
                self._working.pop(order.id, None)
            await self.publish(Topics.ORDERS, order)
        for order_id in list(self._working):
            if order_id not in seen:
                stats["closed"] += 1
                self._working.pop(order_id, None)
                await self.alert(
                    AlertLevel.WARN,
                    f"order {order_id} vanished from the broker book",
                    order=order_id,
                )
        self.log.info("reconciled: %s", stats)
        return stats

    # ------------------------------------------------------------------ introspection
    def strategy_of_orders(self) -> dict[str, str]:
        """Order id -> strategy id, for every order we can trace to an intent."""
        out: dict[str, str] = {}
        for order_id, order in self._orders_seen.items():
            intent = self._intent_for(order)
            if intent is not None:
                out[order_id] = intent.strategy_id
        return out

    def working_orders(self) -> list[Order]:
        return [w.order for w in self._working.values()]

    def plan_for(self, intent_id: str) -> ExecutionPlan | None:
        return self._plans.get(intent_id)
