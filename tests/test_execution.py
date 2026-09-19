"""Execution agent: the approval gate, pricing, slicing, chase, TTL, brackets,
reconciliation and idempotency."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from trading.agents.execution import (
    ExecutionAgent,
    ExecutionConfig,
    OrderRateLimiter,
)
from trading.brokers.base import Broker, Instrument
from trading.brokers.paper import PaperBroker, PaperConfig
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Bar,
    Exchange,
    Fill,
    InstrumentKind,
    Interval,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    ProductType,
    RiskApproval,
    Side,
    Tick,
    Urgency,
)

SYM = "NSE:RELIANCE"
NIFTY = "NFO:NIFTY-OCT26"
TS = datetime(2026, 9, 18, 10, 0, tzinfo=IST)

RELIANCE = Instrument(
    symbol=SYM,
    exchange=Exchange.NSE,
    kind=InstrumentKind.EQUITY,
    broker_id="2885",
    broker_segment="NSE_EQ",
    lot_size=1,
    tick_size=0.10,
)
NIFTY_FUT = Instrument(
    symbol=NIFTY,
    exchange=Exchange.NFO,
    kind=InstrumentKind.FUTURE,
    broker_id="48704",
    broker_segment="NSE_FNO",
    lot_size=65,
    tick_size=0.05,
    freeze_qty=1756,
)


def intent(**kw) -> OrderIntent:
    base = dict(
        ts=TS,
        strategy_id="s1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        product=ProductType.MIS,
        reference_price=2500.0,
        urgency=Urgency.NORMAL,
        limit_band_bps=10.0,
        ttl_seconds=60,
    )
    return OrderIntent(**{**base, **kw})


def approval(
    i: OrderIntent, *, qty: int | None = None, ttl: int = 120, ts: datetime = TS
) -> RiskApproval:
    return RiskApproval(
        intent_id=i.id,
        intent=i,
        ts=ts,
        expires_at=ts + timedelta(seconds=ttl),
        approved_qty=qty or i.qty,
    )


async def build(*, cfg=None, instruments=None, cash=10_000_000.0, clock=None):
    bus = InMemoryBus()
    clock = clock or SimClock(TS)
    broker = PaperBroker(
        config=PaperConfig(
            starting_cash=cash, slippage_bps=0.0, multiplier_for=contract_multiplier
        ),
        clock=clock,
    )
    agent = ExecutionAgent(
        bus,
        broker,
        cfg or ExecutionConfig(),
        instruments=instruments if instruments is not None else {SYM: RELIANCE},
        clock=clock,
    )
    await agent.start()
    queue = broker.update_queue()

    async def drain() -> None:
        while not queue.empty():
            await agent.handle_update(queue.get_nowait())

    return bus, agent, broker, clock, drain


async def quote(bus, *, bid=2499.0, ask=2501.0, ltp=2500.0, symbol=SYM):
    await bus.publish(
        Topics.ticks(symbol),
        Tick(symbol=symbol, ts=TS, ltp=ltp, bid=bid, ask=ask, bid_qty=10, ask_qty=10),
    )


# --------------------------------------------------------------------------- the approval gate


async def test_an_intent_alone_places_nothing():
    """Constraint 3: publishing an intent must never reach the broker."""
    bus, agent, broker, _, _ = await build()
    await bus.publish(Topics.INTENTS, intent())
    assert await broker.orders() == []
    assert agent.placed_count == 0


async def test_unapproved_submission_is_refused_outright():
    _, agent, broker, _, _ = await build()
    with pytest.raises(PermissionError, match="no risk approval"):
        await agent.submit_unapproved(intent())
    assert await broker.orders() == []


async def test_approval_places_the_order():
    bus, _agent, broker, _, _ = await build()
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.APPROVED, approval(i))
    orders = await broker.orders()
    assert len(orders) == 1 and orders[0].qty == 10
    assert orders[0].intent_id == i.id and orders[0].approval_token


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda a: a.model_copy(update={"approved_qty": 999}), "exceeds the intent"),
        (lambda a: a.model_copy(update={"expires_at": TS - timedelta(seconds=1)}), "expired"),
    ],
)
async def test_bad_approvals_are_refused(mutate, fragment):
    bus, agent, broker, _, _ = await build()
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.APPROVED, mutate(approval(i)))
    assert await broker.orders() == []
    assert agent.rejected_without_approval == 1


async def test_a_token_cannot_be_replayed():
    bus, agent, broker, _, _ = await build()
    i = intent()
    a = approval(i)
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.APPROVED, a)
    await bus.publish(Topics.APPROVED, a)
    assert len(await broker.orders()) == 1
    assert agent.rejected_without_approval == 1


async def test_an_approval_that_contradicts_the_intent_we_saw_is_refused():
    bus, agent, broker, _, _ = await build()
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    tampered = approval(i.model_copy(update={"qty": 10, "side": Side.SELL}))
    await bus.publish(Topics.APPROVED, tampered.model_copy(update={"intent_id": i.id}))
    assert await broker.orders() == []
    assert agent.rejected_without_approval == 1


async def test_an_approval_without_a_prior_intent_still_works():
    """Ordering is not guaranteed on Redis, so the approval carries the intent."""
    bus, _agent, broker, _, _ = await build()
    i = intent()
    await bus.publish(Topics.APPROVED, approval(i))
    assert len(await broker.orders()) == 1


async def test_halted_execution_places_nothing():
    bus, agent, broker, _, _ = await build()
    agent.halted = True
    i = intent()
    await bus.publish(Topics.APPROVED, approval(i))
    assert await broker.orders() == []


# --------------------------------------------------------------------------- pricing


async def test_urgency_maps_to_price():
    bus, agent, _, _, _ = await build()
    await quote(bus)
    passive, kind = agent.price_order(intent(urgency=Urgency.PASSIVE))
    assert kind is OrderType.LIMIT and passive == 2499.0  # joins the bid
    normal, _ = agent.price_order(intent(urgency=Urgency.NORMAL))
    assert normal == 2500.0  # the mid
    aggressive, _ = agent.price_order(intent(urgency=Urgency.AGGRESSIVE))
    assert aggressive == pytest.approx(2502.5, abs=0.1)  # crosses the ask, still a limit
    sell, _ = agent.price_order(intent(side=Side.SELL, urgency=Urgency.PASSIVE))
    assert sell == 2501.0  # joins the ask


async def test_aggressive_is_bounded_by_the_intents_band():
    bus, agent, _, _, _ = await build()
    await quote(bus, bid=2600.0, ask=2700.0)  # the market has run away
    price, kind = agent.price_order(intent(urgency=Urgency.AGGRESSIVE, limit_band_bps=10))
    assert kind is OrderType.LIMIT
    assert price == pytest.approx(2502.5, abs=0.05)  # never past reference + 10 bps


async def test_options_never_get_a_market_order():
    opt = "NFO:NIFTY-OCT26-25000-CE"
    cfg = ExecutionConfig(allow_market_orders=True)
    _bus, agent, _, _, _ = await build(cfg=cfg, instruments={})
    _, kind = agent.price_order(
        intent(symbol=opt, urgency=Urgency.AGGRESSIVE, reference_price=100.0)
    )
    assert kind is OrderType.LIMIT
    _, equity_kind = agent.price_order(intent(urgency=Urgency.AGGRESSIVE))
    assert equity_kind is OrderType.MARKET  # allowed here, because it is not an option


async def test_prices_are_rounded_to_the_tick_conservatively():
    _, agent, _, _, _ = await build()
    assert agent.round_to_tick(SYM, 2500.06, Side.BUY) == 2500.0
    assert agent.round_to_tick(SYM, 2500.06, Side.SELL) == 2500.1
    assert agent.tick_size("NSE:UNKNOWN") == 0.05  # default when we have no instrument


async def test_pricing_without_a_quote_falls_back_to_the_reference():
    _, agent, _, _, _ = await build()
    price, _ = agent.price_order(intent(urgency=Urgency.NORMAL))
    assert price == 2500.0


# --------------------------------------------------------------------------- slicing


async def test_slicing_by_participation_and_freeze_limit():
    cfg = ExecutionConfig(max_participation=0.1, volume_lookback_bars=2)
    bus, agent, _, _, _ = await build(cfg=cfg, instruments={SYM: RELIANCE, NIFTY: NIFTY_FUT})
    assert agent.plan_slices(SYM, 100) == [100]  # no volume history yet
    for volume in (1000, 2000):
        await bus.publish(
            Topics.bars(SYM),
            Bar(
                symbol=SYM,
                ts=TS,
                interval=Interval.M1,
                open=1,
                high=1,
                low=1,
                close=1,
                volume=volume,
            ),
        )
    assert agent.plan_slices(SYM, 100) == [100]  # 10% of 1500 = 150, one child is fine
    assert agent.plan_slices(SYM, 400) == [150, 150, 100]
    # the exchange freeze limit binds for index futures
    assert agent.plan_slices(NIFTY, 4000) == [1755, 1755, 490]  # lot-aligned under 1756


async def test_slices_go_out_one_at_a_time():
    cfg = ExecutionConfig(max_participation=0.1, volume_lookback_bars=1)
    bus, agent, broker, _, drain = await build(cfg=cfg)
    await bus.publish(
        Topics.bars(SYM),
        Bar(symbol=SYM, ts=TS, interval=Interval.M1, open=1, high=1, low=1, close=1, volume=100),
    )
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))  # our bid will rest
    i = intent(qty=30, urgency=Urgency.PASSIVE, ttl_seconds=600)
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.APPROVED, approval(i, ttl=600))
    plan = agent.plan_for(i.id)
    assert plan.slices == [10, 10, 10] and plan.placed == 1
    assert len(await broker.orders()) == 1  # only the first child is in the market
    # when a child finishes, the next one goes out
    await agent.cancel(agent.working_orders()[0].id, "test")
    await drain()
    assert plan.placed == 2 and len(await broker.orders()) == 2


# --------------------------------------------------------------------------- fills, brackets


async def test_fill_is_published_and_a_bracket_stop_follows():
    bus, _agent, broker, _, drain = await build()
    fills: list[Fill] = []

    async def on_fill(_t, m):
        fills.append(m)

    await bus.subscribe(Topics.FILLS, on_fill)
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2500.0))
    i = intent(urgency=Urgency.AGGRESSIVE, meta={"stop_loss_pct": 0.01})
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.APPROVED, approval(i))
    await drain()
    assert len(fills) == 1 and fills[0].qty == 10
    stops = [o for o in await broker.orders() if o.order_type is OrderType.SL]
    assert len(stops) == 1
    stop = stops[0]
    assert stop.side is Side.SELL and stop.qty == 10
    assert stop.trigger_price == pytest.approx(2475.0, abs=0.2)  # 1% below the fill
    assert stop.price < stop.trigger_price  # the limit sits under the trigger


async def test_no_bracket_without_a_stop_percentage():
    bus, _agent, broker, _, drain = await build()
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2500.0))
    i = intent(urgency=Urgency.AGGRESSIVE)
    await bus.publish(Topics.APPROVED, approval(i))
    await drain()
    assert [o for o in await broker.orders() if o.order_type is OrderType.SL] == []


async def test_a_fill_derived_from_order_state_gets_estimated_fees():
    """Dhan streams order state, not fills, so the agent synthesises them."""
    bus, agent, _, _, _ = await build()
    fills: list[Fill] = []

    async def on_fill(_t, m):
        fills.append(m)

    await bus.subscribe(Topics.FILLS, on_fill)
    order = Order(
        id="x1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        filled_qty=4,
        avg_fill_price=2500.0,
        order_type=OrderType.LIMIT,
        product=ProductType.MIS,
        status=OrderStatus.PARTIAL,
        created_at=TS,
        updated_at=TS,
    )
    await agent.handle_update(order)
    assert len(fills) == 1 and fills[0].qty == 4 and fills[0].fees.total > 0
    # the rest of the order fills at a different price: only the delta is emitted
    await agent.handle_update(
        order.model_copy(
            update={"filled_qty": 10, "avg_fill_price": 2502.0, "status": OrderStatus.FILLED}
        )
    )
    assert len(fills) == 2 and fills[1].qty == 6
    assert fills[1].price == pytest.approx((2502.0 * 10 - 2500.0 * 4) / 6)


async def test_real_fills_are_not_duplicated_by_synthesis():
    bus, agent, _, _, _ = await build()
    fills: list[Fill] = []

    async def on_fill(_t, m):
        fills.append(m)

    await bus.subscribe(Topics.FILLS, on_fill)
    real = Fill(
        order_id="x1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        price=2500.0,
        ts=TS,
        product=ProductType.MIS,
    )
    await agent.handle_update(real)
    await agent.handle_update(
        Order(
            id="x1",
            symbol=SYM,
            side=Side.BUY,
            qty=10,
            filled_qty=10,
            avg_fill_price=2500.0,
            order_type=OrderType.LIMIT,
            product=ProductType.MIS,
            status=OrderStatus.FILLED,
            created_at=TS,
            updated_at=TS,
        )
    )
    assert len(fills) == 1


# --------------------------------------------------------------------------- chase and TTL


async def test_unfilled_order_is_chased_then_stops_at_the_limit():
    cfg = ExecutionConfig(chase_interval_seconds=5, max_chase_steps=2, max_chase_bps=15)
    clock = SimClock(TS)
    bus, agent, broker, _, _ = await build(cfg=cfg, clock=clock)
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))  # market above our bid
    i = intent(urgency=Urgency.PASSIVE, ttl_seconds=600)
    await bus.publish(Topics.APPROVED, approval(i, ttl=600))
    order = agent.working_orders()[0]
    start = order.price
    for step in range(1, 4):
        clock.set(TS + timedelta(seconds=10 * step))
        await agent.manage()
    working = agent.working_orders()[0]
    assert working.price == pytest.approx(start + 2 * RELIANCE.tick_size)  # two steps, then stops
    assert agent._working[working.id].chase_steps == 2


async def test_chase_will_not_cross_the_max_slippage():
    cfg = ExecutionConfig(chase_interval_seconds=1, max_chase_steps=50, max_chase_bps=1.0)
    clock = SimClock(TS)
    bus, agent, broker, _, _ = await build(cfg=cfg, clock=clock)
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))
    i = intent(urgency=Urgency.PASSIVE, ttl_seconds=600)
    await bus.publish(Topics.APPROVED, approval(i, ttl=600))
    for step in range(1, 12):
        clock.set(TS + timedelta(seconds=5 * step))
        await agent.manage()
    price = agent.working_orders()[0].price
    assert price <= 2500.0 * (1 + 1.0 / 10_000) + 1e-9


async def test_ttl_cancels_what_did_not_fill():
    clock = SimClock(TS)
    bus, agent, broker, _, _ = await build(clock=clock)
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))
    i = intent(urgency=Urgency.PASSIVE, ttl_seconds=30)
    await bus.publish(Topics.APPROVED, approval(i))
    assert len(agent.working_orders()) == 1
    clock.set(TS + timedelta(seconds=31))
    await agent.manage()
    assert agent.working_orders() == []
    assert (await broker.orders())[0].status is OrderStatus.CANCELLED


async def test_kill_switch_cancels_everything_working():
    from trading.core.types import ControlCommand

    bus, agent, broker, _, _ = await build()
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))
    i = intent(urgency=Urgency.PASSIVE, ttl_seconds=600)
    await bus.publish(Topics.APPROVED, approval(i, ttl=600))
    assert len(agent.working_orders()) == 1
    await bus.publish(Topics.CONTROL, ControlCommand(ts=TS, command="KILL", reason="stop"))
    assert agent.halted and agent.working_orders() == []
    assert (await broker.orders())[0].status is OrderStatus.CANCELLED


# --------------------------------------------------------------------------- reconciliation


async def test_reconcile_adopts_broker_orders_and_drops_ghosts():
    bus, agent, broker, _, _ = await build()
    alerts = []

    async def on_alert(_t, m):
        alerts.append(m)

    await bus.subscribe(Topics.ALERTS, on_alert)
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2600.0))
    i = intent(urgency=Urgency.PASSIVE, ttl_seconds=600)
    await bus.publish(Topics.APPROVED, approval(i, ttl=600))
    ours = agent.working_orders()[0]

    stats = await agent.reconcile()
    assert stats["broker_orders"] == 1 and stats["adopted"] == 0
    # someone places an order in the Dhan app while we are running
    from trading.core.types import OrderRequest, new_tag

    await broker.place_order(
        OrderRequest(
            symbol=SYM,
            side=Side.BUY,
            qty=1,
            order_type=OrderType.LIMIT,
            product=ProductType.MIS,
            price=2000.0,
            tag=new_tag(),
        )
    )
    stats = await agent.reconcile()
    assert stats["adopted"] == 1 and stats["unknown"] == 1
    assert any("not ours" in a.message for a in alerts)
    # an order that disappears from the broker book is noticed
    await broker.cancel_order(ours.id)
    broker._orders.pop(ours.id)
    agent._working[ours.id] = agent._working.get(ours.id) or None  # keep our stale view
    stats = await agent.reconcile()
    assert stats["closed"] >= 1
    assert any("vanished" in a.message for a in alerts)


async def test_orders_are_idempotent_by_tag():
    """Constraint 4: our UUID is the broker tag, so a repeat never double-fills."""
    from trading.core.types import OrderRequest

    _bus, _agent, broker, _, _ = await build()
    req = OrderRequest(
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        product=ProductType.MIS,
        price=2000.0,
        tag="fixed-tag-123",
    )
    first = await broker.place_order(req)
    second = await broker.place_order(req)
    assert first.id == second.id == "fixed-tag-123"
    assert len(await broker.orders()) == 1


async def test_every_child_order_gets_its_own_tag():
    cfg = ExecutionConfig(max_participation=0.1, volume_lookback_bars=1)
    bus, _agent, broker, _, drain = await build(cfg=cfg)
    await bus.publish(
        Topics.bars(SYM),
        Bar(symbol=SYM, ts=TS, interval=Interval.M1, open=1, high=1, low=1, close=1, volume=100),
    )
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2400.0))
    i = intent(qty=30, urgency=Urgency.AGGRESSIVE)
    await bus.publish(Topics.APPROVED, approval(i))
    for _ in range(3):
        await drain()
    ids = [o.id for o in await broker.orders()]
    assert len(ids) == len(set(ids)) == 3


# --------------------------------------------------------------------------- rate limiter


async def test_rate_limiter_paces_bursts():
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = OrderRateLimiter(2.0, clock=lambda: now[0])
    original = asyncio.sleep
    asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        for _ in range(4):
            await limiter.acquire()
    finally:
        asyncio.sleep = original  # type: ignore[assignment]
    assert limiter.waits == 2 and all(s > 0 for s in sleeps)


def test_protocols_are_satisfied():
    broker = PaperBroker(config=PaperConfig())
    assert isinstance(broker, Broker)


async def test_bracket_stop_is_cancelled_once_the_position_is_flat():
    """One cancels the other.

    A stop that outlives the position it guarded would trigger later and open a
    brand new position the other way - a long-only strategy would end up short.
    """
    bus, _agent, broker, _, drain = await build()
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2500.0))
    entry = intent(urgency=Urgency.AGGRESSIVE, meta={"stop_loss_pct": 0.01})
    await bus.publish(Topics.APPROVED, approval(entry))
    await drain()
    stops = [o for o in await broker.orders() if o.order_type is OrderType.SL]
    assert len(stops) == 1 and stops[0].status is OrderStatus.TRIGGER_PENDING
    assert stops[0].meta["protective"] is True

    # the strategy exits on its own terms before the stop is ever hit
    exit_intent = intent(side=Side.SELL, urgency=Urgency.AGGRESSIVE)
    await bus.publish(Topics.APPROVED, approval(exit_intent))
    await drain()
    stop = await broker.order_status(stops[0].id)
    assert stop.status is OrderStatus.CANCELLED

    # the market later trades through the old stop level: nothing happens
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2400.0))
    await drain()
    positions = [p for p in await broker.positions() if p.qty]
    assert positions == []


async def test_a_triggered_stop_does_not_cancel_itself():
    bus, _agent, broker, _, drain = await build()
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2500.0))
    await bus.publish(
        Topics.APPROVED, approval(intent(urgency=Urgency.AGGRESSIVE, meta={"stop_loss_pct": 0.01}))
    )
    await drain()
    # between the trigger (~2475) and the stop-limit (~2470): it triggers and fills
    broker.on_tick(Tick(symbol=SYM, ts=TS, ltp=2473.0))
    await drain()
    stop = next(o for o in await broker.orders() if o.order_type is OrderType.SL)
    assert stop.status is OrderStatus.FILLED
    assert [p for p in await broker.positions() if p.qty] == []
