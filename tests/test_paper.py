"""Paper broker tests, including the Phase 1 acceptance test."""

from __future__ import annotations

import asyncio

import pytest

from trading.backtest.costs import compute_fees
from trading.brokers.base import Broker, UnknownOrder
from trading.brokers.paper import PaperBroker, PaperConfig, apply_fill_to_position
from trading.brokers.paper_store import PaperStore
from trading.brokers.symbols import contract_multiplier
from trading.core.types import (
    Fill,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Side,
    Tick,
)

SYM = "NSE:RELIANCE"


def req(
    side,
    qty,
    order_type=OrderType.MARKET,
    price=None,
    trigger=None,
    product=ProductType.CNC,
    symbol=SYM,
    tag=None,
):
    kw = {}
    if tag:
        kw["tag"] = tag
    return OrderRequest(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        product=product,
        price=price,
        trigger_price=trigger,
        **kw,
    )


def cnc_fees(side: Side, qty: int, price: float) -> float:
    """Hand-computed equity delivery fees (Dhan defaults, Oct-2024 statutory rates)."""
    value = qty * price
    stt = value * 0.001
    exchange = value * 0.0000297
    sebi = value * 0.000001
    stamp = value * 0.00015 if side is Side.BUY else 0.0
    gst = 0.18 * (0 + exchange + sebi)
    dp = 14.75 if side is Side.SELL else 0.0
    return stt + exchange + sebi + stamp + gst + dp


def slip(price: float, side: Side, bps: float = 2.0) -> float:
    return round(price * (1 + side.sign * bps / 10_000), 4)


# --------------------------------------------------------------------------- acceptance


async def test_one_day_replay_buy_then_sell_matches_hand_computation(
    synthetic_day, sim_clock, tmp_path
):
    """Replay one NSE session of synthetic 1m bars through the paper broker,
    buy 10 shares with a resting limit, sell them at market, and check cash,
    position and PnL against an independent hand computation."""
    db_url = f"sqlite:///{tmp_path / 'paper.db'}"
    store = PaperStore(db_url)
    cfg = PaperConfig(starting_cash=1_000_000.0, slippage_bps=2.0)
    broker = PaperBroker(config=cfg, store=store, clock=sim_clock)
    assert isinstance(broker, Broker)

    bars = synthetic_day()
    assert len(bars) == 375

    buy_order = sell_order = None
    limit_px = None
    for i, bar in enumerate(bars):
        sim_clock.set(bar.ts)
        broker.on_bar(bar)
        if i == 10:
            limit_px = round(bar.close - 5, 2)
            buy_order = await broker.place_order(req(Side.BUY, 10, OrderType.LIMIT, price=limit_px))
            assert buy_order.status is OrderStatus.OPEN  # not marketable yet
        if i == 200:
            assert (await broker.order_status(buy_order.id)).status is OrderStatus.FILLED
            sell_order = await broker.place_order(req(Side.SELL, 10, OrderType.MARKET))
            assert sell_order.status is OrderStatus.FILLED

    fills = await broker.fills()
    assert [f.side for f in fills] == [Side.BUY, Side.SELL]
    buy_fill, sell_fill = fills

    # the limit was crossed within a bar whose open was above it -> filled at the limit
    assert buy_fill.price == limit_px
    crossing_bar = next(b for b in bars[11:] if b.low <= limit_px)  # after placement
    assert buy_fill.ts == crossing_bar.ts
    assert buy_fill.ts < bars[200].ts
    # market sell at the last close minus 2 bps slippage
    assert sell_fill.price == slip(bars[200].close, Side.SELL)
    assert sell_fill.ts == bars[200].ts

    buy_fees = cnc_fees(Side.BUY, 10, buy_fill.price)
    sell_fees = cnc_fees(Side.SELL, 10, sell_fill.price)
    assert buy_fill.fees.total == pytest.approx(buy_fees, abs=1e-3)
    assert sell_fill.fees.total == pytest.approx(sell_fees, abs=1e-3)

    expected_cash = 1_000_000.0 - 10 * buy_fill.price - buy_fees + 10 * sell_fill.price - sell_fees
    funds = await broker.funds()
    assert funds.cash == pytest.approx(expected_cash, abs=1e-3)
    assert funds.margin_used == 0.0

    positions = await broker.positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.qty == 0
    assert pos.realised_pnl == pytest.approx((sell_fill.price - buy_fill.price) * 10)
    assert pos.fees_paid == pytest.approx(buy_fees + sell_fees, abs=1e-3)
    assert pos.net_pnl == pytest.approx(expected_cash - 1_000_000.0, abs=1e-3)

    # state survives a restart
    reloaded = PaperBroker(config=cfg, store=PaperStore(db_url))
    assert reloaded.cash == pytest.approx(expected_cash, abs=1e-3)
    assert len(await reloaded.fills()) == 2
    assert {o.status for o in await reloaded.orders()} == {OrderStatus.FILLED}
    rpos = (await reloaded.positions())[0]
    assert rpos.qty == 0 and rpos.realised_pnl == pytest.approx(pos.realised_pnl)


# --------------------------------------------------------------------------- unit


def tick(price: float, ts):
    return Tick(symbol=SYM, ts=ts, ltp=price)


async def test_market_buy_fills_at_last_plus_slippage(sim_clock):
    b = PaperBroker(config=PaperConfig(slippage_bps=5.0), clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.BUY, 10))
    assert o.status is OrderStatus.FILLED
    assert o.avg_fill_price == pytest.approx(100.05)
    pos = (await b.positions())[0]
    assert pos.qty == 10 and pos.avg_price == pytest.approx(100.05)


async def test_market_order_without_price_waits_for_first_tick(sim_clock):
    b = PaperBroker(clock=sim_clock)
    o = await b.place_order(req(Side.BUY, 10))
    assert o.status is OrderStatus.OPEN
    b.on_tick(tick(50.0, sim_clock.now()))
    assert (await b.order_status(o.id)).status is OrderStatus.FILLED


async def test_marketable_limit_fills_immediately_capped_at_limit(sim_clock):
    b = PaperBroker(config=PaperConfig(slippage_bps=2.0), clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.BUY, 10, OrderType.LIMIT, price=100.01))
    assert o.status is OrderStatus.FILLED
    assert o.avg_fill_price == pytest.approx(100.01)  # slippage would exceed the limit
    o2 = await b.place_order(req(Side.SELL, 10, OrderType.LIMIT, price=99.0))
    assert o2.avg_fill_price == pytest.approx(99.98)  # 100 - 2 bps, better than the limit


async def test_resting_limit_waits_then_fills_and_cancel_works(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.BUY, 10, OrderType.LIMIT, price=98.0))
    assert o.status is OrderStatus.OPEN
    b.on_tick(tick(99.0, sim_clock.now()))
    assert (await b.order_status(o.id)).status is OrderStatus.OPEN
    cancelled = await b.cancel_order(o.id)
    assert cancelled.status is OrderStatus.CANCELLED
    b.on_tick(tick(97.0, sim_clock.now()))
    assert (await b.order_status(o.id)).status is OrderStatus.CANCELLED
    assert await b.positions() == []
    with pytest.raises(UnknownOrder):
        await b.cancel_order("nope")


async def test_stop_loss_triggers_then_fills(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    await b.place_order(req(Side.BUY, 10))
    sl = await b.place_order(req(Side.SELL, 10, OrderType.SL, price=97.5, trigger=98.0))
    assert sl.status is OrderStatus.TRIGGER_PENDING
    b.on_tick(tick(99.0, sim_clock.now()))
    assert (await b.order_status(sl.id)).status is OrderStatus.TRIGGER_PENDING
    b.on_tick(tick(97.9, sim_clock.now()))
    done = await b.order_status(sl.id)
    assert done.status is OrderStatus.FILLED
    assert done.avg_fill_price == pytest.approx(97.9 * (1 - 0.0002), abs=1e-3)
    assert (await b.positions())[0].qty == 0


async def test_stop_market_fills_at_trigger_with_slippage(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    await b.place_order(req(Side.BUY, 10))
    slm = await b.place_order(req(Side.SELL, 10, OrderType.SLM, trigger=98.0))
    b.on_tick(tick(97.0, sim_clock.now()))
    done = await b.order_status(slm.id)
    assert done.status is OrderStatus.FILLED
    assert done.avg_fill_price == pytest.approx(98.0 * (1 - 0.0002), abs=1e-3)


async def test_insufficient_funds_rejected(sim_clock):
    b = PaperBroker(config=PaperConfig(starting_cash=1000.0), clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.BUY, 11))
    assert o.status is OrderStatus.REJECTED
    assert "insufficient funds" in (o.status_message or "")
    assert await b.positions() == []


async def test_cnc_sell_needs_holdings_but_mis_short_allowed(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.SELL, 5, product=ProductType.CNC))
    assert o.status is OrderStatus.REJECTED
    o = await b.place_order(req(Side.SELL, 5, product=ProductType.MIS))
    assert o.status is OrderStatus.FILLED
    pos = (await b.positions())[0]
    assert pos.qty == -5 and pos.product is ProductType.MIS


async def test_product_rules(sim_clock):
    b = PaperBroker(clock=sim_clock)
    o = await b.place_order(req(Side.BUY, 1, product=ProductType.NRML))
    assert o.status is OrderStatus.REJECTED
    o = await b.place_order(req(Side.BUY, 1, symbol="MCX:GOLDM-OCT26", product=ProductType.CNC))
    assert o.status is OrderStatus.REJECTED
    o = await b.place_order(req(Side.BUY, 1, symbol="NSE:NIFTY50"))
    assert o.status is OrderStatus.REJECTED


async def test_idempotent_tag(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    r = req(Side.BUY, 10, tag="my-uuid")
    o1 = await b.place_order(r)
    o2 = await b.place_order(r)
    assert o1.id == o2.id == "my-uuid"
    assert len(await b.orders()) == 1
    assert (await b.positions())[0].qty == 10


async def test_futures_margin_and_realised_pnl_to_cash(sim_clock):
    sym = "MCX:GOLDM-OCT26"
    b = PaperBroker(config=PaperConfig(starting_cash=100_000.0, slippage_bps=0.0), clock=sim_clock)
    b.on_tick(Tick(symbol=sym, ts=sim_clock.now(), ltp=75_000.0))
    o = await b.place_order(req(Side.BUY, 1, symbol=sym, product=ProductType.NRML))
    assert o.status is OrderStatus.FILLED
    f1 = (await b.fills())[0]
    funds = await b.funds()
    assert funds.cash == pytest.approx(100_000.0 - f1.fees.total)  # no cash for notional
    assert funds.margin_used == pytest.approx(7_500.0)
    b.on_tick(Tick(symbol=sym, ts=sim_clock.now(), ltp=75_500.0))
    assert (await b.funds()).unrealised_pnl == pytest.approx(500.0)
    await b.place_order(req(Side.SELL, 1, symbol=sym, product=ProductType.NRML))
    f2 = (await b.fills())[1]
    funds = await b.funds()
    assert funds.cash == pytest.approx(100_000.0 + 500.0 - f1.fees.total - f2.fees.total)
    assert funds.margin_used == 0.0
    assert (await b.positions())[0].realised_pnl == pytest.approx(500.0)


async def test_modify_order(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    o = await b.place_order(req(Side.BUY, 10, OrderType.LIMIT, price=95.0))
    assert o.status is OrderStatus.OPEN
    o = await b.modify_order(o.id, price=100.0)
    assert o.status is OrderStatus.FILLED and o.price == 100.0
    with pytest.raises(UnknownOrder):
        await b.modify_order(o.id, price=101.0)


async def test_order_updates_stream(sim_clock):
    b = PaperBroker(clock=sim_clock)
    b.on_tick(tick(100.0, sim_clock.now()))
    events: list[Order | Fill] = []

    async def consume():
        async for ev in b.order_updates():
            events.append(ev)
            if len(events) == 2:
                break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await b.place_order(req(Side.BUY, 1))
    await asyncio.wait_for(task, timeout=1)
    assert isinstance(events[0], Fill)
    assert isinstance(events[1], Order) and events[1].status is OrderStatus.FILLED


async def test_ltp_cache_and_bar_gap_fill(sim_clock, synthetic_day):
    b = PaperBroker(clock=sim_clock)
    bars = synthetic_day()
    b.on_bar(bars[0])
    assert await b.ltp([SYM]) == {SYM: bars[0].close}
    # a resting buy limit above a bar's open fills at the (better) open
    o = await b.place_order(req(Side.BUY, 1, OrderType.LIMIT, price=bars[0].close - 100))
    c0 = bars[0].close
    gap = bars[1].model_copy(
        update={"open": c0 - 150, "low": c0 - 160, "close": c0 - 120, "high": c0 - 110}
    )
    b.on_bar(gap)
    assert (await b.order_status(o.id)).avg_fill_price == pytest.approx(gap.open)


def test_apply_fill_to_position_average_cost_and_flip():
    p = Position(symbol=SYM, product=ProductType.MIS)
    assert apply_fill_to_position(p, Side.BUY, 10, 100.0) == 0.0
    assert apply_fill_to_position(p, Side.BUY, 10, 110.0) == 0.0
    assert p.qty == 20 and p.avg_price == pytest.approx(105.0)
    assert apply_fill_to_position(p, Side.SELL, 5, 120.0) == pytest.approx(75.0)
    assert p.qty == 15 and p.avg_price == pytest.approx(105.0)
    # flip through zero: close 15 at 100 (-75), open 5 short at 100
    assert apply_fill_to_position(p, Side.SELL, 20, 100.0) == pytest.approx(-75.0)
    assert p.qty == -5 and p.avg_price == 100.0
    assert p.realised_pnl == pytest.approx(0.0)
    assert apply_fill_to_position(p, Side.BUY, 5, 90.0) == pytest.approx(50.0)
    assert p.qty == 0 and p.avg_price == 0.0


async def test_require_trade_through(sim_clock, synthetic_day):
    b = PaperBroker(config=PaperConfig(require_trade_through=True), clock=sim_clock)
    bars = synthetic_day()
    b.on_bar(bars[0])
    o = await b.place_order(req(Side.BUY, 1, OrderType.LIMIT, price=bars[1].low))
    b.on_bar(bars[1])  # touches the limit exactly -> not enough
    assert (await b.order_status(o.id)).status is OrderStatus.OPEN


async def test_mcx_contract_multiplier_scales_pnl_fees_and_margin(sim_clock):
    """1 lot of GOLDM (100 g quoted per 10 g -> multiplier 10): a Rs 100 move is Rs 1000."""
    sym = "MCX:GOLDM-OCT26"
    b = PaperBroker(
        config=PaperConfig(
            starting_cash=1_000_000.0, slippage_bps=0.0, multiplier_for=contract_multiplier
        ),
        clock=sim_clock,
    )
    b.on_tick(Tick(symbol=sym, ts=sim_clock.now(), ltp=150_000.0))
    await b.place_order(req(Side.BUY, 1, symbol=sym, product=ProductType.NRML))
    f1 = (await b.fills())[0]
    assert f1.multiplier == 10.0 and f1.value == 1_500_000.0
    assert f1.fees.total == pytest.approx(
        compute_fees(sym, Side.BUY, 1, 150_000.0, ProductType.NRML, multiplier=10).total
    )
    assert (await b.funds()).margin_used == pytest.approx(150_000.0)  # 10% of 15 lakh
    b.on_tick(Tick(symbol=sym, ts=sim_clock.now(), ltp=150_100.0))
    pos = (await b.positions())[0]
    assert pos.multiplier == 10.0 and pos.unrealised_pnl == pytest.approx(1000.0)
    await b.place_order(req(Side.SELL, 1, symbol=sym, product=ProductType.NRML))
    f2 = (await b.fills())[1]
    pos = (await b.positions())[0]
    assert pos.qty == 0 and pos.realised_pnl == pytest.approx(1000.0)
    assert (await b.funds()).cash == pytest.approx(
        1_000_000.0 + 1000.0 - f1.fees.total - f2.fees.total
    )
