"""The backtest simulator: no fill at the signal's close, trade-through limits,
gap-aware stops, slippage models and volume participation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading.backtest.sim_broker import FixedSlippage, SimBroker, SimConfig, VolumeSlippage
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Bar,
    Interval,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    ProductType,
    Side,
)

SYM = "NSE:RELIANCE"
T0 = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


def bar(minute: int, o: float, h: float, lo: float, c: float, volume: int = 100_000, symbol=SYM):
    return Bar(
        symbol=symbol,
        ts=T0 + timedelta(minutes=minute),
        interval=Interval.M1,
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=volume,
    )


def req(
    side=Side.BUY,
    qty=10,
    order_type=OrderType.MARKET,
    price=None,
    trigger=None,
    symbol=SYM,
    product=ProductType.MIS,
):
    return OrderRequest(
        symbol=symbol, side=side, qty=qty, order_type=order_type, product=product,
        price=price, trigger_price=trigger,
    )  # fmt: skip


def sim(**kw) -> SimBroker:
    cfg = SimConfig(**{"slippage": FixedSlippage(0.0), **kw})
    return SimBroker(cfg, clock=SimClock(T0))


def test_it_is_in_memory_and_strict():
    b = sim()
    assert b.store is None and b.name == "sim"
    assert b.cfg.require_trade_through and not b.cfg.fill_at_placement


async def test_nothing_fills_at_the_close_that_produced_the_signal():
    """The heart of the simulator: a decision taken on a bar's close cannot also
    trade at that close."""
    b = sim(slippage=FixedSlippage(2.0))
    b.on_bar(bar(0, 100, 101, 99, 100))
    market = await b.place_order(req())
    marketable = await b.place_order(req(order_type=OrderType.LIMIT, price=101.0))
    assert market.status is OrderStatus.OPEN and marketable.status is OrderStatus.OPEN
    assert await b.fills() == []
    b.on_bar(bar(1, 100.5, 102, 100, 101))  # the next bar
    fills = await b.fills()
    assert len(fills) == 2
    assert all(f.ts == T0 + timedelta(minutes=1) for f in fills)
    assert fills[0].price == pytest.approx(100.5 * 1.0002, abs=1e-4)  # next open + slippage
    assert fills[1].price == pytest.approx(100.5 * 1.0002, abs=1e-4)  # crossing limit pays too


async def test_a_crossing_limit_never_pays_past_its_limit():
    b = sim(slippage=FixedSlippage(50.0))
    b.on_bar(bar(0, 100, 101, 99, 100))
    o = await b.place_order(req(order_type=OrderType.LIMIT, price=100.2))
    b.on_bar(bar(1, 100.0, 101, 99.8, 100.5))
    assert (await b.order_status(o.id)).avg_fill_price == 100.2  # 50 bps would be 100.5


async def test_a_resting_limit_needs_a_trade_through_not_a_touch():
    b = sim()
    b.on_bar(bar(0, 100, 101, 99, 100))
    o = await b.place_order(req(order_type=OrderType.LIMIT, price=99.0))
    assert o.meta["marketable"] is False
    b.on_bar(bar(1, 100, 100.5, 99.0, 99.5))  # low touches 99.00
    assert (await b.order_status(o.id)).status is OrderStatus.OPEN
    b.on_bar(bar(2, 99.5, 99.8, 98.9, 99.2))  # trades through
    filled = await b.order_status(o.id)
    assert filled.status is OrderStatus.FILLED and filled.avg_fill_price == 99.0


async def test_a_resting_limit_gapped_through_fills_at_the_open_without_slippage():
    b = sim(slippage=FixedSlippage(10.0))
    b.on_bar(bar(0, 100, 101, 99, 100))
    o = await b.place_order(req(order_type=OrderType.LIMIT, price=99.0))
    b.on_bar(bar(1, 97.0, 98.0, 96.5, 97.5))
    assert (await b.order_status(o.id)).avg_fill_price == 97.0  # hit, not taking


async def test_sell_side_mirrors_buy_side():
    b = sim(slippage=FixedSlippage(2.0))
    b.on_bar(bar(0, 100, 101, 99, 100))
    await b.place_order(req(qty=20))
    b.on_bar(bar(1, 100, 101, 99, 100))
    resting = await b.place_order(req(Side.SELL, 10, OrderType.LIMIT, price=101.0))
    taking = await b.place_order(req(Side.SELL, 10, OrderType.LIMIT, price=99.0))
    b.on_bar(bar(2, 100.0, 101.0, 99.5, 100.5))  # high only touches 101
    assert (await b.order_status(resting.id)).status is OrderStatus.OPEN
    assert (await b.order_status(taking.id)).avg_fill_price == pytest.approx(
        100.0 * 0.9998, abs=1e-4
    )
    b.on_bar(bar(3, 102.0, 102.5, 101.5, 102.0))  # gaps above the resting sell
    assert (await b.order_status(resting.id)).avg_fill_price == 102.0


async def test_stops_fill_at_the_gap_on_bars():
    b = sim()
    b.on_bar(bar(0, 100, 101, 99, 100))
    await b.place_order(req(qty=10))
    b.on_bar(bar(1, 100, 101, 99, 100))
    stop = await b.place_order(req(Side.SELL, 10, OrderType.SLM, trigger=98.0))
    b.on_bar(bar(2, 95.0, 96.0, 94.0, 95.5))
    assert (await b.order_status(stop.id)).avg_fill_price == 95.0


async def test_volume_slippage_grows_with_participation():
    model = VolumeSlippage(base_bps=1.0, impact_bps=20.0)
    order = Order(
        symbol=SYM, side=Side.BUY, qty=1, order_type=OrderType.MARKET, product=ProductType.MIS,
        created_at=T0, updated_at=T0,
    )  # fmt: skip
    assert model.cost_bps(order, 100, 100.0, 10_000) == pytest.approx(1.0 + 20.0 * 0.1)
    assert model.cost_bps(order, 2_500, 100.0, 10_000) == pytest.approx(1.0 + 20.0 * 0.5)
    assert model.cost_bps(order, 50_000, 100.0, 10_000) == pytest.approx(21.0)  # capped at 100%
    assert model.cost_bps(order, 100, 100.0, 0) == pytest.approx(21.0)  # no volume: worst case
    assert "sqrt" in model.describe() and "fixed" in FixedSlippage(3).describe()

    b = sim(slippage=model)
    b.on_bar(bar(0, 100, 101, 99, 100, volume=10_000))
    small = await b.place_order(req(qty=100))
    big = await b.place_order(req(qty=2_500))
    b.on_bar(bar(1, 100, 101, 99, 100, volume=10_000))
    assert (await b.order_status(small.id)).avg_fill_price < (
        await b.order_status(big.id)
    ).avg_fill_price


async def test_participation_cap_spreads_a_big_order_over_bars():
    b = sim(max_participation=0.1, lot_size_for=lambda s: 25)
    b.on_bar(bar(0, 100, 101, 99, 100, volume=1_000))
    o = await b.place_order(req(qty=250))
    b.on_bar(bar(1, 100, 101, 99, 100, volume=1_000))  # 10% of 1000 = 100 = 4 lots
    first = await b.order_status(o.id)
    assert first.status is OrderStatus.PARTIAL and first.filled_qty == 100
    b.on_bar(bar(2, 100, 101, 99, 100, volume=290))  # 29 -> one lot
    assert (await b.order_status(o.id)).filled_qty == 125
    b.on_bar(bar(3, 100, 101, 99, 100, volume=20))  # less than a lot: nothing
    assert (await b.order_status(o.id)).filled_qty == 125
    b.on_bar(bar(4, 100, 101, 99, 100, volume=5_000))
    done = await b.order_status(o.id)
    assert done.status is OrderStatus.FILLED and done.filled_qty == 250
    assert [f.qty for f in await b.fills()] == [100, 25, 125]


async def test_modifying_an_order_does_not_fill_it_on_the_spot():
    b = sim()
    b.on_bar(bar(0, 100, 101, 99, 100))
    o = await b.place_order(req(order_type=OrderType.LIMIT, price=98.0))
    o = await b.modify_order(o.id, price=100.5)  # now crosses
    assert o.status is OrderStatus.OPEN and o.meta["marketable"] is True
    b.on_bar(bar(1, 100.0, 100.8, 99.9, 100.2))
    assert (await b.order_status(o.id)).status is OrderStatus.FILLED


async def test_futures_equity_is_cash_plus_unrealised():
    sym = "MCX:GOLDM-OCT26"
    b = sim(starting_cash=500_000.0)
    b.on_bar(bar(0, 150_000, 150_100, 149_900, 150_000, symbol=sym))
    await b.place_order(req(qty=1, symbol=sym, product=ProductType.NRML))
    b.on_bar(bar(1, 150_000, 150_300, 149_950, 150_200, symbol=sym))
    fees = (await b.fills())[0].fees.total
    # 1 lot x 10 (GOLDM multiplier) x (150,200 - 150,000)
    assert b.positions_value() == pytest.approx(2_000.0)
    assert b.equity() == pytest.approx(500_000.0 - fees + 2_000.0)
    assert (await b.funds()).equity == pytest.approx(b.equity())
