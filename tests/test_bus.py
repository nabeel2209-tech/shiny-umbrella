import os
from datetime import datetime

import pytest
from pydantic import BaseModel

from trading.core.bus import InMemoryBus, RedisBus, Topics, decode, encode, make_bus
from trading.core.types import IST, Bar, Fill, Interval, ProductType, Side


def bar(symbol="NSE:RELIANCE"):
    return Bar(
        symbol=symbol,
        ts=datetime(2026, 9, 18, 9, 15, tzinfo=IST),
        interval=Interval.M1,
        open=100,
        high=101,
        low=99,
        close=100.5,
        volume=10,
    )


def test_encode_decode_round_trip():
    topic, msg = decode(encode(Topics.bars("NSE:RELIANCE"), bar()))
    assert topic == "bars.NSE:RELIANCE"
    assert isinstance(msg, Bar)
    assert msg == bar()
    assert msg.ts.utcoffset() == bar().ts.utcoffset()


def test_unregistered_type_rejected():
    class Foo(BaseModel):
        x: int

    with pytest.raises(TypeError):
        encode("x", Foo(x=1))


async def test_pattern_subscription_and_isolation():
    bus = InMemoryBus()
    got: list[tuple[str, object]] = []

    async def handler(topic, msg):
        got.append((topic, msg))

    await bus.subscribe(Topics.BARS_ALL, handler)
    await bus.publish(Topics.bars("NSE:RELIANCE"), bar())
    await bus.publish(Topics.bars("MCX:GOLDM-OCT26"), bar("MCX:GOLDM-OCT26"))
    await bus.publish(
        Topics.FILLS,
        Fill(
            order_id="o1",
            symbol="NSE:RELIANCE",
            side=Side.BUY,
            qty=1,
            price=100,
            ts=datetime(2026, 9, 18, 9, 15, tzinfo=IST),
            product=ProductType.CNC,
        ),
    )
    assert [t for t, _ in got] == ["bars.NSE:RELIANCE", "bars.MCX:GOLDM-OCT26"]
    assert all(isinstance(m, Bar) for _, m in got)
    assert bus.published == 3


async def test_exact_topic_and_cancel():
    bus = InMemoryBus()
    seen = []

    async def handler(topic, msg):
        seen.append(msg)

    sub = await bus.subscribe(Topics.bars("NSE:TCS"), handler)
    await bus.publish(Topics.bars("NSE:TCS"), bar("NSE:TCS"))
    await bus.publish(Topics.bars("NSE:INFY"), bar("NSE:INFY"))
    assert len(seen) == 1
    await sub.cancel()
    await bus.publish(Topics.bars("NSE:TCS"), bar("NSE:TCS"))
    assert len(seen) == 1


async def test_handler_error_does_not_stop_others():
    bus = InMemoryBus()
    seen = []

    async def bad(topic, msg):
        raise RuntimeError("boom")

    async def good(topic, msg):
        seen.append(msg)

    await bus.subscribe(Topics.BARS_ALL, bad)
    await bus.subscribe(Topics.BARS_ALL, good)
    await bus.publish(Topics.bars("NSE:X"), bar("NSE:X"))
    assert len(seen) == 1
    assert bus.errors == 1


async def test_dispatch_is_synchronous_and_ordered():
    """Handlers run to completion inside publish(), in subscription order, so a
    chain bar -> intent -> approval is deterministic."""
    bus = InMemoryBus()
    order = []

    async def first(topic, msg):
        order.append("first")

    async def second(topic, msg):
        order.append("second")

    await bus.subscribe(Topics.BARS_ALL, first)
    await bus.subscribe(Topics.BARS_ALL, second)
    await bus.publish(Topics.bars("NSE:X"), bar("NSE:X"))
    assert order == ["first", "second"]


def test_make_bus_defaults_to_memory():
    assert isinstance(make_bus(None), InMemoryBus)
    assert isinstance(make_bus(""), InMemoryBus)


@pytest.mark.redis
async def test_redis_bus_round_trip():
    url = os.environ.get("REDIS_URL")
    if not url:
        pytest.skip("REDIS_URL not set")
    bus = RedisBus(url)
    if not await bus.ping():
        pytest.skip("redis not reachable")
    import asyncio

    got: asyncio.Queue = asyncio.Queue()

    async def handler(topic, msg):
        await got.put((topic, msg))

    sub = await bus.subscribe(Topics.BARS_ALL, handler)
    await asyncio.sleep(0.2)  # let psubscribe settle
    await bus.publish(Topics.bars("NSE:RELIANCE"), bar())
    topic, msg = await asyncio.wait_for(got.get(), timeout=3)
    assert topic == "bars.NSE:RELIANCE" and msg == bar()
    await sub.cancel()
    await bus.close()
