"""Market-feed packet decoding, tick state, subscription batching and websocket
reconnect against a local websockets server. No network."""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

from trading.brokers.base import Instrument
from trading.brokers.dhan import (
    CODE_DISCONNECT,
    CODE_FULL,
    CODE_OI,
    CODE_PREV_CLOSE,
    CODE_QUOTE,
    CODE_TICKER,
    DhanConfig,
    DhanMarketFeed,
    DhanOrderFeed,
    FeedState,
    order_from_update,
    parse_packets,
)
from trading.brokers.dhan_instruments import build_symbol_map, read_master
from trading.core.types import IST, Exchange, InstrumentKind, OrderStatus, ProductType, Side

FIXTURE = Path(__file__).parent / "fixtures" / "dhan_master_sample.csv"
LTT = int(datetime(2026, 9, 18, 10, 15, 30, tzinfo=IST).timestamp())
NOW = datetime(2026, 9, 18, 10, 16, tzinfo=IST)


@pytest.fixture(scope="module")
def smap():
    return build_symbol_map(read_master(FIXTURE))


def header(code: int, seg: int, sid: int, total: int) -> bytes:
    return struct.pack("<BHBI", code, total - 8, seg, sid)


def ticker(seg=1, sid=2885, ltp=2501.5, ltt=LTT) -> bytes:
    return header(CODE_TICKER, seg, sid, 16) + struct.pack("<fI", ltp, ltt)


def quote(seg=1, sid=2885, ltp=2501.5, ltq=7, ltt=LTT, vol=123456) -> bytes:
    return header(CODE_QUOTE, seg, sid, 50) + struct.pack(
        "<fHIfIIIffff", ltp, ltq, ltt, 2500.0, vol, 1000, 2000, 2490.0, 2495.0, 2510.0, 2480.0
    )


def full(
    seg=2, sid=48704, ltp=25010.0, ltt=LTT, vol=999, oi=5000, bid=25009.5, ask=25010.5
) -> bytes:
    depth = b"".join(
        struct.pack("<IIHHff", 65 * (i + 1), 130 * (i + 1), 3, 4, bid - i, ask + i)
        for i in range(5)
    )
    return (
        header(CODE_FULL, seg, sid, 162)
        + struct.pack(
            "<fHIfIIIIIIffff",
            ltp,
            65,
            ltt,
            25000.0,
            vol,
            10,
            20,
            oi,
            6000,
            4000,
            24900.0,
            24950.0,
            25100.0,
            24800.0,
        )
        + depth
    )


def oi_packet(seg=2, sid=48704, oi=4321) -> bytes:
    return header(CODE_OI, seg, sid, 12) + struct.pack("<I", oi)


def prev_close(seg=1, sid=2885, pc=2480.0, poi=0) -> bytes:
    return header(CODE_PREV_CLOSE, seg, sid, 16) + struct.pack("<fI", pc, poi)


def disconnect(seg=0, sid=0, code=805) -> bytes:
    return header(CODE_DISCONNECT, seg, sid, 10) + struct.pack("<H", code)


# --------------------------------------------------------------------------- parsing


def test_parse_ticker():
    (p,) = parse_packets(ticker())
    assert (p.code, p.segment, p.security_id) == (CODE_TICKER, 1, 2885)
    assert p.data == {"ltp": pytest.approx(2501.5), "ltt": LTT}


def test_parse_quote_and_full_with_depth():
    (q,) = parse_packets(quote())
    assert q.data["ltp"] == pytest.approx(2501.5) and q.data["ltq"] == 7
    assert q.data["volume"] == 123456 and q.data["low"] == pytest.approx(2480.0)
    (f,) = parse_packets(full())
    assert (f.code, f.segment, f.security_id) == (CODE_FULL, 2, 48704)
    assert f.data["oi"] == 5000 and f.data["oi_high"] == 6000 and f.data["volume"] == 999
    assert len(f.data["depth"]) == 5
    top = f.data["depth"][0]
    assert top["bid"] == pytest.approx(25009.5) and top["ask"] == pytest.approx(25010.5)
    assert top["bid_qty"] == 65 and top["ask_qty"] == 130 and top["bid_orders"] == 3
    assert f.data["depth"][4]["bid"] == pytest.approx(25005.5)


def test_parse_oi_prev_close_disconnect_and_concatenated():
    pk = parse_packets(ticker() + oi_packet() + prev_close() + disconnect())
    assert [p.code for p in pk] == [CODE_TICKER, CODE_OI, CODE_PREV_CLOSE, CODE_DISCONNECT]
    assert pk[1].data == {"oi": 4321}
    assert pk[2].data["prev_close"] == pytest.approx(2480.0)
    assert pk[3].data == {"disconnect_code": 805}


def test_parse_unknown_and_truncated_packets():
    unknown = struct.pack("<BHBI", 1, 8, 0, 13) + b"\x00" * 8  # index packet, length = payload
    (p,) = parse_packets(unknown)
    assert p.code == 1 and p.data["raw"] == b"\x00" * 8
    assert parse_packets(ticker()[:12]) == []  # truncated -> dropped, no exception
    assert parse_packets(b"") == []


# --------------------------------------------------------------------------- state -> ticks


def test_feed_state_builds_ticks_from_packet_mix():
    st = FeedState()
    (t,) = parse_packets(ticker())
    tick = st.apply(t, "NSE:RELIANCE", NOW)
    assert tick is not None and tick.ltp == pytest.approx(2501.5) and tick.volume == 0
    assert tick.ts == datetime(2026, 9, 18, 10, 15, 30, tzinfo=IST) and tick.bid is None
    (q,) = parse_packets(quote(vol=500))
    tick = st.apply(q, "NSE:RELIANCE", NOW)
    assert tick.volume == 500 and tick.ltq == 7
    (t2,) = parse_packets(ticker(ltp=2502.0))
    tick = st.apply(t2, "NSE:RELIANCE", NOW)
    assert tick.ltp == pytest.approx(2502.0) and tick.volume == 500  # volume carried over

    (f,) = parse_packets(full())
    tick = st.apply(f, "NFO:NIFTY-OCT26", NOW)
    assert tick.bid == pytest.approx(25009.5) and tick.ask == pytest.approx(25010.5)
    assert tick.bid_qty == 65 and tick.ask_qty == 130 and tick.oi == 5000
    assert tick.mid == pytest.approx(25010.0)
    (o,) = parse_packets(oi_packet(oi=7777))
    assert st.apply(o, "NFO:NIFTY-OCT26", NOW) is None
    (t3,) = parse_packets(ticker(seg=2, sid=48704, ltp=25011.0))
    assert st.apply(t3, "NFO:NIFTY-OCT26", NOW).oi == 7777
    (pc,) = parse_packets(prev_close())
    assert st.apply(pc, "NSE:RELIANCE", NOW) is None
    assert st.snapshot(1, 2885)["prev_close"] == pytest.approx(2480.0)
    # a zero LTP (pre-open) or zero LTT never produces a bad tick
    (z,) = parse_packets(ticker(sid=1594, ltp=0.0))
    assert st.apply(z, "NSE:INFY", NOW) is None
    (z2,) = parse_packets(ticker(sid=1594, ltp=1500.0, ltt=0))
    assert st.apply(z2, "NSE:INFY", NOW).ts == NOW


# --------------------------------------------------------------------------- subscription


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent.append(msg)


async def test_subscribe_batches_of_100(smap):
    insts = [
        Instrument(
            symbol=f"NSE:S{i}",
            exchange=Exchange.NSE,
            kind=InstrumentKind.EQUITY,
            broker_id=str(i),
            broker_segment="NSE_EQ",
        )
        for i in range(250)
    ]
    cfg = DhanConfig(client_id="1", access_token="t", feed_mode="full")
    ws = FakeWS()
    sent = await DhanMarketFeed(cfg, smap).subscribe(ws, insts)
    assert sent == 3 and len(ws.sent) == 3
    first, last = json.loads(ws.sent[0]), json.loads(ws.sent[-1])
    assert first["RequestCode"] == 21 and first["InstrumentCount"] == 100
    assert first["InstrumentList"][0] == {"ExchangeSegment": "NSE_EQ", "SecurityId": "0"}
    assert last["InstrumentCount"] == 50
    cfg_q = DhanConfig(client_id="1", access_token="t", feed_mode="quote")
    ws2 = FakeWS()
    await DhanMarketFeed(cfg_q, smap).subscribe(ws2, insts[:1])
    assert json.loads(ws2.sent[0])["RequestCode"] == 17


def test_feed_url(smap):
    cfg = DhanConfig(client_id="1000000001", access_token="tok")
    assert DhanMarketFeed(cfg, smap).url() == (
        "wss://api-feed.dhan.co?version=2&token=tok&clientId=1000000001&authType=2"
    )


# --------------------------------------------------------------------------- websocket round trips


async def test_market_feed_reconnects_and_resubscribes(smap):
    received: list[dict] = []
    connections = 0

    async def handler(ws):  # type: ignore[no-untyped-def]
        nonlocal connections
        connections += 1
        received.append(json.loads(await ws.recv()))
        await ws.send(ticker(ltp=2501.5))
        if connections == 1:
            await ws.close(code=1011, reason="server restart")
            return
        await ws.send(ticker(ltp=2502.0) + oi_packet(seg=1, sid=2885, oi=1))
        await ws.send(disconnect(code=805))
        await ws.send(ticker(ltp=2503.0))
        await asyncio.sleep(1)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        cfg = DhanConfig(
            client_id="1",
            access_token="tok",
            feed_url=f"ws://127.0.0.1:{port}/",
            backoff_base=0.01,
            max_backoff=0.02,
        )
        feed = DhanMarketFeed(cfg, smap)
        gen = feed.stream(["NSE:RELIANCE"])
        ticks = []
        async for tick in gen:
            ticks.append(tick)
            if len(ticks) == 4:
                break
        await gen.aclose()
        await feed.close()
    assert connections == 2 and feed.reconnects == 1
    # first connection: one tick, then the server drops us; second: three more
    assert [t.ltp for t in ticks] == [
        pytest.approx(2501.5),
        pytest.approx(2501.5),
        pytest.approx(2502.0),
        pytest.approx(2503.0),
    ]
    assert all(t.symbol == "NSE:RELIANCE" for t in ticks)
    assert received[0]["RequestCode"] == 21 and received[0]["InstrumentCount"] == 1
    assert received[0]["InstrumentList"] == [{"ExchangeSegment": "NSE_EQ", "SecurityId": "2885"}]
    assert received[1] == received[0]  # resubscribed after reconnect
    assert feed.disconnect_codes == [805]


ORDER_ALERT = {
    "Type": "order_alert",
    "Data": {
        "Exchange": "NSE",
        "Segment": "E",
        "Source": "N",
        "SecurityId": "2885",
        "ClientId": "1000000001",
        "ExchOrderNo": "1400000000404591",
        "OrderNo": "1124091136546",
        "Product": "C",
        "TxnType": "B",
        "OrderType": "LMT",
        "Validity": "DAY",
        "DiscQuantity": 0,
        "RemainingQuantity": 6,
        "Quantity": 10,
        "TradedQty": 4,
        "Price": 2500,
        "TriggerPrice": 0,
        "TradedPrice": 2499.5,
        "AvgTradedPrice": 2499.5,
        "OrderDateTime": "2026-09-18 14:39:29",
        "ExchOrderTime": "2026-09-18 14:39:29",
        "LastUpdatedTime": "2026-09-18 14:39:31",
        "Remarks": "NR",
        "ReasonDescription": "CONFIRMED",
        "Instrument": "EQUITY",
        "Symbol": "RELIANCE",
        "ProductName": "CNC",
        "Status": "Pending",
        "LotSize": 1,
        "CorrelationId": "abc-123",
    },
}


def test_order_from_update_mapping(smap):
    o = order_from_update(ORDER_ALERT["Data"], smap, NOW)
    assert o.id == "abc-123" and o.broker_order_id == "1124091136546"
    assert o.symbol == "NSE:RELIANCE" and o.side is Side.BUY and o.product is ProductType.CNC
    assert o.status is OrderStatus.OPEN and o.qty == 10 and o.filled_qty == 4
    assert o.avg_fill_price == 2499.5 and o.price == 2500.0 and o.trigger_price is None
    assert o.created_at == datetime(2026, 9, 18, 14, 39, 29, tzinfo=IST)
    assert o.updated_at == datetime(2026, 9, 18, 14, 39, 31, tzinfo=IST)
    unknown = ORDER_ALERT["Data"] | {
        "SecurityId": "14366",
        "Symbol": "IDEA",
        "CorrelationId": "",
        "Status": "Cancelled",
    }
    o2 = order_from_update(unknown, smap, NOW)
    assert o2.symbol == "NSE:IDEA" and o2.id == "dhan:1124091136546"
    assert o2.status is OrderStatus.CANCELLED


async def test_order_feed_logs_in_and_yields_orders(smap):
    logins: list[dict] = []

    async def handler(ws):  # type: ignore[no-untyped-def]
        logins.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"Type": "connected"}))
        await ws.send("not json")
        await ws.send(json.dumps(ORDER_ALERT))
        await asyncio.sleep(1)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        cfg = DhanConfig(
            client_id="1000000001", access_token="tok", order_update_url=f"ws://127.0.0.1:{port}/"
        )
        feed = DhanOrderFeed(cfg, smap)
        gen = feed.stream()
        order = await anext(gen)
        await gen.aclose()
        await feed.close()
    assert logins == [
        {"LoginReq": {"MsgCode": 42, "ClientId": "1000000001", "Token": "tok"}, "UserType": "SELF"}
    ]
    assert order.id == "abc-123" and order.status is OrderStatus.OPEN and feed.reconnects == 0
