"""DhanBroker REST behaviour against a mocked HTTP layer (respx). No network."""

from __future__ import annotations

import base64
import itertools
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
import respx

from trading.backtest.costs import compute_fees
from trading.brokers.base import AuthError, RateLimited, UnknownOrder
from trading.brokers.dhan import (
    DhanBroker,
    DhanConfig,
    RateLimiter,
    bars_from_chart,
    extract_error,
    intraday_windows,
    jwt_expiry,
)
from trading.brokers.dhan_instruments import build_symbol_map, read_master
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Interval,
    OrderRequest,
    OrderStatus,
    OrderType,
    ProductType,
    Side,
)

BASE = "https://api.dhan.co/v2"
FIXTURE = Path(__file__).parent / "fixtures" / "dhan_master_sample.csv"
NOW = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


def make_jwt(exp: int) -> str:
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'HS256'})}.{b64({'exp': exp, 'sub': 'x'})}.sig"


async def noop_sleep(_: float) -> None:
    return None


@pytest.fixture(scope="module")
def smap():
    return build_symbol_map(read_master(FIXTURE))


@pytest.fixture
def clock():
    return SimClock(NOW)


@pytest.fixture
def cfg(clock):
    return DhanConfig(
        client_id="1000000001",
        access_token=make_jwt(int(clock.now().timestamp()) + 20 * 3600),
        max_retries=2,
        retry_base_delay=0.0,
    )


@pytest.fixture
async def broker(cfg, smap, clock):
    b = DhanBroker(
        cfg, symbols=smap, clock=clock, sleep=noop_sleep, limiter=RateLimiter(sleep=noop_sleep)
    )
    yield b
    await b.close()


BOOK_ROW = {
    "dhanClientId": "1000000001",
    "orderId": "112111182198",
    "correlationId": "abc-123",
    "orderStatus": "PENDING",
    "transactionType": "BUY",
    "exchangeSegment": "NSE_EQ",
    "productType": "CNC",
    "orderType": "LIMIT",
    "validity": "DAY",
    "tradingSymbol": "RELIANCE",
    "securityId": "2885",
    "quantity": 10,
    "disclosedQuantity": 0,
    "price": 2500.0,
    "triggerPrice": 0.0,
    "afterMarketOrder": False,
    "createTime": "2026-09-18 10:05:00",
    "updateTime": "2026-09-18 10:05:30",
    "exchangeTime": "2026-09-18 10:05:01",
    "omsErrorCode": "",
    "omsErrorDescription": "",
    "filledQty": 4,
    "remainingQuantity": 6,
    "averageTradedPrice": 2499.5,
}


# --------------------------------------------------------------------------- pure helpers


def test_jwt_expiry_and_garbage():
    exp = int(NOW.timestamp()) + 3600
    assert jwt_expiry(make_jwt(exp)) == datetime.fromtimestamp(exp, tz=IST)
    assert jwt_expiry("not-a-jwt") is None
    assert jwt_expiry("") is None


def test_extract_error_shapes():
    assert extract_error({"errorType": "Input", "errorCode": "DH-905", "errorMessage": "bad"}) == (
        "DH-905",
        "bad",
    )
    assert extract_error(
        {"status": "failed", "remarks": {"error_code": "DH-907", "error_message": "x"}}
    ) == ("DH-907", "x")
    assert extract_error({"status": "failed", "data": {"805": "Too many requests"}}) == (
        "805",
        "Too many requests",
    )
    assert extract_error({"orderId": "1", "orderStatus": "TRANSIT"}) == (None, None)
    assert extract_error([1, 2]) == (None, None)


def test_intraday_windows():
    start = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2026, 6, 30, 15, 29, tzinfo=IST)
    w = intraday_windows(start, end)
    assert len(w) == 3
    assert w[0][0] == start and w[-1][1] == end
    for (_, hi), (lo, _) in itertools.pairwise(w):
        assert (lo - hi).total_seconds() == 1
    assert intraday_windows(start, start) == [(start, start)]


def test_bars_from_chart_converts_epoch_and_skips_bad_rows():
    t = int(datetime(2026, 9, 18, 9, 15, tzinfo=IST).timestamp())
    data = {
        "open": [100.0, 100.0],
        "high": [101.0, 99.0],  # second row: high < low -> dropped
        "low": [99.0, 100.0],
        "close": [100.5, 100.0],
        "volume": [10, 20],
        "timestamp": [t, t + 60],
        "open_interest": [7, 8],
    }
    bars, bad = bars_from_chart(data, "NFO:NIFTY-OCT26", Interval.M1, with_oi=True)
    assert bad == 1 and len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 18, 9, 15, tzinfo=IST) and bars[0].oi == 7
    bars, _ = bars_from_chart(data, "NSE:RELIANCE", Interval.M1, with_oi=False)
    assert bars[0].oi is None
    assert bars_from_chart({"status": "failed"}, "NSE:X", Interval.D1, with_oi=False) == ([], 0)


async def test_rate_limiter_paces_and_caps():
    t = [0.0]
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    rl = RateLimiter(
        {"x": (2, 3)}, clock=lambda: t[0], sleep=sleep, today=lambda: date(2026, 9, 18)
    )
    await rl.acquire("x")
    await rl.acquire("x")
    assert sleeps == []
    await rl.acquire("x")  # third call inside the same second must wait
    assert len(sleeps) == 1 and sleeps[0] >= 1.0
    assert rl.usage("x") == (3, 3)
    with pytest.raises(RateLimited):
        await rl.acquire("x")  # daily cap


# --------------------------------------------------------------------------- historical


async def test_daily_history_request_and_parse(broker):
    stamps = [int(datetime(2026, 9, d, 0, 0, tzinfo=IST).timestamp()) for d in (16, 17, 18)]
    body = {
        "open": [1, 2, 3],
        "high": [2, 3, 4],
        "low": [0.5, 1.5, 2.5],
        "close": [1.5, 2.5, 3.5],
        "volume": [10, 20, 30],
        "timestamp": stamps,
        "open_interest": [0, 0, 0],
    }
    with respx.mock(base_url=BASE) as router:
        route = router.post("/charts/historical").mock(return_value=httpx.Response(200, json=body))
        bars = await broker.historical(
            "NSE:RELIANCE",
            Interval.D1,
            datetime(2026, 9, 16, tzinfo=IST),
            datetime(2026, 9, 18, tzinfo=IST),
        )
    req = route.calls.last.request
    assert json.loads(req.content) == {
        "securityId": "2885",
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "oi": False,
        "fromDate": "2026-09-16",
        "toDate": "2026-09-19",  # exclusive at Dhan, inclusive for us
    }
    assert req.headers["access-token"] == broker.cfg.token
    assert req.headers["client-id"] == "1000000001"
    assert [b.ts for b in bars] == [datetime(2026, 9, d, tzinfo=IST) for d in (16, 17, 18)]
    assert bars[0].oi is None and bars[-1].close == 3.5 and bars[0].interval is Interval.D1


async def test_intraday_is_chunked_into_90_day_windows_and_deduped(broker):
    start = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    end = datetime(2026, 6, 30, 15, 29, tzinfo=IST)
    t = int(datetime(2026, 3, 2, 9, 15, tzinfo=IST).timestamp())
    same_bar = {
        "open": [100],
        "high": [101],
        "low": [99],
        "close": [100.5],
        "volume": [5],
        "timestamp": [t],
        "open_interest": [1234],
    }
    with respx.mock(base_url=BASE) as router:
        route = router.post("/charts/intraday").mock(
            return_value=httpx.Response(200, json=same_bar)
        )
        bars = await broker.historical("NFO:NIFTY-OCT26", Interval.M1, start, end)
    assert route.call_count == 3
    bodies = [json.loads(c.request.content) for c in route.calls]
    assert bodies[0]["fromDate"] == "2026-01-01 09:15:00"
    assert bodies[0]["interval"] == "1" and bodies[0]["oi"] is True
    assert bodies[0]["instrument"] == "FUTIDX" and bodies[0]["exchangeSegment"] == "NSE_FNO"
    assert bodies[0]["securityId"] == "48704"
    assert bodies[-1]["toDate"] == "2026-06-30 15:30:00"  # one extra bar requested
    assert len(bars) == 1 and bars[0].oi == 1234  # duplicates across windows collapsed


async def test_historical_empty_when_start_after_end(broker):
    assert await broker.historical("NSE:RELIANCE", Interval.M1, NOW, NOW.replace(hour=9)) == []


# --------------------------------------------------------------------------- orders


async def test_place_order_maps_payload_and_is_idempotent(broker):
    req = OrderRequest(
        symbol="NSE:RELIANCE",
        side=Side.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        product=ProductType.CNC,
        price=2500.0,
        tag="abc-123",
    )
    with respx.mock(base_url=BASE) as router:
        route = router.post("/orders").mock(
            return_value=httpx.Response(
                200, json={"orderId": "112111182198", "orderStatus": "TRANSIT"}
            )
        )
        order = await broker.place_order(req)
        again = await broker.place_order(req)
    assert json.loads(route.calls.last.request.content) == {
        "dhanClientId": "1000000001",
        "correlationId": "abc-123",
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "LIMIT",
        "validity": "DAY",
        "securityId": "2885",
        "quantity": 10,
        "disclosedQuantity": 0,
        "price": 2500.0,
        "triggerPrice": 0.0,
        "afterMarketOrder": False,
    }
    assert order.id == "abc-123" and order.broker_order_id == "112111182198"
    assert order.status is OrderStatus.PENDING  # TRANSIT
    assert route.call_count == 1 and again is order


async def test_place_order_mcx_stop_market_mapping(broker):
    req = OrderRequest(
        symbol="MCX:GOLDM-OCT26",
        side=Side.SELL,
        qty=1,
        order_type=OrderType.SLM,
        product=ProductType.NRML,
        trigger_price=149000.0,
    )
    with respx.mock(base_url=BASE) as router:
        route = router.post("/orders").mock(
            return_value=httpx.Response(200, json={"orderId": "9", "orderStatus": "PENDING"})
        )
        order = await broker.place_order(req)
    body = json.loads(route.calls.last.request.content)
    assert body["exchangeSegment"] == "MCX_COMM" and body["securityId"] == "569003"
    assert body["productType"] == "MARGIN" and body["orderType"] == "STOP_LOSS_MARKET"
    assert body["price"] == 0.0 and body["triggerPrice"] == 149000.0 and body["quantity"] == 1
    assert len(body["correlationId"]) <= 30
    assert order.status is OrderStatus.OPEN


async def test_place_order_on_index_is_rejected_locally(broker):
    req = OrderRequest(
        symbol="NSE:NIFTY",
        side=Side.BUY,
        qty=1,
        order_type=OrderType.MARKET,
        product=ProductType.MIS,
    )
    with respx.mock(base_url=BASE):  # any HTTP call would fail: nothing is routed
        order = await broker.place_order(req)
    assert order.status is OrderStatus.REJECTED and "not tradable" in (order.status_message or "")


async def test_input_error_becomes_rejected_order(broker):
    req = OrderRequest(
        symbol="NSE:RELIANCE",
        side=Side.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        product=ProductType.MIS,
    )
    with respx.mock(base_url=BASE) as router:
        router.post("/orders").mock(
            return_value=httpx.Response(
                400,
                json={
                    "errorType": "Input_Exception",
                    "errorCode": "DH-905",
                    "errorMessage": "Missing",
                },
            )
        )
        order = await broker.place_order(req)
    assert order.status is OrderStatus.REJECTED and "DH-905" in (order.status_message or "")


async def test_order_status_modify_cancel(broker):
    with respx.mock(base_url=BASE) as router:
        router.get("/orders/external/abc-123").mock(
            return_value=httpx.Response(200, json=[BOOK_ROW])
        )
        o = await broker.order_status("abc-123")
        assert o.broker_order_id == "112111182198" and o.status is OrderStatus.OPEN
        assert o.filled_qty == 4 and o.avg_fill_price == 2499.5 and o.symbol == "NSE:RELIANCE"
        assert o.created_at == datetime(2026, 9, 18, 10, 5, tzinfo=IST)
        assert o.updated_at == datetime(2026, 9, 18, 10, 5, 30, tzinfo=IST)

        by_id = router.get("/orders/112111182198").mock(
            return_value=httpx.Response(200, json=BOOK_ROW)
        )
        put = router.put("/orders/112111182198").mock(
            return_value=httpx.Response(
                200, json={"orderId": "112111182198", "orderStatus": "TRANSIT"}
            )
        )
        o2 = await broker.modify_order("abc-123", price=2490.0)
        assert json.loads(put.calls.last.request.content) == {
            "dhanClientId": "1000000001",
            "orderId": "112111182198",
            "orderType": "LIMIT",
            "quantity": 10,
            "price": 2490.0,
            "disclosedQuantity": 0,
            "triggerPrice": 0.0,
            "validity": "DAY",
        }
        assert by_id.call_count == 2 and o2.id == "abc-123"

        delete = router.delete("/orders/112111182198").mock(
            return_value=httpx.Response(
                200, json={"orderId": "112111182198", "orderStatus": "CANCELLED"}
            )
        )
        o3 = await broker.cancel_order("abc-123")
        assert o3.status is OrderStatus.CANCELLED and delete.call_count == 1

        router.get("/orders/external/nope").mock(return_value=httpx.Response(200, json=[]))
        with pytest.raises(UnknownOrder):
            await broker.order_status("nope")


async def test_orders_trades_positions_funds(broker):
    manual = BOOK_ROW | {
        "orderId": "555",
        "correlationId": "",
        "orderStatus": "TRADED",
        "filledQty": 10,
    }
    trade = {
        "orderId": "112111182198",
        "exchangeOrderId": "1400000000404591",
        "exchangeTradeId": "15263",
        "transactionType": "BUY",
        "exchangeSegment": "NSE_EQ",
        "productType": "CNC",
        "orderType": "LIMIT",
        "tradingSymbol": "RELIANCE",
        "securityId": "2885",
        "tradedQuantity": 4,
        "tradedPrice": 2499.5,
        "createTime": "2026-09-18 10:05:01",
        "updateTime": "2026-09-18 10:05:01",
        "exchangeTime": "2026-09-18 10:05:01",
    }
    pos = {
        "dhanClientId": "1000000001",
        "tradingSymbol": "GOLDM OCT FUT",
        "securityId": "569003",
        "positionType": "LONG",
        "exchangeSegment": "MCX_COMM",
        "productType": "MARGIN",
        "buyAvg": 150000.0,
        "buyQty": 1,
        "costPrice": 150000.0,
        "sellAvg": 0,
        "sellQty": 0,
        "netQty": 1,
        "realizedProfit": 0,
        "unrealizedProfit": 500,
        "rbiReferenceRate": 1,
        "multiplier": 10,
    }
    with respx.mock(base_url=BASE) as router:
        router.get("/orders").mock(return_value=httpx.Response(200, json=[BOOK_ROW, manual]))
        router.get("/trades").mock(return_value=httpx.Response(200, json=[trade]))
        router.get("/positions").mock(return_value=httpx.Response(200, json=[pos]))
        router.get("/fundlimit").mock(
            return_value=httpx.Response(
                200,
                json={
                    "dhanClientId": "1000000001",
                    "availabelBalance": 98000.5,
                    "sodLimit": 100000,
                    "utilizedAmount": 1999.5,
                    "withdrawableBalance": 98000.5,
                },
            )
        )
        orders, fills = await broker.reconcile()
        positions = await broker.positions()
        funds = await broker.funds()
    assert [o.id for o in orders] == ["abc-123", "dhan:555"]
    assert orders[1].status is OrderStatus.FILLED
    f = fills[0]
    assert f.order_id == "abc-123" and f.symbol == "NSE:RELIANCE" and f.qty == 4
    assert f.ts == datetime(2026, 9, 18, 10, 5, 1, tzinfo=IST)
    assert f.fees.total == compute_fees("NSE:RELIANCE", Side.BUY, 4, 2499.5, ProductType.CNC).total
    p = positions[0]
    assert p.symbol == "MCX:GOLDM-OCT26" and p.qty == 1 and p.avg_price == 150000.0
    assert p.product is ProductType.NRML and p.multiplier == 10.0
    assert funds.cash == 98000.5 and funds.margin_used == 1999.5


async def test_ltp_groups_by_segment(broker):
    with respx.mock(base_url=BASE) as router:
        route = router.post("/marketfeed/ltp").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "NSE_EQ": {"2885": {"last_price": 2501.5}},
                        "NSE_FNO": {"48704": {"last_price": 25010.0}},
                    },
                    "status": "success",
                },
            )
        )
        out = await broker.ltp(["NSE:RELIANCE", "NFO:NIFTY-OCT26"])
    assert json.loads(route.calls.last.request.content) == {"NSE_EQ": [2885], "NSE_FNO": [48704]}
    assert out == {"NSE:RELIANCE": 2501.5, "NFO:NIFTY-OCT26": 25010.0}


# --------------------------------------------------------------------------- errors & tokens


async def test_auth_error(broker):
    with respx.mock(base_url=BASE) as router:
        router.get("/fundlimit").mock(
            return_value=httpx.Response(
                401, json={"errorCode": "DH-901", "errorMessage": "invalid token"}
            )
        )
        with pytest.raises(AuthError):
            await broker.funds()


async def test_rate_limit_retry_then_success_then_give_up(broker):
    ok = httpx.Response(200, json={"availabelBalance": 1.0})
    limited = httpx.Response(429, json={"errorCode": "DH-904", "errorMessage": "slow down"})
    with respx.mock(base_url=BASE) as router:
        route = router.get("/fundlimit").mock(side_effect=[limited, ok])
        assert (await broker.funds()).cash == 1.0
        assert route.call_count == 2
        route.side_effect = [limited, limited, limited]
        with pytest.raises(RateLimited):
            await broker.funds()


async def test_server_error_is_retried(broker):
    with respx.mock(base_url=BASE) as router:
        route = router.get("/fundlimit").mock(
            side_effect=[httpx.Response(502), httpx.Response(200, json={"availabelBalance": 2.0})]
        )
        assert (await broker.funds()).cash == 2.0
        assert route.call_count == 2


async def test_connect_renews_token_close_to_expiry(smap, clock):
    soon = int(clock.now().timestamp()) + 3600  # 1h left < renew_before_hours
    new_token = make_jwt(int(clock.now().timestamp()) + 24 * 3600)
    cfg = DhanConfig(client_id="1000000001", access_token=make_jwt(soon), retry_base_delay=0.0)
    broker = DhanBroker(
        cfg, symbols=smap, clock=clock, sleep=noop_sleep, limiter=RateLimiter(sleep=noop_sleep)
    )
    with respx.mock(base_url=BASE) as router:
        renew = router.post("/RenewToken").mock(
            return_value=httpx.Response(200, json={"accessToken": new_token})
        )
        funds = router.get("/fundlimit").mock(
            return_value=httpx.Response(200, json={"availabelBalance": 5})
        )
        await broker.connect()
    assert renew.call_count == 1
    assert renew.calls.last.request.headers["dhanClientId"] == "1000000001"
    assert broker.client.token == new_token
    assert funds.calls.last.request.headers["access-token"] == new_token
    assert broker.token_status()["hours_left"] is not None
    await broker.close()


async def test_connect_rejects_expired_token(smap, clock):
    cfg = DhanConfig(client_id="1", access_token=make_jwt(int(clock.now().timestamp()) - 10))
    broker = DhanBroker(cfg, symbols=smap, clock=clock)
    with respx.mock(base_url=BASE), pytest.raises(AuthError):
        await broker.connect()
    await broker.close()


def test_symbols_required_before_use(cfg):
    from trading.brokers.base import NotConnected

    with pytest.raises(NotConnected):
        _ = DhanBroker(cfg).symbols
