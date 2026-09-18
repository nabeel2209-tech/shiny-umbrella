from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trading.core.types import (
    IST,
    Bar,
    FeeBreakdown,
    Interval,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Side,
    Tick,
)


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        Tick(symbol="NSE:X", ts=datetime(2026, 9, 18, 10, 0), ltp=100.0)


def test_utc_converted_to_ist():
    t = Tick(symbol="NSE:X", ts=datetime(2026, 9, 18, 4, 30, tzinfo=UTC), ltp=100.0)
    assert t.ts.tzinfo is not None
    assert t.ts.utcoffset() == datetime.now(IST).utcoffset()
    assert (t.ts.hour, t.ts.minute) == (10, 0)


def test_bar_range_validation():
    ts = datetime(2026, 9, 18, 9, 15, tzinfo=IST)
    Bar(symbol="NSE:X", ts=ts, interval=Interval.M1, open=10, high=11, low=9, close=10.5)
    with pytest.raises(ValidationError):
        Bar(symbol="NSE:X", ts=ts, interval=Interval.M1, open=10, high=9.9, low=9, close=10.5)
    with pytest.raises(ValidationError):
        Bar(symbol="NSE:X", ts=ts, interval=Interval.M1, open=10, high=11, low=10.2, close=10.5)


def test_bar_is_frozen():
    ts = datetime(2026, 9, 18, 9, 15, tzinfo=IST)
    b = Bar(symbol="NSE:X", ts=ts, interval=Interval.M1, open=10, high=11, low=9, close=10.5)
    with pytest.raises(ValidationError):
        b.close = 12  # type: ignore[misc]


def test_order_request_needs_price_for_limit():
    with pytest.raises(ValidationError):
        OrderRequest(
            symbol="NSE:X",
            side=Side.BUY,
            qty=1,
            order_type=OrderType.LIMIT,
            product=ProductType.CNC,
        )
    with pytest.raises(ValidationError):
        OrderRequest(
            symbol="NSE:X",
            side=Side.BUY,
            qty=1,
            order_type=OrderType.SL,
            product=ProductType.CNC,
            price=10,
        )
    OrderRequest(
        symbol="NSE:X", side=Side.BUY, qty=1, order_type=OrderType.MARKET, product=ProductType.CNC
    )


def test_order_status_flags():
    assert OrderStatus.FILLED.is_terminal
    assert not OrderStatus.FILLED.is_working
    assert OrderStatus.PARTIAL.is_working
    assert OrderStatus.TRIGGER_PENDING.is_working


def test_side_helpers():
    assert Side.BUY.sign == 1 and Side.SELL.sign == -1
    assert Side.BUY.opposite is Side.SELL


def test_position_pnl():
    p = Position(symbol="NSE:X", product=ProductType.CNC, qty=10, avg_price=100, last_price=105)
    assert p.unrealised_pnl == 50
    assert p.market_value == 1050
    assert p.side is Side.BUY
    p.fees_paid = 5
    p.realised_pnl = 20
    assert p.net_pnl == 65


def test_fee_breakdown_add_and_total():
    a = FeeBreakdown(brokerage=1, stt=2, gst=0.5)
    b = FeeBreakdown(exchange=0.25, other=10)
    c = a + b
    assert c.total == pytest.approx(13.75)
    assert c.stt == 2 and c.other == 10


def test_interval_seconds():
    assert Interval.M5.seconds == 300
    assert Interval.D1.seconds == 86_400
