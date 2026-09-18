"""Core domain types shared by every layer of the platform.

Rules enforced here (see project constraints):
- Every timestamp is timezone-aware and normalised to Asia/Kolkata (IST).
  Naive datetimes are rejected at validation time.
- Symbols above the broker adapter are *canonical* strings such as
  ``NSE:RELIANCE`` or ``MCX:GOLDM-OCT26`` (see ``trading.brokers.symbols``).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

IST = ZoneInfo("Asia/Kolkata")


def to_ist(dt: datetime) -> datetime:
    """Convert an aware datetime to IST. Naive datetimes are a bug: reject them."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("naive datetime: timestamps must be timezone-aware")
    return dt.astimezone(IST)


def now_ist() -> datetime:
    return datetime.now(tz=IST)


ISTDatetime = Annotated[AwareDatetime, AfterValidator(to_ist)]


def new_id() -> str:
    return str(uuid.uuid4())


# Broker order tags (Dhan correlationId) allow at most 30 chars of [A-Za-z0-9 _-].
TAG_PATTERN = r"^[A-Za-z0-9 _-]{1,30}$"


def new_tag() -> str:
    """27-char idempotency tag: 11 hex chars of epoch-ms (sortable) + 16 random hex chars."""
    return f"{int(time.time() * 1000):011x}{uuid.uuid4().hex[:16]}"


# --------------------------------------------------------------------------- enums


class Exchange(StrEnum):
    NSE = "NSE"  # cash equities and indices
    NFO = "NFO"  # NSE futures & options
    MCX = "MCX"  # commodities


class InstrumentKind(StrEnum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    SL = "SL"  # stop-loss limit
    SLM = "SLM"  # stop-loss market


class ProductType(StrEnum):
    MIS = "MIS"  # intraday
    CNC = "CNC"  # equity delivery
    NRML = "NRML"  # carry-forward derivatives / commodities


class Validity(StrEnum):
    DAY = "DAY"
    IOC = "IOC"


class Urgency(StrEnum):
    PASSIVE = "PASSIVE"  # limit at touch, wait
    NORMAL = "NORMAL"  # limit at mid
    AGGRESSIVE = "AGGRESSIVE"  # marketable limit (never raw market on options)


class OrderStatus(StrEnum):
    PENDING = "PENDING"  # created locally, not yet acknowledged by broker
    TRIGGER_PENDING = "TRIGGER_PENDING"  # SL order waiting for trigger
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

    @property
    def is_working(self) -> bool:
        return self in {
            OrderStatus.PENDING,
            OrderStatus.TRIGGER_PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIAL,
        }


class Interval(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86_400}[self.value]


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# --------------------------------------------------------------------------- market data


class Bar(BaseModel):
    """OHLCV bar. ``ts`` is the bar *start* time in IST."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: ISTDatetime
    interval: Interval
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(default=0, ge=0)
    oi: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> Bar:
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(
                f"bar range inconsistent: o={self.open} h={self.high} l={self.low} c={self.close}"
            )
        return self


class Tick(BaseModel):
    """Last-traded / top-of-book snapshot from a live feed."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: ISTDatetime
    ltp: float = Field(gt=0)
    ltq: int = Field(default=0, ge=0)
    volume: int = Field(default=0, ge=0)  # cumulative session volume
    bid: float | None = Field(default=None, gt=0)
    ask: float | None = Field(default=None, gt=0)
    bid_qty: int | None = Field(default=None, ge=0)
    ask_qty: int | None = Field(default=None, ge=0)
    oi: int | None = Field(default=None, ge=0)

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.ltp


# --------------------------------------------------------------------------- signals & intents


class Signal(BaseModel):
    """Raw strategy output, before it is turned into an OrderIntent."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    ts: ISTDatetime
    strategy_id: str
    symbol: str
    score: float  # signed strength; sign = direction
    prob: float | None = Field(default=None, ge=0, le=1)
    expected_edge_bps: float | None = None
    model_version: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    """A proposed trade from a signal agent. Nothing is sent to a broker until the
    risk agent has attached an approval token to it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    ts: ISTDatetime
    strategy_id: str
    symbol: str
    side: Side
    qty: int = Field(gt=0)  # size suggestion; risk may cut it
    product: ProductType
    urgency: Urgency = Urgency.NORMAL
    reference_price: float = Field(gt=0)  # mid at signal time
    limit_band_bps: float = Field(default=10.0, ge=0)  # how far from reference we will go
    ttl_seconds: int = Field(default=300, gt=0)
    expected_edge_bps: float = 0.0  # expected *net* edge; risk applies cost threshold
    prob: float | None = Field(default=None, ge=0, le=1)  # p(win) for Kelly
    payoff_ratio: float | None = Field(default=None, gt=0)  # b for Kelly
    model_version: str | None = None
    signal_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RiskApproval(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str = Field(default_factory=new_id)
    intent_id: str
    ts: ISTDatetime
    expires_at: ISTDatetime
    approved_qty: int = Field(gt=0)
    checks: dict[str, str] = Field(default_factory=dict)  # rule -> outcome text


class RiskRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent_id: str
    ts: ISTDatetime
    rule: str
    reason: str


# --------------------------------------------------------------------------- orders & fills


class FeeBreakdown(BaseModel):
    """Transaction costs for one fill, in INR."""

    model_config = ConfigDict(frozen=True)

    brokerage: float = 0.0
    stt: float = 0.0  # STT / CTT
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0
    other: float = 0.0  # e.g. DP charges

    @property
    def total(self) -> float:
        parts = (
            self.brokerage,
            self.stt,
            self.exchange,
            self.sebi,
            self.stamp,
            self.gst,
            self.other,
        )
        return round(sum(parts), 4)

    def __add__(self, other: FeeBreakdown) -> FeeBreakdown:
        return FeeBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange=self.exchange + other.exchange,
            sebi=self.sebi + other.sebi,
            stamp=self.stamp + other.stamp,
            gst=self.gst + other.gst,
            other=self.other + other.other,
        )


class OrderRequest(BaseModel):
    """What the execution agent hands to a broker adapter."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Side
    qty: int = Field(gt=0)
    order_type: OrderType
    product: ProductType
    price: float | None = Field(default=None, gt=0)
    trigger_price: float | None = Field(default=None, gt=0)
    validity: Validity = Validity.DAY
    # our idempotency key, sent to the broker as its tag / correlation id
    tag: str = Field(default_factory=new_tag, pattern=TAG_PATTERN, max_length=30)
    disclosed_qty: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_prices(self) -> OrderRequest:
        if self.order_type in {OrderType.LIMIT, OrderType.SL} and self.price is None:
            raise ValueError(f"{self.order_type} order needs a price")
        if self.order_type in {OrderType.SL, OrderType.SLM} and self.trigger_price is None:
            raise ValueError(f"{self.order_type} order needs a trigger_price")
        return self


class Order(BaseModel):
    """Our view of an order. ``id`` is our UUID and is also the broker tag."""

    id: str = Field(default_factory=new_id)
    broker_order_id: str | None = None
    symbol: str
    side: Side
    qty: int = Field(gt=0)
    filled_qty: int = Field(default=0, ge=0)
    order_type: OrderType
    product: ProductType
    price: float | None = None
    trigger_price: float | None = None
    validity: Validity = Validity.DAY
    status: OrderStatus = OrderStatus.PENDING
    avg_fill_price: float | None = None
    status_message: str | None = None
    created_at: ISTDatetime
    updated_at: ISTDatetime
    intent_id: str | None = None
    approval_token: str | None = None
    parent_id: str | None = None  # set on child slices
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def remaining_qty(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @classmethod
    def from_request(cls, req: OrderRequest, ts: datetime, **extra: Any) -> Order:
        return cls(
            id=req.tag,
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            product=req.product,
            price=req.price,
            trigger_price=req.trigger_price,
            validity=req.validity,
            created_at=ts,
            updated_at=ts,
            **extra,
        )


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    order_id: str
    broker_order_id: str | None = None
    symbol: str
    side: Side
    qty: int = Field(gt=0)
    price: float = Field(gt=0)
    ts: ISTDatetime
    product: ProductType
    fees: FeeBreakdown = Field(default_factory=FeeBreakdown)
    multiplier: float = Field(default=1.0, gt=0)  # contract multiplier (MCX lots)

    @property
    def value(self) -> float:
        return self.qty * self.price * self.multiplier


class Position(BaseModel):
    """Net position in one instrument for one product type.

    ``realised_pnl`` is gross of fees; ``fees_paid`` is tracked separately so that
    net = realised_pnl - fees_paid.
    """

    symbol: str
    product: ProductType
    qty: int = 0  # signed: +long / -short (lots for MCX)
    avg_price: float = 0.0
    realised_pnl: float = 0.0
    fees_paid: float = 0.0
    last_price: float | None = None
    multiplier: float = Field(default=1.0, gt=0)  # PnL per unit price move per unit qty

    @property
    def unrealised_pnl(self) -> float:
        if self.qty == 0 or self.last_price is None:
            return 0.0
        return (self.last_price - self.avg_price) * self.qty * self.multiplier

    @property
    def net_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl - self.fees_paid

    @property
    def market_value(self) -> float:
        price = self.last_price if self.last_price is not None else self.avg_price
        return self.qty * price * self.multiplier

    @property
    def side(self) -> Side | None:
        if self.qty == 0:
            return None
        return Side.BUY if self.qty > 0 else Side.SELL


class Funds(BaseModel):
    model_config = ConfigDict(frozen=True)

    cash: float  # free cash
    margin_used: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0

    @property
    def available(self) -> float:
        return self.cash - self.margin_used

    @property
    def equity(self) -> float:
        return self.cash + self.unrealised_pnl


# --------------------------------------------------------------------------- ops


class Alert(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_id)
    ts: ISTDatetime
    level: AlertLevel
    source: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class Heartbeat(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts: ISTDatetime
    agent: str
    status: str = "ok"
    data: dict[str, Any] = Field(default_factory=dict)


class ControlCommand(BaseModel):
    """Out-of-band control messages (kill switch, pause, resume)."""

    model_config = ConfigDict(frozen=True)

    ts: ISTDatetime
    command: str  # KILL | PAUSE | RESUME | FLATTEN
    reason: str = ""
    issued_by: str = "system"
