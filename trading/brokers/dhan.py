"""Dhan (DhanHQ v2) broker adapter.

Every fact about the API used here is recorded in ``brokers/README.md``; anything
the docs leave open is a ``TODO(dhan)``. Components:

- ``RateLimiter``   client-side per-second / per-day caps per API category
- ``DhanClient``    httpx wrapper: headers, error mapping, retries, JWT expiry, renew
- ``parse_packets`` binary market-feed decoding; ``FeedState`` turns packets into Ticks
- ``DhanMarketFeed`` / ``DhanOrderFeed``  websocket streams with reconnect + backoff
- ``DhanBroker``    the ``Broker`` implementation the engine uses

Boundary rules (constraint 6): every timestamp leaving this module is tz-aware IST and
every symbol is canonical. Order tags (constraint 4) go to Dhan as ``correlationId``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import random
import struct
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, SecretStr, ValidationError
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from trading.backtest.costs import compute_fees
from trading.brokers.base import (
    AuthError,
    BrokerError,
    Instrument,
    NotConnected,
    RateLimited,
    UnknownOrder,
)
from trading.brokers.dhan_instruments import DHAN_SEGMENT, DhanInstrumentMaster
from trading.brokers.symbols import SymbolMap, UnknownSymbol
from trading.core.clock import Clock, SystemClock
from trading.core.types import (
    IST,
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
    Validity,
    now_ist,
    to_ist,
)

log = logging.getLogger(__name__)

BASE_URL = "https://api.dhan.co/v2"
FEED_URL = "wss://api-feed.dhan.co"
ORDER_UPDATE_URL = "wss://api-order-update.dhan.co"

SEGMENT_CODES = {
    "IDX_I": 0,
    "NSE_EQ": 1,
    "NSE_FNO": 2,
    "NSE_CURRENCY": 3,
    "BSE_EQ": 4,
    "MCX_COMM": 5,
    "BSE_CURRENCY": 7,
    "BSE_FNO": 8,
}
SEGMENT_BY_CODE = {v: k for k, v in SEGMENT_CODES.items()}

FEED_SUBSCRIBE = {"ticker": 15, "quote": 17, "full": 21}
FEED_DISCONNECT = 12
FEED_BATCH = 100
MAX_INSTRUMENTS_PER_CONNECTION = 5000
QUOTE_BATCH = 1000

INTERVAL_MINUTES = {Interval.M1: 1, Interval.M5: 5, Interval.M15: 15, Interval.H1: 60}
INTRADAY_WINDOW_DAYS = 90

PRODUCT_TO_DHAN = {ProductType.MIS: "INTRADAY", ProductType.CNC: "CNC", ProductType.NRML: "MARGIN"}
PRODUCT_FROM_DHAN = {
    "INTRADAY": ProductType.MIS,
    "CNC": ProductType.CNC,
    "MARGIN": ProductType.NRML,
    "MTF": ProductType.CNC,
    "CO": ProductType.MIS,
    "BO": ProductType.MIS,
    # order-update feed short codes
    "I": ProductType.MIS,
    "C": ProductType.CNC,
    "M": ProductType.NRML,
    "F": ProductType.CNC,
    "V": ProductType.MIS,
    "B": ProductType.MIS,
}
ORDER_TYPE_TO_DHAN = {
    OrderType.LIMIT: "LIMIT",
    OrderType.MARKET: "MARKET",
    OrderType.SL: "STOP_LOSS",
    OrderType.SLM: "STOP_LOSS_MARKET",
}
ORDER_TYPE_FROM_DHAN = {v: k for k, v in ORDER_TYPE_TO_DHAN.items()} | {
    "LMT": OrderType.LIMIT,
    "MKT": OrderType.MARKET,
    "SL": OrderType.SL,
    "SLM": OrderType.SLM,
}
STATUS_FROM_DHAN = {
    "TRANSIT": OrderStatus.PENDING,
    "PENDING": OrderStatus.OPEN,
    "PART_TRADED": OrderStatus.PARTIAL,
    "TRADED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
    "TRIGGERED": OrderStatus.OPEN,
    "CLOSED": OrderStatus.FILLED,
}
SIDE_FROM_DHAN = {"BUY": Side.BUY, "SELL": Side.SELL, "B": Side.BUY, "S": Side.SELL}

AUTH_CODES = {"DH-901", "DH-902", "806", "807", "808", "809", "810"}
RATE_CODES = {"DH-904", "805"}
RETRY_CODES = {"DH-908", "DH-909", "800"}
INPUT_CODES = {"DH-905", "DH-906", "DH-907", "811", "812", "813", "814"}

Sleep = Callable[[float], Awaitable[None]]


# --------------------------------------------------------------------------- config


class DhanConfig(BaseModel):
    client_id: str
    access_token: SecretStr
    base_url: str = BASE_URL
    feed_url: str = FEED_URL
    order_update_url: str = ORDER_UPDATE_URL
    feed_mode: Literal["ticker", "quote", "full"] = "full"
    timeout: float = 15.0
    max_retries: int = 3
    retry_base_delay: float = 0.5
    backoff_base: float = 1.0
    max_backoff: float = 60.0
    renew_before_hours: float = 2.0

    @classmethod
    def from_settings(cls, settings: Any) -> DhanConfig:
        return cls(
            client_id=settings.dhan_client_id,
            access_token=settings.dhan_access_token,
            feed_mode=settings.dhan_feed_mode,
        )

    @property
    def token(self) -> str:
        return self.access_token.get_secret_value()


# --------------------------------------------------------------------------- errors


class DhanError(BrokerError):
    def __init__(self, code: str | None, message: str | None, status: int | None = None):
        super().__init__(f"{code or 'error'}: {message or 'no message'} (HTTP {status})")
        self.code = code
        self.message = message
        self.status = status


def extract_error(body: Any) -> tuple[str | None, str | None]:
    """Pull (code, message) out of any of Dhan's error body shapes."""
    if not isinstance(body, dict):
        return None, None
    code = body.get("errorCode")
    if code:
        return str(code), body.get("errorMessage") or body.get("errorType")
    remarks = body.get("remarks")
    if isinstance(remarks, dict) and (remarks.get("error_code") or remarks.get("error_message")):
        return (
            str(remarks.get("error_code") or "failed"),
            remarks.get("error_message") or remarks.get("error_type"),
        )
    if body.get("status") == "failed":
        data = body.get("data")
        if isinstance(data, dict):
            for k, v in data.items():
                if str(k).isdigit():
                    return str(k), str(v)
        return "failed", str(remarks or data or body)
    return None, None


def jwt_expiry(token: str) -> datetime | None:
    """Read the ``exp`` claim of a JWT without verifying it (IST)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = data.get("exp")
        return datetime.fromtimestamp(int(exp), tz=IST) if exp else None
    except Exception:
        return None


def parse_dhan_ts(value: Any) -> datetime | None:
    """Dhan timestamps are naive IST strings like ``2024-09-11 14:39:29``."""
    if not value:
        return None
    s = str(value)
    if s.startswith("0001"):
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=IST)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- rate limiter


@dataclass
class _Bucket:
    per_second: int
    per_day: int | None
    stamps: deque[float] = field(default_factory=deque)
    day: date | None = None
    count: int = 0


class RateLimiter:
    """Token-bucket-ish limiter matching Dhan's published caps (see README)."""

    DEFAULT_LIMITS: dict[str, tuple[int, int | None]] = {
        "order": (10, 7000),
        "data": (5, 100_000),
        "quote": (1, None),
        "nontrading": (20, None),
    }

    def __init__(
        self,
        limits: dict[str, tuple[int, int | None]] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        today: Callable[[], date] = lambda: datetime.now(IST).date(),
    ) -> None:
        self._buckets = {
            k: _Bucket(ps, pdy) for k, (ps, pdy) in (limits or self.DEFAULT_LIMITS).items()
        }
        self._clock = clock
        self._sleep = sleep
        self._today = today
        self._lock = asyncio.Lock()
        self.waits = 0

    async def acquire(self, category: str) -> None:
        b = self._buckets[category]
        async with self._lock:
            today = self._today()
            if b.day != today:
                b.day, b.count = today, 0
            if b.per_day is not None and b.count >= b.per_day:
                raise RateLimited(f"daily limit of {b.per_day} {category} requests reached")
            now = self._clock()
            self._prune(b, now)
            if len(b.stamps) >= b.per_second:
                wait = 1.0 - (now - b.stamps[0]) + 0.01
                self.waits += 1
                await self._sleep(max(wait, 0.0))
                now = self._clock()
                self._prune(b, now)
            b.stamps.append(now)
            b.count += 1

    @staticmethod
    def _prune(b: _Bucket, now: float) -> None:
        while b.stamps and now - b.stamps[0] >= 1.0:
            b.stamps.popleft()

    def usage(self, category: str) -> tuple[int, int | None]:
        b = self._buckets[category]
        return b.count, b.per_day


# --------------------------------------------------------------------------- REST client


class DhanClient:
    def __init__(
        self,
        cfg: DhanConfig,
        *,
        http: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.cfg = cfg
        self._token = cfg.token
        self._http = http or httpx.AsyncClient(timeout=cfg.timeout)
        self._owns_http = http is None
        self.limiter = limiter or RateLimiter()
        self._sleep = sleep

    @property
    def token(self) -> str:
        return self._token

    def set_token(self, token: str) -> None:
        self._token = token

    def headers(self) -> dict[str, str]:
        return {
            "access-token": self._token,
            "client-id": self.cfg.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def token_expiry(self) -> datetime | None:
        return jwt_expiry(self._token)

    async def request(
        self,
        method: str,
        path: str,
        *,
        category: str,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.cfg.base_url}{path}"
        delay = self.cfg.retry_base_delay
        for attempt in range(self.cfg.max_retries + 1):
            last = attempt >= self.cfg.max_retries
            await self.limiter.acquire(category)
            try:
                resp = await self._http.request(
                    method, url, json=json, params=params, headers=self.headers()
                )
            except httpx.TransportError as e:
                if last:
                    raise BrokerError(f"network error calling {path}: {e}") from e
                log.warning("dhan %s %s: %s (retry %d)", method, path, e, attempt + 1)
                await self._sleep(delay)
                delay *= 2
                continue
            body = self._body(resp)
            code, msg = extract_error(body)
            if resp.status_code == 429 or code in RATE_CODES:
                if last:
                    raise RateLimited(msg or "rate limited")
                retry_after = float(resp.headers.get("Retry-After") or 0) or delay
                log.warning("dhan rate limited on %s; retrying in %.1fs", path, retry_after)
                await self._sleep(retry_after)
                delay *= 2
                continue
            if resp.status_code >= 500 or code in RETRY_CODES:
                if last:
                    raise DhanError(code, msg or resp.text[:200], resp.status_code)
                await self._sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (401, 403) or code in AUTH_CODES:
                raise AuthError(f"{code or resp.status_code}: {msg or 'authentication failed'}")
            if resp.status_code >= 400 or code:
                raise DhanError(code, msg or resp.text[:200], resp.status_code)
            return body
        raise BrokerError("unreachable")  # pragma: no cover

    @staticmethod
    def _body(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text} if resp.text else {}

    async def renew_token(self) -> bool:
        """Renew an active token. Returns True if a new token was installed."""
        url = f"{self.cfg.base_url}/RenewToken"
        headers = {"access-token": self._token, "dhanClientId": self.cfg.client_id}
        await self.limiter.acquire("nontrading")
        resp = await self._http.request("POST", url, headers=headers)
        body = self._body(resp)
        code, msg = extract_error(body)
        if resp.status_code >= 400 or code:
            log.warning("token renewal failed: %s %s", code, msg)
            return False
        # TODO(dhan): response shape undocumented; accept the obvious key names
        new = (
            body.get("accessToken") or body.get("access_token") if isinstance(body, dict) else None
        )
        if new:
            self._token = str(new)
            log.info("access token renewed; new expiry %s", self.token_expiry())
            return True
        log.warning("token renewal returned no token; keeping the current one")
        return False

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


# --------------------------------------------------------------------------- historical parsing


def bars_from_chart(
    data: Any, symbol: str, interval: Interval, *, with_oi: bool
) -> tuple[list[Bar], int]:
    """Dhan chart arrays -> Bars. Returns (bars, rows_skipped)."""
    if not isinstance(data, dict):
        return [], 0
    stamps = data.get("timestamp") or []
    cols = {k: data.get(k) or [] for k in ("open", "high", "low", "close", "volume")}
    ois = data.get("open_interest") or []
    out: list[Bar] = []
    bad = 0
    for i, t in enumerate(stamps):
        ts = datetime.fromtimestamp(int(t), tz=IST)
        if interval is Interval.D1:
            ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        oi = int(ois[i]) if with_oi and i < len(ois) and ois[i] is not None else None
        try:
            out.append(
                Bar(
                    symbol=symbol,
                    ts=ts,
                    interval=interval,
                    open=float(cols["open"][i]),
                    high=float(cols["high"][i]),
                    low=float(cols["low"][i]),
                    close=float(cols["close"][i]),
                    volume=int(cols["volume"][i] or 0) if i < len(cols["volume"]) else 0,
                    oi=oi,
                )
            )
        except (ValidationError, ValueError, IndexError, TypeError):
            bad += 1
    if bad:
        log.warning("%s %s: skipped %d malformed candles", symbol, interval.value, bad)
    return out, bad


def intraday_windows(
    start: datetime, end: datetime, *, days: int = INTRADAY_WINDOW_DAYS
) -> list[tuple[datetime, datetime]]:
    out = []
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days) - timedelta(seconds=1))
        out.append((cur, nxt))
        cur = nxt + timedelta(seconds=1)
    return out


# --------------------------------------------------------------------------- feed packets

HEADER = struct.Struct("<BHBI")
_TICKER = struct.Struct("<fI")
_PREV_CLOSE = struct.Struct("<fI")
_OI = struct.Struct("<I")
_QUOTE = struct.Struct("<fHIfIIIffff")
_FULL = struct.Struct("<fHIfIIIIIIffff")
_DEPTH = struct.Struct("<IIHHff")
_DISCONNECT = struct.Struct("<H")

CODE_INDEX, CODE_TICKER, CODE_DEPTH, CODE_QUOTE = 1, 2, 3, 4
CODE_OI, CODE_PREV_CLOSE, CODE_STATUS, CODE_FULL, CODE_DISCONNECT = 5, 6, 7, 8, 50
PACKET_SIZE = {
    CODE_TICKER: 16,
    CODE_DEPTH: 112,
    CODE_QUOTE: 50,
    CODE_OI: 12,
    CODE_PREV_CLOSE: 16,
    CODE_STATUS: 8,
    CODE_FULL: 162,
    CODE_DISCONNECT: 10,
}


@dataclass(frozen=True)
class FeedPacket:
    code: int
    segment: int
    security_id: int
    data: dict[str, Any]


def _depth(body: bytes, offset: int) -> list[dict[str, float]]:
    levels = []
    for i in range(5):
        bq, aq, bo, ao, bp, ap = _DEPTH.unpack_from(body, offset + i * _DEPTH.size)
        levels.append(
            {"bid_qty": bq, "ask_qty": aq, "bid_orders": bo, "ask_orders": ao, "bid": bp, "ask": ap}
        )
    return levels


def _decode(code: int, body: bytes) -> dict[str, Any]:
    if code == CODE_TICKER:
        ltp, ltt = _TICKER.unpack_from(body)
        return {"ltp": ltp, "ltt": ltt}
    if code == CODE_PREV_CLOSE:
        pc, poi = _PREV_CLOSE.unpack_from(body)
        return {"prev_close": pc, "prev_oi": poi}
    if code == CODE_OI:
        (oi,) = _OI.unpack_from(body)
        return {"oi": oi}
    if code == CODE_QUOTE:
        ltp, ltq, ltt, atp, vol, tsq, tbq, o, c, h, lo = _QUOTE.unpack_from(body)
        return {
            "ltp": ltp, "ltq": ltq, "ltt": ltt, "atp": atp, "volume": vol,
            "total_sell_qty": tsq, "total_buy_qty": tbq,
            "open": o, "close": c, "high": h, "low": lo,
        }  # fmt: skip
    if code == CODE_FULL:
        ltp, ltq, ltt, atp, vol, tsq, tbq, oi, oi_hi, oi_lo, o, c, h, lo = _FULL.unpack_from(body)
        return {
            "ltp": ltp, "ltq": ltq, "ltt": ltt, "atp": atp, "volume": vol,
            "total_sell_qty": tsq, "total_buy_qty": tbq,
            "oi": oi, "oi_high": oi_hi, "oi_low": oi_lo,
            "open": o, "close": c, "high": h, "low": lo,
            "depth": _depth(body, _FULL.size),
        }  # fmt: skip
    if code == CODE_DEPTH:
        (ltp,) = struct.unpack_from("<f", body)
        return {"ltp": ltp, "depth": _depth(body, 4)}
    if code == CODE_DISCONNECT:
        (dc,) = _DISCONNECT.unpack_from(body)
        return {"disconnect_code": dc}
    return {}


def parse_packets(message: bytes) -> list[FeedPacket]:
    """Decode one websocket message (normally one packet; tolerates several)."""
    out: list[FeedPacket] = []
    pos, n = 0, len(message)
    while pos + HEADER.size <= n:
        code, length, seg, sid = HEADER.unpack_from(message, pos)
        size = PACKET_SIZE.get(code)
        if size is None:
            # TODO(dhan): index (1) and other undocumented layouts. The docs call the
            # header field the payload length; fall back to reading it as a total length.
            size = HEADER.size + length
            if pos + size > n:
                size = length
            if size < HEADER.size or pos + size > n:
                log.debug(
                    "unknown feed packet code=%s len=%s; dropping rest of message", code, length
                )
                break
            out.append(FeedPacket(code, seg, sid, {"raw": message[pos + HEADER.size : pos + size]}))
            pos += size
            continue
        if pos + size > n:
            log.debug("short feed packet code=%s (%d < %d bytes)", code, n - pos, size)
            break
        body = message[pos + HEADER.size : pos + size]
        out.append(FeedPacket(code, seg, sid, _decode(code, body)))
        pos += size
    return out


class FeedState:
    """Accumulates per-instrument state across packets and produces Ticks."""

    def __init__(self) -> None:
        self._state: dict[tuple[int, int], dict[str, Any]] = {}

    def apply(self, pkt: FeedPacket, symbol: str, now: datetime) -> Tick | None:
        st = self._state.setdefault((pkt.segment, pkt.security_id), {})
        d = pkt.data
        if pkt.code == CODE_OI:
            st["oi"] = d["oi"]
            return None
        if pkt.code == CODE_PREV_CLOSE:
            st["prev_close"], st["prev_oi"] = d["prev_close"], d["prev_oi"]
            return None
        if pkt.code not in (CODE_TICKER, CODE_QUOTE, CODE_FULL, CODE_DEPTH):
            return None
        st.update({k: v for k, v in d.items() if k != "depth"})
        depth = d.get("depth")
        if depth:
            top = depth[0]
            st["bid"] = top["bid"] if top["bid"] > 0 else None
            st["ask"] = top["ask"] if top["ask"] > 0 else None
            st["bid_qty"] = top["bid_qty"]
            st["ask_qty"] = top["ask_qty"]
        ltp = float(st.get("ltp") or 0.0)
        if ltp <= 0:
            return None
        ltt = int(st.get("ltt") or 0)
        ts = datetime.fromtimestamp(ltt, tz=IST) if ltt > 0 else now
        return Tick(
            symbol=symbol,
            ts=ts,
            ltp=ltp,
            ltq=int(st.get("ltq") or 0),
            volume=int(st.get("volume") or 0),
            bid=st.get("bid"),
            ask=st.get("ask"),
            bid_qty=st.get("bid_qty"),
            ask_qty=st.get("ask_qty"),
            oi=int(st["oi"]) if st.get("oi") is not None else None,
        )

    def snapshot(self, segment: int, security_id: int) -> dict[str, Any]:
        return dict(self._state.get((segment, security_id), {}))


# --------------------------------------------------------------------------- websocket feeds


def _backoff(attempt: int, base: float, cap: float) -> float:
    return min(cap, base * 2 ** max(attempt - 1, 0)) * (0.5 + random.random())


def _auth_failure(e: BaseException) -> bool:
    status = getattr(getattr(e, "response", None), "status_code", None)
    return status in (401, 403)


class DhanMarketFeed:
    """Binary market feed -> ``Tick`` stream with auto-reconnect and resubscribe."""

    def __init__(
        self,
        cfg: DhanConfig,
        symbols: SymbolMap,
        *,
        token: Callable[[], str] | None = None,
        connect: Callable[..., Any] = ws_connect,
        sleep: Sleep = asyncio.sleep,
        clock: Clock | None = None,
    ) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self._token = token or (lambda: cfg.token)
        self._connect = connect
        self._sleep = sleep
        self._clock = clock or SystemClock()
        self._closed = False
        self._ws: Any = None
        self.reconnects = 0
        self.disconnect_codes: list[int] = []
        self.state = FeedState()

    def url(self) -> str:
        return (
            f"{self.cfg.feed_url}?version=2&token={self._token()}"
            f"&clientId={self.cfg.client_id}&authType=2"
        )

    async def subscribe(self, ws: Any, instruments: Sequence[Instrument]) -> int:
        """Send subscribe messages in batches of 100. Returns the message count."""
        code = FEED_SUBSCRIBE[self.cfg.feed_mode]
        sent = 0
        for i in range(0, len(instruments), FEED_BATCH):
            batch = instruments[i : i + FEED_BATCH]
            msg = {
                "RequestCode": code,
                "InstrumentCount": len(batch),
                "InstrumentList": [
                    {"ExchangeSegment": x.broker_segment, "SecurityId": str(x.broker_id)}
                    for x in batch
                ],
            }
            await ws.send(json.dumps(msg))
            sent += 1
        return sent

    async def stream(self, symbols: Sequence[str]) -> AsyncIterator[Tick]:
        instruments = [self.symbols.resolve(s) for s in symbols]
        if len(instruments) > MAX_INSTRUMENTS_PER_CONNECTION:
            raise ValueError(
                f"max {MAX_INSTRUMENTS_PER_CONNECTION} instruments per feed connection"
            )
        names = {(SEGMENT_CODES[i.broker_segment], int(i.broker_id)): i.symbol for i in instruments}
        attempt = 0
        while not self._closed:
            try:
                async with self._connect(self.url(), ping_interval=None, max_size=None) as ws:
                    self._ws = ws
                    attempt = 0
                    await self.subscribe(ws, instruments)
                    async for message in ws:
                        if isinstance(message, str):
                            log.info("feed text message: %s", message[:200])
                            continue
                        for pkt in parse_packets(message):
                            if pkt.code == CODE_DISCONNECT:
                                code = pkt.data.get("disconnect_code")
                                self.disconnect_codes.append(int(code or 0))
                                log.warning("feed disconnect packet, code %s", code)
                                continue
                            sym = names.get((pkt.segment, pkt.security_id))
                            if sym is None:
                                continue
                            tick = self.state.apply(pkt, sym, self._clock.now())
                            if tick is not None:
                                yield tick
            except (ConnectionClosed, WebSocketException, OSError, TimeoutError) as e:
                if _auth_failure(e):
                    raise AuthError(f"market feed rejected credentials: {e}") from e
                if self._closed:
                    break
                attempt += 1
                self.reconnects += 1
                delay = _backoff(attempt, self.cfg.backoff_base, self.cfg.max_backoff)
                log.warning("market feed disconnected (%s); reconnecting in %.1fs", e, delay)
                await self._sleep(delay)
            finally:
                self._ws = None

    async def close(self) -> None:
        self._closed = True
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"RequestCode": FEED_DISCONNECT}))
            with contextlib.suppress(Exception):
                await ws.close()


def _segment_from_update(d: dict[str, Any]) -> str | None:
    return DHAN_SEGMENT.get((str(d.get("Exchange", "")), str(d.get("Segment", ""))))


def order_from_update(d: dict[str, Any], symbols: SymbolMap, now: datetime) -> Order:
    """Order-update websocket ``Data`` -> Order."""
    seg = _segment_from_update(d)
    sid = d.get("SecurityId")
    try:
        symbol = symbols.canonical(seg or "", sid or "")
    except UnknownSymbol:
        symbol = f"{d.get('Exchange', '?')}:{d.get('Symbol', sid)}"
    broker_id = str(d.get("OrderNo") or "")
    our_id = str(d.get("CorrelationId") or "") or f"dhan:{broker_id}"
    status = STATUS_FROM_DHAN.get(str(d.get("Status", "")).upper(), OrderStatus.PENDING)
    created = parse_dhan_ts(d.get("OrderDateTime")) or now
    updated = parse_dhan_ts(d.get("LastUpdatedTime")) or created
    avg = float(d.get("AvgTradedPrice") or 0)
    return Order(
        id=our_id,
        broker_order_id=broker_id,
        symbol=symbol,
        side=SIDE_FROM_DHAN.get(str(d.get("TxnType", "")).upper(), Side.BUY),
        qty=int(d.get("Quantity") or 1),
        filled_qty=int(d.get("TradedQty") or 0),
        order_type=ORDER_TYPE_FROM_DHAN.get(str(d.get("OrderType", "")).upper(), OrderType.LIMIT),
        product=PRODUCT_FROM_DHAN.get(str(d.get("Product", "")).upper(), ProductType.MIS),
        price=float(d["Price"]) if d.get("Price") else None,
        trigger_price=float(d["TriggerPrice"]) if d.get("TriggerPrice") else None,
        validity=Validity(str(d.get("Validity", "DAY")).upper()),
        status=status,
        avg_fill_price=avg or None,
        status_message=d.get("ReasonDescription") or d.get("Remarks"),
        created_at=created,
        updated_at=updated,
        meta={"exchange_order_id": d.get("ExchOrderNo"), "traded_price": d.get("TradedPrice")},
    )


class DhanOrderFeed:
    """JSON order-update websocket -> ``Order`` stream with auto-reconnect."""

    def __init__(
        self,
        cfg: DhanConfig,
        symbols: SymbolMap,
        *,
        token: Callable[[], str] | None = None,
        connect: Callable[..., Any] = ws_connect,
        sleep: Sleep = asyncio.sleep,
        clock: Clock | None = None,
    ) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self._token = token or (lambda: cfg.token)
        self._connect = connect
        self._sleep = sleep
        self._clock = clock or SystemClock()
        self._closed = False
        self._ws: Any = None
        self.reconnects = 0

    def login_message(self) -> str:
        return json.dumps(
            {
                "LoginReq": {"MsgCode": 42, "ClientId": self.cfg.client_id, "Token": self._token()},
                "UserType": "SELF",
            }
        )

    async def stream(self) -> AsyncIterator[Order]:
        attempt = 0
        while not self._closed:
            try:
                async with self._connect(self.cfg.order_update_url, max_size=None) as ws:
                    self._ws = ws
                    attempt = 0
                    await ws.send(self.login_message())
                    async for message in ws:
                        try:
                            payload = json.loads(message)
                        except (TypeError, ValueError):
                            log.debug("order feed: non-JSON message %r", message[:100])
                            continue
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("Type") != "order_alert":
                            log.info("order feed message: %s", str(payload)[:200])
                            continue
                        data = payload.get("Data") or {}
                        try:
                            yield order_from_update(data, self.symbols, self._clock.now())
                        except (ValidationError, ValueError) as e:
                            log.warning("order feed: cannot map update %s: %s", data, e)
            except (ConnectionClosed, WebSocketException, OSError, TimeoutError) as e:
                if _auth_failure(e):
                    raise AuthError(f"order feed rejected credentials: {e}") from e
                if self._closed:
                    break
                attempt += 1
                self.reconnects += 1
                delay = _backoff(attempt, self.cfg.backoff_base, self.cfg.max_backoff)
                log.warning("order feed disconnected (%s); reconnecting in %.1fs", e, delay)
                await self._sleep(delay)
            finally:
                self._ws = None

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()


# --------------------------------------------------------------------------- REST mappers


def order_payload(req: OrderRequest, inst: Instrument, client_id: str) -> dict[str, Any]:
    return {
        "dhanClientId": client_id,
        "correlationId": req.tag,
        "transactionType": req.side.value,
        "exchangeSegment": inst.broker_segment,
        "productType": PRODUCT_TO_DHAN[req.product],
        "orderType": ORDER_TYPE_TO_DHAN[req.order_type],
        "validity": req.validity.value,
        "securityId": inst.broker_id,
        "quantity": req.qty,
        "disclosedQuantity": req.disclosed_qty,
        "price": float(req.price or 0),
        "triggerPrice": float(req.trigger_price or 0),
        "afterMarketOrder": False,
    }


def _symbol_for(symbols: SymbolMap, d: dict[str, Any]) -> str:
    seg = str(d.get("exchangeSegment") or "")
    sid = str(d.get("securityId") or "")
    try:
        return symbols.canonical(seg, sid)
    except UnknownSymbol:
        return f"{seg}:{d.get('tradingSymbol') or sid}"


def order_from_book(d: dict[str, Any], symbols: SymbolMap, now: datetime) -> Order:
    broker_id = str(d.get("orderId") or "")
    our_id = str(d.get("correlationId") or "") or f"dhan:{broker_id}"
    created = parse_dhan_ts(d.get("createTime")) or now
    updated = parse_dhan_ts(d.get("updateTime")) or created
    avg = float(d.get("averageTradedPrice") or 0)
    return Order(
        id=our_id,
        broker_order_id=broker_id,
        symbol=_symbol_for(symbols, d),
        side=SIDE_FROM_DHAN[str(d.get("transactionType", "BUY")).upper()],
        qty=int(d.get("quantity") or 0),
        filled_qty=int(d.get("filledQty") or 0),
        order_type=ORDER_TYPE_FROM_DHAN.get(str(d.get("orderType", "")).upper(), OrderType.LIMIT),
        product=PRODUCT_FROM_DHAN.get(str(d.get("productType", "")).upper(), ProductType.MIS),
        price=float(d["price"]) if d.get("price") else None,
        trigger_price=float(d["triggerPrice"]) if d.get("triggerPrice") else None,
        validity=Validity(str(d.get("validity", "DAY")).upper()),
        status=STATUS_FROM_DHAN.get(str(d.get("orderStatus", "")).upper(), OrderStatus.PENDING),
        avg_fill_price=avg or None,
        status_message=d.get("omsErrorDescription") or None,
        created_at=created,
        updated_at=updated,
        meta={"exchange_time": d.get("exchangeTime"), "oms_error_code": d.get("omsErrorCode")},
    )


def fill_from_trade(
    d: dict[str, Any], symbols: SymbolMap, order_ids: dict[str, str], now: datetime
) -> Fill:
    broker_id = str(d.get("orderId") or "")
    symbol = _symbol_for(symbols, d)
    side = SIDE_FROM_DHAN[str(d.get("transactionType", "BUY")).upper()]
    product = PRODUCT_FROM_DHAN.get(str(d.get("productType", "")).upper(), ProductType.MIS)
    qty = int(d.get("tradedQuantity") or 0)
    price = float(d.get("tradedPrice") or 0)
    try:
        mult = symbols.resolve(symbol).multiplier
    except UnknownSymbol:
        mult = 1.0
    try:  # Dhan's trade book carries no charges: estimate with our fee model
        fees = compute_fees(symbol, side, qty, price, product, multiplier=mult)
    except ValueError:
        fees = None
    ts = (
        parse_dhan_ts(d.get("exchangeTime"))
        or parse_dhan_ts(d.get("updateTime"))
        or parse_dhan_ts(d.get("createTime"))
        or now
    )
    trade_id = str(d.get("exchangeTradeId") or "") or f"{broker_id}:{ts.isoformat()}"
    return Fill(
        id=f"dhan:{trade_id}",
        order_id=order_ids.get(broker_id, f"dhan:{broker_id}"),
        broker_order_id=broker_id,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        ts=ts,
        product=product,
        fees=fees if fees is not None else Fill.model_fields["fees"].default_factory(),  # type: ignore[misc]
        multiplier=mult,
    )


def position_from_dhan(d: dict[str, Any], symbols: SymbolMap) -> Position:
    symbol = _symbol_for(symbols, d)
    qty = int(d.get("netQty") or 0)
    avg = (
        float(d.get("buyAvg") or 0) if qty > 0 else float(d.get("sellAvg") or 0) if qty < 0 else 0.0
    )
    try:
        mult = symbols.resolve(symbol).multiplier
    except UnknownSymbol:
        mult = 1.0
    if d.get("multiplier"):
        mult = float(d["multiplier"])
    return Position(
        symbol=symbol,
        product=PRODUCT_FROM_DHAN.get(str(d.get("productType", "")).upper(), ProductType.MIS),
        qty=qty,
        avg_price=avg,
        realised_pnl=float(d.get("realizedProfit") or 0),
        multiplier=mult or 1.0,
    )


def funds_from_dhan(d: dict[str, Any]) -> Funds:
    cash = d.get("availabelBalance", d.get("availableBalance", 0))  # sic: Dhan's spelling
    return Funds(cash=float(cash or 0), margin_used=float(d.get("utilizedAmount") or 0))


# --------------------------------------------------------------------------- broker


class DhanBroker:
    name = "dhan"

    def __init__(
        self,
        cfg: DhanConfig,
        *,
        symbols: SymbolMap | None = None,
        instruments_dir: str = "data/instruments",
        http: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        ws_connect: Callable[..., Any] = ws_connect,
    ) -> None:
        self.cfg = cfg
        self.client = DhanClient(cfg, http=http, limiter=limiter, sleep=sleep)
        self.master = DhanInstrumentMaster(instruments_dir, http=http)
        self._symbols = symbols
        self.clock = clock or SystemClock()
        self._sleep = sleep
        self._ws_connect = ws_connect
        self._orders: dict[str, Order] = {}  # our id -> last known Order
        self._by_broker_id: dict[str, str] = {}  # broker order id -> our id
        self._feeds: list[DhanMarketFeed | DhanOrderFeed] = []

    # ------------------------------------------------------------------ lifecycle
    @property
    def symbols(self) -> SymbolMap:
        if self._symbols is None:
            raise NotConnected("DhanBroker.connect() has not been called")
        return self._symbols

    async def connect(self) -> None:
        if self._symbols is None:
            self._symbols = await self.master.symbol_map()
        exp = self.client.token_expiry()
        now = self.clock.now()
        if exp is not None:
            if exp <= now:
                raise AuthError(f"Dhan access token expired at {exp:%Y-%m-%d %H:%M} IST")
            if exp - now < timedelta(hours=self.cfg.renew_before_hours):
                log.info("access token expires at %s; attempting renewal", exp)
                await self.client.renew_token()
        await self.funds()  # cheapest authenticated call: proves token + connectivity

    async def close(self) -> None:
        for f in self._feeds:
            await f.close()
        self._feeds.clear()
        await self.client.aclose()

    # ------------------------------------------------------------------ market data
    async def instruments(self) -> list[Instrument]:
        return self.symbols.all()

    async def historical(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Bar]:
        inst = self.symbols.resolve(symbol)
        start, end = to_ist(start), to_ist(end)
        if start > end:
            return []
        base = {
            "securityId": inst.broker_id,
            "exchangeSegment": inst.broker_segment,
            "instrument": inst.broker_kind,
            "oi": inst.is_derivative,
        }
        bars: list[Bar] = []
        if interval is Interval.D1:
            body = base | {
                "fromDate": start.date().isoformat(),
                "toDate": (end.date() + timedelta(days=1)).isoformat(),  # exclusive
            }
            data = await self.client.request(
                "POST", "/charts/historical", category="data", json=body
            )
            bars, _ = bars_from_chart(data, inst.symbol, interval, with_oi=inst.is_derivative)
        else:
            minutes = INTERVAL_MINUTES.get(interval)
            if minutes is None:
                raise ValueError(f"Dhan intraday does not support {interval}")
            step = timedelta(seconds=interval.seconds)
            for lo, hi in intraday_windows(start, end):
                body = base | {
                    "interval": str(minutes),
                    "fromDate": lo.strftime("%Y-%m-%d %H:%M:%S"),
                    # TODO(dhan): inclusivity of toDate is undocumented; ask for one extra bar
                    "toDate": (hi + step).strftime("%Y-%m-%d %H:%M:%S"),
                }
                data = await self.client.request(
                    "POST", "/charts/intraday", category="data", json=body
                )
                got, _ = bars_from_chart(data, inst.symbol, interval, with_oi=inst.is_derivative)
                bars.extend(got)
        uniq = {b.ts: b for b in bars if start <= b.ts <= end}
        return [uniq[k] for k in sorted(uniq)]

    def subscribe_live(self, symbols: Sequence[str]) -> AsyncIterator[Tick]:
        feed = DhanMarketFeed(
            self.cfg,
            self.symbols,
            token=lambda: self.client.token,
            connect=self._ws_connect,
            sleep=self._sleep,
            clock=self.clock,
        )
        self._feeds.append(feed)
        return feed.stream(symbols)

    async def ltp(self, symbols: Sequence[str]) -> dict[str, float]:
        insts = [self.symbols.resolve(s) for s in symbols]
        out: dict[str, float] = {}
        for i in range(0, len(insts), QUOTE_BATCH):
            batch = insts[i : i + QUOTE_BATCH]
            body: dict[str, list[int]] = {}
            for inst in batch:
                body.setdefault(inst.broker_segment, []).append(int(inst.broker_id))
            data = await self.client.request("POST", "/marketfeed/ltp", category="quote", json=body)
            payload = data.get("data", {}) if isinstance(data, dict) else {}
            for inst in batch:
                row = (payload.get(inst.broker_segment) or {}).get(str(inst.broker_id))
                if isinstance(row, dict) and row.get("last_price"):
                    out[inst.symbol] = float(row["last_price"])
        return out

    # ------------------------------------------------------------------ orders
    def _remember(self, order: Order) -> Order:
        self._orders[order.id] = order
        if order.broker_order_id:
            self._by_broker_id[order.broker_order_id] = order.id
        return order

    async def place_order(self, req: OrderRequest) -> Order:
        if req.tag in self._orders:  # idempotent: same tag never places twice
            return self._orders[req.tag]
        now = self.clock.now()
        inst = self.symbols.resolve(req.symbol)
        if not inst.tradable:
            return self._remember(
                Order.from_request(
                    req, now, status=OrderStatus.REJECTED, status_message="index is not tradable"
                )
            )
        payload = order_payload(req, inst, self.cfg.client_id)
        try:
            resp = await self.client.request("POST", "/orders", category="order", json=payload)
        except DhanError as e:
            if e.code in INPUT_CODES:
                return self._remember(
                    Order.from_request(req, now, status=OrderStatus.REJECTED, status_message=str(e))
                )
            raise
        resp = resp if isinstance(resp, dict) else {}
        status = STATUS_FROM_DHAN.get(str(resp.get("orderStatus", "")).upper(), OrderStatus.PENDING)
        order = Order.from_request(
            req, now, broker_order_id=str(resp.get("orderId") or "") or None, status=status
        )
        return self._remember(order)

    async def _broker_id(self, order_id: str) -> str:
        cached = self._orders.get(order_id)
        if cached is not None and cached.broker_order_id:
            return cached.broker_order_id
        order = await self.order_status(order_id)
        if not order.broker_order_id:
            raise UnknownOrder(f"order {order_id} has no broker id")
        return order.broker_order_id

    async def modify_order(
        self,
        order_id: str,
        *,
        price: float | None = None,
        trigger_price: float | None = None,
        qty: int | None = None,
        order_type: OrderType | None = None,
    ) -> Order:
        current = await self.order_status(order_id)
        if not current.status.is_working:
            raise UnknownOrder(f"order {order_id} is {current.status}, cannot modify")
        assert current.broker_order_id
        new_type = order_type or current.order_type
        payload = {
            "dhanClientId": self.cfg.client_id,
            "orderId": current.broker_order_id,
            "orderType": ORDER_TYPE_TO_DHAN[new_type],
            "quantity": qty if qty is not None else current.qty,
            "price": float(price if price is not None else current.price or 0),
            "disclosedQuantity": 0,
            "triggerPrice": float(
                trigger_price if trigger_price is not None else current.trigger_price or 0
            ),
            "validity": current.validity.value,
        }
        await self.client.request(
            "PUT", f"/orders/{current.broker_order_id}", category="order", json=payload
        )
        return await self.order_status(order_id)

    async def cancel_order(self, order_id: str) -> Order:
        broker_id = await self._broker_id(order_id)
        resp = await self.client.request("DELETE", f"/orders/{broker_id}", category="order")
        order = self._orders.get(order_id) or await self.order_status(order_id)
        status = STATUS_FROM_DHAN.get(
            str((resp or {}).get("orderStatus", "")).upper() if isinstance(resp, dict) else "",
            OrderStatus.CANCELLED,
        )
        order.status = status
        order.updated_at = self.clock.now()
        return self._remember(order)

    async def order_status(self, order_id: str) -> Order:
        cached = self._orders.get(order_id)
        if cached is not None and cached.broker_order_id:
            data = await self.client.request(
                "GET", f"/orders/{cached.broker_order_id}", category="nontrading"
            )
        else:
            data = await self.client.request(
                "GET", f"/orders/external/{order_id}", category="nontrading"
            )
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict) or not data.get("orderId"):
            raise UnknownOrder(order_id)
        order = order_from_book(data, self.symbols, self.clock.now())
        if cached is not None:  # keep engine-side linkage
            order.intent_id, order.approval_token, order.parent_id = (
                cached.intent_id,
                cached.approval_token,
                cached.parent_id,
            )
        return self._remember(order)

    async def orders(self) -> list[Order]:
        data = await self.client.request("GET", "/orders", category="nontrading")
        rows = (
            data
            if isinstance(data, list)
            else (data or {}).get("data", [])
            if isinstance(data, dict)
            else []
        )
        now = self.clock.now()
        return [
            self._remember(order_from_book(r, self.symbols, now)) for r in rows if r.get("orderId")
        ]

    async def trades(self) -> list[Fill]:
        data = await self.client.request("GET", "/trades", category="nontrading")
        rows = (
            data
            if isinstance(data, list)
            else (data or {}).get("data", [])
            if isinstance(data, dict)
            else []
        )
        now = self.clock.now()
        return [
            fill_from_trade(r, self.symbols, self._by_broker_id, now)
            for r in rows
            if r.get("orderId")
        ]

    async def reconcile(self) -> tuple[list[Order], list[Fill]]:
        """Refresh the local order cache from the broker (constraint 4)."""
        orders = await self.orders()
        fills = await self.trades()
        return orders, fills

    async def positions(self) -> list[Position]:
        data = await self.client.request("GET", "/positions", category="nontrading")
        rows = (
            data
            if isinstance(data, list)
            else (data or {}).get("data", [])
            if isinstance(data, dict)
            else []
        )
        return [position_from_dhan(r, self.symbols) for r in rows]

    async def funds(self) -> Funds:
        data = await self.client.request("GET", "/fundlimit", category="nontrading")
        return funds_from_dhan(data if isinstance(data, dict) else {})

    def order_updates(self) -> AsyncIterator[Order]:
        feed = DhanOrderFeed(
            self.cfg,
            self.symbols,
            token=lambda: self.client.token,
            connect=self._ws_connect,
            sleep=self._sleep,
            clock=self.clock,
        )
        self._feeds.append(feed)
        return feed.stream()

    # ------------------------------------------------------------------ diagnostics
    def cached_orders(self) -> list[Order]:
        return list(self._orders.values())

    def token_status(self) -> dict[str, Any]:
        exp = self.client.token_expiry()
        return {
            "expires_at": exp.isoformat() if exp else None,
            "hours_left": round((exp - now_ist()).total_seconds() / 3600, 2) if exp else None,
        }
