"""Internal message bus.

Topics:
    bars.<symbol>   Bar
    ticks.<symbol>  Tick
    signals         Signal
    intents         OrderIntent          (signal agents -> risk agent)
    approved        RiskApproval         (risk agent -> execution agent)
    rejected        RiskRejection
    orders          Order                (execution agent -> everyone)
    fills           Fill
    alerts          Alert
    heartbeat       Heartbeat
    control         ControlCommand       (kill switch etc.)

Two implementations with the same interface:
- ``InMemoryBus``: for tests and backtests. Dispatches handlers *synchronously*
  (awaited in publish order) so runs are deterministic. ``strict=True`` round-trips
  every message through JSON so a message that would break on Redis breaks here too.
- ``RedisBus``: Redis pub/sub with glob pattern subscriptions.

Messages are pydantic models wrapped in an ``Envelope``; the registry maps the
model name back to its class on decode.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from trading.core.types import (
    Alert,
    Bar,
    ControlCommand,
    Fill,
    Heartbeat,
    Order,
    OrderIntent,
    RiskApproval,
    RiskRejection,
    Signal,
    Tick,
    new_id,
    now_ist,
)

log = logging.getLogger(__name__)


class Topics:
    SIGNALS = "signals"
    INTENTS = "intents"
    APPROVED = "approved"
    REJECTED = "rejected"
    ORDERS = "orders"
    FILLS = "fills"
    ALERTS = "alerts"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"
    BARS_ALL = "bars.*"
    TICKS_ALL = "ticks.*"

    @staticmethod
    def bars(symbol: str) -> str:
        return f"bars.{symbol}"

    @staticmethod
    def ticks(symbol: str) -> str:
        return f"ticks.{symbol}"


# --------------------------------------------------------------------------- envelope


class Envelope(BaseModel):
    id: str = Field(default_factory=new_id)
    topic: str
    type: str
    ts: str
    payload: dict[str, Any]


_REGISTRY: dict[str, type[BaseModel]] = {}


def register_message_type(cls: type[BaseModel]) -> type[BaseModel]:
    _REGISTRY[cls.__name__] = cls
    return cls


for _cls in (
    Bar,
    Tick,
    Signal,
    OrderIntent,
    RiskApproval,
    RiskRejection,
    Order,
    Fill,
    Alert,
    Heartbeat,
    ControlCommand,
):
    register_message_type(_cls)


def encode(topic: str, message: BaseModel) -> bytes:
    name = type(message).__name__
    if name not in _REGISTRY:
        raise TypeError(f"{name} is not a registered bus message type")
    env = Envelope(
        topic=topic,
        type=name,
        ts=now_ist().isoformat(),
        payload=message.model_dump(mode="json"),
    )
    return env.model_dump_json().encode()


def decode(raw: bytes | str) -> tuple[str, BaseModel]:
    env = Envelope.model_validate_json(raw)
    cls = _REGISTRY.get(env.type)
    if cls is None:
        raise TypeError(f"unknown bus message type {env.type!r}")
    return env.topic, cls.model_validate(env.payload)


# --------------------------------------------------------------------------- interface

Handler = Callable[[str, BaseModel], Awaitable[None]]


class Subscription:
    def __init__(self, pattern: str, handler: Handler, cancel: Callable[[], Awaitable[None]]):
        self.pattern = pattern
        self.handler = handler
        self._cancel = cancel
        self.active = True

    async def cancel(self) -> None:
        if self.active:
            self.active = False
            await self._cancel()


@runtime_checkable
class MessageBus(Protocol):
    async def publish(self, topic: str, message: BaseModel) -> None: ...

    async def subscribe(self, pattern: str, handler: Handler) -> Subscription: ...

    async def close(self) -> None: ...


def topic_matches(pattern: str, topic: str) -> bool:
    return fnmatch.fnmatchcase(topic, pattern)


# --------------------------------------------------------------------------- in-memory


class InMemoryBus:
    """Deterministic in-process bus. Handlers are awaited in subscription order."""

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self._subs: list[Subscription] = []
        self.published: int = 0
        self.errors: int = 0

    async def publish(self, topic: str, message: BaseModel) -> None:
        if self.strict:
            topic, message = decode(encode(topic, message))
        self.published += 1
        for sub in list(self._subs):
            if sub.active and topic_matches(sub.pattern, topic):
                try:
                    await sub.handler(topic, message)
                except Exception:
                    self.errors += 1
                    log.exception("handler for %s failed on topic %s", sub.pattern, topic)

    async def subscribe(self, pattern: str, handler: Handler) -> Subscription:
        sub = Subscription(pattern, handler, cancel=lambda: self._remove(pattern, handler))
        self._subs.append(sub)
        return sub

    async def _remove(self, pattern: str, handler: Handler) -> None:
        self._subs = [s for s in self._subs if not (s.pattern == pattern and s.handler is handler)]

    async def close(self) -> None:
        self._subs.clear()


# --------------------------------------------------------------------------- redis


class RedisBus:
    """Redis pub/sub bus. One reader task per subscription."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self.url = url
        self._redis = aioredis.from_url(url)
        self._tasks: set[asyncio.Task[None]] = set()

    async def publish(self, topic: str, message: BaseModel) -> None:
        await self._redis.publish(topic, encode(topic, message))

    async def subscribe(self, pattern: str, handler: Handler) -> Subscription:
        pubsub = self._redis.pubsub()
        await pubsub.psubscribe(pattern)

        async def reader() -> None:
            try:
                async for msg in pubsub.listen():
                    if msg.get("type") != "pmessage":
                        continue
                    try:
                        topic, message = decode(msg["data"])
                        await handler(topic, message)
                    except Exception:
                        log.exception("handler for %s failed", pattern)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(reader(), name=f"redis-sub:{pattern}")
        self._tasks.add(task)

        async def cancel() -> None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._tasks.discard(task)
            with contextlib.suppress(Exception):
                await pubsub.punsubscribe(pattern)
                await pubsub.aclose()

        return Subscription(pattern, handler, cancel)

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        await self._redis.aclose()


def make_bus(redis_url: str | None = None, *, strict: bool = True) -> MessageBus:
    """Redis bus when a URL is configured, otherwise the in-memory bus."""
    if redis_url:
        return RedisBus(redis_url)
    return InMemoryBus(strict=strict)
