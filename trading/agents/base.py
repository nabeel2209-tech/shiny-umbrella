"""Common agent plumbing: bus subscriptions, heartbeats, alerts, lifecycle.

An agent is a small object that subscribes to a few bus topics and publishes to
others. Agents never call each other directly - the bus is the only coupling - so
the same agent runs unchanged against live Dhan, the paper broker or an archive
replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from pydantic import BaseModel

from trading.core.bus import Handler, MessageBus, Subscription, Topics
from trading.core.clock import Clock, SystemClock
from trading.core.types import Alert, AlertLevel, Heartbeat

log = logging.getLogger(__name__)


class Agent:
    """Base class. Subclasses override :meth:`on_start` to add subscriptions."""

    name: str = "agent"

    def __init__(self, bus: MessageBus, *, clock: Clock | None = None) -> None:
        self.bus = bus
        self.clock = clock or SystemClock()
        self.log = logging.getLogger(f"agent.{self.name}")
        self._subs: list[Subscription] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self.running = False
        self.errors = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await self.on_start()
        self.log.info("%s started", self.name)

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for sub in self._subs:
            await sub.cancel()
        self._subs.clear()
        await self.on_stop()
        self.log.info("%s stopped", self.name)

    async def on_start(self) -> None:
        """Subscribe to topics here."""

    async def on_stop(self) -> None:
        """Release resources here."""

    # ------------------------------------------------------------------ helpers
    async def subscribe(self, pattern: str, handler: Handler) -> Subscription:
        sub = await self.bus.subscribe(pattern, self._guard(handler))
        self._subs.append(sub)
        return sub

    def _guard(self, handler: Handler) -> Handler:
        """Wrap a handler so one bad message never kills an agent; it alerts instead."""

        async def wrapped(topic: str, message: BaseModel) -> None:
            try:
                await handler(topic, message)
            except Exception as e:
                self.errors += 1
                self.log.exception("%s failed handling %s", self.name, topic)
                await self.alert(AlertLevel.ERROR, f"{type(e).__name__}: {e}", topic=topic)

        return wrapped

    def spawn(self, coro: Any, *, name: str | None = None) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name or self.name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def publish(self, topic: str, message: BaseModel) -> None:
        await self.bus.publish(topic, message)

    async def alert(self, level: AlertLevel, message: str, **data: Any) -> Alert:
        alert = Alert(
            ts=self.clock.now(), level=level, source=self.name, message=message, data=data
        )
        await self.bus.publish(Topics.ALERTS, alert)
        return alert

    async def heartbeat(self, status: str = "ok", **data: Any) -> None:
        await self.bus.publish(
            Topics.HEARTBEAT,
            Heartbeat(ts=self.clock.now(), agent=self.name, status=status, data=data),
        )
