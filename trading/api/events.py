"""Live event fan-out to websocket clients, and the persistent kill switch."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading.core.types import now_ist

log = logging.getLogger(__name__)


class EventHub:
    """Per-user queues. A slow client drops its oldest events rather than stalling
    the engine that produces them."""

    def __init__(self, maxsize: int = 500) -> None:
        self.maxsize = maxsize
        self._queues: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self.published = 0

    def subscribe(self, user_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.maxsize)
        self._queues.setdefault(user_id, set()).add(q)
        return q

    def unsubscribe(self, user_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._queues.get(user_id, set()).discard(q)

    def publish(self, user_id: str | None, event: dict[str, Any]) -> None:
        """``user_id=None`` broadcasts to everyone (e.g. the kill switch)."""
        event = {"ts": now_ist().isoformat(), **event}
        targets = (
            [q for qs in self._queues.values() for q in qs]
            if user_id is None
            else list(self._queues.get(user_id, ()))
        )
        for q in targets:
            if q.full():
                q.get_nowait()
            q.put_nowait(event)
        self.published += 1

    def clients(self) -> int:
        return sum(len(qs) for qs in self._queues.values())


@dataclass
class KillState:
    killed: bool = False
    reason: str = ""
    by: str = ""
    at: str = ""


class KillSwitch:
    """Platform-wide halt that survives a restart.

    Anyone signed in can pull it; only an admin can release it. While it is
    pulled no engine may start, and every running engine is halted.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def state(self) -> KillState:
        if not self.path.exists():
            return KillState()
        try:
            return KillState(**json.loads(self.path.read_text()))
        except (OSError, ValueError, TypeError) as e:
            # fail closed: a switch we cannot read is a switch that is on
            log.error("kill switch file %s unreadable (%s): treating it as engaged", self.path, e)
            return KillState(True, f"kill switch file unreadable: {e}", "system", "")

    @property
    def engaged(self) -> bool:
        return self.state().killed

    def engage(self, reason: str, by: str) -> KillState:
        return self._write(KillState(True, reason, by, now_ist().isoformat()))

    def release(self, by: str) -> KillState:
        return self._write(KillState(False, "", by, now_ist().isoformat()))

    def _write(self, state: KillState) -> KillState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.__dict__))
        tmp.replace(self.path)
        log.warning(
            "kill switch %s by %s: %s",
            "ENGAGED" if state.killed else "released",
            state.by,
            state.reason,
        )
        return state
