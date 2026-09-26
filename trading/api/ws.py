"""``/ws/live``: engine events (orders, fills, alerts, rejections, signals, kill
switch) pushed to the signed-in user as JSON.

Browsers send the session cookie on a websocket handshake even from another site,
so a cookie-authenticated connection must come from our own origin
(cross-site websocket hijacking). Scripts can instead pass ``?token=`` or an
``Authorization: Bearer`` header.
"""

from __future__ import annotations

import asyncio
import contextlib
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from trading.api.auth import COOKIE_NAME, AuthError

router = APIRouter()


def _same_origin(ws: WebSocket) -> bool:
    origin = ws.headers.get("origin")
    if not origin:
        return False
    return urlparse(origin).netloc == ws.headers.get("host")


@router.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    svc = ws.app.state.services
    header = ws.headers.get("authorization", "")
    token = ws.query_params.get("token") or (
        header[7:] if header.lower().startswith("bearer ") else None
    )
    via_cookie = token is None
    if via_cookie:
        token = ws.cookies.get(COOKIE_NAME)
    try:
        if not token:
            raise AuthError("no credentials")
        if via_cookie and not _same_origin(ws):
            raise AuthError("cross-origin websocket refused")
        user, _ = svc.auth.authenticate(token)
    except AuthError:
        await ws.close(code=4401)
        return
    await ws.accept()
    queue = svc.hub.subscribe(user.id)
    await ws.send_json(
        {"type": "hello", "user": user.username, "kill_switch": svc.kill.state().__dict__}
    )

    async def pump() -> None:
        while True:
            await ws.send_json(await queue.get())

    sender = asyncio.create_task(pump())
    try:
        while True:  # the client sends nothing useful; this just notices it leaving
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await sender
        svc.hub.unsubscribe(user.id, queue)
