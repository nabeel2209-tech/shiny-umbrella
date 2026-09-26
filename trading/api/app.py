"""The FastAPI application: JSON API, websocket, and the HTMX dashboard."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from trading import __version__
from trading.api.deps import LoginRequired
from trading.api.routes.api import gated, public, router
from trading.api.routes.pages import render
from trading.api.routes.pages import router as pages
from trading.api.services import Services, build_services
from trading.api.ws import router as ws_router
from trading.core.config import Settings, get_settings

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"

# Everything the pages load is a file from this origin: no inline scripts or styles,
# no CDN. htmx is vendored under /static.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Cache-Control", "no-store")
        return response


def create_app(
    settings: Settings | None = None,
    *,
    services: Services | None = None,
    admin: tuple[str, str] | None = None,
) -> FastAPI:
    """``admin`` = (username, password) for the first user; defaults to
    DASHBOARD_USER / DASHBOARD_PASSWORD. Existing users are left alone."""
    settings = settings or get_settings()
    svc = services or build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        username, password = admin or (
            settings.dashboard_user,
            settings.dashboard_password.get_secret_value(),
        )
        if svc.auth.get_by_username(username) is None:
            if not password:
                raise RuntimeError("set DASHBOARD_PASSWORD to create the first dashboard user")
            svc.auth.ensure_admin(username, password)
        svc.jobs.start()
        log.info("API up: %d users, kill switch %s", len(svc.auth.users()), svc.kill.engaged)
        yield
        await svc.engines.shutdown()
        await svc.jobs.stop()

    app = FastAPI(title="Trading platform", version=__version__, lifespan=lifespan)
    app.state.services = svc
    app.add_middleware(SecurityHeaders)
    app.include_router(public)
    app.include_router(router)
    if settings.feature_marketplace:
        app.include_router(gated)
    app.include_router(ws_router)
    app.include_router(pages)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(LoginRequired)
    async def to_login(request: Request, exc: LoginRequired) -> Response:
        if request.headers.get("HX-Request"):
            # htmx would swap the login page into a panel: send the whole page there,
            # and come back to the page the panel was on, not to the fragment
            current = urlparse(request.headers.get("HX-Current-URL", ""))
            back = current.path + (f"?{current.query}" if current.query else "")
            target = f"/login?next={quote(back or '/')}"
            return Response(status_code=204, headers={"HX-Redirect": target})
        return RedirectResponse(f"/login?next={quote(exc.next_url)}", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """JSON for the API and htmx; a readable page for a browser."""
        path = request.url.path
        wants_page = (
            not path.startswith(("/api/", "/ui/", "/ws/", "/static/"))
            and not request.headers.get("HX-Request")
            and "text/html" in request.headers.get("accept", "")
        )
        if not wants_page:
            return await http_exception_handler(request, exc)
        headings = {404: "Not found", 403: "Not allowed", 401: "Sign in required"}
        return render(
            request,
            "error.html",
            status_code=exc.status_code,
            heading=headings.get(exc.status_code, "Something went wrong"),
            detail=exc.detail if isinstance(exc.detail, str) else "",
        )

    return app
