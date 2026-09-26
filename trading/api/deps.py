"""Request dependencies: services, the signed-in user, CSRF, admin checks."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from trading.api.auth import COOKIE_NAME, CSRF_HEADER, AuthError, User
from trading.api.services import Services

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class LoginRequired(Exception):
    """Raised by page routes; turned into a redirect to /login."""

    def __init__(self, next_url: str) -> None:
        self.next_url = next_url


def services(request: Request) -> Services:
    return request.app.state.services


def _token(request: Request) -> tuple[str | None, bool]:
    """(token, came from the cookie)."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip(), False
    cookie = request.cookies.get(COOKIE_NAME)
    return cookie, cookie is not None


async def _authenticate(request: Request, svc: Services) -> User | None:
    token, via_cookie = _token(request)
    if not token:
        return None
    try:
        user, sid = svc.auth.authenticate(token)
    except AuthError:
        return None
    request.state.user, request.state.sid, request.state.via_cookie = user, sid, via_cookie
    if via_cookie and request.method not in SAFE_METHODS:
        presented = request.headers.get(CSRF_HEADER)
        if presented is None and request.headers.get("content-type", "").startswith(
            ("application/x-www-form-urlencoded", "multipart/form-data")
        ):
            presented = (await request.form()).get("csrf_token")  # type: ignore[assignment]
        if not svc.auth.check_csrf(sid, presented):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "missing or wrong CSRF token")
    return user


async def current_user(request: Request, svc: Services = Depends(services)) -> User:
    user = await _authenticate(request, svc)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "sign in first", headers={"WWW-Authenticate": "Bearer"}
        )
    return user


async def page_user(request: Request, svc: Services = Depends(services)) -> User:
    user = await _authenticate(request, svc)
    if user is None:
        query = f"?{request.url.query}" if request.url.query else ""
        raise LoginRequired(request.url.path + query)
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user
