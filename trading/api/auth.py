"""Authentication: users, sessions, tokens, CSRF, login throttling.

Single user today, multi-user ready: there is a ``users`` table with roles, and
every resource the API touches is scoped by user. The first user (``admin``) is
created from ``DASHBOARD_USER`` / ``DASHBOARD_PASSWORD`` on startup.

- **Passwords** are hashed with scrypt (stdlib), salted, compared in constant time.
- **Tokens** are ``payload.signature``: base64 JSON ``{uid, sid, exp}`` signed with
  HMAC-SHA256 under ``SESSION_SECRET``. Each carries a session id that must still
  exist in the ``sessions`` table, so logout (and an admin) can revoke it.
- The browser gets the token in an ``HttpOnly``, ``SameSite=Lax`` cookie; scripts
  send it as ``Authorization: Bearer``.
- **CSRF**: a state-changing request authenticated by the *cookie* must also carry
  ``X-CSRF-Token`` - an HMAC of the session id that pages put in a meta tag and
  HTMX sends on every request. A cross-site form can neither read nor set it.
  Bearer requests are exempt: browsers never attach them on their own.
- **Throttling**: five failed logins for a username lock it for a minute.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import Boolean, Float, String, select
from sqlalchemy.orm import Mapped, mapped_column

from trading.core.db import Base, make_engine, make_session_factory
from trading.core.types import now_ist

log = logging.getLogger(__name__)

COOKIE_NAME = "trading_session"
CSRF_HEADER = "X-CSRF-Token"
MAX_FAILURES = 5
LOCKOUT_SECONDS = 60


class Role(StrEnum):
    ADMIN = "admin"  # may trade live, promote and roll back models
    USER = "user"  # paper trading, backtests, own strategies


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[str] = mapped_column(String(40))


class SessionRow(Base):
    __tablename__ = "sessions"
    sid: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    expires_at: Mapped[float] = mapped_column(Float)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[str] = mapped_column(String(40))


@dataclass(frozen=True)
class User:
    id: str
    username: str
    role: Role

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


class AuthError(Exception):
    pass


# --------------------------------------------------------------------------- passwords


def hash_password(password: str, *, n: int = 2**14, r: int = 8, p: int = 1) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    candidate = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32
    )
    return hmac.compare_digest(candidate.hex(), digest)


# --------------------------------------------------------------------------- tokens


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenSigner:
    def __init__(self, secret: str) -> None:
        if len(secret) < 16:
            raise ValueError("session secret must be at least 16 characters")
        self._key = secret.encode()

    def _sig(self, payload: str) -> str:
        return _b64(hmac.new(self._key, payload.encode(), hashlib.sha256).digest())

    def sign(self, claims: dict[str, object]) -> str:
        payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        return f"{payload}.{self._sig(payload)}"

    def verify(self, token: str) -> dict[str, object]:
        try:
            payload, sig = token.split(".")
        except ValueError as e:
            raise AuthError("malformed token") from e
        if not hmac.compare_digest(sig, self._sig(payload)):
            raise AuthError("bad signature")
        claims = json.loads(_unb64(payload))
        if float(claims.get("exp", 0)) < time.time():
            raise AuthError("token expired")
        return claims

    def csrf_for(self, sid: str) -> str:
        return self._sig(f"csrf:{sid}")


# --------------------------------------------------------------------------- store


class AuthStore:
    def __init__(self, db_url: str, secret: str, *, ttl_hours: int = 12) -> None:
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine, tables=[UserRow.__table__, SessionRow.__table__])
        self._session = make_session_factory(self.engine)
        self.signer = TokenSigner(secret)
        self.ttl = ttl_hours * 3600
        self._failures: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str, role: Role = Role.USER) -> User:
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        user = User(id=secrets.token_hex(8), username=username, role=role)
        with self._session() as s, s.begin():
            if s.scalar(select(UserRow).where(UserRow.username == username)):
                raise ValueError(f"user {username!r} exists")
            s.add(
                UserRow(
                    id=user.id,
                    username=username,
                    password_hash=hash_password(password),
                    role=role.value,
                    active=True,
                    created_at=now_ist().isoformat(),
                )
            )
        return user

    def ensure_admin(self, username: str, password: str) -> User:
        """Create the first admin if there are no users yet; otherwise leave them alone."""
        existing = self.get_by_username(username)
        if existing is not None:
            return existing
        with self._session() as s:
            if s.scalar(select(UserRow).limit(1)) is not None:
                raise ValueError(f"users exist but none is called {username!r}")
        log.info("creating the first dashboard user %r", username)
        return self.create_user(username, password, Role.ADMIN)

    def set_password(self, username: str, password: str) -> None:
        """Replace a user's password and end every session they have open."""
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        with self._session() as s, s.begin():
            row = s.scalar(select(UserRow).where(UserRow.username == username))
            if row is None:
                raise KeyError(username)
            row.password_hash = hash_password(password)
            for session in s.scalars(select(SessionRow).where(SessionRow.user_id == row.id)):
                session.revoked = True
        self._failures.pop(username, None)
        log.warning("password changed for %r; their sessions were ended", username)

    def get_by_username(self, username: str) -> User | None:
        with self._session() as s:
            row = s.scalar(select(UserRow).where(UserRow.username == username))
            return _user(row) if row and row.active else None

    def get(self, user_id: str) -> User | None:
        with self._session() as s:
            row = s.get(UserRow, user_id)
            return _user(row) if row and row.active else None

    def users(self) -> list[User]:
        with self._session() as s:
            return [_user(r) for r in s.scalars(select(UserRow).order_by(UserRow.username))]

    # ------------------------------------------------------------------ sessions
    def locked(self, username: str) -> float:
        """Seconds until ``username`` may try again (0 when not locked)."""
        now = time.time()
        recent = [t for t in self._failures.get(username, []) if now - t < LOCKOUT_SECONDS]
        self._failures[username] = recent
        if len(recent) >= MAX_FAILURES:
            return LOCKOUT_SECONDS - (now - recent[0])
        return 0.0

    def login(self, username: str, password: str) -> tuple[User, str]:
        if self.locked(username):
            raise AuthError("too many failed attempts; try again shortly")
        with self._session() as s:
            row = s.scalar(select(UserRow).where(UserRow.username == username))
            ok = row is not None and row.active and verify_password(password, row.password_hash)
            if not ok:
                # spend the same time on unknown users, so timing does not reveal them
                if row is None:
                    verify_password(password, hash_password("x" * 12))
                self._failures.setdefault(username, []).append(time.time())
                raise AuthError("invalid username or password")
            user = _user(row)
        self._failures.pop(username, None)
        sid = secrets.token_hex(16)
        exp = time.time() + self.ttl
        with self._session() as s, s.begin():
            s.add(
                SessionRow(
                    sid=sid,
                    user_id=user.id,
                    expires_at=exp,
                    revoked=False,
                    created_at=now_ist().isoformat(),
                )
            )
        return user, self.signer.sign({"uid": user.id, "sid": sid, "exp": exp})

    def authenticate(self, token: str) -> tuple[User, str]:
        """(user, session id) for a valid, unrevoked token."""
        claims = self.signer.verify(token)
        sid = str(claims["sid"])
        with self._session() as s:
            row = s.get(SessionRow, sid)
            if row is None or row.revoked or row.expires_at < time.time():
                raise AuthError("session ended")
        user = self.get(str(claims["uid"]))
        if user is None:
            raise AuthError("user disabled")
        return user, sid

    def logout(self, sid: str) -> None:
        with self._session() as s, s.begin():
            row = s.get(SessionRow, sid)
            if row is not None:
                row.revoked = True

    def csrf_token(self, sid: str) -> str:
        return self.signer.csrf_for(sid)

    def check_csrf(self, sid: str, presented: str | None) -> bool:
        return presented is not None and hmac.compare_digest(presented, self.csrf_token(sid))


def _user(row: UserRow) -> User:
    return User(id=row.id, username=row.username, role=Role(row.role))
