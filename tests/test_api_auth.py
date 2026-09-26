"""Sign-in, sessions, CSRF, throttling, and the security headers."""

from __future__ import annotations

import pytest

from tests.apikit import PASSWORD, Api
from trading.api.auth import (
    MAX_FAILURES,
    AuthError,
    AuthStore,
    Role,
    TokenSigner,
    hash_password,
    verify_password,
)


@pytest.fixture
def api(tmp_path):
    with Api(tmp_path) as a:
        yield a


# --------------------------------------------------------------------------- units


def test_password_hash_round_trip_and_salt():
    h1, h2 = hash_password("hunter2hunter2"), hash_password("hunter2hunter2")
    assert h1 != h2  # salted
    assert h1.startswith("scrypt$")
    assert verify_password("hunter2hunter2", h1)
    assert not verify_password("hunter2hunter3", h1)
    assert not verify_password("anything", "garbage")


def test_token_signer_rejects_tampering_and_short_secrets():
    signer = TokenSigner("s" * 32)
    token = signer.sign({"sid": "abc", "exp": 4e9})
    assert signer.verify(token)["sid"] == "abc"
    for bad in (token[:-2] + "xx", "no-dot", TokenSigner("t" * 32).sign({"sid": "abc"})):
        with pytest.raises(AuthError):
            signer.verify(bad)
    with pytest.raises(AuthError, match="expired"):
        signer.verify(signer.sign({"sid": "abc", "exp": 1}))
    with pytest.raises(ValueError):
        TokenSigner("short")


def test_store_sessions_expire_and_revoke(tmp_path):
    store = AuthStore(f"sqlite:///{tmp_path / 'a.db'}", "s" * 32, ttl_hours=1)
    store.create_user("alice", "long enough password", role=Role.ADMIN)
    with pytest.raises(ValueError):
        store.create_user("bob", "short")
    _, token = store.login("alice", "long enough password")
    assert store.authenticate(token)[0].username == "alice"
    sid = store.authenticate(token)[1]
    store.logout(sid)
    with pytest.raises(AuthError):
        store.authenticate(token)
    expired = AuthStore(f"sqlite:///{tmp_path / 'a.db'}", "s" * 32, ttl_hours=0)
    _, token2 = expired.login("alice", "long enough password")
    with pytest.raises(AuthError):
        expired.authenticate(token2)


def test_ensure_admin_does_not_reset_an_existing_password(tmp_path):
    store = AuthStore(f"sqlite:///{tmp_path / 'a.db'}", "s" * 32)
    store.ensure_admin("admin", "first password!")
    store.ensure_admin("admin", "second password!")
    store.login("admin", "first password!")
    with pytest.raises(AuthError):
        store.login("admin", "second password!")


# --------------------------------------------------------------------------- routes


def test_healthz_is_public(api):
    r = api.client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "kill_switch": False, "engines": 0}


def test_security_headers(api):
    r = api.client.get("/healthz")
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_login_me_logout(api):
    body = api.login()
    assert body["user"]["username"] == "admin" and body["user"]["role"] == "admin"
    assert body["csrf_token"]
    cookie = api.client.cookies.get("trading_session")
    assert cookie == body["token"]
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert api.client.get("/api/auth/me", headers=headers).json()["username"] == "admin"
    r = api.client.post("/api/auth/logout", headers=headers)
    assert r.status_code == 200
    api.client.cookies.clear()
    assert api.client.get("/api/auth/me", headers=headers).status_code == 401


def test_login_cookie_flags(api):
    r = api.client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie


def test_wrong_password_and_throttle(api):
    for _ in range(MAX_FAILURES):
        r = api.client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
        assert r.status_code == 401
    r = api.client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 401
    assert "too many" in r.json()["detail"]


def test_api_requires_auth(api):
    for path in ("/api/auth/me", "/api/strategies", "/api/engines", "/api/models"):
        r = api.client.get(path)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "Bearer"
    bad = {"Authorization": "Bearer not-a-token"}
    assert api.client.get("/api/auth/me", headers=bad).status_code == 401


def test_cookie_writes_need_csrf_but_bearer_writes_do_not(api):
    body = api.login()  # the client now holds the cookie
    r = api.client.post("/api/control/kill", json={"reason": "no csrf"})
    assert r.status_code == 403
    r = api.client.post(
        "/api/control/kill", json={"reason": "wrong"}, headers={"X-CSRF-Token": "forged"}
    )
    assert r.status_code == 403
    assert not api.services.kill.engaged
    r = api.client.post(
        "/api/control/kill", json={"reason": "right"}, headers={"X-CSRF-Token": body["csrf_token"]}
    )
    assert r.status_code == 200
    api.client.cookies.clear()
    r = api.client.post("/api/control/resume", headers={"Authorization": f"Bearer {body['token']}"})
    assert r.status_code == 200
    # reads never need the token
    api.client.cookies.set("trading_session", body["token"])
    assert api.client.get("/api/control").status_code == 200


def test_csrf_token_is_bound_to_the_session(api):
    first = api.login()["csrf_token"]
    second = api.login()
    assert first != second["csrf_token"]
    r = api.client.post("/api/control/kill", json={}, headers={"X-CSRF-Token": first})
    assert r.status_code == 403
