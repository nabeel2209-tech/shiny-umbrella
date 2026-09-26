"""``/ws/live``: who may connect, and what they receive."""

from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.apikit import ALWAYS_LONG, Api

ORIGIN = {"origin": "http://testserver"}


@pytest.fixture
def api(tmp_path):
    with Api(tmp_path) as a:
        yield a


def refused(api, url, headers=None):
    with (
        pytest.raises(WebSocketDisconnect) as e,
        api.client.websocket_connect(url, headers=dict(headers or {})),
    ):
        pass
    return e.value.code


def test_no_credentials_refused(api):
    assert refused(api, "/ws/live", ORIGIN) == 4401
    assert refused(api, "/ws/live?token=forged", ORIGIN) == 4401


def test_cookie_from_another_site_refused(api):
    """A page on another origin can make the browser send our cookie; the Origin
    check stops cross-site websocket hijacking."""
    api.login()
    assert refused(api, "/ws/live", {"origin": "https://evil.example"}) == 4401
    assert refused(api, "/ws/live", {}) == 4401  # no Origin at all


def test_same_origin_cookie_connects_and_gets_hello(api):
    api.login()
    with api.client.websocket_connect("/ws/live", headers=ORIGIN) as ws:
        hello = ws.receive_json()
    assert hello["type"] == "hello" and hello["user"] == "admin"
    assert hello["kill_switch"]["killed"] is False


def test_token_query_and_bearer_header(api):
    token = api.login()["token"]
    api.client.cookies.clear()
    with api.client.websocket_connect(f"/ws/live?token={token}") as ws:
        assert ws.receive_json()["type"] == "hello"
    with api.client.websocket_connect(
        "/ws/live", headers={"Authorization": f"Bearer {token}"}
    ) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_revoked_session_refused(api):
    body = api.login()
    headers = {"Authorization": f"Bearer {body['token']}"}
    api.client.post("/api/auth/logout", headers=headers)
    assert refused(api, f"/ws/live?token={body['token']}") == 4401


def test_kill_switch_events_reach_everyone(api):
    admin_token = api.login()["token"]
    api.client.cookies.clear()
    bob = api.add_user("bob")
    with api.client.websocket_connect(f"/ws/live?token={admin_token}") as ws:
        ws.receive_json()  # hello
        api.client.post("/api/control/kill", json={"reason": "ws test"}, headers=bob)
        event = ws.receive_json()
    assert event["type"] == "kill" and event["data"]["reason"] == "ws test"
    assert event["data"]["by"] == "bob" and event["ts"]


def test_engine_events_reach_only_their_owner(api):
    admin = api.bearer()
    admin_token = admin["Authorization"][7:]
    bob = api.add_user("bob")
    bob_token = bob["Authorization"][7:]
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=admin)
    with (
        api.client.websocket_connect(f"/ws/live?token={admin_token}") as mine,
        api.client.websocket_connect(f"/ws/live?token={bob_token}") as theirs,
    ):
        mine.receive_json(), theirs.receive_json()  # hellos
        r = api.client.post(
            "/api/engines/paper/start", json={"strategy_ids": ["always_long"]}, headers=admin
        )
        assert r.status_code == 200
        started = mine.receive_json()
        assert started["type"] == "engine" and started["mode"] == "paper"
        assert started["data"]["state"] == "started"
        api.client.post("/api/engines/paper/stop", headers=admin)
        seen = []
        while not seen or seen[-1]["type"] != "engine":
            seen.append(mine.receive_json())
        assert seen[-1]["data"]["state"] == "stopped"
        # the engine's own alerts come through the bridge too (no warmup history here)
        assert any(e["type"] == "alerts" and "warmup" in e["data"]["message"] for e in seen)
        # bob's socket saw none of it: the next thing he gets is a global event
        api.client.post("/api/control/kill", json={"reason": "probe"}, headers=bob)
        assert theirs.receive_json()["type"] == "kill"
    assert api.services.hub.clients() == 0  # both unsubscribed on disconnect
