"""Every JSON route: strategies, backtests, engines, accounts, control, models, and
the SEBI-gated stubs."""

from __future__ import annotations

import time

import pytest

from tests.apikit import ALWAYS_LONG, SYM, TREND_YAML, Api, NotDhan, write_archive
from trading.core.types import Position, ProductType
from trading.training.registry import ModelRegistry


@pytest.fixture
def api(tmp_path):
    with Api(tmp_path) as a:
        yield a


@pytest.fixture
def admin(api):
    return api.bearer()


def create(api, headers, body=None):
    r = api.client.post("/api/strategies", json=body or ALWAYS_LONG, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- strategies


def test_features_lists_names_ops_and_intervals(api, admin):
    body = api.client.get("/api/features", headers=admin).json()
    assert "trend" in body["intraday"] and "rsi_14" in body["intraday"]
    assert "gt" in body["ops"] and "5m" in body["intervals"]


def test_strategy_crud(api, admin):
    assert api.client.get("/api/strategies", headers=admin).json() == []
    created = create(api, admin)
    assert created["id"] == "always_long"
    assert [s["id"] for s in api.client.get("/api/strategies", headers=admin).json()] == [
        "always_long"
    ]
    got = api.client.get("/api/strategies/always_long", headers=admin).json()
    assert got["sizing"]["qty"] == 10 and "always_long" in got["yaml"]

    r = api.client.post("/api/strategies", json=ALWAYS_LONG, headers=admin)
    assert r.status_code == 409

    changed = ALWAYS_LONG | {"name": "Renamed"}
    r = api.client.put("/api/strategies/always_long", json=changed, headers=admin)
    assert r.status_code == 200 and r.json()["name"] == "Renamed"
    r = api.client.put("/api/strategies/always_long", json=ALWAYS_LONG | {"id": "x"}, headers=admin)
    assert r.status_code == 422  # the id in the body must match the URL
    r = api.client.put("/api/strategies/missing", json=ALWAYS_LONG, headers=admin)
    assert r.status_code == 404

    r = api.client.delete("/api/strategies/always_long", headers=admin)
    assert r.status_code == 200 and ".trash" in r.json()["recoverable_at"]
    assert api.client.get("/api/strategies/always_long", headers=admin).status_code == 404
    assert api.client.delete("/api/strategies/always_long", headers=admin).status_code == 404


def test_create_from_yaml_and_validation_errors(api, admin):
    r = api.client.post("/api/strategies", json={"yaml": TREND_YAML}, headers=admin)
    assert r.status_code == 201 and r.json()["id"] == "trend_test"

    bad = ALWAYS_LONG | {
        "id": "bad",
        "rules": {"long": {"all": [{"feature": "nope", "op": "gt", "value": 1}]}},
    }
    r = api.client.post("/api/strategies", json=bad, headers=admin)
    assert r.status_code == 422
    assert "unknown feature: nope" in r.json()["detail"]["errors"]

    r = api.client.post("/api/strategies", json={"yaml": "id: [unclosed"}, headers=admin)
    assert r.status_code == 422

    r = api.client.post("/api/strategies", json=ALWAYS_LONG | {"symbols": []}, headers=admin)
    assert r.status_code == 422


def test_validate_does_not_save(api, admin):
    ok = api.client.post("/api/strategies/validate", json=ALWAYS_LONG, headers=admin).json()
    assert ok["valid"] and ok["errors"] == [] and "always_long" in ok["yaml"]
    bad = api.client.post(
        "/api/strategies/validate", json={"yaml": "id: x\nsymbols: []\n"}, headers=admin
    ).json()
    assert not bad["valid"] and bad["errors"] and bad["yaml"] is None
    assert api.client.get("/api/strategies", headers=admin).json() == []


def test_templates_are_the_shipped_examples(api, admin):
    ids = {t["id"] for t in api.client.get("/api/strategies/templates", headers=admin).json()}
    assert {"trend_reliance", "goldm_meanrev"} <= ids


def test_strategies_are_per_user(api, admin):
    create(api, admin)
    bob = api.add_user("bob")
    assert api.client.get("/api/strategies", headers=bob).json() == []
    assert api.client.get("/api/strategies/always_long", headers=bob).status_code == 404
    create(api, bob)  # same id, different user: no clash


# --------------------------------------------------------------------------- backtests


def test_backtest_job_lifecycle(api, admin, calendar):
    days = write_archive(api.settings.archive_dir, calendar)
    create(api, admin)
    r = api.client.post(
        "/api/backtests",
        json={"strategy_ids": ["always_long"], "start": str(days[0]), "end": str(days[-1])},
        headers=admin,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    assert r.json()["status"] in ("queued", "running")

    job = api.wait_for_job(job_id, admin)
    assert job["status"] == "done", job
    assert job["summary"]["metrics"]["trades"] >= 1
    assert job["summary"]["config"]["strategies"] == ["always_long"]

    equity = api.client.get(f"/api/backtests/{job_id}/equity", headers=admin).json()
    assert equity and {"ts", "equity"} <= set(equity[0])
    trades = api.client.get(f"/api/backtests/{job_id}/trades", headers=admin).json()
    assert trades and trades[0]["symbol"] == SYM

    listed = api.client.get("/api/backtests", headers=admin).json()
    assert [j["id"] for j in listed] == [job_id]


def test_backtest_errors(api, admin):
    body = {"strategy_ids": ["nope"], "start": "2026-09-01", "end": "2026-09-02"}
    assert api.client.post("/api/backtests", json=body, headers=admin).status_code == 404
    create(api, admin)
    backwards = {"strategy_ids": ["always_long"], "start": "2026-09-10", "end": "2026-09-01"}
    assert api.client.post("/api/backtests", json=backwards, headers=admin).status_code == 422
    too_long = {"strategy_ids": ["always_long"], "start": "2020-01-01", "end": "2026-09-01"}
    assert api.client.post("/api/backtests", json=too_long, headers=admin).status_code == 422
    assert api.client.get("/api/backtests/nope", headers=admin).status_code == 404


def test_backtest_without_data_fails_and_has_no_results(api, admin):
    create(api, admin)  # nothing archived
    body = {"strategy_ids": ["always_long"], "start": "2026-09-14", "end": "2026-09-15"}
    job = api.wait_for_job(
        api.client.post("/api/backtests", json=body, headers=admin).json()["id"], admin
    )
    assert job["status"] == "failed" and job["error"]
    for part in ("equity", "trades"):
        r = api.client.get(f"/api/backtests/{job['id']}/{part}", headers=admin)
        assert r.status_code == 409 and r.json()["detail"] == "backtest is failed"


def test_backtest_of_a_derivative_without_lot_sizes_fails_loudly(api, admin, calendar):
    days = write_archive(api.settings.archive_dir, calendar)
    fut = ALWAYS_LONG | {"id": "fut", "symbols": ["NFO:NIFTY-OCT26"], "product": "NRML"}
    create(api, admin, fut)
    r = api.client.post(
        "/api/backtests",
        json={"strategy_ids": ["fut"], "start": str(days[0]), "end": str(days[-1])},
        headers=admin,
    )
    job = api.wait_for_job(r.json()["id"], admin)
    assert job["status"] == "failed"
    assert "lot size" in job["error"].lower()


def test_backtests_are_per_user(api, admin, calendar):
    days = write_archive(api.settings.archive_dir, calendar)
    create(api, admin)
    r = api.client.post(
        "/api/backtests",
        json={"strategy_ids": ["always_long"], "start": str(days[0]), "end": str(days[-1])},
        headers=admin,
    )
    bob = api.add_user("bob")
    assert api.client.get(f"/api/backtests/{r.json()['id']}", headers=bob).status_code == 404
    assert api.client.get("/api/backtests", headers=bob).json() == []
    api.wait_for_job(r.json()["id"], admin)


# --------------------------------------------------------------------------- engines


def start(api, headers, mode="paper", ids=("always_long",), confirm=None):
    body = {"strategy_ids": list(ids)}
    if confirm is not None:
        body["confirm"] = confirm
    return api.client.post(f"/api/engines/{mode}/start", json=body, headers=headers)


def test_paper_engine_start_status_stop(api, admin):
    create(api, admin)
    engines = api.client.get("/api/engines", headers=admin).json()
    assert engines["paper"]["running"] is False and engines["live"]["live_allowed"] is False

    r = start(api, admin)
    assert r.status_code == 200, r.text
    status = r.json()
    assert status["running"] and status["strategies"] == ["always_long"]
    assert status["engine"]["live"] is False and status["engine"]["broker"] == "paper"
    assert api.client.get("/healthz").json()["engines"] == 1

    assert start(api, admin).status_code == 409  # already running

    r = api.client.post("/api/engines/paper/stop", headers=admin)
    assert r.status_code == 200 and r.json()["running"] is False
    assert api.feed.closed
    assert api.client.post("/api/engines/paper/stop", headers=admin).status_code == 409


def test_engine_start_errors(api, admin):
    assert start(api, admin, ids=("missing",)).status_code == 404
    r = api.client.post("/api/engines/paper/start", json={"strategy_ids": []}, headers=admin)
    assert r.status_code == 422
    assert api.client.post("/api/engines/sideways/start", json={}, headers=admin).status_code == 422


def test_derivative_engine_refused_without_lot_sizes(api, admin):
    fut = ALWAYS_LONG | {"id": "fut", "symbols": ["NFO:NIFTY-OCT26"], "product": "NRML"}
    create(api, admin, fut)
    r = start(api, admin, ids=("fut",))
    assert r.status_code == 422
    assert "NFO:NIFTY-OCT26" in r.json()["detail"]
    assert api.client.get("/api/engines", headers=admin).json()["paper"]["running"] is False


def test_live_trading_refusals(tmp_path):
    with Api(tmp_path) as api:  # defaults: LIVE_TRADING false, paper broker
        admin = api.bearer()
        create(api, admin)
        r = start(api, admin, mode="live", confirm="LIVE")
        assert r.status_code == 403 and "LIVE_TRADING" in r.json()["detail"]


def test_live_trading_needs_dhan_admin_and_confirmation(tmp_path):
    live = {"live_trading": True, "broker": "dhan", "dhan_client_id": "1"}
    with Api(tmp_path, **live) as api:
        admin = api.bearer()
        create(api, admin)
        assert api.client.get("/api/engines", headers=admin).json()["live"]["live_allowed"]

        r = start(api, admin, mode="live")
        assert r.status_code == 400 and "LIVE" in r.json()["detail"]
        r = start(api, admin, mode="live", confirm="yes")
        assert r.status_code == 400

        bob = api.add_user("bob")
        create(api, bob)
        r = start(api, bob, mode="live", confirm="LIVE")
        assert r.status_code == 403 and "admin" in r.json()["detail"]

        r = start(api, admin, mode="live", confirm="LIVE")
        assert r.status_code == 200, r.text
        assert r.json()["engine"]["live"] is True and r.json()["engine"]["broker"] == "dhan"
        assert api.client.post("/api/engines/live/stop", headers=admin).status_code == 200


def test_live_engine_refuses_a_broker_that_is_not_dhan(tmp_path):
    live = {"live_trading": True, "broker": "dhan", "dhan_client_id": "1"}
    with Api(tmp_path, live_broker=NotDhan(), **live) as api:
        admin = api.bearer()
        create(api, admin)
        r = start(api, admin, mode="live", confirm="LIVE")
        assert r.status_code == 403 and "Dhan" in r.json()["detail"]
        assert api.live_broker.md.closed


def test_live_broker_failure_is_a_502(tmp_path):
    async def boom():
        raise ConnectionError("dhan is down")

    live = {"live_trading": True, "broker": "dhan", "dhan_client_id": "1"}
    with Api(tmp_path, **live) as api:
        api.services.engines.live_broker_factory = boom
        admin = api.bearer()
        create(api, admin)
        r = start(api, admin, mode="live", confirm="LIVE")
        assert r.status_code == 502 and "dhan is down" in r.json()["detail"]


# --------------------------------------------------------------------------- flatten & accounts


def seed_position(api, qty=10):
    cash = api.settings.paper_starting_cash
    api.services.paper_store.save_account("paper-admin", cash, cash)
    api.services.paper_store.upsert_position(
        "paper-admin",
        Position(symbol=SYM, product=ProductType.MIS, qty=qty, avg_price=2500.0, last_price=2510),
    )


def test_flatten_sends_exits_through_risk(api, admin):
    create(api, admin)
    assert api.client.post("/api/engines/paper/flatten", headers=admin).status_code == 409
    seed_position(api)
    assert start(api, admin).status_code == 200
    r = api.client.post("/api/engines/paper/flatten", headers=admin)
    assert r.status_code == 200
    assert r.json()["exits"] == [{"symbol": SYM, "side": "SELL", "qty": 10}]
    engine = api.client.get("/api/engines", headers=admin).json()["paper"]["engine"]
    assert engine["approved"] + engine["rejected"] == 1  # the risk agent saw the exit
    api.client.post("/api/engines/paper/stop", headers=admin)


def test_accounts(api, admin):
    seed_position(api)
    paper = api.client.get("/api/accounts/paper", headers=admin).json()
    assert paper["running"] is False
    assert paper["funds"]["cash"] == api.settings.paper_starting_cash
    assert [p["symbol"] for p in paper["positions"]] == [SYM]
    assert paper["orders"] == [] and paper["fills"] == []
    r = api.client.get("/api/accounts/live", headers=admin)
    assert r.status_code == 409  # nothing to show until the live engine runs


def test_paper_reset(api, admin):
    seed_position(api)
    r = api.client.post("/api/accounts/paper/reset", json={"confirm": "yes"}, headers=admin)
    assert r.status_code == 400 and "paper-admin" in r.json()["detail"]
    r = api.client.post("/api/accounts/paper/reset", json={"confirm": "paper-admin"}, headers=admin)
    assert r.status_code == 200
    assert api.client.get("/api/accounts/paper", headers=admin).json()["positions"] == []


def test_paper_reset_refused_while_running(api, admin):
    create(api, admin)
    start(api, admin)
    r = api.client.post("/api/accounts/paper/reset", json={"confirm": "paper-admin"}, headers=admin)
    assert r.status_code == 409
    api.client.post("/api/engines/paper/stop", headers=admin)


def test_paper_accounts_are_per_user(api, admin):
    seed_position(api)
    bob = api.add_user("bob")
    assert api.client.get("/api/accounts/paper", headers=bob).json()["positions"] == []


# --------------------------------------------------------------------------- kill switch


def test_kill_switch_stops_engines_and_blocks_starts(api, admin):
    create(api, admin)
    start(api, admin)
    r = api.client.post("/api/control/kill", json={"reason": "test"}, headers=admin)
    assert r.status_code == 200 and r.json()["killed"] and r.json()["by"] == "admin"
    assert api.client.get("/api/control", headers=admin).json()["reason"] == "test"
    engine = api.client.get("/api/engines", headers=admin).json()["paper"]["engine"]
    assert engine["killed"] is True
    assert api.client.post("/api/engines/paper/flatten", headers=admin).status_code == 423
    api.client.post("/api/engines/paper/stop", headers=admin)
    r = start(api, admin)
    assert r.status_code == 423 and "kill switch" in r.json()["detail"]


def test_anyone_can_kill_only_admin_can_resume(api, admin):
    bob = api.add_user("bob")
    assert api.client.post("/api/control/kill", json={}, headers=bob).status_code == 200
    assert api.client.post("/api/control/resume", headers=bob).status_code == 403
    assert api.services.kill.engaged
    r = api.client.post("/api/control/resume", headers=admin)
    assert r.status_code == 200 and r.json()["killed"] is False


def test_kill_switch_survives_a_restart(tmp_path):
    with Api(tmp_path) as api:
        api.client.post("/api/control/kill", json={"reason": "persist"}, headers=api.bearer())
    with Api(tmp_path) as api:
        state = api.client.get("/api/control", headers=api.bearer()).json()
        assert state["killed"] and state["reason"] == "persist"


# --------------------------------------------------------------------------- models


@pytest.fixture
def registry(api, planted_ridge):
    reg = ModelRegistry(api.settings.models_dir)
    v1 = reg.register("alpha", planted_ridge)
    v2 = reg.register("alpha", planted_ridge)
    reg.set_live("alpha", v1, reason="first")
    reg.set_live("alpha", v2, reason="second")
    reg.record_decision("alpha", v2, {"promote": True, "failures": [], "candidate": {"sharpe": 1}})
    return reg


def test_model_routes(api, admin, registry):
    listed = api.client.get("/api/models", headers=admin).json()
    assert listed == [{"name": "alpha", "live": "v0002", "versions": 2}]
    detail = api.client.get("/api/models/alpha", headers=admin).json()
    assert detail["live"]["version"] == "v0002"
    assert [v["version"] for v in detail["versions"]] == ["v0001", "v0002"]
    assert detail["versions"][1]["gate"] is True and detail["versions"][0]["gate"] is None
    assert [h["event"] for h in detail["history"]][-2:] == ["promoted", "promoted"]
    meta = api.client.get("/api/models/alpha/v0001", headers=admin).json()
    assert meta["model"]["kind"] and meta["decisions"] == []
    assert api.client.get("/api/models/alpha/v0009", headers=admin).status_code == 404
    assert api.client.get("/api/models/nope", headers=admin).status_code == 404


def test_rollback_and_manual_promotion_are_admin_only(api, admin, registry):
    bob = api.add_user("bob")
    body = {"reason": "bad fills"}
    assert api.client.post("/api/models/alpha/rollback", json=body, headers=bob).status_code == 403
    r = api.client.post("/api/models/alpha/rollback", json=body, headers=admin)
    assert r.status_code == 200 and r.json() == {"live": "v0001"}
    r = api.client.post("/api/models/alpha/rollback", json=body, headers=admin)
    assert r.status_code == 409  # nothing further back

    promote = {"version": "v0002", "reason": "manual override"}
    r = api.client.post("/api/models/alpha/promote", json=promote, headers=admin)
    assert r.status_code == 400  # bypassing the gate needs confirm
    r = api.client.post(
        "/api/models/alpha/promote", json=promote | {"confirm": True}, headers=admin
    )
    assert r.status_code == 200 and registry.live_version("alpha") == "v0002"
    r = api.client.post("/api/models/alpha/promote", json=promote | {"confirm": True}, headers=bob)
    assert r.status_code == 403
    missing = {"version": "v0009", "reason": "typo", "confirm": True}
    assert (
        api.client.post("/api/models/alpha/promote", json=missing, headers=admin).status_code == 404
    )


# --------------------------------------------------------------------------- SEBI-gated stubs


@pytest.mark.parametrize("path", ["/api/marketplace", "/api/auto-trading", "/api/ai-strategies"])
def test_gated_routes_absent_when_the_flag_is_off(api, admin, path):
    assert api.client.get(path, headers=admin).status_code == 404


@pytest.mark.parametrize("path", ["/api/marketplace", "/api/auto-trading", "/api/ai-strategies"])
def test_gated_routes_say_not_available_when_the_flag_is_on(tmp_path, path):
    with Api(tmp_path, feature_marketplace=True) as api:
        assert api.client.get(path).status_code == 401  # still behind sign-in
        r = api.client.get(path, headers=api.bearer())
        assert r.status_code == 501 and "SEBI" in r.json()["detail"]


def test_shutdown_stops_running_engines(tmp_path):
    with Api(tmp_path) as api:
        admin = api.bearer()
        create(api, admin)
        start(api, admin)
        feed = api.feed
    deadline = time.monotonic() + 5
    while not feed.closed and time.monotonic() < deadline:
        time.sleep(0.05)
    assert feed.closed
