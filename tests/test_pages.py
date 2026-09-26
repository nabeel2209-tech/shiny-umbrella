"""The HTMX dashboard: every page, every fragment, and the browser-side rules
(sign-in redirects, CSRF on form posts, no inline scripts or styles)."""

from __future__ import annotations

import re

import pytest

from tests.apikit import ALWAYS_LONG, PASSWORD, TREND_YAML, Api, write_archive
from trading.training.registry import ModelRegistry

PAGES = [
    "/",
    "/builder",
    "/builder?template=trend_reliance",
    "/backtests",
    "/practise",
    "/algo",
    "/models",
    "/how-it-works",
    "/marketplace",
    "/auto-trading",
    "/ai-strategies",
]
HTML = {"accept": "text/html"}


@pytest.fixture
def api(tmp_path):
    with Api(tmp_path) as a:
        yield a


@pytest.fixture
def browser(api):
    """Signed in through the login form; returns headers an htmx request carries."""
    csrf = api.browser_login()
    return {"X-CSRF-Token": csrf, "HX-Request": "true"}


RULE_FORM = {
    "id": "form_made",
    "name": "Made in the form",
    "symbols": "NSE:RELIANCE, NSE:TCS",
    "interval": "5m",
    "product": "MIS",
    "enabled": "1",
    "long_join": "all",
    "long_feature": ["trend", "rsi_14", ""],
    "long_op": ["gt", "lt", "gt"],
    "long_value": ["0.0005", "70", ""],
    "exit_long_join": "any",
    "exit_long_feature": ["trend"],
    "exit_long_op": ["lt"],
    "exit_long_value": ["0"],
    "short_join": "all",
    "exit_short_join": "all",
    "expected_edge_bps": "25",
    "payoff_ratio": "1.5",
    "sizing_mode": "fixed_notional",
    "sizing_qty": "1",
    "sizing_notional": "200000",
    "sizing_fraction": "0.02",
    "urgency": "NORMAL",
    "limit_band_bps": "8",
    "ttl_seconds": "300",
    "stop_loss_pct": "1.5",
    "take_profit_pct": "",
    "max_holding_bars": "24",
    "notes": "from a test",
}


# --------------------------------------------------------------------------- sign-in


@pytest.mark.parametrize("path", PAGES)
def test_pages_need_sign_in(api, path):
    r = api.client.get(path, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login?next=/")


def test_redirect_keeps_the_query_string(api):
    r = api.client.get("/builder?template=trend_reliance", follow_redirects=False)
    assert r.headers["location"] == "/login?next=/builder%3Ftemplate%3Dtrend_reliance"


def test_login_form(api):
    assert "Sign in" in api.client.get("/login").text
    r = api.client.post("/login", data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401 and "invalid username or password" in r.text.lower()
    r = api.client.post(
        "/login",
        data={"username": "admin", "password": PASSWORD, "next": "/backtests"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/backtests"
    assert "httponly" in r.headers["set-cookie"].lower()


@pytest.mark.parametrize(
    "target",
    [
        "//evil.example/x",
        "https://evil.example/",
        "/\\evil.example",
        "javascript:alert(1)",
        "",
        "x",
    ],
)
def test_login_never_redirects_off_site(api, target):
    r = api.client.post(
        "/login",
        data={"username": "admin", "password": PASSWORD, "next": target},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/"


def test_htmx_request_without_session_redirects_the_whole_page(api):
    r = api.client.get(
        "/ui/accounts/paper",
        headers={"HX-Request": "true", "HX-Current-URL": "http://testserver/practise"},
        follow_redirects=False,
    )
    assert r.status_code == 204
    assert r.headers["HX-Redirect"] == "/login?next=/practise"


def test_logout_form_needs_csrf_and_ends_the_session(api):
    csrf = api.browser_login()
    assert api.client.post("/logout", follow_redirects=False).status_code == 403
    r = api.client.post("/logout", data={"csrf_token": csrf}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert api.client.get("/", follow_redirects=False).status_code == 303


# --------------------------------------------------------------------------- every page


@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders_with_the_full_nav(api, browser, path):
    r = api.client.get(path)
    assert r.status_code == 200, r.text
    for label in ("Strategy Builder", "Backtesting", "Algo Trading", "Practise", "How It Works"):
        assert label in r.text
    for label in ("Marketplace", "Auto Trading", "AI Strategies"):
        assert label in r.text
    assert "Kill switch" in r.text
    assert f'"X-CSRF-Token": "{browser["X-CSRF-Token"]}"' in r.text


@pytest.mark.parametrize("path", PAGES)
def test_pages_have_no_inline_script_or_style(api, browser, path):
    """The CSP forbids them; a page that used one would silently break."""
    text = api.client.get(path).text
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", text)
    assert not re.search(r"\sstyle\s*=", text)
    assert "<style" not in text and "hx-on" not in text and "javascript:" not in text


@pytest.mark.parametrize("page", ["marketplace", "auto-trading", "ai-strategies"])
def test_coming_soon_pages_explain_the_sebi_gate(api, browser, page):
    text = api.client.get(f"/{page}").text
    assert "coming soon" in text and "SEBI" in text and "FEATURE_MARKETPLACE" in text


def test_unknown_page_is_a_readable_404(api, browser):
    r = api.client.get("/no-such-page", headers=HTML)
    assert r.status_code == 404 and "Not found" in r.text and "<html" in r.text
    r = api.client.get("/api/no-such-route", headers=HTML)
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_static_assets_are_served(api):
    assert "htmx" in api.client.get("/static/vendor/htmx.min.js").text[:200]
    assert api.client.get("/static/app.js").status_code == 200
    assert "--series-1" in api.client.get("/static/app.css").text


# --------------------------------------------------------------------------- strategy builder


def test_builder_template_prefills_the_form(api, browser):
    text = api.client.get("/builder?template=trend_reliance").text
    assert 'value="trend_reliance"' in text and "NSE:RELIANCE" in text
    assert "<option selected>trend</option>" in text


def test_builder_row_fragment(api, browser):
    r = api.client.get("/ui/builder/row?group=exit_long", headers=browser)
    assert r.status_code == 200
    assert 'name="exit_long_feature"' in r.text and "data-remove-row" in r.text
    assert api.client.get("/ui/builder/row?group=sideways", headers=browser).status_code == 404


def test_builder_preview_shows_yaml_or_errors(api, browser):
    r = api.client.post("/ui/builder/preview", data=RULE_FORM, headers=browser)
    assert r.status_code == 200 and "Valid" in r.text
    assert "stop_loss_pct: 0.015" in r.text  # the form's percent is stored as a fraction
    assert "NSE:TCS" in r.text
    bad = RULE_FORM | {"long_feature": ["nope"], "long_op": ["gt"], "long_value": ["1"]}
    r = api.client.post("/ui/builder/preview", data=bad, headers=browser)
    assert "unknown feature for 5m bars: nope" in r.text
    r = api.client.post(
        "/ui/builder/preview", data=RULE_FORM | {"payoff_ratio": "lots"}, headers=browser
    )
    assert "payoff_ratio: not a number" in r.text
    assert api.client.get("/api/strategies", headers=browser).json() == []


def test_builder_save_edit_delete(api, browser):
    r = api.client.post("/ui/builder/save", data=RULE_FORM, headers=browser)
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/builder/form_made?saved=1"
    saved = api.services.strategies.get("admin", "form_made")
    assert saved.execution.stop_loss_pct == pytest.approx(0.015)
    assert saved.execution.max_holding_bars == 24 and saved.sizing.notional == 200_000

    r = api.client.post("/ui/builder/save", data=RULE_FORM, headers=browser)
    assert r.status_code == 409 and "already exists" in r.text

    edited = RULE_FORM | {"editing": "1", "name": "Renamed"}
    r = api.client.post("/ui/builder/save", data=edited, headers=browser)
    assert r.status_code == 204
    assert api.services.strategies.get("admin", "form_made").name == "Renamed"

    page = api.client.get("/builder/form_made?saved=1").text
    assert "Saved." in page and 'value="Renamed"' in page and "readonly" in page

    r = api.client.post("/ui/strategies/form_made/delete", headers=browser)
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/builder?deleted=form_made"
    assert api.client.get("/builder/form_made", headers=HTML).status_code == 404
    assert api.client.post("/ui/strategies/form_made/delete", headers=browser).status_code == 404


def test_builder_save_invalid_is_422_with_reasons(api, browser):
    r = api.client.post(
        "/ui/builder/save", data=RULE_FORM | {"symbols": "RELIANCE"}, headers=browser
    )
    assert r.status_code == 422 and "Not valid yet" in r.text


def test_builder_save_yaml(api, browser):
    r = api.client.post("/ui/builder/save-yaml", data={"yaml": TREND_YAML}, headers=browser)
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/builder/trend_test?saved=1"
    r = api.client.post(
        "/ui/builder/save-yaml",
        data={"yaml": TREND_YAML.replace("Trend test", "Changed")},
        headers=browser,
    )
    assert r.status_code == 204  # same id: an edit
    assert api.services.strategies.get("admin", "trend_test").name == "Changed"
    r = api.client.post("/ui/builder/save-yaml", data={"yaml": "id: [oops"}, headers=browser)
    assert r.status_code == 422


def test_nested_rules_are_flagged_as_yaml_only(api, browser):
    extra = "[{feature: rsi_14, op: lt, value: 30}, {feature: ret_1, op: gt, other: ret_5}]"
    nested = TREND_YAML.replace(
        "      - {feature: trend, op: gt, value: 0.0005}",
        f"      - {{feature: trend, op: gt, value: 0.0005}}\n      - any: {extra}",
    )
    r = api.client.post("/ui/builder/save-yaml", data={"yaml": nested}, headers=browser)
    assert r.status_code == 204, r.text
    text = api.client.get("/builder/trend_test").text
    assert "Edit it in YAML below" in text


def test_ui_posts_need_csrf(api, browser):
    no_csrf = {"HX-Request": "true"}
    assert api.client.post("/ui/builder/save", data=RULE_FORM, headers=no_csrf).status_code == 403
    assert api.client.post("/ui/control/kill", headers=no_csrf).status_code == 403
    # a plain form post may carry the token as a field instead of a header
    r = api.client.post(
        "/ui/control/kill", data={"csrf_token": browser["X-CSRF-Token"], "reason": "field"}
    )
    assert r.status_code == 200 and api.services.kill.engaged


# --------------------------------------------------------------------------- backtesting


def test_backtest_form_to_results_page(api, browser, calendar):
    days = write_archive(api.settings.archive_dir, calendar)
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=browser)
    r = api.client.post(
        "/ui/backtests",
        data={
            "strategy_ids": ["always_long"],
            "start": str(days[0]),
            "end": str(days[-1]),
            "name": "From the form",
            "initial_cash": "500000",
            "slippage_bps": "3",
            "participation": "10",
            "liquidate": "1",
        },
        headers=browser,
    )
    assert r.status_code == 200 and "From the form" in r.text
    job = api.client.get("/api/backtests", headers=browser).json()[0]
    assert job["params"]["participation"] == pytest.approx(0.10)
    assert job["params"]["initial_cash"] == 500_000
    job = api.wait_for_job(job["id"], browser)
    assert job["status"] == "done", job

    jobs = api.client.get("/ui/backtests/jobs", headers=browser).text
    assert "done" in jobs and "every 2s" not in jobs  # nothing running: no polling

    page = api.client.get(f"/backtests/{job['id']}").text
    for text in ("Net P&amp;L", "Sharpe", "Max drawdown", "Closed trades", "Costs paid"):
        assert text in page
    assert page.count('class="chart-svg"') == 2  # equity and drawdown, never one dual-axis chart
    assert "data-points=" in page and "Table view: daily closes" in page
    assert "By strategy" in page and "always_long" in page


def test_backtest_form_errors(api, browser):
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=browser)
    r = api.client.post(
        "/ui/backtests",
        data={"strategy_ids": ["always_long"], "start": "2026-09-10", "end": "2026-09-01"},
        headers=browser,
    )
    assert r.status_code == 422 and "end is before start" in r.text
    assert "Value error" not in r.text
    r = api.client.post(
        "/ui/backtests", data={"strategy_ids": ["always_long"], "start": "soon"}, headers=browser
    )
    assert r.status_code == 422 and "check the dates" in r.text
    r = api.client.post(
        "/ui/backtests",
        data={"strategy_ids": ["ghost"], "start": "2026-09-01", "end": "2026-09-02"},
        headers=browser,
    )
    assert r.status_code == 422 and "ghost" in r.text


def test_backtest_detail_pages(api, browser):
    assert api.client.get("/backtests/nope", headers=HTML).status_code == 404
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=browser)
    job = api.client.post(
        "/api/backtests",
        json={"strategy_ids": ["always_long"], "start": "2026-09-14", "end": "2026-09-15"},
        headers=browser,
    ).json()
    job = api.wait_for_job(job["id"], browser)  # no archive: fails
    page = api.client.get(f"/backtests/{job['id']}").text
    assert "Failed." in page and "ingest them first" in page


# --------------------------------------------------------------------------- practise & algo


def test_practise_engine_panel_start_flatten_stop(api, browser):
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=browser)
    r = api.client.post("/ui/engines/paper/flatten", headers=browser)
    assert "not running" in r.text
    r = api.client.post(
        "/ui/engines/paper/start", data={"strategy_ids": ["always_long"]}, headers=browser
    )
    assert "running" in r.text and "every 5s" in r.text and "Flatten all" in r.text
    assert "running" in api.client.get("/ui/engines/paper/panel", headers=browser).text
    r = api.client.post("/ui/engines/paper/flatten", headers=browser)
    assert "sent 0 exit order(s)" in r.text
    r = api.client.post("/ui/engines/paper/stop", headers=browser)
    assert "stopped" in r.text and "Start paper engine" in r.text
    assert api.client.post("/ui/engines/paper/dance", headers=browser).status_code == 404


def test_account_fragment_and_reset(api, browser):
    r = api.client.get("/ui/accounts/paper", headers=browser)
    assert "₹10,00,000" in r.text and "No open positions." in r.text
    r = api.client.post("/ui/accounts/paper/reset", data={"confirm": "nope"}, headers=browser)
    assert "type paper-admin to confirm" in r.text
    r = api.client.post(
        "/ui/accounts/paper/reset", data={"confirm": "paper-admin"}, headers=browser
    )
    assert "reset to its starting cash" in r.text
    r = api.client.get("/ui/accounts/live", headers=browser)
    assert "the live engine is not running" in r.text


def test_algo_page_explains_why_live_is_off(api, browser):
    api.client.post("/api/strategies", json=ALWAYS_LONG, headers=browser)
    page = api.client.get("/algo").text
    assert "Live trading is off." in page and "LIVE_TRADING=true" in page
    r = api.client.post(
        "/ui/engines/live/start",
        data={"strategy_ids": ["always_long"], "confirm": "LIVE"},
        headers=browser,
    )
    assert "LIVE_TRADING is not true" in r.text


def test_algo_page_asks_for_live_confirmation(tmp_path):
    with Api(tmp_path, live_trading=True, broker="dhan", dhan_client_id="1") as api:
        headers = {"X-CSRF-Token": api.browser_login(), "HX-Request": "true"}
        api.client.post("/api/strategies", json=ALWAYS_LONG, headers=headers)
        page = api.client.get("/algo").text
        assert "Live trading is off." not in page and "Type <code>LIVE</code>" in page
        r = api.client.post(
            "/ui/engines/live/start", data={"strategy_ids": ["always_long"]}, headers=headers
        )
        assert "type LIVE to confirm" in r.text
        r = api.client.post(
            "/ui/engines/live/start",
            data={"strategy_ids": ["always_long"], "confirm": "LIVE"},
            headers=headers,
        )
        assert "running" in r.text and "broker dhan" in r.text
        api.client.post("/ui/engines/live/stop", headers=headers)


# --------------------------------------------------------------------------- kill switch


def test_kill_banner_fragments(api, browser):
    r = api.client.post("/ui/control/kill", data={"reason": "from the page"}, headers=browser)
    assert "Kill switch engaged." in r.text and "from the page" in r.text and "Release" in r.text
    assert "Kill switch engaged." in api.client.get("/ui/control/banner", headers=browser).text
    assert "Kill switch engaged." in api.client.get("/").text  # on every page
    r = api.client.post("/ui/control/resume", headers=browser)
    assert "Kill switch engaged." not in r.text and not api.services.kill.engaged
    assert api.client.post("/ui/control/explode", headers=browser).status_code == 404


def test_only_admin_sees_release(api):
    api.services.auth.create_user("bob", "another long password")
    csrf = api.browser_login("bob", "another long password")
    headers = {"X-CSRF-Token": csrf, "HX-Request": "true"}
    r = api.client.post("/ui/control/kill", headers=headers)
    assert "Only an admin can release it." in r.text
    r = api.client.post("/ui/control/resume", headers=headers)
    assert "only an admin may release" in r.text and api.services.kill.engaged


# --------------------------------------------------------------------------- models


def test_model_pages_and_rollback(api, browser, planted_ridge):
    assert "No models yet" in api.client.get("/models").text
    reg = ModelRegistry(api.settings.models_dir)
    v1, v2 = reg.register("alpha", planted_ridge), reg.register("alpha", planted_ridge)
    reg.set_live("alpha", v1, reason="first")
    reg.set_live("alpha", v2, reason="second")
    reg.record_decision(
        "alpha", v1, {"promote": False, "failures": ["too few trades"], "candidate": {}}
    )
    assert "alpha" in api.client.get("/models").text
    page = api.client.get("/models/alpha").text
    assert "v0002" in page and "Roll back to v0001" in page and "too few trades" in page
    assert api.client.get("/models/nope", headers=HTML).status_code == 404

    r = api.client.post("/ui/models/alpha/rollback", data={"reason": "test"}, headers=browser)
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/models/alpha"
    assert reg.live_version("alpha") == "v0001"
    assert api.client.post("/ui/models/alpha/rollback", headers=browser).status_code == 409


def test_non_admin_cannot_roll_back(api, planted_ridge):
    reg = ModelRegistry(api.settings.models_dir)
    v1, v2 = reg.register("alpha", planted_ridge), reg.register("alpha", planted_ridge)
    reg.set_live("alpha", v1, reason="first")
    reg.set_live("alpha", v2, reason="second")
    api.services.auth.create_user("bob", "another long password")
    csrf = api.browser_login("bob", "another long password")
    assert "Roll back to" not in api.client.get("/models/alpha").text
    r = api.client.post("/ui/models/alpha/rollback", headers={"X-CSRF-Token": csrf})
    assert r.status_code == 403 and reg.live_version("alpha") == "v0002"
