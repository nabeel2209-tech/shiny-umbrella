"""The API's building blocks, without HTTP: formatting, charts, the builder's form
mapping, the strategy store, the kill switch, the event hub, and the serve script."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest
from starlette.datastructures import FormData

from scripts.serve import live_gate
from tests.apikit import ALWAYS_LONG, TREND_YAML, make_settings
from trading.api.builder import form_rows, form_to_dict
from trading.api.charts import line_chart, lttb, nice_ticks
from trading.api.events import EventHub, KillSwitch
from trading.api.formatting import (
    MISSING,
    duration,
    indian_grouping,
    inr,
    inr_compact,
    num,
    pct,
    signed_pct,
    when,
)
from trading.api.routes.pages import _safe_next
from trading.api.strategy_store import (
    StrategyExists,
    StrategyNotFound,
    StrategyStore,
    parse_strategy_yaml,
    validate_strategy_dict,
)
from trading.core.types import IST
from trading.strategies.schema import StrategyConfig

# --------------------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("n", "text"),
    [(0, "0"), (999, "999"), (1000, "1,000"), (123456, "1,23,456"), (1234567, "12,34,567"),
     (123456789, "12,34,56,789"), (-1234567, "-12,34,567")],
)  # fmt: skip
def test_indian_grouping(n, text):
    assert indian_grouping(n) == text


def test_rupees():
    assert inr(1036007.01) == "₹10,36,007.01"
    assert inr(-687.97) == "-₹687.97"
    assert inr(999.6, 0) == "₹1,000"  # rounds before grouping
    assert inr(0.999) == "₹1.00"
    assert inr(-0.001) == "₹0.00"  # no negative zero
    assert inr(None) == MISSING and inr(float("nan")) == MISSING
    assert inr_compact(950) == "₹950"
    assert inr_compact(12_345) == "₹12.3K"
    assert inr_compact(1_036_007) == "₹10.36L"
    assert inr_compact(123_456_789) == "₹12.35Cr"
    assert inr_compact(-5_900) == "-₹5.9K"


def test_percent_number_time_duration():
    assert pct(-0.0382) == "-3.82%" and pct(None) == MISSING
    assert signed_pct(0.0028) == "+0.28%" and signed_pct(-0.01, 1) == "-1.0%"
    assert num(1234567) == "12,34,567" and num(0.4912) == "0.49" and num(None) == MISSING
    utc = datetime(2026, 9, 26, 8, 33, tzinfo=IST).astimezone(tz=None).isoformat()
    assert when(utc) == "26 Sep 08:33"
    assert when("2026-09-26T03:03:00+00:00") == "26 Sep 08:33"  # shown in IST
    assert when("") == MISSING
    assert duration(1800) == "30 min" and duration(5400) == "1.5 h" and duration("") == MISSING


# --------------------------------------------------------------------------- charts


def test_lttb_keeps_ends_and_extremes():
    values = [0.0] * 1000
    values[500] = 10.0
    keep = lttb(values, 50)
    assert len(keep) == 50 and keep[0] == 0 and keep[-1] == 999 and 500 in keep
    assert lttb([1.0, 2.0, 3.0], 10) == [0, 1, 2]


def test_nice_ticks():
    assert nice_ticks(995_000, 1_037_000) == [980_000, 1_000_000, 1_020_000, 1_040_000]
    ticks = nice_ticks(-0.0382, 0.0)
    assert ticks[0] <= -0.0382 and ticks[-1] >= 0.0
    assert len(nice_ticks(5.0, 5.0)) >= 2  # flat series still gets an axis


def series(n=300):
    start = datetime(2026, 9, 14, 9, 15, tzinfo=IST)
    stamps = [start + timedelta(minutes=5 * i) for i in range(n)]
    values = [1_000_000 + 50 * i for i in range(n)]
    return stamps, values


def test_line_chart_is_one_series_with_hover_data():
    stamps, values = series()
    chart = line_chart(stamps, values, chart_id="equity", reference=1_000_000)
    svg = chart.svg
    assert svg.startswith('<svg class="chart-svg"') and svg.endswith("</svg>")
    assert svg.count('class="line"') == 1 and svg.count('class="area"') == 1
    assert 'class="reference"' in svg and 'aria-labelledby="equity-title"' in svg
    assert "<script" not in svg and "style=" not in svg
    points = json.loads(chart.points)
    assert len(points["x"]) == len(points["y"]) == len(points["t"]) == len(points["v"])
    assert points["v"][-1] == "₹10.15L" and points["w"] == 760


def test_line_chart_decimates_long_series_and_escapes_text():
    stamps, values = series(5000)
    chart = line_chart(stamps, values, chart_id="x", fmt=lambda v: "<b>&</b>", max_points=400)
    assert len(json.loads(chart.points)["x"]) == 400
    assert "<b>" not in chart.svg and "&lt;b&gt;&amp;&lt;/b&gt;" in chart.svg


def test_line_chart_empty():
    chart = line_chart([], [], chart_id="none")
    assert chart.svg == "" and chart.points == "{}"


# --------------------------------------------------------------------------- builder form


def formdata(fields: dict) -> FormData:
    items = []
    for key, value in fields.items():
        for v in value if isinstance(value, list) else [value]:
            items.append((key, v))
    return FormData(items)


def test_form_to_dict_builds_a_valid_strategy():
    data, errors = form_to_dict(
        formdata(
            {
                "id": "f",
                "symbols": "nse:reliance, NSE:TCS",
                "enabled": "1",
                "long_join": "any",
                "long_feature": ["trend", " ", "rsi_14"],
                "long_op": ["gt", "gt", "lt"],
                "long_value": ["0.001", "", "30"],
                "exit_long_always": "1",
                "stop_loss_pct": "2",
                "model_name": "alpha",
                "sizing_max_qty": "",
            }
        )
    )
    assert errors == []
    assert data["symbols"] == ["NSE:RELIANCE", "NSE:TCS"]
    assert data["rules"]["long"] == {
        "any": [
            {"feature": "trend", "op": "gt", "value": 0.001},
            {"feature": "rsi_14", "op": "lt", "value": 30.0},
        ]
    }
    assert data["rules"]["exit_long"] == {"always": True}
    assert "short" not in data["rules"]
    assert data["execution"]["stop_loss_pct"] == 0.02 and data["sizing"]["max_qty"] is None
    assert data["model"]["name"] == "alpha" and data["model"]["version"] is None
    StrategyConfig.model_validate(data)


def test_form_to_dict_reports_bad_numbers():
    _, errors = form_to_dict(
        formdata({"id": "f", "long_feature": "trend", "long_op": "gt", "long_value": "high",
                  "payoff_ratio": "big"})
    )  # fmt: skip
    assert "long row 1: value must be a number" in errors
    assert "payoff_ratio: not a number" in errors


def test_form_rows_round_trip_and_nested_flag():
    strategy = StrategyConfig.from_yaml_str(TREND_YAML)
    rows = form_rows(strategy)
    assert rows["long"]["rows"] == [{"feature": "trend", "op": "gt", "value": 0.0005}]
    assert rows["exit_long"]["join"] == "any" and not rows["long"]["nested"]
    assert rows["short"]["rows"] == [{"feature": "", "op": "gt", "value": ""}]
    nested = StrategyConfig.model_validate(
        ALWAYS_LONG
        | {"rules": {"long": {"all": [{"feature": "ret_1", "op": "gt", "other": "ret_5"}]}}}
    )
    assert form_rows(nested)["long"]["nested"]
    assert form_rows(None)["long"]["rows"][0]["feature"] == ""


# --------------------------------------------------------------------------- strategy store


def test_strategy_store(tmp_path):
    store = StrategyStore(tmp_path)
    s = StrategyConfig.model_validate(ALWAYS_LONG)
    store.create("alice", s)
    with pytest.raises(StrategyExists):
        store.create("alice", s)
    assert [x.id for x in store.list("alice")] == ["always_long"]
    assert store.list("bob") == []
    with pytest.raises(StrategyNotFound):
        store.get("alice", "../../etc/passwd")
    with pytest.raises(ValueError):
        store.list("../alice")
    with pytest.raises(ValueError):
        store.update("alice", "always_long", s.model_copy(update={"id": "other"}))
    moved = store.delete("alice", "always_long")
    assert moved.parent.name == ".trash" and moved.exists()
    assert store.list("alice") == []  # the trash is not listed


def test_parse_and_validate_never_raise():
    assert parse_strategy_yaml("id: [")[1][0].startswith("YAML:")
    assert parse_strategy_yaml("- a list") == (
        None,
        ["the YAML must be a mapping of strategy fields"],
    )
    strategy, errors = validate_strategy_dict(ALWAYS_LONG | {"symbols": ["RELIANCE"]})
    assert strategy is None and errors


# --------------------------------------------------------------------------- kill switch & hub


def test_kill_switch_persists_and_fails_closed(tmp_path):
    path = tmp_path / "state" / "kill.json"
    kill = KillSwitch(path)
    assert not kill.engaged
    kill.engage("test", "alice")
    assert KillSwitch(path).state().by == "alice" and KillSwitch(path).engaged
    kill.release("admin")
    assert not KillSwitch(path).engaged
    path.write_text("{not json")
    state = KillSwitch(path).state()
    assert state.killed and "unreadable" in state.reason


async def test_event_hub_routes_and_drops_oldest():
    hub = EventHub(maxsize=2)
    alice, bob = hub.subscribe("alice"), hub.subscribe("bob")
    hub.publish("alice", {"type": "orders", "n": 1})
    hub.publish(None, {"type": "kill"})
    assert bob.qsize() == 1 and (await bob.get())["type"] == "kill"
    hub.publish("alice", {"type": "orders", "n": 2})
    assert alice.qsize() == 2  # the first order event was dropped, not the engine blocked
    assert [(await alice.get())["type"] for _ in range(2)] == ["kill", "orders"]
    hub.unsubscribe("alice", alice)
    hub.unsubscribe("bob", bob)
    assert hub.clients() == 0
    await asyncio.sleep(0)


# --------------------------------------------------------------------------- pages helpers


@pytest.mark.parametrize(
    ("target", "safe"),
    [("/backtests", "/backtests"), ("/builder?template=x", "/builder?template=x"),
     ("//evil.example", "/"), ("https://evil.example", "/"), ("", "/"), (None, "/"),
     ("builder", "/"), ("/\\evil.example", "/"), ("/\\/evil.example", "/"),
     ("/ok\r\nSet-Cookie: x=1", "/")],
)  # fmt: skip
def test_safe_next(target, safe):
    assert _safe_next(target) == safe


# --------------------------------------------------------------------------- serve


def test_live_gate_leaves_paper_alone(tmp_path):
    settings = make_settings(tmp_path)
    assert live_gate(settings, prompt=lambda _: "LIVE") is settings


def test_live_gate_needs_the_startup_confirmation(tmp_path):
    settings = make_settings(tmp_path, live_trading=True, broker="dhan", dhan_client_id="1")
    assert live_gate(settings, prompt=lambda _: "LIVE").live_trading is True
    assert live_gate(settings, prompt=lambda _: "yes").live_trading is False
    assert settings.live_trading is True  # the original is untouched


def test_live_gate_without_a_terminal_means_paper(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, live_trading=True, broker="dhan", dhan_client_id="1")
    monkeypatch.setattr("sys.stdin", open("/dev/null"))  # noqa: SIM115
    assert live_gate(settings).live_trading is False
