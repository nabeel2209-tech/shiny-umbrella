"""Monitor agent: alert log, heartbeats, kill switch and slippage tracking."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading.agents.monitor import MonitorAgent, MonitorConfig, SlippageRecord
from trading.agents.portfolio import Portfolio
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Alert,
    AlertLevel,
    Fill,
    Heartbeat,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    ProductType,
    RiskRejection,
    Side,
    Tick,
)

SYM = "NSE:RELIANCE"
TS = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


async def build(**kw):
    bus = InMemoryBus()
    portfolio = Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    agent = MonitorAgent(bus, portfolio, MonitorConfig(**kw), clock=SimClock(TS))
    await agent.start()
    return bus, agent, portfolio


def intent(**kw) -> OrderIntent:
    base = dict(
        ts=TS,
        strategy_id="s1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        product=ProductType.MIS,
        reference_price=2500.0,
    )
    return OrderIntent(**{**base, **kw})


def order(intent_id: str, oid: str = "o1") -> Order:
    return Order(
        id=oid,
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        product=ProductType.MIS,
        status=OrderStatus.OPEN,
        created_at=TS,
        updated_at=TS,
        intent_id=intent_id,
    )


def fill(oid: str = "o1", price: float = 2500.0, side=Side.BUY, qty: int = 10) -> Fill:
    return Fill(
        order_id=oid, symbol=SYM, side=side, qty=qty, price=price, ts=TS, product=ProductType.MIS
    )


async def test_alerts_are_logged_and_counted():
    bus, agent, _ = await build()
    await bus.publish(
        Topics.ALERTS, Alert(ts=TS, level=AlertLevel.WARN, source="risk", message="careful")
    )
    await bus.publish(
        Topics.ALERTS, Alert(ts=TS, level=AlertLevel.CRITICAL, source="exec", message="bad")
    )
    assert len(agent.alerts) == 2
    assert agent.alerts[-1].level is AlertLevel.CRITICAL
    assert agent.status()["alerts"] == 2


async def test_alert_log_is_bounded():
    bus, agent, _ = await build(alert_log_size=3)
    for i in range(10):
        await bus.publish(
            Topics.ALERTS, Alert(ts=TS, level=AlertLevel.INFO, source="x", message=str(i))
        )
    assert [a.message for a in agent.alerts] == ["7", "8", "9"]


async def test_heartbeats_and_staleness():
    bus, agent, _ = await build(heartbeat_timeout_seconds=30, watch_agents=["data", "risk"])
    await bus.publish(Topics.HEARTBEAT, Heartbeat(ts=TS, agent="data"))
    assert agent.stale_agents(TS) == ["risk"]  # never beat at all
    assert agent.stale_agents(TS + timedelta(seconds=45)) == ["data", "risk"]
    stale = await agent.check_health(TS + timedelta(seconds=45))
    assert stale == ["data", "risk"]
    assert any("no heartbeat from data" in a.message for a in agent.alerts)


async def test_counts_intents_orders_fills_and_rejections():
    bus, agent, _ = await build()
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.ORDERS, order(i.id))
    await bus.publish(Topics.FILLS, fill())
    await bus.publish(
        Topics.REJECTED, RiskRejection(intent_id="x", ts=TS, rule="cost_threshold", reason="thin")
    )
    assert agent.counts == {"intents": 1, "orders": 1, "fills": 1, "rejections": 1}
    assert agent.rejections == {"cost_threshold": 1}


async def test_kill_switch_is_broadcast_and_latched():
    bus, agent, _ = await build()
    commands = []

    async def on_control(_t, m):
        commands.append(m)

    await bus.subscribe(Topics.CONTROL, on_control)
    cmd = await agent.kill("losing too fast")
    assert cmd.command == "KILL" and commands[-1].reason == "losing too fast"
    assert agent.killed and agent.status()["killed"] is True
    await agent.resume()
    assert not agent.killed
    await agent.flatten("end of day")
    assert commands[-1].command == "FLATTEN"


async def test_slippage_measures_achieved_against_mid_at_signal():
    bus, agent, _ = await build()
    await bus.publish(
        Topics.ticks(SYM), Tick(symbol=SYM, ts=TS, ltp=2500.0, bid=2499.0, ask=2501.0)
    )
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.ORDERS, order(i.id))
    await bus.publish(Topics.FILLS, fill(price=2502.5))  # paid 10 bps over the mid
    assert len(agent.slippage) == 1
    record = agent.slippage[0]
    assert record.mid_at_signal == 2500.0 and record.achieved == 2502.5
    assert record.slippage_bps == pytest.approx(10.0)
    summary = agent.slippage_summary()
    assert summary["count"] == 1 and summary["mean_bps"] == pytest.approx(10.0)
    assert summary["total_cost"] == pytest.approx(25.0)  # 10 shares x 2.50


def test_slippage_sign_is_cost_for_both_sides():
    buy = SlippageRecord(TS, SYM, "s", Side.BUY, 1, 100.0, 101.0)
    sell = SlippageRecord(TS, SYM, "s", Side.SELL, 1, 100.0, 99.0)
    good = SlippageRecord(TS, SYM, "s", Side.BUY, 1, 100.0, 99.0)
    assert buy.slippage_bps == pytest.approx(100.0)  # paid up
    assert sell.slippage_bps == pytest.approx(100.0)  # sold lower: also a cost
    assert good.slippage_bps == pytest.approx(-100.0)  # price improvement


async def test_slippage_can_be_filtered_by_strategy():
    bus, agent, _ = await build()
    for n, (strategy_id, price) in enumerate([("a", 2505.0), ("b", 2500.0)]):
        i = intent(strategy_id=strategy_id)
        await bus.publish(Topics.INTENTS, i)
        await bus.publish(Topics.ORDERS, order(i.id, oid=f"o{n}"))
        await bus.publish(Topics.FILLS, fill(oid=f"o{n}", price=price))
    assert agent.slippage_summary("a")["mean_bps"] == pytest.approx(20.0)
    assert agent.slippage_summary("b")["mean_bps"] == pytest.approx(0.0)
    assert agent.slippage_summary("missing")["count"] == 0


async def test_fills_without_a_known_intent_are_ignored_for_slippage():
    bus, agent, _ = await build()
    await bus.publish(Topics.FILLS, fill(oid="unknown"))
    assert len(agent.slippage) == 0
    assert agent.counts["fills"] == 1


async def test_status_reports_the_portfolio():
    _bus, agent, portfolio = await build()
    portfolio.apply_fill(fill())
    portfolio.mark(SYM, 2510.0)
    status = agent.status()["portfolio"]
    assert status["open_positions"] == 1
    assert status["gross_exposure"] == pytest.approx(25_100.0)
    assert status["equity"] > 1_000_000


async def test_protective_exits_are_excluded_from_slippage():
    """A stop-loss fill measures the stop, not how well the signal was executed."""
    bus, agent, _ = await build()
    await bus.publish(
        Topics.ticks(SYM), Tick(symbol=SYM, ts=TS, ltp=2500.0, bid=2499.0, ask=2501.0)
    )
    i = intent()
    await bus.publish(Topics.INTENTS, i)
    await bus.publish(Topics.ORDERS, order(i.id))
    await bus.publish(Topics.FILLS, fill(price=2500.0))

    stop = order(i.id, oid="stop1")
    stop.meta["protective"] = True
    await bus.publish(Topics.ORDERS, stop)
    await bus.publish(Topics.FILLS, fill(oid="stop1", price=2475.0, side=Side.SELL))

    assert agent.counts["fills"] == 2
    assert agent.slippage_summary()["count"] == 1  # only the entry
    assert agent.slippage_summary()["mean_bps"] == pytest.approx(0.0)
