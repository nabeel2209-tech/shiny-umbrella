"""Risk agent: one test per rule, plus trimming, exits and the kill switch.

Constraint 3 lives here - nothing trades unless this agent says so.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from trading.agents.portfolio import Portfolio
from trading.agents.risk import RiskAgent, RiskLimits, kelly_fraction
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    ControlCommand,
    Fill,
    OrderIntent,
    ProductType,
    RiskApproval,
    RiskRejection,
    Side,
)

SYM = "NSE:RELIANCE"
OPEN_TIME = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


@pytest.fixture
def portfolio():
    p = Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    p.roll_day(OPEN_TIME)
    return p


@pytest.fixture
def clock():
    return SimClock(OPEN_TIME)


def make_agent(calendar, portfolio, clock, limits=None, **kw):
    return RiskAgent(InMemoryBus(), portfolio, calendar, limits or RiskLimits(), clock=clock, **kw)


def intent(**kw) -> OrderIntent:
    base = dict(
        ts=OPEN_TIME,
        strategy_id="s1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        product=ProductType.MIS,
        reference_price=2500.0,
        expected_edge_bps=50.0,
    )
    return OrderIntent(**{**base, **kw})


def fill(symbol=SYM, side=Side.BUY, qty=10, price=2500.0, product=ProductType.MIS, ts=OPEN_TIME):
    return Fill(
        order_id="o1", symbol=symbol, side=side, qty=qty, price=price, ts=ts, product=product
    )


def check(agent, i=None, **kw):
    approval, rejection = agent.evaluate(i or intent(**kw))
    return approval, rejection


# --------------------------------------------------------------------------- happy path


def test_a_clean_intent_is_approved_with_a_token(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    approval, rejection = check(agent)
    assert rejection is None and approval is not None
    assert approval.approved_qty == 10
    assert approval.token and approval.expires_at > clock.now()
    assert approval.intent.id == approval.intent_id  # self-contained
    assert {"kill_switch", "market_hours", "instrument", "daily_loss", "cost_threshold"} <= set(
        approval.checks
    )
    assert "trimmed" not in approval.checks


# --------------------------------------------------------------------------- rule: kill switch


async def test_kill_switch_blocks_everything_until_resume(calendar, portfolio, clock):
    bus = InMemoryBus()
    agent = RiskAgent(bus, portfolio, calendar, clock=clock)
    await agent.start()
    await bus.publish(
        Topics.CONTROL, ControlCommand(ts=clock.now(), command="KILL", reason="by hand")
    )
    assert agent.killed
    _, rejection = check(agent)
    assert rejection is not None and rejection.rule == "kill_switch"
    assert "by hand" in rejection.reason
    await bus.publish(Topics.CONTROL, ControlCommand(ts=clock.now(), command="RESUME"))
    assert not agent.killed
    approval, _ = check(agent)
    assert approval is not None


# --------------------------------------------------------------------------- rule: strategy enabled


async def test_pausing_one_strategy_leaves_the_others_alone(calendar, portfolio, clock):
    bus = InMemoryBus()
    agent = RiskAgent(bus, portfolio, calendar, clock=clock)
    await agent.start()
    await bus.publish(Topics.CONTROL, ControlCommand(ts=clock.now(), command="PAUSE", reason="s1"))
    _, rejection = check(agent, strategy_id="s1")
    assert rejection is not None and rejection.rule == "strategy_enabled"
    approval, _ = check(agent, strategy_id="s2")
    assert approval is not None
    await bus.publish(
        Topics.CONTROL, ControlCommand(ts=clock.now(), command="UNPAUSE", reason="s1")
    )
    assert check(agent, strategy_id="s1")[0] is not None


# --------------------------------------------------------------------------- rule: market hours


def test_market_hours_guard(calendar, portfolio):
    before_open = SimClock(datetime(2026, 9, 18, 8, 0, tzinfo=IST))
    agent = make_agent(calendar, portfolio, before_open)
    _, rejection = check(agent)
    assert rejection is not None and rejection.rule == "market_hours"
    assert "next open 2026-09-18 09:15" in rejection.reason

    holiday = SimClock(datetime(2026, 10, 2, 11, 0, tzinfo=IST))
    assert check(make_agent(calendar, portfolio, holiday))[1].rule == "market_hours"

    # MCX runs late: 22:00 is closed for NSE but open for commodities
    late = SimClock(datetime(2026, 9, 18, 22, 0, tzinfo=IST))
    # 1 GOLDM lot is 100 g quoted per 10 g, so its contract value is 15 lakh -
    # the default per-instrument and per-order caps are too small to hold one
    agent = make_agent(
        calendar,
        portfolio,
        late,
        RiskLimits(max_order_value=2_000_000, max_position_value=2_000_000),
    )
    assert check(agent)[1].rule == "market_hours"
    approval, _ = check(
        agent,
        i=intent(
            symbol="MCX:GOLDM-OCT26",
            product=ProductType.NRML,
            qty=1,
            reference_price=150_000.0,
        ),
    )
    assert approval is not None

    off = make_agent(calendar, portfolio, before_open, RiskLimits(require_market_open=False))
    assert check(off)[0] is not None


def test_exits_can_be_allowed_when_the_market_is_shut(calendar, portfolio):
    after_close = SimClock(datetime(2026, 9, 18, 16, 0, tzinfo=IST))
    portfolio.apply_fill(fill())  # long 10
    limits = RiskLimits(allow_exits_when_closed=True)
    agent = make_agent(calendar, portfolio, after_close, limits)
    approval, _ = check(agent, side=Side.SELL)
    assert approval is not None and "exit allowed" in approval.checks["market_hours"]
    # an entry is still refused
    assert check(agent, side=Side.BUY)[1].rule == "market_hours"


# --------------------------------------------------------------------------- rule: instrument


@pytest.mark.parametrize(
    "kw,fragment",
    [
        ({"symbol": "NSE:NIFTY"}, "index"),
        ({"symbol": SYM, "product": ProductType.NRML}, "MIS or CNC"),
        ({"symbol": "NFO:NIFTY-OCT26", "product": ProductType.CNC}, "MIS or NRML"),
    ],
)
def test_instrument_rule_rejects_impossible_combinations(calendar, portfolio, clock, kw, fragment):
    agent = make_agent(calendar, portfolio, clock)
    _, rejection = check(agent, **kw)
    assert rejection is not None and rejection.rule == "instrument"
    assert fragment in rejection.reason


def test_quantity_must_be_a_whole_number_of_lots(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock, lot_size_for={"NFO:NIFTY-OCT26": 65})
    i = intent(symbol="NFO:NIFTY-OCT26", product=ProductType.NRML, qty=70, reference_price=25_000.0)
    _, rejection = check(agent, i=i)
    assert rejection is not None and rejection.rule == "instrument" and "lot 65" in rejection.reason


def test_single_order_value_cap(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock, RiskLimits(max_order_value=10_000))
    _, rejection = check(agent, qty=10)  # 10 x 2500 = 25,000
    assert (
        rejection is not None and rejection.rule == "instrument" and "over cap" in rejection.reason
    )


# --------------------------------------------------------------------------- rule: daily loss


def test_daily_loss_limit_blocks_new_trades(calendar, portfolio, clock):
    agent = make_agent(
        calendar, portfolio, clock, RiskLimits(max_daily_loss=5_000, max_daily_loss_fraction=None)
    )
    portfolio.apply_fill(fill(qty=100, price=2500.0))
    portfolio.mark(SYM, 2440.0)  # -6,000 on the day
    assert portfolio.day_pnl < -5_000
    _, rejection = check(agent)
    assert rejection is not None and rejection.rule == "daily_loss"


def test_daily_loss_fraction_is_the_tighter_of_the_two(calendar, portfolio, clock):
    limits = RiskLimits(max_daily_loss=100_000, max_daily_loss_fraction=0.01)  # 1% of 1,000,000
    agent = make_agent(calendar, portfolio, clock, limits)
    portfolio.apply_fill(fill(qty=100, price=2500.0))
    portfolio.mark(SYM, 2380.0)  # -12,000: inside 100,000 but past 10,000
    assert check(agent)[1].rule == "daily_loss"


async def test_breaching_the_loss_limit_halts_entries_for_the_day_only(calendar, portfolio, clock):
    """Entries stop the moment the limit is hit; exits still go through, so the
    losing position can be closed; the next trading day starts fresh."""
    bus = InMemoryBus()
    alerts = []

    async def on_alert(_t, m):
        alerts.append(m)

    await bus.subscribe(Topics.ALERTS, on_alert)
    limits = RiskLimits(max_daily_loss=1_000, max_daily_loss_fraction=None)
    agent = RiskAgent(bus, portfolio, calendar, limits, clock=clock)
    await agent.start()
    entry = fill(qty=10, price=2500.0)
    portfolio.apply_fill(entry)
    portfolio.mark(SYM, 2300.0)  # -2,000 open loss
    await bus.publish(Topics.FILLS, entry)

    assert agent.halted_day == OPEN_TIME.date() and not agent.killed
    assert any(a.level == "CRITICAL" and "entries halted" in a.message for a in alerts)
    _, rejection = check(agent, i=intent(symbol="NSE:TCS"))
    assert rejection.rule == "daily_loss" and "halted" in rejection.reason
    approval, _ = check(agent, side=Side.SELL)  # closing the loser is allowed
    assert approval is not None

    # recover the loss the same day: still halted, the halt is for the day
    portfolio.mark(SYM, 2600.0)
    assert check(agent, i=intent(symbol="NSE:TCS"))[1].rule == "daily_loss"

    # next session: a new day, entries resume
    next_day = datetime(2026, 9, 21, 10, 0, tzinfo=IST)
    clock.set(next_day)
    portfolio.mark(SYM, 2600.0, next_day)
    assert check(agent, i=intent(symbol="NSE:TCS", ts=next_day))[0] is not None


def test_a_manual_kill_is_different_from_a_loss_halt(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    portfolio.apply_fill(fill(qty=10))
    agent.killed = True
    assert check(agent, side=Side.SELL)[1].rule == "kill_switch"  # a kill stops exits too


# --------------------------------------------------------------------------- rule: intraday cutoff


def test_no_new_mis_entries_near_the_close(calendar, portfolio):
    late = SimClock(datetime(2026, 9, 18, 15, 16, tzinfo=IST))  # cutoff is 15:15
    agent = make_agent(calendar, portfolio, late)
    _, rejection = check(agent, product=ProductType.MIS)
    assert rejection.rule == "intraday_cutoff" and "15:15" in rejection.reason
    # delivery is not squared off, so it is unaffected
    assert check(agent, product=ProductType.CNC, qty=100, expected_edge_bps=80.0)[0] is not None
    # and an MIS exit is still allowed
    portfolio.apply_fill(fill(qty=10, ts=late.now()))
    assert check(agent, side=Side.SELL, product=ProductType.MIS)[0] is not None


def test_mis_cutoff_follows_each_exchange_close(calendar, portfolio):
    # MCX closes 23:55 in September (US DST), so 23:30 is still before its cutoff
    evening = SimClock(datetime(2026, 9, 18, 23, 30, tzinfo=IST))
    agent = make_agent(
        calendar, portfolio, evening, RiskLimits(max_order_value=5e6, max_position_value=5e6)
    )
    gold = intent(
        symbol="MCX:GOLDM-OCT26", product=ProductType.MIS, qty=1, reference_price=150_000.0
    )
    assert check(agent, i=gold)[0] is not None
    evening.set(datetime(2026, 9, 18, 23, 41, tzinfo=IST))
    assert check(agent, i=gold)[1].rule == "intraday_cutoff"
    off = make_agent(
        calendar,
        portfolio,
        evening,
        RiskLimits(mis_entry_cutoff_minutes=None, max_order_value=5e6, max_position_value=5e6),
    )
    assert check(off, i=gold)[0] is not None


# --------------------------------------------------------------------------- rule: cost threshold


def test_cost_threshold_rejects_an_edge_that_cannot_pay_for_the_trade(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    _, rejection = check(agent, expected_edge_bps=1.0)
    assert rejection is not None and rejection.rule == "cost_threshold"
    assert "does not clear" in rejection.reason
    assert check(agent, expected_edge_bps=50.0)[0] is not None


def test_cost_threshold_scales_with_the_instrument(calendar, portfolio, clock):
    """Delivery pays STT both ways, so it needs a much bigger edge than intraday."""
    agent = make_agent(calendar, portfolio, clock)
    # 100 shares at 2,500: intraday brokerage caps out at Rs 20 a side, so the
    # round trip is ~9 bps including slippage and a 20 bps edge clears 1.5x it
    assert check(agent, qty=100, product=ProductType.MIS, expected_edge_bps=20.0)[0] is not None
    # the same trade held overnight pays 0.1% STT each way: ~32 bps, so 20 fails
    rejection = check(agent, qty=100, product=ProductType.CNC, expected_edge_bps=20.0)[1]
    assert rejection.rule == "cost_threshold"
    assert check(agent, qty=100, product=ProductType.CNC, expected_edge_bps=60.0)[0] is not None


def test_edge_multiple_is_configurable(calendar, portfolio, clock):
    strict = make_agent(calendar, portfolio, clock, RiskLimits(min_edge_multiple=10.0))
    assert check(strict, expected_edge_bps=50.0)[1].rule == "cost_threshold"


# --------------------------------------------------------------------------- rule: Kelly


def test_kelly_formula():
    # f = p/a - q/b with a = 1%, b = payoff x a
    assert kelly_fraction(0.55, 1.0, 0.01) == pytest.approx(0.55 / 0.01 - 0.45 / 0.01)
    assert kelly_fraction(0.45, 1.0, 0.01) < 0  # no edge
    assert kelly_fraction(0.5, 2.0, 0.02) == pytest.approx(0.5 / 0.02 - 0.5 / 0.04)
    for bad in [(0.0, 1.0, 0.01), (1.0, 1.0, 0.01), (0.6, 0.0, 0.01), (0.6, 1.0, 0.0)]:
        assert kelly_fraction(*bad) == 0.0


def test_kelly_rejects_a_losing_bet(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    _, rejection = check(agent, prob=0.40, payoff_ratio=1.0)
    assert rejection is not None and rejection.rule == "kelly_size"
    assert "not positive" in rejection.reason


def test_kelly_cap_trims_the_size(calendar, portfolio, clock):
    limits = RiskLimits(max_kelly_fraction=0.02)  # 2% of 1,000,000 = 20,000 = 8 shares
    agent = make_agent(calendar, portfolio, clock, limits)
    approval, _ = check(agent, qty=100, prob=0.9, payoff_ratio=3.0)
    assert approval is not None and approval.approved_qty == 8
    assert approval.checks["trimmed"] == "100 -> 8"
    assert "capped 0.0200" in approval.checks["kelly_size"]


def test_kelly_is_skipped_without_a_probability(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    approval, _ = check(agent, prob=None)
    assert approval is not None and approval.approved_qty == 10
    assert "no probability" in approval.checks["kelly_size"]


# --------------------------------------------------------------------------- rule: position limit


def test_position_limit_trims_then_rejects(calendar, portfolio, clock):
    limits = RiskLimits(max_position_value=50_000)  # 20 shares at 2,500
    agent = make_agent(calendar, portfolio, clock, limits)
    approval, _ = check(agent, qty=30)
    assert approval is not None and approval.approved_qty == 20

    portfolio.apply_fill(fill(qty=20))  # now full
    _, rejection = check(agent, qty=5)
    assert rejection is not None and rejection.rule == "position_limit"
    assert "50,000" in rejection.reason


def test_position_limit_counts_only_the_same_product(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock, RiskLimits(max_position_value=50_000))
    portfolio.apply_fill(fill(qty=20, product=ProductType.CNC))
    approval, _ = check(agent, product=ProductType.MIS, qty=20)
    assert approval is not None and approval.approved_qty == 20


# --------------------------------------------------------------------------- rule: gross exposure


def test_gross_exposure_trims_across_instruments(calendar, portfolio, clock):
    limits = RiskLimits(max_gross_exposure=100_000, max_position_value=100_000)
    agent = make_agent(calendar, portfolio, clock, limits)
    portfolio.apply_fill(fill(symbol="NSE:TCS", qty=30, price=2500.0))  # 75,000 used
    portfolio.mark("NSE:TCS", 2500.0)
    approval, _ = check(agent, qty=30)  # wants 75,000, only 25,000 left
    assert approval is not None and approval.approved_qty == 10
    portfolio.apply_fill(fill(symbol="NSE:TCS", qty=10, price=2500.0))
    portfolio.mark("NSE:TCS", 2500.0)
    _, rejection = check(agent, qty=1)
    assert rejection is not None and rejection.rule == "gross_exposure"


def test_trim_to_zero_is_a_rejection(calendar, portfolio, clock):
    limits = RiskLimits(max_position_value=1_000)  # less than one 2,500 share
    agent = make_agent(calendar, portfolio, clock, limits)
    _, rejection = check(agent, qty=10)
    assert rejection is not None and rejection.rule == "position_limit"


# --------------------------------------------------------------------------- exits


def test_exits_skip_the_entry_rules(calendar, portfolio, clock):
    """Closing risk must never be blocked by an edge, Kelly or exposure limit."""
    limits = RiskLimits(max_position_value=1.0, max_gross_exposure=1.0, min_edge_multiple=1000.0)
    agent = make_agent(calendar, portfolio, clock, limits)
    portfolio.apply_fill(fill(qty=50))
    approval, _ = check(agent, side=Side.SELL, qty=50, expected_edge_bps=0.0, prob=0.1)
    assert approval is not None and approval.approved_qty == 50
    assert "cost_threshold" not in approval.checks and "kelly_size" not in approval.checks
    # but the kill switch still applies to exits
    agent.killed = True
    assert check(agent, side=Side.SELL, qty=50)[1].rule == "kill_switch"


def test_increasing_a_short_is_not_an_exit(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock)
    portfolio.apply_fill(fill(side=Side.SELL, qty=10))
    assert not agent._is_exit(intent(side=Side.SELL))
    assert agent._is_exit(intent(side=Side.BUY))


# --------------------------------------------------------------------------- bus behaviour


async def test_agent_publishes_approvals_and_rejections(calendar, portfolio, clock):
    bus = InMemoryBus()
    approvals: list[RiskApproval] = []
    rejections: list[RiskRejection] = []

    async def on_approved(_t, m):
        approvals.append(m)

    async def on_rejected(_t, m):
        rejections.append(m)

    await bus.subscribe(Topics.APPROVED, on_approved)
    await bus.subscribe(Topics.REJECTED, on_rejected)
    agent = RiskAgent(bus, portfolio, calendar, clock=clock)
    await agent.start()
    good = intent()
    bad = intent(expected_edge_bps=0.1)
    await bus.publish(Topics.INTENTS, good)
    await bus.publish(Topics.INTENTS, bad)
    assert [a.intent_id for a in approvals] == [good.id]
    assert [r.intent_id for r in rejections] == [bad.id]
    assert agent.approved_count == 1 and agent.rejected_count == 1
    assert agent.rejections == {"cost_threshold": 1}


def test_approval_ttl(calendar, portfolio, clock):
    agent = make_agent(calendar, portfolio, clock, RiskLimits(approval_ttl_seconds=45))
    approval, _ = check(agent)
    assert approval.expires_at == clock.now() + timedelta(seconds=45)
