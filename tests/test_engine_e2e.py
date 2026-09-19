"""Phase 3 acceptance test.

Replays archived bars through the whole engine on the in-memory bus - data agent
to features, a rule-based strategy to an intent, the risk agent to an approval,
the execution agent to the simulated broker - and checks the resulting cash,
position and PnL in SQLite against fees computed by hand in this file.

Nothing here calls the production cost model: the expected numbers are worked out
independently so a mistake in ``backtest/costs.py`` cannot hide behind itself.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from trading.agents.data import merge_bars
from trading.agents.engine import EngineConfig, TradingEngine, confirm_live_trading
from trading.agents.execution import ExecutionConfig
from trading.agents.risk import RiskLimits
from trading.brokers.base import Instrument
from trading.brokers.paper import PaperBroker, PaperConfig
from trading.brokers.paper_store import PaperStore
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import InMemoryBus
from trading.core.clock import SimClock
from trading.core.config import Settings
from trading.core.types import (
    Exchange,
    InstrumentKind,
    Interval,
    OrderStatus,
    ProductType,
    Side,
)
from trading.strategies.schema import StrategyConfig
from trading.training.ingest import Archive, bars_to_frame

from .conftest import make_bars

SYM = "NSE:RELIANCE"
DAY = date(2026, 9, 18)
STARTING_CASH = 1_000_000.0
SLIPPAGE_BPS = 2.0

# 41 flat bars to warm the features, a step up that triggers the entry, then a
# step down four bars later that triggers the exit.
PRICES = [100.0] * 41 + [101.0] * 4 + [100.0] * 11

STRATEGY = {
    "id": "e2e_step",
    "name": "Buy the step, sell the drop",
    "symbols": [SYM],
    "interval": "1m",
    "product": "MIS",
    "expected_edge_bps": 30.0,
    "rules": {
        "long": {"all": [{"feature": "ret_1", "op": "gt", "value": 0.001}]},
        "exit_long": {"all": [{"feature": "ret_1", "op": "lt", "value": -0.001}]},
    },
    "sizing": {"mode": "fixed_qty", "qty": 10},
    "execution": {"urgency": "AGGRESSIVE", "limit_band_bps": 10, "ttl_seconds": 300},
}

RELIANCE = Instrument(
    symbol=SYM,
    exchange=Exchange.NSE,
    kind=InstrumentKind.EQUITY,
    broker_id="2885",
    broker_segment="NSE_EQ",
    lot_size=1,
    tick_size=0.05,
)


# --------------------------------------------------------------------------- hand-computed fees


def intraday_equity_fees(side: Side, qty: int, price: float) -> float:
    """NSE equity intraday charges, written out from the rate card.

    Brokerage 0.03% capped at Rs 20 a side; STT 0.025% on the sell leg only;
    exchange transaction charge 0.00297%; SEBI turnover fee Rs 10 per crore;
    stamp duty 0.003% on the buy leg only; GST 18% on brokerage + exchange + SEBI.
    """
    value = qty * price
    brokerage = min(0.0003 * value, 20.0)
    stt = 0.00025 * value if side is Side.SELL else 0.0
    exchange = 0.0000297 * value
    sebi = 0.000001 * value
    stamp = 0.00003 * value if side is Side.BUY else 0.0
    gst = 0.18 * (brokerage + exchange + sebi)
    return brokerage + stt + exchange + sebi + stamp + gst


def slipped(price: float, side: Side) -> float:
    return round(price * (1 + side.sign * SLIPPAGE_BPS / 10_000), 4)


# --------------------------------------------------------------------------- harness


def build_engine(bus, broker, calendar, clock, strategy_dict, **limits_kw):
    strategy = StrategyConfig.model_validate(strategy_dict)
    return TradingEngine(
        bus,
        broker,
        calendar,
        EngineConfig(
            strategies=[strategy],
            limits=RiskLimits(**limits_kw),
            execution=ExecutionConfig(place_bracket_stop=False),
            starting_equity=STARTING_CASH,
        ),
        instruments={SYM: RELIANCE},
        clock=clock,
        live=False,
    )


def make_broker(store, clock, cash=STARTING_CASH):
    return PaperBroker(
        config=PaperConfig(
            starting_cash=cash,
            slippage_bps=SLIPPAGE_BPS,
            account_id="e2e",
            multiplier_for=contract_multiplier,
        ),
        store=store,
        clock=clock,
    )


@pytest.fixture
def archived_bars(tmp_path, calendar):
    """Write the bars to a real Parquet archive and read them back out."""
    archive = Archive(tmp_path / "archive")
    bars = make_bars(calendar, PRICES, symbol=SYM, day=DAY)
    archive.write(SYM, Interval.M1, bars_to_frame(bars))
    replayed = archive.read_bars(SYM, Interval.M1, DAY, DAY)
    assert replayed == bars  # the archive round trip is lossless
    return replayed


# --------------------------------------------------------------------------- the acceptance test


async def test_archive_replay_produces_hand_computed_cash_and_pnl(
    tmp_path, calendar, archived_bars
):
    db_url = f"sqlite:///{tmp_path / 'engine.db'}"
    clock = SimClock(archived_bars[0].ts - timedelta(minutes=1))
    broker = make_broker(PaperStore(db_url), clock)
    bus = InMemoryBus()
    engine = build_engine(bus, broker, calendar, clock, STRATEGY)

    await engine.start(reconcile=False)
    assert await engine.replay(archived_bars) == len(archived_bars)
    await engine.stop()

    # --- the strategy fired exactly twice: one entry, one exit
    status = engine.status()
    assert status["intents"] == 2
    assert status["approved"] == 2 and status["rejected"] == 0
    assert status["orders"] == 2 and status["fills"] == 2

    fills = await broker.fills()
    entry, exit_ = fills
    assert [f.side for f in fills] == [Side.BUY, Side.SELL]

    # --- prices: AGGRESSIVE crosses with a marketable limit, filled at last +/- slippage
    expected_entry_price = slipped(101.0, Side.BUY)
    expected_exit_price = slipped(100.0, Side.SELL)
    assert entry.price == expected_entry_price == 101.0202
    assert exit_.price == expected_exit_price == 99.98
    assert entry.ts == datetime.combine(DAY, datetime.min.time(), tzinfo=entry.ts.tzinfo).replace(
        hour=9, minute=56
    )
    assert exit_.ts.hour == 10 and exit_.ts.minute == 0

    # --- fees, computed independently above
    entry_fees = intraday_equity_fees(Side.BUY, 10, expected_entry_price)
    exit_fees = intraday_equity_fees(Side.SELL, 10, expected_exit_price)
    assert entry.fees.total == pytest.approx(entry_fees, abs=1e-3)
    assert exit_.fees.total == pytest.approx(exit_fees, abs=1e-3)
    # each component is rounded to 4 decimals by the fee model, hence the tolerance
    assert entry.fees.stt == 0.0
    assert exit_.fees.stt == pytest.approx(0.00025 * 10 * 99.98, abs=1e-3)
    assert entry.fees.stamp > 0 and exit_.fees.stamp == 0.0

    # --- cash and PnL
    gross_pnl = (expected_exit_price - expected_entry_price) * 10
    expected_cash = (
        STARTING_CASH
        - 10 * expected_entry_price
        - entry_fees
        + 10 * expected_exit_price
        - exit_fees
    )
    assert gross_pnl == pytest.approx(-10.402)

    funds = await broker.funds()
    assert funds.cash == pytest.approx(expected_cash, abs=1e-3)
    assert funds.margin_used == 0.0

    positions = await broker.positions()
    assert len(positions) == 1
    position = positions[0]
    assert position.qty == 0 and position.product is ProductType.MIS
    assert position.realised_pnl == pytest.approx(gross_pnl)
    assert position.fees_paid == pytest.approx(entry_fees + exit_fees, abs=1e-3)
    assert position.net_pnl == pytest.approx(expected_cash - STARTING_CASH, abs=1e-3)

    # the engine's own view agrees with the broker's
    assert engine.portfolio.net_qty(SYM, ProductType.MIS) == 0
    assert engine.portfolio.realised_pnl == pytest.approx(gross_pnl)
    assert engine.portfolio.equity == pytest.approx(expected_cash, abs=1e-3)

    # --- and all of it survives a restart, read back from SQLite
    reloaded = make_broker(PaperStore(db_url), clock)
    assert reloaded.cash == pytest.approx(expected_cash, abs=1e-3)
    assert len(await reloaded.fills()) == 2
    assert {o.status for o in await reloaded.orders()} == {OrderStatus.FILLED}
    restored = (await reloaded.positions())[0]
    assert restored.qty == 0
    assert restored.realised_pnl == pytest.approx(gross_pnl)
    assert restored.fees_paid == pytest.approx(entry_fees + exit_fees, abs=1e-3)


# --------------------------------------------------------------------------- the risk gate


async def test_the_same_run_is_blocked_when_the_edge_cannot_pay_the_costs(
    tmp_path, calendar, archived_bars
):
    """Identical bars and rules, but the strategy claims a 1 bp edge: nothing trades."""
    clock = SimClock(archived_bars[0].ts - timedelta(minutes=1))
    broker = make_broker(PaperStore(f"sqlite:///{tmp_path / 'thin.db'}"), clock)
    engine = build_engine(
        InMemoryBus(), broker, calendar, clock, {**STRATEGY, "expected_edge_bps": 1.0}
    )
    await engine.start(reconcile=False)
    await engine.replay(archived_bars)
    await engine.stop()

    assert engine.status()["intents"] == 1  # the rule still fired
    assert engine.risk.approved_count == 0
    assert engine.risk.rejections == {"cost_threshold": 1}
    assert await broker.orders() == []
    assert (await broker.funds()).cash == STARTING_CASH


async def test_the_kill_switch_stops_the_engine_mid_replay(tmp_path, calendar, archived_bars):
    clock = SimClock(archived_bars[0].ts - timedelta(minutes=1))
    broker = make_broker(PaperStore(f"sqlite:///{tmp_path / 'kill.db'}"), clock)
    bus = InMemoryBus()
    engine = build_engine(bus, broker, calendar, clock, STRATEGY)
    await engine.start(reconcile=False)
    await engine.monitor.kill("manual stop")
    await engine.replay(archived_bars)
    await engine.stop()

    assert engine.risk.killed and engine.execution.halted
    assert engine.risk.rejections == {"kill_switch": 1}
    assert await broker.orders() == []


async def test_position_limit_trims_the_order_that_reaches_the_broker(
    tmp_path, calendar, archived_bars
):
    """Risk cuts the size and execution honours the cut, not the strategy's wish."""
    clock = SimClock(archived_bars[0].ts - timedelta(minutes=1))
    broker = make_broker(PaperStore(f"sqlite:///{tmp_path / 'trim.db'}"), clock)
    strategy = {**STRATEGY, "sizing": {"mode": "fixed_qty", "qty": 100}}
    engine = build_engine(
        InMemoryBus(), broker, calendar, clock, strategy, max_position_value=606.0
    )
    await engine.start(reconcile=False)
    await engine.replay(archived_bars)
    await engine.stop()

    orders = await broker.orders()
    assert orders[0].qty == 6  # 606 / 101 = 6 shares, not the 100 asked for
    assert engine.portfolio.net_qty(SYM, ProductType.MIS) == 0  # opened and closed


async def test_multi_symbol_replay_keeps_positions_separate(tmp_path, calendar):
    other = "NSE:TCS"
    clock = SimClock(calendar.session_bars("NSE", DAY, Interval.M1)[0] - timedelta(minutes=1))
    broker = make_broker(PaperStore(f"sqlite:///{tmp_path / 'multi.db'}"), clock)
    strategy = {**STRATEGY, "symbols": [SYM, other]}
    engine = build_engine(
        InMemoryBus(),
        broker,
        calendar,
        clock,
        strategy,
    )
    engine.instruments[other] = RELIANCE.model_copy(update={"symbol": other, "broker_id": "11536"})
    bars = merge_bars(
        [
            make_bars(calendar, PRICES, symbol=SYM, day=DAY),
            make_bars(calendar, PRICES, symbol=other, day=DAY),
        ]
    )
    await engine.start(reconcile=False)
    await engine.replay(bars)
    await engine.stop()

    assert engine.status()["intents"] == 4  # an entry and an exit for each symbol
    positions = {p.symbol: p for p in await broker.positions()}
    assert set(positions) == {SYM, other}
    for position in positions.values():
        assert position.qty == 0
        assert position.realised_pnl == pytest.approx(-10.402)


# --------------------------------------------------------------------------- constraint 5


def test_live_trading_needs_all_three_conditions():
    """Constraint 5: LIVE_TRADING, a Dhan broker, and a typed confirmation."""
    paper = Settings(_env_file=None)
    assert confirm_live_trading(paper, "paper", prompt=lambda _: "LIVE") is False

    live_env = Settings(_env_file=None, live_trading=True, broker="dhan", dhan_client_id="1")
    assert confirm_live_trading(live_env, "paper", prompt=lambda _: "LIVE") is False  # wrong broker
    assert confirm_live_trading(live_env, "dhan", prompt=lambda _: "yes") is False  # not the word
    assert confirm_live_trading(live_env, "dhan", prompt=lambda _: "") is False
    assert confirm_live_trading(live_env, "dhan", prompt=lambda _: " LIVE ") is True

    asked: list[str] = []

    def record(message: str) -> str:
        asked.append(message)
        return "no"

    confirm_live_trading(live_env, "dhan", prompt=record)
    assert "LIVE TRADING" in asked[0] and "real money" in asked[0]


async def test_engine_status_is_reportable(tmp_path, calendar, archived_bars):
    clock = SimClock(archived_bars[0].ts - timedelta(minutes=1))
    broker = make_broker(PaperStore(f"sqlite:///{tmp_path / 'status.db'}"), clock)
    engine = build_engine(InMemoryBus(), broker, calendar, clock, STRATEGY)
    await engine.start(reconcile=False)
    await engine.replay(archived_bars)
    status = engine.status()
    await engine.stop()
    assert status["broker"] == "paper" and status["live"] is False
    assert status["strategies"] == ["e2e_step"]
    assert status["bars"] == len(archived_bars)
    assert status["portfolio"]["open_positions"] == 0
    assert status["slippage"]["count"] == 2
    assert status["slippage"]["mean_bps"] == pytest.approx(2.0)  # the paper slippage model
