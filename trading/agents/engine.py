"""Assembles the five agents into a running engine.

The wiring is the same for live, paper and replay - only the broker and the driver
differ - which is the whole point: a backtest exercises the production code path.

Subscription order matters on the in-memory bus, which dispatches handlers
synchronously in the order they subscribed. The simulated broker is wired to
``bars.*`` **first**, so a bar fills resting orders before any strategy gets to
react to that same bar. Without that, a backtest would fill on prices a live
system could never have had.

Time: when a bar is processed, the clock reads the moment that bar *completed*
(``MarketCalendar.bar_end``), not its start. A decision taken on the 09:56 bar can
only exist from 09:57, and the risk agent's market-hours and cut-off rules must
see that.

MIS square-off: brokers close intraday positions shortly before the close. Dhan
does it on its own; for the paper and backtest brokers the engine does it
``mis_square_off_minutes`` before each exchange's close, so an intraday strategy
can never carry a position overnight in a simulation when it could not live.

Constraint 5 lives here too: :func:`confirm_live_trading` is the startup gate that
real orders must pass, on top of ``LIVE_TRADING=true`` and a Dhan broker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trading.agents.data import DataAgent, DataAgentConfig
from trading.agents.execution import ExecutionAgent, ExecutionConfig
from trading.agents.monitor import MonitorAgent, MonitorConfig
from trading.agents.portfolio import Portfolio
from trading.agents.risk import RiskAgent, RiskLimits
from trading.agents.signal import ModelProvider, SignalAgent
from trading.brokers.base import Broker, Instrument
from trading.brokers.lots import LotSizes
from trading.brokers.paper import PaperBroker
from trading.brokers.symbols import contract_multiplier, parse_symbol
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock, MarketCalendar, SystemClock
from trading.core.config import Settings
from trading.core.types import Bar, Fill, Interval, Order, ProductType
from trading.features.features import DEFAULT_SPEC, FeatureSpec
from trading.strategies.schema import StrategyConfig

log = logging.getLogger(__name__)


class LiveTradingRefused(RuntimeError):
    """The live-trading gate was not satisfied; nothing was started."""


def confirm_live_trading(
    settings: Settings,
    broker_name: str,
    *,
    prompt: Callable[[str], str] | None = None,
) -> bool:
    """Constraint 5: real orders need all three of LIVE_TRADING, Dhan, and a human.

    Returns True only when every condition holds. Any other combination means the
    engine runs on paper, which is the default.
    """
    if not settings.live_trading:
        log.info("LIVE_TRADING is false: running on paper")
        return False
    if broker_name != "dhan":
        log.warning("LIVE_TRADING is true but the broker is %r: running on paper", broker_name)
        return False
    ask = prompt or input
    answer = ask(
        "\n"
        "*** LIVE TRADING ***\n"
        f"    broker      : {broker_name}\n"
        f"    client id   : {settings.dhan_client_id}\n"
        "Real orders will be sent with real money.\n"
        "Type LIVE to confirm, anything else to stay on paper: "
    )
    confirmed = answer.strip() == "LIVE"
    log.warning("live trading %s", "CONFIRMED" if confirmed else "declined; running on paper")
    return confirmed


@dataclass
class EngineConfig:
    strategies: list[StrategyConfig]
    limits: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    spec: FeatureSpec = field(default_factory=lambda: DEFAULT_SPEC)
    starting_equity: float = 1_000_000.0
    publish_ticks: bool = True
    manage_interval_seconds: float = 1.0
    mis_square_off_minutes: int | None = 10  # before each exchange's close; None: off

    @property
    def symbols(self) -> list[str]:
        seen: list[str] = []
        for s in self.strategies:
            for sym in s.symbols:
                if sym not in seen:
                    seen.append(sym)
        return seen

    @property
    def intervals(self) -> list[Interval]:
        return sorted({s.interval for s in self.strategies}, key=lambda i: i.seconds)


class TradingEngine:
    def __init__(
        self,
        bus: MessageBus,
        broker: Broker,
        calendar: MarketCalendar,
        config: EngineConfig,
        *,
        instruments: dict[str, Instrument] | None = None,
        models: ModelProvider | None = None,
        clock: Clock | None = None,
        live: bool = False,
    ) -> None:
        self.bus = bus
        self.broker = broker
        self.calendar = calendar
        self.cfg = config
        self.clock = clock or SystemClock()
        self.instruments = instruments or {}
        self.live = live
        lots = LotSizes.from_instruments(self.instruments)
        lots.require(config.symbols)  # fail before anything starts, never guess a lot
        self.lots = lots

        self.portfolio = Portfolio(
            starting_equity=config.starting_equity, multiplier_for=contract_multiplier
        )
        self.data = DataAgent(
            bus,
            calendar,
            DataAgentConfig(
                symbols=config.symbols,
                intervals=config.intervals,
                spec=config.spec,
                publish_ticks=config.publish_ticks,
            ),
            source=broker,
            clock=self.clock,
        )
        self.signals = [
            SignalAgent(
                bus, strategy, self.portfolio, models=models, lot_size_for=lots, clock=self.clock
            )
            for strategy in config.strategies
        ]
        self.risk = RiskAgent(
            bus, self.portfolio, calendar, config.limits, lot_size_for=lots, clock=self.clock
        )
        self.execution = ExecutionAgent(
            bus, broker, config.execution, instruments=self.instruments, lots=lots, clock=self.clock
        )
        self.monitor = MonitorAgent(bus, self.portfolio, config.monitor, clock=self.clock)
        self._queue: asyncio.Queue[Order | Fill] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.started = False

    @property
    def agents(self) -> list[object]:
        # execution subscribes to intents before risk does, so on an ordered bus its
        # cache is populated in time to cross-check the approval it will receive
        return [self.data, *self.signals, self.execution, self.risk, self.monitor]

    # ------------------------------------------------------------------ lifecycle
    async def start(self, *, reconcile: bool = True) -> None:
        """Start every agent. Order of subscription is deliberate - see the module docstring."""
        if self.started:
            return
        # 1. a simulated broker must see each bar before any strategy reacts to it
        if isinstance(self.broker, PaperBroker):
            await self.bus.subscribe(Topics.BARS_ALL, self._feed_broker)
            self._queue = self.broker.update_queue()
        # 2. the portfolio must see fills before risk evaluates the next intent
        await self.bus.subscribe(Topics.FILLS, self._apply_fill)
        await self.bus.subscribe(Topics.BARS_ALL, self._mark_price)
        # 3. agents
        for agent in self.agents:
            await agent.start()  # type: ignore[attr-defined]
        if reconcile:
            await self.sync()
        self.started = True
        log.info(
            "engine started: %d strategies, %d symbols, live=%s",
            len(self.cfg.strategies),
            len(self.cfg.symbols),
            self.live,
        )

    async def stop(self) -> None:
        if not self.started:
            return
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for agent in self.agents:
            await agent.stop()  # type: ignore[attr-defined]
        if self._queue is not None and isinstance(self.broker, PaperBroker):
            self.broker.release_queue(self._queue)
            self._queue = None
        self.started = False

    async def sync(self) -> dict[str, int]:
        """Reconcile orders and adopt the broker's positions (constraint 4)."""
        stats = await self.execution.reconcile()
        try:
            funds = await self.broker.funds()
            positions = await self.broker.positions()
        except Exception as e:
            log.warning("could not sync portfolio from broker: %s", e)
            return stats
        self.portfolio.sync_from_broker(funds, positions)
        self.portfolio.starting_equity = funds.equity - self.portfolio.net_pnl
        return stats

    # ------------------------------------------------------------------ bus wiring
    async def _feed_broker(self, _topic: str, bar: Bar) -> None:
        """Let the simulated broker match resting orders against this bar, then
        hand every resulting update to the execution agent before anything else
        sees the bar."""
        self.broker.on_bar(bar)  # type: ignore[union-attr]
        await self.drain_updates()

    async def _apply_fill(self, _topic: str, fill: Fill) -> None:
        self.portfolio.apply_fill(fill)

    async def _mark_price(self, _topic: str, bar: Bar) -> None:
        self.portfolio.mark(bar.symbol, bar.close, bar.ts)

    async def drain_updates(self) -> int:
        """Hand queued broker updates to the execution agent. Deterministic: no sleeps."""
        if self._queue is None:
            return 0
        n = 0
        while not self._queue.empty():
            await self.execution.handle_update(self._queue.get_nowait())
            n += 1
        return n

    # ------------------------------------------------------------------ drivers
    def bar_end(self, bar: Bar) -> datetime:
        return self.calendar.bar_end(parse_symbol(bar.symbol).exchange, bar.ts, bar.interval)

    async def step(self, bar: Bar) -> datetime:
        """Process one completed bar; returns the time it completed."""
        end = self.bar_end(bar)
        self.clock_set(end)
        await self.data.emit_bar(bar)
        await self.drain_updates()
        await self.execution.manage(end)
        await self.drain_updates()
        await self.square_off_due(end)
        return end

    async def replay(self, bars: Iterable[Bar]) -> int:
        """Drive the engine from archived bars (backtests and tests).

        Bars must arrive in order of completion time; for one interval that is
        simply timestamp order.
        """
        n = 0
        for bar in bars:
            await self.step(bar)
            n += 1
        return n

    async def square_off_due(self, now: datetime | None = None) -> int:
        """Close MIS positions once an exchange is inside its square-off window.

        Only for brokers we simulate; a real broker squares off on its own.
        """
        minutes = self.cfg.mis_square_off_minutes
        if minutes is None or not isinstance(self.broker, PaperBroker):
            return 0
        now = now or self.clock.now()
        due: list[str] = []
        for pos in await self.broker.positions():
            if pos.qty == 0 or pos.product is not ProductType.MIS:
                continue
            bounds = self.calendar.session_bounds(parse_symbol(pos.symbol).exchange, now.date())
            if bounds and now >= bounds[1] - timedelta(minutes=minutes):
                due.append(pos.symbol)
        if not due:
            return 0
        await self.execution.cancel_where(
            lambda o: o.product is ProductType.MIS and o.symbol in due, "MIS square-off"
        )
        closed = await self.broker.liquidate(
            product=ProductType.MIS, symbols=due, reason="MIS square-off"
        )
        await self.drain_updates()
        log.info("squared off %d MIS positions at %s", len(closed), now)
        return len(closed)

    def clock_set(self, ts: datetime) -> None:
        setter = getattr(self.clock, "set", None)
        if setter is not None:
            setter(ts)

    async def run_live(self, *, warmup: bool = True) -> None:
        """Stream the broker's live feed until stopped."""
        if warmup:
            await self.data.warmup()
        self._tasks.append(asyncio.create_task(self.execution.pump_updates(), name="exec-pump"))
        self._tasks.append(asyncio.create_task(self._manage_loop(), name="exec-manage"))
        await self.data.run_live()

    async def _manage_loop(self) -> None:
        while self.started:
            await asyncio.sleep(self.cfg.manage_interval_seconds)
            await self.execution.manage()
            await self.square_off_due()
            await self.monitor.check_health()

    def status(self) -> dict[str, object]:
        return {
            "live": self.live,
            "broker": getattr(self.broker, "name", "?"),
            "strategies": [s.strategy.id for s in self.signals],
            "bars": self.data.bars_published,
            "intents": sum(s.intents_emitted for s in self.signals),
            "approved": self.risk.approved_count,
            "rejected": self.risk.rejected_count,
            "orders": self.execution.placed_count,
            "fills": self.execution.fills_published,
            **self.monitor.status(),
        }


def build_engine(
    bus: MessageBus,
    broker: Broker,
    calendar: MarketCalendar,
    strategies: Sequence[StrategyConfig],
    **kw,  # type: ignore[no-untyped-def]
) -> TradingEngine:
    return TradingEngine(bus, broker, calendar, EngineConfig(strategies=list(strategies)), **kw)
