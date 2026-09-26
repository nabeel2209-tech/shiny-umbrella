"""Starting, stopping and watching trading engines from the API.

One engine per (user, mode). ``paper`` trades a persistent fake-cash account on the
real Dhan feed; ``live`` sends real orders and therefore needs everything
constraint 5 asks for - ``LIVE_TRADING=true``, ``BROKER=dhan`` and a human typing
``LIVE`` (the web form's equivalent of the CLI's startup prompt) - and, in the API,
an admin.

The manager also carries out the kill switch (persisted, see ``events.KillSwitch``)
and "flatten": closing every position by sending **exit intents through the risk
agent**, so even an emergency close never bypasses constraint 3.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from trading.agents.engine import EngineConfig, TradingEngine, confirm_live_trading
from trading.api.auth import User
from trading.api.events import EventHub, KillSwitch
from trading.api.strategy_store import StrategyNotFound, StrategyStore
from trading.brokers.base import Broker, MarketData, NotConnected
from trading.brokers.lots import LotSizes, MissingLotSize
from trading.brokers.paper import PaperBroker, PaperConfig
from trading.brokers.paper_store import PaperStore
from trading.brokers.symbols import SymbolMap, UnknownSymbol, contract_multiplier
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import MarketCalendar
from trading.core.config import Settings
from trading.core.types import OrderIntent, Side, Urgency, now_ist
from trading.training.registry import ModelRegistry
from trading.training.signal_log import SignalLog

log = logging.getLogger(__name__)

BRIDGED_TOPICS = (
    Topics.ORDERS,
    Topics.FILLS,
    Topics.ALERTS,
    Topics.REJECTED,
    Topics.SIGNALS,
    Topics.CONTROL,
)


class EngineMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class EngineError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class EngineHandle:
    user: User
    mode: EngineMode
    engine: TradingEngine
    broker: Broker
    source: MarketData | None
    strategies: list[str]
    started_at: str
    task: asyncio.Task[None] | None = None
    error: str | None = None
    subscriptions: list[Any] = field(default_factory=list)


BrokerFactory = Callable[[], Awaitable[Any]]
SymbolMapLoader = Callable[[], Awaitable[SymbolMap | None]]


def live_trading_refusal(
    settings: Settings, user: User, confirm: str | None
) -> tuple[int, str] | None:
    """Why live trading cannot start, or None. Every condition is reported by name."""
    if not user.is_admin:
        return 403, "only an admin may trade the live account"
    if not settings.live_trading:
        return 403, "LIVE_TRADING is not true in the environment"
    if settings.broker != "dhan":
        return 403, f"BROKER is {settings.broker!r}; live trading needs dhan"
    if not confirm_live_trading(settings, settings.broker, prompt=lambda _: confirm or ""):
        return 400, "type LIVE to confirm that real orders will be sent"
    return None


def paper_account(user: User) -> str:
    return f"paper-{user.username}"


class EngineManager:
    def __init__(
        self,
        settings: Settings,
        calendar: MarketCalendar,
        strategies: StrategyStore,
        registry: ModelRegistry,
        signal_log: SignalLog,
        paper_store: PaperStore,
        hub: EventHub,
        kill: KillSwitch,
        *,
        feed_factory: BrokerFactory,
        live_broker_factory: BrokerFactory,
        symbol_map_loader: SymbolMapLoader,
    ) -> None:
        self.settings = settings
        self.calendar = calendar
        self.strategies = strategies
        self.registry = registry
        self.signal_log = signal_log
        self.paper_store = paper_store
        self.hub = hub
        self.kill_switch = kill
        self.feed_factory = feed_factory
        self.live_broker_factory = live_broker_factory
        self.symbol_map_loader = symbol_map_loader
        self._handles: dict[tuple[str, EngineMode], EngineHandle] = {}
        self._lock = asyncio.Lock()

    def get(self, user: User, mode: EngineMode) -> EngineHandle | None:
        return self._handles.get((user.id, mode))

    def running(self) -> list[EngineHandle]:
        return list(self._handles.values())

    # ------------------------------------------------------------------ start / stop
    async def start(
        self,
        user: User,
        mode: EngineMode,
        strategy_ids: list[str],
        *,
        confirm: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            state = self.kill_switch.state()
            if state.killed:
                raise EngineError(423, f"kill switch engaged by {state.by}: {state.reason}")
            if self.get(user, mode) is not None:
                raise EngineError(409, f"the {mode.value} engine is already running")
            if mode is EngineMode.LIVE:
                refusal = live_trading_refusal(self.settings, user, confirm)
                if refusal:
                    raise EngineError(*refusal)
            try:
                strategies = self.strategies.get_many(user.username, strategy_ids)
            except StrategyNotFound as e:
                raise EngineError(404, f"no strategy {e.args[0]!r}") from e
            if not strategies:
                raise EngineError(422, "choose at least one strategy")
            handle = await self._build(user, mode, strategies)
            self._handles[(user.id, mode)] = handle
        try:
            await handle.engine.start(reconcile=True)
        except Exception as e:
            await self._teardown(handle)
            self._handles.pop((user.id, mode), None)
            raise EngineError(502, f"engine failed to start: {e}") from e
        handle.task = asyncio.create_task(self._run(handle), name=f"engine-{user.username}-{mode}")
        self.hub.publish(
            user.id, {"type": "engine", "mode": mode.value, "data": {"state": "started"}}
        )
        log.warning("%s engine started for %s: %s", mode.value, user.username, handle.strategies)
        return self.status(user, mode)

    async def _build(self, user: User, mode: EngineMode, strategies: list[Any]) -> EngineHandle:
        symbols = sorted({s for st in strategies for s in st.symbols})
        try:
            if mode is EngineMode.LIVE:
                broker = await self.live_broker_factory()
                source: MarketData | None = broker
                if getattr(broker, "name", None) != "dhan":  # constraint 5, checked again
                    await _close(broker, source)
                    raise EngineError(403, "live trading is only allowed through the Dhan broker")
            else:
                source = await self.feed_factory()
                broker = PaperBroker(
                    source,
                    config=PaperConfig(
                        starting_cash=self.settings.paper_starting_cash,
                        slippage_bps=self.settings.paper_slippage_bps,
                        account_id=paper_account(user),
                        multiplier_for=contract_multiplier,
                    ),
                    store=self.paper_store,
                )
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(502, f"could not reach the broker: {e}") from e

        try:  # a connected Dhan broker already holds the instrument master
            symbol_map = source.symbols  # type: ignore[union-attr]
        except (AttributeError, NotConnected):
            symbol_map = await self.symbol_map_loader()
        try:
            LotSizes.from_symbol_map(symbol_map, symbols)
        except MissingLotSize as e:
            await _close(broker, source)
            raise EngineError(422, str(e)) from e
        instruments = {}
        if symbol_map is not None:
            for symbol in symbols:
                with contextlib.suppress(UnknownSymbol):
                    instruments[symbol] = symbol_map.resolve(symbol)

        bus = InMemoryBus()
        cfg = EngineConfig(strategies=strategies, starting_equity=self.settings.paper_starting_cash)
        engine = TradingEngine(
            bus,
            broker,
            self.calendar,
            cfg,
            instruments=instruments,
            models=ModelRegistry(self.registry.root, expected_spec=cfg.spec),
            signal_log=self.signal_log,
            live=mode is EngineMode.LIVE,
        )
        handle = EngineHandle(
            user=user,
            mode=mode,
            engine=engine,
            broker=broker,
            source=source,
            strategies=[s.id for s in strategies],
            started_at=now_ist().isoformat(),
        )
        for topic in BRIDGED_TOPICS:
            handle.subscriptions.append(await bus.subscribe(topic, self._bridge(handle)))
        return handle

    def _bridge(self, handle: EngineHandle):  # type: ignore[no-untyped-def]
        async def forward(topic: str, message: BaseModel) -> None:
            self.hub.publish(
                handle.user.id,
                {"type": topic, "mode": handle.mode.value, "data": message.model_dump(mode="json")},
            )

        return forward

    async def _run(self, handle: EngineHandle) -> None:
        try:
            await handle.engine.run_live(warmup=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            handle.error = f"{type(e).__name__}: {e}"
            log.exception("%s engine for %s died", handle.mode.value, handle.user.username)
            self.hub.publish(
                handle.user.id,
                {
                    "type": "engine",
                    "mode": handle.mode.value,
                    "data": {"state": "error", "error": handle.error},
                },
            )

    async def stop(
        self, user: User, mode: EngineMode, *, reason: str = "stopped by user"
    ) -> dict[str, Any]:
        handle = self._handles.pop((user.id, mode), None)
        if handle is None:
            raise EngineError(409, f"the {mode.value} engine is not running")
        await self._teardown(handle)
        self.hub.publish(
            user.id,
            {"type": "engine", "mode": mode.value, "data": {"state": "stopped", "reason": reason}},
        )
        log.warning("%s engine stopped for %s: %s", mode.value, user.username, reason)
        return {"running": False, "mode": mode.value, "reason": reason}

    async def _teardown(self, handle: EngineHandle) -> None:
        if handle.task is not None:
            handle.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handle.task
        with contextlib.suppress(Exception):
            await handle.engine.stop()
        for sub in handle.subscriptions:
            with contextlib.suppress(Exception):
                await sub.cancel()
        await _close(handle.broker, handle.source)

    async def shutdown(self) -> None:
        for (uid, mode), handle in list(self._handles.items()):
            self._handles.pop((uid, mode), None)
            await self._teardown(handle)

    # ------------------------------------------------------------------ status
    def status(self, user: User, mode: EngineMode) -> dict[str, Any]:
        handle = self.get(user, mode)
        kill = self.kill_switch.state()
        out: dict[str, Any] = {
            "mode": mode.value,
            "running": handle is not None,
            "kill_switch": kill.__dict__,
        }
        if mode is EngineMode.LIVE:
            out["live_allowed"] = self.settings.live_trading and self.settings.broker == "dhan"
        if handle is not None:
            out.update(
                strategies=handle.strategies,
                started_at=handle.started_at,
                error=handle.error,
                engine=handle.engine.status(),
            )
        return out

    # ------------------------------------------------------------------ control
    async def kill(self, user: User, reason: str) -> dict[str, Any]:
        state = self.kill_switch.engage(reason or "kill switch", user.username)
        for handle in self.running():
            await handle.engine.monitor.kill(reason or "kill switch", issued_by=user.username)
        self.hub.publish(None, {"type": "kill", "data": state.__dict__})
        return state.__dict__

    async def resume(self, user: User) -> dict[str, Any]:
        if not user.is_admin:
            raise EngineError(403, "only an admin may release the kill switch")
        state = self.kill_switch.release(user.username)
        for handle in self.running():
            await handle.engine.monitor.resume(issued_by=user.username)
        self.hub.publish(None, {"type": "resume", "data": state.__dict__})
        return state.__dict__

    async def flatten(self, user: User, mode: EngineMode) -> dict[str, Any]:
        """Close every open position with exit intents that pass through risk."""
        handle = self.get(user, mode)
        if handle is None:
            raise EngineError(409, f"the {mode.value} engine is not running")
        if self.kill_switch.engaged:
            raise EngineError(423, "the kill switch stops all orders, exits too; release it first")
        await handle.engine.execution.cancel_all("flatten requested")
        sent = []
        for pos in await handle.broker.positions():
            if pos.qty == 0:
                continue
            price = (
                handle.engine.portfolio.last_price.get(pos.symbol)
                or pos.last_price
                or pos.avg_price
            )
            intent = OrderIntent(
                ts=now_ist(),
                strategy_id="flatten",
                symbol=pos.symbol,
                side=Side.SELL if pos.qty > 0 else Side.BUY,
                qty=abs(pos.qty),
                product=pos.product,
                urgency=Urgency.AGGRESSIVE,
                reference_price=price,
                limit_band_bps=50.0,
                meta={"reason": f"flatten by {user.username}", "is_exit": True},
            )
            await handle.engine.bus.publish(Topics.INTENTS, intent)
            sent.append({"symbol": pos.symbol, "side": intent.side.value, "qty": intent.qty})
        return {"exits": sent}

    # ------------------------------------------------------------------ portfolio
    async def account(self, user: User, mode: EngineMode) -> dict[str, Any]:
        handle = self.get(user, mode)
        if handle is not None:
            broker: Broker = handle.broker
        elif mode is EngineMode.PAPER:
            broker = PaperBroker(
                config=PaperConfig(
                    starting_cash=self.settings.paper_starting_cash, account_id=paper_account(user)
                ),
                store=self.paper_store,
            )
        else:
            raise EngineError(409, "the live engine is not running")
        funds = await broker.funds()
        positions = await broker.positions()
        orders = sorted(await broker.orders(), key=lambda o: o.updated_at, reverse=True)
        fills = await broker.fills() if hasattr(broker, "fills") else []  # type: ignore[attr-defined]
        return {
            "mode": mode.value,
            "running": handle is not None,
            "funds": {
                "cash": round(funds.cash, 2),
                "equity": round(funds.equity, 2),
                "margin_used": round(funds.margin_used, 2),
                "realised_pnl": round(funds.realised_pnl, 2),
                "unrealised_pnl": round(funds.unrealised_pnl, 2),
            },
            "positions": [
                p.model_dump(mode="json") | {"net_pnl": round(p.net_pnl, 2)} for p in positions
            ],
            "orders": [o.model_dump(mode="json") for o in orders[:200]],
            "fills": [
                f.model_dump(mode="json") | {"fee_total": round(f.fees.total, 2)}
                for f in sorted(fills, key=lambda f: f.ts, reverse=True)[:200]
            ],
        }

    def reset_paper(self, user: User, confirm: str) -> None:
        account = paper_account(user)
        if self.get(user, EngineMode.PAPER) is not None:
            raise EngineError(409, "stop the paper engine before resetting its account")
        if confirm != account:
            raise EngineError(400, f"type {account} to confirm wiping this paper account")
        self.paper_store.reset(account)


async def _close(broker: Any, source: Any) -> None:
    for thing in {id(broker): broker, id(source): source}.values():
        if thing is not None:
            with contextlib.suppress(Exception):
                await thing.close()
