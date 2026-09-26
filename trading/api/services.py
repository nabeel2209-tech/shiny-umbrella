"""Everything the API needs, built once per process (and replaceable in tests)."""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading.api.auth import AuthStore
from trading.api.engines import BrokerFactory, EngineError, EngineManager, SymbolMapLoader
from trading.api.events import EventHub, KillSwitch
from trading.api.jobs import BacktestJobs
from trading.api.strategy_store import StrategyStore
from trading.backtest.runner import DEFAULT_OUTPUT
from trading.brokers.dhan_instruments import ensure_symbol_map
from trading.brokers.paper_store import PaperStore
from trading.core.clock import MarketCalendar
from trading.core.config import Settings
from trading.training.ingest import Archive
from trading.training.registry import ModelRegistry
from trading.training.signal_log import SignalLog

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    calendar: MarketCalendar
    archive: Archive
    registry: ModelRegistry
    auth: AuthStore
    strategies: StrategyStore
    jobs: BacktestJobs
    engines: EngineManager
    hub: EventHub
    kill: KillSwitch
    paper_store: PaperStore
    signal_log: SignalLog
    backtests_dir: Path


def _dhan_factory(settings: Settings) -> BrokerFactory:
    async def make() -> Any:
        if not (settings.dhan_client_id and settings.dhan_access_token.get_secret_value()):
            raise EngineError(
                400,
                "this needs Dhan's market data: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env",
            )
        from trading.brokers.dhan import DhanBroker, DhanConfig

        broker = DhanBroker(
            DhanConfig.from_settings(settings), instruments_dir=str(settings.instruments_dir)
        )
        await broker.connect()
        return broker

    return make


def build_services(
    settings: Settings,
    *,
    feed_factory: BrokerFactory | None = None,
    live_broker_factory: BrokerFactory | None = None,
    symbol_map_loader: SymbolMapLoader | None = None,
    backtests_dir: Path | None = None,
) -> Services:
    secret = settings.session_secret.get_secret_value()
    if not secret:
        secret = secrets.token_hex(32)
        log.warning("SESSION_SECRET is not set: sessions will not survive a restart")
    if symbol_map_loader is None:

        async def symbol_map_loader() -> Any:
            return await ensure_symbol_map(settings.instruments_dir)

    calendar = MarketCalendar.load(settings.holidays_file, mcx_close=settings.mcx_close)
    archive = Archive(settings.archive_dir, corporate_actions=settings.corporate_actions_file)
    registry = ModelRegistry(settings.models_dir)
    strategies = StrategyStore(settings.strategies_dir)
    hub = EventHub()
    kill = KillSwitch(settings.state_dir / "kill.json")
    paper_store = PaperStore(settings.db_url)
    signal_log = SignalLog(settings.db_url)
    out_dir = backtests_dir or DEFAULT_OUTPUT
    jobs = BacktestJobs(
        settings.db_url,
        calendar=calendar,
        archive=archive,
        registry=registry,
        output_dir=out_dir,
        symbol_map_loader=symbol_map_loader,
    )
    engines = EngineManager(
        settings,
        calendar,
        strategies,
        registry,
        signal_log,
        paper_store,
        hub,
        kill,
        feed_factory=feed_factory or _dhan_factory(settings),
        live_broker_factory=live_broker_factory or _dhan_factory(settings),
        symbol_map_loader=symbol_map_loader,
    )
    return Services(
        settings=settings,
        calendar=calendar,
        archive=archive,
        registry=registry,
        auth=AuthStore(settings.db_url, secret, ttl_hours=settings.session_ttl_hours),
        strategies=strategies,
        jobs=jobs,
        engines=engines,
        hub=hub,
        kill=kill,
        paper_store=paper_store,
        signal_log=signal_log,
        backtests_dir=out_dir,
    )
