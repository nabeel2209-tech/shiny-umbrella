"""Run the full engine against the paper broker on Dhan's live feed.

Fake cash, fake fills, **real quotes** - the honest dress rehearsal before any live
capital. Positions, orders and cash persist to SQLite, so stopping and restarting
resumes the same paper account.

    python scripts/run_paper.py --strategies trading/strategies/examples
    python scripts/run_paper.py --replay 2026-09-14:2026-09-18

Without Dhan credentials, ``--replay`` drives the same engine from the Parquet
archive instead, which needs no network.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from datetime import date, datetime, time
from pathlib import Path

from trading.agents.data import merge_bars
from trading.agents.engine import EngineConfig, TradingEngine, confirm_live_trading
from trading.agents.execution import ExecutionConfig
from trading.agents.risk import RiskLimits
from trading.brokers.base import Instrument
from trading.brokers.dhan_instruments import ensure_symbol_map
from trading.brokers.lots import LotSizes, MissingLotSize
from trading.brokers.paper import PaperBroker, PaperConfig
from trading.brokers.paper_store import PaperStore
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import make_bus
from trading.core.clock import MarketCalendar, SimClock, SystemClock
from trading.core.config import get_settings
from trading.core.types import IST
from trading.features.features import DEFAULT_SPEC
from trading.strategies.schema import load_strategies
from trading.training.ingest import Archive
from trading.training.registry import ModelRegistry
from trading.training.signal_log import SignalLog

log = logging.getLogger("run_paper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--strategies",
        type=Path,
        default=Path("trading/strategies/examples"),
        help="directory of strategy YAML files",
    )
    ap.add_argument("--account", default="paper", help="paper account id in SQLite")
    ap.add_argument("--cash", type=float, help="starting cash (default PAPER_STARTING_CASH)")
    ap.add_argument(
        "--replay",
        metavar="START:END",
        help="replay archived bars instead of the live feed, e.g. 2026-09-14:2026-09-18",
    )
    ap.add_argument("--reset", action="store_true", help="wipe this paper account first")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = get_settings()
    calendar = MarketCalendar.load(settings.holidays_file, mcx_close=settings.mcx_close)
    strategies = load_strategies(args.strategies)
    if not strategies:
        print(f"no strategies found in {args.strategies}", file=sys.stderr)
        return 2
    log.info("loaded %d strategies: %s", len(strategies), ", ".join(s.id for s in strategies))

    # Constraint 5: this runner is paper-only, but the gate is still evaluated and
    # logged so the behaviour is identical to the live runner.
    confirm_live_trading(settings, "paper")

    store = PaperStore(settings.db_url)
    if args.reset:
        confirm = input(f"wipe paper account {args.account!r}? type YES: ")
        if confirm.strip() != "YES":
            print("aborted")
            return 1
        store.reset(args.account)

    cash = args.cash if args.cash is not None else settings.paper_starting_cash
    replay_range: tuple[date, date] | None = None
    if args.replay:
        start_s, _, end_s = args.replay.partition(":")
        replay_range = (date.fromisoformat(start_s), date.fromisoformat(end_s or start_s))
        # a replay clock starts at the beginning of the window, not at "now"
        clock = SimClock(datetime.combine(replay_range[0], time(0, 0), tzinfo=IST))
    else:
        clock = SystemClock()
    symbols = sorted({s for st in strategies for s in st.symbols})
    instruments: dict[str, Instrument] = {}
    data_source = None

    if args.replay:
        # no broker connection: size from the instrument master (downloaded if
        # missing); a derivative that cannot be sized stops here
        symbol_map = await ensure_symbol_map(settings.instruments_dir)
        try:
            LotSizes.from_symbol_map(symbol_map, symbols)
        except MissingLotSize as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        for sym in symbols:
            if symbol_map is not None and sym in symbol_map:
                instruments[sym] = symbol_map.resolve(sym)
    else:
        from trading.brokers.dhan import DhanBroker, DhanConfig

        problems = settings.problems()
        if problems:
            for p in problems:
                print(f"config problem: {p}", file=sys.stderr)
            return 2
        dhan = DhanBroker(
            DhanConfig.from_settings(settings), instruments_dir=str(settings.instruments_dir)
        )
        await dhan.connect()
        data_source = dhan
        for sym in symbols:
            instruments[sym] = dhan.symbols.resolve(sym)

    broker = PaperBroker(
        data_source,
        config=PaperConfig(
            starting_cash=cash,
            slippage_bps=settings.paper_slippage_bps,
            account_id=args.account,
            multiplier_for=contract_multiplier,
        ),
        store=store,
        clock=clock,
    )
    bus = make_bus(settings.redis_url if not args.replay else None)
    engine = TradingEngine(
        bus,
        broker,
        calendar,
        EngineConfig(
            strategies=strategies,
            limits=RiskLimits(),
            execution=ExecutionConfig(),
            starting_equity=cash,
        ),
        instruments=instruments,
        models=ModelRegistry(settings.models_dir, expected_spec=DEFAULT_SPEC),
        signal_log=SignalLog(settings.db_url),
        clock=clock,
        live=False,
    )

    await engine.start(reconcile=not args.replay)
    stop = asyncio.Event()
    with contextlib.suppress(NotImplementedError):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
    try:
        if replay_range is not None:
            bars = load_archive_bars(settings.archive_dir, strategies, *replay_range)
            log.info("replaying %d bars", len(bars))
            await engine.replay(bars)
        else:
            runner = asyncio.create_task(engine.run_live())
            await stop.wait()
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner
    finally:
        await engine.stop()
        await broker.close()

    print_summary(engine, await broker.funds(), await broker.positions())
    return 0


def load_archive_bars(archive_dir: Path, strategies, start: date, end: date):  # type: ignore[no-untyped-def]
    archive = Archive(archive_dir)
    streams = []
    for strategy in strategies:
        for symbol in strategy.symbols:
            bars = archive.read_bars(symbol, strategy.interval, start, end)
            if not bars:
                log.warning("no %s bars for %s in the archive", strategy.interval.value, symbol)
            streams.append(bars)
    return merge_bars(streams)


def print_summary(engine: TradingEngine, funds, positions) -> None:  # type: ignore[no-untyped-def]
    status = engine.status()
    print("\n" + "=" * 70)
    print("paper session summary")
    print("=" * 70)
    for key in ("bars", "intents", "approved", "rejected", "orders", "fills"):
        print(f"  {key:12} {status[key]}")
    if status["rejections"]:
        print(f"  {'by rule':12} {status['rejections']}")
    print(f"  {'slippage':12} {status['slippage']}")
    print(f"  {'cash':12} {funds.cash:,.2f}")
    print(f"  {'equity':12} {funds.equity:,.2f}")
    print(f"  {'day pnl':12} {engine.portfolio.day_pnl:,.2f}")
    for pos in positions:
        if pos.qty:
            print(
                f"  position     {pos.symbol} {pos.qty:+d} @ {pos.avg_price:.2f} "
                f"pnl {pos.net_pnl:,.2f}"
            )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
