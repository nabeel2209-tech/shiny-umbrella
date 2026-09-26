"""Backtest strategies over the Parquet archive with the production engine.

Examples:
    python scripts/run_backtest.py --start 2026-01-01 --end 2026-06-30
    python scripts/run_backtest.py --strategies my_strategies/ --start 2026-06-01 \\
        --end 2026-06-30 --impact --participation 0.1 --liquidate
    python scripts/run_backtest.py --list

Results land in data/backtests/<run_id>/ (summary.json plus trades, equity, fills
and orders as CSV). No credentials are needed. Lot sizes come from Dhan's public
instrument master, downloaded on startup when today's copy is missing; equities
default to one share, and a derivative that cannot be sized is an error.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from trading.agents.risk import RiskLimits
from trading.backtest.runner import DEFAULT_OUTPUT, BacktestConfig, list_runs, run_backtest
from trading.backtest.sim_broker import FixedSlippage, VolumeSlippage
from trading.brokers.dhan_instruments import ensure_symbol_map
from trading.brokers.lots import LotSizes, MissingLotSize
from trading.brokers.symbols import UnknownSymbol
from trading.core.clock import MarketCalendar
from trading.core.config import get_settings
from trading.strategies.schema import load_strategies
from trading.training.ingest import Archive

log = logging.getLogger("run_backtest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--strategies", type=Path, default=Path("trading/strategies/examples"))
    ap.add_argument("--only", help="comma-separated strategy ids to include")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0, help="fixed slippage")
    ap.add_argument("--impact", action="store_true", help="square-root volume impact instead")
    ap.add_argument("--participation", type=float, help="cap fills at this share of bar volume")
    ap.add_argument("--liquidate", action="store_true", help="close everything at the end")
    ap.add_argument("--max-position", type=float, help="per-instrument cap (default risk limit)")
    ap.add_argument("--max-gross", type=float, help="gross exposure cap")
    ap.add_argument("--raw", action="store_true", help="skip corporate-action adjustment")
    ap.add_argument("--name", default="")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--list", action="store_true", help="list saved runs and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def print_runs(root: Path) -> None:
    runs = list_runs(root)
    if not runs:
        print(f"no runs in {root}")
        return
    for r in runs:
        m, c = r["metrics"], r["config"]
        print(
            f"{r['run_id']:40} {c['start']}..{c['end']}  {','.join(c['strategies']):30} "
            f"pnl {m['net_pnl']:>12,.0f}  sharpe {m['sharpe']:>6.2f}  "
            f"dd {m['max_drawdown_pct'] * 100:>6.2f}%  trades {m['trades']}"
        )


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.list:
        print_runs(args.output)
        return 0
    if args.start is None or args.end is None:
        print("--start and --end are required", file=sys.stderr)
        return 2

    settings = get_settings()
    calendar = MarketCalendar.load(settings.holidays_file, mcx_close=settings.mcx_close)
    strategies = load_strategies(args.strategies)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        strategies = [s for s in strategies if s.id in wanted]
    if not strategies:
        print("no strategies selected", file=sys.stderr)
        return 2

    # lot sizes: the instrument master is downloaded if today's copy is missing;
    # equities default to 1, a derivative that cannot be sized stops the run
    symbols = sorted({s for st in strategies for s in st.symbols})
    symbol_map = await ensure_symbol_map(settings.instruments_dir)
    try:
        LotSizes.from_symbol_map(symbol_map, symbols)
    except MissingLotSize as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    instruments = {}
    if symbol_map is not None:
        for symbol in symbols:
            try:
                instruments[symbol] = symbol_map.resolve(symbol)
            except UnknownSymbol:
                log.warning("%s is not in the instrument master; trading single shares", symbol)

    limits = RiskLimits()
    if args.max_position is not None:
        limits.max_position_value = args.max_position
        limits.max_order_value = max(limits.max_order_value, args.max_position)
    if args.max_gross is not None:
        limits.max_gross_exposure = args.max_gross

    cfg = BacktestConfig(
        strategies=strategies,
        start=args.start,
        end=args.end,
        initial_cash=args.cash,
        limits=limits,
        slippage=VolumeSlippage() if args.impact else FixedSlippage(args.slippage_bps),
        max_participation=args.participation,
        instruments=instruments,
        liquidate_at_end=args.liquidate,
        adjusted=not args.raw,
        name=args.name,
    )
    archive = Archive(settings.archive_dir, corporate_actions=settings.corporate_actions_file)
    result = await run_backtest(cfg, calendar, archive=archive, output=args.output)
    print(result.summary_text())
    for sid, stats in result.per_strategy.items():
        print(f"  {sid:24} trades {stats['trades']:>4}  net {stats['net_pnl']:>12,.2f}")
    print(f"\nsaved to {result.path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
