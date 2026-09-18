"""Pull daily and intraday bars from Dhan into the Parquet archive.

Examples:
    python scripts/ingest_history.py --symbols NSE:RELIANCE --intervals 1m,1d --days 10
    python scripts/ingest_history.py --universe nifty100 --intervals 1d --start 2021-01-01
    python scripts/ingest_history.py --symbols MCX:GOLDM-OCT26 --intervals 5m            # top-up

Needs DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env and a Data API subscription.
Nothing in the archive is ever deleted; re-runs merge and replace overlapping bars.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from trading.brokers.dhan import DhanBroker, DhanConfig
from trading.core.clock import MarketCalendar
from trading.core.config import get_settings
from trading.core.types import IST, Interval
from trading.core.universe import COMMODITY_UNDERLYINGS, INDEX_SYMBOLS, load_nifty100
from trading.training.ingest import Archive, ingest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--symbols", help="comma-separated canonical symbols")
    ap.add_argument(
        "--universe", choices=["nifty100", "indices", "gold", "all"], help="named symbol set"
    )
    ap.add_argument("--intervals", default="1m,1d", help="comma-separated: 1m,5m,15m,1h,1d")
    ap.add_argument(
        "--start", type=date.fromisoformat, help="YYYY-MM-DD (default: top-up from archive)"
    )
    ap.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, help="shortcut: start = today - days")
    ap.add_argument("--archive", type=Path, help="archive dir (default ARCHIVE_DIR)")
    ap.add_argument("--report-json", type=Path, help="write per-symbol results here")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    calendar = MarketCalendar.load(settings.holidays_file, mcx_close=settings.mcx_close)
    archive = Archive(
        args.archive or settings.archive_dir, corporate_actions=settings.corporate_actions_file
    )
    intervals = [Interval(i) for i in args.intervals.split(",")]

    symbols: list[str] = []
    if args.symbols:
        symbols += [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.universe in {"nifty100", "all"}:
        symbols += load_nifty100(settings.instruments_dir)
    if args.universe in {"indices", "all"}:
        symbols += INDEX_SYMBOLS
    if not symbols and args.universe != "gold":
        print("nothing to do: pass --symbols or --universe", file=sys.stderr)
        return 2

    start = args.start
    if args.days:
        start = datetime.now(IST).date() - timedelta(days=args.days)

    if args.dry_run:  # no credentials or network needed
        for s in symbols:
            for i in intervals:
                last = archive.last_ts(s, i)
                nxt = start or (last + timedelta(seconds=i.seconds) if last else "lookback")
                print(f"{s:32} {i.value:3} archive last bar: {last}  -> fetch from {nxt}")
        if args.universe in {"gold", "all"}:
            print("MCX front-month contracts are resolved from the instrument master at run time")
        return 0

    cfg = DhanConfig.from_settings(settings)
    broker = DhanBroker(cfg, instruments_dir=str(settings.instruments_dir))
    await broker.connect()
    try:
        if args.universe in {"gold", "all"}:
            today = datetime.now(IST).date()
            for u in COMMODITY_UNDERLYINGS:
                front = broker.symbols.front_month("MCX", u, today, min_days_to_expiry=3)
                symbols.append(front.symbol)
        results = await ingest(
            broker, archive, symbols, intervals, calendar, start=start, end=args.end
        )
    finally:
        await broker.close()

    problems = 0
    for r in results:
        print("-" * 78)
        rng = (
            f"{r.range[0]:%Y-%m-%d %H:%M} .. {r.range[1]:%Y-%m-%d %H:%M}"
            if r.range
            else "nothing new"
        )
        print(
            f"{r.symbol} {r.interval.value}: fetched {r.fetched} bars in {r.requests} "
            f"requests ({rng}), wrote {r.written}"
        )
        if r.error:
            problems += 1
            print(f"  ERROR: {r.error}")
        if r.report:
            print(r.report.summary())
            if not r.report.ok:
                problems += 1
    if args.report_json:
        args.report_json.write_text(
            json.dumps([_result_json(r) for r in results], indent=2, default=str)
        )
    print("-" * 78)
    print(f"{len(results)} symbol/intervals, {problems} with problems")
    return 1 if problems else 0


def _result_json(r) -> dict:  # type: ignore[no-untyped-def]
    d = {
        "symbol": r.symbol,
        "interval": r.interval.value,
        "fetched": r.fetched,
        "written": r.written,
        "requests": r.requests,
        "error": r.error,
    }
    if r.report:
        rep = r.report
        d["report"] = {
            "start": rep.start,
            "end": rep.end,
            "trading_days": rep.trading_days,
            "days_with_bars": rep.days_with_bars,
            "days_missing": rep.days_missing,
            "bars_expected": rep.bars_expected,
            "bars_present": rep.bars_present,
            "gaps": rep.gaps,
            "duplicates": rep.duplicates,
            "out_of_session": rep.out_of_session,
            "price_errors": rep.price_errors,
            "zero_volume_bars": rep.zero_volume_bars,
            "split_candidates": rep.split_candidates,
            "notes": rep.notes,
        }
    return d


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
