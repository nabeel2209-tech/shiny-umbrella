"""Train, register, promote and roll back models; run the nightly job.

    python scripts/train.py fit --name nifty_5m --symbols NSE:RELIANCE,NSE:INFY \\
        --interval 5m --as-of 2026-09-25 --horizon 6 --kind ridge --promote
    python scripts/train.py list [--name nifty_5m]
    python scripts/train.py promote nifty_5m v0004 --reason "manual after review"
    python scripts/train.py rollback nifty_5m --reason "bad fills since Tuesday"
    python scripts/train.py nightly --jobs trading/training/jobs.yaml --once

``fit`` without ``--promote`` only registers the candidate; with it, the candidate
goes through the promotion gate on the held-out sessions. ``promote`` moves the
live pointer by hand and bypasses the gate - it asks for confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path

from trading.brokers.dhan_instruments import ensure_symbol_map
from trading.brokers.lots import LotSizes, MissingLotSize
from trading.core.clock import MarketCalendar
from trading.core.config import get_settings
from trading.core.types import IST, Interval, ProductType, now_ist
from trading.training.ingest import Archive
from trading.training.labels import LabelKind, LabelSpec
from trading.training.registry import ModelRegistry, RegistryError
from trading.training.schedule import ModelJob, NightlyConfig, NightlyJob
from trading.training.train import ModelKind, TrainConfig

MODELS_DIR = Path("data/models")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", type=Path, default=MODELS_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="train and register one model")
    fit.add_argument("--name", required=True)
    fit.add_argument("--symbols", required=True, help="comma-separated canonical symbols")
    fit.add_argument("--interval", default="5m")
    fit.add_argument("--as-of", type=date.fromisoformat, help="last day of data (default today)")
    fit.add_argument("--lookback-days", type=int, default=120)
    fit.add_argument("--holdout-days", type=int, default=10)
    fit.add_argument("--kind", choices=[k.value for k in ModelKind], default="ridge")
    fit.add_argument("--label", choices=[k.value for k in LabelKind], default="forward_return")
    fit.add_argument("--horizon", type=int, default=6)
    fit.add_argument("--product", choices=[p.value for p in ProductType], default="MIS")
    fit.add_argument("--embargo", type=int, default=0)
    fit.add_argument("--promote", action="store_true", help="submit to the promotion gate")

    ls = sub.add_parser("list", help="registered models and versions")
    ls.add_argument("--name")

    pr = sub.add_parser("promote", help="set the live version by hand (bypasses the gate)")
    pr.add_argument("name")
    pr.add_argument("version")
    pr.add_argument("--reason", required=True)
    pr.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    rb = sub.add_parser("rollback", help="move the live pointer back one version")
    rb.add_argument("name")
    rb.add_argument("--reason", required=True)

    ni = sub.add_parser("nightly", help="run the nightly job")
    ni.add_argument("--jobs", type=Path, required=True)
    ni.add_argument("--once", action="store_true", help="run now and exit")
    ni.add_argument("--as-of", type=date.fromisoformat)
    return ap.parse_args(argv)


def cmd_list(registry: ModelRegistry, name: str | None) -> int:
    names = [name] if name else registry.names()
    if not names:
        print("no models registered")
        return 0
    for n in names:
        live = registry.live_version(n)
        print(f"{n}  (live: {live or '-'})")
        for v in registry.versions(n):
            meta = registry.metadata(n, v)
            decisions = registry.decisions(n, v)
            holdout = decisions[-1]["candidate"] if decisions else {}
            model = meta["model"]
            gate = (
                "passed"
                if decisions and decisions[-1]["promote"]
                else ("refused" if decisions else "not gated")
            )
            print(
                f"  {'*' if v == live else ' '} {v}  {model['kind']:8} "
                f"trained to {model['train_window']['end'][:16]}  "
                f"oos IC {model['cv']['oos_ic']:+.3f}  "
                f"holdout Sharpe {holdout.get('sharpe', '-')}  "
                f"gate {gate}"
            )
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = get_settings()
    registry = ModelRegistry(args.models)

    if args.cmd == "list":
        return cmd_list(registry, args.name)
    if args.cmd == "promote":
        if not args.yes:
            answer = input(f"make {args.name} {args.version} live WITHOUT the gate? type YES: ")
            if answer.strip() != "YES":
                print("aborted")
                return 1
        try:
            registry.set_live(args.name, args.version, reason=f"manual: {args.reason}", actor="cli")
        except RegistryError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"{args.name} live -> {args.version}")
        return 0
    if args.cmd == "rollback":
        try:
            version = registry.rollback(args.name, reason=args.reason, actor="cli")
        except RegistryError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"{args.name} live -> {version}")
        return 0

    calendar = MarketCalendar.load(settings.holidays_file, mcx_close=settings.mcx_close)
    archive = Archive(settings.archive_dir, corporate_actions=settings.corporate_actions_file)
    if args.cmd == "fit":
        jobs = [
            ModelJob(
                name=args.name,
                symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
                interval=Interval(args.interval),
                label=LabelSpec(
                    kind=LabelKind(args.label),
                    horizon=args.horizon,
                    product=ProductType(args.product),
                ),
                train=TrainConfig(kind=ModelKind(args.kind), embargo=args.embargo),
                lookback_days=args.lookback_days,
                holdout_days=args.holdout_days,
            )
        ]
        cfg = NightlyConfig(jobs=jobs, ingest=False)
    else:
        cfg = NightlyConfig.from_yaml(args.jobs)

    symbols = sorted({s for j in cfg.jobs for s in j.symbols})
    try:
        lots = LotSizes.from_symbol_map(await ensure_symbol_map(settings.instruments_dir), symbols)
    except MissingLotSize as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    source = None
    if args.cmd == "nightly" and cfg.ingest and settings.dhan_client_id:
        from trading.brokers.dhan import DhanBroker, DhanConfig

        source = DhanBroker(
            DhanConfig.from_settings(settings), instruments_dir=str(settings.instruments_dir)
        )
        await source.connect()
    job = NightlyJob(archive, registry, calendar, cfg, source=source, lots=lots)
    try:
        as_of = args.as_of or now_ist().astimezone(IST).date()
        if args.cmd == "fit":
            try:
                result = job.run_job(cfg.jobs[0], as_of, promote=args.promote)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.once:
            report = await job.run(as_of)
            print(json.dumps(report.get("models", {}), indent=2, default=str))
            print(f"report: {report['path']}")
            return 0
        await job.run_forever()
        return 0
    finally:
        if source is not None:
            await source.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
