"""The nightly retraining job and the weekly drift report.

Retraining is a scheduled, validated, reversible batch job - never something that
happens inside the live tick loop (constraint 2). Each night, after MCX closes
(the last exchange to shut), :class:`NightlyJob`:

1. **tops up the archive** with the day's bars, if it has a market-data source;
2. **attaches outcomes** to the signals the paper/live engines logged, so the live
   model's realised IC and hit rate are known;
3. for every configured model: builds a dataset over ``lookback_days`` from the
   archive (which now includes the day just traded), holds out the last
   ``holdout_days`` sessions, **trains** a candidate on the rest, **registers** it
   with its out-of-fold and holdout metrics and the live model's realised record,
   and **promotes** it only if the gate passes on that holdout;
4. on ``drift_weekday`` writes the **drift report**: realised against expected,
   for the models (logged predictions vs holdout promises) and for trading
   (paper/live fills vs a backtest of the same week);
5. writes everything it did to ``report_dir/nightly-<date>.json``.

One model failing does not stop the others; its error goes into the report.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading.backtest.metrics import match_trades, trade_stats
from trading.brokers.base import MarketData
from trading.brokers.lots import LotSizes
from trading.core.clock import Clock, MarketCalendar, SystemClock
from trading.core.types import Fill, Interval, now_ist
from trading.features.features import DEFAULT_SPEC, FeatureSpec
from trading.training.dataset import build_dataset, load_frames
from trading.training.evaluate import EvalConfig, evaluate_oos
from trading.training.ingest import Archive, ingest
from trading.training.labels import LabelKind, LabelSpec
from trading.training.promote import PromotionGate, promote_if_better
from trading.training.registry import ModelRegistry
from trading.training.signal_log import SignalLog
from trading.training.splits import holdout_split
from trading.training.train import ModelKind, TrainConfig, fit_model, information_coefficient

log = logging.getLogger(__name__)


@dataclass
class ModelJob:
    name: str
    symbols: list[str]
    interval: Interval = Interval.M5
    label: LabelSpec = field(default_factory=LabelSpec)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    gate: PromotionGate = field(default_factory=PromotionGate)
    spec: FeatureSpec = field(default_factory=lambda: DEFAULT_SPEC)
    lookback_days: int = 120  # calendar days of history per retrain
    holdout_days: int = 10  # trading sessions held out for the gate

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelJob:
        label = dict(d.get("label", {}))
        if "kind" in label:
            label["kind"] = LabelKind(label["kind"])
        train = dict(d.get("train", {}))
        if "kind" in train:
            train["kind"] = ModelKind(train["kind"])
        for key in ("ridge_alphas", "lgbm_grid"):
            if key in train:
                train[key] = tuple(train[key])
        from trading.core.types import ProductType

        if "product" in label:
            label["product"] = ProductType(label["product"])
        return cls(
            name=d["name"],
            symbols=list(d["symbols"]),
            interval=Interval(d.get("interval", "5m")),
            label=LabelSpec(**label),
            train=TrainConfig(**train),
            evaluation=EvalConfig(**d.get("evaluation", {})),
            gate=PromotionGate(**d.get("gate", {})),
            lookback_days=int(d.get("lookback_days", 120)),
            holdout_days=int(d.get("holdout_days", 10)),
        )


@dataclass
class NightlyConfig:
    jobs: list[ModelJob]
    ingest: bool = True
    minutes_after_close: int = 15
    drift_weekday: int = 4  # Friday
    report_dir: Path = Path("data/reports")
    adjusted: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> NightlyConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            jobs=[ModelJob.from_dict(j) for j in raw["jobs"]],
            ingest=bool(raw.get("ingest", True)),
            minutes_after_close=int(raw.get("minutes_after_close", 15)),
            drift_weekday=int(raw.get("drift_weekday", 4)),
            report_dir=Path(raw.get("report_dir", "data/reports")),
            adjusted=bool(raw.get("adjusted", True)),
        )


BacktestFn = Callable[[date, date], Awaitable[Any]]
FillsFn = Callable[[date, date], Sequence[Fill]]


class NightlyJob:
    def __init__(
        self,
        archive: Archive,
        registry: ModelRegistry,
        calendar: MarketCalendar,
        cfg: NightlyConfig,
        *,
        source: MarketData | None = None,
        lots: LotSizes | None = None,
        signal_log: SignalLog | None = None,
        realised_fills: FillsFn | None = None,
        backtest: BacktestFn | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.archive = archive
        self.registry = registry
        self.calendar = calendar
        self.cfg = cfg
        self.source = source
        self.lots = lots or LotSizes()
        self.signal_log = signal_log
        self.realised_fills = realised_fills
        self.backtest = backtest
        self.clock = clock or SystemClock()
        for job in cfg.jobs:
            self.lots.require(job.symbols)  # never train on a guessed lot size

    # ------------------------------------------------------------------ timing
    def next_run(self, now: datetime) -> datetime:
        """The next MCX close plus the buffer (MCX is the last market to shut)."""
        day = now.date()
        for _ in range(15):
            bounds = self.calendar.session_bounds("MCX", day)
            if bounds is not None:
                at = bounds[1] + timedelta(minutes=self.cfg.minutes_after_close)
                if at > now:
                    return at
            day += timedelta(days=1)
        raise RuntimeError("no MCX session in the next two weeks")

    def session_of(self, run_at: datetime) -> date:
        """The trading day a run belongs to. MCX closes at 23:30/23:55, so the run
        usually starts after midnight - it is still the previous day's retrain."""
        return (run_at - timedelta(minutes=self.cfg.minutes_after_close)).date()

    async def run_forever(self) -> None:
        while True:
            now = self.clock.now()
            at = self.next_run(now)
            log.info("next nightly run at %s", at)
            await asyncio.sleep(max(0.0, (at - now).total_seconds()))
            try:
                await self.run(self.session_of(at))
            except Exception:  # keep the scheduler alive; tomorrow is another run
                log.exception("nightly run failed")

    # ------------------------------------------------------------------ the run
    async def run(self, as_of: date | None = None) -> dict[str, Any]:
        as_of = as_of or self.clock.now().date()
        report: dict[str, Any] = {"as_of": as_of.isoformat(), "started_at": now_ist().isoformat()}
        report["ingest"] = await self._ingest(as_of)
        report["outcomes"] = self._attach_outcomes()
        report["models"] = {}
        for job in self.cfg.jobs:
            try:
                report["models"][job.name] = self.run_job(job, as_of)
            except Exception as e:
                log.exception("nightly job %s failed", job.name)
                report["models"][job.name] = {"error": f"{type(e).__name__}: {e}"}
        if as_of.weekday() == self.cfg.drift_weekday:
            report["drift"] = await self.drift(as_of)
        report["finished_at"] = now_ist().isoformat()
        self.cfg.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.report_dir / f"nightly-{as_of.isoformat()}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        report["path"] = str(path)
        return report

    async def _ingest(self, as_of: date) -> dict[str, Any]:
        if not (self.cfg.ingest and self.source is not None):
            return {"skipped": "no market data source" if self.source is None else "disabled"}
        pairs = sorted({(s, j.interval) for j in self.cfg.jobs for s in j.symbols})
        out = {}
        for symbol, interval in pairs:
            (res,) = await ingest(
                self.source, self.archive, [symbol], [interval], self.calendar, end=as_of
            )
            out[f"{symbol} {interval.value}"] = {
                "fetched": res.fetched,
                "error": res.error,
                "ok": res.report.ok if res.report else None,
            }
        return out

    def _attach_outcomes(self) -> dict[str, Any]:
        if self.signal_log is None:
            return {"skipped": "no signal log"}
        horizons = {j.name: j.label.horizon for j in self.cfg.jobs}
        return {"attached": self.signal_log.attach_outcomes(self.archive, horizons=horizons)}

    def holdout_start(self, job: ModelJob, as_of: date) -> date:
        exchange = "MCX" if all(s.startswith("MCX:") for s in job.symbols) else "NSE"
        day = (
            as_of
            if self.calendar.is_trading_day(exchange, as_of)
            else (self.calendar.previous_trading_day(exchange, as_of))
        )
        for _ in range(job.holdout_days - 1):
            day = self.calendar.previous_trading_day(exchange, day)
        return day

    def run_job(self, job: ModelJob, as_of: date, *, promote: bool = True) -> dict[str, Any]:
        start = as_of - timedelta(days=job.lookback_days)
        frames = load_frames(
            self.archive,
            job.symbols,
            job.interval,
            start,
            as_of,
            calendar=self.calendar,
            adjusted=self.cfg.adjusted,
        )
        ds = build_dataset(frames, job.interval, spec=job.spec, label=job.label, lots=self.lots)
        if len(ds.labelled) == 0:
            raise ValueError(
                f"no usable {job.interval.value} bars for {', '.join(job.symbols)} between "
                f"{start} and {as_of} in the archive - ingest them first"
            )
        hold_day = self.holdout_start(job, as_of)
        n_hold = int(ds.timestamps.dt.date.ge(hold_day).groupby(ds.timestamps).first().sum())
        train_rows, _ = holdout_split(ds.timestamps, holdout=n_hold, purge=job.label.horizon)
        model = fit_model(ds.take(train_rows), job.train)
        oos = evaluate_oos(model, job.evaluation)
        live_before = self.registry.live_version(job.name)
        version = self.registry.register(
            job.name,
            model,
            metrics={
                "oos": oos.metrics,
                "oos_per_fold": oos.per_fold,
                "holdout_start": hold_day.isoformat(),
                "live_realised": self.realised_model_stats(job.name, hold_day, as_of),
            },
            actor="nightly",
        )
        if not promote:
            return {
                "version": version,
                "promoted": False,
                "failures": ["not submitted to the gate"],
                "live_before": live_before,
                "live_after": live_before,
                "rows": len(ds),
                "train_rows": len(train_rows),
                "holdout_start": hold_day.isoformat(),
                "oos": {k: oos.metrics.get(k) for k in ("ic", "sharpe", "trades", "hit_rate")},
            }
        window_start = datetime.combine(hold_day, datetime.min.time(), tzinfo=ds.timestamps.dt.tz)
        decision = promote_if_better(
            self.registry,
            job.name,
            version,
            frames,
            window_start,
            eval_cfg=job.evaluation,
            gate=job.gate,
            lots=self.lots,
            actor="nightly",
        )
        return {
            "version": version,
            "promoted": decision.promote,
            "failures": decision.failures,
            "live_before": live_before,
            "live_after": self.registry.live_version(job.name),
            "rows": len(ds),
            "train_rows": len(train_rows),
            "holdout_start": hold_day.isoformat(),
            "oos": {k: oos.metrics.get(k) for k in ("ic", "sharpe", "trades", "hit_rate")},
            "holdout": decision.to_dict()["candidate"],
            "live_holdout": decision.to_dict()["live"],
        }

    # ------------------------------------------------------------------ drift
    def realised_model_stats(self, model: str, start: date, end: date) -> dict[str, Any]:
        """How the model's logged live/paper predictions actually turned out."""
        if self.signal_log is None:
            return {"count": 0}
        df = self.signal_log.frame(start, end, model=model)
        df = df[df["outcome"].notna()] if not df.empty else df
        if df.empty:
            return {"count": 0}
        score, outcome = df["score"].to_numpy(dtype=float), df["outcome"].to_numpy(dtype=float)
        moved = outcome != 0
        return {
            "count": len(df),
            "ic": round(information_coefficient(score, outcome), 4),
            "hit_rate": round(float((np.sign(score[moved]) == np.sign(outcome[moved])).mean()), 4)
            if moved.any()
            else None,
            "mean_predicted_edge_bps": round(float(df["expected_edge_bps"].abs().mean()), 2),
            "mean_realised_bps": round(float((np.sign(score) * outcome).mean() * 1e4), 2),
            "versions": sorted(df["model_version"].dropna().unique().tolist()),
        }

    async def drift(self, as_of: date) -> dict[str, Any]:
        """Realised vs expected over the last week, for models and for trading."""
        week_start = as_of - timedelta(days=6)
        out: dict[str, Any] = {"week": [week_start.isoformat(), as_of.isoformat()], "models": {}}
        flags: list[str] = []
        for job in self.cfg.jobs:
            realised = self.realised_model_stats(job.name, week_start, as_of)
            live = self.registry.live_version(job.name)
            expected = {}
            if live is not None:
                meta = self.registry.metadata(job.name, live)
                decisions = self.registry.decisions(job.name, live)
                promised = decisions[-1]["candidate"] if decisions else {}
                expected = {
                    "version": live,
                    "ic": promised.get("ic", meta["metrics"].get("oos", {}).get("ic")),
                    "hit_rate": promised.get("hit_rate"),
                }
            out["models"][job.name] = {"realised": realised, "expected": expected}
            if realised.get("count", 0) >= 20 and expected.get("ic") is not None:
                if realised["ic"] < expected["ic"] - 0.1:
                    flags.append(
                        f"{job.name}: realised IC {realised['ic']} vs {expected['ic']} promised"
                    )
                if realised["ic"] < 0:
                    flags.append(f"{job.name}: live predictions are anti-correlated with outcomes")
        out["trading"] = await self._trading_drift(week_start, as_of, flags)
        out["flags"] = flags
        return out

    async def _trading_drift(self, start: date, end: date, flags: list[str]) -> dict[str, Any]:
        if self.realised_fills is None:
            return {"skipped": "no realised fills source"}
        fills = [f for f in self.realised_fills(start, end) if start <= f.ts.date() <= end]
        trades = match_trades(fills)
        realised = {
            **trade_stats(trades),
            "net_pnl": round(sum(t.net_pnl for t in trades if not t.is_open), 2),
            "fees": round(sum(f.fees.total for f in fills), 2),
            "fills": len(fills),
        }
        out: dict[str, Any] = {"realised": realised}
        if self.backtest is not None:
            result = await self.backtest(start, end)
            m = result.metrics
            expected = {k: m.get(k) for k in ("trades", "hit_rate", "net_pnl", "fills")}
            expected["fees"] = m.get("fees", {}).get("total")
            out["backtest"] = expected
            flags.extend(_trading_flags(realised, expected))
        return out


def _trading_flags(realised: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Where live/paper trading has drifted from what the backtest said it would do."""
    flags = []
    got, want = realised.get("trades") or 0, expected.get("trades") or 0
    if want and abs(got - want) / want > 0.25:
        flags.append(f"trade count {got} vs {want} backtested")
    hit, hit_bt = realised.get("hit_rate"), expected.get("hit_rate")
    if hit is not None and hit_bt is not None and abs(hit - hit_bt) > 0.10:
        flags.append(f"hit rate {hit:.0%} vs {hit_bt:.0%} backtested")
    pnl, pnl_bt = realised.get("net_pnl") or 0.0, expected.get("net_pnl") or 0.0
    if got and (pnl > 0) != (pnl_bt > 0):
        flags.append(f"net PnL {pnl:,.2f} vs {pnl_bt:,.2f} backtested")
    return flags


def load_jobs_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Nightly report models section as a table (for the dashboard)."""
    rows = [{"model": k, **v} for k, v in report.get("models", {}).items()]
    return pd.DataFrame(rows)
