"""Backtest runner: the production engine replayed over the archive.

Nothing here re-implements trading logic. The runner builds the same
``TradingEngine`` the paper and live runners use - data, signal, risk, execution
and monitor agents on an in-memory bus - swaps the broker for :class:`SimBroker`,
and feeds it archived bars in the order they would have completed. What differs
from live is only what must: the clock is simulated and fills are simulated.

Per run it writes, under ``<output>/<run_id>/``::

    summary.json    config, metrics, per-strategy breakdown, timings
    trades.csv      round trips (FIFO), open positions marked to market
    equity.csv      equity, cash, positions value, exposure and drawdown per bar
    fills.csv       every fill with its fee breakdown
    orders.csv      every order and its final state
    strategies/     the strategy YAML exactly as it was run

Warmup: the feature buffers are primed with the bars before ``start`` (the same
thing the live data agent does from broker history), so a strategy is fully warm
on the first bar of the window and cannot trade on a half-built history.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

import pandas as pd

from trading.agents.data import resample_bars
from trading.agents.engine import EngineConfig, TradingEngine
from trading.agents.execution import ExecutionConfig
from trading.agents.risk import RiskLimits
from trading.backtest.costs import DEFAULT_FEES, FeeSchedule
from trading.backtest.metrics import (
    Trade,
    equity_stats,
    fee_breakdown,
    match_trades,
    trade_stats,
    turnover,
)
from trading.backtest.sim_broker import FixedSlippage, SimBroker, SimConfig, SlippageModel
from trading.brokers.base import Instrument
from trading.brokers.lots import LotSizes
from trading.brokers.symbols import parse_symbol
from trading.core.bus import InMemoryBus
from trading.core.clock import MarketCalendar, SimClock
from trading.core.types import IST, Bar, Fill, Interval, Order, now_ist
from trading.features.features import DEFAULT_SPEC, FeatureSpec
from trading.strategies.schema import StrategyConfig
from trading.training.ingest import Archive

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/backtests")


@dataclass
class BacktestConfig:
    strategies: list[StrategyConfig]
    start: date
    end: date
    initial_cash: float = 1_000_000.0
    limits: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    slippage: SlippageModel = field(default_factory=FixedSlippage)
    max_participation: float | None = None
    fee_schedule: FeeSchedule = field(default_factory=lambda: DEFAULT_FEES)
    spec: FeatureSpec = field(default_factory=lambda: DEFAULT_SPEC)
    instruments: dict[str, Instrument] = field(default_factory=dict)
    liquidate_at_end: bool = False
    mis_square_off_minutes: int | None = 10
    enforce_market_hours: bool | None = None  # None: on, unless a strategy trades daily bars
    adjusted: bool = True  # apply confirmed corporate actions to equity bars
    precompute_features: bool = True  # batch features once; False recomputes per bar
    risk_free_rate: float = 0.0  # annual, for Sharpe / Sortino
    name: str = ""
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")
        if not self.strategies:
            raise ValueError("a backtest needs at least one strategy")
        ids = [s.id for s in self.strategies]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate strategy ids: {ids}")

    @property
    def market_hours_enforced(self) -> bool:
        if self.enforce_market_hours is not None:
            return self.enforce_market_hours
        # a daily bar completes at the close, when the market is shut; its orders
        # are market-on-open orders for the next session, which the simulator
        # fills at that open, so the live-only guard does not apply
        return not any(s.interval is Interval.D1 for s in self.strategies)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_cash": self.initial_cash,
            "strategies": [s.id for s in self.strategies],
            "slippage": self.slippage.describe(),
            "max_participation": self.max_participation,
            "liquidate_at_end": self.liquidate_at_end,
            "mis_square_off_minutes": self.mis_square_off_minutes,
            "market_hours_enforced": self.market_hours_enforced,
            "adjusted": self.adjusted,
            "risk_free_rate": self.risk_free_rate,
            "feature_spec": dataclasses.asdict(self.spec),
            "limits": {
                k: v for k, v in dataclasses.asdict(self.limits).items() if k != "fee_schedule"
            },
            "execution": dataclasses.asdict(self.execution),
        }

    def fingerprint(self) -> str:
        """Stable hash of everything that determines the result."""
        payload = json.dumps(
            {"config": self.describe(), "yaml": [s.to_yaml() for s in self.strategies]},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class BacktestResult:
    run_id: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    per_strategy: dict[str, dict[str, Any]]
    trades: list[Trade]
    equity: pd.DataFrame
    fills: list[Fill]
    orders: list[Order]
    strategies: list[StrategyConfig]
    elapsed_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: now_ist().isoformat())
    path: Path | None = None

    # ------------------------------------------------------------------ io
    def save(self, root: Path | str = DEFAULT_OUTPUT) -> Path:
        out = Path(root) / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "strategies").mkdir(exist_ok=True)
        for strategy in self.strategies:
            (out / "strategies" / f"{strategy.id}.yaml").write_text(strategy.to_yaml())
        _write_csv(out / "trades.csv", [t.to_row() for t in self.trades], TRADE_COLUMNS)
        equity = self.equity.copy()
        if not equity.empty:
            equity["ts"] = equity["ts"].map(lambda t: t.isoformat())
        equity.to_csv(out / "equity.csv", index=False)
        _write_csv(out / "fills.csv", [_fill_row(f) for f in self.fills], FILL_COLUMNS)
        _write_csv(out / "orders.csv", [_order_row(o) for o in self.orders], ORDER_COLUMNS)
        summary = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "config": self.config,
            "metrics": self.metrics,
            "per_strategy": self.per_strategy,
            "files": ["trades.csv", "equity.csv", "fills.csv", "orders.csv", "strategies/"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        self.path = out
        return out

    def summary_text(self) -> str:
        m = self.metrics
        pct = lambda v: "n/a" if v is None else f"{v * 100:.2f}%"  # noqa: E731
        lines = [
            f"backtest {self.run_id}  {self.config['start']} .. {self.config['end']}",
            f"  strategies     {', '.join(self.config['strategies'])}",
            f"  net pnl        {m['net_pnl']:,.2f}  ({pct(m['total_return'])})",
            f"  final equity   {m['final_equity']:,.2f}",
            f"  sharpe         {m['sharpe']:.2f} ({m['sharpe_basis']}, "
            f"{m['sharpe_observations']} returns)",
            f"  sortino        {'n/a' if m['sortino'] is None else format(m['sortino'], '.2f')}",
            f"  max drawdown   {m['max_drawdown']:,.2f}  ({pct(m['max_drawdown_pct'])})",
            f"  trades         {m['trades']} closed, {m['open_trades']} open, "
            f"hit rate {pct(m['hit_rate'])}",
            f"  turnover       {m['turnover']:.2f}x capital ({m['traded_value']:,.0f} traded)",
            f"  fees           {m['fees']['total']:,.2f}",
            f"  achieved-mid   {m['achieved_vs_mid']['mean_bps']:.2f} bps mean over "
            f"{m['achieved_vs_mid']['count']} fills",
            f"  exposure       {pct(m['exposure'])} of bars",
        ]
        if m["rejections"]:
            lines.append(f"  rejections     {m['rejections']}")
        return "\n".join(lines)


TRADE_COLUMNS = [
    "strategy_id", "symbol", "product", "direction", "qty", "entry_ts", "entry_price",
    "exit_ts", "exit_price", "multiplier", "gross_pnl", "fees", "net_pnl", "return_pct",
    "holding_seconds", "is_open", "entry_order_id", "exit_order_id",
]  # fmt: skip
FILL_COLUMNS = [
    "ts", "order_id", "symbol", "side", "qty", "price", "product", "multiplier", "value",
    "brokerage", "stt", "exchange", "sebi", "stamp", "gst", "other", "fees",
]  # fmt: skip
ORDER_COLUMNS = [
    "id", "symbol", "side", "qty", "filled_qty", "order_type", "product", "price",
    "trigger_price", "status", "avg_fill_price", "created_at", "updated_at", "intent_id",
    "parent_id", "note",
]  # fmt: skip


def _fill_row(f: Fill) -> dict[str, object]:
    return {
        "ts": f.ts.isoformat(),
        "order_id": f.order_id,
        "symbol": f.symbol,
        "side": f.side.value,
        "qty": f.qty,
        "price": f.price,
        "product": f.product.value,
        "multiplier": f.multiplier,
        "value": round(f.value, 4),
        **{
            k: getattr(f.fees, k)
            for k in ("brokerage", "stt", "exchange", "sebi", "stamp", "gst", "other")
        },
        "fees": f.fees.total,
    }


def _order_row(o: Order) -> dict[str, object]:
    note = o.meta.get("liquidation") or ("protective" if o.meta.get("protective") else "")
    return {
        "id": o.id,
        "symbol": o.symbol,
        "side": o.side.value,
        "qty": o.qty,
        "filled_qty": o.filled_qty,
        "order_type": o.order_type.value,
        "product": o.product.value,
        "price": o.price,
        "trigger_price": o.trigger_price,
        "status": o.status.value,
        "avg_fill_price": o.avg_fill_price,
        "created_at": o.created_at.isoformat(),
        "updated_at": o.updated_at.isoformat(),
        "intent_id": o.intent_id or "",
        "parent_id": o.parent_id or "",
        "note": note,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):  # numpy scalars
        return value.item()  # type: ignore[attr-defined]
    return str(value)


def load_summary(run_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def list_runs(root: Path | str = DEFAULT_OUTPUT) -> list[dict[str, Any]]:
    """Every saved run's summary, newest first (the dashboard's backtest list)."""
    base = Path(root)
    if not base.exists():
        return []
    runs = [load_summary(p) for p in base.iterdir() if (p / "summary.json").exists()]
    return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)


# --------------------------------------------------------------------------- runner


class BacktestRunner:
    def __init__(self, calendar: MarketCalendar, archive: Archive | None = None) -> None:
        self.calendar = calendar
        self.archive = archive

    # ------------------------------------------------------------------ data
    def warmup_start(self, symbol: str, interval: Interval, start: date, spec: FeatureSpec) -> date:
        exchange = parse_symbol(symbol).exchange
        probe = (
            start
            if self.calendar.is_trading_day(exchange, start)
            else (self.calendar.next_trading_day(exchange, start))
        )
        per_session = max(1, len(self.calendar.session_bars(exchange, probe, interval)))
        sessions = math.ceil(spec.buffer_bars(per_session) / per_session) + 1
        day = start
        for _ in range(sessions):
            day = self.calendar.previous_trading_day(exchange, day)
        return day

    def load_bars(self, cfg: BacktestConfig) -> tuple[list[Bar], list[Bar]]:
        """(warmup bars before ``start``, bars inside the window) from the archive.

        A missing interval is built from 1-minute bars with the live bar builder.
        """
        if self.archive is None:
            raise ValueError("no archive configured; pass bars to run() instead")
        warm: list[Bar] = []
        window: list[Bar] = []
        seen: set[tuple[str, Interval]] = set()
        for strategy in cfg.strategies:
            for symbol in strategy.symbols:
                key = (symbol, strategy.interval)
                if key in seen:
                    continue
                seen.add(key)
                first = self.warmup_start(symbol, strategy.interval, cfg.start, cfg.spec)
                bars = self._read(symbol, strategy.interval, first, cfg.end, cfg.adjusted)
                if not bars:
                    log.warning("no %s bars for %s in the archive", strategy.interval.value, symbol)
                for bar in bars:
                    (warm if bar.ts.date() < cfg.start else window).append(bar)
        return warm, window

    def _read(
        self, symbol: str, interval: Interval, start: date, end: date, adjusted: bool
    ) -> list[Bar]:
        assert self.archive is not None
        if self.archive.partitions(symbol, interval) or interval is Interval.M1:
            return self.archive.read_bars(symbol, interval, start, end, adjusted=adjusted)
        minute = self.archive.read_bars(symbol, Interval.M1, start, end, adjusted=adjusted)
        return resample_bars(minute, interval, self.calendar) if minute else []

    def order_by_completion(self, bars: Sequence[Bar]) -> list[Bar]:
        """The order bars become *known*: a daily bar lands after that day's intraday bars."""

        def key(bar: Bar) -> tuple[datetime, int, str]:
            end = self.calendar.bar_end(parse_symbol(bar.symbol).exchange, bar.ts, bar.interval)
            return end, bar.interval.seconds, bar.symbol

        return sorted(bars, key=key)

    # ------------------------------------------------------------------ run
    async def run(
        self,
        cfg: BacktestConfig,
        *,
        bars: Sequence[Bar] | None = None,
        warmup_bars: Sequence[Bar] | None = None,
    ) -> BacktestResult:
        started = time.perf_counter()
        if bars is None:
            warm, window = self.load_bars(cfg)
        else:
            window = [b for b in bars if cfg.start <= b.ts.date() <= cfg.end]
            warm = list(warmup_bars or []) + [b for b in bars if b.ts.date() < cfg.start]
        window = self.order_by_completion(window)

        lots = LotSizes.from_instruments(cfg.instruments)
        lots.require({s for strategy in cfg.strategies for s in strategy.symbols})
        clock = SimClock(datetime.combine(cfg.start, dtime(0, 0), tzinfo=IST))
        broker = SimBroker(
            SimConfig(
                starting_cash=cfg.initial_cash,
                slippage=cfg.slippage,
                fee_schedule=cfg.fee_schedule,
                max_participation=cfg.max_participation,
                lot_size_for=lots,
            ),
            clock=clock,
        )
        limits = dataclasses.replace(
            cfg.limits, require_market_open=cfg.market_hours_enforced, fee_schedule=cfg.fee_schedule
        )
        # no wall-clock throttling inside simulated time (see ExecutionConfig)
        execution = dataclasses.replace(cfg.execution, orders_per_second=None)
        engine = TradingEngine(
            InMemoryBus(),
            broker,
            self.calendar,
            EngineConfig(
                strategies=cfg.strategies,
                limits=limits,
                execution=execution,
                spec=cfg.spec,
                starting_equity=cfg.initial_cash,
                publish_ticks=False,
                mis_square_off_minutes=cfg.mis_square_off_minutes,
            ),
            instruments=cfg.instruments,
            clock=clock,
            live=False,
        )
        await engine.start(reconcile=False)
        engine.data.prime(sorted(warm, key=lambda b: (b.ts, b.symbol)))
        if cfg.precompute_features:
            engine.data.precompute([*warm, *window])

        curve: dict[datetime, dict[str, float]] = {}
        try:
            for bar in window:
                end = await engine.step(bar)
                curve[end] = self._snapshot(broker)
            if cfg.liquidate_at_end and window:
                await broker.liquidate(reason="end of backtest")
                await engine.drain_updates()
                curve[clock.now()] = self._snapshot(broker)
        finally:
            await engine.stop()

        return await self._result(cfg, engine, broker, curve, window, time.perf_counter() - started)

    @staticmethod
    def _snapshot(broker: SimBroker) -> dict[str, float]:
        gross = 0.0
        for pos in broker._positions.values():
            if pos.qty:
                price = pos.last_price if pos.last_price is not None else pos.avg_price
                gross += abs(pos.qty) * price * pos.multiplier
        return {
            "equity": broker.equity(),
            "cash": broker.cash,
            "positions_value": broker.positions_value(),
            "gross_exposure": gross,
        }

    async def _result(
        self,
        cfg: BacktestConfig,
        engine: TradingEngine,
        broker: SimBroker,
        curve: dict[datetime, dict[str, float]],
        window: Sequence[Bar],
        elapsed: float,
    ) -> BacktestResult:
        equity = pd.DataFrame([{"ts": ts, **row} for ts, row in sorted(curve.items())])
        if not equity.empty:
            series = pd.Series(equity["equity"].to_numpy(), index=pd.DatetimeIndex(equity["ts"]))
            peaks = series.cummax().clip(lower=cfg.initial_cash)
            equity["drawdown_pct"] = (series / peaks - 1.0).to_numpy()
        else:
            series = pd.Series(dtype=float)
        fills = await broker.fills()
        orders = await broker.orders()
        strategy_of = engine.execution.strategy_of_orders()
        as_of = equity["ts"].iloc[-1] if not equity.empty else None
        trades = match_trades(fills, strategy_of=strategy_of, marks=dict(broker._last), as_of=as_of)

        monitor = engine.monitor
        metrics: dict[str, Any] = {
            **equity_stats(series, cfg.initial_cash, risk_free_rate=cfg.risk_free_rate),
            **trade_stats(trades),
            **turnover(fills, cfg.initial_cash),
            "fees": fee_breakdown(fills),
            "bars": len(window),
            "intents": sum(s.intents_emitted for s in engine.signals),
            "approved": engine.risk.approved_count,
            "rejected": engine.risk.rejected_count,
            "rejections": dict(engine.risk.rejections),
            "orders": len(orders),
            "fills": len(fills),
            "achieved_vs_mid": monitor.slippage_summary(),
            "exposure": (
                round(float((equity["gross_exposure"] > 0).mean()), 4) if not equity.empty else 0.0
            ),
        }
        per_strategy = {}
        for strategy in cfg.strategies:
            own = [t for t in trades if t.strategy_id == strategy.id]
            per_strategy[strategy.id] = {
                **trade_stats(own),
                "net_pnl": round(sum(t.net_pnl for t in own), 2),
                "fees": round(sum(t.fees for t in own), 2),
                "achieved_vs_mid": monitor.slippage_summary(strategy.id),
            }
        run_id = cfg.run_id or f"bt-{now_ist():%Y%m%d-%H%M%S}-{cfg.fingerprint()[:8]}"
        return BacktestResult(
            run_id=run_id,
            config={**cfg.describe(), "fingerprint": cfg.fingerprint()},
            metrics=metrics,
            per_strategy=per_strategy,
            trades=trades,
            equity=equity,
            fills=fills,
            orders=orders,
            strategies=list(cfg.strategies),
            elapsed_seconds=elapsed,
        )


async def run_backtest(
    cfg: BacktestConfig,
    calendar: MarketCalendar,
    *,
    archive: Archive | None = None,
    bars: Sequence[Bar] | None = None,
    output: Path | str | None = DEFAULT_OUTPUT,
) -> BacktestResult:
    """Run and (unless ``output`` is None) save a backtest."""
    result = await BacktestRunner(calendar, archive).run(cfg, bars=bars)
    if output is not None:
        result.save(output)
    return result
