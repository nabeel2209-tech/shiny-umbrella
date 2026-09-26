"""Backtest jobs: a persisted queue with one background worker.

``submit`` records the request and returns at once; the worker runs each backtest
in its own thread (with its own event loop - a backtest is self-contained: an
in-memory bus and a simulated broker), so a long run never blocks the API. Jobs
left queued by a restart are picked up again; ones that were running are marked
failed, because half a backtest is not a result.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from trading.agents.risk import RiskLimits
from trading.backtest.runner import BacktestConfig, BacktestRunner, load_summary
from trading.backtest.sim_broker import FixedSlippage, VolumeSlippage
from trading.brokers.lots import LotSizes
from trading.brokers.symbols import SymbolMap, UnknownSymbol
from trading.core.clock import MarketCalendar
from trading.core.db import Base, make_engine, make_session_factory
from trading.core.types import now_ist
from trading.strategies.schema import StrategyConfig
from trading.training.ingest import Archive
from trading.training.registry import ModelRegistry

log = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    strategy_ids: list[str] = Field(min_length=1)
    start: date
    end: date
    name: str = Field(default="", max_length=120)
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    slippage_bps: float = Field(default=2.0, ge=0, le=500)
    impact: bool = False  # square-root volume impact instead of fixed slippage
    participation: float | None = Field(default=None, gt=0, le=1)
    liquidate: bool = False
    max_position_value: float | None = Field(default=None, gt=0)
    max_gross_exposure: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _dates(self) -> BacktestRequest:
        if self.end < self.start:
            raise ValueError("end is before start")
        if (self.end - self.start).days > 3 * 366:
            raise ValueError("at most three years per backtest")
        return self


class JobRow(Base):
    __tablename__ = "backtest_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued running done failed
    params: Mapped[str] = mapped_column(Text)
    strategies_yaml: Mapped[str] = mapped_column(Text)  # the exact strategies submitted
    run_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40))
    started_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)


def job_dict(row: JobRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "status": row.status,
        "params": json.loads(row.params),
        "run_id": row.run_id,
        "error": row.error,
        "created_at": row.created_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
    }


SymbolMapLoader = Callable[[], Any]  # async () -> SymbolMap | None


class BacktestJobs:
    def __init__(
        self,
        db_url: str,
        *,
        calendar: MarketCalendar,
        archive: Archive,
        registry: ModelRegistry,
        output_dir: Path,
        symbol_map_loader: SymbolMapLoader,
    ) -> None:
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine, tables=[JobRow.__table__])
        self._session = make_session_factory(self.engine)
        self.calendar = calendar
        self.archive = archive
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.symbol_map_loader = symbol_map_loader
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ api
    def submit(
        self, user_id: str, req: BacktestRequest, strategies: list[StrategyConfig]
    ) -> dict[str, Any]:
        row = JobRow(
            id=secrets.token_hex(8),
            user_id=user_id,
            status="queued",
            params=req.model_dump_json(),
            strategies_yaml=json.dumps([s.to_yaml() for s in strategies]),
            created_at=now_ist().isoformat(),
        )
        with self._session() as s, s.begin():
            s.add(row)
        self._queue.put_nowait(row.id)
        return job_dict(row)

    def get(self, user_id: str, job_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            row = s.get(JobRow, job_id)
            return job_dict(row) if row and row.user_id == user_id else None

    def list(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(
                select(JobRow)
                .where(JobRow.user_id == user_id)
                .order_by(JobRow.created_at.desc())
                .limit(limit)
            ).all()
            return [job_dict(r) for r in rows]

    def result_dir(self, run_id: str) -> Path:
        return self.output_dir / run_id

    def summary(self, run_id: str) -> dict[str, Any]:
        return load_summary(self.result_dir(run_id))

    # ------------------------------------------------------------------ worker
    def start(self) -> None:
        with self._session() as s, s.begin():
            for row in s.scalars(select(JobRow).where(JobRow.status.in_(["queued", "running"]))):
                if row.status == "running":
                    row.status, row.error = "failed", "interrupted by a restart"
                    row.finished_at = now_ist().isoformat()
                else:
                    self._queue.put_nowait(row.id)
        self._worker = asyncio.create_task(self._work(), name="backtest-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def _work(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self.run_job(job_id)
            except Exception:  # the worker must outlive any one job
                log.exception("backtest job %s crashed", job_id)

    async def run_job(self, job_id: str) -> None:
        with self._session() as s, s.begin():
            row = s.get(JobRow, job_id)
            if row is None or row.status != "queued":
                return
            row.status, row.started_at = "running", now_ist().isoformat()
            params, yaml_list = row.params, row.strategies_yaml
        try:
            symbol_map = await self.symbol_map_loader()
            run_id = await asyncio.to_thread(
                self._run_in_thread, job_id, params, yaml_list, symbol_map
            )
            status, error = "done", None
        except Exception as e:
            log.warning("backtest job %s failed: %s", job_id, e)
            run_id, status, error = None, "failed", f"{type(e).__name__}: {e}"
        with self._session() as s, s.begin():
            row = s.get(JobRow, job_id)
            assert row is not None
            row.status, row.run_id, row.error = status, run_id, error
            row.finished_at = now_ist().isoformat()

    def _run_in_thread(
        self, job_id: str, params: str, yaml_list: str, symbol_map: SymbolMap | None
    ) -> str:
        return asyncio.run(self._run(job_id, params, yaml_list, symbol_map))

    async def _run(
        self, job_id: str, params: str, yaml_list: str, symbol_map: SymbolMap | None
    ) -> str:
        req = BacktestRequest.model_validate_json(params)
        strategies = [StrategyConfig.from_yaml_str(y) for y in json.loads(yaml_list)]
        symbols = sorted({s for st in strategies for s in st.symbols})
        LotSizes.from_symbol_map(symbol_map, symbols)  # a derivative with no lot size stops here
        instruments = {}
        if symbol_map is not None:
            for symbol in symbols:
                with contextlib.suppress(UnknownSymbol):  # an unlisted equity trades single shares
                    instruments[symbol] = symbol_map.resolve(symbol)
        limits = RiskLimits()
        if req.max_position_value:
            limits.max_position_value = req.max_position_value
            limits.max_order_value = max(limits.max_order_value, req.max_position_value)
        if req.max_gross_exposure:
            limits.max_gross_exposure = req.max_gross_exposure
        cfg = BacktestConfig(
            strategies=strategies,
            start=req.start,
            end=req.end,
            initial_cash=req.initial_cash,
            limits=limits,
            slippage=VolumeSlippage() if req.impact else FixedSlippage(req.slippage_bps),
            max_participation=req.participation,
            instruments=instruments,
            liquidate_at_end=req.liquidate,
            name=req.name,
            run_id=f"bt-{now_ist():%Y%m%d-%H%M%S}-{job_id}",
        )
        runner = BacktestRunner(self.calendar, self.archive, models=self.registry)
        result = await runner.run(cfg)
        if not result.metrics.get("bars"):
            # an empty run looks like "the strategy did nothing"; say what is missing
            raise ValueError(
                f"no archived bars for {', '.join(symbols)} between {req.start} and {req.end}: "
                "ingest them first (scripts/ingest_history.py)"
            )
        result.save(self.output_dir)
        return result.run_id
