"""Every signal the engine emits, with its realised outcome filled in later.

The engine's :class:`SignalLogger` agent writes each ``Signal`` (score, calibrated
probability, model version) to SQLite as it happens. The nightly job then looks up
what actually followed in the archive - the gross return over the model's label
horizon - so the live model's *realised* IC and hit rate can be compared with what
its holdout promised (the weekly drift report) and recorded against the next
candidate.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import Float, Integer, String, select, update
from sqlalchemy.orm import Mapped, mapped_column

from trading.agents.base import Agent
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock
from trading.core.db import Base, make_engine, make_session_factory
from trading.core.types import Interval, Signal
from trading.training.ingest import Archive

log = logging.getLogger(__name__)


class SignalRow(Base):
    __tablename__ = "signal_log"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[str] = mapped_column(String(40), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    interval: Mapped[str] = mapped_column(String(8))
    model: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    score: Mapped[float] = mapped_column(Float)
    prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_edge_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(String(64), default="")
    outcome: Mapped[float | None] = mapped_column(Float, nullable=True)  # gross return
    outcome_horizon: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SignalLog:
    def __init__(self, db_url: str = "sqlite:///data/state.db") -> None:
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine, tables=[SignalRow.__table__])
        self._session = make_session_factory(self.engine)

    def record(self, signal: Signal) -> None:
        with self._session() as s, s.begin():
            s.merge(
                SignalRow(
                    id=signal.id,
                    ts=signal.ts.isoformat(),
                    day=signal.ts.date().isoformat(),
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    interval=str(signal.meta.get("interval", "")),
                    model=signal.meta.get("model"),
                    model_version=signal.model_version,
                    score=signal.score,
                    prob=signal.prob,
                    expected_edge_bps=signal.expected_edge_bps,
                    reason=str(signal.meta.get("reason", ""))[:64],
                )
            )

    def frame(
        self,
        start: date | None = None,
        end: date | None = None,
        *,
        model: str | None = None,
    ) -> pd.DataFrame:
        q = select(SignalRow)
        if start is not None:
            q = q.where(SignalRow.day >= start.isoformat())
        if end is not None:
            q = q.where(SignalRow.day <= end.isoformat())
        if model is not None:
            q = q.where(SignalRow.model == model)
        with self._session() as s:
            rows = s.scalars(q.order_by(SignalRow.ts)).all()
        cols = [c.name for c in SignalRow.__table__.columns]
        df = pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows], columns=cols)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def attach_outcomes(
        self,
        archive: Archive,
        *,
        horizons: Mapping[str, int],
        default_horizon: int | None = None,
        adjusted: bool = True,
    ) -> int:
        """Fill in the gross forward return of every signal that now has one.

        ``horizons`` maps a model name to its label horizon; signals from
        rule-only strategies use ``default_horizon`` or stay empty.
        """
        with self._session() as s:
            pending = s.scalars(select(SignalRow).where(SignalRow.outcome.is_(None))).all()
            todo = [
                (r.id, r.symbol, r.interval, r.ts, horizons.get(r.model or "", default_horizon))
                for r in pending
            ]
        updates: list[tuple[str, float, int]] = []
        groups: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
        for sid, symbol, interval, ts, h in todo:
            if h and interval:
                groups.setdefault((symbol, interval), []).append((sid, ts, h))
        for (symbol, interval), items in groups.items():
            first = min(datetime.fromisoformat(t) for _, t, _ in items).date()
            last = max(datetime.fromisoformat(t) for _, t, _ in items).date()
            bars = archive.read(
                symbol, Interval(interval), first, last + timedelta(days=10), adjusted=adjusted
            )
            if bars.empty:
                continue
            position = {ts: i for i, ts in enumerate(bars["ts"])}
            close = bars["close"].to_numpy(dtype=float)
            for sid, ts, h in items:
                i = position.get(pd.Timestamp(ts))
                if i is not None and i + h < len(close):
                    updates.append((sid, close[i + h] / close[i] - 1.0, h))
        if updates:
            with self._session() as s, s.begin():
                for sid, outcome, h in updates:
                    s.execute(
                        update(SignalRow)
                        .where(SignalRow.id == sid)
                        .values(outcome=outcome, outcome_horizon=h)
                    )
        return len(updates)


class SignalLogger(Agent):
    """Writes every signal on the bus to the log."""

    name = "signal_log"

    def __init__(self, bus: MessageBus, log_: SignalLog, *, clock: Clock | None = None) -> None:
        super().__init__(bus, clock=clock)
        self.store = log_
        self.recorded = 0

    async def on_start(self) -> None:
        await self.subscribe(Topics.SIGNALS, self._on_signal)

    async def _on_signal(self, _topic: str, signal: Signal) -> None:  # type: ignore[override]
        self.store.record(signal)
        self.recorded += 1
