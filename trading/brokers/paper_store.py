"""SQLite/SQLAlchemy persistence for the paper broker.

Rows keep a few queryable columns plus the full pydantic payload as JSON so the
schema stays stable while the models evolve. Timestamps are stored as ISO
strings to preserve the IST offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import JSON, Float, String, delete, select
from sqlalchemy.orm import Mapped, mapped_column

from trading.core.db import Base, make_engine, make_session_factory
from trading.core.types import Fill, Order, Position, now_ist


class PaperAccountRow(Base):
    __tablename__ = "paper_account"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    starting_cash: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[str] = mapped_column(String(40))


class PaperOrderRow(Base):
    __tablename__ = "paper_orders"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)


class PaperFillRow(Base):
    __tablename__ = "paper_fills"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)


class PaperPositionRow(Base):
    __tablename__ = "paper_positions"
    key: Mapped[str] = mapped_column(String(160), primary_key=True)  # account|symbol|product
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    product: Mapped[str] = mapped_column(String(10))
    payload: Mapped[dict] = mapped_column(JSON)


@dataclass
class PaperState:
    starting_cash: float
    cash: float
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)


class PaperStore:
    def __init__(self, db_url: str = "sqlite:///data/state.db") -> None:
        self.engine = make_engine(db_url)
        Base.metadata.create_all(self.engine)
        self._session = make_session_factory(self.engine)

    # ------------------------------------------------------------------ load
    def load(self, account_id: str) -> PaperState | None:
        with self._session() as s:
            acct = s.get(PaperAccountRow, account_id)
            if acct is None:
                return None
            orders = s.scalars(
                select(PaperOrderRow).where(PaperOrderRow.account_id == account_id)
            ).all()
            fills = s.scalars(
                select(PaperFillRow).where(PaperFillRow.account_id == account_id)
            ).all()
            positions = s.scalars(
                select(PaperPositionRow).where(PaperPositionRow.account_id == account_id)
            ).all()
            return PaperState(
                starting_cash=acct.starting_cash,
                cash=acct.cash,
                orders=[Order.model_validate(r.payload) for r in orders],
                fills=sorted((Fill.model_validate(r.payload) for r in fills), key=lambda f: f.ts),
                positions=[Position.model_validate(r.payload) for r in positions],
            )

    # ------------------------------------------------------------------ save
    def save_account(self, account_id: str, starting_cash: float, cash: float) -> None:
        with self._session() as s, s.begin():
            s.merge(
                PaperAccountRow(
                    id=account_id,
                    starting_cash=starting_cash,
                    cash=cash,
                    updated_at=now_ist().isoformat(),
                )
            )

    def upsert_order(self, account_id: str, order: Order) -> None:
        with self._session() as s, s.begin():
            s.merge(
                PaperOrderRow(
                    id=order.id,
                    account_id=account_id,
                    symbol=order.symbol,
                    status=order.status.value,
                    updated_at=order.updated_at.isoformat(),
                    payload=order.model_dump(mode="json"),
                )
            )

    def add_fill(self, account_id: str, fill: Fill) -> None:
        with self._session() as s, s.begin():
            s.merge(
                PaperFillRow(
                    id=fill.id,
                    account_id=account_id,
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    ts=fill.ts.isoformat(),
                    payload=fill.model_dump(mode="json"),
                )
            )

    def upsert_position(self, account_id: str, pos: Position) -> None:
        with self._session() as s, s.begin():
            s.merge(
                PaperPositionRow(
                    key=f"{account_id}|{pos.symbol}|{pos.product.value}",
                    account_id=account_id,
                    symbol=pos.symbol,
                    product=pos.product.value,
                    payload=pos.model_dump(mode="json"),
                )
            )

    def reset(self, account_id: str) -> None:
        """Destructive: wipe everything for one paper account. Only call explicitly."""
        with self._session() as s, s.begin():
            for table in (PaperOrderRow, PaperFillRow, PaperPositionRow):
                s.execute(delete(table).where(table.account_id == account_id))
            s.execute(delete(PaperAccountRow).where(PaperAccountRow.id == account_id))
