"""Backtest metrics: round-trip trades from fills, and equity-curve statistics.

Kept separate from the runner so the same numbers can be computed for a paper or
live account from its fills and equity history (the weekly drift report in
Phase 5 compares exactly these against the backtest).

Conventions
-----------
- Trades are matched **FIFO** per (symbol, product). A fill that reverses a
  position closes the old lots and opens a new one with the remainder.
- Fees are allocated to trades pro rata by quantity, entry and exit separately.
- Returns are simple. Sharpe and Sortino are annualised from **daily** returns
  (252 sessions) once there are at least ``MIN_DAILY_OBSERVATIONS`` of them;
  below that a daily Sharpe is noise (two days can produce a Sharpe of 400), so
  per-bar returns scaled by bars per session are used instead. ``sharpe_basis``
  and ``sharpe_observations`` say which, and how many returns went in.
- A ratio that is undefined - Sortino with no losing period - is ``None``, not 0.
- Drawdown includes the starting capital as the first peak, so losing money on
  day one is a drawdown.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from trading.core.types import Fill, ProductType

TRADING_DAYS = 252
MIN_DAILY_OBSERVATIONS = 10


@dataclass
class Trade:
    """One round trip (or an open position marked to market at the end)."""

    symbol: str
    product: ProductType
    direction: str  # LONG | SHORT
    qty: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime | None
    exit_price: float
    multiplier: float
    gross_pnl: float
    fees: float
    strategy_id: str = ""
    is_open: bool = False
    entry_order_id: str = ""
    exit_order_id: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def notional(self) -> float:
        return self.entry_price * self.qty * self.multiplier

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.notional if self.notional else 0.0

    @property
    def holding_seconds(self) -> float | None:
        if self.exit_ts is None:
            return None
        return (self.exit_ts - self.entry_ts).total_seconds()

    def to_row(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "product": self.product.value,
            "direction": self.direction,
            "qty": self.qty,
            "entry_ts": self.entry_ts.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_ts": self.exit_ts.isoformat() if self.exit_ts else "",
            "exit_price": round(self.exit_price, 4),
            "multiplier": self.multiplier,
            "gross_pnl": round(self.gross_pnl, 4),
            "fees": round(self.fees, 4),
            "net_pnl": round(self.net_pnl, 4),
            "return_pct": round(self.return_pct, 6),
            "holding_seconds": self.holding_seconds,
            "is_open": self.is_open,
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
        }


@dataclass
class _Lot:
    qty: int
    direction: int  # +1 long, -1 short
    price: float
    ts: datetime
    fee_per_unit: float
    order_id: str
    multiplier: float
    strategy_id: str


def match_trades(
    fills: Iterable[Fill],
    *,
    strategy_of: Mapping[str, str] | None = None,
    marks: Mapping[str, float] | None = None,
    as_of: datetime | None = None,
) -> list[Trade]:
    """FIFO round trips from fills.

    ``strategy_of`` maps an order id to its strategy. Lots still open at the end
    are returned as open trades valued at ``marks[symbol]`` (their entry price if
    no mark is given), carrying their entry fees only.
    """
    strategy_of = strategy_of or {}
    books: dict[tuple[str, ProductType], deque[_Lot]] = {}
    trades: list[Trade] = []
    for fill in sorted(fills, key=lambda f: f.ts):
        sign = fill.side.sign
        fee_unit = fill.fees.total / fill.qty
        book = books.setdefault((fill.symbol, fill.product), deque())
        left = fill.qty
        while left > 0 and book and book[0].direction != sign:
            lot = book[0]
            matched = min(lot.qty, left)
            trades.append(
                Trade(
                    symbol=fill.symbol,
                    product=fill.product,
                    direction="LONG" if lot.direction > 0 else "SHORT",
                    qty=matched,
                    entry_ts=lot.ts,
                    entry_price=lot.price,
                    exit_ts=fill.ts,
                    exit_price=fill.price,
                    multiplier=lot.multiplier,
                    gross_pnl=(fill.price - lot.price) * matched * lot.direction * lot.multiplier,
                    fees=(lot.fee_per_unit + fee_unit) * matched,
                    strategy_id=lot.strategy_id,
                    entry_order_id=lot.order_id,
                    exit_order_id=fill.order_id,
                )
            )
            lot.qty -= matched
            left -= matched
            if lot.qty == 0:
                book.popleft()
        if left > 0:
            book.append(
                _Lot(
                    qty=left,
                    direction=sign,
                    price=fill.price,
                    ts=fill.ts,
                    fee_per_unit=fee_unit,
                    order_id=fill.order_id,
                    multiplier=fill.multiplier,
                    strategy_id=strategy_of.get(fill.order_id, ""),
                )
            )
    marks = marks or {}
    for (symbol, product), book in books.items():
        for lot in book:
            mark = marks.get(symbol, lot.price)
            trades.append(
                Trade(
                    symbol=symbol,
                    product=product,
                    direction="LONG" if lot.direction > 0 else "SHORT",
                    qty=lot.qty,
                    entry_ts=lot.ts,
                    entry_price=lot.price,
                    exit_ts=as_of,
                    exit_price=mark,
                    multiplier=lot.multiplier,
                    gross_pnl=(mark - lot.price) * lot.qty * lot.direction * lot.multiplier,
                    fees=lot.fee_per_unit * lot.qty,
                    strategy_id=lot.strategy_id,
                    is_open=True,
                    entry_order_id=lot.order_id,
                )
            )
    return trades


# --------------------------------------------------------------------------- trade statistics


def trade_stats(trades: Sequence[Trade]) -> dict[str, object]:
    """Statistics over *closed* trades; open ones are counted but not scored."""
    closed = [t for t in trades if not t.is_open]
    out: dict[str, object] = {
        "trades": len(closed),
        "open_trades": len(trades) - len(closed),
    }
    if not closed:
        out.update(
            hit_rate=None,
            avg_win=None,
            avg_loss=None,
            profit_factor=None,
            expectancy=None,
            best_trade=None,
            worst_trade=None,
            avg_holding_seconds=None,
        )
        return out
    pnl = np.array([t.net_pnl for t in closed])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = float(-losses.sum())
    holding = [t.holding_seconds for t in closed if t.holding_seconds is not None]
    out.update(
        hit_rate=round(len(wins) / len(closed), 4),
        avg_win=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 2) if len(losses) else 0.0,
        profit_factor=round(float(wins.sum()) / gross_loss, 4) if gross_loss > 0 else None,
        expectancy=round(float(pnl.mean()), 2),
        best_trade=round(float(pnl.max()), 2),
        worst_trade=round(float(pnl.min()), 2),
        avg_holding_seconds=round(float(np.mean(holding)), 1) if holding else None,
    )
    return out


# --------------------------------------------------------------------------- equity statistics


@dataclass
class DrawdownStats:
    max_drawdown: float = 0.0  # rupees, <= 0
    max_drawdown_pct: float = 0.0  # fraction, <= 0
    peak_ts: datetime | None = None
    trough_ts: datetime | None = None
    recovered_ts: datetime | None = None
    longest_underwater_seconds: float = 0.0
    points: list[float] = field(default_factory=list)


def drawdown(equity: pd.Series, initial: float) -> DrawdownStats:
    """Peak-to-trough on an equity series indexed by timestamp."""
    if equity.empty:
        return DrawdownStats()
    values = np.concatenate([[initial], equity.to_numpy(dtype=float)])
    stamps: list[datetime | None] = [None, *equity.index.to_pydatetime()]
    peaks = np.maximum.accumulate(values)
    dd_abs = values - peaks
    dd_pct = np.where(peaks > 0, values / peaks - 1.0, 0.0)
    trough = int(np.argmin(dd_pct))
    stats = DrawdownStats(
        max_drawdown=float(dd_abs.min()),
        max_drawdown_pct=float(dd_pct.min()),
        points=[float(x) for x in dd_pct[1:]],
    )
    if stats.max_drawdown_pct < 0:
        peak_index = int(np.argmax(values[: trough + 1]))
        stats.peak_ts = stamps[peak_index] or (stamps[1] if len(stamps) > 1 else None)
        stats.trough_ts = stamps[trough]
        after = np.nonzero(values[trough:] >= peaks[trough])[0]
        if len(after):
            stats.recovered_ts = stamps[trough + int(after[0])]
    # longest time spent below a previous peak
    longest, start = 0.0, None
    for value, peak, ts in zip(values, peaks, stamps, strict=True):
        if ts is None:
            continue
        if value < peak:
            start = start or ts
            longest = max(longest, (ts - start).total_seconds())
        else:
            start = None
    stats.longest_underwater_seconds = longest
    return stats


def _annualised_ratio(returns: pd.Series, periods_per_year: float, rf_per_period: float) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_per_period
    sd = float(excess.std(ddof=1))
    if not math.isfinite(sd) or sd <= 1e-15:
        return 0.0
    return float(excess.mean()) / sd * math.sqrt(periods_per_year)


def _sortino(returns: pd.Series, periods_per_year: float, rf_per_period: float) -> float | None:
    """None when there is no downside at all - the ratio is undefined, not zero."""
    if len(returns) < 2:
        return None
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    dd = math.sqrt(float((downside**2).sum()) / len(excess))
    if dd <= 1e-15:
        return None
    return float(excess.mean()) / dd * math.sqrt(periods_per_year)


def equity_stats(
    equity: pd.Series, initial: float, *, risk_free_rate: float = 0.0
) -> dict[str, object]:
    """Returns, risk-adjusted returns and drawdown from an equity series indexed
    by timestamp. ``risk_free_rate`` is annual."""
    if equity.empty:
        return {
            "final_equity": initial,
            "net_pnl": 0.0,
            "total_return": 0.0,
            "annualised_return": None,
            "annualised_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": None,
            "sharpe_basis": "none",
            "sharpe_observations": 0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "days": 0,
        }
    final = float(equity.iloc[-1])
    total = final / initial - 1.0
    daily = equity.groupby(equity.index.date).last()
    days = len(daily)
    daily_returns = daily / daily.shift(1).fillna(initial) - 1.0
    rf_day = risk_free_rate / TRADING_DAYS
    if days >= MIN_DAILY_OBSERVATIONS:
        basis, returns, per_year = "daily", daily_returns, float(TRADING_DAYS)
        rf_period = rf_day
    else:
        bar_returns = equity / equity.shift(1).fillna(initial) - 1.0
        bars_per_day = len(equity) / max(days, 1)
        basis, returns, per_year = "bar", bar_returns, TRADING_DAYS * bars_per_day
        rf_period = rf_day / bars_per_day
    vol = float(returns.std(ddof=1)) * math.sqrt(per_year) if len(returns) > 1 else 0.0
    dd = drawdown(equity, initial)
    sortino = _sortino(returns, per_year, rf_period)
    return {
        "final_equity": round(final, 2),
        "net_pnl": round(final - initial, 2),
        "total_return": round(total, 6),
        "annualised_return": (
            round((1.0 + total) ** (TRADING_DAYS / days) - 1.0, 6) if days >= 20 else None
        ),
        "annualised_volatility": round(vol, 6) if math.isfinite(vol) else 0.0,
        "sharpe": round(_annualised_ratio(returns, per_year, rf_period), 4),
        "sortino": round(sortino, 4) if sortino is not None else None,
        "sharpe_basis": basis,
        "sharpe_observations": len(returns),
        "max_drawdown": round(dd.max_drawdown, 2),
        "max_drawdown_pct": round(dd.max_drawdown_pct, 6),
        "drawdown_peak": dd.peak_ts.isoformat() if dd.peak_ts else None,
        "drawdown_trough": dd.trough_ts.isoformat() if dd.trough_ts else None,
        "drawdown_recovered": dd.recovered_ts.isoformat() if dd.recovered_ts else None,
        "longest_underwater_seconds": dd.longest_underwater_seconds,
        "days": days,
    }


def fee_breakdown(fills: Iterable[Fill]) -> dict[str, float]:
    parts = ("brokerage", "stt", "exchange", "sebi", "stamp", "gst", "other")
    totals = dict.fromkeys(parts, 0.0)
    for fill in fills:
        for part in parts:
            totals[part] += getattr(fill.fees, part)
    totals = {k: round(v, 2) for k, v in totals.items()}
    totals["total"] = round(sum(v for k, v in totals.items()), 2)
    return totals


def turnover(fills: Iterable[Fill], initial: float) -> dict[str, float]:
    traded = sum(abs(f.value) for f in fills)
    return {
        "traded_value": round(traded, 2),
        "turnover": round(traded / initial, 4) if initial else 0.0,
    }
