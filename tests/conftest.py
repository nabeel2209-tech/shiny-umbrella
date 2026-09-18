"""Shared fixtures."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime

import pytest

from trading.core.clock import DEFAULT_HOLIDAYS_FILE, MarketCalendar, SimClock
from trading.core.types import IST, Bar, Interval

TRADING_DAY = date(2026, 9, 18)  # a Friday, not a holiday


@pytest.fixture(scope="session")
def calendar() -> MarketCalendar:
    return MarketCalendar.load(DEFAULT_HOLIDAYS_FILE)


@pytest.fixture
def trading_day() -> date:
    return TRADING_DAY


def make_synthetic_day(
    calendar: MarketCalendar,
    symbol: str = "NSE:RELIANCE",
    day: date = TRADING_DAY,
    *,
    base: float = 2500.0,
    amplitude: float = 20.0,
    period: float = 30.0,
    exchange: str = "NSE",
) -> list[Bar]:
    """Deterministic 1-minute bars: close follows a sine wave, open = previous close,
    high/low = extremes +/- 1. The path is smooth so a bar never gaps."""
    bars: list[Bar] = []
    prev_close = base
    for i, ts in enumerate(calendar.session_bars(exchange, day, Interval.M1)):
        close = round(base + amplitude * math.sin(i / period), 2)
        o = prev_close
        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                interval=Interval.M1,
                open=o,
                high=round(max(o, close) + 1, 2),
                low=round(min(o, close) - 1, 2),
                close=close,
                volume=1000,
            )
        )
        prev_close = close
    return bars


@pytest.fixture
def synthetic_day(calendar: MarketCalendar) -> Callable[..., list[Bar]]:
    def _make(**kw: object) -> list[Bar]:
        return make_synthetic_day(calendar, **kw)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def ist(trading_day: date) -> Callable[[int, int], datetime]:
    def _at(hour: int, minute: int = 0, second: int = 0) -> datetime:
        return datetime(
            trading_day.year, trading_day.month, trading_day.day, hour, minute, second, tzinfo=IST
        )

    return _at


@pytest.fixture
def sim_clock(ist: Callable[[int, int], datetime]) -> SimClock:
    return SimClock(ist(9, 0))
