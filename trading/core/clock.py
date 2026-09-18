"""Market calendar and clock abstraction.

Sessions (IST):
- NSE / NFO: 09:15-15:30
- MCX: 09:00-23:30, extended to 23:55 while US daylight-saving time is in force
  (second Sunday of March to first Sunday of November), both configurable.

Holidays and special sessions are loaded from a JSON file (see
``data/holidays/holidays.json``). A special session's close may be ``null`` meaning
"the regular close for that date". Session intervals are half-open ``[open, close)``.
Daily bars are stamped at 00:00 IST of the session date.

``Clock`` is the source of "now" for agents: ``SystemClock`` in live/paper mode,
``SimClock`` for archive replay and backtests so that the same code runs in both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol

from trading.core.types import IST, Exchange, Interval, now_ist, to_ist

DEFAULT_HOLIDAYS_FILE = Path(__file__).resolve().parents[2] / "data" / "holidays" / "holidays.json"

NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
MCX_OPEN = time(9, 0)
MCX_CLOSE = time(23, 30)
MCX_DST_CLOSE = time(23, 55)


def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)  # first Sunday
    return d + timedelta(weeks=n - 1)


def us_dst_active(d: date) -> bool:
    """US daylight-saving window: second Sunday of March to first Sunday of November."""
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


@dataclass(frozen=True)
class Session:
    open: time
    close: time | None  # None only inside ``special_sessions``: use the regular close

    def bounds(self, d: date) -> tuple[datetime, datetime]:
        assert self.close is not None, "unresolved session close"
        return (
            datetime.combine(d, self.open, tzinfo=IST),
            datetime.combine(d, self.close, tzinfo=IST),
        )


def _calendar_key(exchange: Exchange | str) -> str:
    """NFO trades on the NSE calendar."""
    ex = Exchange(exchange)
    return "NSE" if ex in {Exchange.NSE, Exchange.NFO} else ex.value


@dataclass
class MarketCalendar:
    sessions: dict[str, Session]
    holidays: dict[str, set[date]] = field(default_factory=dict)
    special_sessions: dict[str, dict[date, Session]] = field(default_factory=dict)
    verified: bool = False
    mcx_dst_close: time | None = MCX_DST_CLOSE  # None disables the DST extension

    # ----------------------------------------------------------------- construction
    @classmethod
    def load(
        cls,
        holidays_file: Path | str | None = None,
        *,
        mcx_close: time = MCX_CLOSE,
        mcx_dst_close: time | None = MCX_DST_CLOSE,
    ) -> MarketCalendar:
        path = Path(holidays_file) if holidays_file else DEFAULT_HOLIDAYS_FILE
        raw = json.loads(path.read_text()) if path.exists() else {}
        holidays: dict[str, set[date]] = {}
        special: dict[str, dict[date, Session]] = {}
        for key in ("NSE", "MCX"):
            block = raw.get(key, {})
            holidays[key] = {date.fromisoformat(d) for d in block.get("holidays", [])}
            special[key] = {
                date.fromisoformat(d): Session(
                    time.fromisoformat(o), time.fromisoformat(c) if c else None
                )
                for d, (o, c) in block.get("special_sessions", {}).items()
            }
        return cls(
            sessions={
                "NSE": Session(NSE_OPEN, NSE_CLOSE),
                "MCX": Session(MCX_OPEN, mcx_close),
            },
            holidays=holidays,
            special_sessions=special,
            verified=bool(raw.get("_verified", False)),
            mcx_dst_close=mcx_dst_close,
        )

    # ----------------------------------------------------------------- queries
    def regular_close(self, exchange: Exchange | str, d: date) -> time:
        key = _calendar_key(exchange)
        if key == "MCX" and self.mcx_dst_close is not None and us_dst_active(d):
            return self.mcx_dst_close
        close = self.sessions[key].close
        assert close is not None
        return close

    def session(self, exchange: Exchange | str, d: date) -> Session | None:
        """The session for ``d`` or None if the exchange is closed that day."""
        key = _calendar_key(exchange)
        special = self.special_sessions.get(key, {}).get(d)
        if special is not None:
            close = special.close if special.close is not None else self.regular_close(key, d)
            return Session(special.open, close)
        if d.weekday() >= 5 or d in self.holidays.get(key, set()):
            return None
        return Session(self.sessions[key].open, self.regular_close(key, d))

    def is_trading_day(self, exchange: Exchange | str, d: date) -> bool:
        return self.session(exchange, d) is not None

    def session_bounds(self, exchange: Exchange | str, d: date) -> tuple[datetime, datetime] | None:
        s = self.session(exchange, d)
        return s.bounds(d) if s else None

    def is_open(self, exchange: Exchange | str, ts: datetime) -> bool:
        ts = to_ist(ts)
        bounds = self.session_bounds(exchange, ts.date())
        if bounds is None:
            return False
        open_, close = bounds
        return open_ <= ts < close

    def next_open(self, exchange: Exchange | str, ts: datetime) -> datetime:
        """Earliest session open strictly after ``ts`` (or today's open if it is still ahead)."""
        ts = to_ist(ts)
        d = ts.date()
        for _ in range(400):  # guard against a broken calendar
            bounds = self.session_bounds(exchange, d)
            if bounds and bounds[0] > ts:
                return bounds[0]
            d += timedelta(days=1)
        raise RuntimeError(f"no session found for {exchange} within 400 days of {ts}")

    def next_close(self, exchange: Exchange | str, ts: datetime) -> datetime:
        """Close of the current session if open, else close of the next session."""
        ts = to_ist(ts)
        bounds = self.session_bounds(exchange, ts.date())
        if bounds and ts < bounds[1]:
            return bounds[1]
        nxt = self.next_open(exchange, ts)
        return self.session_bounds(exchange, nxt.date())[1]  # type: ignore[index]

    def next_trading_day(self, exchange: Exchange | str, d: date) -> date:
        d += timedelta(days=1)
        while not self.is_trading_day(exchange, d):
            d += timedelta(days=1)
        return d

    def previous_trading_day(self, exchange: Exchange | str, d: date) -> date:
        d -= timedelta(days=1)
        while not self.is_trading_day(exchange, d):
            d -= timedelta(days=1)
        return d

    def trading_days(self, exchange: Exchange | str, start: date, end: date) -> list[date]:
        """Inclusive range of trading days."""
        out = []
        d = start
        while d <= end:
            if self.is_trading_day(exchange, d):
                out.append(d)
            d += timedelta(days=1)
        return out

    def session_bars(
        self, exchange: Exchange | str, d: date, interval: Interval | str = Interval.M1
    ) -> list[datetime]:
        """Bar start timestamps for one session (empty list if closed).

        The daily bar is stamped at 00:00 IST of the session date.
        """
        interval = Interval(interval)
        bounds = self.session_bounds(exchange, d)
        if bounds is None:
            return []
        open_, close = bounds
        if interval is Interval.D1:
            return [datetime.combine(d, time(0, 0), tzinfo=IST)]
        step = timedelta(seconds=interval.seconds)
        out = []
        t = open_
        while t < close:
            out.append(t)
            t += step
        return out

    def bar_start(self, ts: datetime, interval: Interval | str) -> datetime:
        """Floor ``ts`` to the start of its bar (session-relative for intraday)."""
        interval = Interval(interval)
        ts = to_ist(ts)
        if interval is Interval.D1:
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        secs = interval.seconds
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = int((ts - midnight).total_seconds())
        return midnight + timedelta(seconds=elapsed - elapsed % secs)


# ----------------------------------------------------------------------------- clocks


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return now_ist()


class SimClock:
    """Manually advanced clock for replay and backtests."""

    def __init__(self, start: datetime) -> None:
        self._now = to_ist(start)

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        ts = to_ist(ts)
        if ts < self._now:
            raise ValueError(f"SimClock cannot go backwards: {ts} < {self._now}")
        self._now = ts

    def advance(self, delta: timedelta) -> None:
        self._now += delta


_default_calendar: MarketCalendar | None = None


def get_calendar() -> MarketCalendar:
    """Process-wide calendar loaded from the default holidays file (or settings)."""
    global _default_calendar
    if _default_calendar is None:
        from trading.core.config import get_settings

        s = get_settings()
        _default_calendar = MarketCalendar.load(s.holidays_file, mcx_close=s.mcx_close)
    return _default_calendar
