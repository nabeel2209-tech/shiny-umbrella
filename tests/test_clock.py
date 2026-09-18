from datetime import UTC, date, datetime, time, timedelta

import pytest

from trading.core.clock import DEFAULT_HOLIDAYS_FILE, MarketCalendar, SimClock, us_dst_active
from trading.core.types import IST, Interval


def at(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST)


def test_nse_session_edges(calendar):
    assert not calendar.is_open("NSE", at(2026, 9, 18, 9, 14, 59))
    assert calendar.is_open("NSE", at(2026, 9, 18, 9, 15))
    assert calendar.is_open("NSE", at(2026, 9, 18, 15, 29, 59))
    assert not calendar.is_open("NSE", at(2026, 9, 18, 15, 30))
    assert calendar.is_open("NFO", at(2026, 9, 18, 12, 0))  # NFO follows NSE


def test_mcx_session_follows_us_dst(calendar):
    # September: US DST in force -> MCX closes 23:55
    assert not calendar.is_open("MCX", at(2026, 9, 18, 8, 59))
    assert calendar.is_open("MCX", at(2026, 9, 18, 9, 0))
    assert calendar.is_open("MCX", at(2026, 9, 18, 23, 30))
    assert not calendar.is_open("MCX", at(2026, 9, 18, 23, 55))
    # January: no DST -> 23:30
    assert calendar.is_open("MCX", at(2026, 1, 16, 23, 29))
    assert not calendar.is_open("MCX", at(2026, 1, 16, 23, 30))
    assert calendar.regular_close("MCX", date(2026, 3, 6)) == time(23, 30)
    assert calendar.regular_close("MCX", date(2026, 3, 9)) == time(23, 55)
    assert calendar.regular_close("NSE", date(2026, 3, 9)) == time(15, 30)


def test_us_dst_window():
    assert not us_dst_active(date(2026, 3, 7))
    assert us_dst_active(date(2026, 3, 8))  # second Sunday of March
    assert us_dst_active(date(2026, 10, 31))
    assert not us_dst_active(date(2026, 11, 1))  # first Sunday of November


def test_mcx_close_configurable():
    cal = MarketCalendar.load(DEFAULT_HOLIDAYS_FILE, mcx_close=time(23, 55), mcx_dst_close=None)
    assert cal.is_open("MCX", at(2026, 1, 16, 23, 40))
    assert len(cal.session_bars("MCX", date(2026, 1, 16))) == 895
    fixed = MarketCalendar.load(DEFAULT_HOLIDAYS_FILE, mcx_dst_close=None)
    assert not fixed.is_open("MCX", at(2026, 9, 18, 23, 30))


def test_weekend_and_holidays(calendar):
    assert calendar.verified
    assert not calendar.is_trading_day("NSE", date(2026, 9, 19))  # Saturday
    assert not calendar.is_trading_day("NSE", date(2026, 9, 20))  # Sunday
    assert not calendar.is_trading_day("NSE", date(2026, 10, 2))  # Gandhi Jayanti
    assert not calendar.is_trading_day("MCX", date(2026, 10, 2))
    assert not calendar.is_open("NSE", at(2026, 10, 2, 10, 0))
    assert not calendar.is_trading_day("NSE", date(2026, 1, 15))  # municipal election (NSE API)
    assert not calendar.is_trading_day("NFO", date(2026, 1, 15))
    assert calendar.is_trading_day("NSE", date(2026, 1, 16))
    # MCX on New Year's Day: morning only
    assert calendar.session_bounds("MCX", date(2026, 1, 1)) == (
        at(2026, 1, 1, 9, 0),
        at(2026, 1, 1, 17, 0),
    )


def test_mcx_evening_only_session_on_nse_holiday(calendar):
    d = date(2026, 3, 3)
    assert not calendar.is_trading_day("NSE", d)
    assert calendar.is_trading_day("MCX", d)
    assert not calendar.is_open("MCX", at(2026, 3, 3, 10, 0))
    assert calendar.is_open("MCX", at(2026, 3, 3, 18, 0))
    assert calendar.session_bounds("MCX", d) == (at(2026, 3, 3, 17, 0), at(2026, 3, 3, 23, 30))
    # same pattern inside the DST window resolves the null close to 23:55
    assert calendar.session_bounds("MCX", date(2026, 3, 26)) == (
        at(2026, 3, 26, 17, 0),
        at(2026, 3, 26, 23, 55),
    )
    assert not calendar.is_trading_day("NSE", date(2026, 3, 26))


def test_next_open(calendar):
    # before today's open -> today
    assert calendar.next_open("NSE", at(2026, 9, 18, 8, 0)) == at(2026, 9, 18, 9, 15)
    # during the session -> next trading day
    assert calendar.next_open("NSE", at(2026, 9, 18, 12, 0)) == at(2026, 9, 21, 9, 15)
    # Friday evening -> Monday
    assert calendar.next_open("NSE", at(2026, 9, 18, 16, 0)) == at(2026, 9, 21, 9, 15)
    # Thursday 1 Oct evening: Fri 2 Oct is a holiday, then the weekend
    assert calendar.next_open("NSE", at(2026, 10, 1, 16, 0)) == at(2026, 10, 5, 9, 15)


def test_next_close(calendar):
    assert calendar.next_close("NSE", at(2026, 9, 18, 12, 0)) == at(2026, 9, 18, 15, 30)
    assert calendar.next_close("NSE", at(2026, 9, 18, 16, 0)) == at(2026, 9, 21, 15, 30)


def test_trading_day_navigation(calendar):
    assert calendar.next_trading_day("NSE", date(2026, 9, 18)) == date(2026, 9, 21)
    assert calendar.previous_trading_day("NSE", date(2026, 10, 5)) == date(2026, 10, 1)
    days = calendar.trading_days("NSE", date(2026, 9, 28), date(2026, 10, 5))
    assert days == [
        date(2026, 9, 28),
        date(2026, 9, 29),
        date(2026, 9, 30),
        date(2026, 10, 1),
        date(2026, 10, 5),
    ]


def test_session_bars(calendar):
    d = date(2026, 9, 18)
    m1 = calendar.session_bars("NSE", d, Interval.M1)
    assert len(m1) == 375
    assert m1[0] == at(2026, 9, 18, 9, 15)
    assert m1[-1] == at(2026, 9, 18, 15, 29)
    assert len(calendar.session_bars("NSE", d, "5m")) == 75
    assert calendar.session_bars("NSE", d, Interval.D1) == [at(2026, 9, 18, 0, 0)]
    assert len(calendar.session_bars("MCX", d, Interval.M1)) == 895  # DST close 23:55
    assert len(calendar.session_bars("MCX", date(2026, 1, 16), Interval.M1)) == 870
    assert calendar.session_bars("NSE", date(2026, 9, 19)) == []


def test_bar_start(calendar):
    ts = at(2026, 9, 18, 10, 17, 42)
    assert calendar.bar_start(ts, Interval.M1) == at(2026, 9, 18, 10, 17)
    assert calendar.bar_start(ts, Interval.M5) == at(2026, 9, 18, 10, 15)
    assert calendar.bar_start(ts, Interval.H1) == at(2026, 9, 18, 10, 0)
    assert calendar.bar_start(ts, Interval.D1) == at(2026, 9, 18, 0, 0)


def test_naive_and_utc_inputs(calendar):
    with pytest.raises(ValueError, match="naive"):
        calendar.is_open("NSE", datetime(2026, 9, 18, 10, 0))
    assert calendar.is_open("NSE", datetime(2026, 9, 18, 4, 30, tzinfo=UTC))  # 10:00 IST


def test_sim_clock():
    c = SimClock(at(2026, 9, 18, 9, 0))
    c.advance(timedelta(minutes=15))
    assert c.now() == at(2026, 9, 18, 9, 15)
    c.set(at(2026, 9, 18, 9, 16))
    with pytest.raises(ValueError):
        c.set(at(2026, 9, 18, 9, 0))
    with pytest.raises(ValueError):
        SimClock(datetime(2026, 9, 18, 9, 0))


def test_missing_holidays_file_means_no_holidays(tmp_path):
    cal = MarketCalendar.load(tmp_path / "nope.json")
    assert cal.is_trading_day("NSE", date(2026, 10, 2))
    assert not cal.verified
