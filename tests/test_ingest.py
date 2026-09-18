"""Archive round trips, validation / gap reports, corporate actions, incremental ingest."""

from __future__ import annotations

import itertools
import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from trading.core.clock import MarketCalendar
from trading.core.types import IST, Bar, Interval
from trading.training.ingest import (
    Archive,
    CorporateAction,
    apply_corporate_actions,
    bars_to_frame,
    detect_split_candidates,
    ingest,
    ingest_symbol,
    next_bar_start,
    plan_windows,
    validate_bars,
)

from .conftest import make_synthetic_day

SYM = "NSE:RELIANCE"
TODAY = date(2026, 9, 25)  # after all test data; a Friday


def daily_bars(calendar: MarketCalendar, start: date, end: date, symbol: str = SYM, base=2500.0):
    out = []
    for i, d in enumerate(calendar.trading_days("NSE", start, end)):
        close = base + i
        out.append(
            Bar(
                symbol=symbol,
                ts=datetime.combine(d, datetime.min.time(), tzinfo=IST),
                interval=Interval.D1,
                open=close - 1,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=1_000_000,
            )
        )
    return out


class FakeSource:
    """MarketData stub that fabricates synthetic bars for any requested range."""

    name = "fake"

    def __init__(self, calendar: MarketCalendar, fail_for: set[str] | None = None) -> None:
        self.calendar = calendar
        self.calls: list[tuple[str, Interval, datetime, datetime]] = []
        self.fail_for = fail_for or set()

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def instruments(self):
        return []

    async def ltp(self, symbols):
        return {}

    def subscribe_live(self, symbols):
        raise NotImplementedError

    async def historical(self, symbol, interval, start, end):
        self.calls.append((symbol, interval, start, end))
        if symbol in self.fail_for:
            raise RuntimeError("boom")
        if interval is Interval.D1:
            bars = daily_bars(self.calendar, start.date(), end.date(), symbol)
        else:
            bars = []
            for d in self.calendar.trading_days("NSE", start.date(), end.date()):
                bars += make_synthetic_day(self.calendar, symbol, d)
        return [b for b in bars if start <= b.ts <= end]


@pytest.fixture
def archive(tmp_path):
    return Archive(tmp_path / "archive", corporate_actions=tmp_path / "ca.json")


# --------------------------------------------------------------------------- archive


def test_write_read_round_trip_and_partitions(archive, calendar):
    bars = make_synthetic_day(calendar, SYM, date(2026, 9, 17)) + make_synthetic_day(
        calendar, SYM, date(2026, 9, 18)
    )
    stats = archive.write(SYM, Interval.M1, bars_to_frame(bars))
    assert stats.rows_written == 750 and stats.partitions == ["2026-09-17", "2026-09-18"]
    assert (archive.root / "1m" / "NSE_RELIANCE" / "2026-09-18.parquet").exists()
    assert archive.partitions(SYM, "1m") == ["2026-09-17", "2026-09-18"]
    assert archive.symbols("1m") == [SYM]
    assert archive.dates(SYM, "1m") == [date(2026, 9, 17), date(2026, 9, 18)]
    assert archive.last_ts(SYM, "1m") == datetime(2026, 9, 18, 15, 29, tzinfo=IST)

    df = archive.read(SYM, "1m")
    assert len(df) == 750 and str(df["ts"].dt.tz) == "Asia/Kolkata"
    back = archive.read_bars(SYM, Interval.M1, date(2026, 9, 18), date(2026, 9, 18))
    assert back == make_synthetic_day(calendar, SYM, date(2026, 9, 18))
    part = archive.read(
        SYM,
        "1m",
        datetime(2026, 9, 18, 10, 0, tzinfo=IST),
        datetime(2026, 9, 18, 10, 4, tzinfo=IST),
    )
    assert len(part) == 5 and part["ts"].iloc[0] == pd.Timestamp("2026-09-18 10:00", tz=IST)
    assert archive.read("NSE:NOPE", "1m").empty
    assert archive.last_ts("NSE:NOPE", "1m") is None


def test_rewrite_replaces_overlapping_rows(archive, calendar):
    bars = make_synthetic_day(calendar, SYM, date(2026, 9, 18))
    archive.write(SYM, Interval.M1, bars_to_frame(bars))
    changed = [
        b.model_copy(update={"close": b.close + 100, "high": b.high + 100}) for b in bars[:10]
    ]
    archive.write(SYM, Interval.M1, bars_to_frame(changed))
    df = archive.read(SYM, "1m")
    assert len(df) == 375  # no duplicates
    assert df["close"].iloc[0] == pytest.approx(bars[0].close + 100)
    assert df["close"].iloc[20] == pytest.approx(bars[20].close)


def test_daily_partitions_by_year(archive, calendar):
    bars = daily_bars(calendar, date(2025, 12, 20), date(2026, 1, 10))
    archive.write(SYM, Interval.D1, bars_to_frame(bars))
    assert archive.partitions(SYM, "1d") == ["2025", "2026"]
    df = archive.read(SYM, "1d", date(2026, 1, 1), date(2026, 1, 31))
    assert (df["ts"].dt.year == 2026).all() and len(df) == len(
        calendar.trading_days("NSE", date(2026, 1, 1), date(2026, 1, 10))
    )
    assert archive.dates(SYM, "1d")[0] == date(2025, 12, 22)


# --------------------------------------------------------------------------- validation


def test_validate_clean_day_is_ok(archive, calendar):
    df = bars_to_frame(make_synthetic_day(calendar, SYM, date(2026, 9, 18)))
    rep = validate_bars(df, SYM, "1m", calendar, today=TODAY)
    assert rep.ok and rep.coverage == 1.0 and rep.bars_expected == 375 == rep.bars_present
    assert rep.trading_days == 1 and rep.days_with_bars == 1 and rep.gaps == []
    assert "375 bars" in rep.summary()


def test_validate_reports_gaps_duplicates_stray_and_prices(calendar):
    bars = make_synthetic_day(calendar, SYM, date(2026, 9, 18))
    df = bars_to_frame(bars[:100] + bars[105:])  # 5 missing in-session bars
    dup = bars_to_frame([bars[0]])
    df = pd.concat([df, dup]).reset_index(drop=True)  # duplicate ts
    stray = bars[0].model_copy(update={"ts": datetime(2026, 9, 18, 8, 0, tzinfo=IST)})
    bad = bars[1].model_copy(update={"ts": datetime(2026, 9, 18, 9, 16, tzinfo=IST)})
    df = pd.concat([df, bars_to_frame([stray])]).reset_index(drop=True)
    df.loc[df["ts"] == pd.Timestamp(bad.ts), "high"] = 1.0  # high below open/close
    rep = validate_bars(df, SYM, "1m", calendar, today=TODAY)
    assert not rep.ok
    assert rep.gaps == [(date(2026, 9, 18), 5)]
    assert rep.duplicates == 1
    assert rep.out_of_session == 1
    assert rep.price_errors == 1
    assert any("09:00" in n for n in rep.notes)
    assert "in-session gaps" in rep.summary() and "price errors: 1" in rep.summary()


def test_validate_missing_days_and_partial_today(calendar):
    df = bars_to_frame(make_synthetic_day(calendar, SYM, date(2026, 9, 16)))
    rep = validate_bars(
        df, SYM, "1m", calendar, start=date(2026, 9, 14), end=date(2026, 9, 18), today=TODAY
    )
    # 14 Sep is a holiday; 15, 17, 18 are missing trading days
    assert rep.trading_days == 4 and rep.days_missing == [
        date(2026, 9, 15),
        date(2026, 9, 17),
        date(2026, 9, 18),
    ]
    assert rep.bars_expected == 4 * 375 and rep.bars_present == 375
    half = bars_to_frame(make_synthetic_day(calendar, SYM, date(2026, 9, 18))[:100])
    rep2 = validate_bars(half, SYM, "1m", calendar, today=date(2026, 9, 18))
    assert rep2.gaps == [] and any("partial" in n for n in rep2.notes)
    rep3 = validate_bars(bars_to_frame([]), SYM, "1m", calendar, today=TODAY)
    assert rep3.notes == ["no bars"]


def test_validate_daily_detects_split_candidates(calendar):
    bars = daily_bars(calendar, date(2026, 8, 3), date(2026, 8, 14))
    halved = [
        b
        if b.ts.date() < date(2026, 8, 10)
        else b.model_copy(
            update={"open": b.open / 2, "high": b.high / 2, "low": b.low / 2, "close": b.close / 2}
        )
        for b in bars
    ]
    df = bars_to_frame(halved)
    rep = validate_bars(df, SYM, "1d", calendar, today=TODAY)
    assert rep.ok and len(rep.split_candidates) == 1
    d, ratio, nearest = rep.split_candidates[0]
    assert d == date(2026, 8, 10) and nearest == 2 and ratio == pytest.approx(2.0, rel=0.01)
    assert "possible corporate action" in rep.summary()
    assert detect_split_candidates(bars_to_frame(bars)) == []


# --------------------------------------------------------------------------- corporate actions


def test_corporate_actions_apply_on_read_only(archive, calendar, tmp_path):
    bars = daily_bars(calendar, date(2026, 8, 3), date(2026, 8, 14))
    archive.write(SYM, Interval.D1, bars_to_frame(bars))
    (tmp_path / "ca.json").write_text(
        json.dumps(
            {"actions": {SYM: [{"ex_date": "2026-08-10", "ratio": 2.0, "note": "1:1 bonus"}]}}
        )
    )
    raw = archive.read(SYM, "1d")
    adj = archive.read(SYM, "1d", adjusted=True)
    before = adj["ts"].dt.date < date(2026, 8, 10)
    assert (adj.loc[before, "close"] * 2 == raw.loc[before, "close"]).all()
    assert (adj.loc[before, "volume"] == raw.loc[before, "volume"] * 2).all()
    assert (adj.loc[~before, "close"] == raw.loc[~before, "close"]).all()
    assert archive.corporate_actions()[SYM][0] == CorporateAction(
        SYM, date(2026, 8, 10), 2.0, "1:1 bonus"
    )
    assert apply_corporate_actions(raw, []).equals(raw)


# --------------------------------------------------------------------------- ingest


def test_plan_windows():
    start = datetime(2026, 1, 1, 9, 15, tzinfo=IST)
    end = start + timedelta(days=200)
    w = plan_windows(start, end, days=90)
    assert len(w) == 3 and w[0][0] == start and w[-1][1] == end
    assert all(b[0] == a[1] + timedelta(seconds=1) for a, b in itertools.pairwise(w))


async def test_ingest_then_top_up(archive, calendar):
    src = FakeSource(calendar)
    first = await ingest_symbol(
        src,
        archive,
        SYM,
        "1m",
        calendar,
        start=date(2026, 9, 14),
        end=date(2026, 9, 16),
        today=date(2026, 9, 16),
    )
    assert first.error is None and first.fetched == 750 == first.written and first.requests == 1
    assert first.report is not None and first.report.ok and first.report.trading_days == 2
    assert src.calls[0][2] == datetime(2026, 9, 14, tzinfo=IST)

    second = await ingest_symbol(
        src, archive, SYM, "1m", calendar, end=date(2026, 9, 18), today=date(2026, 9, 18)
    )
    assert src.calls[-1][2] == datetime(2026, 9, 17, 9, 15, tzinfo=IST)  # next session's first bar
    assert second.fetched == 750 and second.report.ok
    assert len(archive.read(SYM, "1m")) == 1500
    assert archive.dates(SYM, "1m") == [
        date(2026, 9, 15),
        date(2026, 9, 16),
        date(2026, 9, 17),
        date(2026, 9, 18),
    ]

    third = await ingest_symbol(
        src, archive, SYM, "1m", calendar, end=date(2026, 9, 18), today=date(2026, 9, 18)
    )
    assert third.range is None and third.requests == 0 and third.report.ok  # nothing new


async def test_ingest_daily_and_error_isolation(archive, calendar):
    src = FakeSource(calendar, fail_for={"NSE:BAD"})
    results = await ingest(
        src,
        archive,
        ["NSE:BAD", SYM],
        ["1d"],
        calendar,
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        today=TODAY,
    )
    bad, good = results
    assert bad.error is not None and "boom" in bad.error and bad.written == 0
    assert good.error is None and good.report.ok
    assert good.written == len(calendar.trading_days("NSE", date(2026, 8, 1), date(2026, 8, 31)))
    assert archive.partitions(SYM, "1d") == ["2026"]


async def test_ingest_lookback_when_archive_empty(archive, calendar):
    src = FakeSource(calendar)
    res = await ingest_symbol(
        src,
        archive,
        SYM,
        "1d",
        calendar,
        end=date(2026, 9, 18),
        lookback_days=10,
        today=date(2026, 9, 18),
    )
    assert src.calls[0][2].date() == date(2026, 9, 8)
    assert res.report.ok and res.fetched == len(
        calendar.trading_days("NSE", date(2026, 9, 8), date(2026, 9, 18))
    )


def test_next_bar_start_follows_sessions(calendar):
    assert next_bar_start(
        calendar, "NSE", datetime(2026, 9, 18, 10, 0, tzinfo=IST), Interval.M1
    ) == datetime(2026, 9, 18, 10, 1, tzinfo=IST)
    # last bar of Friday -> Monday's first bar
    assert next_bar_start(
        calendar, "NSE", datetime(2026, 9, 18, 15, 29, tzinfo=IST), Interval.M1
    ) == datetime(2026, 9, 21, 9, 15, tzinfo=IST)
    assert next_bar_start(
        calendar, "NSE", datetime(2026, 9, 18, tzinfo=IST), Interval.D1
    ) == datetime(2026, 9, 21, tzinfo=IST)
    # MCX evening: 23:54 on a DST day is the last 1m bar
    assert next_bar_start(
        calendar, "MCX", datetime(2026, 9, 18, 23, 54, tzinfo=IST), Interval.M1
    ) == datetime(2026, 9, 21, 9, 0, tzinfo=IST)
