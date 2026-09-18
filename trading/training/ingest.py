"""Bar archive: Parquet store, validation, incremental ingest, corporate actions.

Layout (``ARCHIVE_DIR``)::

    archive/1m/NSE_RELIANCE/2026-09-18.parquet     one file per session day (intraday)
    archive/1d/NSE_RELIANCE/2026.parquet           one file per year (daily)

Files hold columns ``ts`` (tz-aware IST), ``open, high, low, close`` (float),
``volume`` (int), ``oi`` (nullable int). The archive is *raw*: corporate actions are
applied on read (``adjusted=True``) from ``data/corporate_actions.json`` so the store
is never rewritten (constraint 10).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from trading.brokers.base import MarketData
from trading.brokers.symbols import parse_symbol
from trading.core.clock import MarketCalendar
from trading.core.types import IST, Bar, InstrumentKind, Interval, to_ist

log = logging.getLogger(__name__)

BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "oi"]
INTRADAY_WINDOW_DAYS = 90  # Dhan intraday limit per request


def symbol_to_dirname(symbol: str) -> str:
    return symbol.replace(":", "_", 1)


def dirname_to_symbol(name: str) -> str:
    return name.replace("_", ":", 1)


def bars_to_frame(bars: Sequence[Bar]) -> pd.DataFrame:
    if not bars:
        return empty_frame()
    df = pd.DataFrame(
        {
            "ts": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "oi": [b.oi for b in bars],
        }
    )
    return normalise_frame(df)


def frame_to_bars(df: pd.DataFrame, symbol: str, interval: Interval) -> list[Bar]:
    out = []
    for row in df.itertuples(index=False):
        oi = None if pd.isna(row.oi) else int(row.oi)
        out.append(
            Bar(
                symbol=symbol,
                ts=row.ts.to_pydatetime(),
                interval=interval,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
                oi=oi,
            )
        )
    return out


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS})
    df["ts"] = pd.Series(dtype="datetime64[ns, Asia/Kolkata]")
    df["volume"] = df["volume"].astype("int64")
    df["oi"] = df["oi"].astype("Int64")
    return df


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate on ts (last wins), enforce dtypes and IST."""
    df = df.copy()
    ts = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    df["ts"] = ts
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["oi"] = (
        df["oi"].astype("Int64") if "oi" in df else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    df = df[BAR_COLUMNS].sort_values("ts").drop_duplicates("ts", keep="last")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- corporate actions


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: date
    ratio: float  # new shares per old share (2.0 for a 1:1 bonus or 2-for-1 split)
    note: str = ""


def load_corporate_actions(path: Path | str) -> dict[str, list[CorporateAction]]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text()).get("actions", {})
    out: dict[str, list[CorporateAction]] = {}
    for symbol, items in raw.items():
        out[symbol] = sorted(
            (
                CorporateAction(
                    symbol, date.fromisoformat(i["ex_date"]), float(i["ratio"]), i.get("note", "")
                )
                for i in items
            ),
            key=lambda a: a.ex_date,
        )
    return out


def apply_corporate_actions(df: pd.DataFrame, actions: Sequence[CorporateAction]) -> pd.DataFrame:
    """Back-adjust bars before each ex-date so the series is continuous."""
    if df.empty or not actions:
        return df
    df = df.copy()
    dates = df["ts"].dt.date
    for a in actions:
        mask = dates < a.ex_date
        if not mask.any():
            continue
        for c in ("open", "high", "low", "close"):
            df.loc[mask, c] = df.loc[mask, c] / a.ratio
        df.loc[mask, "volume"] = (df.loc[mask, "volume"] * a.ratio).round().astype("int64")
    return df


def detect_split_candidates(
    df: pd.DataFrame, *, threshold: float = 0.35
) -> list[tuple[date, float, float]]:
    """Daily close-to-close ratios far from 1 that look like splits/bonuses.

    Returns (date, observed ratio prev/close, nearest simple ratio)."""
    if len(df) < 2:
        return []
    closes = df["close"].to_numpy()
    out = []
    candidates = [2, 3, 4, 5, 10, 1.5, 0.5, 0.25, 0.2, 0.1]
    for i in range(1, len(closes)):
        if closes[i] <= 0 or closes[i - 1] <= 0:
            continue
        r = closes[i - 1] / closes[i]
        if abs(r - 1) >= threshold:
            nearest = min(candidates, key=lambda c: abs(c - r))
            out.append((df["ts"].iloc[i].date(), round(float(r), 4), nearest))
    return out


# --------------------------------------------------------------------------- archive


@dataclass
class WriteStats:
    symbol: str
    interval: Interval
    rows_in: int = 0
    rows_written: int = 0
    partitions: list[str] = field(default_factory=list)


class Archive:
    def __init__(self, root: Path | str, *, corporate_actions: Path | str | None = None) -> None:
        self.root = Path(root)
        self.actions_path = Path(corporate_actions) if corporate_actions else None
        self._actions: dict[str, list[CorporateAction]] | None = None

    # ------------------------------------------------------------------ paths
    def symbol_dir(self, symbol: str, interval: Interval | str) -> Path:
        return self.root / Interval(interval).value / symbol_to_dirname(symbol)

    @staticmethod
    def partition_key(ts: datetime | pd.Timestamp, interval: Interval) -> str:
        ts = to_ist(ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts)
        return str(ts.year) if interval is Interval.D1 else ts.date().isoformat()

    def partitions(self, symbol: str, interval: Interval | str) -> list[str]:
        d = self.symbol_dir(symbol, interval)
        return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []

    def symbols(self, interval: Interval | str) -> list[str]:
        d = self.root / Interval(interval).value
        return (
            sorted(dirname_to_symbol(p.name) for p in d.iterdir() if p.is_dir())
            if d.exists()
            else []
        )

    def dates(self, symbol: str, interval: Interval | str) -> list[date]:
        interval = Interval(interval)
        if interval is Interval.D1:
            df = self.read(symbol, interval)
            return sorted(set(df["ts"].dt.date)) if not df.empty else []
        return [date.fromisoformat(k) for k in self.partitions(symbol, interval)]

    def last_ts(self, symbol: str, interval: Interval | str) -> datetime | None:
        keys = self.partitions(symbol, interval)
        if not keys:
            return None
        df = pd.read_parquet(self.symbol_dir(symbol, interval) / f"{keys[-1]}.parquet")
        return df["ts"].max().to_pydatetime() if not df.empty else None

    # ------------------------------------------------------------------ io
    def write(self, symbol: str, interval: Interval | str, df: pd.DataFrame) -> WriteStats:
        """Merge ``df`` into the archive (existing rows with the same ts are replaced)."""
        interval = Interval(interval)
        stats = WriteStats(symbol, interval, rows_in=len(df))
        if df.empty:
            return stats
        df = normalise_frame(df)
        d = self.symbol_dir(symbol, interval)
        d.mkdir(parents=True, exist_ok=True)
        keys = df["ts"].map(lambda t: self.partition_key(t, interval))
        for key, part in df.groupby(keys):
            path = d / f"{key}.parquet"
            if path.exists():
                merged = normalise_frame(pd.concat([pd.read_parquet(path), part]))
            else:
                merged = part.reset_index(drop=True)
            tmp = path.with_suffix(".tmp")
            merged.to_parquet(tmp, index=False)
            tmp.replace(path)
            stats.rows_written += len(part)
            stats.partitions.append(str(key))
        return stats

    def read(
        self,
        symbol: str,
        interval: Interval | str,
        start: datetime | date | None = None,
        end: datetime | date | None = None,
        *,
        adjusted: bool = False,
    ) -> pd.DataFrame:
        """Bars in ``[start, end]`` (inclusive; dates cover whole days)."""
        interval = Interval(interval)
        keys = self.partitions(symbol, interval)
        if not keys:
            return empty_frame()
        lo, hi = self._bounds(start, end)
        parts = []
        for key in keys:
            if not self._key_overlaps(key, interval, lo, hi):
                continue
            parts.append(pd.read_parquet(self.symbol_dir(symbol, interval) / f"{key}.parquet"))
        if not parts:
            return empty_frame()
        df = normalise_frame(pd.concat(parts))
        if lo is not None:
            df = df[df["ts"] >= lo]
        if hi is not None:
            df = df[df["ts"] <= hi]
        df = df.reset_index(drop=True)
        if adjusted:
            df = apply_corporate_actions(df, self.corporate_actions().get(symbol, []))
        return df

    def read_bars(
        self,
        symbol: str,
        interval: Interval | str,
        start: datetime | date | None = None,
        end: datetime | date | None = None,
        *,
        adjusted: bool = False,
    ) -> list[Bar]:
        return frame_to_bars(
            self.read(symbol, interval, start, end, adjusted=adjusted), symbol, Interval(interval)
        )

    def corporate_actions(self) -> dict[str, list[CorporateAction]]:
        if self._actions is None:
            self._actions = load_corporate_actions(self.actions_path) if self.actions_path else {}
        return self._actions

    @staticmethod
    def _bounds(start, end):  # type: ignore[no-untyped-def]
        def lo_of(v):  # type: ignore[no-untyped-def]
            if v is None:
                return None
            if isinstance(v, datetime):
                return to_ist(v)
            return datetime.combine(v, datetime.min.time(), tzinfo=IST)

        def hi_of(v):  # type: ignore[no-untyped-def]
            if v is None:
                return None
            if isinstance(v, datetime):
                return to_ist(v)
            return datetime.combine(v, datetime.max.time(), tzinfo=IST)

        return lo_of(start), hi_of(end)

    @staticmethod
    def _key_overlaps(key: str, interval: Interval, lo, hi) -> bool:  # type: ignore[no-untyped-def]
        if interval is Interval.D1:
            y = int(key)
            k_lo = datetime(y, 1, 1, tzinfo=IST)
            k_hi = datetime(y, 12, 31, 23, 59, 59, tzinfo=IST)
        else:
            d = date.fromisoformat(key)
            k_lo = datetime.combine(d, datetime.min.time(), tzinfo=IST)
            k_hi = datetime.combine(d, datetime.max.time(), tzinfo=IST)
        return (lo is None or k_hi >= lo) and (hi is None or k_lo <= hi)


# --------------------------------------------------------------------------- validation


@dataclass
class GapReport:
    symbol: str
    interval: Interval
    start: date | None = None
    end: date | None = None
    trading_days: int = 0
    days_with_bars: int = 0
    days_missing: list[date] = field(default_factory=list)
    bars_expected: int = 0
    bars_present: int = 0
    gaps: list[tuple[date, int]] = field(default_factory=list)  # (day, missing bars)
    duplicates: int = 0
    out_of_session: int = 0
    price_errors: int = 0
    zero_volume_bars: int = 0
    split_candidates: list[tuple[date, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.duplicates or self.out_of_session or self.price_errors or self.days_missing
        )

    @property
    def coverage(self) -> float:
        return self.bars_present / self.bars_expected if self.bars_expected else 1.0

    def summary(self) -> str:
        lines = [
            f"{self.symbol} {self.interval.value}: {self.bars_present} bars over "
            f"{self.days_with_bars}/{self.trading_days} trading days "
            f"({self.start} .. {self.end}), coverage {self.coverage:.1%}",
        ]
        if self.days_missing:
            shown = ", ".join(str(d) for d in self.days_missing[:5])
            more = f" (+{len(self.days_missing) - 5} more)" if len(self.days_missing) > 5 else ""
            lines.append(f"  missing days: {shown}{more}")
        if self.gaps:
            worst = sorted(self.gaps, key=lambda g: -g[1])[:5]
            lines.append(
                f"  in-session gaps on {len(self.gaps)} days, worst: "
                + ", ".join(f"{d} (-{n})" for d, n in worst)
            )
        for label, n in (
            ("duplicates", self.duplicates),
            ("out-of-session bars", self.out_of_session),
            ("price errors", self.price_errors),
            ("zero-volume bars", self.zero_volume_bars),
        ):
            if n:
                lines.append(f"  {label}: {n}")
        for d, r, nearest in self.split_candidates:
            lines.append(f"  possible corporate action on {d}: prev/close = {r} (~{nearest})")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def validate_bars(
    df: pd.DataFrame,
    symbol: str,
    interval: Interval | str,
    calendar: MarketCalendar,
    *,
    start: date | None = None,
    end: date | None = None,
    today: date | None = None,
) -> GapReport:
    """Check completeness and sanity of one symbol's bars.

    ``start``/``end`` default to the first/last bar; pass them to check that a
    requested range is fully covered. A partial *current* day is reported as a note,
    not a gap.
    """
    interval = Interval(interval)
    parsed = parse_symbol(symbol)
    exchange = parsed.exchange
    rep = GapReport(symbol, interval)
    if df.empty and start is None:
        rep.notes.append("no bars")
        return rep
    df = df.sort_values("ts").reset_index(drop=True)
    bar_dates = df["ts"].dt.date if not df.empty else pd.Series([], dtype=object)
    rep.start = start or bar_dates.iloc[0]
    rep.end = end or bar_dates.iloc[-1]
    today = today or datetime.now(IST).date()

    # duplicates & prices
    rep.duplicates = int(df["ts"].duplicated().sum())
    bad = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    rep.price_errors = int(bad.sum())
    if parsed.kind is not InstrumentKind.INDEX:
        rep.zero_volume_bars = int((df["volume"] <= 0).sum())

    # per-day completeness
    days = calendar.trading_days(exchange, rep.start, rep.end)
    rep.trading_days = len(days)
    by_day = {d: g for d, g in df.groupby(bar_dates)} if not df.empty else {}
    present_days = set(by_day)
    rep.days_with_bars = len(present_days & set(days))
    stray = present_days - set(days)
    if stray:
        rep.out_of_session += int(sum(len(by_day[d]) for d in stray))
        rep.notes.append(f"bars on {len(stray)} non-trading days, e.g. {sorted(stray)[0]}")
    for d in days:
        expected = calendar.session_bars(exchange, d, interval)
        rep.bars_expected += len(expected)
        g = by_day.get(d)
        if g is None:
            if d == today:
                rep.notes.append("no bars yet for today")
            else:
                rep.days_missing.append(d)
            continue
        present = set(g["ts"].dt.to_pydatetime())
        exp_set = set(expected)
        rep.bars_present += len(present & exp_set)
        outside = len(present - exp_set)
        rep.out_of_session += outside
        missing = len(exp_set - present)
        if missing:
            if d == today:
                rep.notes.append(f"today is partial: {missing} bars still to come")
            else:
                rep.gaps.append((d, missing))
    if interval is Interval.D1:
        rep.split_candidates = detect_split_candidates(df)
    if rep.out_of_session and interval is not Interval.D1 and not df.empty:
        first = df["ts"].iloc[0]
        if first.hour < 9:
            rep.notes.append("bars before 09:00 IST - check the epoch/timezone conversion")
    return rep


# --------------------------------------------------------------------------- ingest


@dataclass
class IngestResult:
    symbol: str
    interval: Interval
    fetched: int = 0
    written: int = 0
    requests: int = 0
    range: tuple[datetime, datetime] | None = None
    report: GapReport | None = None
    error: str | None = None


def _bar_start_for(interval: Interval, calendar: MarketCalendar, exchange, d: date) -> datetime:  # type: ignore[no-untyped-def]
    bars = calendar.session_bars(exchange, d, interval)
    return bars[0] if bars else datetime.combine(d, datetime.min.time(), tzinfo=IST)


def next_bar_start(
    calendar: MarketCalendar, exchange: object, last: datetime, interval: Interval
) -> datetime:
    """Start of the first bar after ``last`` according to the session calendar."""
    if interval is Interval.D1:
        nd = calendar.next_trading_day(exchange, last.date())  # type: ignore[arg-type]
        return datetime.combine(nd, time(0, 0), tzinfo=IST)
    candidate = last + timedelta(seconds=interval.seconds)
    bounds = calendar.session_bounds(exchange, last.date())  # type: ignore[arg-type]
    if bounds and candidate < bounds[1]:
        return candidate
    nd = calendar.next_trading_day(exchange, last.date())  # type: ignore[arg-type]
    return calendar.session_bars(exchange, nd, interval)[0]  # type: ignore[arg-type]


def plan_windows(
    start: datetime, end: datetime, *, days: int = INTRADAY_WINDOW_DAYS
) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into windows of at most ``days`` days."""
    out = []
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days) - timedelta(seconds=1))
        out.append((cur, nxt))
        cur = nxt + timedelta(seconds=1)
    return out


async def ingest_symbol(
    source: MarketData,
    archive: Archive,
    symbol: str,
    interval: Interval | str,
    calendar: MarketCalendar,
    *,
    start: datetime | date | None = None,
    end: datetime | date | None = None,
    lookback_days: int = 30,
    validate: bool = True,
    today: date | None = None,
) -> IngestResult:
    """Fetch bars from ``source`` and merge them into the archive.

    Without ``start`` the ingest tops up from the last archived bar (or
    ``lookback_days`` back when the archive is empty). Nothing is deleted.
    """
    interval = Interval(interval)
    parsed = parse_symbol(symbol)
    res = IngestResult(symbol, interval)
    today = today or datetime.now(IST).date()
    end_dt = archive._bounds(None, end or today)[1]
    if start is None:
        last = archive.last_ts(symbol, interval)
        if last is not None:
            start_dt = next_bar_start(calendar, parsed.exchange, last, interval)
        else:
            start_dt = _bar_start_for(
                interval, calendar, parsed.exchange, today - timedelta(days=lookback_days)
            )
    else:
        start_dt = archive._bounds(start, None)[0]
    assert start_dt is not None and end_dt is not None
    if start_dt > end_dt:
        res.report = (
            validate_bars(archive.read(symbol, interval), symbol, interval, calendar, today=today)
            if validate
            else None
        )
        return res
    res.range = (start_dt, end_dt)
    windows = (
        plan_windows(start_dt, end_dt) if interval is not Interval.D1 else [(start_dt, end_dt)]
    )
    frames = []
    try:
        for lo, hi in windows:
            bars = await source.historical(symbol, interval, lo, hi)
            res.requests += 1
            res.fetched += len(bars)
            if bars:
                df = bars_to_frame(bars)
                stats = archive.write(symbol, interval, df)
                res.written += stats.rows_written
                frames.append(df)
    except Exception as e:  # keep going with other symbols; report the failure
        log.exception("ingest %s %s failed", symbol, interval.value)
        res.error = f"{type(e).__name__}: {e}"
    if validate:  # read whole days so a mid-day top-up start does not look like a gap
        got = archive.read(symbol, interval, start_dt.date(), end_dt)
        res.report = validate_bars(
            got, symbol, interval, calendar, start=start_dt.date(), end=end_dt.date(), today=today
        )
    return res


async def ingest(
    source: MarketData,
    archive: Archive,
    symbols: Sequence[str],
    intervals: Sequence[Interval | str],
    calendar: MarketCalendar,
    **kw,  # type: ignore[no-untyped-def]
) -> list[IngestResult]:
    out = []
    for symbol in symbols:
        for interval in intervals:
            out.append(await ingest_symbol(source, archive, symbol, interval, calendar, **kw))
    return out
