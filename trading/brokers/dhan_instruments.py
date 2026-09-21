"""Dhan instrument master -> ``SymbolMap``.

Downloads the detailed scrip master CSV (public, ~35 MB, refreshed daily by Dhan),
keeps a per-day Parquet cache in ``data/instruments/`` and builds canonical
symbols. Column conventions are documented in ``brokers/README.md``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

from trading.brokers.base import Instrument
from trading.brokers.symbols import (
    MCX_CONTRACT_MULTIPLIER,
    SymbolError,
    SymbolMap,
    expiry_token,
    index_underlying,
    make_symbol,
)
from trading.core.types import IST, Exchange, InstrumentKind, OptionType

log = logging.getLogger(__name__)

INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

COLUMNS = [
    "EXCH_ID",
    "SEGMENT",
    "SECURITY_ID",
    "ISIN",
    "INSTRUMENT",
    "UNDERLYING_SYMBOL",
    "SYMBOL_NAME",
    "DISPLAY_NAME",
    "INSTRUMENT_TYPE",
    "SERIES",
    "LOT_SIZE",
    "SM_EXPIRY_DATE",
    "STRIKE_PRICE",
    "OPTION_TYPE",
    "TICK_SIZE",
    "EXPIRY_FLAG",
    "SM_FREEZE_QTY",
]

DHAN_SEGMENT = {  # (EXCH_ID, SEGMENT) -> Dhan exchangeSegment string
    ("NSE", "E"): "NSE_EQ",
    ("NSE", "I"): "IDX_I",
    ("NSE", "D"): "NSE_FNO",
    ("MCX", "M"): "MCX_COMM",
}
EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}
EQUITY_TYPES = {"ES", "ETF"}
MONTHLY_FLAGS = {"M", "Q", "H"}


def read_master(source: str | Path | BytesIO) -> pd.DataFrame:
    """Read the detailed CSV keeping only the columns and rows we use."""
    df = pd.read_csv(
        source,
        usecols=lambda c: c in COLUMNS,
        dtype={
            "SECURITY_ID": str,
            "ISIN": str,
            "SERIES": str,
            "OPTION_TYPE": str,
            "EXPIRY_FLAG": str,
            "SM_EXPIRY_DATE": str,
        },
        low_memory=False,
    )
    return filter_master(df)


def filter_master(df: pd.DataFrame) -> pd.DataFrame:
    keys = list(zip(df["EXCH_ID"], df["SEGMENT"], strict=True))
    keep = pd.Series([k in DHAN_SEGMENT for k in keys], index=df.index)
    out = df[keep].copy()
    out["SECURITY_ID"] = out["SECURITY_ID"].astype(str)
    return out.reset_index(drop=True)


def _tick(raw: float | None) -> float:
    """Dhan lists ticks in paise for tradable rows (10.0 = Rs 0.10) and rupees for indices."""
    if raw is None or pd.isna(raw) or raw <= 0:
        return 0.05
    return round(raw / 100, 4) if raw >= 1 else float(raw)


def _int(v: object, default: int) -> int:
    try:
        n = int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def _expiry(v: object) -> date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v)[:10]
    if s.startswith("0001"):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def build_symbol_map(df: pd.DataFrame, *, source: str = INSTRUMENT_MASTER_URL) -> SymbolMap:
    smap = SymbolMap(source=source)
    skipped = 0
    for row in df.itertuples(index=False):
        segment = DHAN_SEGMENT.get((row.EXCH_ID, row.SEGMENT))
        if segment is None:
            continue
        try:
            inst, aliases = _instrument_from_row(row, segment)
        except (SymbolError, ValueError) as e:
            skipped += 1
            log.debug("skip %s %s %s: %s", row.EXCH_ID, row.SEGMENT, row.SECURITY_ID, e)
            continue
        if inst is None:
            continue
        if inst.symbol in smap._instruments:
            existing = smap._instruments[inst.symbol]
            # keep the nearer contract when two rows map to the same monthly name
            if (existing.expiry or date.max) <= (inst.expiry or date.max):
                log.warning(
                    "duplicate canonical %s (ids %s, %s)",
                    inst.symbol,
                    existing.broker_id,
                    inst.broker_id,
                )
                continue
            del smap._instruments[inst.symbol]
        smap.add(inst, aliases)
    log.info("symbol map: %d instruments (%d rows skipped)", len(smap), skipped)
    return smap


def _instrument_from_row(row, segment: str) -> tuple[Instrument | None, list[str]]:  # type: ignore[no-untyped-def]
    instrument = str(row.INSTRUMENT)
    lot = _int(row.LOT_SIZE, 1)
    tick = _tick(row.TICK_SIZE)
    freeze = _int(row.SM_FREEZE_QTY, 0) or None
    exch = Exchange(row.EXCH_ID) if row.EXCH_ID != "NSE" or row.SEGMENT != "D" else Exchange.NFO

    if segment == "NSE_EQ":
        if str(row.INSTRUMENT_TYPE) not in EQUITY_TYPES or str(row.SERIES) not in EQUITY_SERIES:
            return None, []
        ticker = str(row.UNDERLYING_SYMBOL).strip().upper()
        inst = Instrument(
            symbol=make_symbol(Exchange.NSE, ticker),
            exchange=Exchange.NSE,
            kind=InstrumentKind.EQUITY,
            broker_id=str(row.SECURITY_ID),
            broker_segment=segment,
            broker_kind=instrument,
            name=str(row.DISPLAY_NAME),
            lot_size=lot,
            tick_size=tick,
            freeze_qty=freeze,
            isin=None if pd.isna(row.ISIN) else str(row.ISIN),
            series=str(row.SERIES),
            underlying=ticker,
        )
        return inst, []

    if segment == "IDX_I":
        name = index_underlying(str(row.SYMBOL_NAME))
        inst = Instrument(
            symbol=make_symbol(Exchange.NSE, name),
            exchange=Exchange.NSE,
            kind=InstrumentKind.INDEX,
            broker_id=str(row.SECURITY_ID),
            broker_segment=segment,
            broker_kind=instrument,
            name=str(row.DISPLAY_NAME),
            tick_size=0.05,
            underlying=name,
        )
        return inst, []

    # derivatives (NSE_FNO / MCX_COMM)
    expiry = _expiry(row.SM_EXPIRY_DATE)
    if expiry is None:
        raise ValueError("derivative without expiry")
    flag = None if pd.isna(row.EXPIRY_FLAG) else str(row.EXPIRY_FLAG)
    monthly = flag in MONTHLY_FLAGS
    underlying = (
        (str(row.SYMBOL_NAME) if segment == "MCX_COMM" else str(row.UNDERLYING_SYMBOL))
        .strip()
        .upper()
    )
    is_option = instrument.startswith("OPT")
    strike = float(row.STRIKE_PRICE) if is_option else None
    opt = OptionType(str(row.OPTION_TYPE)) if is_option else None
    if is_option and (strike is None or strike <= 0):
        raise ValueError("option without strike")
    primary_token = expiry_token(expiry, monthly=monthly)
    day_token = expiry_token(expiry, monthly=False)
    symbol = make_symbol(exch, underlying, primary_token, strike, opt)
    aliases = [make_symbol(exch, underlying, day_token, strike, opt)] if monthly else []
    multiplier = MCX_CONTRACT_MULTIPLIER.get(underlying, 1.0) if exch is Exchange.MCX else 1.0
    inst = Instrument(
        symbol=symbol,
        exchange=exch,
        kind=InstrumentKind.OPTION if is_option else InstrumentKind.FUTURE,
        broker_id=str(row.SECURITY_ID),
        broker_segment=segment,
        broker_kind=instrument,
        name=str(row.DISPLAY_NAME),
        lot_size=lot,
        tick_size=tick,
        multiplier=float(multiplier),
        freeze_qty=freeze,
        expiry=expiry,
        expiry_flag=flag,
        strike=strike,
        option_type=opt,
        underlying=underlying,
    )
    return inst, aliases


def load_cached_symbol_map(cache_dir: Path | str = "data/instruments") -> SymbolMap | None:
    """The newest cached instrument master, without touching the network.

    Backtests use it for lot sizes, ticks and freeze limits; when nothing is
    cached they fall back to lot 1 rather than failing offline.
    """
    files = sorted(Path(cache_dir).glob("dhan_master_*.parquet"))
    if not files:
        return None
    return build_symbol_map(pd.read_parquet(files[-1]), source=str(files[-1]))


class DhanInstrumentMaster:
    """Daily-cached download of the Dhan scrip master."""

    def __init__(
        self,
        cache_dir: Path | str = "data/instruments",
        *,
        url: str = INSTRUMENT_MASTER_URL,
        http: httpx.AsyncClient | None = None,
        keep_days: int = 7,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.url = url
        self._http = http
        self.keep_days = keep_days

    def cache_path(self, d: date) -> Path:
        return self.cache_dir / f"dhan_master_{d.isoformat()}.parquet"

    async def load(self, *, force: bool = False, today: date | None = None) -> pd.DataFrame:
        today = today or datetime.now(IST).date()
        path = self.cache_path(today)
        if path.exists() and not force:
            return pd.read_parquet(path)
        df = await self._download()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        self._prune(today)
        return df

    async def symbol_map(self, *, force: bool = False, today: date | None = None) -> SymbolMap:
        return build_symbol_map(await self.load(force=force, today=today), source=self.url)

    async def _download(self) -> pd.DataFrame:
        log.info("downloading Dhan instrument master from %s", self.url)
        client = self._http or httpx.AsyncClient(timeout=120)
        try:
            resp = await client.get(self.url)
            resp.raise_for_status()
            return read_master(BytesIO(resp.content))
        finally:
            if self._http is None:
                await client.aclose()

    def _prune(self, today: date) -> None:
        for p in self.cache_dir.glob("dhan_master_*.parquet"):
            try:
                d = date.fromisoformat(p.stem.removeprefix("dhan_master_"))
            except ValueError:
                continue
            if (today - d).days > self.keep_days:
                p.unlink(missing_ok=True)
