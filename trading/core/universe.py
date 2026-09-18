"""Instrument universes: Nifty 100 constituents (from NSE's public CSV) and the
index/commodity symbols the platform trades."""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import httpx

from trading.core.types import IST

log = logging.getLogger(__name__)

NIFTY100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

INDEX_SYMBOLS = [
    "NSE:NIFTY",
    "NSE:BANKNIFTY",
    "NSE:FINNIFTY",
    "NSE:MIDCPNIFTY",
    "NSE:NIFTY_100",
    "NSE:INDIA_VIX",
]
COMMODITY_UNDERLYINGS = ["GOLDM"]


def parse_nifty100(csv_text: str) -> list[str]:
    rows = list(csv.DictReader(StringIO(csv_text)))
    out = [f"NSE:{r['Symbol'].strip().upper()}" for r in rows if r.get("Symbol")]
    if len(out) < 90:
        raise ValueError(f"Nifty 100 list looks wrong: {len(out)} rows")
    return out


def load_nifty100(
    cache_dir: Path | str = "data/instruments", *, max_age_days: int = 7
) -> list[str]:
    """Canonical symbols of the current Nifty 100 constituents (cached)."""
    cache = Path(cache_dir) / "nifty100.csv"
    today = datetime.now(IST).date()
    if cache.exists():
        age = (today - date.fromtimestamp(cache.stat().st_mtime)).days
        if age <= max_age_days:
            return parse_nifty100(cache.read_text())
    try:
        r = httpx.get(NIFTY100_URL, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
        r.raise_for_status()
        symbols = parse_nifty100(r.text)
    except Exception as e:  # fall back to a stale cache rather than failing the run
        if cache.exists():
            log.warning("Nifty 100 refresh failed (%s); using cached list", e)
            return parse_nifty100(cache.read_text())
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return symbols
