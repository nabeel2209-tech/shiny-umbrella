"""Canonical symbols.

Everything above the broker adapter uses canonical strings; adapters translate
to/from broker security IDs (the Dhan map is built in Phase 2).

Format::

    NSE:RELIANCE                  equity
    NSE:NIFTY50                   index spot (not tradable)
    NFO:NIFTY-OCT26               monthly index future      (trailing "-FUT" accepted)
    NFO:NIFTY-OCT26-25000-CE      monthly option
    NFO:NIFTY-14OCT26-25000-CE    day-specific (weekly) option
    NFO:RELIANCE-OCT26            stock future
    MCX:GOLDM-OCT26               commodity future
    MCX:GOLDM-OCT26-70000-CE      commodity option

A monthly expiry token (``OCT26``) is resolved to the exact expiry date from the
instrument master; a day-specific token (``14OCT26``) is the expiry date itself.
Underlyings may contain hyphens (``NSE:BAJAJ-AUTO``); tokens are parsed from the right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from trading.core.types import Exchange, InstrumentKind, OptionType

INDEX_NAMES = {
    "NIFTY",
    "NIFTY50",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "INDIAVIX",
    "SENSEX",
    "BANKEX",
}

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
_EXPIRY_RE = re.compile(r"^(\d{1,2})?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})$")
_STRIKE_RE = re.compile(r"^\d+(\.\d+)?$")
_UNDERLYING_RE = re.compile(r"^[A-Z0-9&\-]+$")


class SymbolError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedSymbol:
    exchange: Exchange
    underlying: str
    kind: InstrumentKind
    expiry: str | None = None  # raw token, e.g. "OCT26" or "14OCT26"
    strike: float | None = None
    option_type: OptionType | None = None

    @property
    def canonical(self) -> str:
        return make_symbol(
            self.exchange, self.underlying, self.expiry, self.strike, self.option_type
        )

    @property
    def is_derivative(self) -> bool:
        return self.kind in {InstrumentKind.FUTURE, InstrumentKind.OPTION}

    @property
    def expiry_month(self) -> tuple[int, int] | None:
        """(year, month) of the expiry token, or None for cash instruments."""
        if not self.expiry:
            return None
        m = _EXPIRY_RE.match(self.expiry)
        assert m  # validated on construction
        return 2000 + int(m.group(3)), _MONTHS[m.group(2)]

    @property
    def expiry_date(self) -> date | None:
        """Exact expiry date if the token carries a day (``14OCT26``), else None."""
        if not self.expiry:
            return None
        m = _EXPIRY_RE.match(self.expiry)
        assert m
        if m.group(1) is None:
            return None
        return date(2000 + int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))


def _format_strike(strike: float) -> str:
    return str(int(strike)) if float(strike).is_integer() else f"{strike:g}"


def make_symbol(
    exchange: Exchange | str,
    underlying: str,
    expiry: str | None = None,
    strike: float | None = None,
    option_type: OptionType | str | None = None,
) -> str:
    ex = Exchange(exchange)
    parts = [underlying.upper()]
    if expiry:
        parts.append(expiry.upper())
    if strike is not None:
        if option_type is None:
            raise SymbolError("strike given without option type")
        parts.append(_format_strike(strike))
        parts.append(OptionType(option_type).value)
    return f"{ex.value}:{'-'.join(parts)}"


def expiry_token(expiry: date, *, monthly: bool = False) -> str:
    """Build an expiry token from a date: ``14OCT26`` or ``OCT26`` when ``monthly``."""
    mon = list(_MONTHS)[expiry.month - 1]
    yy = f"{expiry.year % 100:02d}"
    return f"{mon}{yy}" if monthly else f"{expiry.day:02d}{mon}{yy}"


def parse_symbol(symbol: str) -> ParsedSymbol:
    if ":" not in symbol:
        raise SymbolError(f"symbol {symbol!r} must look like EXCHANGE:NAME")
    ex_raw, rest = symbol.split(":", 1)
    try:
        exchange = Exchange(ex_raw.upper())
    except ValueError as e:
        raise SymbolError(f"unknown exchange {ex_raw!r} in {symbol!r}") from e
    tokens = rest.upper().split("-")
    if tokens and tokens[-1] == "FUT":  # tolerate an explicit FUT suffix
        tokens = tokens[:-1]
        if len(tokens) < 2 or not _EXPIRY_RE.match(tokens[-1]):
            raise SymbolError(f"FUT suffix without expiry in {symbol!r}")

    expiry = strike = option_type = None
    if len(tokens) >= 4 and tokens[-1] in {"CE", "PE"}:
        if not _STRIKE_RE.match(tokens[-2]) or not _EXPIRY_RE.match(tokens[-3]):
            raise SymbolError(f"bad option symbol {symbol!r}")
        option_type = OptionType(tokens[-1])
        strike = float(tokens[-2])
        expiry = tokens[-3]
        underlying = "-".join(tokens[:-3])
        kind = InstrumentKind.OPTION
    elif len(tokens) >= 2 and _EXPIRY_RE.match(tokens[-1]):
        expiry = tokens[-1]
        underlying = "-".join(tokens[:-1])
        kind = InstrumentKind.FUTURE
    else:
        underlying = "-".join(tokens)
        if exchange is not Exchange.NSE:
            raise SymbolError(f"{exchange} symbols need an expiry: {symbol!r}")
        kind = InstrumentKind.INDEX if underlying in INDEX_NAMES else InstrumentKind.EQUITY

    if not underlying or not _UNDERLYING_RE.match(underlying):
        raise SymbolError(f"bad underlying in {symbol!r}")
    if exchange is Exchange.NSE and kind in {InstrumentKind.FUTURE, InstrumentKind.OPTION}:
        raise SymbolError(f"derivatives belong on NFO, not NSE: {symbol!r}")
    return ParsedSymbol(exchange, underlying, kind, expiry, strike, option_type)


def is_valid_symbol(symbol: str) -> bool:
    try:
        parse_symbol(symbol)
    except SymbolError:
        return False
    return True
