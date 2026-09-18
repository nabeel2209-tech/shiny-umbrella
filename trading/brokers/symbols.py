"""Canonical symbols and the symbol map.

Everything above the broker adapter uses canonical strings; adapters translate
to/from broker security IDs through a ``SymbolMap`` built from the broker's
instrument master (Dhan: ``dhan_instruments.py``).

Format::

    NSE:RELIANCE                  equity (ticker; may contain '-' or '&')
    NSE:NIFTY, NSE:NIFTY_100      index spot (not tradable; spaces -> '_')
    NFO:NIFTY-OCT26               monthly index future      (trailing "-FUT" accepted)
    NFO:NIFTY-OCT26-25000-CE      monthly option
    NFO:NIFTY-06OCT26-25000-CE    day-specific (weekly) option
    NFO:RELIANCE-OCT26            stock future
    MCX:GOLDM-OCT26               commodity future (qty in lots, see multipliers)
    MCX:GOLDM-OCT26-150000-CE     commodity option

A monthly expiry token (``OCT26``) is resolved to the exact expiry date from the
instrument master; a day-specific token (``06OCT26``) is the expiry date itself.
Monthly contracts are also reachable through their day-specific alias.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from trading.brokers.base import Instrument
from trading.core.types import Exchange, InstrumentKind, OptionType, now_ist

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

# MCX contract multipliers: contract value = price * lots * multiplier
# (trading unit / quotation unit from the MCX contract specifications). Verify
# against the current spec before trading a new commodity; unknown symbols get 1.
MCX_CONTRACT_MULTIPLIER: dict[str, float] = {
    "GOLD": 100,  # 1 kg, quoted per 10 g
    "GOLDM": 10,  # 100 g, quoted per 10 g
    "GOLDGUINEA": 1,  # 8 g, quoted per 8 g
    "GOLDPETAL": 1,  # 1 g, quoted per 1 g
    "GOLDTEN": 1,  # 10 g, quoted per 10 g
    "SILVER": 30,  # 30 kg, quoted per kg
    "SILVERM": 5,  # 5 kg
    "SILVERMIC": 1,  # 1 kg
    "CRUDEOIL": 100,  # 100 bbl, quoted per bbl
    "CRUDEOILM": 10,
    "NATURALGAS": 1250,  # 1250 mmBtu
    "NATGASMINI": 250,
    "COPPER": 2500,  # kg
    "ZINC": 5000,
    "ZINCMINI": 1000,
    "LEAD": 5000,
    "LEADMINI": 1000,
    "ALUMINIUM": 5000,
    "ALUMINI": 1000,
    "NICKEL": 1500,
}

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
_MONTH_NAMES = list(_MONTHS)
_EXPIRY_RE = re.compile(r"^(\d{1,2})?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})$")
_STRIKE_RE = re.compile(r"^\d+(\.\d+)?$")
_UNDERLYING_RE = re.compile(r"^[A-Z0-9&_\-]+$")


class SymbolError(ValueError):
    pass


class UnknownSymbol(SymbolError):
    pass


@dataclass(frozen=True)
class ParsedSymbol:
    exchange: Exchange
    underlying: str
    kind: InstrumentKind
    expiry: str | None = None  # raw token, e.g. "OCT26" or "06OCT26"
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
        """Exact expiry date if the token carries a day (``06OCT26``), else None."""
        if not self.expiry:
            return None
        m = _EXPIRY_RE.match(self.expiry)
        assert m
        if m.group(1) is None:
            return None
        return date(2000 + int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))

    @property
    def is_monthly_token(self) -> bool:
        return bool(self.expiry) and self.expiry_date is None


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
    """Build an expiry token from a date: ``06OCT26`` or ``OCT26`` when ``monthly``."""
    mon = _MONTH_NAMES[expiry.month - 1]
    yy = f"{expiry.year % 100:02d}"
    return f"{mon}{yy}" if monthly else f"{expiry.day:02d}{mon}{yy}"


def index_underlying(name: str) -> str:
    """Broker index name -> canonical underlying (``NIFTY 100`` -> ``NIFTY_100``)."""
    return re.sub(r"\s+", "_", name.strip().upper())


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
        # index names are either well known or contain '_' (from a space); tickers never do
        is_index = underlying in INDEX_NAMES or "_" in underlying
        kind = InstrumentKind.INDEX if is_index else InstrumentKind.EQUITY

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


def contract_multiplier(symbol: str | ParsedSymbol) -> float:
    """Contract multiplier for a canonical symbol (1 for everything but MCX)."""
    p = parse_symbol(symbol) if isinstance(symbol, str) else symbol
    if p.exchange is Exchange.MCX:
        return float(MCX_CONTRACT_MULTIPLIER.get(p.underlying, 1.0))
    return 1.0


# --------------------------------------------------------------------------- symbol map


@dataclass
class SymbolMap:
    """Canonical symbol <-> broker (segment, security id), plus contract lookups."""

    built_at: datetime = field(default_factory=now_ist)
    source: str = ""
    _instruments: dict[str, Instrument] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)
    _by_broker: dict[tuple[str, str], str] = field(default_factory=dict)
    _contracts: dict[tuple[Exchange, str, InstrumentKind], list[Instrument]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------ building
    def add(self, inst: Instrument, aliases: Iterable[str] = ()) -> None:
        if inst.symbol in self._instruments:
            raise SymbolError(f"duplicate canonical symbol {inst.symbol}")
        self._instruments[inst.symbol] = inst
        self._by_broker[(inst.broker_segment, str(inst.broker_id))] = inst.symbol
        for a in aliases:
            self._aliases.setdefault(a, inst.symbol)
        if inst.is_derivative and inst.underlying:
            key = (inst.exchange, inst.underlying, inst.kind)
            self._contracts.setdefault(key, []).append(inst)

    def __len__(self) -> int:
        return len(self._instruments)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        try:
            self.resolve(symbol)
        except SymbolError:
            return False
        return True

    # ------------------------------------------------------------------ lookups
    def resolve(self, symbol: str) -> Instrument:
        canonical = parse_symbol(symbol).canonical
        inst = self._instruments.get(canonical)
        if inst is None:
            alias = self._aliases.get(canonical)
            inst = self._instruments.get(alias) if alias else None
        if inst is None:
            raise UnknownSymbol(f"{symbol} is not in the instrument master")
        return inst

    def broker_id(self, symbol: str) -> tuple[str, str]:
        inst = self.resolve(symbol)
        return inst.broker_segment, inst.broker_id

    def canonical(self, segment: str, security_id: str | int) -> str:
        try:
            return self._by_broker[(segment, str(security_id))]
        except KeyError as e:
            raise UnknownSymbol(f"{segment}:{security_id} is not in the instrument master") from e

    def instrument_by_broker(self, segment: str, security_id: str | int) -> Instrument:
        return self._instruments[self.canonical(segment, security_id)]

    def all(self) -> list[Instrument]:
        return list(self._instruments.values())

    def contracts(
        self,
        exchange: Exchange | str,
        underlying: str,
        kind: InstrumentKind = InstrumentKind.FUTURE,
    ) -> list[Instrument]:
        key = (Exchange(exchange), underlying.upper(), kind)
        return sorted(
            self._contracts.get(key, []),
            key=lambda i: (i.expiry or date.max, i.strike or 0.0, i.option_type or ""),
        )

    def expiries(
        self,
        exchange: Exchange | str,
        underlying: str,
        kind: InstrumentKind = InstrumentKind.FUTURE,
    ) -> list[date]:
        return sorted({i.expiry for i in self.contracts(exchange, underlying, kind) if i.expiry})

    def front_month(
        self,
        exchange: Exchange | str,
        underlying: str,
        as_of: date,
        *,
        min_days_to_expiry: int = 0,
    ) -> Instrument:
        """Nearest future expiring at least ``min_days_to_expiry`` days after ``as_of``."""
        for inst in self.contracts(exchange, underlying, InstrumentKind.FUTURE):
            if inst.expiry and (inst.expiry - as_of).days >= min_days_to_expiry:
                return inst
        raise UnknownSymbol(f"no {exchange}:{underlying} future expiring after {as_of}")

    def option(
        self,
        exchange: Exchange | str,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: OptionType | str,
    ) -> Instrument:
        ot = OptionType(option_type)
        for inst in self.contracts(exchange, underlying, InstrumentKind.OPTION):
            if inst.expiry == expiry and inst.strike == strike and inst.option_type is ot:
                return inst
        raise UnknownSymbol(f"no {exchange}:{underlying} {expiry} {strike} {ot} option")

    def multipliers(self, symbols: Sequence[str]) -> dict[str, float]:
        return {s: self.resolve(s).multiplier for s in symbols}
