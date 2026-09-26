"""Lot sizes, with one rule for what is not known.

NSE cash equities trade in single shares, so an equity that is missing from the
instrument master defaults to a lot of 1. A derivative's lot is set by the
exchange and revised several times a year (NIFTY went 50 -> 25 -> 75 -> 65), so a
missing one is an **error**: sizing a NIFTY future as if one contract were one
unit would be wrong by a factor of 65, silently.

Every agent, the simulators and the backtest runner take a :class:`LotSizes`; the
engine and the runner call :meth:`LotSizes.require` at startup so a missing
derivative lot stops the run before anything trades.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from trading.brokers.base import Instrument
from trading.brokers.symbols import SymbolError, SymbolMap, UnknownSymbol, parse_symbol

EQUITY_LOT = 1

MISSING_HELP = (
    "derivative lot sizes come from Dhan's instrument master "
    "(https://images.dhan.co/api-data/api-scrip-master-detailed.csv), cached daily in "
    "data/instruments/. It is not cached and could not be downloaded; re-run with an "
    "internet connection and it is fetched automatically (no login needed). "
    "Refusing to guess a lot size of 1."
)


class MissingLotSize(SymbolError):
    """A derivative whose lot size is not known."""

    def __init__(self, symbols: Iterable[str], detail: str = MISSING_HELP) -> None:
        self.symbols = sorted(set(symbols))
        super().__init__(f"lot size unknown for {', '.join(self.symbols)}: {detail}")


class LotSizes:
    """Symbol -> lot size, defaulting to 1 for NSE cash and failing for derivatives."""

    def __init__(self, known: Mapping[str, int] | None = None, *, source: str = "") -> None:
        self._known = {parse_symbol(s).canonical: int(v) for s, v in (known or {}).items()}
        for symbol, lot in self._known.items():
            if lot < 1:
                raise ValueError(f"lot size for {symbol} must be >= 1, got {lot}")
        self.source = source

    # ------------------------------------------------------------------ construction
    @classmethod
    def of(cls, value: LotSizes | Mapping[str, int] | None) -> LotSizes:
        return value if isinstance(value, LotSizes) else cls(value)

    @classmethod
    def from_instruments(cls, instruments: Mapping[str, Instrument]) -> LotSizes:
        return cls({s: i.lot_size for s, i in instruments.items()}, source="instruments")

    @classmethod
    def from_symbol_map(cls, symbol_map: SymbolMap | None, symbols: Iterable[str]) -> LotSizes:
        """Resolve ``symbols`` against the instrument master, applying the policy.

        Raises :class:`MissingLotSize` naming every derivative it cannot size.
        """
        known: dict[str, int] = {}
        missing: list[str] = []
        for symbol in symbols:
            try:
                if symbol_map is None:
                    raise UnknownSymbol(symbol)
                known[symbol] = symbol_map.resolve(symbol).lot_size
            except UnknownSymbol:
                if parse_symbol(symbol).is_derivative:
                    missing.append(symbol)
        if missing:
            detail = (
                MISSING_HELP
                if symbol_map is None
                else (
                    "not in the instrument master - the contract may have expired or the "
                    "symbol may be misspelt"
                )
            )
            raise MissingLotSize(missing, detail)
        return cls(known, source=symbol_map.source if symbol_map else "equity default")

    # ------------------------------------------------------------------ lookups
    def get(self, symbol: str) -> int:
        canonical = parse_symbol(symbol).canonical
        lot = self._known.get(canonical)
        if lot is not None:
            return lot
        if parse_symbol(canonical).is_derivative:
            raise MissingLotSize([canonical])
        return EQUITY_LOT

    __call__ = get

    def known(self, symbol: str) -> bool:
        try:
            self.get(symbol)
        except MissingLotSize:
            return False
        return True

    def require(self, symbols: Iterable[str]) -> None:
        """Fail fast, naming every symbol that cannot be sized."""
        missing = [s for s in symbols if not self.known(s)]
        if missing:
            raise MissingLotSize(missing)

    def as_dict(self) -> dict[str, int]:
        return dict(self._known)

    def __repr__(self) -> str:
        return f"LotSizes({len(self._known)} known, source={self.source!r})"
