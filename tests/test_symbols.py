from datetime import date

import pytest

from trading.brokers.symbols import (
    SymbolError,
    expiry_token,
    is_valid_symbol,
    make_symbol,
    parse_symbol,
)
from trading.core.types import Exchange, InstrumentKind, OptionType


@pytest.mark.parametrize(
    "symbol,kind,underlying,expiry,strike,opt",
    [
        ("NSE:RELIANCE", InstrumentKind.EQUITY, "RELIANCE", None, None, None),
        ("NSE:BAJAJ-AUTO", InstrumentKind.EQUITY, "BAJAJ-AUTO", None, None, None),
        ("NSE:M&M", InstrumentKind.EQUITY, "M&M", None, None, None),
        ("NSE:NIFTY50", InstrumentKind.INDEX, "NIFTY50", None, None, None),
        ("NFO:NIFTY-OCT26", InstrumentKind.FUTURE, "NIFTY", "OCT26", None, None),
        ("NFO:BAJAJ-AUTO-OCT26", InstrumentKind.FUTURE, "BAJAJ-AUTO", "OCT26", None, None),
        (
            "NFO:NIFTY-14OCT26-25000-CE",
            InstrumentKind.OPTION,
            "NIFTY",
            "14OCT26",
            25000.0,
            OptionType.CE,
        ),
        ("MCX:GOLDM-OCT26", InstrumentKind.FUTURE, "GOLDM", "OCT26", None, None),
        (
            "MCX:GOLDM-OCT26-70000-PE",
            InstrumentKind.OPTION,
            "GOLDM",
            "OCT26",
            70000.0,
            OptionType.PE,
        ),
    ],
)
def test_parse(symbol, kind, underlying, expiry, strike, opt):
    p = parse_symbol(symbol)
    assert p.kind is kind
    assert p.underlying == underlying
    assert p.expiry == expiry
    assert p.strike == strike
    assert p.option_type is opt
    assert p.canonical == symbol  # round trip


def test_fut_suffix_tolerated_and_normalised():
    assert parse_symbol("NFO:NIFTY-OCT26-FUT").canonical == "NFO:NIFTY-OCT26"
    assert parse_symbol("nfo:nifty-oct26").canonical == "NFO:NIFTY-OCT26"


def test_expiry_helpers():
    assert parse_symbol("NFO:NIFTY-OCT26").expiry_month == (2026, 10)
    assert parse_symbol("NFO:NIFTY-OCT26").expiry_date is None
    assert parse_symbol("NFO:NIFTY-14OCT26-25000-CE").expiry_date == date(2026, 10, 14)
    assert expiry_token(date(2026, 10, 14)) == "14OCT26"
    assert expiry_token(date(2026, 10, 14), monthly=True) == "OCT26"


def test_make_symbol():
    assert make_symbol(Exchange.NFO, "nifty", "OCT26", 25000, "CE") == "NFO:NIFTY-OCT26-25000-CE"
    assert make_symbol("MCX", "GOLDM", "OCT26") == "MCX:GOLDM-OCT26"
    assert make_symbol("NFO", "NIFTY", "OCT26", 25050.5, OptionType.PE).endswith("-25050.5-PE")


@pytest.mark.parametrize(
    "bad",
    [
        "RELIANCE",  # no exchange
        "BSE:RELIANCE",  # unknown exchange
        "NFO:NIFTY",  # derivative without expiry
        "MCX:GOLDM",
        "NSE:NIFTY-OCT26",  # derivative on the cash segment
        "NFO:NIFTY-OCT26-25000",  # strike without CE/PE
        "NFO:NIFTY-OCT26-FUT-FUT",
        "NFO:-OCT26",
        "NSE:reliance!",
    ],
)
def test_bad_symbols(bad):
    with pytest.raises(SymbolError):
        parse_symbol(bad)
    assert not is_valid_symbol(bad)
