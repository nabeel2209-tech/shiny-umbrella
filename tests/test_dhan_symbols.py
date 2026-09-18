"""Symbol map built from a slice of the real Dhan instrument master (tests/fixtures)."""

from datetime import date
from pathlib import Path

import pytest

from trading.brokers.base import Instrument
from trading.brokers.dhan_instruments import build_symbol_map, read_master
from trading.brokers.symbols import (
    SymbolError,
    SymbolMap,
    UnknownSymbol,
    contract_multiplier,
    parse_symbol,
)
from trading.core.types import Exchange, InstrumentKind, OptionType

FIXTURE = Path(__file__).parent / "fixtures" / "dhan_master_sample.csv"


@pytest.fixture(scope="module")
def smap() -> SymbolMap:
    return build_symbol_map(read_master(FIXTURE))


def test_fixture_size_and_noise_filtered(smap):
    # 7 equities + 6 indices + 6 index futs + 8 NIFTY options + 3 stock futs + 4 stock options
    # + 33 MCX futures + 4 MCX options; BSE / currency / NSE-commodity rows are dropped
    assert len(smap) == 71
    with pytest.raises(UnknownSymbol):
        smap.canonical("MCX_COMM", "121601")  # that id is an NSE 'M' segment row
    with pytest.raises(UnknownSymbol):
        smap.canonical("BSE_EQ", "200072")


def test_equities(smap):
    r = smap.resolve("NSE:RELIANCE")
    assert r.broker_id == "2885"
    assert r.broker_segment == "NSE_EQ" and r.broker_kind == "EQUITY"
    assert r.kind is InstrumentKind.EQUITY and r.tradable
    assert r.tick_size == pytest.approx(0.10)  # master says 10.0 paise
    assert r.isin == "INE002A01018" and r.series == "EQ" and r.lot_size == 1
    assert r.freeze_qty == 67662 and r.multiplier == 1.0
    assert smap.resolve("nse:bajaj-auto").tick_size == pytest.approx(1.0)
    assert smap.resolve("NSE:M&M").broker_id == "2031"
    assert smap.resolve("NSE:RELINFRA").series == "BE"
    assert smap.resolve("NSE:SETFNIF50").kind is InstrumentKind.EQUITY  # ETF rows are kept


def test_indices(smap):
    n = smap.resolve("NSE:NIFTY")
    assert n.broker_id == "13" and n.broker_segment == "IDX_I" and n.broker_kind == "INDEX"
    assert n.kind is InstrumentKind.INDEX and not n.tradable
    assert smap.resolve("NSE:NIFTY_100").broker_id == "17"
    assert smap.resolve("NSE:INDIA_VIX").broker_id == "21"
    assert smap.resolve("NSE:BANKNIFTY").broker_id == "25"
    assert parse_symbol("NSE:NIFTY_100").kind is InstrumentKind.INDEX
    assert parse_symbol("NSE:NIFTYBEES").kind is InstrumentKind.EQUITY


def test_futures_monthly_and_aliases(smap):
    f = smap.resolve("NFO:NIFTY-OCT26")
    assert f.broker_id == "48704" and f.broker_segment == "NSE_FNO" and f.broker_kind == "FUTIDX"
    assert f.exchange is Exchange.NFO and f.kind is InstrumentKind.FUTURE
    assert f.expiry == date(2026, 10, 27) and f.expiry_flag == "M"
    assert f.lot_size == 65 and f.freeze_qty == 1756 and f.tick_size == pytest.approx(0.10)
    assert f.underlying == "NIFTY"
    assert smap.resolve("NFO:NIFTY-27OCT26") is f  # day-specific alias
    assert smap.resolve("NFO:NIFTY-OCT26-FUT") is f
    assert smap.resolve("NFO:BANKNIFTY-SEP26").broker_id == "68390"
    assert smap.resolve("NFO:BANKNIFTY-SEP26").tick_size == pytest.approx(0.20)
    assert smap.resolve("NFO:RELIANCE-OCT26").broker_id == "48987"
    assert smap.resolve("NFO:RELIANCE-OCT26").lot_size == 500


def test_weekly_options_are_day_specific(smap):
    w = smap.resolve("NFO:NIFTY-06OCT26-25000-CE")
    assert w.broker_id == "40865" and w.expiry_flag == "W" and w.expiry == date(2026, 10, 6)
    assert w.strike == 25000.0 and w.option_type is OptionType.CE and w.broker_kind == "OPTIDX"
    assert smap.resolve("NFO:NIFTY-13OCT26-25000-PE").broker_id == "44779"
    m = smap.resolve("NFO:NIFTY-OCT26-25000-CE")
    assert m.broker_id == "51440" and m.expiry == date(2026, 10, 27) and m.expiry_flag == "M"
    assert smap.resolve("NFO:NIFTY-27OCT26-25000-CE") is m
    with pytest.raises(UnknownSymbol):
        smap.resolve("NFO:NIFTY-OCT26-25050-CE")
    assert smap.resolve("NFO:RELIANCE-SEP26-700-PE").broker_id == "106278"


def test_mcx(smap):
    g = smap.resolve("MCX:GOLDM-OCT26")
    assert g.broker_id == "569003" and g.broker_segment == "MCX_COMM" and g.broker_kind == "FUTCOM"
    assert g.exchange is Exchange.MCX and g.expiry == date(2026, 10, 5)
    assert g.lot_size == 1 and g.multiplier == 10.0 and g.tick_size == pytest.approx(1.0)
    assert g.freeze_qty == 100
    assert smap.resolve("MCX:GOLD-OCT26").multiplier == 100.0
    assert smap.resolve("MCX:SILVERM-NOV26").multiplier == 5.0
    assert smap.resolve("MCX:CRUDEOIL-SEP26").broker_id == "565899"
    o = smap.resolve("MCX:GOLDM-SEP26-150000-CE")
    assert o.broker_id == "578784" and o.expiry == date(2026, 9, 25) and o.tick_size == 0.5
    assert smap.resolve("MCX:GOLDM-25SEP26-150000-CE") is o
    assert smap.resolve("MCX:GOLDM-OCT26-150000-PE").broker_id == "582038"


def test_reverse_lookup(smap):
    assert smap.canonical("NSE_EQ", 2885) == "NSE:RELIANCE"
    assert smap.canonical("NSE_FNO", "40865") == "NFO:NIFTY-06OCT26-25000-CE"
    assert smap.canonical("MCX_COMM", "569003") == "MCX:GOLDM-OCT26"
    assert smap.instrument_by_broker("IDX_I", "13").symbol == "NSE:NIFTY"
    assert smap.broker_id("NFO:NIFTY-27OCT26") == ("NSE_FNO", "48704")
    with pytest.raises(UnknownSymbol):
        smap.canonical("NSE_EQ", "999999")
    assert "NSE:RELIANCE" in smap and "NSE:NOPE" not in smap and 42 not in smap


def test_contract_helpers(smap):
    assert smap.expiries("NFO", "NIFTY") == [
        date(2026, 9, 29),
        date(2026, 10, 27),
        date(2026, 11, 23),
    ]
    assert smap.front_month("NFO", "NIFTY", date(2026, 9, 18)).symbol == "NFO:NIFTY-SEP26"
    assert (
        smap.front_month("NFO", "NIFTY", date(2026, 9, 18), min_days_to_expiry=15).symbol
        == "NFO:NIFTY-OCT26"
    )
    assert smap.front_month("MCX", "GOLDM", date(2026, 10, 6)).symbol == "MCX:GOLDM-NOV26"
    with pytest.raises(UnknownSymbol):
        smap.front_month("NFO", "NIFTY", date(2027, 1, 1))
    assert smap.option("NFO", "NIFTY", date(2026, 10, 6), 25000, "CE").broker_id == "40865"
    with pytest.raises(UnknownSymbol):
        smap.option("NFO", "NIFTY", date(2026, 10, 6), 26000, "CE")
    assert smap.multipliers(["NSE:RELIANCE", "MCX:GOLDM-OCT26"]) == {
        "NSE:RELIANCE": 1.0,
        "MCX:GOLDM-OCT26": 10.0,
    }
    assert len(smap.contracts("NFO", "NIFTY", InstrumentKind.OPTION)) == 8


def test_contract_multiplier_function():
    assert contract_multiplier("MCX:GOLDM-OCT26") == 10.0
    assert contract_multiplier("MCX:GOLD-OCT26") == 100.0
    assert contract_multiplier("MCX:UNKNOWNTHING-OCT26") == 1.0
    assert contract_multiplier("NSE:RELIANCE") == 1.0
    assert contract_multiplier("NFO:NIFTY-OCT26") == 1.0


def test_duplicate_add_rejected():
    m = SymbolMap()
    inst = Instrument(
        symbol="NSE:X", exchange=Exchange.NSE, kind=InstrumentKind.EQUITY, broker_id="1"
    )
    m.add(inst)
    with pytest.raises(SymbolError):
        m.add(inst)
