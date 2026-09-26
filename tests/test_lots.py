"""Lot-size policy: equities default to 1, derivatives never do; the instrument
master is fetched on startup when missing."""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from trading.brokers.dhan_instruments import (
    INSTRUMENT_MASTER_URL,
    DhanInstrumentMaster,
    build_symbol_map,
    ensure_symbol_map,
    read_master,
)
from trading.brokers.lots import LotSizes, MissingLotSize

FIXTURE = Path(__file__).parent / "fixtures" / "dhan_master_sample.csv"
TODAY = date(2026, 9, 26)


@pytest.fixture(scope="module")
def smap():
    return build_symbol_map(read_master(FIXTURE))


# --------------------------------------------------------------------------- the policy


def test_equities_default_to_one_share():
    lots = LotSizes()
    assert lots.get("NSE:RELIANCE") == 1
    assert lots("NSE:BAJAJ-AUTO") == 1  # callable, for the simulators
    assert lots.known("NSE:ANYTHING")


@pytest.mark.parametrize(
    "symbol", ["NFO:NIFTY-OCT26", "NFO:NIFTY-06OCT26-25000-CE", "MCX:GOLDM-OCT26"]
)
def test_derivatives_never_default(symbol):
    lots = LotSizes()
    with pytest.raises(MissingLotSize, match="Refusing to guess"):
        lots.get(symbol)
    assert not lots.known(symbol)


def test_known_values_are_used_and_names_are_canonical():
    lots = LotSizes({"nfo:nifty-oct26-fut": 65, "MCX:GOLDM-OCT26": 1})
    assert lots.get("NFO:NIFTY-OCT26") == 65
    assert lots.get("MCX:GOLDM-OCT26") == 1
    assert lots.as_dict() == {"NFO:NIFTY-OCT26": 65, "MCX:GOLDM-OCT26": 1}
    with pytest.raises(ValueError):
        LotSizes({"NFO:NIFTY-OCT26": 0})


def test_require_names_every_missing_derivative_at_once():
    lots = LotSizes({"NFO:NIFTY-OCT26": 65})
    lots.require(["NSE:RELIANCE", "NFO:NIFTY-OCT26"])  # fine
    with pytest.raises(MissingLotSize) as err:
        lots.require(["MCX:GOLDM-OCT26", "NFO:BANKNIFTY-OCT26", "NSE:TCS"])
    assert err.value.symbols == ["MCX:GOLDM-OCT26", "NFO:BANKNIFTY-OCT26"]


def test_from_symbol_map(smap):
    lots = LotSizes.from_symbol_map(
        smap, ["NFO:NIFTY-OCT26", "MCX:GOLDM-OCT26", "NSE:RELIANCE", "NSE:NOTLISTED"]
    )
    assert lots.get("NFO:NIFTY-OCT26") == 65 and lots.get("MCX:GOLDM-OCT26") == 1
    assert lots.get("NSE:NOTLISTED") == 1  # an equity missing from the master is still 1
    with pytest.raises(MissingLotSize, match="not in the instrument master"):
        LotSizes.from_symbol_map(smap, ["NFO:NIFTY-OCT99"])


def test_no_master_at_all():
    assert LotSizes.from_symbol_map(None, ["NSE:RELIANCE"]).get("NSE:RELIANCE") == 1
    with pytest.raises(MissingLotSize, match="could not be downloaded"):
        LotSizes.from_symbol_map(None, ["NSE:RELIANCE", "MCX:GOLDM-OCT26"])


# --------------------------------------------------------------------------- download on startup


def csv_bytes() -> bytes:
    return FIXTURE.read_bytes()


async def test_downloads_the_master_when_missing(tmp_path):
    with respx.mock(assert_all_called=True) as router:
        route = router.get(INSTRUMENT_MASTER_URL).mock(
            return_value=httpx.Response(200, content=csv_bytes())
        )
        smap = await ensure_symbol_map(tmp_path, today=TODAY)
    assert route.call_count == 1
    assert smap is not None and smap.resolve("NFO:NIFTY-OCT26").lot_size == 65
    assert (tmp_path / f"dhan_master_{TODAY.isoformat()}.parquet").exists()


async def test_uses_todays_cache_without_touching_the_network(tmp_path):
    DhanInstrumentMaster(tmp_path)
    read_master(BytesIO(csv_bytes())).to_parquet(
        tmp_path / f"dhan_master_{TODAY.isoformat()}.parquet"
    )
    with respx.mock(assert_all_called=False) as router:
        route = router.get(INSTRUMENT_MASTER_URL)
        smap = await ensure_symbol_map(tmp_path, today=TODAY)
    assert route.call_count == 0 and smap is not None


async def test_falls_back_to_a_stale_copy_when_offline(tmp_path):
    yesterday = TODAY - timedelta(days=1)
    read_master(BytesIO(csv_bytes())).to_parquet(
        tmp_path / f"dhan_master_{yesterday.isoformat()}.parquet"
    )
    with respx.mock() as router:
        router.get(INSTRUMENT_MASTER_URL).mock(side_effect=httpx.ConnectError("offline"))
        smap = await ensure_symbol_map(tmp_path, today=TODAY)
    assert smap is not None and "2026-09-25" in smap.source


async def test_nothing_cached_and_offline_means_none(tmp_path):
    with respx.mock() as router:
        router.get(INSTRUMENT_MASTER_URL).mock(side_effect=httpx.ConnectError("offline"))
        assert await ensure_symbol_map(tmp_path, today=TODAY) is None
    assert await ensure_symbol_map(tmp_path, today=TODAY, download=False) is None


# --------------------------------------------------------------------------- fail fast at startup


def test_engine_refuses_to_start_a_derivative_strategy_without_lots(calendar):
    from trading.agents.engine import EngineConfig, TradingEngine
    from trading.brokers.paper import PaperBroker
    from trading.core.bus import InMemoryBus
    from trading.strategies.schema import StrategyConfig

    gold = StrategyConfig.model_validate(
        {
            "id": "gold",
            "symbols": ["MCX:GOLDM-OCT26"],
            "product": "NRML",
            "expected_edge_bps": 20,
            "rules": {"long": {"always": True}},
        }
    )
    with pytest.raises(MissingLotSize, match="MCX:GOLDM-OCT26"):
        TradingEngine(InMemoryBus(), PaperBroker(), calendar, EngineConfig(strategies=[gold]))


async def test_backtest_refuses_before_loading_any_data(calendar):
    from trading.backtest.runner import BacktestConfig, BacktestRunner
    from trading.strategies.schema import StrategyConfig

    nifty = StrategyConfig.model_validate(
        {
            "id": "nifty",
            "symbols": ["NFO:NIFTY-OCT26"],
            "product": "NRML",
            "expected_edge_bps": 20,
            "rules": {"long": {"always": True}},
        }
    )
    cfg = BacktestConfig(strategies=[nifty], start=date(2026, 9, 17), end=date(2026, 9, 18))
    with pytest.raises(MissingLotSize):
        await BacktestRunner(calendar).run(cfg, bars=[])


async def test_backtest_cli_exits_with_the_error_when_offline(tmp_path, monkeypatch, capsys):
    import scripts.run_backtest as cli

    async def offline(*_a, **_k):
        return None

    monkeypatch.setattr(cli, "ensure_symbol_map", offline)
    code = await cli.main(
        ["--strategies", "trading/strategies/examples", "--only", "goldm_meanrev",
         "--start", "2026-09-17", "--end", "2026-09-18", "--output", str(tmp_path)]
    )  # fmt: skip
    assert code == 2
    err = capsys.readouterr().err
    assert "lot size unknown for MCX:GOLDM-OCT26" in err and "Refusing to guess" in err


def test_the_fixture_master_is_the_real_format():
    df = pd.read_csv(FIXTURE)
    assert {"LOT_SIZE", "SM_EXPIRY_DATE", "SECURITY_ID"} <= set(df.columns)
