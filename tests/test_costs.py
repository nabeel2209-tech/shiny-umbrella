import pytest

from trading.backtest.costs import (
    DEFAULT_FEES,
    ZERO_FEES,
    compute_fees,
    round_trip_cost_bps,
    round_trip_fees,
    segment_for,
)
from trading.core.types import ProductType, Side


def test_segments():
    assert segment_for("NSE:RELIANCE", ProductType.CNC) == "equity_delivery"
    assert segment_for("NSE:RELIANCE", ProductType.MIS) == "equity_intraday"
    assert segment_for("NFO:NIFTY-OCT26", ProductType.NRML) == "futures"
    assert segment_for("NFO:NIFTY-OCT26-25000-CE", ProductType.MIS) == "options"
    assert segment_for("MCX:GOLDM-OCT26", ProductType.NRML) == "mcx_futures"
    assert segment_for("MCX:GOLDM-OCT26-70000-CE", ProductType.NRML) == "mcx_options"
    with pytest.raises(ValueError):
        segment_for("NSE:NIFTY50", ProductType.CNC)


def test_equity_delivery_buy_hand_computed():
    # 10 x 2500 = 25,000 delivery buy
    f = compute_fees("NSE:RELIANCE", Side.BUY, 10, 2500.0, ProductType.CNC)
    assert f.brokerage == 0.0
    assert f.stt == pytest.approx(25_000 * 0.001)  # 25.00
    assert f.exchange == pytest.approx(25_000 * 0.0000297, abs=1e-4)  # 0.7425
    assert f.sebi == pytest.approx(25_000 * 0.000001, abs=1e-4)  # 0.025
    assert f.stamp == pytest.approx(25_000 * 0.00015, abs=1e-4)  # 3.75
    assert f.gst == pytest.approx(0.18 * (0.7425 + 0.025), abs=1e-3)
    assert f.other == 0.0
    assert f.total == pytest.approx(25 + 0.7425 + 0.025 + 3.75 + 0.13815, abs=1e-3)


def test_equity_delivery_sell_has_dp_charge_and_no_stamp():
    f = compute_fees("NSE:RELIANCE", Side.SELL, 10, 2600.0, ProductType.CNC)
    assert f.stamp == 0.0
    assert f.stt == pytest.approx(26.0)
    assert f.other == pytest.approx(14.75)


def test_intraday_brokerage_capped_at_20():
    # 100 x 2500 = 250,000 -> 0.03% = 75 > cap 20
    f = compute_fees("NSE:RELIANCE", Side.BUY, 100, 2500.0, ProductType.MIS)
    assert f.brokerage == 20.0
    assert f.stt == 0.0  # intraday STT on sell side only
    assert f.stamp == pytest.approx(250_000 * 0.00003, abs=1e-4)
    assert f.gst == pytest.approx(0.18 * (20 + 7.425 + 0.25), abs=1e-3)
    # small trade: 0.03% below the cap
    small = compute_fees("NSE:RELIANCE", Side.BUY, 2, 2500.0, ProductType.MIS)
    assert small.brokerage == pytest.approx(5000 * 0.0003)


def test_futures_and_options_side_rules():
    fb = compute_fees("NFO:NIFTY-OCT26", Side.BUY, 75, 25_000.0, ProductType.NRML)
    fs = compute_fees("NFO:NIFTY-OCT26", Side.SELL, 75, 25_000.0, ProductType.NRML)
    assert fb.stt == 0.0 and fs.stt == pytest.approx(75 * 25_000 * 0.0002)
    assert fb.stamp > 0 and fs.stamp == 0.0
    ob = compute_fees("NFO:NIFTY-OCT26-25000-CE", Side.BUY, 75, 100.0, ProductType.NRML)
    os_ = compute_fees("NFO:NIFTY-OCT26-25000-CE", Side.SELL, 75, 100.0, ProductType.NRML)
    assert ob.stt == 0.0 and os_.stt == pytest.approx(7500 * 0.001)
    assert ob.exchange == pytest.approx(7500 * 0.0003503, abs=1e-4)


def test_round_trip_and_bps():
    rt = round_trip_fees("NSE:RELIANCE", 10, 2500.0, ProductType.CNC)
    buy = compute_fees("NSE:RELIANCE", Side.BUY, 10, 2500.0, ProductType.CNC)
    sell = compute_fees("NSE:RELIANCE", Side.SELL, 10, 2500.0, ProductType.CNC)
    assert rt.total == pytest.approx(buy.total + sell.total)
    bps = round_trip_cost_bps("NSE:RELIANCE", 10, 2500.0, ProductType.CNC)
    assert bps == pytest.approx(rt.total / 25_000 * 10_000)
    assert round_trip_cost_bps(
        "NSE:RELIANCE", 10, 2500.0, ProductType.CNC, slippage_bps=3
    ) == pytest.approx(bps + 6)
    # delivery is more expensive than intraday, which is more expensive than futures
    assert bps > round_trip_cost_bps("NSE:RELIANCE", 100, 2500.0, ProductType.MIS)
    assert round_trip_cost_bps("NSE:RELIANCE", 100, 2500.0, ProductType.MIS) > round_trip_cost_bps(
        "NFO:NIFTY-OCT26", 75, 25_000.0, ProductType.NRML
    )


def test_zero_schedule():
    f = compute_fees("NSE:RELIANCE", Side.BUY, 10, 2500.0, ProductType.CNC, schedule=ZERO_FEES)
    assert f.total == 0.0
    assert DEFAULT_FEES is not ZERO_FEES
