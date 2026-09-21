"""Round-trip trade matching and equity statistics, checked by hand."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from trading.backtest.metrics import (
    MIN_DAILY_OBSERVATIONS,
    drawdown,
    equity_stats,
    fee_breakdown,
    match_trades,
    trade_stats,
    turnover,
)
from trading.core.types import IST, FeeBreakdown, Fill, ProductType, Side

SYM = "NSE:RELIANCE"
T0 = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


def fill(
    side, qty, price, minute=0, fee=0.0, oid="o", symbol=SYM, mult=1.0, product=ProductType.MIS
):
    return Fill(
        order_id=oid, symbol=symbol, side=side, qty=qty, price=price,
        ts=T0 + timedelta(minutes=minute), product=product,
        fees=FeeBreakdown(brokerage=fee), multiplier=mult,
    )  # fmt: skip


# --------------------------------------------------------------------------- FIFO matching


def test_fifo_closes_the_oldest_lot_first():
    trades = match_trades(
        [
            fill(Side.BUY, 10, 100.0, 0, oid="a"),
            fill(Side.BUY, 10, 110.0, 1, oid="b"),
            fill(Side.SELL, 15, 120.0, 2, oid="c"),
        ],
        marks={SYM: 115.0},
    )
    closed = [t for t in trades if not t.is_open]
    assert [(t.qty, t.entry_price, t.gross_pnl) for t in closed] == [
        (10, 100.0, 200.0),
        (5, 110.0, 50.0),
    ]
    assert [t.entry_order_id for t in closed] == ["a", "b"] and closed[0].exit_order_id == "c"
    (still_open,) = [t for t in trades if t.is_open]
    assert still_open.qty == 5 and still_open.gross_pnl == pytest.approx(25.0)  # marked at 115


def test_shorts_and_a_reversal_in_one_fill():
    trades = match_trades(
        [
            fill(Side.SELL, 10, 100.0, 0),
            fill(Side.BUY, 15, 95.0, 1),  # cover 10 (+50), go long 5
            fill(Side.SELL, 5, 97.0, 2),  # close the long (+10)
        ]
    )
    assert [(t.direction, t.qty, t.gross_pnl) for t in trades] == [
        ("SHORT", 10, 50.0),
        ("LONG", 5, 10.0),
    ]
    assert not any(t.is_open for t in trades)


def test_fees_are_split_pro_rata_between_trades():
    trades = match_trades(
        [
            fill(Side.BUY, 10, 100.0, 0, fee=10.0),
            fill(Side.SELL, 4, 101.0, 1, fee=2.0),
            fill(Side.SELL, 6, 102.0, 2, fee=3.0),
        ]
    )
    # entry fee 1.00/share; exits 0.50/share each
    assert [t.fees for t in trades] == [pytest.approx(4 * 1.0 + 2.0), pytest.approx(6 * 1.0 + 3.0)]
    assert trades[0].net_pnl == pytest.approx(4.0 - 6.0)
    assert sum(t.fees for t in trades) == pytest.approx(15.0)


def test_multiplier_and_attribution():
    (trade,) = match_trades(
        [
            fill(
                Side.BUY,
                1,
                150_000.0,
                0,
                oid="x",
                symbol="MCX:GOLDM-OCT26",
                mult=10,
                product=ProductType.NRML,
            ),
            fill(
                Side.SELL,
                1,
                150_100.0,
                5,
                oid="y",
                symbol="MCX:GOLDM-OCT26",
                mult=10,
                product=ProductType.NRML,
            ),
        ],
        strategy_of={"x": "gold_rev"},
    )
    assert trade.gross_pnl == pytest.approx(1_000.0) and trade.strategy_id == "gold_rev"
    assert trade.notional == 1_500_000.0 and trade.holding_seconds == 300
    assert trade.to_row()["product"] == "NRML"


def test_products_are_matched_separately():
    trades = match_trades(
        [
            fill(Side.BUY, 10, 100.0, 0, product=ProductType.CNC),
            fill(Side.SELL, 10, 101.0, 1, product=ProductType.MIS),
        ]
    )
    assert len(trades) == 2 and all(t.is_open for t in trades)


# --------------------------------------------------------------------------- trade stats


def test_trade_stats():
    trades = match_trades(
        [
            fill(Side.BUY, 1, 100.0, 0), fill(Side.SELL, 1, 110.0, 1),  # +10
            fill(Side.BUY, 1, 100.0, 2), fill(Side.SELL, 1, 95.0, 3),   # -5
            fill(Side.BUY, 1, 100.0, 4), fill(Side.SELL, 1, 120.0, 5),  # +20
        ]
    )  # fmt: skip
    stats = trade_stats(trades)
    assert stats["trades"] == 3 and stats["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert stats["avg_win"] == 15.0 and stats["avg_loss"] == -5.0
    assert stats["profit_factor"] == pytest.approx(6.0)
    assert stats["best_trade"] == 20.0 and stats["worst_trade"] == -5.0
    assert stats["avg_holding_seconds"] == 60.0


def test_trade_stats_with_nothing_closed():
    stats = trade_stats(match_trades([fill(Side.BUY, 1, 100.0)]))
    assert stats["trades"] == 0 and stats["open_trades"] == 1
    assert stats["hit_rate"] is None and stats["profit_factor"] is None


# --------------------------------------------------------------------------- equity


def curve(values, *, days: bool = True):
    step = timedelta(days=1) if days else timedelta(minutes=1)
    idx = pd.DatetimeIndex([T0 + step * i for i in range(len(values))])
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_counts_the_starting_capital_as_the_first_peak():
    dd = drawdown(curve([99_000, 98_000, 101_000]), 100_000)
    assert dd.max_drawdown == -2_000 and dd.max_drawdown_pct == pytest.approx(-0.02)
    assert dd.trough_ts == T0 + timedelta(days=1)
    assert dd.recovered_ts == T0 + timedelta(days=2)


def test_drawdown_on_a_known_path():
    dd = drawdown(curve([110, 120, 90, 100, 130, 125]), 100)
    assert dd.max_drawdown == -30 and dd.max_drawdown_pct == pytest.approx(-0.25)
    assert dd.peak_ts == T0 + timedelta(days=1) and dd.trough_ts == T0 + timedelta(days=2)
    assert dd.longest_underwater_seconds == 86_400  # below 120 from day 2 to day 3
    assert drawdown(curve([]), 100).max_drawdown == 0.0


def test_sharpe_uses_daily_returns_only_with_enough_days():
    rng = [100_000 * (1 + 0.001 * ((i % 3) - 0.5)) for i in range(MIN_DAILY_OBSERVATIONS)]
    stats = equity_stats(curve(rng), 100_000)
    assert (
        stats["sharpe_basis"] == "daily" and stats["sharpe_observations"] == MIN_DAILY_OBSERVATIONS
    )
    short = equity_stats(curve([100_100, 100_050, 100_200], days=False), 100_000)
    assert short["sharpe_basis"] == "bar" and short["days"] == 1


def test_known_sharpe():
    values = [100_000.0]
    for r in [0.01, -0.005, 0.02, 0.0, -0.01, 0.015, 0.005, -0.002, 0.01, 0.003]:
        values.append(values[-1] * (1 + r))
    stats = equity_stats(curve(values[1:]), values[0])
    rets = pd.Series(values).pct_change().dropna()
    expected = rets.mean() / rets.std(ddof=1) * (252**0.5)
    assert stats["sharpe"] == pytest.approx(expected, rel=1e-3)
    assert stats["sortino"] is not None and stats["sortino"] > stats["sharpe"]
    assert stats["total_return"] == pytest.approx(values[-1] / values[0] - 1, abs=1e-6)
    lower = equity_stats(curve(values[1:]), values[0], risk_free_rate=0.5)["sharpe"]
    assert lower < stats["sharpe"]


def test_undefined_ratios_are_none_not_zero():
    up_only = equity_stats(curve([100_000 + 10 * i for i in range(1, 12)]), 100_000)
    assert up_only["sortino"] is None  # no losing day: undefined
    flat = equity_stats(curve([100_000.0] * 12), 100_000)
    assert flat["sharpe"] == 0.0 and flat["max_drawdown"] == 0.0 and flat["net_pnl"] == 0.0
    empty = equity_stats(pd.Series(dtype=float), 100_000)
    assert empty["final_equity"] == 100_000 and empty["sharpe_basis"] == "none"


def test_annualised_return_needs_a_month():
    assert equity_stats(curve([101_000] * 5), 100_000)["annualised_return"] is None
    long_run = equity_stats(curve([100_000 * 1.0005 ** (i + 1) for i in range(252)]), 100_000)
    assert long_run["annualised_return"] == pytest.approx(1.0005**252 - 1, rel=1e-3)


def test_fee_breakdown_and_turnover():
    common = {"symbol": SYM, "qty": 10, "ts": T0, "product": ProductType.CNC}
    fills = [
        Fill(order_id="a", side=Side.BUY, price=100.0, **common,
             fees=FeeBreakdown(stt=1.0, stamp=0.15, gst=0.01)),
        Fill(order_id="b", side=Side.SELL, price=110.0, **common,
             fees=FeeBreakdown(stt=1.1, other=14.75)),
    ]  # fmt: skip
    fees = fee_breakdown(fills)
    assert fees["stt"] == 2.1 and fees["other"] == 14.75 and fees["total"] == pytest.approx(17.01)
    t = turnover(fills, 10_000.0)
    assert t["traded_value"] == 2_100.0 and t["turnover"] == 0.21
