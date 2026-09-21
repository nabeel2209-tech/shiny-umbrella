"""Phase 4 acceptance tests, plus the runner's behaviour.

1. A trivial always-flat strategy produces zero trades and a flat equity curve.
2. Buy-and-hold reproduces the archive's own return: exactly, when trading is
   free; and exactly *minus the costs*, worked out by hand here, when it is not.
"""

from __future__ import annotations

import csv
import itertools
import json
from datetime import date, datetime, time

import pytest

from trading.agents.risk import RiskLimits
from trading.backtest.costs import ZERO_FEES
from trading.backtest.runner import (
    BacktestConfig,
    BacktestRunner,
    list_runs,
    load_summary,
    run_backtest,
)
from trading.backtest.sim_broker import FixedSlippage
from trading.core.types import IST, Bar, Interval, ProductType, Side
from trading.strategies.schema import StrategyConfig
from trading.training.ingest import Archive, bars_to_frame

from .conftest import make_bars

ETF = "NSE:NIFTYBEES"
WARMUP_DAY = date(2026, 9, 16)
DAY1 = date(2026, 9, 17)
DAY2 = date(2026, 9, 18)
CASH = 1_000_000.0
P0 = 250.0  # close of the first bar in the window: where buy-and-hold enters
PN = 262.5  # last close in the window: +5%
VOLUME = 5_000_000  # liquid enough that nothing is sliced
# A limit buy is funds-checked at its *limit*, as Dhan blocks margin at the limit
# price: 4,000 shares with a limit 10 bps over 250 would need Rs 10,01,000. So the
# benchmark holds 3,990 shares - 99.75% of the capital - and the rest stays cash.
QTY = 3990

GENEROUS = RiskLimits(
    max_position_value=5e6,
    max_order_value=5e6,
    max_gross_exposure=5e6,
    max_daily_loss=1e9,
    max_daily_loss_fraction=None,
)


def path(points: list[tuple[int, float]], n: int) -> list[float]:
    """Piecewise-linear prices through (index, price) anchors, on a 0.05 tick."""
    out = []
    for i in range(n):
        for (i0, p0), (i1, p1) in itertools.pairwise(points):
            if i0 <= i <= i1:
                price = p0 + (p1 - p0) * (i - i0) / max(i1 - i0, 1)
                out.append(round(round(price / 0.05) * 0.05, 2))
                break
    return out


def etf_bars(calendar) -> list[Bar]:
    n = len(calendar.session_bars("NSE", DAY1, Interval.M1))
    warm = make_bars(
        calendar, path([(0, 245.0), (n - 1, 249.0)], n), symbol=ETF, day=WARMUP_DAY, volume=VOLUME
    )
    # day one opens at 250, dips to 240 around midday, closes at 255
    one = make_bars(
        calendar,
        path([(0, P0), (180, 240.0), (n - 1, 255.0)], n),
        symbol=ETF,
        day=DAY1,
        volume=VOLUME,
        start=warm[-1].close,
    )
    two = make_bars(
        calendar,
        path([(0, 255.0), (100, 253.0), (n - 1, PN)], n),
        symbol=ETF,
        day=DAY2,
        volume=VOLUME,
        start=one[-1].close,
    )
    assert one[0].close == P0 and two[-1].close == PN
    return warm + one + two


@pytest.fixture
def archive(tmp_path, calendar):
    a = Archive(tmp_path / "archive")
    a.write(ETF, Interval.M1, bars_to_frame(etf_bars(calendar)))
    return a


def hold(**kw) -> StrategyConfig:
    base = {
        "id": "buy_and_hold",
        "symbols": [ETF],
        "interval": "1m",
        "product": "CNC",
        "expected_edge_bps": 100.0,
        "rules": {"long": {"always": True}},
        "sizing": {"mode": "fixed_qty", "qty": QTY},
        "execution": {"urgency": "AGGRESSIVE", "limit_band_bps": 10, "ttl_seconds": 300},
    }
    return StrategyConfig.model_validate({**base, **kw})


def flat() -> StrategyConfig:
    return StrategyConfig.model_validate(
        {
            "id": "always_flat",
            "symbols": [ETF],
            "interval": "1m",
            "product": "MIS",
            "expected_edge_bps": 100.0,
            # RSI lives in [0, 100]: this can never be true
            "rules": {"long": {"all": [{"feature": "rsi_14", "op": "gt", "value": 101.0}]}},
        }
    )


def config(*strategies, **kw) -> BacktestConfig:
    base = dict(
        strategies=list(strategies), start=DAY1, end=DAY2, initial_cash=CASH, limits=GENEROUS
    )
    return BacktestConfig(**{**base, **kw})


def delivery_fees(side: Side, qty: int, price: float) -> float:
    """NSE delivery charges from the rate card: no brokerage; STT 0.1% both legs;
    exchange 0.00297%; SEBI Rs 10/crore; stamp 0.015% on buys; GST 18% on
    exchange + SEBI; DP charge Rs 14.75 per sell."""
    value = qty * price
    exchange, sebi = 0.0000297 * value, 0.000001 * value
    stamp = 0.00015 * value if side is Side.BUY else 0.0
    dp = 14.75 if side is Side.SELL else 0.0
    return 0.001 * value + exchange + sebi + stamp + 0.18 * (exchange + sebi) + dp


# =========================================================================== acceptance


async def test_always_flat_strategy_makes_no_trades(calendar, archive, tmp_path):
    result = await run_backtest(config(flat()), calendar, archive=archive, output=tmp_path / "runs")
    m = result.metrics
    assert m["intents"] == m["orders"] == m["fills"] == 0
    assert result.trades == [] and m["trades"] == 0 and m["open_trades"] == 0
    assert m["net_pnl"] == 0.0 and m["total_return"] == 0.0 and m["final_equity"] == CASH
    assert m["max_drawdown"] == 0.0 and m["sharpe"] == 0.0 and m["turnover"] == 0.0
    assert m["fees"]["total"] == 0.0 and m["exposure"] == 0.0 and m["hit_rate"] is None
    # it did run: every bar of the window was processed and valued
    assert m["bars"] == 750 and len(result.equity) == 750
    assert set(result.equity["equity"]) == {CASH}
    # the features were warm from the first bar (warmup came from the day before)
    assert m["rejected"] == 0
    saved = result.path
    assert (saved / "summary.json").exists()
    with (saved / "trades.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1 and rows[0][:3] == ["strategy_id", "symbol", "product"]  # header only


async def test_buy_and_hold_reproduces_the_archive_return_when_trading_is_free(calendar, archive):
    """No fees, no slippage: the capital that is invested earns exactly what the
    archive earned, and the portfolio earns that times the fraction invested."""
    cfg = config(hold(), fee_schedule=ZERO_FEES, slippage=FixedSlippage(0.0), liquidate_at_end=True)
    result = await BacktestRunner(calendar, archive).run(cfg)
    archive_return = PN / P0 - 1.0  # +5%

    entry, exit_ = result.fills
    assert entry.side is Side.BUY and entry.qty == QTY and entry.price == P0
    assert exit_.side is Side.SELL and exit_.price == PN
    # decided on the first bar's close (09:15 bar, known at 09:16), filled at the next open
    assert entry.ts == datetime.combine(DAY1, time(9, 16), tzinfo=IST)

    invested = QTY * P0
    assert result.metrics["net_pnl"] == pytest.approx(QTY * (PN - P0))
    assert result.metrics["net_pnl"] / invested == pytest.approx(archive_return, abs=1e-12)
    assert result.metrics["total_return"] == pytest.approx(
        invested / CASH * archive_return, abs=1e-9
    )


async def test_buy_and_hold_return_is_the_archive_return_minus_costs(calendar, archive):
    """Real Indian delivery charges and 2 bps slippage each way, computed by hand."""
    cfg = config(hold(), slippage=FixedSlippage(2.0), liquidate_at_end=True)
    result = await BacktestRunner(calendar, archive).run(cfg)
    entry_price = round(P0 * 1.0002, 4)  # next open + 2 bps, inside the 10 bps limit
    exit_price = round(PN * 0.9998, 4)  # closed at the last price - 2 bps
    buy_fees = delivery_fees(Side.BUY, QTY, entry_price)
    sell_fees = delivery_fees(Side.SELL, QTY, exit_price)

    entry, exit_ = result.fills
    assert entry.price == entry_price and exit_.price == exit_price
    assert entry.fees.total == pytest.approx(buy_fees, abs=0.01)
    assert exit_.fees.total == pytest.approx(sell_fees, abs=0.01)

    expected_equity = CASH - QTY * entry_price - buy_fees + QTY * exit_price - sell_fees
    assert result.metrics["final_equity"] == pytest.approx(expected_equity, abs=0.02)

    # the same number, read as "the archive's return minus what trading cost"
    archive_return = PN / P0 - 1.0
    slippage_cost = QTY * (entry_price - P0) + QTY * (PN - exit_price)
    costs = slippage_cost + buy_fees + sell_fees
    expected_return = (QTY * P0 / CASH) * archive_return - costs / CASH
    assert result.metrics["total_return"] == pytest.approx(expected_return, abs=1e-6)
    assert result.metrics["fees"]["total"] == pytest.approx(buy_fees + sell_fees, abs=0.02)
    assert result.metrics["fees"]["other"] == 14.75  # one DP charge, on the sell
    assert costs / CASH == pytest.approx(0.0026, abs=0.0002)  # ~26 bps: mostly STT

    (trade,) = result.trades
    assert not trade.is_open and trade.direction == "LONG" and trade.qty == QTY
    assert trade.net_pnl == pytest.approx(expected_equity - CASH, abs=0.02)
    assert result.metrics["hit_rate"] == 1.0


async def test_buy_and_hold_marked_to_market_without_liquidating(calendar, archive):
    cfg = config(hold(), fee_schedule=ZERO_FEES, slippage=FixedSlippage(0.0))
    result = await BacktestRunner(calendar, archive).run(cfg)
    assert len(result.fills) == 1  # never sold
    (trade,) = result.trades
    assert trade.is_open and trade.exit_price == PN
    cash_left = CASH - QTY * P0
    assert result.metrics["final_equity"] == pytest.approx(cash_left + QTY * PN)
    assert result.metrics["open_trades"] == 1 and result.metrics["trades"] == 0
    # it rode the day-one dip from 250 to 240
    trough = cash_left + QTY * 240.0
    assert result.metrics["max_drawdown_pct"] == pytest.approx(trough / CASH - 1, abs=1e-6)
    assert result.metrics["exposure"] > 0.99  # invested from the second bar on


# =========================================================================== realism


async def test_fills_come_from_the_next_bar_never_the_signal_bar(calendar):
    """A gap between the signal bar's close and the next open must be paid."""
    stamps = calendar.session_bars("NSE", DAY1, Interval.M1)
    closes = [100.0, 100.0, 100.0, 104.0, 104.5, 105.0]
    bars, prev = [], 100.0
    for ts, close in zip(stamps, closes, strict=False):
        o = 103.0 if close == 104.0 else prev  # bar 3 gaps up at the open
        bars.append(Bar(symbol=ETF, ts=ts, interval=Interval.M1, open=o, high=max(o, close),
                        low=min(o, close), close=close, volume=VOLUME))  # fmt: skip
        prev = close
    strategy = hold(
        sizing={"mode": "fixed_qty", "qty": 10},
        execution={"urgency": "AGGRESSIVE", "limit_band_bps": 500},
    )
    result = await BacktestRunner(calendar).run(
        config(strategy, end=DAY1, fee_schedule=ZERO_FEES, slippage=FixedSlippage(0.0)), bars=bars
    )
    (entry,) = result.fills
    assert entry.ts == bars[1].ts and entry.price == bars[1].open  # not bars[0].close
    assert result.metrics["achieved_vs_mid"]["count"] == 1


async def test_intraday_positions_are_squared_off_before_the_close(calendar, archive):
    """An MIS position can never be carried overnight in a simulation."""
    strategy = hold(id="intraday_long", product="MIS")
    result = await BacktestRunner(calendar, archive).run(
        config(strategy, slippage=FixedSlippage(0.0))
    )
    square_offs = [o for o in result.orders if o.meta.get("liquidation") == "MIS square-off"]
    assert len(square_offs) == 2  # once each day
    assert {o.created_at.time() for o in square_offs} == {time(15, 20)}
    closes = result.equity[result.equity["ts"].map(lambda t: t.time() >= time(15, 20))]
    assert (closes["gross_exposure"] == 0).all()  # flat into every close
    assert result.metrics["rejections"].get("intraday_cutoff", 0) > 0  # no re-entry after 15:15
    assert all(t.product is ProductType.MIS and not t.is_open for t in result.trades)


async def test_warmup_means_the_first_window_bar_is_already_warm(calendar, archive):
    first_bar_rule = {
        "id": "warm_check",
        "symbols": [ETF],
        "interval": "1m",
        "product": "CNC",
        "expected_edge_bps": 100.0,
        "rules": {
            "long": {"all": [{"feature": "trend", "op": "gt", "value": -1.0}]}
        },  # true once warm
        # 1,000 shares: at 10 shares the Rs 14.75 DP charge alone is ~60 bps and the
        # cost gate (rightly) refuses the trade
        "sizing": {"mode": "fixed_qty", "qty": 1000},
        "execution": {"urgency": "AGGRESSIVE", "limit_band_bps": 50},
    }
    strategy = StrategyConfig.model_validate(first_bar_rule)
    warmed = await BacktestRunner(calendar, archive).run(config(strategy))
    assert warmed.fills[0].ts == datetime.combine(DAY1, time(9, 16), tzinfo=IST)
    # the same bars without the day before: nothing can fire until the features fill
    cold_bars = [b for b in archive.read_bars(ETF, Interval.M1, DAY1, DAY2)]
    cold = await BacktestRunner(calendar).run(config(strategy), bars=cold_bars)
    assert cold.fills[0].ts > datetime.combine(DAY1, time(9, 40), tzinfo=IST)


async def test_daily_bars_trade_at_the_next_sessions_open(calendar, tmp_path):
    days = calendar.trading_days("NSE", date(2026, 8, 3), date(2026, 8, 31))
    closes = [200.0 + 2 * i for i in range(len(days))]
    bars, prev = [], closes[0]
    for day, close in zip(days, closes, strict=True):
        (b,) = make_bars(calendar, [close], symbol=ETF, day=day, interval=Interval.D1, start=prev)
        bars.append(b)
        prev = close
    cfg = BacktestConfig(
        strategies=[hold(interval="1d", sizing={"mode": "fixed_qty", "qty": 100})],
        start=days[0], end=days[-1], initial_cash=CASH, limits=GENEROUS,
        fee_schedule=ZERO_FEES, slippage=FixedSlippage(0.0), liquidate_at_end=True,
    )  # fmt: skip
    assert not cfg.market_hours_enforced  # a daily bar completes after the close
    result = await BacktestRunner(calendar).run(cfg, bars=bars)
    entry, _exit = result.fills
    assert entry.ts.date() == days[1] and entry.price == closes[0]  # next session's open
    assert result.metrics["net_pnl"] == pytest.approx(100 * (closes[-1] - closes[0]))
    assert result.metrics["sharpe_basis"] == "daily"
    assert result.metrics["days"] == len(days)


async def test_intervals_missing_from_the_archive_are_built_from_minutes(calendar, archive):
    five = hold(id="five_minute", interval="5m", sizing={"mode": "fixed_qty", "qty": 1000})
    result = await BacktestRunner(calendar, archive).run(config(five))
    assert result.metrics["bars"] == 2 * 75
    assert result.fills[0].ts == datetime.combine(DAY1, time(9, 20), tzinfo=IST)


async def test_the_risk_gate_still_applies(calendar, archive):
    thin = hold(id="thin_edge", expected_edge_bps=1.0)
    result = await BacktestRunner(calendar, archive).run(config(thin))
    assert result.fills == [] and result.metrics["rejections"]["cost_threshold"] >= 1


async def test_tiny_delivery_trades_are_refused_by_the_cost_gate(calendar, archive):
    """10 shares of a Rs 250 ETF is Rs 2,500: the flat DP charge makes it ~60 bps."""
    tiny = hold(id="tiny", sizing={"mode": "fixed_qty", "qty": 10})
    result = await BacktestRunner(calendar, archive).run(config(tiny))
    assert result.fills == []
    assert set(result.metrics["rejections"]) <= {"cost_threshold", "market_hours"}


async def test_precomputed_features_match_recomputing_every_bar(calendar, archive):
    """The replay speed-up must not change a single decision."""
    import dataclasses

    strategy = StrategyConfig.model_validate(
        {
            "id": "trend",
            "symbols": [ETF],
            "interval": "1m",
            "product": "CNC",
            "expected_edge_bps": 100.0,
            "rules": {
                "long": {"all": [{"feature": "trend", "op": "gt", "value": 0.0005}]},
                "exit_long": {"all": [{"feature": "trend", "op": "lt", "value": -0.0005}]},
            },
            "sizing": {"mode": "fixed_qty", "qty": 1000},
            "execution": {"urgency": "AGGRESSIVE", "limit_band_bps": 20},
        }
    )
    fast = await BacktestRunner(calendar, archive).run(config(strategy))
    slow_cfg = dataclasses.replace(config(strategy), precompute_features=False)
    slow = await BacktestRunner(calendar, archive).run(slow_cfg)
    assert fast.metrics["fills"] > 2  # it actually trades
    assert [(f.side, f.qty, f.price, f.ts) for f in fast.fills] == [
        (f.side, f.qty, f.price, f.ts) for f in slow.fills
    ]
    assert fast.metrics["net_pnl"] == slow.metrics["net_pnl"]


# =========================================================================== bookkeeping


async def test_two_strategies_are_reported_separately(calendar, archive):
    result = await BacktestRunner(calendar, archive).run(
        config(hold(sizing={"mode": "fixed_qty", "qty": 100}), flat(), liquidate_at_end=True)
    )
    assert set(result.per_strategy) == {"buy_and_hold", "always_flat"}
    assert result.per_strategy["buy_and_hold"]["trades"] == 1
    assert result.per_strategy["always_flat"]["trades"] == 0
    assert result.per_strategy["always_flat"]["net_pnl"] == 0.0


async def test_results_are_saved_as_json_and_csv(calendar, archive, tmp_path):
    root = tmp_path / "runs"
    cfg = config(hold(), liquidate_at_end=True, name="benchmark", run_id="bench-1")
    result = await run_backtest(cfg, calendar, archive=archive, output=root)
    out = root / "bench-1"
    assert result.path == out
    for name in ("summary.json", "trades.csv", "equity.csv", "fills.csv", "orders.csv"):
        assert (out / name).exists(), name
    assert (out / "strategies" / "buy_and_hold.yaml").exists()
    summary = load_summary(out)
    assert summary["run_id"] == "bench-1" and summary["config"]["name"] == "benchmark"
    assert summary["metrics"]["fills"] == 2
    assert summary["config"]["fingerprint"] == cfg.fingerprint()
    json.dumps(summary)  # plain JSON all the way down
    trades = list(csv.DictReader((out / "trades.csv").open()))
    assert len(trades) == 1 and float(trades[0]["net_pnl"]) == pytest.approx(
        result.metrics["net_pnl"], abs=0.02
    )
    equity = list(csv.DictReader((out / "equity.csv").open()))
    assert {"ts", "equity", "cash", "positions_value", "gross_exposure", "drawdown_pct"} <= set(
        equity[0]
    )
    fills = list(csv.DictReader((out / "fills.csv").open()))
    assert [f["side"] for f in fills] == ["BUY", "SELL"] and float(fills[0]["stt"]) > 0
    orders = list(csv.DictReader((out / "orders.csv").open()))
    assert orders[-1]["note"] == "end of backtest"
    assert [r["run_id"] for r in list_runs(root)] == ["bench-1"]
    assert "net pnl" in result.summary_text()


async def test_runs_are_deterministic(calendar, archive):
    cfg = config(hold(), flat(), liquidate_at_end=True)
    first = await BacktestRunner(calendar, archive).run(cfg)
    second = await BacktestRunner(calendar, archive).run(cfg)
    assert first.metrics == second.metrics
    assert first.equity["equity"].tolist() == second.equity["equity"].tolist()
    assert cfg.fingerprint() == config(hold(), flat(), liquidate_at_end=True).fingerprint()
    assert cfg.fingerprint() != config(hold(), liquidate_at_end=True).fingerprint()


def test_config_validation():
    with pytest.raises(ValueError, match="before start"):
        BacktestConfig(strategies=[hold()], start=DAY2, end=DAY1)
    with pytest.raises(ValueError, match="duplicate"):
        BacktestConfig(strategies=[hold(), hold()], start=DAY1, end=DAY2)
    with pytest.raises(ValueError, match="at least one"):
        BacktestConfig(strategies=[], start=DAY1, end=DAY2)
