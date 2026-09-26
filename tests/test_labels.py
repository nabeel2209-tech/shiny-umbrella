"""Labels net of costs (constraint 8), triple barrier, session masking, uniqueness."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading.backtest.costs import round_trip_cost_bps
from trading.core.types import IST, ProductType
from trading.training.ingest import bars_to_frame
from trading.training.labels import (
    LabelKind,
    LabelSpec,
    cost_fraction,
    forward_return,
    label_uniqueness,
    make_labels,
    net_of_costs,
    triple_barrier,
)

from .conftest import make_bars

SYM = "NSE:RELIANCE"


def frame_from(rows):
    """rows: (open, high, low, close) per minute from 10:00."""
    t0 = datetime(2026, 9, 18, 10, 0, tzinfo=IST)
    return pd.DataFrame(
        {
            "ts": [t0 + timedelta(minutes=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 100,
            "oi": pd.array([None] * len(rows), dtype="Int64"),
        }
    )


def test_forward_return():
    f = frame_from([(100, 100, 100, c) for c in (100, 101, 103, 102)])
    r = forward_return(f, 2)
    assert r.iloc[0] == pytest.approx(0.03) and r.iloc[1] == pytest.approx(102 / 101 - 1)
    assert r.iloc[2:].isna().all()


def test_net_of_costs_keeps_only_the_move_beyond_a_round_trip():
    gross = pd.Series([0.0030, -0.0030, 0.0005, -0.0005, 0.0, np.nan])
    net = net_of_costs(gross, 0.001)
    assert net.iloc[0] == pytest.approx(0.002) and net.iloc[1] == pytest.approx(-0.002)
    assert net.iloc[2] == 0.0 and net.iloc[3] == 0.0 and net.iloc[4] == 0.0  # too small to pay
    assert np.isnan(net.iloc[5])


def test_cost_comes_from_the_cost_model():
    spec = LabelSpec(product=ProductType.MIS, notional=250_000, slippage_bps=2.0)
    expected = round_trip_cost_bps(SYM, 100, 2500.0, ProductType.MIS, slippage_bps=2.0) / 10_000
    assert cost_fraction(SYM, 2500.0, spec) == pytest.approx(expected)
    # delivery pays STT on both legs: far more expensive to label profitably
    cnc = cost_fraction(SYM, 2500.0, LabelSpec(product=ProductType.CNC, notional=250_000))
    assert cnc > 2.5 * expected  # ~27 bps against ~9
    assert cost_fraction(SYM, 2500.0, LabelSpec(cost_bps=7.0)) == pytest.approx(0.0007)
    # lot-aligned for derivatives: NIFTY at 25,000 with a lot of 65
    fut = cost_fraction("NFO:NIFTY-OCT26", 25_000.0, LabelSpec(product=ProductType.NRML), lot=65)
    assert fut == pytest.approx(
        round_trip_cost_bps("NFO:NIFTY-OCT26", 65, 25_000.0, ProductType.NRML, slippage_bps=2.0)
        / 1e4
    )


def test_intraday_labels_never_span_the_overnight_gap(calendar):
    day1 = make_bars(calendar, [100.0 + i * 0.01 for i in range(375)], day=date(2026, 9, 17))
    day2 = make_bars(calendar, [110.0] * 375, day=date(2026, 9, 18), start=110.0)
    frame = bars_to_frame(day1 + day2)
    mis = make_labels(frame, SYM, LabelSpec(horizon=10, product=ProductType.MIS))
    assert mis.gross.iloc[364] == pytest.approx(day1[374].close / day1[364].close - 1)
    assert mis.gross.iloc[365:375].isna().all()  # would need day-two prices
    assert mis.gross.iloc[375] == pytest.approx(0.0)
    cnc = make_labels(frame, SYM, LabelSpec(horizon=10, product=ProductType.CNC))
    assert cnc.gross.iloc[370] == pytest.approx(110.0 / day1[370].close - 1)  # holds overnight


def test_triple_barrier_first_touch_wins():
    spec = LabelSpec(
        kind=LabelKind.TRIPLE_BARRIER, horizon=3, profit_take=0.02, stop_loss=0.01, atr_scaled=False
    )
    up = frame_from(
        [
            (100, 100, 100, 100),
            (100, 101, 99.5, 101),
            (101, 102.5, 100.5, 102),
            (102, 102, 101, 101),
            (101, 101, 101, 101),
        ]
    )
    gross, outcome = triple_barrier(up, spec)
    assert outcome.iloc[0] == 1 and gross.iloc[0] == pytest.approx(0.02)  # 102 touched at bar 2
    down = frame_from(
        [
            (100, 100, 100, 100),
            (100, 100.5, 98.8, 99),
            (99, 103, 98, 102),
            (102, 102, 102, 102),
            (102, 102, 102, 102),
        ]
    )
    gross, outcome = triple_barrier(down, spec)
    assert outcome.iloc[0] == -1 and gross.iloc[0] == pytest.approx(-0.01)
    both = frame_from(
        [
            (100, 100, 100, 100),
            (100, 103, 98, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
        ]
    )
    _, outcome = triple_barrier(both, spec)
    assert outcome.iloc[0] == -1  # one bar hits both: assume the stop came first
    flat = frame_from([(100, 100.5, 99.5, 100.3)] * 5)
    gross, outcome = triple_barrier(flat, spec)
    assert outcome.iloc[0] == 0 and gross.iloc[0] == pytest.approx(0.0)  # timeout at the close


def test_triple_barrier_widths_scale_with_atr(calendar):
    bars = make_bars(calendar, [100 + (i % 7) * 0.3 for i in range(60)])
    labels = make_labels(
        bars_to_frame(bars), SYM, LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon=5)
    )
    valid = labels.outcome.dropna()
    assert set(valid.unique()) <= {-1.0, 0.0, 1.0} and len(valid) > 0
    assert labels.gross.iloc[:13].isna().all()  # no ATR yet: no barrier, no label


def test_uniqueness():
    # every bar starts a 3-bar label: interior labels share each bar with 3 others
    u = label_uniqueness(np.ones(10, dtype=bool), 3)
    assert u[4] == pytest.approx(1 / 3)
    assert u[0] > u[4]  # the first label shares its first bar with fewer neighbours
    # a lone label is fully unique; invalid labels weigh nothing
    lone = label_uniqueness(np.array([False, True, False, False, False, False]), 3)
    assert lone[1] == pytest.approx(1.0) and lone[0] == 0.0
    # two labels overlapping in two of their three bars
    pair = label_uniqueness(np.array([True, True, False, False, False]), 3)
    assert pair[0] == pytest.approx((1 + 0.5 + 0.5) / 3)


def test_spec_round_trip_and_validation():
    spec = LabelSpec(kind=LabelKind.TRIPLE_BARRIER, horizon=8, product=ProductType.NRML)
    assert LabelSpec.from_dict(spec.to_dict()) == spec
    assert spec.spans_sessions and not LabelSpec(product=ProductType.MIS).spans_sessions
    with pytest.raises(ValueError):
        LabelSpec(horizon=0)
    with pytest.raises(ValueError):
        LabelSpec(profit_take=-1)
