"""Features: causality, batch/incremental parity (constraint 1), warmup."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading.core.types import Interval
from trading.features.features import (
    DEFAULT_SPEC,
    FeatureSpec,
    bars_to_feature_frame,
    compute_features,
    feature_names,
    is_warm,
    latest_features,
)

from .conftest import make_bars, make_synthetic_day


@pytest.fixture
def two_days(calendar):
    return make_synthetic_day(calendar, day=date(2026, 9, 17)) + make_synthetic_day(
        calendar, day=date(2026, 9, 18)
    )


def test_feature_names_and_spec():
    names = feature_names()
    assert names[:3] == ["ret_1", "ret_5", "ret_15"]
    assert {"trend", "rsi_14", "atr_pct_14", "vwap_dist", "tod", "overnight_gap"} <= set(names)
    # daily bars have no session-anchored features
    daily = feature_names(interval=Interval.D1)
    assert not {"vwap_dist", "tod", "overnight_gap"} & set(daily)
    assert DEFAULT_SPEC.warmup_bars == 31  # longest window (30) + 1 for the return
    assert DEFAULT_SPEC.buffer_bars(375) == 376
    assert FeatureSpec(sma_slow=100).warmup_bars == 101


def test_columns_match_names(two_days):
    frame = compute_features(bars_to_feature_frame(two_days))
    assert list(frame.columns) == feature_names()
    assert len(frame) == len(two_days)


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        compute_features(pd.DataFrame({"ts": [], "close": []}))


def test_causality_future_bars_cannot_change_the_past(two_days):
    """The defining property: appending bars must not move any earlier value."""
    prefix = compute_features(bars_to_feature_frame(two_days[:400]))
    full = compute_features(bars_to_feature_frame(two_days))
    pd.testing.assert_frame_equal(prefix, full.iloc[:400], check_exact=False, rtol=1e-9)


def test_batch_and_incremental_agree(two_days):
    """Constraint 1 in numbers: the live rolling-buffer path reproduces the batch
    values a model was trained on, to floating-point round-off."""
    full = compute_features(bars_to_feature_frame(two_days))
    buffer = DEFAULT_SPEC.buffer_bars(375)
    for index in (500, 600, len(two_days) - 1):
        window = two_days[max(0, index + 1 - buffer) : index + 1]
        values, warm = latest_features(window)
        assert warm
        for name, expected in full.iloc[index].items():
            assert values[name] == pytest.approx(expected, rel=1e-9, abs=1e-12), name


def test_warmup_rows_are_not_warm(calendar):
    bars = make_bars(calendar, [100.0 + i * 0.1 for i in range(40)])
    frame = compute_features(bars_to_feature_frame(bars))
    assert not is_warm(frame.iloc[0])
    # the slowest window is the 30-bar SMA, so row 28 cannot be complete
    assert not is_warm(frame.iloc[DEFAULT_SPEC.sma_slow - 2])
    # and warmup_bars is enough for every feature, which is the contract
    assert is_warm(frame.iloc[DEFAULT_SPEC.warmup_bars - 1])
    assert latest_features(bars[: DEFAULT_SPEC.warmup_bars])[1]
    values, warm = latest_features(bars[:10])
    assert not warm
    assert all(np.isfinite(v) for v in values.values())  # NaNs never leak downstream
    assert latest_features([]) == ({}, False)


def test_known_values(calendar):
    """Spot-check the arithmetic against values computed by hand."""
    prices = [100.0] * 20 + [101.0] * 40
    bars = make_bars(calendar, prices)
    frame = compute_features(bars_to_feature_frame(bars))
    jump = frame.iloc[20]
    assert jump["ret_1"] == pytest.approx(np.log(1.01))
    assert frame.iloc[21]["ret_1"] == pytest.approx(0.0)
    # 15 bars after the step, ret_15 still spans it
    assert frame.iloc[34]["ret_15"] == pytest.approx(np.log(1.01))
    assert frame.iloc[35]["ret_15"] == pytest.approx(0.0)
    # a single up-move inside a flat window pins RSI at 100 (no losses at all)
    assert frame.iloc[25][f"rsi_{DEFAULT_SPEC.rsi}"] == pytest.approx(100.0)
    last = frame.iloc[-1]
    assert last["trend"] == pytest.approx(0.0)  # both SMAs sit at 101 again
    # the last 20 bars are perfectly flat, so "where in the range" is the midpoint
    assert last[f"range_pos_{DEFAULT_SPEC.range_window}"] == pytest.approx(0.5)
    # halfway through the step the close sits at the top of the 20-bar range
    assert frame.iloc[20][f"range_pos_{DEFAULT_SPEC.range_window}"] == pytest.approx(1.0)


def test_session_features(calendar):
    day_one = make_bars(calendar, [100.0] * 50, day=date(2026, 9, 17))
    day_two = make_bars(calendar, [102.0] * 50, day=date(2026, 9, 18), start=102.0)
    frame = compute_features(bars_to_feature_frame(day_one + day_two))
    assert frame["tod"].iloc[0] == pytest.approx(0.0)
    assert frame["tod"].iloc[49] == pytest.approx(49 / 60)
    assert frame["tod"].iloc[50] == pytest.approx(0.0)  # resets at the new session
    assert frame["overnight_gap"].iloc[0] == pytest.approx(0.0)  # no previous session
    assert frame["overnight_gap"].iloc[50] == pytest.approx(0.02)  # 100 -> 102
    assert frame["overnight_gap"].iloc[99] == pytest.approx(0.02)  # constant all day
    assert frame["vwap_dist"].iloc[60] == pytest.approx(0.0)  # flat prices sit on VWAP


def test_daily_interval_skips_intraday_features(calendar):
    bars = make_bars(calendar, [100.0 + i for i in range(40)])
    frame = compute_features(bars_to_feature_frame(bars), interval=Interval.D1)
    assert "vwap_dist" not in frame.columns
    assert "ret_1" in frame.columns
    values, warm = latest_features(bars, interval=Interval.D1)
    assert warm and "tod" not in values


def test_zero_volume_does_not_produce_nan(calendar):
    bars = make_bars(calendar, [100.0 + i * 0.01 for i in range(40)], volume=0)
    values, warm = latest_features(bars)
    assert warm
    assert all(np.isfinite(v) for v in values.values())
