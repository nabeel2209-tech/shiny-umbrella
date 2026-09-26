"""Synthetic minute bars with a known, planted signal, for training tests.

``planted_frame`` simulates a price that mean-reverts toward its own 30-bar
average: the next bar's expected return is ``-kappa * (close / SMA30 - 1)``, which
is exactly the ``dist_sma_30`` feature. ``kappa=0`` gives a pure random walk with
the same volatility - the no-signal control.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading.core.clock import MarketCalendar
from trading.core.types import Interval
from trading.training.ingest import BAR_COLUMNS


def planted_frame(
    calendar: MarketCalendar,
    start: date,
    days: int,
    *,
    kappa: float = 0.08,
    sigma: float = 0.0008,
    seed: int = 1,
    start_price: float = 1000.0,
    symbol: str = "NSE:PLANTED",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = []
    d = start
    while len(sessions) < days:
        if calendar.is_trading_day("NSE", d):
            sessions.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    rows = []
    closes: list[float] = [start_price]
    window_sum = start_price
    for day in sessions:
        for ts in calendar.session_bars("NSE", day, Interval.M1):
            prev = closes[-1]
            n = min(len(closes), 30)
            sma = window_sum / n
            deviation = prev / sma - 1.0
            ret = -kappa * deviation + sigma * rng.standard_normal()
            close = prev * (1.0 + ret)
            wick = abs(rng.standard_normal()) * sigma * 0.3
            rows.append(
                (
                    ts,
                    prev,
                    max(prev, close) * (1 + wick),
                    min(prev, close) * (1 - wick),
                    close,
                    int(rng.lognormal(10, 0.4)),
                    None,
                )
            )
            closes.append(close)
            window_sum += close
            if len(closes) > 30:
                window_sum -= closes[-31]
    frame = pd.DataFrame(rows, columns=BAR_COLUMNS)
    frame["oi"] = frame["oi"].astype("Int64")
    frame.attrs["symbol"] = symbol
    return frame
