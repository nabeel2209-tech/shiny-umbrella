"""Shared fixtures."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import pytest

from trading.core.clock import DEFAULT_HOLIDAYS_FILE, MarketCalendar, SimClock
from trading.core.types import IST, Bar, Interval

TRADING_DAY = date(2026, 9, 18)  # a Friday, not a holiday


@pytest.fixture(scope="session")
def calendar() -> MarketCalendar:
    return MarketCalendar.load(DEFAULT_HOLIDAYS_FILE)


@pytest.fixture
def trading_day() -> date:
    return TRADING_DAY


def make_synthetic_day(
    calendar: MarketCalendar,
    symbol: str = "NSE:RELIANCE",
    day: date = TRADING_DAY,
    *,
    base: float = 2500.0,
    amplitude: float = 20.0,
    period: float = 30.0,
    exchange: str = "NSE",
) -> list[Bar]:
    """Deterministic 1-minute bars: close follows a sine wave, open = previous close,
    high/low = extremes +/- 1. The path is smooth so a bar never gaps."""
    bars: list[Bar] = []
    prev_close = base
    for i, ts in enumerate(calendar.session_bars(exchange, day, Interval.M1)):
        close = round(base + amplitude * math.sin(i / period), 2)
        o = prev_close
        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                interval=Interval.M1,
                open=o,
                high=round(max(o, close) + 1, 2),
                low=round(min(o, close) - 1, 2),
                close=close,
                volume=1000,
            )
        )
        prev_close = close
    return bars


@pytest.fixture
def synthetic_day(calendar: MarketCalendar) -> Callable[..., list[Bar]]:
    def _make(**kw: object) -> list[Bar]:
        return make_synthetic_day(calendar, **kw)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def ist(trading_day: date) -> Callable[[int, int], datetime]:
    def _at(hour: int, minute: int = 0, second: int = 0) -> datetime:
        return datetime(
            trading_day.year, trading_day.month, trading_day.day, hour, minute, second, tzinfo=IST
        )

    return _at


@pytest.fixture
def sim_clock(ist: Callable[[int, int], datetime]) -> SimClock:
    return SimClock(ist(9, 0))


def make_bars(
    calendar: MarketCalendar,
    prices: list[float],
    symbol: str = "NSE:RELIANCE",
    day: date = TRADING_DAY,
    *,
    interval: Interval = Interval.M1,
    exchange: str = "NSE",
    volume: int = 1000,
    start: float | None = None,
) -> list[Bar]:
    """Bars following an exact close path, so a test can hand-compute everything.

    Each bar opens at the previous close and its range just covers open..close, so
    there are no gaps and no spurious intrabar extremes.
    """
    stamps = calendar.session_bars(exchange, day, interval)
    if len(prices) > len(stamps):
        raise ValueError(f"{len(prices)} prices do not fit in {len(stamps)} bars on {day}")
    out: list[Bar] = []
    prev = prices[0] if start is None else start
    for ts, close in zip(stamps, prices, strict=False):
        out.append(
            Bar(
                symbol=symbol,
                ts=ts,
                interval=interval,
                open=prev,
                high=max(prev, close),
                low=min(prev, close),
                close=close,
                volume=volume,
            )
        )
        prev = close
    return out


@pytest.fixture
def bars_from_prices(calendar: MarketCalendar) -> Callable[..., list[Bar]]:
    def _make(prices: list[float], **kw: object) -> list[Bar]:
        return make_bars(calendar, prices, **kw)  # type: ignore[arg-type]

    return _make


# --------------------------------------------------------------------------- training fixtures
# Trained once per session: the planted-signal and random-walk datasets and models
# shared by the training, evaluation, registry, promotion and schedule tests.

PLANTED_SYMBOL = "NSE:PLANTED"
PLANTED_SESSIONS = 30
HOLDOUT_SESSIONS = 8
PLANTED_HORIZON = 10


@dataclass
class PlantedData:
    frame: pd.DataFrame
    dataset: object
    train: object
    holdout: object
    holdout_start: datetime


def _planted(calendar: MarketCalendar, *, kappa: float, seed: int) -> PlantedData:
    from trading.training.dataset import build_dataset
    from trading.training.labels import LabelSpec
    from trading.training.splits import holdout_split

    from .planted import planted_frame

    frame = planted_frame(calendar, date(2026, 6, 1), PLANTED_SESSIONS, kappa=kappa, seed=seed)
    ds = build_dataset(
        {PLANTED_SYMBOL: frame}, Interval.M1, label=LabelSpec(horizon=PLANTED_HORIZON)
    )
    per_session = len(calendar.session_bars("NSE", date(2026, 6, 1), Interval.M1))
    train, held = holdout_split(
        ds.timestamps, holdout=HOLDOUT_SESSIONS * per_session, purge=PLANTED_HORIZON
    )
    holdout = ds.take(held)
    return PlantedData(frame, ds, ds.take(train), holdout, holdout.meta["ts"].min().to_pydatetime())


@pytest.fixture(scope="session")
def planted_data(calendar: MarketCalendar) -> PlantedData:
    return _planted(calendar, kappa=0.08, seed=1)


@pytest.fixture(scope="session")
def noise_data(calendar: MarketCalendar) -> PlantedData:
    return _planted(calendar, kappa=0.0, seed=2)


@pytest.fixture(scope="session")
def planted_ridge(planted_data: PlantedData):
    from trading.training.train import ModelKind, TrainConfig, fit_model

    return fit_model(planted_data.train, TrainConfig(kind=ModelKind.RIDGE))


@pytest.fixture(scope="session")
def planted_lgbm(planted_data: PlantedData):
    from trading.training.train import ModelKind, TrainConfig, fit_model

    return fit_model(planted_data.train, TrainConfig(kind=ModelKind.LIGHTGBM))


@pytest.fixture(scope="session")
def noise_lgbm(noise_data: PlantedData):
    from trading.training.train import ModelKind, TrainConfig, fit_model

    return fit_model(noise_data.train, TrainConfig(kind=ModelKind.LIGHTGBM))
