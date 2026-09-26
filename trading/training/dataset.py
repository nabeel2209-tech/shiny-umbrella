"""Model datasets: features from ``features.py`` (constraint 1), labels from
``labels.py`` (constraint 8), for one or many symbols of one interval.

Every warm bar is a row. Rows whose label cannot be formed - the last ``horizon``
bars, or bars whose horizon would cross a session an intraday product cannot hold -
keep their features but have a NaN label: they are predicted on in evaluation (a
position already on the book still earns there) but never trained on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from trading.backtest.costs import DEFAULT_FEES, FeeSchedule
from trading.brokers.lots import LotSizes
from trading.core.clock import MarketCalendar
from trading.core.types import Interval
from trading.features.features import DEFAULT_SPEC, FeatureSpec, compute_features, feature_names
from trading.training.ingest import Archive, bars_to_frame, empty_frame
from trading.training.labels import LabelSpec, make_labels

META_COLUMNS = ["ts", "symbol", "close", "ret_next", "gross", "cost", "uniqueness", "outcome"]


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series  # net label, NaN where no label can be formed
    meta: pd.DataFrame  # META_COLUMNS
    spec: FeatureSpec
    label: LabelSpec
    interval: Interval
    symbols: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def labelled(self) -> np.ndarray:
        return np.nonzero(self.y.notna().to_numpy())[0]

    @property
    def timestamps(self) -> pd.Series:
        return self.meta["ts"]

    def take(self, rows: np.ndarray) -> Dataset:
        rows = np.asarray(rows)
        return Dataset(
            X=self.X.iloc[rows].reset_index(drop=True),
            y=self.y.iloc[rows].reset_index(drop=True),
            meta=self.meta.iloc[rows].reset_index(drop=True),
            spec=self.spec,
            label=self.label,
            interval=self.interval,
            symbols=self.symbols,
        )

    def window(self) -> tuple[str | None, str | None]:
        if self.meta.empty:
            return None, None
        return self.meta["ts"].min().isoformat(), self.meta["ts"].max().isoformat()


def build_dataset(
    frames: Mapping[str, pd.DataFrame],
    interval: Interval | str,
    *,
    spec: FeatureSpec = DEFAULT_SPEC,
    label: LabelSpec | None = None,
    lots: LotSizes | None = None,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> Dataset:
    """``frames`` maps a symbol to its bars in archive layout (ts + OHLCV)."""
    interval = Interval(interval)
    label = label or LabelSpec()
    lots = lots or LotSizes()
    names = feature_names(spec, interval)
    parts_x, parts_y, parts_meta = [], [], []
    for symbol, frame in sorted(frames.items()):
        if frame.empty:
            continue
        bars = frame.sort_values("ts").reset_index(drop=True)
        feats = compute_features(bars, spec, interval)
        labels = make_labels(bars, symbol, label, lot=lots.get(symbol), schedule=schedule)
        close = bars["close"].astype(float)
        ret_next = close.shift(-1) / close - 1.0
        if not label.spans_sessions:  # an intraday book is flat overnight
            same_day = bars["ts"].dt.date.shift(-1) == bars["ts"].dt.date
            ret_next = ret_next.where(same_day, 0.0)
        warm = feats.notna().all(axis=1).to_numpy()
        meta = pd.DataFrame(
            {
                "ts": bars["ts"],
                "symbol": symbol,
                "close": close,
                "ret_next": ret_next.fillna(0.0),
                "gross": labels.gross,
                "cost": labels.cost,
                "uniqueness": labels.uniqueness,
                "outcome": labels.outcome,
            }
        )
        parts_x.append(feats.loc[warm, names])
        parts_y.append(labels.net.loc[warm])
        parts_meta.append(meta.loc[warm])
    if not parts_x:
        empty = pd.DataFrame(columns=names, dtype=float)
        return Dataset(
            empty, pd.Series(dtype=float), pd.DataFrame(columns=META_COLUMNS), spec, label, interval
        )
    X = pd.concat(parts_x, ignore_index=True)
    y = pd.concat(parts_y, ignore_index=True)
    meta = pd.concat(parts_meta, ignore_index=True)
    order = np.lexsort((meta["symbol"].to_numpy(), meta["ts"].to_numpy()))  # by time, then symbol
    return Dataset(
        X=X.iloc[order].reset_index(drop=True),
        y=y.iloc[order].reset_index(drop=True),
        meta=meta.iloc[order].reset_index(drop=True),
        spec=spec,
        label=label,
        interval=interval,
        symbols=sorted(frames),
    )


def load_frames(
    archive: Archive,
    symbols: Sequence[str],
    interval: Interval | str,
    start: date,
    end: date,
    *,
    calendar: MarketCalendar,
    adjusted: bool = True,
) -> dict[str, pd.DataFrame]:
    """Bars per symbol in archive layout. An interval that is not archived is built
    from 1-minute bars with the live ``BarBuilder`` (``agents.data.resample_bars``),
    exactly as the backtester does - so a model is trained on the bars it will see."""
    from trading.agents.data import resample_bars  # agents -> training never imports back

    interval = Interval(interval)
    out = {}
    for symbol in symbols:
        if interval is Interval.M1 or archive.partitions(symbol, interval):
            out[symbol] = archive.read(symbol, interval, start, end, adjusted=adjusted)
            continue
        minute = archive.read_bars(symbol, Interval.M1, start, end, adjusted=adjusted)
        out[symbol] = (
            bars_to_frame(resample_bars(minute, interval, calendar)) if minute else empty_frame()
        )
    return out


def dataset_from_archive(
    archive: Archive,
    symbols: Sequence[str],
    interval: Interval | str,
    start: date,
    end: date,
    *,
    spec: FeatureSpec = DEFAULT_SPEC,
    label: LabelSpec | None = None,
    lots: LotSizes | None = None,
    adjusted: bool = True,
    schedule: FeeSchedule = DEFAULT_FEES,
    calendar: MarketCalendar | None = None,
) -> Dataset:
    if calendar is None:
        from trading.core.clock import get_calendar

        calendar = get_calendar()
    frames = load_frames(
        archive, symbols, interval, start, end, calendar=calendar, adjusted=adjusted
    )
    return build_dataset(frames, interval, spec=spec, label=label, lots=lots, schedule=schedule)
