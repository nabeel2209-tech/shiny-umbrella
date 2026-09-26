"""Feature computation — the single implementation, used by BOTH training and live.

Constraint 1: this module is imported by ``trading/agents/data.py`` (live feed and
archive replay) and by the training pipeline. Feature code lives here and nowhere
else.

Two properties are enforced by tests:

1. **Causal.** A feature at bar *t* uses only bars at or before *t*. Nothing peeks
   into the future, so labels and features can never leak into each other.
2. **Incremental parity.** Every feature uses a *finite rolling window* - no
   recursive smoothing (classic EMA / Wilder RSI / Wilder ATR), whose value would
   depend on all history back to the first bar ever seen. So a live agent holding
   the last ``buffer_bars`` bars reproduces the batch value to floating-point
   round-off (relative 1e-9; pandas accumulates rolling sums incrementally, which
   moves the last couple of bits depending on where the window starts). RSI and
   ATR therefore use Cutler's (simple-average) formulation.

Session-anchored features (VWAP distance, time of day, overnight gap) need the
first bar of the current session in the window, which is why the buffer size is
``max(warmup_bars, bars_per_session + 1)`` rather than just the longest window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.core.types import Bar, Interval

OHLCV = ("open", "high", "low", "close", "volume")
EPS = 1e-12


@dataclass(frozen=True)
class FeatureSpec:
    """Windows for the feature set. Persisted alongside a trained model so that a
    model is always scored with the features it was fitted on."""

    returns: tuple[int, ...] = (1, 5, 15)
    sma_fast: int = 10
    sma_slow: int = 30
    rsi: int = 14
    atr: int = 14
    vol: int = 20
    volume: int = 20
    range_window: int = 20
    version: str = "v1"

    @property
    def windows(self) -> tuple[int, ...]:
        return (
            *self.returns,
            self.sma_fast,
            self.sma_slow,
            self.rsi,
            self.atr,
            self.vol,
            self.volume,
            self.range_window,
        )

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every feature is defined.

        Windows applied to *returns* (volatility) need one extra bar because the
        first return is undefined, so this is one more than the longest window -
        a bar of slack rather than an exactly tight bound.
        """
        return max(self.windows) + 1

    def buffer_bars(self, bars_per_session: int = 0) -> int:
        """How many bars a live agent must keep to reproduce batch values exactly."""
        return max(self.warmup_bars, bars_per_session + 1)

    def names(self, *, intraday: bool = True) -> list[str]:
        out = [f"ret_{n}" for n in self.returns]
        out += [
            "trend",
            f"dist_sma_{self.sma_slow}",
            f"rsi_{self.rsi}",
            f"atr_pct_{self.atr}",
            f"vol_{self.vol}",
            f"volume_z_{self.volume}",
            f"range_pos_{self.range_window}",
        ]
        if intraday:
            out += ["vwap_dist", "tod", "overnight_gap"]
        return out


DEFAULT_SPEC = FeatureSpec()


def is_intraday(interval: Interval | str) -> bool:
    return Interval(interval) is not Interval.D1


def feature_names(
    spec: FeatureSpec = DEFAULT_SPEC, interval: Interval | str = Interval.M1
) -> list[str]:
    return spec.names(intraday=is_intraday(interval))


# --------------------------------------------------------------------------- helpers


def _rolling(s: pd.Series, n: int) -> pd.core.window.rolling.Rolling:
    """Rolling window that yields NaN until it is completely filled."""
    return s.rolling(n, min_periods=n)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Cutler's RSI: simple averages of gains and losses over a finite window."""
    delta = close.diff()
    gain = _rolling(delta.clip(lower=0), n).mean()
    loss = _rolling((-delta).clip(lower=0), n).mean()
    rs = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr_pct(bars: pd.DataFrame, window: int) -> pd.Series:
    """Average true range over ``window`` bars as a fraction of the close (Cutler).

    Public because triple-barrier labels size their barriers with it - one ATR
    implementation for features and labels alike.
    """
    atr = _rolling(_true_range(bars), window).mean()
    return atr / (bars["close"].astype(float) + EPS)


def _session(ts: pd.Series) -> pd.Series:
    return ts.dt.date


# --------------------------------------------------------------------------- main entry point


def compute_features(
    bars: pd.DataFrame,
    spec: FeatureSpec = DEFAULT_SPEC,
    interval: Interval | str = Interval.M1,
) -> pd.DataFrame:
    """Features for every bar in ``bars`` (archive frame layout: ts + OHLCV).

    Rows without enough history hold NaN; use :func:`is_warm` or ``dropna()``.
    The returned frame has the same index as ``bars``.
    """
    missing = [c for c in OHLCV if c not in bars.columns]
    if missing:
        raise ValueError(f"bars frame is missing columns: {missing}")
    intraday = is_intraday(interval)
    df = bars
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)

    log_close = np.log(close.clip(lower=EPS))
    for n in spec.returns:
        out[f"ret_{n}"] = log_close.diff(n)

    sma_fast = _rolling(close, spec.sma_fast).mean()
    sma_slow = _rolling(close, spec.sma_slow).mean()
    out["trend"] = sma_fast / (sma_slow + EPS) - 1.0
    out[f"dist_sma_{spec.sma_slow}"] = close / (sma_slow + EPS) - 1.0

    out[f"rsi_{spec.rsi}"] = _rsi(close, spec.rsi)

    out[f"atr_pct_{spec.atr}"] = atr_pct(df, spec.atr)

    out[f"vol_{spec.vol}"] = _rolling(out["ret_1"], spec.vol).std()

    volume = df["volume"].astype(float)
    vol_mean = _rolling(volume, spec.volume).mean()
    vol_std = _rolling(volume, spec.volume).std()
    out[f"volume_z_{spec.volume}"] = (volume - vol_mean) / (vol_std + EPS)

    hi = _rolling(df["high"].astype(float), spec.range_window).max()
    lo = _rolling(df["low"].astype(float), spec.range_window).min()
    span = hi - lo
    # a window with no range (a halted or perfectly flat instrument) has no
    # meaningful position in it; call that the midpoint rather than an edge
    out[f"range_pos_{spec.range_window}"] = ((close - lo) / span).where(span > 0, 0.5)

    if intraday:
        session = _session(df["ts"])
        typical = (df["high"] + df["low"] + close) / 3.0
        cum_pv = (typical * volume).groupby(session).cumsum()
        cum_v = volume.groupby(session).cumsum()
        vwap = cum_pv / cum_v.where(cum_v > 0)
        vwap = vwap.fillna(close)  # no volume yet (index feeds): fall back to price
        out["vwap_dist"] = close / (vwap + EPS) - 1.0

        session_start = df["ts"].groupby(session).transform("first")
        out["tod"] = (df["ts"] - session_start).dt.total_seconds() / 3600.0

        session_open = df["open"].astype(float).groupby(session).transform("first")
        prev_session_close = close.groupby(session).last().shift(1)
        prev_close = session.map(prev_session_close)
        out["overnight_gap"] = (session_open / (prev_close.astype(float) + EPS) - 1.0).fillna(0.0)

    return out[spec.names(intraday=intraday)]


def is_warm(row: pd.Series) -> bool:
    """True when every feature in ``row`` is finite."""
    return bool(np.isfinite(pd.to_numeric(row, errors="coerce")).all())


def bars_to_feature_frame(bars: list[Bar]) -> pd.DataFrame:
    """Bars -> the frame layout ``compute_features`` expects (same as the archive)."""
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([b.ts for b in bars], utc=True).tz_convert(bars[0].ts.tzinfo)
            if bars
            else pd.Series(dtype="datetime64[ns, UTC]"),
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )


def latest_features(
    bars: list[Bar],
    spec: FeatureSpec = DEFAULT_SPEC,
    interval: Interval | str = Interval.M1,
) -> tuple[dict[str, float], bool]:
    """Features for the last bar of ``bars``; returns ``(values, warm)``.

    This is what the live data agent calls on its rolling buffer. It runs the very
    same :func:`compute_features`, so live and training values agree to round-off.
    """
    if not bars:
        return {}, False
    frame = compute_features(bars_to_feature_frame(bars), spec, interval)
    row = frame.iloc[-1]
    warm = is_warm(row)
    values = {k: (float(v) if np.isfinite(v) else 0.0) for k, v in row.items()}
    return values, warm
