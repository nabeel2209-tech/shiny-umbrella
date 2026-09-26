"""Training labels, net of trading costs (constraint 8).

Two label kinds:

``forward_return``
    The close-to-close return over the next ``horizon`` bars.
``triple_barrier``
    Whichever comes first within ``horizon`` bars: an upper barrier (profit take),
    a lower barrier (stop), or the horizon (timeout). Barriers are multiples of the
    bar's ATR by default. When one bar touches both, the stop is assumed first -
    the conservative reading of a bar we cannot see inside.

Both are turned into a **net** label with the same rule::

    net = sign(gross) * max(|gross| - cost, 0)

- the profit from trading in the direction of the move after paying a full round
trip, and zero for a move too small to pay for itself. The model therefore learns
that small moves are worth nothing, and the sign of the label is always the side
worth taking. ``cost`` is the round-trip cost from ``backtest/costs.py`` for the
label's product and a representative trade size, plus slippage on both legs.

For intraday products a label never spans the overnight gap: an MIS position is
squared off before the close, so a horizon that would cross the session boundary
has no label (``cross_sessions=False``).

``label_uniqueness`` gives the average uniqueness of each label (López de Prado):
with every bar starting an ``h``-bar label, neighbouring labels share most of their
returns, and weighting each by 1/overlap stops the training set from counting the
same move ``h`` times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from trading.backtest.costs import DEFAULT_FEES, FeeSchedule, round_trip_cost_bps
from trading.brokers.symbols import contract_multiplier
from trading.core.types import ProductType
from trading.features.features import atr_pct


class LabelKind(StrEnum):
    FORWARD_RETURN = "forward_return"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass(frozen=True)
class LabelSpec:
    kind: LabelKind = LabelKind.FORWARD_RETURN
    horizon: int = 12  # bars
    product: ProductType = ProductType.MIS
    notional: float = 250_000.0  # trade size used to price the costs
    slippage_bps: float = 2.0  # per side, on top of fees
    cost_bps: float | None = None  # override the computed round-trip cost
    cross_sessions: bool | None = None  # None: False for MIS, True otherwise
    profit_take: float = 2.0  # triple barrier, in ATRs (or fractions if atr_scaled=False)
    stop_loss: float = 1.0
    atr_scaled: bool = True
    atr_window: int = 14

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least one bar")
        if self.profit_take <= 0 or self.stop_loss <= 0:
            raise ValueError("barriers must be positive")

    @property
    def spans_sessions(self) -> bool:
        if self.cross_sessions is not None:
            return self.cross_sessions
        return self.product is not ProductType.MIS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["product"] = self.product.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelSpec:
        return cls(**{**d, "kind": LabelKind(d["kind"]), "product": ProductType(d["product"])})


def cost_fraction(
    symbol: str,
    price: float,
    spec: LabelSpec,
    *,
    lot: int = 1,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> float:
    """Round-trip cost as a fraction of trade value, at a representative size."""
    if spec.cost_bps is not None:
        return spec.cost_bps / 10_000
    mult = contract_multiplier(symbol)
    qty = max(lot, int(spec.notional / max(price * mult, 1e-9)) // lot * lot)
    bps = round_trip_cost_bps(
        symbol,
        qty,
        price,
        spec.product,
        slippage_bps=spec.slippage_bps,
        schedule=schedule,
        multiplier=mult,
    )
    return bps / 10_000


def net_of_costs(gross: pd.Series, cost: float) -> pd.Series:
    """``sign(g) * max(|g| - cost, 0)``: the move you could have kept."""
    return np.sign(gross) * (gross.abs() - cost).clip(lower=0.0)


def _session_mask(ts: pd.Series, horizon: int) -> np.ndarray:
    """True where the bar ``horizon`` ahead is in the same session."""
    day = ts.dt.date.to_numpy()
    ahead = np.empty_like(day)
    ahead[:-horizon] = day[horizon:] if horizon < len(day) else day[:0]
    same = np.zeros(len(day), dtype=bool)
    if horizon < len(day):
        same[:-horizon] = day[:-horizon] == ahead[:-horizon]
    return same


def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    close = bars["close"].astype(float)
    return close.shift(-horizon) / close - 1.0


def triple_barrier(bars: pd.DataFrame, spec: LabelSpec) -> tuple[pd.Series, pd.Series]:
    """(gross outcome return, outcome) per bar; outcome +1 profit, -1 stop, 0 timeout."""
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n, h = len(close), spec.horizon
    atr = atr_pct(bars, spec.atr_window).to_numpy(dtype=float)
    width = atr if spec.atr_scaled else np.ones(n)
    up = spec.profit_take * width
    dn = spec.stop_loss * width
    gross = np.full(n, np.nan)
    outcome = np.full(n, np.nan)
    for i in range(n - h):
        if not np.isfinite(up[i]):
            continue
        upper, lower = close[i] * (1 + up[i]), close[i] * (1 - dn[i])
        result, kind = close[i + h] / close[i] - 1.0, 0.0
        for j in range(i + 1, i + h + 1):
            hit_low, hit_high = low[j] <= lower, high[j] >= upper
            if hit_low:  # also when both are hit in one bar: assume the stop first
                result, kind = -dn[i], -1.0
                break
            if hit_high:
                result, kind = up[i], 1.0
                break
        gross[i], outcome[i] = result, kind
    return pd.Series(gross, index=bars.index), pd.Series(outcome, index=bars.index)


@dataclass
class Labels:
    net: pd.Series  # the training target
    gross: pd.Series  # before costs, for evaluation
    outcome: pd.Series  # triple barrier: +1/-1/0; forward return: sign of gross
    cost: float  # round-trip cost fraction used
    uniqueness: pd.Series


def make_labels(
    bars: pd.DataFrame,
    symbol: str,
    spec: LabelSpec,
    *,
    lot: int = 1,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> Labels:
    """Labels for one symbol's bars (archive frame layout, sorted by ts)."""
    if spec.kind is LabelKind.TRIPLE_BARRIER:
        gross, outcome = triple_barrier(bars, spec)
    else:
        gross = forward_return(bars, spec.horizon)
        outcome = np.sign(gross)
    if not spec.spans_sessions:
        same = _session_mask(bars["ts"], spec.horizon)
        gross = gross.where(same)
        outcome = outcome.where(same)
    price = float(bars["close"].median()) if len(bars) else 1.0
    cost = cost_fraction(symbol, price, spec, lot=lot, schedule=schedule)
    net = net_of_costs(gross, cost)
    valid = gross.notna().to_numpy()
    return Labels(
        net=net,
        gross=gross,
        outcome=outcome,
        cost=cost,
        uniqueness=pd.Series(label_uniqueness(valid, spec.horizon), index=bars.index),
    )


def label_uniqueness(valid: np.ndarray, horizon: int) -> np.ndarray:
    """Average uniqueness of each label spanning bars (i, i + horizon].

    ``c_t`` counts the labels whose span covers bar t; a label's uniqueness is the
    mean of ``1 / c_t`` over its span. Invalid labels get 0.
    """
    n = len(valid)
    starts = valid.astype(float)
    # label i covers bars i+1 .. i+h: add at i+1, remove after i+h
    delta = np.zeros(n + horizon + 2)
    idx = np.nonzero(valid)[0]
    np.add.at(delta, idx + 1, 1.0)
    np.add.at(delta, idx + horizon + 1, -1.0)
    concurrency = np.cumsum(delta)[: n + horizon + 1]
    inv = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1.0), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    out = np.zeros(n)
    for i in idx:
        out[i] = (csum[i + horizon + 1] - csum[i + 1]) / horizon
    return out * starts
