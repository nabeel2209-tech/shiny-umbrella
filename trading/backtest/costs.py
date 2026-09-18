"""Indian transaction cost model.

Used by the paper broker, the sim broker, label generation (labels are net of
costs) and the risk agent's cost threshold.

Defaults follow Dhan's published pricing and the statutory rates in force after
the 1 Oct 2024 revisions. They are *defaults*: verify against
https://dhan.co/pricing and the exchange circulars, and override via
``FeeSchedule`` if your account differs. Slippage is not a fee; the paper/sim
brokers model it separately, and ``round_trip_cost_bps`` can add an assumption.

Brokerage is charged per order; we compute it per fill, which slightly
over-estimates cost for orders that fill in several parts.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading.brokers.symbols import ParsedSymbol, parse_symbol
from trading.core.types import Exchange, FeeBreakdown, InstrumentKind, ProductType, Side


@dataclass(frozen=True)
class SegmentFees:
    brokerage_pct: float  # fraction of trade value
    brokerage_cap: float | None  # INR per order ("Rs 20 or x%, whichever lower")
    stt_buy_pct: float  # STT / CTT on buy side
    stt_sell_pct: float  # STT / CTT on sell side
    exchange_pct: float  # exchange transaction charge
    sebi_pct: float  # SEBI turnover fee (Rs 10 / crore)
    stamp_buy_pct: float  # stamp duty, buy side only
    gst_pct: float = 0.18  # on brokerage + exchange + SEBI
    dp_charge_sell: float = 0.0  # INR per sell order (delivery only)


SEBI = 0.000001  # Rs 10 per crore

DEFAULT_FEES_BY_SEGMENT: dict[str, SegmentFees] = {
    "equity_delivery": SegmentFees(
        brokerage_pct=0.0,
        brokerage_cap=None,
        stt_buy_pct=0.001,
        stt_sell_pct=0.001,
        exchange_pct=0.0000297,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00015,
        dp_charge_sell=14.75,  # Dhan: Rs 12.50 + GST per scrip per day; verify
    ),
    "equity_intraday": SegmentFees(
        brokerage_pct=0.0003,
        brokerage_cap=20.0,
        stt_buy_pct=0.0,
        stt_sell_pct=0.00025,
        exchange_pct=0.0000297,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00003,
    ),
    "futures": SegmentFees(
        brokerage_pct=0.0003,
        brokerage_cap=20.0,
        stt_buy_pct=0.0,
        stt_sell_pct=0.0002,
        exchange_pct=0.0000173,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00002,
    ),
    "options": SegmentFees(  # rates apply to premium value
        brokerage_pct=0.0003,
        brokerage_cap=20.0,
        stt_buy_pct=0.0,
        stt_sell_pct=0.001,
        exchange_pct=0.0003503,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00003,
    ),
    "mcx_futures": SegmentFees(
        brokerage_pct=0.0003,
        brokerage_cap=20.0,
        stt_buy_pct=0.0,
        stt_sell_pct=0.0001,  # CTT, non-agri
        exchange_pct=0.000021,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00002,
    ),
    "mcx_options": SegmentFees(
        brokerage_pct=0.0003,
        brokerage_cap=20.0,
        stt_buy_pct=0.0,
        stt_sell_pct=0.0005,  # CTT on premium
        exchange_pct=0.000418,
        sebi_pct=SEBI,
        stamp_buy_pct=0.00003,
    ),
}


@dataclass(frozen=True)
class FeeSchedule:
    segments: dict[str, SegmentFees]

    def for_segment(self, segment: str) -> SegmentFees:
        return self.segments[segment]


DEFAULT_FEES = FeeSchedule(DEFAULT_FEES_BY_SEGMENT)
ZERO_FEES = FeeSchedule(
    {k: SegmentFees(0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for k in DEFAULT_FEES_BY_SEGMENT}
)


def segment_for(symbol: str | ParsedSymbol, product: ProductType) -> str:
    p = parse_symbol(symbol) if isinstance(symbol, str) else symbol
    if p.exchange is Exchange.MCX:
        return "mcx_options" if p.kind is InstrumentKind.OPTION else "mcx_futures"
    if p.kind is InstrumentKind.OPTION:
        return "options"
    if p.kind is InstrumentKind.FUTURE:
        return "futures"
    if p.kind is InstrumentKind.INDEX:
        raise ValueError(f"{p.canonical} is an index and cannot be traded")
    return "equity_delivery" if product is ProductType.CNC else "equity_intraday"


def compute_fees(
    symbol: str | ParsedSymbol,
    side: Side,
    qty: int,
    price: float,
    product: ProductType,
    *,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> FeeBreakdown:
    """Fees for one fill of ``qty`` units at ``price``."""
    seg = schedule.for_segment(segment_for(symbol, product))
    value = qty * price
    brokerage = value * seg.brokerage_pct
    if seg.brokerage_cap is not None:
        brokerage = min(brokerage, seg.brokerage_cap)
    stt = value * (seg.stt_buy_pct if side is Side.BUY else seg.stt_sell_pct)
    exchange = value * seg.exchange_pct
    sebi = value * seg.sebi_pct
    stamp = value * seg.stamp_buy_pct if side is Side.BUY else 0.0
    gst = (brokerage + exchange + sebi) * seg.gst_pct
    other = seg.dp_charge_sell if side is Side.SELL else 0.0
    return FeeBreakdown(
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange=round(exchange, 4),
        sebi=round(sebi, 4),
        stamp=round(stamp, 4),
        gst=round(gst, 4),
        other=round(other, 4),
    )


def round_trip_fees(
    symbol: str | ParsedSymbol,
    qty: int,
    price: float,
    product: ProductType,
    *,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> FeeBreakdown:
    """Buy + sell of ``qty`` at the same price."""
    return compute_fees(symbol, Side.BUY, qty, price, product, schedule=schedule) + compute_fees(
        symbol, Side.SELL, qty, price, product, schedule=schedule
    )


def round_trip_cost_bps(
    symbol: str | ParsedSymbol,
    qty: int,
    price: float,
    product: ProductType,
    *,
    slippage_bps: float = 0.0,
    schedule: FeeSchedule = DEFAULT_FEES,
) -> float:
    """Round-trip cost as basis points of one-way trade value, incl. an optional
    slippage assumption applied on both legs. This is the hurdle an expected edge
    must clear (risk agent cost threshold)."""
    value = qty * price
    if value <= 0:
        raise ValueError("trade value must be positive")
    fees = round_trip_fees(symbol, qty, price, product, schedule=schedule).total
    return fees / value * 10_000 + 2 * slippage_bps
