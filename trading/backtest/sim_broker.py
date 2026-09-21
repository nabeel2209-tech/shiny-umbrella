"""Simulated broker for backtests.

It is the paper broker with the optimism taken out:

- **Nothing fills when it is placed.** An order decided on a bar's close waits for
  the next bar. Filling it at the close that produced the signal is a quiet
  look-ahead that makes almost any strategy look better.
- **Limits fill only if price trades through them.** A bar whose low merely
  *touches* a buy limit leaves it unfilled - at the touch you would have been
  behind the queue.
- **Stops meet the market where it is**, so a stop gapped through at the open
  fills at the open, not at its trigger.
- **Taking liquidity costs slippage** from a pluggable model; a resting order
  that gets hit does not.
- **Indian costs** on every fill, from ``backtest/costs.py``.
- Optional **volume participation cap**, which turns large orders into partial
  fills across several bars.

Everything runs in memory: a backtest must not touch the paper account's SQLite.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from trading.backtest.costs import DEFAULT_FEES, FeeSchedule
from trading.brokers.paper import PaperBroker, PaperConfig, PriceCtx
from trading.brokers.symbols import contract_multiplier
from trading.core.clock import Clock
from trading.core.types import Order


class SlippageModel(Protocol):
    """Basis points paid on a fill that takes liquidity."""

    def cost_bps(self, order: Order, qty: int, price: float, bar_volume: int | None) -> float: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class FixedSlippage:
    bps: float = 2.0

    def cost_bps(self, order: Order, qty: int, price: float, bar_volume: int | None) -> float:
        return self.bps

    def describe(self) -> str:
        return f"fixed {self.bps} bps"


@dataclass(frozen=True)
class VolumeSlippage:
    """Square-root market impact: ``base + impact * sqrt(qty / bar volume)``.

    ``impact_bps`` is the cost of trading a whole bar's volume. With no volume
    (an index, or a bar with no trades) the full impact is charged - an order
    into an empty market is expensive, not free.
    """

    base_bps: float = 1.0
    impact_bps: float = 25.0

    def cost_bps(self, order: Order, qty: int, price: float, bar_volume: int | None) -> float:
        if not bar_volume:
            return self.base_bps + self.impact_bps
        participation = min(qty / bar_volume, 1.0)
        return self.base_bps + self.impact_bps * math.sqrt(participation)

    def describe(self) -> str:
        return f"{self.base_bps} bps + {self.impact_bps} bps x sqrt(participation)"


@dataclass
class SimConfig:
    starting_cash: float = 1_000_000.0
    slippage: SlippageModel = field(default_factory=FixedSlippage)
    fee_schedule: FeeSchedule = field(default_factory=lambda: DEFAULT_FEES)
    max_participation: float | None = None  # e.g. 0.1 = at most 10% of a bar's volume
    futures_margin_pct: float = 0.10
    lot_size_for: Callable[[str], int] | None = None
    multiplier_for: Callable[[str], float] = contract_multiplier

    def describe(self) -> dict[str, object]:
        return {
            "starting_cash": self.starting_cash,
            "slippage": self.slippage.describe(),
            "max_participation": self.max_participation,
            "futures_margin_pct": self.futures_margin_pct,
            "fills": "next bar; limits need a trade-through; stops fill at the gap",
        }


class SimBroker(PaperBroker):
    name = "sim"

    def __init__(self, config: SimConfig | None = None, *, clock: Clock | None = None) -> None:
        self.sim = config or SimConfig()
        super().__init__(
            None,
            config=PaperConfig(
                starting_cash=self.sim.starting_cash,
                slippage_bps=0.0,  # replaced by the slippage model below
                fee_schedule=self.sim.fee_schedule,
                futures_margin_pct=self.sim.futures_margin_pct,
                require_trade_through=True,
                fill_at_placement=False,
                max_participation=self.sim.max_participation,
                lot_size_for=self.sim.lot_size_for,
                multiplier_for=self.sim.multiplier_for,
                account_id="backtest",
            ),
            store=None,
            clock=clock,
        )

    def _slippage_bps(self, order: Order, qty: int, px: PriceCtx) -> float:
        return self.sim.slippage.cost_bps(order, qty, px.last, px.volume)
