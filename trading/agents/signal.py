"""Signal agent: features in, OrderIntents out.

One agent per strategy. It never talks to a broker and never sizes past what the
strategy asks for - the risk agent is the only thing that can approve a trade, and
it may cut the size (constraint 3).

Two modes, combinable:

- **rule-based** - the YAML condition tree in ``strategies/schema.py``
- **model-based** - a score from the model registry (Phase 5). The model version is
  looked up on every bar through a :class:`ModelProvider`, so a newly promoted
  version is picked up without a restart.

When both are configured the rules act as a filter: the model must agree with the
direction the rules allow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.agents.base import Agent
from trading.agents.portfolio import Portfolio
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock
from trading.core.types import (
    AlertLevel,
    FeatureVector,
    Fill,
    Order,
    OrderIntent,
    RiskRejection,
    Side,
    Signal,
)
from trading.strategies.schema import SizingMode, StrategyConfig


@dataclass(frozen=True)
class ModelPrediction:
    score: float  # signed; sign is the direction
    version: str
    prob: float | None = None  # p(win), for Kelly sizing
    expected_edge_bps: float | None = None


class ModelProvider(Protocol):
    """What the signal agent needs from the model registry (Phase 5 implements it)."""

    def live_version(self, name: str) -> str | None: ...

    def predict(
        self, name: str, version: str | None, features: dict[str, float]
    ) -> ModelPrediction | None: ...


class SignalAgent(Agent):
    def __init__(
        self,
        bus: MessageBus,
        strategy: StrategyConfig,
        portfolio: Portfolio,
        *,
        models: ModelProvider | None = None,
        lot_size_for: dict[str, int] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.name = f"signal:{strategy.id}"
        super().__init__(bus, clock=clock)
        self.strategy = strategy
        self.portfolio = portfolio
        self.models = models
        self.lot_sizes = lot_size_for or {}
        self._pending: dict[str, str] = {}  # symbol -> intent id awaiting an outcome
        self._model_version: str | None = None
        self.signals_emitted = 0
        self.intents_emitted = 0
        self.skipped_cold = 0

    async def on_start(self) -> None:
        for symbol in self.strategy.symbols:
            await self.subscribe(Topics.features(symbol), self._on_features)
        await self.subscribe(Topics.FILLS, self._on_fill)
        await self.subscribe(Topics.REJECTED, self._on_rejection)
        await self.subscribe(Topics.ORDERS, self._on_order)

    # ------------------------------------------------------------------ inbound
    async def _on_fill(self, _topic: str, fill: Fill) -> None:  # type: ignore[override]
        self._pending.pop(fill.symbol, None)

    async def _on_rejection(self, _topic: str, rej: RiskRejection) -> None:  # type: ignore[override]
        self._clear_pending(rej.intent_id)

    async def _on_order(self, _topic: str, order: Order) -> None:  # type: ignore[override]
        """A cancelled or rejected order must not leave the strategy stuck."""
        if order.status.is_terminal and order.intent_id:
            self._clear_pending(order.intent_id)

    def _clear_pending(self, intent_id: str) -> None:
        for symbol, pending_id in list(self._pending.items()):
            if pending_id == intent_id:
                del self._pending[symbol]

    async def _on_features(self, _topic: str, fv: FeatureVector) -> None:  # type: ignore[override]
        if not self.strategy.enabled or fv.interval is not self.strategy.interval:
            return
        if not fv.warm:
            self.skipped_cold += 1
            return
        if fv.symbol in self._pending:
            return  # an intent for this symbol is still in flight
        decision = self.decide(fv)
        if decision is None:
            return
        side, qty, prediction, reason = decision
        signal = Signal(
            ts=fv.ts,
            strategy_id=self.strategy.id,
            symbol=fv.symbol,
            score=prediction.score if prediction else (1.0 if side is Side.BUY else -1.0),
            prob=prediction.prob if prediction else None,
            expected_edge_bps=prediction.expected_edge_bps if prediction else None,
            model_version=prediction.version if prediction else None,
            meta={"reason": reason, "interval": fv.interval.value},
        )
        await self.publish(Topics.SIGNALS, signal)
        self.signals_emitted += 1
        intent = self.build_intent(fv, side, qty, signal)
        self._pending[fv.symbol] = intent.id
        await self.publish(Topics.INTENTS, intent)
        self.intents_emitted += 1

    # ------------------------------------------------------------------ decision
    def decide(self, fv: FeatureVector) -> tuple[Side, int, ModelPrediction | None, str] | None:
        """(side, qty, prediction, reason) or None to stand pat."""
        s = self.strategy
        held = self.portfolio.net_qty(fv.symbol, s.product)
        prediction = self._predict(fv)
        if s.model.enabled and prediction is None:
            return None

        if held == 0:
            for name, side in (("long", Side.BUY), ("short", Side.SELL)):
                rule = s.rules.get(name)
                if rule is None:
                    continue
                if not self._fires(rule, fv):
                    continue
                if prediction is not None and not self._model_agrees(prediction, side):
                    continue
                qty = self.size(fv, side, prediction)
                if qty <= 0:
                    return None
                return side, qty, prediction, f"rule:{name}"
            if not s.rules and prediction is not None:  # model-only strategy
                side = Side.BUY if prediction.score > 0 else Side.SELL
                qty = self.size(fv, side, prediction)
                if qty <= 0:
                    return None
                return side, qty, prediction, "model"
            return None

        # in a position: only exits
        name = "exit_long" if held > 0 else "exit_short"
        rule = s.rules.get(name)
        exit_side = Side.SELL if held > 0 else Side.BUY
        if rule is not None and self._fires(rule, fv):
            return exit_side, abs(held), prediction, f"rule:{name}"
        if rule is None and not s.rules and prediction is not None:
            flipped = (held > 0 and prediction.score < 0) or (held < 0 and prediction.score > 0)
            if flipped and abs(prediction.score) >= s.model.min_abs_score:
                return exit_side, abs(held), prediction, "model:flip"
        return None

    def _fires(self, rule, fv: FeatureVector) -> bool:  # type: ignore[no-untyped-def]
        try:
            return rule.evaluate(fv.values)
        except KeyError as e:
            self.log.error("strategy %s references unknown feature %s", self.strategy.id, e)
            self.spawn(
                self.alert(AlertLevel.ERROR, f"unknown feature {e} in strategy {self.strategy.id}")
            )
            return False

    def _model_agrees(self, prediction: ModelPrediction, side: Side) -> bool:
        if abs(prediction.score) < self.strategy.model.min_abs_score:
            return False
        return (prediction.score > 0) == (side is Side.BUY)

    def _predict(self, fv: FeatureVector) -> ModelPrediction | None:
        ref = self.strategy.model
        if not ref.enabled or self.models is None:
            return None
        name = ref.name
        assert name is not None
        version = ref.version or self.models.live_version(name)
        if version != self._model_version:
            if self._model_version is not None:
                self.log.info(
                    "strategy %s switching model %s -> %s",
                    self.strategy.id,
                    self._model_version,
                    version,
                )
            self._model_version = version
        if version is None:
            return None
        return self.models.predict(name, version, fv.values)

    # ------------------------------------------------------------------ sizing
    def size(self, fv: FeatureVector, side: Side, prediction: ModelPrediction | None) -> int:
        """The strategy's size *suggestion*; the risk agent caps it."""
        s = self.strategy.sizing
        price = fv.bar.close
        mult = contract_multiplier(fv.symbol)
        lot = max(1, self.lot_sizes.get(fv.symbol, 1))
        match s.mode:
            case SizingMode.FIXED_QTY:
                qty = s.qty
            case SizingMode.FIXED_NOTIONAL:
                qty = int(s.notional / max(price * mult, 1e-9))
            case SizingMode.KELLY:
                budget = self.portfolio.equity * s.fraction
                qty = int(budget / max(price * mult, 1e-9))
        qty = (qty // lot) * lot
        if s.max_qty is not None:
            qty = min(qty, (s.max_qty // lot) * lot)
        return max(qty, 0)

    def build_intent(self, fv: FeatureVector, side: Side, qty: int, signal: Signal) -> OrderIntent:
        s = self.strategy
        edge = signal.expected_edge_bps
        if edge is None:
            edge = (
                abs(signal.score) * s.model.edge_scale_bps
                if s.model.enabled
                else s.expected_edge_bps
            )
        return OrderIntent(
            ts=fv.ts,
            strategy_id=s.id,
            symbol=fv.symbol,
            side=side,
            qty=qty,
            product=s.product,
            urgency=s.execution.urgency,
            reference_price=fv.bar.close,
            limit_band_bps=s.execution.limit_band_bps,
            ttl_seconds=s.execution.ttl_seconds,
            expected_edge_bps=float(edge),
            prob=signal.prob,
            payoff_ratio=s.payoff_ratio,
            model_version=signal.model_version,
            signal_id=signal.id,
            meta={
                "reason": signal.meta.get("reason", ""),
                "stop_loss_pct": s.execution.stop_loss_pct,
                "take_profit_pct": s.execution.take_profit_pct,
                "is_exit": self.portfolio.net_qty(fv.symbol, s.product) != 0,
            },
        )
