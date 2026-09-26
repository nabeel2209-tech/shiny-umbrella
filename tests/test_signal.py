"""Signal agent: rules, exits, sizing, model integration and hot reload."""

from __future__ import annotations

from datetime import datetime

import pytest

from trading.agents.portfolio import Portfolio
from trading.agents.signal import ModelPrediction, SignalAgent
from trading.brokers.symbols import contract_multiplier
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Bar,
    FeatureVector,
    Fill,
    Interval,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    ProductType,
    RiskRejection,
    Side,
    Signal,
    Urgency,
)
from trading.strategies.schema import StrategyConfig

SYM = "NSE:RELIANCE"
TS = datetime(2026, 9, 18, 10, 0, tzinfo=IST)

BASE = {
    "id": "s1",
    "symbols": [SYM],
    "interval": "1m",
    "product": "MIS",
    "expected_edge_bps": 30.0,
    "rules": {
        "long": {"all": [{"feature": "trend", "op": "gt", "value": 0.0}]},
        "short": {"all": [{"feature": "trend", "op": "lt", "value": -0.01}]},
        "exit_long": {"all": [{"feature": "trend", "op": "lt", "value": 0.0}]},
        "exit_short": {"all": [{"feature": "trend", "op": "gt", "value": 0.0}]},
    },
    "sizing": {"mode": "fixed_qty", "qty": 10},
    "execution": {"urgency": "PASSIVE", "limit_band_bps": 5, "ttl_seconds": 120},
}


def strategy(**overrides) -> StrategyConfig:
    return StrategyConfig.model_validate({**BASE, **overrides})


def fv(
    trend: float = 0.01,
    *,
    warm: bool = True,
    close: float = 2500.0,
    interval=Interval.M1,
    symbol=SYM,
):
    bar = Bar(
        symbol=symbol,
        ts=TS,
        interval=interval,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100,
    )
    return FeatureVector(
        symbol=symbol,
        ts=TS,
        interval=interval,
        bar=bar,
        values={"trend": trend, "rsi_14": 50.0},
        warm=warm,
    )


class StubModels:
    def __init__(self, prediction: ModelPrediction | None = None, version: str = "v1") -> None:
        self.prediction = prediction
        self.version = version
        self.calls: list[tuple[str, str | None]] = []

    def live_version(self, name: str) -> str | None:
        return self.version

    def predict(self, name, version, features):
        self.calls.append((name, version))
        if self.prediction is None:
            return None
        return ModelPrediction(
            score=self.prediction.score,
            version=version or self.version,
            prob=self.prediction.prob,
            expected_edge_bps=self.prediction.expected_edge_bps,
        )


async def build(strategy_config=None, *, models=None, portfolio=None, lots=None):
    bus = InMemoryBus()
    signals: list[Signal] = []
    intents: list[OrderIntent] = []

    async def on_signal(_t, m):
        signals.append(m)

    async def on_intent(_t, m):
        intents.append(m)

    await bus.subscribe(Topics.SIGNALS, on_signal)
    await bus.subscribe(Topics.INTENTS, on_intent)
    p = portfolio or Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    agent = SignalAgent(
        bus,
        strategy_config or strategy(),
        p,
        models=models,
        lot_size_for=lots,
        clock=SimClock(TS),
    )
    await agent.start()
    return bus, agent, p, signals, intents


# --------------------------------------------------------------------------- rules


async def test_long_rule_fires_and_builds_an_intent():
    bus, _agent, _, signals, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert len(intents) == 1
    i = intents[0]
    assert i.side is Side.BUY and i.qty == 10 and i.symbol == SYM
    assert i.product is ProductType.MIS and i.urgency is Urgency.PASSIVE
    assert i.reference_price == 2500.0 and i.limit_band_bps == 5 and i.ttl_seconds == 120
    assert i.expected_edge_bps == 30.0  # declared by the strategy, no model
    assert i.strategy_id == "s1" and i.signal_id == signals[0].id
    assert i.meta["reason"] == "rule:long" and i.meta["is_exit"] is False
    assert signals[0].score == 1.0 and signals[0].model_version is None


async def test_short_rule_fires_when_its_own_condition_is_met():
    bus, _, _, _, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=-0.005))  # neither long nor short
    assert intents == []
    await bus.publish(Topics.features(SYM), fv(trend=-0.02))
    assert intents[0].side is Side.SELL


async def test_no_signal_until_features_are_warm():
    bus, agent, _, _, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=0.02, warm=False))
    assert intents == [] and agent.skipped_cold == 1


async def test_only_the_strategy_interval_is_acted_on():
    bus, _, _, _, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=0.02, interval=Interval.M5))
    assert intents == []


async def test_disabled_strategy_does_nothing():
    bus, _, _, _, intents = await build(strategy(enabled=False))
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert intents == []


# --------------------------------------------------------------------------- position awareness


async def test_one_intent_at_a_time_per_symbol():
    bus, _, _, _, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    await bus.publish(Topics.features(SYM), fv(trend=0.03))
    assert len(intents) == 1  # still waiting on the first


async def test_pending_clears_on_fill_rejection_and_dead_order():
    bus, agent, _portfolio, _, intents = await build()
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    await bus.publish(
        Topics.REJECTED, RiskRejection(intent_id=intents[0].id, ts=TS, rule="x", reason="y")
    )
    assert agent._pending == {}
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert len(intents) == 2
    dead = Order(
        id="o1",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        product=ProductType.MIS,
        status=OrderStatus.CANCELLED,
        created_at=TS,
        updated_at=TS,
        intent_id=intents[1].id,
    )
    await bus.publish(Topics.ORDERS, dead)
    assert agent._pending == {}


async def test_exit_rule_closes_an_open_position():
    portfolio = Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    bus, _agent, _, _, intents = await build(portfolio=portfolio)
    portfolio.apply_fill(
        Fill(
            order_id="o",
            symbol=SYM,
            side=Side.BUY,
            qty=7,
            price=2500.0,
            ts=TS,
            product=ProductType.MIS,
        )
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02))  # long rule, but already long
    assert intents == []
    await bus.publish(Topics.features(SYM), fv(trend=-0.005))  # exit_long fires
    assert len(intents) == 1
    assert intents[0].side is Side.SELL and intents[0].qty == 7
    assert intents[0].meta["is_exit"] is True and intents[0].meta["reason"] == "rule:exit_long"


async def test_short_position_exits_with_a_buy():
    portfolio = Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    bus, _, _, _, intents = await build(portfolio=portfolio)
    portfolio.apply_fill(
        Fill(
            order_id="o",
            symbol=SYM,
            side=Side.SELL,
            qty=4,
            price=2500.0,
            ts=TS,
            product=ProductType.MIS,
        )
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert intents[0].side is Side.BUY and intents[0].qty == 4


# --------------------------------------------------------------------------- sizing


async def test_fixed_notional_sizing():
    bus, _, _, _, intents = await build(
        strategy(sizing={"mode": "fixed_notional", "notional": 100_000})
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02, close=2500.0))
    assert intents[0].qty == 40


async def test_a_notional_smaller_than_one_lot_produces_nothing():
    cfg = strategy(
        symbols=["NFO:NIFTY-OCT26"],
        product="NRML",
        sizing={"mode": "fixed_notional", "notional": 1_000_000},
    )
    bus, _, _, _, intents = await build(cfg, lots={"NFO:NIFTY-OCT26": 65})
    await bus.publish(
        Topics.features("NFO:NIFTY-OCT26"),
        fv(trend=0.02, close=25_000.0, symbol="NFO:NIFTY-OCT26"),
    )
    assert intents == []  # 40 shares is less than one 65-lot


async def test_sizing_respects_lot_size_and_max_qty():
    cfg = strategy(
        symbols=["NFO:NIFTY-OCT26"],
        product="NRML",
        sizing={"mode": "fixed_notional", "notional": 5_000_000, "max_qty": 130},
    )
    bus, _, _, _, intents = await build(cfg, lots={"NFO:NIFTY-OCT26": 65})
    await bus.publish(
        Topics.features("NFO:NIFTY-OCT26"),
        fv(trend=0.02, close=25_000.0, symbol="NFO:NIFTY-OCT26"),
    )
    # 5,000,000/25,000 = 200 -> 3 lots of 65 = 195 -> max_qty trims to 2 lots
    assert intents[0].qty == 130


async def test_mcx_sizing_uses_the_contract_multiplier():
    cfg = strategy(
        symbols=["MCX:GOLDM-OCT26"],
        product="NRML",
        sizing={"mode": "fixed_notional", "notional": 3_000_000},
    )
    bus, _, _, _, intents = await build(cfg, lots={"MCX:GOLDM-OCT26": 1})
    await bus.publish(
        Topics.features("MCX:GOLDM-OCT26"),
        fv(trend=0.02, close=150_000.0, symbol="MCX:GOLDM-OCT26"),
    )
    assert intents[0].qty == 2  # 3,000,000 / (150,000 x 10) = 2 lots


async def test_kelly_sizing_passes_probability_through():
    cfg = strategy(
        sizing={"mode": "kelly", "fraction": 0.05},
        model={"name": "m", "edge_scale_bps": 100.0},
        payoff_ratio=2.0,
        rules={},
    )
    models = StubModels(ModelPrediction(score=0.5, version="v1", prob=0.6))
    bus, _, _, _, intents = await build(cfg, models=models)
    await bus.publish(Topics.features(SYM), fv(trend=0.0))
    i = intents[0]
    assert i.qty == 20  # 5% of 1,000,000 = 50,000 / 2,500
    assert i.prob == 0.6 and i.payoff_ratio == 2.0
    assert i.expected_edge_bps == pytest.approx(50.0)  # |score| x edge_scale
    assert i.model_version == "v1"


async def test_zero_size_emits_nothing():
    bus, _, _, _, intents = await build(
        strategy(sizing={"mode": "fixed_notional", "notional": 100})
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02, close=2500.0))
    assert intents == []


# --------------------------------------------------------------------------- models


async def test_model_must_agree_with_the_rule_direction():
    models = StubModels(ModelPrediction(score=-0.8, version="v1"))
    cfg = strategy(model={"name": "m", "min_abs_score": 0.1})
    bus, _, _, _, intents = await build(cfg, models=models)
    await bus.publish(Topics.features(SYM), fv(trend=0.02))  # long rule, bearish model
    assert intents == []
    models.prediction = ModelPrediction(score=0.8, version="v1")
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert len(intents) == 1 and intents[0].side is Side.BUY


async def test_weak_predictions_are_ignored():
    models = StubModels(ModelPrediction(score=0.05, version="v1"))
    bus, _, _, _, intents = await build(
        strategy(model={"name": "m", "min_abs_score": 0.5}), models=models
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert intents == []


async def test_model_only_strategy_trades_on_the_score_and_flips_out():
    portfolio = Portfolio(starting_equity=1_000_000.0, multiplier_for=contract_multiplier)
    models = StubModels(ModelPrediction(score=0.9, version="v1"))
    cfg = strategy(rules={}, model={"name": "m", "min_abs_score": 0.2})
    bus, _, _, _, intents = await build(cfg, models=models, portfolio=portfolio)
    await bus.publish(Topics.features(SYM), fv(trend=0.0))
    assert intents[0].side is Side.BUY and intents[0].meta["reason"] == "model"
    entry = Fill(
        order_id="o",
        symbol=SYM,
        side=Side.BUY,
        qty=10,
        price=2500.0,
        ts=TS,
        product=ProductType.MIS,
    )
    portfolio.apply_fill(entry)
    await bus.publish(Topics.FILLS, entry)  # clears the pending flag
    models.prediction = ModelPrediction(score=-0.9, version="v1")
    await bus.publish(Topics.features(SYM), fv(trend=0.0))
    assert intents[1].side is Side.SELL and intents[1].meta["reason"] == "model:flip"


async def test_a_newly_promoted_version_is_picked_up_without_a_restart():
    """Phase 5 promotion must reach a running signal agent."""
    models = StubModels(ModelPrediction(score=0.9, version="v1"), version="v1")
    bus, agent, _, _, intents = await build(strategy(model={"name": "m"}), models=models)
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert agent._model_version == "v1" and intents[0].model_version == "v1"
    models.version = "v2"  # promote
    await bus.publish(
        Topics.FILLS,
        Fill(
            order_id="o",
            symbol=SYM,
            side=Side.BUY,
            qty=10,
            price=2500.0,
            ts=TS,
            product=ProductType.MIS,
        ),
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert agent._model_version == "v2"
    assert models.calls[-1] == ("m", "v2")


async def test_pinned_version_ignores_promotions():
    models = StubModels(ModelPrediction(score=0.9, version="v1"), version="v9")
    bus, _, _, _, _intents = await build(
        strategy(model={"name": "m", "version": "v1"}), models=models
    )
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert models.calls == [("m", "v1")]


async def test_model_strategy_waits_when_there_is_no_prediction():
    bus, _, _, _, intents = await build(strategy(model={"name": "m"}), models=StubModels(None))
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert intents == []


async def test_unknown_feature_alerts_instead_of_crashing():
    bus, agent, _, _, intents = await build(
        strategy(rules={"long": {"all": [{"feature": "nope", "op": "gt", "value": 0}]}})
    )
    alerts = []

    async def on_alert(_t, m):
        alerts.append(m)

    await bus.subscribe(Topics.ALERTS, on_alert)
    await bus.publish(Topics.features(SYM), fv(trend=0.02))
    assert intents == []
    assert agent.errors == 0  # handled, not an unhandled exception


async def test_a_featureless_strategy_acts_before_warmup():
    cfg = strategy(rules={"long": {"always": True}})
    bus, agent, _, _, intents = await build(cfg)
    await bus.publish(Topics.features(SYM), fv(warm=False))
    assert len(intents) == 1 and agent.skipped_cold == 0


async def test_a_derivative_strategy_without_lot_sizes_fails_at_construction():
    from trading.brokers.lots import MissingLotSize

    with pytest.raises(MissingLotSize, match="NFO:NIFTY-OCT26"):
        await build(strategy(symbols=["NFO:NIFTY-OCT26"], product="NRML"))
