"""Model registry: versions, the live pointer, rollback, and hot reload into a
running signal agent (Phase 5 acceptance)."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from trading.agents.portfolio import Portfolio
from trading.agents.signal import SignalAgent
from trading.core.bus import InMemoryBus, Topics
from trading.core.clock import SimClock
from trading.core.types import (
    IST,
    Bar,
    FeatureVector,
    Fill,
    Interval,
    OrderIntent,
    ProductType,
)
from trading.features.features import FeatureSpec
from trading.strategies.schema import StrategyConfig
from trading.training.registry import ModelRegistry, RegistryError

from .conftest import PLANTED_SYMBOL


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models")


def test_register_versions_and_metadata(registry, planted_ridge, planted_lgbm, planted_data):
    v1 = registry.register("planted", planted_ridge, metrics={"note": "first"})
    v2 = registry.register("planted", planted_lgbm)
    assert (v1, v2) == ("v0001", "v0002") and registry.versions("planted") == ["v0001", "v0002"]
    assert registry.names() == ["planted"]
    meta = registry.metadata("planted", v1)
    assert meta["model"]["kind"] == "ridge" and meta["metrics"] == {"note": "first"}
    assert meta["model"]["train_window"]["end"] < planted_data.holdout_start.isoformat()
    loaded = ModelRegistry(registry.root).load("planted", v1)  # a fresh process
    X = planted_data.holdout.X
    assert np.array_equal(loaded.predict(X), planted_ridge.predict(X))
    assert [e["event"] for e in registry.history("planted")] == ["registered", "registered"]
    with pytest.raises(RegistryError):
        registry.register("Bad Name!", planted_ridge)


def test_live_pointer_and_multi_step_rollback(registry, planted_ridge):
    for _ in range(3):
        registry.register("m", planted_ridge)
    assert registry.live_version("m") is None
    registry.set_live("m", "v0001", reason="first")
    registry.set_live("m", "v0002", reason="better")
    registry.set_live("m", "v0003", reason="better still")
    assert registry.live_info("m")["stack"] == ["v0002", "v0001"]
    assert registry.rollback("m", reason="v3 misbehaving") == "v0002"
    assert registry.rollback("m", reason="v2 too") == "v0001"  # walks back, no ping-pong
    assert registry.live_version("m") == "v0001"
    with pytest.raises(RegistryError, match="nothing to roll back"):
        registry.rollback("m")
    with pytest.raises(RegistryError):
        registry.set_live("m", "v0099", reason="typo")
    events = [(e["event"], e["version"]) for e in registry.history("m")[3:]]
    assert events == [
        ("promoted", "v0001"),
        ("promoted", "v0002"),
        ("promoted", "v0003"),
        ("rolled_back", "v0002"),
        ("rolled_back", "v0001"),
    ]


def test_a_pointer_moved_by_another_process_is_seen_immediately(registry, planted_ridge):
    registry.register("m", planted_ridge)
    registry.register("m", planted_ridge)
    registry.set_live("m", "v0001", reason="a")
    reader = ModelRegistry(registry.root)  # e.g. the running engine
    assert reader.live_version("m") == "v0001"
    ModelRegistry(registry.root).set_live("m", "v0002", reason="nightly job, other process")
    assert reader.live_version("m") == "v0002"


def features_of(planted_data, i: int) -> dict[str, float]:
    return {k: float(v) for k, v in planted_data.holdout.X.iloc[i].items()}


def test_provider_predicts_and_never_raises(registry, planted_ridge, planted_data):
    v = registry.register("m", planted_ridge)
    p = registry.predict("m", v, features_of(planted_data, 10), symbol=PLANTED_SYMBOL)
    assert p is not None and p.version == v and p.prob is not None
    assert registry.predict("m", None, {}) is None
    assert registry.predict("m", "v0404", features_of(planted_data, 10)) is None  # unknown version
    assert registry.predict("m", v, {"ret_1": 0.0}) is None  # missing features: logged, not raised


def test_provider_refuses_a_model_trained_on_other_features(tmp_path, planted_ridge, planted_data):
    writer = ModelRegistry(tmp_path)
    v = writer.register("m", planted_ridge)
    strict = ModelRegistry(tmp_path, expected_spec=FeatureSpec(sma_slow=50))
    assert strict.predict("m", v, features_of(planted_data, 0)) is None
    assert ModelRegistry(tmp_path, expected_spec=planted_ridge.spec).predict(
        "m", v, features_of(planted_data, 0)
    )


# =========================================================================== acceptance


async def test_signal_agent_picks_up_a_newly_promoted_version_without_restart(
    tmp_path, planted_ridge, planted_lgbm, planted_data
):
    engine_side = ModelRegistry(tmp_path)  # what the running engine holds
    nightly_side = ModelRegistry(tmp_path)  # the nightly job, a separate process
    v1 = nightly_side.register("planted", planted_ridge)
    v2 = nightly_side.register("planted", planted_lgbm)
    nightly_side.set_live("planted", v1, reason="initial")

    strategy = StrategyConfig.model_validate(
        {
            "id": "model_driven",
            "symbols": [PLANTED_SYMBOL],
            "interval": "1m",
            "product": "MIS",
            "model": {"name": "planted", "min_abs_score": 0.0},
            "sizing": {"mode": "fixed_qty", "qty": 10},
        }
    )
    bus = InMemoryBus()
    intents: list[OrderIntent] = []

    async def on_intent(_t, m):
        intents.append(m)

    await bus.subscribe(Topics.INTENTS, on_intent)
    portfolio = Portfolio()
    agent = SignalAgent(
        bus, strategy, portfolio, models=engine_side, clock=SimClock(planted_data.holdout_start)
    )
    await agent.start()

    def fv(i: int) -> FeatureVector:
        close = float(planted_data.holdout.meta["close"].iloc[i])
        ts = planted_data.holdout.meta["ts"].iloc[i].to_pydatetime()
        bar = Bar(
            symbol=PLANTED_SYMBOL,
            ts=ts,
            interval=Interval.M1,
            open=close,
            high=close,
            low=close,
            close=close,
        )
        return FeatureVector(
            symbol=PLANTED_SYMBOL,
            ts=ts,
            interval=Interval.M1,
            bar=bar,
            values=features_of(planted_data, i),
            warm=True,
        )

    async def settle(intent: OrderIntent) -> None:
        """Pretend the order filled and was closed, so the agent is flat and free again."""
        for side in (intent.side, intent.side.opposite):
            fill = Fill(
                order_id="x",
                symbol=PLANTED_SYMBOL,
                side=side,
                qty=10,
                price=1000.0,
                ts=datetime(2026, 7, 1, 10, 0, tzinfo=IST),
                product=ProductType.MIS,
            )
            portfolio.apply_fill(fill)
            await bus.publish(Topics.FILLS, fill)

    await bus.publish(Topics.features(PLANTED_SYMBOL), fv(100))
    assert intents[-1].model_version == v1
    await settle(intents[-1])

    nightly_side.set_live("planted", v2, reason="gate passed")  # promoted while running

    await bus.publish(Topics.features(PLANTED_SYMBOL), fv(101))
    assert intents[-1].model_version == v2  # next bar, same agent, no restart
    assert agent._model_version == v2
    # the prediction really came from the new model
    assert intents[-1].expected_edge_bps == pytest.approx(
        planted_lgbm.predict_one(
            features_of(planted_data, 101), v2, PLANTED_SYMBOL
        ).expected_edge_bps
    )
    await settle(intents[-1])

    nightly_side.rollback("planted", reason="changed our minds")
    await bus.publish(Topics.features(PLANTED_SYMBOL), fv(102))
    assert intents[-1].model_version == v1  # rollback reaches it the same way
    json.dumps(nightly_side.live_info("planted"))
