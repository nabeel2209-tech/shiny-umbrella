"""Strategy YAML: parsing, validation and rule evaluation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading.core.types import Interval, ProductType, Urgency
from trading.features.features import feature_names
from trading.strategies.schema import (
    Condition,
    Op,
    RuleGroup,
    SizingMode,
    StrategyConfig,
    load_strategies,
)

EXAMPLES = "trading/strategies/examples"

MINIMAL = """
id: demo
symbols: [NSE:RELIANCE]
interval: 5m
product: MIS
expected_edge_bps: 20
rules:
  long:
    all:
      - {feature: trend, op: gt, value: 0.0}
"""


def test_minimal_yaml_round_trip():
    s = StrategyConfig.from_yaml_str(MINIMAL)
    assert s.id == "demo" and s.interval is Interval.M5 and s.product is ProductType.MIS
    assert s.sizing.mode is SizingMode.FIXED_QTY and s.execution.urgency is Urgency.NORMAL
    assert not s.model.enabled
    assert StrategyConfig.from_yaml_str(s.to_yaml()) == s


def test_shipped_examples_are_valid_and_use_real_features():
    strategies = load_strategies(EXAMPLES)
    assert [s.id for s in strategies] == ["goldm_meanrev", "trend_reliance"]
    available = set(feature_names()) | set(feature_names(interval=Interval.D1))
    for s in strategies:
        assert s.validate_features(available) == []
        assert s.expected_edge_bps > 0  # must be able to clear its costs


def test_condition_operators():
    f = {"a": 2.0, "b": -3.0}
    assert Condition(feature="a", op=Op.GT, value=1).evaluate(f)
    assert not Condition(feature="a", op=Op.GT, value=2).evaluate(f)
    assert Condition(feature="a", op=Op.GTE, value=2).evaluate(f)
    assert Condition(feature="b", op=Op.LT, value=0).evaluate(f)
    assert Condition(feature="a", op=Op.LTE, value=2).evaluate(f)
    assert Condition(feature="a", op=Op.EQ, value=2).evaluate(f)
    assert Condition(feature="a", op=Op.NEQ, value=3).evaluate(f)
    assert Condition(feature="b", op=Op.ABS_GT, value=2).evaluate(f)
    assert Condition(feature="a", op=Op.ABS_LT, value=3).evaluate(f)
    assert Condition(feature="a", op=Op.GT, other="b").evaluate(f)
    with pytest.raises(KeyError):
        Condition(feature="missing", op=Op.GT, value=0).evaluate(f)


def test_condition_needs_exactly_one_right_hand_side():
    with pytest.raises(ValidationError):
        Condition(feature="a")
    with pytest.raises(ValidationError):
        Condition(feature="a", value=1, other="b")
    with pytest.raises(ValidationError):
        Condition(feature="a", value=1, nonsense=2)  # extra fields are rejected


def test_rule_group_logic():
    f = {"a": 2.0, "b": -3.0}
    assert RuleGroup(all=[Condition(feature="a", op=Op.GT, value=1)]).evaluate(f)
    assert not RuleGroup(
        all=[Condition(feature="a", op=Op.GT, value=1), Condition(feature="b", op=Op.GT, value=0)]
    ).evaluate(f)
    assert RuleGroup(
        any=[Condition(feature="a", op=Op.GT, value=99), Condition(feature="b", op=Op.LT, value=0)]
    ).evaluate(f)
    assert RuleGroup(none=[Condition(feature="a", op=Op.GT, value=99)]).evaluate(f)
    assert not RuleGroup(none=[Condition(feature="a", op=Op.GT, value=1)]).evaluate(f)
    # nesting, and all/any combined must both hold
    nested = RuleGroup(
        all=[Condition(feature="a", op=Op.GT, value=1)],
        any=[
            RuleGroup(all=[Condition(feature="b", op=Op.LT, value=-5)]),
            RuleGroup(all=[Condition(feature="b", op=Op.LT, value=0)]),
        ],
    )
    assert nested.evaluate(f)
    assert nested.features_used() == {"a", "b"}
    with pytest.raises(ValidationError):
        RuleGroup()


def test_strategy_validation_rules():
    with pytest.raises(ValidationError):  # no rules and no model
        StrategyConfig(id="x", symbols=["NSE:RELIANCE"])
    with pytest.raises(ValidationError, match="EXCHANGE:NAME"):  # SymbolError is a ValueError
        StrategyConfig.from_yaml_str(MINIMAL.replace("NSE:RELIANCE", "NOPE"))
    with pytest.raises(ValidationError):  # CNC on a derivative
        StrategyConfig.from_yaml_str(
            MINIMAL.replace("NSE:RELIANCE", "NFO:NIFTY-OCT26").replace("MIS", "CNC")
        )
    with pytest.raises(ValidationError):  # id must be a slug
        StrategyConfig.from_yaml_str(MINIMAL.replace("id: demo", "id: Demo Strategy!"))
    # a model-only strategy needs no rules
    model_only = StrategyConfig.from_yaml_str(
        MINIMAL.split("rules:")[0] + "model: {name: ridge_v1, min_abs_score: 0.2}"
    )
    assert model_only.model.enabled and model_only.rules == {}


def test_unknown_features_are_reported():
    s = StrategyConfig.from_yaml_str(MINIMAL.replace("trend", "made_up_feature"))
    assert s.validate_features(feature_names()) == ["made_up_feature"]


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(MINIMAL)
    (tmp_path / "b.yaml").write_text(MINIMAL)
    with pytest.raises(ValueError, match="duplicate strategy ids"):
        load_strategies(tmp_path)
    assert load_strategies(tmp_path / "nothing-here") == []


def test_always_rule_and_feature_needs():
    hold = StrategyConfig.model_validate(
        {
            "id": "hold",
            "symbols": ["NSE:NIFTYBEES"],
            "product": "CNC",
            "expected_edge_bps": 100,
            "rules": {"long": {"always": True}},
        }
    )
    assert hold.rules["long"].evaluate({}) is True
    assert hold.features_used() == set()
    assert hold.needs_features is False  # can act before any feature is warm
    assert StrategyConfig.from_yaml_str(MINIMAL).needs_features is True
    assert RuleGroup(always=True, all=[Condition(feature="a", value=0)]).evaluate({"a": 1.0})
    assert not RuleGroup(always=True, all=[Condition(feature="a", value=0)]).evaluate({"a": -1.0})
