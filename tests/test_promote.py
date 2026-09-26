"""Promotion gate (Phase 5 acceptance: a worse model is refused)."""

from __future__ import annotations

import pytest

from trading.training.promote import PromotionGate, check_gate, promote_if_better
from trading.training.registry import ModelRegistry

from .conftest import PLANTED_SYMBOL

WINDOW = ["2026-07-01 09:15:00+05:30", "2026-07-10 15:29:00+05:30"]
BALANCED = {"a": 0.4, "b": 0.35, "c": 0.25}


def metrics(**kw):
    return {"sharpe": 2.0, "max_drawdown_pct": -0.02, "trades": 100, "window": WINDOW, **kw}


# --------------------------------------------------------------------------- the rules


def test_a_better_candidate_passes():
    d = check_gate(metrics(sharpe=2.5), BALANCED, metrics(sharpe=2.0), PromotionGate())
    assert d.promote and d.failures == []


def test_worse_or_equal_sharpe_is_refused():
    worse = check_gate(metrics(sharpe=1.5), BALANCED, metrics(sharpe=2.0), PromotionGate())
    assert not worse.promote and "does not beat" in worse.failures[0]
    tie = check_gate(metrics(sharpe=2.0), BALANCED, metrics(sharpe=2.0), PromotionGate())
    assert not tie.promote
    margin = check_gate(
        metrics(sharpe=2.2),
        BALANCED,
        metrics(sharpe=2.0),
        PromotionGate(min_sharpe_improvement=0.5),
    )
    assert not margin.promote


def test_first_model_needs_a_positive_sharpe():
    assert check_gate(metrics(sharpe=0.3), BALANCED, None, PromotionGate()).promote
    d = check_gate(metrics(sharpe=-0.1), BALANCED, None, PromotionGate())
    assert not d.promote and "not positive" in d.failures[0]


def test_drawdown_limit():
    d = check_gate(metrics(sharpe=3.0, max_drawdown_pct=-0.25), BALANCED, None, PromotionGate())
    assert not d.promote and "drawdown 25.00%" in d.failures[0]


def test_no_feature_may_carry_more_than_half_the_model():
    d = check_gate(metrics(sharpe=3.0), {"a": 0.6, "b": 0.4}, None, PromotionGate())
    assert not d.promote and "a carries 60.0%" in d.failures[0]
    assert check_gate(metrics(sharpe=3.0), {"a": 0.5, "b": 0.5}, None, PromotionGate()).promote
    assert not check_gate(metrics(sharpe=3.0), {}, None, PromotionGate()).promote


def test_too_few_trades_and_different_windows():
    assert not check_gate(metrics(trades=5), BALANCED, None, PromotionGate()).promote
    other = metrics(sharpe=1.0, window=["2026-06-01", "2026-06-05"])
    d = check_gate(metrics(sharpe=3.0), BALANCED, other, PromotionGate())
    assert not d.promote and "same window" in d.failures[0]


# =========================================================================== acceptance


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models")


def frames(planted_data):
    return {PLANTED_SYMBOL: planted_data.frame}


def test_first_model_is_promoted_when_it_passes(registry, planted_ridge, planted_data):
    v = registry.register("planted", planted_ridge)
    decision = promote_if_better(
        registry, "planted", v, frames(planted_data), planted_data.holdout_start
    )
    assert decision.promote, decision.failures
    assert registry.live_version("planted") == v
    assert decision.live is None and decision.candidate["sharpe"] > 0
    assert registry.decisions("planted", v)[-1]["promote"] is True


def test_the_gate_refuses_a_worse_model(registry, planted_ridge, noise_lgbm, planted_data):
    """Live: a model that learnt the planted signal. Candidate: one fitted to a
    random walk. Scored on the same recent window, the candidate is worse - so the
    live pointer must not move."""
    good = registry.register("planted", planted_ridge)
    registry.set_live("planted", good, reason="baseline")
    worse = registry.register("planted", noise_lgbm)

    decision = promote_if_better(
        registry, "planted", worse, frames(planted_data), planted_data.holdout_start
    )

    assert not decision.promote
    assert any("does not beat" in f for f in decision.failures)
    assert decision.candidate["sharpe"] < decision.live["sharpe"]
    assert decision.candidate["window"] == decision.live["window"]  # the same window
    assert registry.live_version("planted") == good  # untouched
    recorded = registry.decisions("planted", worse)[-1]
    assert recorded["promote"] is False and recorded["live_version"] == good
    assert registry.history("planted")[-1]["event"] == "refused"


def test_a_better_model_replaces_a_worse_one_and_can_be_rolled_back(
    registry, planted_ridge, noise_lgbm, planted_data
):
    weak = registry.register("planted", noise_lgbm)
    registry.set_live("planted", weak, reason="stopgap")
    strong = registry.register("planted", planted_ridge)
    decision = promote_if_better(
        registry, "planted", strong, frames(planted_data), planted_data.holdout_start
    )
    assert decision.promote and registry.live_version("planted") == strong
    assert decision.candidate["sharpe"] > decision.live["sharpe"]
    assert registry.rollback("planted", reason="test") == weak


def test_optional_walk_forward_requirement():
    gate = PromotionGate(min_oos_sharpe=0.0)
    lucky = check_gate(metrics(sharpe=5.0), BALANCED, None, gate, oos_sharpe=-0.2)
    assert not lucky.promote and "walk-forward Sharpe -0.2" in lucky.failures[0]
    assert check_gate(metrics(sharpe=5.0), BALANCED, None, gate, oos_sharpe=0.8).promote
    # off by default: the gate is exactly the three rules plus a minimum trade count
    assert check_gate(metrics(sharpe=5.0), BALANCED, None, PromotionGate(), oos_sharpe=-0.2).promote
