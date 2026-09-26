"""Trading evaluation: the risk agent's sizing rule applied to predictions."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading.core.types import IST
from trading.training.evaluate import (
    EvalConfig,
    evaluate_model,
    evaluate_oos,
    simulate,
    target_weights,
)

T0 = datetime(2026, 9, 18, 10, 0, tzinfo=IST)


def test_weights_follow_the_risk_agents_rule():
    cfg = EvalConfig(kelly_cap=0.05, min_edge_multiple=1.5, use_kelly=False)
    cost = np.full(5, 0.001)
    pred = np.array([0.0002, 0.0008, -0.0008, 0.003, 0.003])
    can = np.array([True, True, True, True, False])
    w = target_weights(pred, cost, can, cfg)
    # gross edge = |pred| + cost must beat 1.5 x cost, i.e. |pred| > 0.0005
    assert list(w) == [0.0, 0.05, -0.05, 0.05, 0.0]  # the last cannot complete its horizon


def test_kelly_sizing_and_refusal():
    cfg = EvalConfig(kelly_cap=0.05, kelly_loss_fraction=0.01)
    pred = np.array([0.002, 0.002])
    w = target_weights(
        pred, np.full(2, 0.001), np.ones(2, bool), cfg, prob=np.array([0.7, 0.3]), payoff=1.0
    )
    assert w[0] == pytest.approx(0.05)  # f = 0.7/0.01 - 0.3/0.01 = 40, capped
    assert w[1] == 0.0  # p = 0.3: negative Kelly, no bet


def frame(weights, returns, cost=0.0, symbol="NSE:X"):
    n = len(weights)
    return pd.DataFrame(
        {
            "ts": [T0 + timedelta(minutes=i) for i in range(n)],
            "symbol": symbol,
            "weight": weights,
            "ret_next": returns,
            "cost": cost,
            "gross": returns,
        }
    )


def test_simulation_by_hand():
    cfg = EvalConfig(initial_equity=100_000.0)
    result = simulate(frame([0.1, 0.1, 0.0], [0.01, -0.02, 0.05]), horizon=1, cfg=cfg)
    equity = result.equity["equity"].tolist()
    assert equity[0] == pytest.approx(100_000 * 1.001)
    assert equity[1] == pytest.approx(100_000 * 1.001 * 0.998)
    assert equity[2] == pytest.approx(equity[1])  # flat on the last bar
    assert result.metrics["trades"] == 2 and result.metrics["turnover"] == pytest.approx(0.2)


def test_positions_overlap_over_the_horizon():
    cfg = EvalConfig(initial_equity=100_000.0)
    result = simulate(frame([0.1, 0.0, 0.0, 0.0], [0.0, 0.01, 0.01, 0.01]), horizon=2, cfg=cfg)
    # the one signal is held for two bars at half size each
    eq = result.equity["equity"].to_numpy()
    assert eq[1] / eq[0] - 1 == pytest.approx(0.05 * 0.01)
    assert eq[2] == pytest.approx(eq[1])


def test_costs_are_charged_on_position_changes():
    cfg = EvalConfig(initial_equity=100_000.0)
    free = simulate(frame([0.1, -0.1, 0.1], [0.0, 0.0, 0.0]), horizon=1, cfg=cfg)
    costly = simulate(frame([0.1, -0.1, 0.1], [0.0, 0.0, 0.0], cost=0.002), horizon=1, cfg=cfg)
    assert free.metrics["net_pnl"] == 0.0
    # turnover 0.1 + 0.2 + 0.2 = 0.5 of capital, half a round trip each unit
    expected = 100_000 * (1 - 0.0001) * (1 - 0.0002) ** 2
    assert costly.metrics["final_equity"] == pytest.approx(expected, abs=0.01)  # 2-dp rounding


def test_symbols_share_capital_equally():
    cfg = EvalConfig(initial_equity=100_000.0)
    both = pd.concat([frame([0.1], [0.01], symbol="NSE:A"), frame([0.1], [-0.01], symbol="NSE:B")])
    assert simulate(both, horizon=1, cfg=cfg).metrics["net_pnl"] == pytest.approx(0.0)


def test_planted_model_trades_profitably_out_of_sample(planted_ridge, planted_data):
    result = evaluate_model(planted_ridge, planted_data.holdout)
    m = result.metrics
    assert m["sharpe"] > 1.0 and m["net_pnl"] > 0 and m["trades"] > 100
    assert m["hit_rate"] > 0.5 and m["ic"] > 0.2
    assert m["max_drawdown_pct"] > -0.05
    assert m["window"][0].startswith(str(planted_data.holdout_start.date()))
    assert m["importances"] == planted_ridge.importances


def test_noise_model_does_not_make_money(noise_lgbm, noise_data):
    m = evaluate_model(noise_lgbm, noise_data.holdout).metrics
    assert m["sharpe"] <= 0.5  # nothing to find; costs make any trading a loss


def test_out_of_fold_evaluation_reports_every_fold(planted_ridge):
    result = evaluate_oos(planted_ridge)
    assert set(result.per_fold) == set(range(len(planted_ridge.cv["folds"])))
    assert result.metrics["ic"] == planted_ridge.cv["oos_ic"]
    assert result.metrics["sharpe"] > 0


def test_empty_inputs(planted_ridge, planted_data):
    empty = planted_data.holdout.take(np.array([], dtype=int))
    m = evaluate_model(planted_ridge, empty).metrics
    assert m["trades"] == 0 and m["sharpe"] == 0.0
