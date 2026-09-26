"""Training: the Phase 5 acceptance test (a planted signal is recovered), plus
hyperparameter selection, weights, determinism and inference."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading.training.dataset import build_dataset
from trading.training.labels import LabelSpec
from trading.training.train import (
    ModelKind,
    TrainConfig,
    fit_model,
    information_coefficient,
    sample_weights,
)

from .conftest import PLANTED_SYMBOL

# =========================================================================== acceptance


@pytest.mark.parametrize("model_fixture", ["planted_ridge", "planted_lgbm"])
def test_a_planted_signal_is_recovered(model_fixture, planted_data, request):
    """The price mean-reverts to its 30-bar average, so the next move is driven by
    dist_sma_30. Both model kinds must find that, out of sample."""
    model = request.getfixturevalue(model_fixture)
    assert model.cv["oos_ic"] > 0.2  # walk-forward validation folds
    holdout_pred = model.predict(planted_data.holdout.X)
    lab = planted_data.holdout.labelled
    holdout_ic = information_coefficient(holdout_pred[lab], planted_data.holdout.y.to_numpy()[lab])
    assert holdout_ic > 0.2  # eight sessions it never saw
    assert next(iter(model.importances)) == "dist_sma_30"  # the planted driver ranks first
    assert model.direction_accuracy > 0.55


def test_ridge_recovers_the_direction_of_the_planted_effect(planted_ridge):
    coef = dict(
        zip(
            planted_ridge.feature_names,
            planted_ridge.estimator.named_steps["ridge"].coef_,
            strict=True,
        )
    )
    assert coef["dist_sma_30"] < 0  # above its average, the price falls: mean reversion


def test_a_random_walk_yields_no_skill(noise_lgbm, noise_data):
    assert abs(noise_lgbm.cv["oos_ic"]) < 0.08
    pred = noise_lgbm.predict(noise_data.holdout.X)
    lab = noise_data.holdout.labelled
    assert abs(information_coefficient(pred[lab], noise_data.holdout.y.to_numpy()[lab])) < 0.08


# =========================================================================== protocol


def test_hyperparameters_are_chosen_on_validation_folds_only(planted_ridge, planted_data):
    cv = planted_ridge.cv
    assert len(cv["candidates"]) == len(TrainConfig().ridge_alphas)
    best = max(cv["candidates"], key=lambda c: c["mean_ic"])
    assert cv["chosen"] == best["params"] == planted_ridge.params
    # no fold ever touched the holdout
    last_validation = max(pd.Timestamp(f["validation_end"]) for f in cv["folds"])
    assert last_validation < pd.Timestamp(planted_data.holdout_start)
    for fold in cv["folds"]:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
        assert fold["purge"] >= 10  # the label horizon
    # the model was fitted on the training part only
    assert pd.Timestamp(planted_ridge.train_window[1]) < pd.Timestamp(planted_data.holdout_start)


def test_lightgbm_never_bags_rows(planted_lgbm):
    params = planted_lgbm.estimator.get_params()
    assert params["subsample"] == 1.0 and params["deterministic"] is True


def test_training_is_deterministic(planted_data):
    cfg = TrainConfig(kind=ModelKind.LIGHTGBM, lgbm_grid=(TrainConfig().lgbm_grid[0],))
    a, b = fit_model(planted_data.train, cfg), fit_model(planted_data.train, cfg)
    X = planted_data.holdout.X
    assert np.array_equal(a.predict(X), b.predict(X))


def test_sample_weights_combine_uniqueness_and_recency(planted_data):
    ds = planted_data.train
    rows = ds.labelled
    recent = sample_weights(ds, rows, TrainConfig(use_uniqueness=False, recency_half_life_days=5))
    assert recent.mean() == pytest.approx(1.0)
    assert recent[-1] > recent[0] * 4  # three weeks old at a 5-day half-life
    unique = sample_weights(ds, rows, TrainConfig(recency_half_life_days=None))
    uniq = ds.meta["uniqueness"].to_numpy()[rows]
    assert np.allclose(unique, uniq / uniq.mean())
    flat = sample_weights(ds, rows, TrainConfig(use_uniqueness=False, recency_half_life_days=None))
    assert np.allclose(flat, 1.0)


def test_too_few_rows(planted_data):
    with pytest.raises(ValueError, match="labelled rows"):
        fit_model(planted_data.train.take(np.arange(300)), TrainConfig(min_train=500))


# =========================================================================== inference


def test_predict_one_matches_batch_and_speaks_the_risk_agents_language(planted_ridge, planted_data):
    row = planted_data.holdout.X.iloc[123]
    features = {k: float(v) for k, v in row.items()}
    p = planted_ridge.predict_one(features, "v0007", symbol=PLANTED_SYMBOL)
    assert p.version == "v0007"
    assert p.score == pytest.approx(planted_ridge.predict(planted_data.holdout.X.iloc[[123]])[0])
    # net prediction -> gross edge for the risk agent's cost threshold
    cost = planted_ridge.cost(PLANTED_SYMBOL)
    assert 5e-4 < cost < 2e-3
    assert p.expected_edge_bps == pytest.approx((abs(p.score) + cost) * 10_000)
    assert 0.0 < p.prob < 1.0 and p.payoff_ratio == planted_ridge.payoff_ratio
    assert p.horizon == 10  # the model says how long its view lasts
    with pytest.raises(KeyError, match="dist_sma_30"):
        planted_ridge.predict_one({k: v for k, v in features.items() if k != "dist_sma_30"}, "v1")


def test_calibration_is_monotone_for_a_real_signal(planted_ridge):
    p = planted_ridge.prob_correct(np.array([0.0, 0.0005, 0.002]))
    assert p[0] < p[1] < p[2]  # bigger predictions are more often right


def test_metadata_is_plain_json(planted_lgbm):
    meta = planted_lgbm.metadata()
    json.dumps(meta)
    assert meta["kind"] == "lightgbm" and meta["label"]["horizon"] == 10
    assert meta["feature_names"] == planted_lgbm.feature_names
    assert abs(sum(meta["importances"].values()) - 1.0) < 1e-3
    assert meta["cost_by_symbol"][PLANTED_SYMBOL] > 0


def test_labels_are_net_of_costs_in_the_dataset(planted_data):
    """Constraint 8: the training target is the move left after a round trip."""
    ds = planted_data.dataset
    lab = ds.labelled
    gross, net, cost = (
        ds.meta["gross"].to_numpy()[lab],
        ds.y.to_numpy()[lab],
        ds.meta["cost"].to_numpy()[lab],
    )
    small = np.abs(gross) <= cost
    assert np.all(net[small] == 0)
    assert np.allclose(np.abs(net[~small]), np.abs(gross[~small]) - cost[~small])


def test_datasets_stack_symbols_in_time_order(calendar):
    from datetime import date

    from .planted import planted_frame

    a = planted_frame(calendar, date(2026, 6, 1), 3, seed=4)
    b = planted_frame(calendar, date(2026, 6, 1), 3, seed=5)
    ds = build_dataset({"NSE:AAA": a, "NSE:BBB": b}, "1m", label=LabelSpec(horizon=5))
    ts = ds.meta["ts"]
    assert ts.is_monotonic_increasing
    assert set(ds.symbols) == {"NSE:AAA", "NSE:BBB"}
    assert (ds.meta.groupby("ts")["symbol"].nunique() == 2).all()
