"""Model training: ridge regression or LightGBM on walk-forward folds.

Hyperparameters (ridge's lambda, LightGBM's tree settings) are chosen **only** on
validation folds from :func:`splits.walk_forward`: every candidate is fitted on
each fold's training rows and scored on that fold's validation rows by the
information coefficient (Spearman correlation of prediction and label), and the
best mean wins. The final model is then fitted on every labelled row it was given.
Nothing is shuffled; LightGBM runs deterministically without row bagging.

Sample weights are ``uniqueness x recency``: labels that share most of their
horizon with neighbours count for less (see ``labels.label_uniqueness``), and old
rows decay with a half-life in days.

Predictions are of the **net** label (after a round trip), so they are not what
the risk agent's cost threshold expects: that rule compares a *gross* edge with
the round-trip cost times a multiple. :meth:`TrainedModel.gross_edge` converts back
(net + the cost the label was built with) so the one rule applies unchanged, and
it reduces to "trade when the predicted net edge beats (multiple - 1) x cost"
instead of charging the costs twice.

Out-of-fold predictions of the chosen settings are kept on the model. They are
honest (each came from a model that never saw that row) and serve three purposes:
the trading evaluation in ``evaluate.py``, a logistic calibration of
P(direction right | |prediction|) for Kelly sizing, and the payoff ratio b.

Models are pickled with joblib. Only load registry files this platform wrote.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading import __version__
from trading.agents.signal import ModelPrediction
from trading.core.types import Interval, now_ist
from trading.features.features import FeatureSpec
from trading.training.dataset import Dataset
from trading.training.labels import LabelSpec
from trading.training.splits import walk_forward

log = logging.getLogger(__name__)


class ModelKind(StrEnum):
    RIDGE = "ridge"
    LIGHTGBM = "lightgbm"


DEFAULT_LGBM_GRID: tuple[dict[str, Any], ...] = (
    {"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 150, "min_child_samples": 100},
    {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 250, "min_child_samples": 200},
    {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 300, "min_child_samples": 300},
)


@dataclass
class TrainConfig:
    kind: ModelKind = ModelKind.RIDGE
    n_folds: int = 5
    embargo: int = 0  # extra bars between training and validation, beyond the purge
    min_train: int = 500
    expanding: bool = True
    train_size: int | None = None
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    lgbm_grid: tuple[dict[str, Any], ...] = DEFAULT_LGBM_GRID
    recency_half_life_days: float | None = 60.0
    use_uniqueness: bool = True
    seed: int = 7

    def candidates(self) -> list[dict[str, Any]]:
        if self.kind is ModelKind.RIDGE:
            return [{"alpha": a} for a in self.ridge_alphas]
        return [dict(p) for p in self.lgbm_grid]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["lgbm_grid"] = [dict(p) for p in self.lgbm_grid]
        return d


# --------------------------------------------------------------------------- estimators


def make_estimator(kind: ModelKind, params: dict[str, Any], seed: int) -> Any:
    if kind is ModelKind.RIDGE:
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=params["alpha"]))])
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        **params,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        subsample=1.0,  # no row bagging: training rows are used as given, in time order
        colsample_bytree=1.0,
        n_jobs=1,
        verbose=-1,
    )


def fit_estimator(est: Any, kind: ModelKind, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
    if kind is ModelKind.RIDGE:
        est.fit(X, y, scale__sample_weight=w, ridge__sample_weight=w)
    else:
        est.fit(X, y, sample_weight=w)
    return est


def information_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation; 0 when either side is constant.

    Constancy is tested with the range, not the standard deviation: a tree model
    that predicts one value everywhere leaves a std of ~1e-21 from float noise.
    """
    if len(pred) < 3 or np.ptp(pred) == 0 or np.ptp(target) == 0:
        return 0.0
    ic = pd.Series(pred).rank().corr(pd.Series(target).rank())
    return float(ic) if math.isfinite(ic) else 0.0


def sample_weights(ds: Dataset, rows: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    w = np.ones(len(rows))
    if cfg.use_uniqueness:
        w *= ds.meta["uniqueness"].to_numpy()[rows]
    if cfg.recency_half_life_days:
        ts = ds.meta["ts"].iloc[rows]
        age = (ts.max() - ts).dt.total_seconds().to_numpy() / 86_400
        w *= 0.5 ** (age / cfg.recency_half_life_days)
    w = np.where(w > 0, w, 0.0)
    return w / w.mean() if w.mean() > 0 else np.ones(len(rows))


# --------------------------------------------------------------------------- the model


@dataclass
class TrainedModel:
    kind: ModelKind
    params: dict[str, Any]
    estimator: Any
    feature_names: list[str]
    spec: FeatureSpec
    label: LabelSpec
    interval: Interval
    symbols: list[str]
    train_window: tuple[str | None, str | None]
    n_train: int
    cv: dict[str, Any]
    oos: pd.DataFrame  # out-of-fold predictions: ts, symbol, fold, y, gross, cost, ret_next, pred
    calibrator: Any | None = None
    calibration_scale: float = 1.0
    payoff_ratio: float | None = None
    direction_accuracy: float | None = None
    importances: dict[str, float] = field(default_factory=dict)
    cost_by_symbol: dict[str, float] = field(default_factory=dict)  # label round-trip cost
    trained_at: str = field(default_factory=lambda: now_ist().isoformat())
    platform_version: str = __version__

    # ------------------------------------------------------------------ inference
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = X[self.feature_names].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else X
        return np.asarray(self.estimator.predict(arr), dtype=float)

    def prob_correct(self, abs_pred: np.ndarray) -> np.ndarray | None:
        """P(direction is right | |prediction|), from the out-of-fold calibration."""
        if self.calibrator is None:
            return None
        x = (np.abs(abs_pred) / self.calibration_scale).reshape(-1, 1)
        return self.calibrator.predict_proba(x)[:, 1]

    def cost(self, symbol: str | None = None) -> float:
        """Round-trip cost fraction the labels were built with."""
        if symbol is not None and symbol in self.cost_by_symbol:
            return self.cost_by_symbol[symbol]
        costs = list(self.cost_by_symbol.values())
        return float(np.median(costs)) if costs else 0.0

    def gross_edge(self, pred: float, symbol: str | None = None) -> float:
        """The gross edge a net-of-cost prediction implies (see module docstring)."""
        return abs(pred) + self.cost(symbol)

    def predict_one(
        self, features: dict[str, float], version: str, symbol: str | None = None
    ) -> ModelPrediction:
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise KeyError(f"model needs features the data agent did not send: {missing}")
        row = np.array([[features[n] for n in self.feature_names]], dtype=float)
        score = float(self.predict(row)[0])
        prob = self.prob_correct(np.array([score]))
        return ModelPrediction(
            score=score,
            version=version,
            prob=float(prob[0]) if prob is not None else None,
            expected_edge_bps=self.gross_edge(score, symbol) * 10_000,
            payoff_ratio=self.payoff_ratio,
            horizon=self.label.horizon,
        )

    # ------------------------------------------------------------------ description
    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_spec": asdict(self.spec),
            "label": self.label.to_dict(),
            "interval": self.interval.value,
            "symbols": self.symbols,
            "train_window": {"start": self.train_window[0], "end": self.train_window[1]},
            "n_train": self.n_train,
            "cv": self.cv,
            "payoff_ratio": self.payoff_ratio,
            "direction_accuracy": self.direction_accuracy,
            "calibrated": self.calibrator is not None,
            "importances": self.importances,
            "cost_by_symbol": self.cost_by_symbol,
            "trained_at": self.trained_at,
            "platform_version": self.platform_version,
        }


def feature_importances(model: Any, kind: ModelKind, names: list[str]) -> dict[str, float]:
    """Shares that sum to 1: |coefficient| on standardised features for ridge, total
    split gain for LightGBM."""
    if kind is ModelKind.RIDGE:
        raw = np.abs(model.named_steps["ridge"].coef_)
    else:
        raw = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    total = raw.sum()
    shares = raw / total if total > 0 else np.zeros_like(raw)
    return dict(
        sorted(zip(names, (round(float(s), 6) for s in shares), strict=True), key=lambda kv: -kv[1])
    )


def _calibrate(oos: pd.DataFrame) -> tuple[Any | None, float, float | None]:
    """Logistic P(right direction | |pred|) on out-of-fold rows with a real move."""
    rows = oos[(oos["pred"] != 0) & (oos["gross"] != 0)]
    if len(rows) < 50:
        return None, 1.0, None
    right = (np.sign(rows["pred"]) == np.sign(rows["gross"])).to_numpy().astype(int)
    accuracy = float(right.mean())
    if right.min() == right.max():
        return None, 1.0, accuracy
    scale = float(rows["pred"].abs().std()) or 1.0
    x = (rows["pred"].abs().to_numpy() / scale).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = LogisticRegression(C=1.0).fit(x, right)
    return cal, scale, accuracy


def _payoff(oos: pd.DataFrame) -> float | None:
    realised = np.sign(oos["pred"]) * oos["gross"]
    wins, losses = realised[realised > 0], realised[realised < 0]
    if not len(wins) or not len(losses):
        return None
    return round(float(wins.mean() / -losses.mean()), 4)


def fit_model(ds: Dataset, cfg: TrainConfig | None = None) -> TrainedModel:
    """Choose hyperparameters on walk-forward folds, then fit on all labelled rows."""
    cfg = cfg or TrainConfig()
    lab = ds.labelled
    if len(lab) < cfg.min_train:
        raise ValueError(f"only {len(lab)} labelled rows; need at least {cfg.min_train}")
    X_all = ds.X.to_numpy(dtype=float)
    y_all = ds.y.to_numpy(dtype=float)
    horizon = ds.label.horizon
    folds = walk_forward(
        ds.timestamps.iloc[lab],
        n_folds=cfg.n_folds,
        purge=horizon,
        embargo=cfg.embargo,
        horizon=horizon,
        min_train=cfg.min_train,
        expanding=cfg.expanding,
        train_size=cfg.train_size,
    )
    scores: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    oos_by_candidate: list[list[pd.DataFrame]] = []
    for c, params in enumerate(cfg.candidates()):
        fold_ics, fold_parts = [], []
        for fold in folds:
            tr, va = lab[fold.train], lab[fold.validation]
            est = fit_estimator(
                make_estimator(cfg.kind, params, cfg.seed),
                cfg.kind,
                X_all[tr],
                y_all[tr],
                sample_weights(ds, tr, cfg),
            )
            pred = np.asarray(est.predict(X_all[va]), dtype=float)
            fold_ics.append(information_coefficient(pred, y_all[va]))
            part = ds.meta.iloc[va][["ts", "symbol", "gross", "cost", "ret_next"]].copy()
            part["fold"], part["y"], part["pred"] = fold.index, y_all[va], pred
            fold_parts.append(part)
        mean_ic = float(np.mean(fold_ics))
        scores.append(
            {
                "params": params,
                "fold_ic": [round(x, 5) for x in fold_ics],
                "mean_ic": round(mean_ic, 5),
            }
        )
        oos_by_candidate.append(fold_parts)
        if best is None or mean_ic > best[0] + 1e-12:
            best = (mean_ic, c)
    assert best is not None
    chosen = cfg.candidates()[best[1]]
    oos = pd.concat(oos_by_candidate[best[1]], ignore_index=True)

    final = fit_estimator(
        make_estimator(cfg.kind, chosen, cfg.seed),
        cfg.kind,
        X_all[lab],
        y_all[lab],
        sample_weights(ds, lab, cfg),
    )
    calibrator, scale, accuracy = _calibrate(oos)
    window = (ds.meta["ts"].iloc[lab].min().isoformat(), ds.meta["ts"].iloc[lab].max().isoformat())
    model = TrainedModel(
        kind=cfg.kind,
        params=chosen,
        estimator=final,
        feature_names=ds.feature_names,
        spec=ds.spec,
        label=ds.label,
        interval=ds.interval,
        symbols=ds.symbols,
        train_window=window,
        n_train=len(lab),
        cv={
            "selection": "mean validation IC over walk-forward folds",
            "folds": [f.describe() for f in folds],
            "candidates": scores,
            "chosen": chosen,
            "oos_ic": round(
                information_coefficient(oos["pred"].to_numpy(), oos["y"].to_numpy()), 5
            ),
            "config": cfg.to_dict(),
        },
        oos=oos,
        calibrator=calibrator,
        calibration_scale=scale,
        payoff_ratio=_payoff(oos),
        direction_accuracy=round(accuracy, 4) if accuracy is not None else None,
        importances=feature_importances(final, cfg.kind, ds.feature_names),
        cost_by_symbol={
            str(s): float(c) for s, c in ds.meta.groupby("symbol")["cost"].first().items()
        },
    )
    log.info(
        "trained %s %s on %d rows: oos IC %.4f, chosen %s",
        cfg.kind.value,
        ds.symbols,
        len(lab),
        model.cv["oos_ic"],
        chosen,
    )
    return model
