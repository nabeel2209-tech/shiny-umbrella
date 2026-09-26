"""Trading evaluation of model predictions.

Predictions are turned into positions with the **same rule the risk agent applies
live**: direction = sign(prediction); size = the Kelly fraction ``f = p/a - q/b``
(``risk.kelly_fraction``, with p from the model's calibration and b its payoff
ratio), capped at ``kelly_cap``; and nothing at all when the gross edge does not
clear the round-trip cost times ``min_edge_multiple``. The prediction is net of
costs, so the gross edge is ``|prediction| + cost`` - see ``train.py``. Returns, drawdown and
Sharpe then come from ``backtest/metrics.py``, the same code that scores a full
engine backtest.

Positions overlap: a prediction made at bar *t* is a view on the next *h* bars, so
each one is held for *h* bars with 1/h of its size (the standard overlapping-
portfolio construction). Costs are charged on every change of position. Several
symbols share the capital equally. An intraday (MIS) book is flat overnight.

This is a vectorised stand-in for the engine: fast enough to score every fold of
every candidate, faithful to the sizing and cost rules, but without order-level
detail (queue position, partial fills, TTLs). The engine backtest (Phase 4)
remains the final word before any strategy trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading.agents.risk import kelly_fraction
from trading.backtest.metrics import equity_stats
from trading.training.dataset import Dataset
from trading.training.train import TrainedModel, information_coefficient


@dataclass(frozen=True)
class EvalConfig:
    initial_equity: float = 1_000_000.0
    kelly_cap: float = 0.05  # RiskLimits.max_kelly_fraction
    kelly_loss_fraction: float = 0.01  # RiskLimits.kelly_loss_fraction ('a')
    min_edge_multiple: float = 1.5  # RiskLimits.min_edge_multiple
    use_kelly: bool = True  # False: every trade is kelly_cap in size


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    equity: pd.DataFrame  # ts, equity, gross_exposure
    per_fold: dict[int, dict[str, Any]] = field(default_factory=dict)


def target_weights(
    pred: np.ndarray,
    cost: np.ndarray,
    can_trade: np.ndarray,
    cfg: EvalConfig,
    *,
    prob: np.ndarray | None = None,
    payoff: float | None = None,
) -> np.ndarray:
    """Signed fraction of capital each (net-of-cost) prediction asks for."""
    gross_edge = np.abs(pred) + cost
    edge_ok = gross_edge > cost * cfg.min_edge_multiple
    size = np.full(len(pred), cfg.kelly_cap)
    if cfg.use_kelly and prob is not None:
        b = payoff or 1.0
        f = np.array([kelly_fraction(p, b, cfg.kelly_loss_fraction) for p in prob])
        size = np.clip(f, 0.0, cfg.kelly_cap)
    return np.sign(pred) * size * (edge_ok & can_trade)


def simulate(frame: pd.DataFrame, horizon: int, cfg: EvalConfig) -> EvalResult:
    """``frame`` needs ts, symbol, weight, ret_next, cost, gross (one row per bar)."""
    if frame.empty:
        return EvalResult(
            metrics=_empty_metrics(cfg), equity=pd.DataFrame(columns=["ts", "equity"])
        )
    frame = frame.sort_values(["symbol", "ts"]).copy()
    # each signal is held for `horizon` bars at 1/horizon of its size
    frame["position"] = frame.groupby("symbol")["weight"].transform(
        lambda w: w.rolling(horizon, min_periods=1).sum() / horizon
    )
    frame["trade"] = frame.groupby("symbol")["position"].diff().fillna(frame["position"]).abs()
    frame["pnl"] = frame["position"] * frame["ret_next"] - frame["trade"] * frame["cost"] / 2
    n_symbols = frame["symbol"].nunique()
    by_time = frame.groupby("ts").agg(
        pnl=("pnl", "sum"),
        gross_exposure=("position", lambda p: p.abs().sum()),
        turnover=("trade", "sum"),
    )
    ret = by_time["pnl"] / n_symbols
    equity = cfg.initial_equity * (1.0 + ret).cumprod()
    series = pd.Series(equity.to_numpy(), index=pd.DatetimeIndex(by_time.index))
    stats = equity_stats(series, cfg.initial_equity)
    traded = frame[frame["weight"] != 0]
    realised = np.sign(traded["weight"]) * traded["gross"] - traded["cost"]
    metrics = {
        **stats,
        "turnover": round(float(by_time["turnover"].sum() / n_symbols), 4),
        "trades": len(traded),
        "hit_rate": round(float((realised > 0).mean()), 4) if len(traded) else None,
        "exposure": round(float((by_time["gross_exposure"] > 0).mean()), 4),
        "avg_gross_exposure": round(float(by_time["gross_exposure"].mean() / n_symbols), 6),
        "bars": len(by_time),
        "window": [str(by_time.index.min()), str(by_time.index.max())],
    }
    out = pd.DataFrame(
        {
            "ts": by_time.index,
            "equity": equity.to_numpy(),
            "gross_exposure": by_time["gross_exposure"].to_numpy(),
        }
    )
    return EvalResult(metrics=metrics, equity=out)


def _empty_metrics(cfg: EvalConfig) -> dict[str, Any]:
    return {
        **equity_stats(pd.Series(dtype=float), cfg.initial_equity),
        "turnover": 0.0,
        "trades": 0,
        "hit_rate": None,
        "exposure": 0.0,
        "avg_gross_exposure": 0.0,
        "bars": 0,
        "window": [None, None],
    }


def evaluate_model(model: TrainedModel, ds: Dataset, cfg: EvalConfig | None = None) -> EvalResult:
    """Predict every row of ``ds`` with ``model`` and trade the predictions."""
    cfg = cfg or EvalConfig()
    if len(ds) == 0:
        return EvalResult(
            metrics=_empty_metrics(cfg), equity=pd.DataFrame(columns=["ts", "equity"])
        )
    pred = model.predict(ds.X)
    prob = model.prob_correct(pred)
    frame = ds.meta[["ts", "symbol", "ret_next", "cost", "gross"]].copy()
    frame["weight"] = target_weights(
        pred,
        ds.meta["cost"].to_numpy(),
        ds.y.notna().to_numpy(),  # no new trade where the horizon cannot complete
        cfg,
        prob=prob,
        payoff=model.payoff_ratio,
    )
    result = simulate(frame, model.label.horizon, cfg)
    lab = ds.labelled
    result.metrics["ic"] = round(information_coefficient(pred[lab], ds.y.to_numpy()[lab]), 5)
    result.metrics["importances"] = model.importances
    return result


def evaluate_oos(model: TrainedModel, cfg: EvalConfig | None = None) -> EvalResult:
    """Trade the out-of-fold predictions kept from training, fold by fold and pooled."""
    cfg = cfg or EvalConfig()
    oos = model.oos
    prob = model.prob_correct(oos["pred"].to_numpy())
    frame = oos[["ts", "symbol", "ret_next", "cost", "gross", "fold"]].copy()
    frame["weight"] = target_weights(
        oos["pred"].to_numpy(),
        oos["cost"].to_numpy(),
        np.ones(len(oos), dtype=bool),
        cfg,
        prob=prob,
        payoff=model.payoff_ratio,
    )
    pooled = simulate(frame.drop(columns="fold"), model.label.horizon, cfg)
    pooled.metrics["ic"] = model.cv.get("oos_ic")
    pooled.metrics["importances"] = model.importances
    for fold, part in frame.groupby("fold"):
        pooled.per_fold[int(fold)] = simulate(
            part.drop(columns="fold"), model.label.horizon, cfg
        ).metrics
    return pooled
