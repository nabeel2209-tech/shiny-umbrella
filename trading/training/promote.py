"""Promotion gate and rollback.

A candidate replaces the live model only if, on the **same recent window**:

1. its Sharpe beats the live model's by at least ``min_sharpe_improvement`` (and,
   with no live model yet, is positive);
2. its maximum drawdown stays within ``max_drawdown_pct``;
3. no single feature carries more than ``max_feature_importance`` of the model -
   a model that leans on one input is one data glitch away from nonsense;
4. it traded at least ``min_trades`` times, so the Sharpe means something.

The window is the holdout the candidate never saw. Every model is deployed exactly
as it was evaluated (no refit on the holdout), so the live model - trained on a
window that ended before its own holdout - has not seen this one either, and the
comparison is out of sample for both. Each is scored with its own feature spec and
label horizon, rebuilt from the same bars.

Every decision, pass or fail, is written next to the candidate and into the
model's history. Rollback is just moving the pointer (``ModelRegistry.rollback``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading.brokers.lots import LotSizes
from trading.training.dataset import build_dataset
from trading.training.evaluate import EvalConfig, EvalResult, evaluate_model, evaluate_oos
from trading.training.registry import ModelRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionGate:
    min_sharpe_improvement: float = 0.0
    max_drawdown_pct: float = 0.20  # of equity, as a positive fraction
    max_feature_importance: float = 0.50
    min_trades: int = 20
    require_positive_sharpe: bool = True
    # Optional, off by default: also require the walk-forward folds (not just the
    # holdout) to trade profitably - a model that only shines on one short window
    # may have got lucky there.
    min_oos_sharpe: float | None = None


@dataclass
class GateDecision:
    promote: bool
    failures: list[str]
    checks: dict[str, str]
    candidate: dict[str, Any]
    live: dict[str, Any] | None = None
    live_version: str | None = None
    candidate_version: str | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    actor: str = "gate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "failures": self.failures,
            "checks": self.checks,
            "candidate_version": self.candidate_version,
            "live_version": self.live_version,
            "candidate": _headline(self.candidate),
            "live": _headline(self.live) if self.live else None,
            "gate": self.gate,
            "actor": self.actor,
        }


def _headline(m: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sharpe",
        "total_return",
        "max_drawdown_pct",
        "trades",
        "hit_rate",
        "turnover",
        "ic",
        "window",
    )
    return {k: m.get(k) for k in keys}


def check_gate(
    candidate: Mapping[str, Any],
    importances: Mapping[str, float],
    live: Mapping[str, Any] | None,
    gate: PromotionGate,
    *,
    oos_sharpe: float | None = None,
) -> GateDecision:
    """Pure decision from metrics on a shared window; no side effects."""
    failures: list[str] = []
    checks: dict[str, str] = {}
    sharpe = float(candidate.get("sharpe") or 0.0)

    if live is not None:
        if live.get("window") != candidate.get("window"):
            failures.append("candidate and live were not evaluated on the same window")
        bar = float(live.get("sharpe") or 0.0) + gate.min_sharpe_improvement
        ok = sharpe > bar
        checks["sharpe"] = (
            f"{sharpe:.3f} vs live {live.get('sharpe')} (+{gate.min_sharpe_improvement})"
        )
        if not ok:
            failures.append(f"Sharpe {sharpe:.3f} does not beat the live model's {bar:.3f}")
    if gate.require_positive_sharpe and sharpe <= 0:
        failures.append(f"Sharpe {sharpe:.3f} is not positive")
    checks.setdefault("sharpe", f"{sharpe:.3f}")

    drawdown = abs(float(candidate.get("max_drawdown_pct") or 0.0))
    checks["drawdown"] = f"{drawdown:.2%} vs limit {gate.max_drawdown_pct:.2%}"
    if drawdown > gate.max_drawdown_pct:
        failures.append(f"drawdown {drawdown:.2%} exceeds {gate.max_drawdown_pct:.2%}")

    if importances:
        top_name, top_share = max(importances.items(), key=lambda kv: kv[1])
        checks["concentration"] = (
            f"{top_name} {top_share:.1%} vs limit {gate.max_feature_importance:.0%}"
        )
        if top_share > gate.max_feature_importance:
            limit = gate.max_feature_importance
            failures.append(f"{top_name} carries {top_share:.1%} of the model (> {limit:.0%})")
    else:
        failures.append("no feature importances recorded")

    if gate.min_oos_sharpe is not None:
        checks["oos_sharpe"] = f"{oos_sharpe} vs minimum {gate.min_oos_sharpe}"
        if oos_sharpe is None or oos_sharpe <= gate.min_oos_sharpe:
            failures.append(f"walk-forward Sharpe {oos_sharpe} is not above {gate.min_oos_sharpe}")

    trades = int(candidate.get("trades") or 0)
    checks["trades"] = f"{trades} vs minimum {gate.min_trades}"
    if trades < gate.min_trades:
        failures.append(f"only {trades} trades on the window (< {gate.min_trades})")

    return GateDecision(
        promote=not failures,
        failures=failures,
        checks=checks,
        candidate=dict(candidate),
        live=dict(live) if live is not None else None,
        gate=asdict(gate),
    )


def evaluate_on_window(
    registry: ModelRegistry,
    name: str,
    version: str,
    frames: Mapping[str, pd.DataFrame],
    window_start: datetime,
    *,
    eval_cfg: EvalConfig,
    lots: LotSizes | None = None,
) -> EvalResult:
    """Score a registered version on bars from ``window_start`` on, rebuilding its
    features and labels from ``frames`` (which must include enough earlier bars to
    warm the features)."""
    model = registry.load(name, version)
    ds = build_dataset(frames, model.interval, spec=model.spec, label=model.label, lots=lots)
    rows = (ds.meta["ts"] >= pd.Timestamp(window_start)).to_numpy().nonzero()[0]
    return evaluate_model(model, ds.take(rows), eval_cfg)


def promote_if_better(
    registry: ModelRegistry,
    name: str,
    version: str,
    frames: Mapping[str, pd.DataFrame],
    window_start: datetime,
    *,
    eval_cfg: EvalConfig | None = None,
    gate: PromotionGate | None = None,
    lots: LotSizes | None = None,
    actor: str = "gate",
) -> GateDecision:
    eval_cfg = eval_cfg or EvalConfig()
    gate = gate or PromotionGate()
    candidate = evaluate_on_window(
        registry, name, version, frames, window_start, eval_cfg=eval_cfg, lots=lots
    )
    live_version = registry.live_version(name)
    live = None
    if live_version is not None and live_version != version:
        live = evaluate_on_window(
            registry, name, live_version, frames, window_start, eval_cfg=eval_cfg, lots=lots
        ).metrics
    model = registry.load(name, version)
    oos_sharpe = (
        evaluate_oos(model, eval_cfg).metrics["sharpe"] if gate.min_oos_sharpe is not None else None
    )
    decision = check_gate(candidate.metrics, model.importances, live, gate, oos_sharpe=oos_sharpe)
    decision.candidate_version, decision.live_version, decision.actor = version, live_version, actor
    registry.record_decision(name, version, decision.to_dict())
    if decision.promote:
        registry.set_live(
            name,
            version,
            reason=f"gate passed: Sharpe {candidate.metrics['sharpe']:.3f}"
            + (f" vs {live['sharpe']:.3f} ({live_version})" if live else " (first live model)"),
            actor=actor,
        )
    else:
        log.info("%s %s not promoted: %s", name, version, "; ".join(decision.failures))
    return decision
