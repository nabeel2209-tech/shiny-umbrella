"""Strategy configuration (YAML) and its rule evaluator.

A strategy is data, not code: the dashboard's Strategy Builder writes this YAML and
the signal agent reads it. Conditions are an explicit tree of comparisons -
deliberately **not** Python expressions, so a strategy from the UI can never
execute arbitrary code.

```yaml
id: ema_cross_reliance
name: Trend follow RELIANCE
symbols: [NSE:RELIANCE]
interval: 5m
product: MIS
model: {name: null}                 # rule-only; set a name to use the registry
sizing: {mode: fixed_qty, qty: 10}
execution: {urgency: NORMAL, limit_band_bps: 10, ttl_seconds: 300}
rules:
  long:
    all:
      - {feature: trend, op: gt, value: 0.0}
      - {feature: rsi_14, op: lt, value: 70}
  exit_long:
    any:
      - {feature: trend, op: lt, value: 0.0}
```

Rule names: ``long`` / ``short`` open a position, ``exit_long`` / ``exit_short``
close one. A strategy with no ``short`` block is long-only.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading.brokers.symbols import parse_symbol
from trading.core.types import Interval, ProductType, Urgency


class Op(StrEnum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    ABS_GT = "abs_gt"  # |feature| > value
    ABS_LT = "abs_lt"

    def apply(self, left: float, right: float) -> bool:
        match self:
            case Op.GT:
                return left > right
            case Op.GTE:
                return left >= right
            case Op.LT:
                return left < right
            case Op.LTE:
                return left <= right
            case Op.EQ:
                return left == right
            case Op.NEQ:
                return left != right
            case Op.ABS_GT:
                return abs(left) > right
            case Op.ABS_LT:
                return abs(left) < right
        raise AssertionError(f"unhandled op {self}")  # pragma: no cover


class Condition(BaseModel):
    """One comparison: ``feature op (value | other)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature: str
    op: Op = Op.GT
    value: float | None = None
    other: str | None = None  # compare against another feature instead of a constant

    @model_validator(mode="after")
    def _one_rhs(self) -> Condition:
        if (self.value is None) == (self.other is None):
            raise ValueError("condition needs exactly one of 'value' or 'other'")
        return self

    def evaluate(self, features: dict[str, float]) -> bool:
        if self.feature not in features:
            raise KeyError(self.feature)
        left = features[self.feature]
        if self.other is not None:
            if self.other not in features:
                raise KeyError(self.other)
            right = features[self.other]
        else:
            right = float(self.value)  # type: ignore[arg-type]
        return self.op.apply(left, right)

    def describe(self) -> str:
        rhs = self.other if self.other is not None else self.value
        return f"{self.feature} {self.op.value} {rhs}"


class RuleGroup(BaseModel):
    """``all`` (AND) / ``any`` (OR) / ``none`` (NOR) over conditions and subgroups.

    ``always: true`` is a rule that holds unconditionally - what a buy-and-hold
    benchmark needs, since it reads no features at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    always: bool = False
    all: list[Condition | RuleGroup] = Field(default_factory=list)
    any: list[Condition | RuleGroup] = Field(default_factory=list)
    none: list[Condition | RuleGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_empty(self) -> RuleGroup:
        if not (self.always or self.all or self.any or self.none):
            raise ValueError("rule group is empty")
        return self

    def evaluate(self, features: dict[str, float]) -> bool:
        ok = True
        if self.all:
            ok = ok and all(c.evaluate(features) for c in self.all)
        if self.any:
            ok = ok and any(c.evaluate(features) for c in self.any)
        if self.none:
            ok = ok and not any(c.evaluate(features) for c in self.none)
        return ok

    def features_used(self) -> set[str]:
        out: set[str] = set()
        for c in (*self.all, *self.any, *self.none):
            if isinstance(c, Condition):
                out.add(c.feature)
                if c.other:
                    out.add(c.other)
            else:
                out |= c.features_used()
        return out


class SizingMode(StrEnum):
    FIXED_QTY = "fixed_qty"
    FIXED_NOTIONAL = "fixed_notional"
    KELLY = "kelly"  # size suggestion from prob/payoff; the risk agent caps it


class Sizing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: SizingMode = SizingMode.FIXED_QTY
    qty: int = Field(default=1, gt=0)
    notional: float = Field(default=100_000.0, gt=0)
    fraction: float = Field(default=0.02, gt=0, le=1)  # of equity, for kelly mode
    max_qty: int | None = Field(default=None, gt=0)


class ExecutionPrefs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    urgency: Urgency = Urgency.NORMAL
    limit_band_bps: float = Field(default=10.0, ge=0)
    ttl_seconds: int = Field(default=300, gt=0)
    stop_loss_pct: float | None = Field(default=None, gt=0)  # bracket stop on fill
    # exit after this many bars in the position; a model strategy defaults to the
    # model's label horizon, so it is traded over the span it was trained to predict
    max_holding_bars: int | None = Field(default=None, gt=0)
    take_profit_pct: float | None = Field(default=None, gt=0)


class ModelRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None  # registry model name; None = rule-only strategy
    version: str | None = None  # None = follow the registry's "live" pointer
    min_abs_score: float = Field(default=0.0, ge=0)  # ignore weaker predictions
    edge_scale_bps: float = Field(default=10_000.0, gt=0)  # score -> expected edge bps

    @property
    def enabled(self) -> bool:
        return self.name is not None


class StrategyConfig(BaseModel):
    """One strategy. ``id`` is unique and appears on every Signal and OrderIntent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_\-]{1,64}$")
    name: str = ""
    enabled: bool = True
    symbols: list[str] = Field(min_length=1)
    interval: Interval = Interval.M5
    product: ProductType = ProductType.MIS
    rules: dict[Literal["long", "short", "exit_long", "exit_short"], RuleGroup] = Field(
        default_factory=dict
    )
    model: ModelRef = Field(default_factory=ModelRef)
    sizing: Sizing = Field(default_factory=Sizing)
    execution: ExecutionPrefs = Field(default_factory=ExecutionPrefs)
    payoff_ratio: float = Field(default=1.0, gt=0)  # b in Kelly, if the model has no view
    expected_edge_bps: float = Field(default=0.0, ge=0)
    """Edge the strategy claims, in basis points, used when there is no model.

    The risk agent's cost threshold (constraint 8) compares this against the
    round-trip cost of the trade, so a rule-based strategy that cannot state an
    edge larger than its costs will never be approved - which is the point.
    """
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> StrategyConfig:
        for s in self.symbols:
            parse_symbol(s)  # raises SymbolError on a malformed symbol
        if not self.rules and not self.model.enabled:
            raise ValueError("strategy has neither rules nor a model")
        if self.product is ProductType.CNC and any(
            parse_symbol(s).is_derivative for s in self.symbols
        ):
            raise ValueError("CNC is for equities only")
        return self

    def features_used(self) -> set[str]:
        out: set[str] = set()
        for group in self.rules.values():
            out |= group.features_used()
        return out

    @property
    def needs_features(self) -> bool:
        """Whether signals must wait for warm features. A strategy that reads none
        (``always`` rules, no model) can act on the very first bar."""
        return bool(self.features_used()) or self.model.enabled

    def validate_features(self, available: list[str] | set[str]) -> list[str]:
        """Feature names the rules reference that the data agent does not produce."""
        return sorted(self.features_used() - set(available))

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: Path | str) -> StrategyConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_yaml_str(cls, text: str) -> StrategyConfig:
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        data: dict[str, Any] = self.model_dump(mode="json", exclude_defaults=False)
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def load_strategies(directory: Path | str) -> list[StrategyConfig]:
    """Every ``*.yaml`` in ``directory``, sorted by id. Duplicate ids are an error."""
    out = [StrategyConfig.from_yaml(p) for p in sorted(Path(directory).glob("*.yaml"))]
    ids = [s.id for s in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate strategy ids: {sorted(dupes)}")
    return sorted(out, key=lambda s: s.id)
