"""Strategy Builder: HTML form fields <-> strategy dict.

The form covers what most strategies need: four rule groups (long / short /
exit_long / exit_short), each a list of ``feature op value`` rows joined by ALL or
ANY, or ``always``; sizing; execution; an optional model. Anything more (nested
groups, comparing two features) is written in the YAML tab, which accepts the full
schema. Either way, what gets saved is validated by ``StrategyConfig``.
"""

from __future__ import annotations

from typing import Any

from starlette.datastructures import FormData

from trading.strategies.schema import Condition, RuleGroup, StrategyConfig

RULE_GROUPS = ("long", "short", "exit_long", "exit_short")
GROUP_TITLES = {
    "long": "Enter long when",
    "short": "Enter short when",
    "exit_long": "Exit a long when",
    "exit_short": "Exit a short when",
}


def _num(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _int(value: str | None) -> int | None:
    v = _num(value)
    return None if v is None else int(v)


def form_to_dict(form: FormData) -> tuple[dict[str, Any], list[str]]:
    """Form fields -> a strategy dict (not yet validated) and any parse errors."""
    errors: list[str] = []
    get = form.get

    def field(name: str, convert, default=None):  # type: ignore[no-untyped-def]
        try:
            value = convert(get(name))
        except (TypeError, ValueError):
            errors.append(f"{name}: not a number")
            return default
        return default if value is None else value

    symbols = [
        s.strip().upper() for s in str(get("symbols", "")).replace(",", " ").split() if s.strip()
    ]
    rules: dict[str, Any] = {}
    for group in RULE_GROUPS:
        if get(f"{group}_always"):
            rules[group] = {"always": True}
            continue
        features = form.getlist(f"{group}_feature")
        ops = form.getlist(f"{group}_op")
        values = form.getlist(f"{group}_value")
        conditions = []
        for i, (feature, op, value) in enumerate(zip(features, ops, values, strict=False)):
            if not str(feature).strip():
                continue
            try:
                conditions.append(
                    {"feature": str(feature).strip(), "op": str(op), "value": float(value)}
                )
            except (TypeError, ValueError):
                errors.append(f"{group} row {i + 1}: value must be a number")
        if conditions:
            rules[group] = {str(get(f"{group}_join") or "all"): conditions}

    data: dict[str, Any] = {
        "id": str(get("id", "")).strip(),
        "name": str(get("name", "")).strip(),
        "enabled": bool(get("enabled")),
        "symbols": symbols,
        "interval": get("interval") or "5m",
        "product": get("product") or "MIS",
        "rules": rules,
        "expected_edge_bps": field("expected_edge_bps", _num, 0.0),
        "payoff_ratio": field("payoff_ratio", _num, 1.0),
        "notes": str(get("notes", "")).strip(),
        "sizing": {
            "mode": get("sizing_mode") or "fixed_qty",
            "qty": field("sizing_qty", _int, 1),
            "notional": field("sizing_notional", _num, 100_000.0),
            "fraction": field("sizing_fraction", _num, 0.02),
            "max_qty": field("sizing_max_qty", _int),
        },
        "execution": {
            "urgency": get("urgency") or "NORMAL",
            "limit_band_bps": field("limit_band_bps", _num, 10.0),
            "ttl_seconds": field("ttl_seconds", _int, 300),
            "stop_loss_pct": _pct(field("stop_loss_pct", _num)),
            "take_profit_pct": _pct(field("take_profit_pct", _num)),
            "max_holding_bars": field("max_holding_bars", _int),
        },
    }
    model_name = str(get("model_name", "")).strip()
    if model_name:
        data["model"] = {
            "name": model_name,
            "version": str(get("model_version", "")).strip() or None,
            "min_abs_score": field("model_min_abs_score", _num, 0.0),
        }
    return data, errors


def _pct(value: float | None) -> float | None:
    """The form asks for percent (1 = 1%); the schema stores a fraction."""
    return None if value is None else value / 100


def form_rows(strategy: StrategyConfig | None) -> dict[str, dict[str, Any]]:
    """Rule groups as simple rows for the form; nested groups are YAML-only."""
    out: dict[str, dict[str, Any]] = {}
    for group in RULE_GROUPS:
        rule: RuleGroup | None = strategy.rules.get(group) if strategy else None  # type: ignore[assignment]
        rows: list[dict[str, Any]] = []
        join, always, nested = "all", False, False
        if rule is not None:
            always = rule.always
            items = rule.all or rule.any
            join = "any" if rule.any and not rule.all else "all"
            nested = bool(rule.none) or any(not isinstance(c, Condition) for c in items)
            for c in items:
                if isinstance(c, Condition) and c.value is not None:
                    rows.append({"feature": c.feature, "op": c.op.value, "value": c.value})
                else:
                    nested = True
        out[group] = {
            "rows": rows or [{"feature": "", "op": "gt", "value": ""}],
            "join": join,
            "always": always,
            "nested": nested,
        }
    return out
