"""Number and time formatting for the dashboard (Jinja filters)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from trading.core.types import IST

MISSING = "\u2013"  # an en dash for values that do not exist


def indian_grouping(n: int) -> str:
    """1234567 -> '12,34,567' (lakh / crore grouping)."""
    s = str(abs(n))
    if len(s) <= 3:
        head = s
    else:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        head = ",".join([head, *parts]) + "," + tail if head else ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + head


def inr(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return MISSING
    v = round(float(value), decimals)  # round first, so 999.6 shows as 1,000
    whole = int(abs(v))
    frac = f"{abs(v) - whole:.{decimals}f}"[1:] if decimals else ""
    return ("-" if v < 0 else "") + "₹" + indian_grouping(whole) + frac


def inr_compact(value: Any) -> str:
    """₹950 / ₹12.3K / ₹4.25L / ₹1.05Cr."""
    if value is None:
        return MISSING
    v = float(value)
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f}L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:.1f}K"
    return f"{sign}₹{a:,.0f}"


def pct(value: Any, decimals: int = 2) -> str:
    if value is None:
        return MISSING
    return f"{float(value) * 100:.{decimals}f}%"


def signed_pct(value: Any, decimals: int = 2) -> str:
    if value is None:
        return MISSING
    return f"{float(value) * 100:+.{decimals}f}%"


def num(value: Any, decimals: int = 2) -> str:
    if value is None:
        return MISSING
    if isinstance(value, int):
        return indian_grouping(value)
    return f"{float(value):,.{decimals}f}"


def when(value: Any) -> str:
    """ISO timestamp -> '26 Sep 14:03' in IST."""
    if not value:
        return MISSING
    ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST)
    return ts.strftime("%d %b %H:%M")


def duration(seconds: Any) -> str:
    if seconds in (None, ""):
        return MISSING
    s = float(seconds)
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86_400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86_400:.1f} d"


FILTERS = {
    "inr": inr,
    "inr_compact": inr_compact,
    "pct": pct,
    "signed_pct": signed_pct,
    "num": num,
    "when": when,
    "duration": duration,
}
