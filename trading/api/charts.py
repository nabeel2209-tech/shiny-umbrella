"""Server-rendered SVG line charts for the dashboard.

No chart library: an equity curve is a path, a few hairlines and some text. The
page stays fast, works with JavaScript off, and passes the strict CSP. A small
script (``static/app.js``) adds the hover layer - a crosshair that snaps to the
nearest bar, and the same readout from the keyboard - using the per-point data
embedded on the ``<figure>``.

Design rules (from the house data-viz guidance): one series per chart, so no
legend box - the title names it; a 2px line with a 10% area wash; solid hairline
gridlines; direct-label only the last value; never two y-axes (equity and
drawdown are two charts sharing the x-axis); and a table view of the same data.

The x-axis is *bar-indexed*, not clock time: trading happens in sessions, and a
time axis would spend most of its width on nights and weekends.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape

from trading.api.formatting import inr_compact, pct


def lttb(values: Sequence[float], threshold: int) -> list[int]:
    """Largest-triangle-three-buckets: indices of ``threshold`` points that keep the
    visual shape (peaks and troughs survive, unlike taking every n-th point)."""
    n = len(values)
    if threshold >= n or threshold < 3:
        return list(range(n))
    out = [0]
    bucket = (n - 2) / (threshold - 2)
    a = 0
    for i in range(threshold - 2):
        start = (math.floor((i + 1) * bucket)) + 1
        end = min((math.floor((i + 2) * bucket)) + 1, n)
        nxt_start = (math.floor((i + 2) * bucket)) + 1
        nxt_end = min((math.floor((i + 3) * bucket)) + 1, n)
        nxt = range(nxt_start, nxt_end) if nxt_start < nxt_end else range(n - 1, n)
        avg_x = sum(nxt) / len(nxt)
        avg_y = sum(values[j] for j in nxt) / len(nxt)
        best, best_area = start, -1.0
        for j in range(start, end):
            area = abs((a - avg_x) * (values[j] - values[a]) - (a - j) * (avg_y - values[a]))
            if area > best_area:
                best, best_area = j, area
        out.append(best)
        a = best
    out.append(n - 1)
    return out


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]: steps of 1, 2 or 5 x 10^k."""
    if hi == lo:
        pad = abs(hi) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.floor(lo / step) * step
    ticks = []
    t = first
    while t <= hi + step * 0.5:
        ticks.append(round(t, 10))
        t += step
    return ticks


@dataclass
class Chart:
    svg: str
    points: str  # JSON for the hover layer


def line_chart(
    stamps: Sequence[datetime],
    values: Sequence[float],
    *,
    chart_id: str,
    fmt: Callable[[float], str] = inr_compact,
    reference: float | None = None,
    area_to: str = "bottom",  # "bottom" or "reference" (a drawdown hangs from zero)
    width: int = 760,
    height: int = 240,
    max_points: int = 600,
) -> Chart:
    left, right, top, bottom = 68, 72, 14, 30
    pw, ph = width - left - right, height - top - bottom
    if not values:
        return Chart(svg="", points="{}")
    keep = lttb(values, max_points)
    xs_idx = keep
    vals = [values[i] for i in keep]
    lo, hi = min(vals), max(vals)
    if reference is not None:
        lo, hi = min(lo, reference), max(hi, reference)
    ticks = nice_ticks(lo, hi)
    y_lo, y_hi = ticks[0], ticks[-1]
    span = (y_hi - y_lo) or 1.0
    n = len(values)

    def x_of(i: int) -> float:
        return left + (i / max(n - 1, 1)) * pw

    def y_of(v: float) -> float:
        return top + (1 - (v - y_lo) / span) * ph

    pts = [(x_of(i), y_of(v)) for i, v in zip(xs_idx, vals, strict=True)]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    floor_y = y_of(reference) if area_to == "reference" and reference is not None else top + ph
    area = f"{line} L{pts[-1][0]:.1f},{floor_y:.1f} L{pts[0][0]:.1f},{floor_y:.1f} Z"

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{chart_id}-title" preserveAspectRatio="xMidYMid meet">'
    ]
    add = parts.append
    base = top + ph
    for t in ticks:  # hairline grid + y labels
        y = y_of(t)
        add(_el("line", cls="grid", x1=left, x2=left + pw, y1=y, y2=y))
        add(_el("text", fmt(t), cls="tick", x=left - 8, y=y + 4, anchor="end"))
    # x labels at the first bar of each session, at most six of them
    starts = [i for i in range(n) if i == 0 or stamps[i].date() != stamps[i - 1].date()]
    stride = max(1, math.ceil(len(starts) / 6))
    for i in starts[::stride]:
        x = x_of(i)
        add(_el("line", cls="axis", x1=x, x2=x, y1=base, y2=base + 4))
        add(_el("text", f"{stamps[i]:%d %b}", cls="tick", x=x, y=base + 18, anchor="middle"))
    add(_el("line", cls="axis", x1=left, x2=left + pw, y1=base, y2=base))
    if reference is not None:
        ry = y_of(reference)
        add(_el("line", cls="reference", x1=left, x2=left + pw, y1=ry, y2=ry))
    add(_el("path", cls="area", d=area))
    add(_el("path", cls="line", d=line))
    ex, ey = pts[-1]
    add(_el("circle", cls="dot", cx=ex, cy=ey, r=4))
    add(_el("text", fmt(vals[-1]), cls="end-label", x=ex + 8, y=ey + 4))
    # hover layer, drawn by app.js
    add(_el("line", cls="crosshair", x1=0, x2=0, y1=top, y2=base, visibility="hidden"))
    add(_el("circle", cls="dot hover-dot", cx=0, cy=0, r=4, visibility="hidden"))
    parts.append("</svg>")
    data = {
        "x": [round(x, 1) for x, _ in pts],
        "y": [round(y, 1) for _, y in pts],
        "t": [f"{stamps[i]:%d %b %H:%M}" for i in xs_idx],
        "v": [fmt(v) for v in vals],
        "w": width,
    }
    return Chart(svg="".join(parts), points=json.dumps(data, separators=(",", ":")))


def _el(
    tag: str, text: str | None = None, *, cls: str = "", anchor: str = "", **attrs: object
) -> str:
    """One SVG element; numbers are rounded, text is escaped."""
    fields = [f'class="{cls}"'] if cls else []
    if anchor:
        fields.append(f'text-anchor="{anchor}"')
    for key, value in attrs.items():
        shown = f"{value:.1f}" if isinstance(value, float) else value
        fields.append(f'{key}="{escape(str(shown), quote=True)}"')
    opening = f"<{tag} {' '.join(fields)}"
    return f"{opening}>{escape(text)}</{tag}>" if text is not None else f"{opening}/>"


def pct_fmt(v: float) -> str:
    return pct(v, 1)
