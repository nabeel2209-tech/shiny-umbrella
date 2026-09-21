"""Data agent: ticks (or archived bars) in, bars + features out.

One code path for three sources, which is what makes a backtest trustworthy:

- **live**    broker websocket ticks -> :class:`BarBuilder` -> bars
- **replay**  bars read from the Parquet archive (backtests, Phase 4)
- **warmup**  history pulled from the broker so features are already warm when the
              first live bar closes

Features always come from ``trading.features.features`` (constraint 1) - computed
on a rolling buffer big enough to reproduce batch values exactly (see that module).
Downstream agents only ever see ``Bar`` on ``bars.<symbol>`` and ``FeatureVector``
on ``features.<symbol>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from trading.agents.base import Agent
from trading.brokers.base import MarketData
from trading.brokers.symbols import parse_symbol
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock, MarketCalendar
from trading.core.types import AlertLevel, Bar, FeatureVector, Interval, Tick
from trading.features.features import (
    DEFAULT_SPEC,
    FeatureSpec,
    bars_to_feature_frame,
    compute_features,
    is_warm,
    latest_features,
)


class BarBuilder:
    """Aggregates ticks (or smaller bars) into bars of one interval.

    Returns the *completed* bar when a new window starts, so a published bar is
    always final - no partial bars reach a strategy.
    """

    def __init__(self, symbol: str, interval: Interval, calendar: MarketCalendar) -> None:
        self.symbol = symbol
        self.interval = interval
        self.calendar = calendar
        self.start: datetime | None = None
        self.open = self.high = self.low = self.close = 0.0
        self.volume = 0
        self.oi: int | None = None
        self._prev_cum_volume: int | None = None
        self._session: date | None = None

    @property
    def pending(self) -> bool:
        return self.start is not None

    def _bar(self) -> Bar:
        assert self.start is not None
        return Bar(
            symbol=self.symbol,
            ts=self.start,
            interval=self.interval,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            oi=self.oi,
        )

    def _open_window(self, start: datetime, price: float, volume: int, oi: int | None) -> None:
        self.start = start
        self.open = self.high = self.low = self.close = price
        self.volume = volume
        self.oi = oi

    def _tick_volume(self, tick: Tick) -> int:
        """Ticks carry *cumulative* session volume; turn that into a per-tick delta."""
        if tick.ts.date() != self._session:
            self._session = tick.ts.date()
            self._prev_cum_volume = None
        if tick.volume <= 0:
            return tick.ltq
        prev = self._prev_cum_volume
        self._prev_cum_volume = tick.volume
        if prev is None or tick.volume < prev:  # first tick of the session, or a reset
            return tick.ltq
        return tick.volume - prev

    def on_tick(self, tick: Tick) -> Bar | None:
        start = self.calendar.bar_start(tick.ts, self.interval)
        delta = self._tick_volume(tick)
        if self.start is None:
            self._open_window(start, tick.ltp, delta, tick.oi)
            return None
        if start > self.start:
            completed = self._bar()
            self._open_window(start, tick.ltp, delta, tick.oi)
            return completed
        if start < self.start:  # late tick from a window we already published
            return None
        self.high = max(self.high, tick.ltp)
        self.low = min(self.low, tick.ltp)
        self.close = tick.ltp
        self.volume += delta
        if tick.oi is not None:
            self.oi = tick.oi
        return None

    def on_bar(self, bar: Bar) -> Bar | None:
        """Aggregate a smaller bar (e.g. 1m) into this interval (e.g. 5m)."""
        start = self.calendar.bar_start(bar.ts, self.interval)
        if self.start is None:
            self._open_window(start, bar.open, bar.volume, bar.oi)
            self.high, self.low, self.close = bar.high, bar.low, bar.close
            return None
        if start > self.start:
            completed = self._bar()
            self._open_window(start, bar.open, bar.volume, bar.oi)
            self.high, self.low, self.close = bar.high, bar.low, bar.close
            return completed
        if start < self.start:
            return None
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        if bar.oi is not None:
            self.oi = bar.oi
        return None

    def close_if_due(self, now: datetime) -> Bar | None:
        """Close the window when its end has passed (a bar with no trades in it)."""
        if self.start is None:
            return None
        if now < self.start + timedelta(seconds=self.interval.seconds):
            return None
        completed = self._bar()
        self.start = None
        return completed

    def flush(self) -> Bar | None:
        if self.start is None:
            return None
        completed = self._bar()
        self.start = None
        return completed


@dataclass
class DataAgentConfig:
    symbols: list[str]
    intervals: list[Interval] = field(default_factory=lambda: [Interval.M1])
    spec: FeatureSpec = field(default_factory=lambda: DEFAULT_SPEC)
    publish_ticks: bool = True
    publish_features: bool = True
    flush_interval_seconds: float = 1.0


class DataAgent(Agent):
    name = "data"

    def __init__(
        self,
        bus: MessageBus,
        calendar: MarketCalendar,
        config: DataAgentConfig,
        *,
        source: MarketData | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(bus, clock=clock)
        self.calendar = calendar
        self.cfg = config
        self.source = source
        self._builders: dict[tuple[str, Interval], BarBuilder] = {}
        self._buffers: dict[tuple[str, Interval], deque[Bar]] = {}
        self._precomputed: dict[
            tuple[str, Interval], dict[datetime, tuple[Bar, dict[str, float], bool]]
        ] = {}
        self.bars_published = 0
        self.ticks_seen = 0

    # ------------------------------------------------------------------ buffers
    def _buffer_size(self, symbol: str, interval: Interval) -> int:
        exchange = parse_symbol(symbol).exchange
        day = self.clock.now().date()
        if not self.calendar.is_trading_day(exchange, day):
            day = self.calendar.next_trading_day(exchange, day)
        per_session = len(self.calendar.session_bars(exchange, day, interval))
        return self.cfg.spec.buffer_bars(per_session)

    def buffer(self, symbol: str, interval: Interval) -> deque[Bar]:
        key = (symbol, interval)
        buf = self._buffers.get(key)
        if buf is None:
            buf = deque(maxlen=self._buffer_size(symbol, interval))
            self._buffers[key] = buf
        return buf

    def builder(self, symbol: str, interval: Interval) -> BarBuilder:
        key = (symbol, interval)
        b = self._builders.get(key)
        if b is None:
            b = BarBuilder(symbol, interval, self.calendar)
            self._builders[key] = b
        return b

    # ------------------------------------------------------------------ publishing
    async def emit_bar(self, bar: Bar) -> None:
        """Publish a completed bar and the features computed from it."""
        buf = self.buffer(bar.symbol, bar.interval)
        if buf and bar.ts <= buf[-1].ts:  # replayed or duplicate bar
            if bar.ts == buf[-1].ts:
                buf[-1] = bar
            else:
                return
        else:
            buf.append(bar)
        self.bars_published += 1
        await self.publish(Topics.bars(bar.symbol), bar)
        if not self.cfg.publish_features:
            return
        known = self._precomputed.get((bar.symbol, bar.interval), {}).get(bar.ts)
        if known is not None and known[0] == bar:
            _, values, warm = known
        else:
            values, warm = latest_features(list(buf), self.cfg.spec, bar.interval)
        await self.publish(
            Topics.features(bar.symbol),
            FeatureVector(
                symbol=bar.symbol,
                ts=bar.ts,
                interval=bar.interval,
                bar=bar,
                values=values,
                warm=warm,
            ),
        )

    async def on_tick(self, tick: Tick) -> None:
        self.ticks_seen += 1
        if self.cfg.publish_ticks:
            await self.publish(Topics.ticks(tick.symbol), tick)
        for interval in self.cfg.intervals:
            completed = self.builder(tick.symbol, interval).on_tick(tick)
            if completed is not None:
                await self.emit_bar(completed)

    async def flush_due(self, now: datetime | None = None) -> None:
        """Close windows whose time has passed even if no tick arrived."""
        now = now or self.clock.now()
        for builder in list(self._builders.values()):
            completed = builder.close_if_due(now)
            if completed is not None:
                await self.emit_bar(completed)

    # ------------------------------------------------------------------ replay speed-up
    def precompute(self, bars: Iterable[Bar]) -> int:
        """Compute features for a whole known series in one pass.

        Only for replay, where every bar is known in advance. It calls the same
        :func:`compute_features` as the incremental path and as training; because
        that function is causal, row *t* of the batch equals what the rolling
        buffer would produce at *t* (to round-off - tested). Recomputing a 376-bar
        window on every bar costs ~9 ms, which makes a year of minute bars a
        quarter of an hour; a lookup costs nothing. A bar that differs from the
        one precomputed (a restatement) falls back to the incremental path.
        """
        groups: dict[tuple[str, Interval], list[Bar]] = {}
        for bar in bars:
            groups.setdefault((bar.symbol, bar.interval), []).append(bar)
        n = 0
        for (symbol, interval), series in groups.items():
            series = sorted({b.ts: b for b in series}.values(), key=lambda b: b.ts)
            frame = compute_features(bars_to_feature_frame(series), self.cfg.spec, interval)
            lookup = self._precomputed.setdefault((symbol, interval), {})
            for bar, (_, row) in zip(series, frame.iterrows(), strict=True):
                values = {k: (float(v) if np.isfinite(v) else 0.0) for k, v in row.items()}
                lookup[bar.ts] = (bar, values, is_warm(row))
                n += 1
        return n

    # ------------------------------------------------------------------ warmup
    def prime(self, bars: Iterable[Bar]) -> int:
        """Load history into the feature buffers without publishing anything.

        The same thing :meth:`warmup` does from a broker, for callers that already
        hold the bars (a backtest loading the days before its start date).
        """
        n = 0
        for bar in bars:
            buf = self.buffer(bar.symbol, bar.interval)
            if buf and bar.ts <= buf[-1].ts:
                continue
            buf.append(bar)
            n += 1
        return n

    async def warmup(self, *, source: MarketData | None = None, end: datetime | None = None) -> int:
        """Fill the feature buffers from history so the first live bar is already warm."""
        src = source or self.source
        if src is None:
            return 0
        end = end or self.clock.now()
        loaded = 0
        for symbol in self.cfg.symbols:
            for interval in self.cfg.intervals:
                need = self._buffer_size(symbol, interval)
                span = timedelta(seconds=interval.seconds * need * 3 + 86_400)
                try:
                    bars = await src.historical(symbol, interval, end - span, end)
                except Exception as e:
                    await self.alert(
                        AlertLevel.WARN, f"warmup failed for {symbol} {interval.value}: {e}"
                    )
                    continue
                buf = self.buffer(symbol, interval)
                for bar in bars[-need:]:
                    buf.append(bar)
                loaded += min(len(bars), need)
                if len(buf) < self.cfg.spec.warmup_bars:
                    await self.alert(
                        AlertLevel.WARN,
                        f"{symbol} {interval.value}: only {len(buf)} warmup bars, "
                        f"need {self.cfg.spec.warmup_bars}",
                    )
        self.log.info("warmup loaded %d bars", loaded)
        return loaded

    # ------------------------------------------------------------------ run modes
    async def run_live(self, source: MarketData | None = None) -> None:
        """Stream ticks until stopped. Reconnection is the adapter's job."""
        src = source or self.source
        if src is None:
            raise ValueError("data agent has no market data source")
        flusher = self.spawn(self._flush_loop(), name="data-flush")
        try:
            async for tick in src.subscribe_live(self.cfg.symbols):
                await self.on_tick(tick)
                if not self.running:
                    break
        finally:
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flusher

    async def _flush_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.cfg.flush_interval_seconds)
            await self.flush_due()
            await self.heartbeat(bars=self.bars_published, ticks=self.ticks_seen)

    async def replay_bars(self, bars: Iterable[Bar]) -> int:
        """Publish pre-built bars in order (archive replay and tests)."""
        n = 0
        for bar in bars:
            await self.emit_bar(bar)
            n += 1
        return n

    async def replay_ticks(self, ticks: Sequence[Tick]) -> int:
        for tick in ticks:
            await self.on_tick(tick)
        for builder in list(self._builders.values()):
            completed = builder.flush()
            if completed is not None:
                await self.emit_bar(completed)
        return len(ticks)


def resample_bars(bars: Iterable[Bar], interval: Interval, calendar: MarketCalendar) -> list[Bar]:
    """Aggregate smaller bars into ``interval`` with the same :class:`BarBuilder`
    the live agent uses, so a backtest on resampled 1m data sees what live would."""
    builders: dict[str, BarBuilder] = {}
    out: list[Bar] = []
    for bar in bars:
        builder = builders.setdefault(bar.symbol, BarBuilder(bar.symbol, interval, calendar))
        completed = builder.on_bar(bar)
        if completed is not None:
            out.append(completed)
    for builder in builders.values():
        last = builder.flush()
        if last is not None:
            out.append(last)
    return sorted(out, key=lambda b: (b.ts, b.symbol))


def merge_bars(streams: Sequence[Sequence[Bar]]) -> list[Bar]:
    """Interleave per-symbol bar lists into one chronological stream.

    Ties break on symbol so a replay is deterministic regardless of input order.
    """
    return list(heapq.merge(*streams, key=lambda b: (b.ts, b.symbol)))
