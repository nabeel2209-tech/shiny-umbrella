"""Bar building from ticks, aggregation, and what the data agent publishes."""

from __future__ import annotations

from datetime import datetime

import pytest

from trading.agents.data import BarBuilder, DataAgent, DataAgentConfig, merge_bars
from trading.core.bus import InMemoryBus, Topics
from trading.core.types import IST, Bar, FeatureVector, Interval, Tick
from trading.features.features import DEFAULT_SPEC

from .conftest import make_bars, make_synthetic_day

SYM = "NSE:RELIANCE"


def tick(
    hour: int, minute: int, second: int, ltp: float, *, volume: int = 0, ltq: int = 0, symbol=SYM
):
    return Tick(
        symbol=symbol,
        ts=datetime(2026, 9, 18, hour, minute, second, tzinfo=IST),
        ltp=ltp,
        volume=volume,
        ltq=ltq,
    )


# --------------------------------------------------------------------------- BarBuilder


def test_ticks_build_ohlc_and_roll_over(calendar):
    b = BarBuilder(SYM, Interval.M1, calendar)
    assert b.on_tick(tick(9, 15, 0, 100.0, volume=1000)) is None
    assert b.on_tick(tick(9, 15, 20, 102.0, volume=1100)) is None
    assert b.on_tick(tick(9, 15, 50, 99.0, volume=1150)) is None
    assert b.pending
    done = b.on_tick(tick(9, 16, 1, 101.0, volume=1200))
    assert done is not None
    assert (done.open, done.high, done.low, done.close) == (100.0, 102.0, 99.0, 99.0)
    assert done.ts == datetime(2026, 9, 18, 9, 15, tzinfo=IST)
    assert done.interval is Interval.M1
    # cumulative feed volume becomes per-bar volume; the first tick has no baseline
    assert done.volume == 150
    nxt = b.flush()
    assert nxt is not None and nxt.ts == datetime(2026, 9, 18, 9, 16, tzinfo=IST)
    assert nxt.volume == 50 and b.flush() is None


def test_volume_uses_ltq_when_the_feed_has_no_cumulative_volume(calendar):
    b = BarBuilder(SYM, Interval.M1, calendar)
    b.on_tick(tick(9, 15, 0, 100.0, ltq=7))
    b.on_tick(tick(9, 15, 30, 100.0, ltq=3))
    assert b.flush().volume == 10


def test_volume_resets_across_sessions(calendar):
    b = BarBuilder(SYM, Interval.M1, calendar)
    b.on_tick(tick(15, 29, 0, 100.0, volume=50_000))
    b.on_tick(tick(15, 29, 30, 100.0, volume=50_500))
    first = b.flush()
    assert first.volume == 500
    # next session starts from zero again; the drop must not become a negative delta
    nxt = Tick(
        symbol=SYM, ts=datetime(2026, 9, 21, 9, 15, tzinfo=IST), ltp=100.0, volume=10, ltq=10
    )
    assert b.on_tick(nxt) is None
    assert b.flush().volume == 10


def test_late_tick_from_a_published_window_is_ignored(calendar):
    b = BarBuilder(SYM, Interval.M1, calendar)
    b.on_tick(tick(9, 16, 0, 100.0))
    assert b.on_tick(tick(9, 15, 59, 95.0)) is None
    assert b.flush().low == 100.0


def test_five_minute_aggregation_from_one_minute_bars(calendar):
    b = BarBuilder(SYM, Interval.M5, calendar)
    ones = make_bars(calendar, [100.0, 103.0, 99.0, 101.0, 102.0, 105.0])
    completed = [b.on_bar(bar) for bar in ones]
    assert completed[:5] == [None] * 5
    five = completed[5]
    assert five is not None
    assert five.ts == datetime(2026, 9, 18, 9, 15, tzinfo=IST)
    assert five.interval is Interval.M5
    assert (five.open, five.high, five.low, five.close) == (100.0, 103.0, 99.0, 102.0)
    assert five.volume == 5000


def test_close_if_due_closes_a_bar_with_no_trades(calendar):
    b = BarBuilder(SYM, Interval.M1, calendar)
    b.on_tick(tick(9, 15, 10, 100.0))
    assert b.close_if_due(datetime(2026, 9, 18, 9, 15, 59, tzinfo=IST)) is None
    done = b.close_if_due(datetime(2026, 9, 18, 9, 16, 0, tzinfo=IST))
    assert done is not None and done.close == 100.0
    assert not b.pending


# --------------------------------------------------------------------------- DataAgent


async def collect(bus: InMemoryBus) -> tuple[list[Bar], list[FeatureVector], list[Tick]]:
    bars: list[Bar] = []
    features: list[FeatureVector] = []
    ticks: list[Tick] = []

    async def on_bar(_t, m):
        bars.append(m)

    async def on_feature(_t, m):
        features.append(m)

    async def on_tick(_t, m):
        ticks.append(m)

    await bus.subscribe(Topics.BARS_ALL, on_bar)
    await bus.subscribe(Topics.FEATURES_ALL, on_feature)
    await bus.subscribe(Topics.TICKS_ALL, on_tick)
    return bars, features, ticks


def agent(bus, calendar, clock, **kw):
    cfg = DataAgentConfig(symbols=[SYM], intervals=[Interval.M1], **kw)
    return DataAgent(bus, calendar, cfg, clock=clock)


async def test_replay_publishes_bars_then_features(calendar, sim_clock):
    bus = InMemoryBus()
    bars, features, _ = await collect(bus)
    a = agent(bus, calendar, sim_clock)
    day = make_synthetic_day(calendar)
    await a.replay_bars(day)
    assert len(bars) == len(features) == 375 == a.bars_published
    assert [b.ts for b in bars] == [f.ts for f in features]
    assert features[0].bar == bars[0] and features[0].symbol == SYM
    assert not features[0].warm
    assert features[DEFAULT_SPEC.warmup_bars - 1].warm
    assert features[-1].warm
    assert {"ret_1", "trend", "rsi_14"} <= set(features[-1].values)


async def test_features_match_the_batch_computation(calendar, sim_clock):
    """The agent must publish exactly what training would compute (constraint 1)."""
    from trading.features.features import bars_to_feature_frame, compute_features

    bus = InMemoryBus()
    _, features, _ = await collect(bus)
    day = make_synthetic_day(calendar)
    await agent(bus, calendar, sim_clock).replay_bars(day)
    expected = compute_features(bars_to_feature_frame(day)).iloc[-1]
    for name, value in expected.items():
        assert features[-1].values[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


async def test_buffer_is_capped_at_a_session_plus_warmup(calendar, sim_clock):
    bus = InMemoryBus()
    a = agent(bus, calendar, sim_clock)
    await a.replay_bars(make_synthetic_day(calendar))
    buf = a.buffer(SYM, Interval.M1)
    assert buf.maxlen == DEFAULT_SPEC.buffer_bars(375) == 376
    assert len(buf) == 375


async def test_ticks_flow_through_and_build_bars(calendar, sim_clock):
    bus = InMemoryBus()
    bars, _, ticks = await collect(bus)
    a = agent(bus, calendar, sim_clock)
    await a.replay_ticks(
        [tick(9, 15, 0, 100.0, ltq=5), tick(9, 15, 30, 101.0, ltq=5), tick(9, 16, 0, 99.0, ltq=5)]
    )
    assert len(ticks) == 3 == a.ticks_seen
    assert [b.close for b in bars] == [101.0, 99.0]  # flush() closes the last window


async def test_publish_ticks_can_be_switched_off(calendar, sim_clock):
    bus = InMemoryBus()
    _, _, ticks = await collect(bus)
    a = agent(bus, calendar, sim_clock, publish_ticks=False)
    await a.replay_ticks([tick(9, 15, 0, 100.0, ltq=1)])
    assert ticks == []


async def test_duplicate_and_out_of_order_bars(calendar, sim_clock):
    bus = InMemoryBus()
    bars, _, _ = await collect(bus)
    a = agent(bus, calendar, sim_clock)
    day = make_synthetic_day(calendar)[:5]
    await a.replay_bars(day)
    restated = day[-1].model_copy(update={"close": day[-1].high, "volume": 9999})
    await a.emit_bar(restated)
    assert a.buffer(SYM, Interval.M1)[-1].volume == 9999  # a restated bar replaces it
    assert len(a.buffer(SYM, Interval.M1)) == 5
    await a.emit_bar(day[0])  # a stale bar is dropped, not prepended or republished
    assert len(a.buffer(SYM, Interval.M1)) == 5
    assert len(bars) == 6  # 5 original + the restatement


async def test_flush_due_closes_stale_windows(calendar, sim_clock):
    bus = InMemoryBus()
    bars, _, _ = await collect(bus)
    a = agent(bus, calendar, sim_clock)
    await a.on_tick(tick(9, 15, 10, 100.0, ltq=1))
    await a.flush_due(datetime(2026, 9, 18, 9, 15, 30, tzinfo=IST))
    assert bars == []
    await a.flush_due(datetime(2026, 9, 18, 9, 16, 0, tzinfo=IST))
    assert len(bars) == 1 and bars[0].close == 100.0


async def test_warmup_loads_history(calendar, sim_clock):
    class Source:
        name = "stub"
        calls: list[tuple] = []

        async def historical(self, symbol, interval, start, end):
            Source.calls.append((symbol, interval, start, end))
            return make_synthetic_day(calendar, symbol)

    bus = InMemoryBus()
    a = agent(bus, calendar, sim_clock)
    loaded = await a.warmup(source=Source(), end=datetime(2026, 9, 18, 9, 15, tzinfo=IST))
    # the source has a whole session (375); the buffer wants 376 and takes what it can
    assert loaded == 375 and len(a.buffer(SYM, Interval.M1)) == 375
    assert Source.calls[0][0] == SYM and Source.calls[0][1] is Interval.M1


async def test_warmup_failure_alerts_but_does_not_raise(calendar, sim_clock):
    class Broken:
        name = "broken"

        async def historical(self, *a, **k):
            raise RuntimeError("no data plan")

    bus = InMemoryBus()
    alerts = []

    async def on_alert(_t, m):
        alerts.append(m)

    await bus.subscribe(Topics.ALERTS, on_alert)
    a = agent(bus, calendar, sim_clock)
    assert await a.warmup(source=Broken()) == 0
    assert any("no data plan" in x.message for x in alerts)


def test_merge_bars_is_chronological_and_deterministic(calendar):
    a = make_bars(calendar, [100.0, 101.0], symbol="NSE:AAA")
    b = make_bars(calendar, [200.0, 201.0], symbol="NSE:BBB")
    merged = merge_bars([b, a])
    assert [(x.ts.minute, x.symbol) for x in merged] == [
        (15, "NSE:AAA"),
        (15, "NSE:BBB"),
        (16, "NSE:AAA"),
        (16, "NSE:BBB"),
    ]
    assert merge_bars([a, b]) == merged
