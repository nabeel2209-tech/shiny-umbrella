"""The nightly job, the weekly drift report, the signal log and the training CLI."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from trading.brokers.lots import LotSizes, MissingLotSize
from trading.core.bus import InMemoryBus, Topics
from trading.core.types import IST, FeeBreakdown, Fill, Interval, ProductType, Side, Signal
from trading.training.ingest import Archive
from trading.training.labels import LabelSpec
from trading.training.registry import ModelRegistry
from trading.training.schedule import ModelJob, NightlyConfig, NightlyJob, _trading_flags
from trading.training.signal_log import SignalLog, SignalLogger
from trading.training.train import ModelKind, TrainConfig

from .conftest import HOLDOUT_SESSIONS, PLANTED_HORIZON, PLANTED_SYMBOL


@pytest.fixture(scope="module")
def planted_archive(tmp_path_factory, planted_data):
    archive = Archive(tmp_path_factory.mktemp("archive"))
    archive.write(PLANTED_SYMBOL, Interval.M1, planted_data.frame)
    return archive


@pytest.fixture
def last_day(planted_data) -> date:
    return planted_data.frame["ts"].iloc[-1].date()


def job(**kw) -> ModelJob:
    base = dict(
        name="planted",
        symbols=[PLANTED_SYMBOL],
        interval=Interval.M1,
        label=LabelSpec(horizon=PLANTED_HORIZON),
        train=TrainConfig(kind=ModelKind.RIDGE),
        lookback_days=60,
        holdout_days=HOLDOUT_SESSIONS,
    )
    return ModelJob(**{**base, **kw})


def nightly(archive, calendar, tmp_path, *jobs, **kw) -> NightlyJob:
    cfg = NightlyConfig(jobs=list(jobs) or [job()], report_dir=tmp_path / "reports")
    return NightlyJob(archive, ModelRegistry(tmp_path / "models"), calendar, cfg, **kw)


# --------------------------------------------------------------------------- timing


def test_the_job_runs_after_mcx_closes_and_belongs_to_that_session(
    calendar, planted_archive, tmp_path
):
    n = nightly(planted_archive, calendar, tmp_path)
    friday_noon = datetime(2026, 9, 18, 12, 0, tzinfo=IST)
    at = n.next_run(friday_noon)
    assert at == datetime(2026, 9, 19, 0, 10, tzinfo=IST)  # 23:55 close (US DST) + 15 min
    assert n.session_of(at) == date(2026, 9, 18)  # after midnight, still Friday's retrain
    # Saturday: no MCX session until Monday
    assert n.next_run(at + timedelta(minutes=5)) == datetime(2026, 9, 22, 0, 10, tzinfo=IST)
    # in winter MCX closes at 23:30
    assert n.next_run(datetime(2026, 12, 1, 9, 0, tzinfo=IST)) == datetime(
        2026, 12, 1, 23, 45, tzinfo=IST
    )


# --------------------------------------------------------------------------- the nightly run


def test_nightly_run_trains_registers_and_gates(calendar, planted_archive, tmp_path, last_day):
    n = nightly(planted_archive, calendar, tmp_path)
    first = n.run_job(job(), last_day)
    assert first["version"] == "v0001" and first["promoted"], first["failures"]
    assert first["live_before"] is None and first["live_after"] == "v0001"
    assert first["oos"]["ic"] > 0.2 and first["holdout"]["sharpe"] > 0
    meta = n.registry.metadata("planted", "v0001")
    assert meta["metrics"]["holdout_start"] == first["holdout_start"]
    trained_to = pd_ts(meta["model"]["train_window"]["end"]).date()
    assert trained_to < date.fromisoformat(first["holdout_start"])  # never saw the holdout

    # the same data again: an identical model cannot *beat* the live one, so no churn
    second = n.run_job(job(), last_day)
    assert second["version"] == "v0002" and not second["promoted"]
    assert any("does not beat" in f for f in second["failures"])
    assert n.registry.live_version("planted") == "v0001"


def pd_ts(value):
    import pandas as pd

    return pd.Timestamp(value)


async def test_run_writes_a_report_and_isolates_failures(
    calendar, planted_archive, tmp_path, last_day
):
    broken = job(name="missing_data", symbols=["NSE:NOTARCHIVED"])
    n = nightly(planted_archive, calendar, tmp_path, job(), broken)
    report = await n.run(last_day)
    assert report["models"]["planted"]["promoted"] is True
    assert "error" in report["models"]["missing_data"]  # recorded, did not stop the other
    assert report["ingest"] == {"skipped": "no market data source"}
    assert report["outcomes"] == {"skipped": "no signal log"}
    assert (tmp_path / "reports" / f"nightly-{last_day.isoformat()}.json").exists()


async def test_drift_report_only_on_the_configured_weekday(
    calendar, planted_archive, tmp_path, last_day
):
    n = nightly(planted_archive, calendar, tmp_path)
    n.cfg.drift_weekday = last_day.weekday()
    assert "drift" in await n.run(last_day)
    n.cfg.drift_weekday = (last_day.weekday() + 1) % 7
    assert "drift" not in await n.run(last_day)


async def test_ingest_tops_up_each_symbol_when_a_source_exists(
    calendar, planted_archive, tmp_path, last_day
):
    class Source:
        name = "stub"
        calls: list = []

        async def historical(self, symbol, interval, start, end):
            Source.calls.append(symbol)
            return []

    n = nightly(planted_archive, calendar, tmp_path, source=Source())
    report = await n.run(last_day)
    assert list(report["ingest"]) == [f"{PLANTED_SYMBOL} 1m"]


def test_derivative_jobs_need_lot_sizes(calendar, planted_archive, tmp_path):
    with pytest.raises(MissingLotSize):
        nightly(planted_archive, calendar, tmp_path, job(name="gold", symbols=["MCX:GOLDM-OCT26"]))
    ok = nightly(
        planted_archive,
        calendar,
        tmp_path,
        job(name="gold", symbols=["MCX:GOLDM-OCT26"]),
        lots=LotSizes({"MCX:GOLDM-OCT26": 1}),
    )
    assert ok.lots.get("MCX:GOLDM-OCT26") == 1


def test_jobs_file_round_trip(tmp_path):
    cfg = NightlyConfig.from_yaml("trading/training/jobs.example.yaml")
    j = cfg.jobs[0]
    assert j.interval is Interval.M5 and j.label.product is ProductType.MIS
    assert j.train.kind is ModelKind.RIDGE and j.gate.min_trades == 50


# --------------------------------------------------------------------------- signal log


def signal(ts, score, *, model="planted", version="v0001", interval="1m") -> Signal:
    return Signal(
        ts=ts,
        strategy_id="s",
        symbol=PLANTED_SYMBOL,
        score=score,
        prob=0.6,
        expected_edge_bps=abs(score) * 1e4 + 9,
        model_version=version,
        meta={"interval": interval, "model": model, "reason": "model"},
    )


async def test_signals_are_logged_and_outcomes_attached(tmp_path, planted_archive, planted_data):
    log = SignalLog(f"sqlite:///{tmp_path / 'signals.db'}")
    bus = InMemoryBus()
    await SignalLogger(bus, log).start()
    frame = planted_data.frame
    i = 5000
    ts = frame["ts"].iloc[i].to_pydatetime()
    await bus.publish(Topics.SIGNALS, signal(ts, 0.001))
    await bus.publish(Topics.SIGNALS, signal(ts, 0.001, model=None))  # a rule-only strategy
    assert len(log.frame()) == 2
    attached = log.attach_outcomes(planted_archive, horizons={"planted": PLANTED_HORIZON})
    assert attached == 1  # rule-only signal has no horizon
    row = log.frame(model="planted").iloc[0]
    expected = frame["close"].iloc[i + PLANTED_HORIZON] / frame["close"].iloc[i] - 1
    assert row["outcome"] == pytest.approx(expected) and row["outcome_horizon"] == PLANTED_HORIZON
    assert log.attach_outcomes(planted_archive, horizons={}, default_horizon=5) == 1


async def test_drift_flags_a_live_model_that_stopped_working(
    calendar, planted_archive, tmp_path, planted_ridge, planted_data, last_day
):
    from trading.training.promote import promote_if_better

    log = SignalLog(f"sqlite:///{tmp_path / 'signals.db'}")
    n = nightly(planted_archive, calendar, tmp_path, signal_log=log)
    v = n.registry.register("planted", planted_ridge)
    promote_if_better(
        n.registry, "planted", v, {PLANTED_SYMBOL: planted_data.frame}, planted_data.holdout_start
    )
    frame = planted_data.frame
    start = len(frame) - 5 * 375
    for k in range(40):  # predictions that point the wrong way
        i = start + k * 45
        move = frame["close"].iloc[i + PLANTED_HORIZON] / frame["close"].iloc[i] - 1
        log.record(signal(frame["ts"].iloc[i].to_pydatetime(), -move, version=v))
    log.attach_outcomes(planted_archive, horizons={"planted": PLANTED_HORIZON})
    drift = await n.drift(last_day)
    model = drift["models"]["planted"]
    assert model["realised"]["count"] >= 20 and model["realised"]["ic"] < -0.9
    assert model["expected"]["version"] == v and model["expected"]["ic"] > 0.2
    assert any("anti-correlated" in f for f in drift["flags"])


def test_trading_drift_flags():
    realised = {"trades": 4, "hit_rate": 0.25, "net_pnl": -500.0}
    expected = {"trades": 10, "hit_rate": 0.60, "net_pnl": 2_000.0}
    flags = _trading_flags(realised, expected)
    assert len(flags) == 3
    assert not _trading_flags({"trades": 10, "hit_rate": 0.58, "net_pnl": 1_800.0}, expected)


async def test_trading_drift_compares_fills_with_a_backtest(
    calendar, planted_archive, tmp_path, last_day
):
    class Result:
        metrics = {
            "trades": 1,
            "hit_rate": 1.0,
            "net_pnl": 100.0,
            "fills": 2,
            "fees": {"total": 2.0},
        }

    async def backtest(start, end):
        return Result()

    t = datetime.combine(last_day, datetime.min.time(), tzinfo=IST) + timedelta(hours=10)

    def fills(start, end):
        common = {"symbol": PLANTED_SYMBOL, "qty": 10, "product": ProductType.MIS}
        fee = FeeBreakdown(brokerage=1.0)
        return [
            Fill(order_id="a", side=Side.BUY, price=100.0, ts=t, fees=fee, **common),
            Fill(
                order_id="b",
                side=Side.SELL,
                price=95.0,
                ts=t + timedelta(minutes=5),
                fees=fee,
                **common,
            ),
        ]

    n = nightly(planted_archive, calendar, tmp_path, realised_fills=fills, backtest=backtest)
    drift = await n.drift(last_day)
    assert drift["trading"]["realised"]["net_pnl"] == pytest.approx(-52.0)
    assert drift["trading"]["backtest"]["net_pnl"] == 100.0
    assert any("net PnL" in f for f in drift["flags"])


def test_engine_can_log_its_signals(calendar, tmp_path):
    from trading.agents.engine import EngineConfig, TradingEngine
    from trading.brokers.paper import PaperBroker
    from trading.strategies.schema import StrategyConfig

    s = StrategyConfig.model_validate(
        {
            "id": "x",
            "symbols": ["NSE:RELIANCE"],
            "expected_edge_bps": 20,
            "rules": {"long": {"always": True}},
        }
    )
    log = SignalLog(f"sqlite:///{tmp_path / 's.db'}")
    engine = TradingEngine(
        InMemoryBus(), PaperBroker(), calendar, EngineConfig(strategies=[s]), signal_log=log
    )
    assert engine.signal_logger in engine.agents


# --------------------------------------------------------------------------- CLI


async def test_train_cli_fit_list_and_rollback(
    tmp_path, planted_archive, last_day, monkeypatch, capsys
):
    import scripts.train as cli
    from trading.core import config

    async def no_master(*_a, **_k):
        return None  # an equity-only job needs no instrument master

    monkeypatch.setattr(cli, "ensure_symbol_map", no_master)
    monkeypatch.setenv("ARCHIVE_DIR", str(planted_archive.root))
    config.get_settings.cache_clear()
    try:
        models = str(tmp_path / "models")
        args = [
            "--models", models, "fit", "--name", "planted", "--symbols", PLANTED_SYMBOL,
            "--interval", "1m", "--as-of", last_day.isoformat(), "--lookback-days", "60",
            "--holdout-days", str(HOLDOUT_SESSIONS), "--horizon", str(PLANTED_HORIZON),
        ]  # fmt: skip
        assert await cli.main([*args, "--promote"]) == 0
        assert await cli.main(args) == 0  # registered, not gated
        capsys.readouterr()
        assert await cli.main(["--models", models, "list"]) == 0
        listing = capsys.readouterr().out
        assert (
            "planted  (live: v0001)" in listing
            and "gate passed" in listing
            and "not gated" in listing
        )
        assert (
            await cli.main(
                ["--models", models, "promote", "planted", "v0002", "--reason", "t", "--yes"]
            )
            == 0
        )
        assert await cli.main(["--models", models, "rollback", "planted", "--reason", "t"]) == 0
        assert ModelRegistry(models).live_version("planted") == "v0001"
        assert await cli.main(["--models", models, "rollback", "planted", "--reason", "t"]) == 2
    finally:
        config.get_settings.cache_clear()


def test_a_model_interval_missing_from_the_archive_is_built_from_minutes(
    calendar, planted_archive, planted_data
):
    from trading.training.dataset import load_frames

    first = planted_data.frame["ts"].iloc[0].date()
    frames = load_frames(
        planted_archive,
        [PLANTED_SYMBOL],
        Interval.M5,
        first,
        first + timedelta(days=3),
        calendar=calendar,
    )
    five = frames[PLANTED_SYMBOL]
    assert len(five) == 75 * len(set(five["ts"].dt.date))  # 75 five-minute bars a session
    one = planted_data.frame[planted_data.frame["ts"].dt.date == first]
    assert (
        five["open"].iloc[0] == one["open"].iloc[0]
        and five["close"].iloc[0] == one["close"].iloc[4]
    )
    assert five["high"].iloc[0] == one["high"].iloc[:5].max()


def test_a_job_with_no_data_says_so(calendar, planted_archive, tmp_path, last_day):
    n = nightly(planted_archive, calendar, tmp_path)
    with pytest.raises(ValueError, match="no usable 1m bars for NSE:NOTARCHIVED"):
        n.run_job(job(symbols=["NSE:NOTARCHIVED"]), last_day)
