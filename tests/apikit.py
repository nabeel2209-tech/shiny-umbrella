"""Shared pieces for the API and dashboard tests: an app wired to temporary
directories, a market-data feed that never touches the network, and helpers."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.auth import Role
from trading.api.services import Services, build_services
from trading.brokers.base import Instrument
from trading.brokers.paper import PaperBroker, PaperConfig
from trading.core.clock import MarketCalendar
from trading.core.config import Settings
from trading.core.types import Bar, Interval, Tick
from trading.training.ingest import Archive, bars_to_frame

PASSWORD = "correct horse battery staple"
USER_PASSWORD = "another long password"
SYM = "NSE:RELIANCE"
FIRST_DAY = date(2026, 9, 17)

ALWAYS_LONG = {
    "id": "always_long",
    "name": "Always long",
    "symbols": [SYM],
    "interval": "5m",
    "product": "MIS",
    "rules": {"long": {"always": True}},
    "expected_edge_bps": 100.0,
    "sizing": {"mode": "fixed_qty", "qty": 10},
}

TREND_YAML = """\
id: trend_test
name: Trend test
symbols: [NSE:RELIANCE]
interval: 5m
rules:
  long:
    all:
      - {feature: trend, op: gt, value: 0.0005}
  exit_long:
    any:
      - {feature: trend, op: lt, value: 0.0}
expected_edge_bps: 25
"""


class FakeFeed:
    """Market data that never connects anywhere: no history, and ticks only when a
    test puts them on the queue."""

    name = "fake-feed"

    def __init__(self) -> None:
        self.ticks: asyncio.Queue[Tick] = asyncio.Queue()
        self.closed = False

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def instruments(self) -> list[Instrument]:
        return []

    async def historical(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> list[Bar]:
        return []

    async def subscribe_live(self, symbols: Sequence[str]) -> AsyncIterator[Tick]:
        while True:
            yield await self.ticks.get()

    async def ltp(self, symbols: Sequence[str]) -> dict[str, float]:
        return dict.fromkeys(symbols, 2500.0)


class FakeDhan(PaperBroker):
    """Stands in for DhanBroker in live-mode tests: same name, simulated fills."""

    name = "dhan"

    def __init__(self) -> None:
        super().__init__(FakeFeed(), config=PaperConfig(account_id="fake-dhan"))


class NotDhan(FakeDhan):
    name = "paper"


def make_settings(tmp: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        db_url=f"sqlite:///{tmp / 'state.db'}",
        archive_dir=tmp / "archive",
        models_dir=tmp / "models",
        instruments_dir=tmp / "instruments",
        strategies_dir=tmp / "strategies",
        state_dir=tmp / "state",
        session_secret="test-secret-" + "x" * 24,
        **overrides,
    )


def write_archive(root: Path, calendar: MarketCalendar, days: int = 2) -> list[date]:
    """A few sessions of smooth 1-minute bars for SYM."""
    archive = Archive(root)
    written, day, price = [], FIRST_DAY, 2500.0
    while len(written) < days:
        if calendar.is_trading_day("NSE", day):
            bars = []
            for i, ts in enumerate(calendar.session_bars("NSE", day, Interval.M1)):
                close = round(2500.0 + 20 * math.sin(i / 30) + len(written) * 5, 2)
                bars.append(
                    Bar(
                        symbol=SYM,
                        ts=ts,
                        interval=Interval.M1,
                        open=price,
                        high=round(max(price, close) + 1, 2),
                        low=round(min(price, close) - 1, 2),
                        close=close,
                        volume=5000,
                    )
                )
                price = close
            archive.write(SYM, Interval.M1, bars_to_frame(bars))
            written.append(day)
        day += timedelta(days=1)
    return written


class Api:
    """A running app plus the services behind it."""

    def __init__(
        self,
        tmp: Path,
        *,
        symbol_map: Any = None,
        live_broker: Any = None,
        **settings: Any,
    ) -> None:
        self.settings = make_settings(tmp, **settings)
        self.feed = FakeFeed()
        self.live_broker = live_broker

        async def feed_factory() -> Any:
            return self.feed

        async def live_factory() -> Any:
            return self.live_broker or FakeDhan()

        async def loader() -> Any:
            return symbol_map

        self.services: Services = build_services(
            self.settings,
            feed_factory=feed_factory,
            live_broker_factory=live_factory,
            symbol_map_loader=loader,
            backtests_dir=tmp / "backtests",
        )
        self.app = create_app(self.settings, services=self.services, admin=("admin", PASSWORD))
        self.client = TestClient(self.app, base_url="http://testserver")

    def __enter__(self) -> Api:
        self.client.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.client.__exit__(*exc)

    # ------------------------------------------------------------------ auth helpers
    def login(self, username: str = "admin", password: str = PASSWORD) -> dict[str, Any]:
        r = self.client.post("/api/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return r.json()

    def bearer(self, username: str = "admin", password: str = PASSWORD) -> dict[str, str]:
        """Headers for a token-authenticated script (no cookie, no CSRF)."""
        token = self.login(username, password)["token"]
        self.client.cookies.clear()
        return {"Authorization": f"Bearer {token}"}

    def add_user(self, username: str = "bob") -> dict[str, str]:
        self.services.auth.create_user(username, USER_PASSWORD, role=Role.USER)
        return self.bearer(username, USER_PASSWORD)

    def browser_login(self, username: str = "admin", password: str = PASSWORD) -> str:
        """Sign in through the login form, like a browser. Returns the CSRF token."""
        r = self.client.post(
            "/login", data={"username": username, "password": password}, follow_redirects=False
        )
        assert r.status_code == 303, r.text
        return self.csrf()

    def csrf(self) -> str:
        sid = self.services.auth.authenticate(self.client.cookies.get("trading_session"))[1]
        return self.services.auth.csrf_token(sid)

    def wait_for_job(self, job_id: str, headers: dict[str, str], timeout: float = 60) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/backtests/{job_id}", headers=headers).json()
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(0.1)
        raise AssertionError(f"backtest {job_id} did not finish: {job}")
