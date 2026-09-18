"""Live Dhan integration: ingests one symbol and prints bar counts and the gap report.

Skipped unless ``DHAN_INTEGRATION=1`` and credentials are present in the environment /
``.env``. Never run in CI. Uses a temporary archive so the real one is untouched.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from trading.brokers.dhan import DhanBroker, DhanConfig
from trading.core.clock import MarketCalendar
from trading.core.config import Settings
from trading.core.types import IST
from trading.training.ingest import Archive, ingest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("DHAN_INTEGRATION") != "1", reason="set DHAN_INTEGRATION=1 to run"
)
async def test_ingest_one_symbol_live(tmp_path, capsys):
    settings = Settings()
    if not settings.dhan_client_id or not settings.dhan_access_token.get_secret_value():
        pytest.skip("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not configured")
    calendar = MarketCalendar.load(settings.holidays_file)
    archive = Archive(tmp_path / "archive")
    broker = DhanBroker(
        DhanConfig.from_settings(settings), instruments_dir=str(settings.instruments_dir)
    )
    await broker.connect()
    try:
        today = datetime.now(IST).date()
        results = await ingest(
            broker,
            archive,
            ["NSE:RELIANCE"],
            ["1m", "1d"],
            calendar,
            start=today - timedelta(days=7),
            end=today,
        )
        ltp = await broker.ltp(["NSE:RELIANCE"])
    finally:
        await broker.close()
    with capsys.disabled():
        print()
        print("token:", broker.token_status())
        print("ltp:", ltp)
        for r in results:
            print(
                f"{r.symbol} {r.interval.value}: fetched={r.fetched} written={r.written} "
                f"requests={r.requests} error={r.error}"
            )
            if r.report:
                print(r.report.summary())
    for r in results:
        assert r.error is None
        assert r.fetched > 0
