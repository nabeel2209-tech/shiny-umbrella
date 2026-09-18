# Multi-agent algo trading platform (India, Dhan first)

A modular trading platform for NSE equities, NSE index F&O and MCX gold, built in
phases. The trading engine is four agents (data → signal → risk → execution) on an
internal message bus; brokers plug in behind one adapter interface; models are
trained offline, versioned, and promoted through a gate — never inside the live loop.

```
Dashboard & API ──► Training pipeline ──► Model registry ──► Signal agents
                                                                  │
      Storage ◄──── Data agent ──► Signal agents ──► Risk agent ──► Execution
                                                                  │
                                     Broker adapter interface ────┘
                              (DhanBroker | PaperBroker | future brokers)
```

## Hard constraints

1. `features/features.py` is the single feature implementation for training and live.
2. No per-tick model updates. Retraining is a scheduled, validated, reversible batch job.
3. Every `OrderIntent` passes the risk agent; execution rejects anything without an approval token.
4. Orders are idempotent (our UUID is the broker tag); reconcile against the broker before placing.
5. Real orders only when `LIVE_TRADING=true` **and** the broker is `DhanBroker` **and** a startup prompt is confirmed. Default is paper.
6. Timestamps are tz-aware IST at the adapter boundary; canonical symbols only above the adapter.
7. Walk-forward splits with a purge gap equal to the label horizon. No shuffling across time.
8. Labels are net of costs (`backtest/costs.py`).
9. Dhan API details come from the DhanHQ v2 docs, recorded in `trading/brokers/README.md`.
10. Destructive actions (deleting data, rewriting the archive) require explicit confirmation.

## Setup

```bash
make setup          # uv venv + deps + copies .env.example to .env
make check          # ruff + pytest
```

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv). Secrets live only in
`.env` (git-ignored); see `.env.example` for every key.

## Build order

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core types, market clock, bus, broker interface, paper broker, cost model | done |
| 2 | Dhan adapter, symbol map, archive ingest | done |
| 3 | Trading engine (data / signal / risk / execution / monitor agents) | |
| 4 | Backtester | |
| 5 | Training pipeline, registry, promotion gate, nightly schedule | |
| 6 | FastAPI + dashboard | |
| 7 | Hardening: docker compose, logging, alerts, runbook | |

## Phase 1 — what exists

| Module | Purpose |
|--------|---------|
| `trading/core/types.py` | Pydantic models: `Bar`, `Tick`, `Signal`, `OrderIntent`, `RiskApproval`/`RiskRejection`, `OrderRequest`, `Order`, `Fill`, `Position`, `Funds`, `Alert`, `Heartbeat`, `ControlCommand`, enums. All timestamps validated as tz-aware and normalised to IST. |
| `trading/core/clock.py` | `MarketCalendar` (NSE 09:15–15:30, MCX 09:00–23:30 configurable, holidays + special sessions from JSON): `is_open`, `next_open`, `next_close`, `session_bars`, `bar_start`, trading-day navigation. `SystemClock` / `SimClock` so live and replay share code. |
| `trading/core/bus.py` | `Topics`, JSON envelope + type registry, `InMemoryBus` (deterministic, JSON round-trip in strict mode) and `RedisBus` (pattern pub/sub), `make_bus()`. |
| `trading/core/config.py` | `Settings` from `.env`; `live_orders_allowed`, `problems()` for startup validation. |
| `trading/core/db.py` | SQLAlchemy engine helper (SQLite default, Postgres-ready). |
| `trading/brokers/symbols.py` | Canonical symbol grammar: `NSE:RELIANCE`, `NFO:NIFTY-OCT26`, `NFO:NIFTY-14OCT26-25000-CE`, `MCX:GOLDM-OCT26`. |
| `trading/brokers/base.py` | `MarketData`, `OrderRouter`, `Broker` Protocols; `Instrument`; adapter exceptions. |
| `trading/brokers/paper.py` | `PaperBroker`: wraps any `MarketData` for quotes, simulates fills (limit/market/SL/SLM, slippage), applies Indian costs, tracks cash/positions/PnL/margin, streams order updates. |
| `trading/brokers/paper_store.py` | SQLite persistence for the paper account. |
| `trading/backtest/costs.py` | Indian fee model per segment (brokerage, STT/CTT, exchange, SEBI, stamp, GST, DP) and `round_trip_cost_bps` for the risk threshold. |
| `data/holidays/holidays.json` | 2026 NSE holidays from the official NSE API (`scripts/fetch_holidays.py`); MCX split sessions from dhan.co/market-holiday. |

Run the tests:

```bash
make check
```

The Phase 1 acceptance test (`tests/test_paper.py::test_one_day_replay_buy_then_sell_matches_hand_computation`)
replays a full NSE session of synthetic 1-minute bars through `PaperBroker`, buys with a
resting limit, sells at market, and checks cash, position and PnL against hand-computed
fees — then reloads the account from SQLite and checks it again.

## Phase 2 — what exists

Read [`trading/brokers/README.md`](trading/brokers/README.md) first: it records everything taken
from the DhanHQ v2 docs (endpoints, limits, packet layouts, instrument-master quirks) and the
open `TODO(dhan)` items.

| Module | Purpose |
|--------|---------|
| `trading/brokers/dhan.py` | `DhanBroker`: REST client with rate limiter, retries, JWT expiry + renewal; historical daily/intraday (90-day chunking, epoch→IST); binary market-feed decoder and websocket stream with reconnect/resubscribe; order-update websocket; order/position/funds mapping; idempotent `place_order` by tag; `reconcile()`. |
| `trading/brokers/dhan_instruments.py` | Downloads the detailed scrip master (public CSV), caches it per day in `data/instruments/`, builds the symbol map. |
| `trading/brokers/symbols.py` | `SymbolMap`: canonical ↔ `(segment, security id)`, aliases (monthly ↔ day-specific), `front_month()`, `option()`, MCX contract multipliers. |
| `trading/training/ingest.py` | `Archive` (Parquet: `1m/<symbol>/<date>.parquet`, `1d/<symbol>/<year>.parquet`), `validate_bars()` gap/duplicate/price/session report, split detection, corporate actions applied on read, incremental `ingest()` that tops up from the next expected bar. |
| `trading/core/universe.py` | Nifty 100 constituents from NSE's CSV (cached), index symbols. |
| `scripts/ingest_history.py` | CLI: `--symbols` / `--universe nifty100|indices|gold|all`, `--intervals`, `--start/--end/--days`, `--dry-run`, `--report-json`. Prints bar counts and the gap report per symbol. |
| `scripts/fetch_holidays.py` | Refreshes the NSE block of `data/holidays/holidays.json` from the NSE holiday API (MCX block is kept by hand). |
| `data/corporate_actions.json` | Confirmed splits/bonuses (ratio per ex-date); the archive itself is never rewritten. |
| `tests/fixtures/dhan_master_sample.csv` | 77 real rows of the instrument master used by the symbol-map tests. |

Also changed in Phase 2: order tags are 27 chars (Dhan's `correlationId` limit is 30),
`Instrument`/`Position`/`Fill` carry a contract `multiplier` (MCX quantity is in lots),
and the market calendar extends MCX to 23:55 during US daylight-saving time.

### Running it

1. Put `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` in `.env` (24-hour token from
   web.dhan.co → My Profile → Access DhanHQ APIs). Data APIs need Dhan's paid Data API
   subscription; order APIs additionally need a static IP whitelisted in the portal.
2. Ingest one symbol and read the gap report:

```bash
.venv/bin/python scripts/ingest_history.py --symbols NSE:RELIANCE --intervals 1m,1d --days 10
```

3. Or run the live integration test (skipped by default, never in CI):

```bash
DHAN_INTEGRATION=1 .venv/bin/python -m pytest -m integration -s tests/test_dhan_integration.py
```

4. Whole universe, daily bars since 2021, then keep topping up nightly:

```bash
.venv/bin/python scripts/ingest_history.py --universe all --intervals 1d --start 2021-01-01
.venv/bin/python scripts/ingest_history.py --universe all --intervals 1m,1d
```

`--dry-run` shows what would be fetched without credentials.
