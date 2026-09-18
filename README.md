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
| 2 | Dhan adapter, symbol map, archive ingest | |
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
| `data/holidays/holidays.json` | 2026 NSE/MCX holidays and MCX evening-only sessions. **Unverified — check against exchange circulars.** |

Run the tests:

```bash
make check
```

The Phase 1 acceptance test (`tests/test_paper.py::test_one_day_replay_buy_then_sell_matches_hand_computation`)
replays a full NSE session of synthetic 1-minute bars through `PaperBroker`, buys with a
resting limit, sells at market, and checks cash, position and PnL against hand-computed
fees — then reloads the account from SQLite and checks it again.
