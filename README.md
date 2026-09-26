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
make serve          # dashboard on http://127.0.0.1:8000 (set DASHBOARD_PASSWORD first)
```

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv). Secrets live only in
`.env` (git-ignored); see `.env.example` for every key.

## Build order

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core types, market clock, bus, broker interface, paper broker, cost model | done |
| 2 | Dhan adapter, symbol map, archive ingest | done |
| 3 | Trading engine (data / signal / risk / execution / monitor agents) | done |
| 4 | Backtester | done |
| 5 | Training pipeline, registry, promotion gate, nightly schedule | done |
| 6 | FastAPI + dashboard | done |
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

## Phase 3 — what exists

Five agents on the message bus. They never call each other: the bus is the only
coupling, so the same code runs against live Dhan, the paper broker or an archive
replay.

```
ticks/bars ─► data ─► features.<symbol> ─► signal ─► intents ─► risk ─► approved ─► execution ─► broker
                                                                  │                      │
                                                            rejected                   fills ─► portfolio
                                                                  └──────── monitor ◄────┘
```

| Module | Purpose |
|--------|---------|
| `trading/features/features.py` | **The** feature implementation (constraint 1), imported by the data agent and by training. Causal, finite-window only, so the live rolling buffer reproduces batch values to round-off. |
| `trading/strategies/schema.py` | Strategy YAML: an explicit condition tree (`{feature, op, value}` with `all`/`any`/`none`), sizing, execution prefs, model reference. No `eval`, so a strategy from the UI cannot run code. |
| `trading/agents/data.py` | `BarBuilder` (ticks → bars, cumulative-volume deltas, session resets, aggregation) and `DataAgent` (live / replay / warmup, publishes `Bar` + `FeatureVector`). |
| `trading/agents/signal.py` | One agent per strategy: rules and/or a model score, position-aware entries and exits, sizing (fixed qty / notional / Kelly), emits `OrderIntent`. Picks up a newly promoted model version without a restart. |
| `trading/agents/risk.py` | Nine rules — kill switch, strategy paused, market hours, instrument sanity, daily loss, cost threshold, Kelly cap, per-instrument limit, gross exposure. Trimming rules cut the size; exits skip the entry-only rules. Issues a one-shot `RiskApproval`. |
| `trading/agents/execution.py` | Order state machine, urgency → price (never a raw market order on options), slicing by participation and freeze limit, chase with step/slippage caps, TTL cancel, bracket stop on fill with OCO cancellation, rate limiting, reconciliation. |
| `trading/agents/monitor.py` | Alert log, heartbeats and staleness, kill switch, achieved-vs-mid slippage per strategy. |
| `trading/agents/portfolio.py` | Positions, day PnL and exposure rebuilt from the fill stream; reconciled from the broker at startup. |
| `trading/agents/engine.py` | Assembles the five agents, orders the bus subscriptions so a simulated broker sees each bar before any strategy reacts to it, and holds the live-trading gate (constraint 5). |
| `scripts/run_paper.py` | The whole engine on the paper broker: Dhan's live feed, or `--replay` from the archive. |

### Running it

```bash
# replay two archived days through the full engine (no network, no credentials)
.venv/bin/python scripts/run_paper.py --strategies trading/strategies/examples --replay 2026-09-17:2026-09-18

# live Dhan feed, fake cash and fills, state persisted to SQLite
.venv/bin/python scripts/run_paper.py --strategies trading/strategies/examples
```

The acceptance test is `tests/test_engine_e2e.py`: it writes bars to a real Parquet
archive, reads them back, replays them through the engine, and checks the cash,
position and PnL that land in SQLite against fees worked out by hand in the test
file — so an error in `backtest/costs.py` cannot hide behind itself.

### Decisions worth knowing

- **A strategy must declare `expected_edge_bps`.** The cost threshold (constraint 8)
  compares it against the round-trip cost, so a rule-based strategy that cannot
  state an edge bigger than its costs never gets approved. That is deliberate.
- **`RiskApproval` carries the intent it approves.** The execution agent therefore
  needs no prior sighting of the intent, which matters because Redis does not
  guarantee ordering and agents restart independently.
- **Exits skip the entry-only rules** (edge, Kelly, position and exposure limits).
  Closing risk is never blocked by a risk limit; the kill switch still applies.
- **Bracket stops are OCO.** When a position goes flat the outstanding stop is
  cancelled, otherwise it would later trigger and open a fresh position the other
  way — a long-only strategy would quietly end the day short.

## Phase 4 — what exists

The backtester is the production engine replayed over the archive. `BacktestRunner`
builds the same `TradingEngine` the paper and live runners use and swaps in a
simulated broker; nothing about signals, risk or execution is re-implemented.

| Module | Purpose |
|--------|---------|
| `trading/backtest/sim_broker.py` | `SimBroker`: the paper broker with the optimism removed — nothing fills on the bar that produced the signal, limits need a trade-through (a touch is not a fill), stops fill at the gap, liquidity-taking fills pay slippage from a pluggable model (`FixedSlippage`, `VolumeSlippage` square-root impact), optional volume participation cap with partial fills, Indian costs on every fill. In memory only. |
| `trading/backtest/metrics.py` | FIFO round-trip trades from fills (fees split pro rata, open lots marked to market), trade stats (hit rate, profit factor, expectancy), equity stats (return, Sharpe/Sortino, drawdown with duration), fee breakdown, turnover. Reusable for paper/live accounts. |
| `trading/backtest/runner.py` | `BacktestConfig` / `BacktestRunner` / `BacktestResult`: loads bars (resampling from 1m when an interval is not archived), primes features from the days before `start`, replays bars in completion order, records the equity curve, saves JSON + CSV. `list_runs()` for the dashboard. |
| `scripts/run_backtest.py` | CLI: `--start/--end`, `--only`, `--impact`, `--participation`, `--liquidate`, `--list`. |
| `trading/strategies/benchmarks/` | Buy-and-hold NIFTYBEES, the yardstick to run beside any equity strategy. |

Each run writes `data/backtests/<run_id>/`: `summary.json` (config with a
fingerprint, metrics, per-strategy breakdown), `trades.csv`, `equity.csv`,
`fills.csv`, `orders.csv` and the strategy YAML exactly as run.

```bash
.venv/bin/python scripts/run_backtest.py --start 2026-08-03 --end 2026-09-18
.venv/bin/python scripts/run_backtest.py --strategies trading/strategies/benchmarks \
    --start 2026-01-01 --end 2026-06-30 --liquidate --max-position 1000000
.venv/bin/python scripts/run_backtest.py --list
```

Acceptance tests (`tests/test_backtest.py`): an always-flat strategy makes zero
trades with a flat equity curve; buy-and-hold earns exactly the archive's return on
the capital it invests when trading is free, and exactly that minus hand-computed
delivery charges and slippage when it is not.

### What makes it honest

- **Decisions happen when a bar completes.** The engine's clock reads the bar's
  end (`MarketCalendar.bar_end`), and the order fills on the *next* bar.
- **Limits need a trade-through; stops fill at the gap.** Both are the usual ways a
  backtest flatters itself.
- **MIS is squared off 10 minutes before each exchange's close**, as the broker
  would, and new MIS entries stop 15 minutes before the close (`intraday_cutoff`),
  so an intraday strategy can never hold overnight in simulation.
- **Warm features from the first bar** — the buffers are primed from history
  before `start`, exactly as the live data agent does from broker history.
- **Fast without a second feature implementation.** Features for a known series
  are computed once with the same `compute_features` training uses; a test proves
  the decisions are identical to recomputing on every bar (~0.13 ms/bar vs ~3.6).

### Changed in earlier phases along the way

- `Funds.equity` was `cash + unrealised PnL`, which undercounts a long equity
  position by its whole cost. It is now `cash + positions_value`.
- Stops (paper and sim) fill at the gap price when the market jumps through them.
- The daily loss limit now halts **entries for that day** and resumes the next
  day; it no longer latches the kill switch or blocks exits. A manual KILL still
  stops everything until RESUME.
- The order rate limiter is wall-clock, so backtests switch it off.

## Phase 5 — what exists

Models are trained offline, validated walk-forward, versioned, and promoted through
a gate — never updated inside the live loop (constraint 2).

| Module | Purpose |
|--------|---------|
| `trading/training/labels.py` | Forward-return and triple-barrier labels, **net of costs** (constraint 8): `sign(g) * max(|g| - cost, 0)` with the round trip from `costs.py`. Intraday labels never span the overnight gap. Average-uniqueness weights for overlapping labels. |
| `trading/training/splits.py` | Walk-forward folds on unique timestamps (several symbols never split by row), a purge of at least the label horizon (refused otherwise, constraint 7), an optional embargo, and a holdout split. Nothing is shuffled. |
| `trading/training/dataset.py` | Features from `features.py` (constraint 1) + labels, for one or many symbols; builds 5m/15m from archived 1m bars with the live bar builder. |
| `trading/training/train.py` | Ridge or LightGBM; hyperparameters chosen only by mean validation IC across folds; weights = uniqueness x recency; out-of-fold predictions kept for a logistic calibration of P(right direction) and the payoff ratio; feature importances. |
| `trading/training/evaluate.py` | The risk agent's position rule on predictions — direction from the sign, Kelly-capped size, nothing below the cost hurdle — with overlapping holdings; Sharpe, PnL, drawdown and turnover from the backtester's `metrics.py`. |
| `trading/training/registry.py` | Write-once versions (`data/models/<name>/v0001/`), a single `live.json` pointer with a rollback stack, history and gate decisions. Also the signal agent's model provider: a promotion reaches a running engine on its next bar. |
| `trading/training/promote.py` | The gate: better holdout Sharpe than the live model on the same window, drawdown within limit, no feature over 50% of importance, a minimum trade count. Optional: require positive walk-forward Sharpe too. |
| `trading/training/signal_log.py` | Every engine signal to SQLite; the nightly job fills in the realised outcome. |
| `trading/training/schedule.py` | Nightly after the MCX close: ingest top-up, attach outcomes, retrain, register, gate. Weekly drift report: live predictions vs holdout promises, paper/live fills vs a backtest of the same week. |
| `scripts/train.py` | `fit`, `list`, `promote` (manual, bypasses the gate, asks first), `rollback`, `nightly --jobs FILE [--once]`. |

```bash
.venv/bin/python scripts/train.py fit --name reliance_5m --symbols NSE:RELIANCE \
    --interval 5m --as-of 2026-09-25 --horizon 6 --promote
.venv/bin/python scripts/train.py list
.venv/bin/python scripts/train.py rollback reliance_5m --reason "bad week"
cp trading/training/jobs.example.yaml trading/training/jobs.yaml   # then edit
.venv/bin/python scripts/train.py nightly --jobs trading/training/jobs.yaml
```

A model-driven strategy names its model (`model: {name: reliance_5m}`) and trades
whatever version is live; `run_backtest.py` and `run_paper.py` pick up the registry
(`MODELS_DIR`). A backtest over a window the live model was trained on prints an
in-sample warning.

Acceptance tests: a planted mean-reversion signal is recovered by both model kinds
(IC > 0.2 out of sample, the planted feature ranked first, the right sign) while a
random walk yields none (`test_train.py`); the gate refuses a worse model and leaves
the live pointer alone (`test_promote.py`); a running signal agent switches to a
newly promoted version, and back on rollback, without a restart (`test_registry.py`).

### Design decisions

- **Deployed = evaluated.** A candidate is trained on data ending before its
  holdout and deployed exactly as evaluated — no refit on the holdout. The live
  model was trained the same way, so it has not seen today's holdout either and the
  comparison is out of sample for both. Cost: the newest `holdout_days` are not in
  the deployed model.
- **Net predictions, gross hurdle.** Labels are net of costs, so a prediction is a
  net edge; the risk agent's rule wants a gross one. `gross = |prediction| + cost`,
  converted in one place, so costs are not charged twice.
- **A model is traded over its horizon.** Predictions carry the label horizon and a
  model-driven position exits after that many bars (or `max_holding_bars`).
- Lot sizes: equities default to one share; a derivative with no known lot is an
  error everywhere (engine, backtest, training), and the instrument master is
  downloaded on startup when missing.

## Phase 6 — what exists

A FastAPI app serving a JSON API, a websocket of live events, and an HTMX dashboard
(server-rendered HTML, one vendored script, no front-end build).

| Module | Purpose |
|--------|---------|
| `trading/api/app.py` | App factory, security headers (strict CSP: no inline script or style, `frame-ancestors 'none'`, `no-store`), sign-in redirects, readable HTML error pages. |
| `trading/api/auth.py` | Users (admin / user roles, multi-user ready), scrypt password hashes, HMAC-signed session tokens backed by a revocable session table, per-session CSRF tokens, login throttling. |
| `trading/api/deps.py` | Who is calling: `Authorization: Bearer` for scripts, an HttpOnly SameSite=Lax cookie for the browser. Cookie-authenticated writes need the CSRF token (header or form field). |
| `trading/api/routes/api.py` | The JSON API (list below). |
| `trading/api/routes/pages.py` | Dashboard pages and the `/ui/...` fragments HTMX swaps in. |
| `trading/api/ws.py` | `/ws/live`: orders, fills, signals, rejections, alerts and kill-switch events for the signed-in user. Cookie connections must come from our own Origin. |
| `trading/api/engines.py` | One paper and one live engine per user, built from the same `TradingEngine` as `run_paper.py`; start / stop / flatten / account; the live-trading refusals. |
| `trading/api/jobs.py` | Backtests as a background queue (SQLite-backed, one worker, survives restarts: queued jobs re-run, interrupted ones are marked failed). |
| `trading/api/strategy_store.py` | Per-user strategy YAML files, validated on every write; delete moves to `.trash/`. |
| `trading/api/events.py` | Event hub (per-user queues that drop the oldest event for a slow client), and the persisted kill switch. |
| `trading/api/builder.py` | Strategy Builder form ⇄ strategy dict. |
| `trading/api/charts.py`, `formatting.py` | Server-rendered SVG line charts (LTTB-decimated, crosshair data for hover); rupees in lakh/crore grouping, IST times. |
| `trading/web/` | Jinja templates, `app.css` (light and dark), `app.js` (websocket feed, chart hover and keyboard, builder rows), `vendor/htmx.min.js` (see `VENDORED.md`). |
| `scripts/serve.py` | Runs it: asks for LIVE at startup when `LIVE_TRADING=true`, refuses to start without a dashboard password (or `--dev`). |

JSON API (all under `/api`, all need sign-in except `/healthz` and login):

```
POST /auth/login  POST /auth/logout  GET /auth/me  GET /features
GET|POST /strategies  GET /strategies/templates  POST /strategies/validate
GET|PUT|DELETE /strategies/{id}
POST /backtests  GET /backtests  GET /backtests/{id}  GET /backtests/{id}/equity|trades
GET /engines  POST /engines/{paper|live}/start|stop|flatten
GET /accounts/{paper|live}  POST /accounts/paper/reset
GET /control  POST /control/kill  POST /control/resume (admin)
GET /models  GET /models/{name}  GET /models/{name}/{version}
POST /models/{name}/rollback (admin)  POST /models/{name}/promote (admin, confirm)
GET /marketplace|auto-trading|ai-strategies  → 501, only mounted when FEATURE_MARKETPLACE=true
```

Pages: Strategy Builder (form → YAML, or the YAML itself), Backtesting (run, then
results: stat tiles, equity and drawdown charts, daily table, per-strategy and trade
tables), Algo Trading (own Dhan account), Practise (paper account with fake cash),
Models, How It Works. Marketplace, Auto Trading and AI Strategies are in the nav and
say "coming soon": offering algos to others needs SEBI algo-provider registration.

```bash
# in .env: DASHBOARD_PASSWORD=... and SESSION_SECRET=$(python -c "import secrets;print(secrets.token_hex(32))")
make serve
# a script, without the browser:
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login -H 'content-type: application/json' \
    -d '{"username":"admin","password":"..."}' | python -c "import json,sys;print(json.load(sys.stdin)['token'])")
curl -s localhost:8000/api/engines -H "Authorization: Bearer $TOKEN"
```

Acceptance tests: every route above, both ways it can go
(`tests/test_api_routes.py`, `test_api_auth.py`, `test_api_ws.py`, `test_pages.py`,
`test_api_units.py`).

### Safety rules as the dashboard applies them

- **Live orders** need `LIVE_TRADING=true`, `BROKER=dhan`, LIVE typed at the
  `scripts/serve.py` startup prompt, an admin, LIVE typed again on the Algo Trading
  page, and a broker that really is Dhan. Any one missing means paper, with the
  reason shown.
- **Kill switch** on every page. Anyone signed in can engage it; only an admin can
  release it. It halts running engines (exits included), blocks starts and flatten,
  and survives a restart. An unreadable kill-switch file counts as engaged.
- **Flatten** sends exit intents through the risk agent like any other order.
- **Lot sizes**: a derivative with no known lot size is refused when an engine starts
  and fails its backtest, with the reason.
- A backtest over bars that are not in the archive fails and says to ingest them,
  rather than reporting an empty result.

### Manual walkthrough

Run this after any change to the dashboard. Use a scratch data folder so it does not
touch your paper account: `export DB_URL=sqlite:///data/walkthrough.db
STRATEGIES_DIR=data/walkthrough/strategies STATE_DIR=data/walkthrough/state`.

1. **Start.** `make serve` without `DASHBOARD_PASSWORD` refuses to start and says
   why; `python -m scripts.serve --dev` prints a one-off admin password.
2. **Sign in.** `/` redirects to `/login`. A wrong password shows an error; five
   wrong ones lock the account for a minute. Sign in; the header shows "live" (the
   websocket is connected).
3. **Nav.** All nine items are there. Marketplace, Auto Trading and AI Strategies say
   "coming soon" and mention SEBI.
4. **Strategy Builder.** Start from the "Trend follow RELIANCE" template. The YAML
   preview updates as you type. Change the id, add a condition, remove one, set a
   1.5% stop loss (the YAML shows `stop_loss_pct: 0.015`). Type a feature value of
   `abc`: the preview lists the error. Save; you land on the edit page with "Saved.".
   Save a strategy with an existing id: "already exists". Open "Edit as YAML", change
   the name, save. Delete a strategy; it disappears from the list.
5. **Backtesting.** Pick the strategy and a window you have ingested
   (`scripts/ingest_history.py`). The job appears as queued, then done, without
   reloading. Open it: tiles, equity chart with the starting-capital line, drawdown
   chart, daily table under "Table view", per-strategy and trade tables. Hover a
   chart: crosshair and tooltip; Tab to it and use the arrow keys. Run one over a
   window with no data: it fails with "ingest them first". Switch the OS to dark
   mode: the page follows.
6. **Practise.** Start the paper engine with your strategy (needs Dhan credentials
   for market data). The panel shows running and refreshes itself; the activity feed
   shows engine events as they happen (started, alerts, signals, orders, fills). Flatten: "sent N exit order(s)". Stop. Reset the paper
   account: typing anything but `paper-<you>` is refused.
7. **Kill switch.** Press it with a reason: the red banner appears on every page,
   the activity feed logs it, starting an engine is refused. As a non-admin user the
   banner says only an admin can release it. Release it as admin.
8. **Algo Trading.** With `LIVE_TRADING=false` the page explains what is missing and
   the start button is disabled. With `LIVE_TRADING=true` and `BROKER=dhan`, the
   startup prompt must be answered LIVE, and the start form asks for LIVE again.
   **Do this step only with a strategy sized to one share, and stop the engine
   straight after.**
9. **Models.** After `scripts/train.py fit ... --promote` twice, the model page shows
   both versions, the gate decisions and the history; roll back as admin.
10. **Sign out**, then check `/` redirects to sign-in again.

### Design decisions

- **Server-rendered HTML + HTMX, no front-end build.** Charts are SVG drawn on the
  server; `app.js` only adds hover, the websocket feed, and builder rows. One chart
  per measure (equity and drawdown are two charts, never a dual axis).
- **CSP without `unsafe-inline`.** Everything is a file from this origin, and a
  test fails any page that grows an inline script or style.
- **Bearer tokens for scripts, cookies for the browser.** Only cookie requests need
  the CSRF token, because only cookies are sent automatically by a browser.
- **Per-user state.** Strategies, backtests, paper accounts and engines are keyed by
  user, so adding users later needs no data migration. The live account and the
  kill switch are platform-wide.
- **One backtest worker.** Backtests run one at a time in a thread, so a long one
  cannot starve the live engines' event loop.
