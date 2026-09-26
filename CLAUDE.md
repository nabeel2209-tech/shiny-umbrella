# Working on this repo

Multi-agent algo trading platform for India (Dhan first). `README.md` has the
architecture, the hard constraints, and what each phase built; read it first.

## How the work is run

- The platform is built in 7 phases, in order. At the end of each phase: run
  `make check`, summarise what exists (files, how to run, decisions to confirm),
  commit, and **stop for the user's review** before starting the next phase. Never
  skip a phase's acceptance test.
- Phases 1-6 are done. **Phase 7** (docker compose, structured logging, email/Telegram
  alert hooks, config validation on startup, `make check`, README setup + build order
  + "first week" runbook) has not started: wait for the user's go-ahead.
- Ask before anything destructive: deleting data, rewriting the archive, force-pushing.
- The hard constraints in `README.md` always apply (feature parity, no per-tick model
  updates, every intent through risk, idempotent orders, live only with
  LIVE_TRADING + Dhan + confirmation, IST everywhere, walk-forward only, net-of-cost
  labels).
- Do not invent Dhan API details. Read the DhanHQ v2 docs and record findings in
  `trading/brokers/README.md`; mark unknowns `TODO(dhan)`.
- Lot sizes: never default a derivative (`NFO:`, `MCX:`) to 1. NSE equities may
  default to 1. Use `trading/brokers/lots.py` (`LotSizes`) and
  `dhan_instruments.ensure_symbol_map` (downloads the instrument master when missing);
  a derivative with no known lot is an error that stops the run.

## Decisions the user has not answered yet

Re-ask these rather than assuming an answer.

Phase 5:
1. Add an engine-backtest check to the promotion gate, or keep the vectorised
   evaluator? (Same holdout window: evaluator Sharpe 7.42, engine 1.50; same direction.)
2. Keep deploying each model exactly as evaluated (no refit on the holdout)?
3. "Retrain including logged paper/live predictions and outcomes" is implemented as
   adding realised outcomes to the training data, not meta-labelling. Right?
4. Keep the optional `min_oos_sharpe` gate rule off by default?

Phase 6:
1. User management: only the `.env` admin exists; add an "add user" admin page?
2. One platform-wide live Dhan account, admins only; per-user credentials later?
3. Practise needs Dhan data credentials; add an archive-replay mode that needs none?
4. With no terminal (Docker/systemd) nobody can answer the LIVE startup prompt, so
   the dashboard is paper-only. Keep that, or design another confirmation? (Phase 7.)
5. Saved strategy YAML writes every default field; omit defaults instead?
6. Backtests run one at a time; fine?

## New machine

```bash
make setup                          # venv, deps, copies .env.example to .env
python -m scripts.set_password      # create or reset the dashboard admin
make check && make serve            # http://127.0.0.1:8000
```

`.env` (Dhan credentials, dashboard settings) and `data/` are never committed. Dhan
order APIs only accept whitelisted static IPs, so live orders work only from the
machine whose IP is whitelisted in the Dhan portal (changing it locks for 7 days).
