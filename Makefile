PY := .venv/bin/python
RUFF := .venv/bin/ruff

.PHONY: setup check lint format test test-redis

setup:            ## create venv and install everything
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	cp -n .env.example .env || true

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

test:
	$(PY) -m pytest -q

test-redis:       ## also run the Redis bus test (needs REDIS_URL)
	REDIS_URL=$${REDIS_URL:-redis://localhost:6379/0} $(PY) -m pytest -q -m redis

check: lint test  ## what CI runs
