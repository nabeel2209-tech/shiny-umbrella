"""Per-user strategy files: ``<strategies_dir>/<username>/<id>.yaml``.

Strategies are validated by ``StrategyConfig`` on every write, so the engine never
meets a file it cannot load. Deleting moves the file to ``.trash/`` with a
timestamp rather than removing it - a strategy is someone's work, and an undo is
cheap. The shipped examples are read-only templates.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from pydantic import ValidationError

from trading.core.types import now_ist
from trading.strategies.schema import StrategyConfig, load_strategies

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "strategies"
USERNAME_RE = re.compile(r"^[a-z0-9_\-]{1,64}$")


class StrategyExists(ValueError):
    pass


class StrategyNotFound(KeyError):
    pass


class StrategyStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, username: str) -> Path:
        if not USERNAME_RE.match(username):
            raise ValueError(f"bad username {username!r}")
        d = self.root / username
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, username: str, strategy_id: str) -> Path:
        if not re.match(r"^[a-z0-9_\-]{1,64}$", strategy_id):
            raise StrategyNotFound(strategy_id)
        return self._dir(username) / f"{strategy_id}.yaml"

    def list(self, username: str) -> list[StrategyConfig]:
        return load_strategies(self._dir(username))

    def get(self, username: str, strategy_id: str) -> StrategyConfig:
        path = self._path(username, strategy_id)
        if not path.exists():
            raise StrategyNotFound(strategy_id)
        return StrategyConfig.from_yaml(path)

    def get_many(self, username: str, ids: list[str]) -> list[StrategyConfig]:
        return [self.get(username, i) for i in ids]

    def create(self, username: str, strategy: StrategyConfig) -> StrategyConfig:
        path = self._path(username, strategy.id)
        if path.exists():
            raise StrategyExists(strategy.id)
        return self._write(path, strategy)

    def update(self, username: str, strategy_id: str, strategy: StrategyConfig) -> StrategyConfig:
        path = self._path(username, strategy_id)
        if not path.exists():
            raise StrategyNotFound(strategy_id)
        if strategy.id != strategy_id:
            raise ValueError("the id in the body must match the one in the URL")
        return self._write(path, strategy)

    def delete(self, username: str, strategy_id: str) -> Path:
        """Soft delete: returns where the file went."""
        path = self._path(username, strategy_id)
        if not path.exists():
            raise StrategyNotFound(strategy_id)
        trash = self._dir(username) / ".trash"
        trash.mkdir(exist_ok=True)
        target = trash / f"{strategy_id}.{now_ist():%Y%m%d-%H%M%S}.yaml"
        shutil.move(path, target)
        return target

    @staticmethod
    def _write(path: Path, strategy: StrategyConfig) -> StrategyConfig:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(strategy.to_yaml())
        tmp.replace(path)
        return strategy

    # ------------------------------------------------------------------ templates
    def templates(self) -> list[StrategyConfig]:
        out = []
        for sub in ("examples", "benchmarks"):
            d = TEMPLATES_DIR / sub
            if d.exists():
                out.extend(load_strategies(d))
        return out


def parse_strategy_yaml(text: str) -> tuple[StrategyConfig | None, list[str]]:
    """(strategy, errors) - never raises on user input."""
    import yaml

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return None, [f"YAML: {e}"]
    if not isinstance(data, dict):
        return None, ["the YAML must be a mapping of strategy fields"]
    return validate_strategy_dict(data)


def validate_strategy_dict(data: dict) -> tuple[StrategyConfig | None, list[str]]:
    try:
        return StrategyConfig.model_validate(data), []
    except ValidationError as e:
        return None, [
            f"{'.'.join(str(x) for x in err['loc']) or 'strategy'}: {err['msg']}"
            for err in e.errors()
        ]
    except ValueError as e:  # e.g. a malformed symbol
        return None, [str(e)]
