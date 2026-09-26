"""Versioned model registry with a single ``live`` pointer.

Layout (``data/models/``)::

    <name>/v0001/model.joblib      the fitted TrainedModel
    <name>/v0001/metadata.json     features, label, training window, CV, metrics
    <name>/v0001/decisions.jsonl   every promotion-gate decision about this version
    <name>/live.json               {"version": "v0003", "stack": ["v0002", "v0001"], ...}
    <name>/history.jsonl           registered / promoted / rolled_back / refused events

Versions are write-once. Promotion and rollback only move the pointer: rollback
pops the previous live version off ``stack``, so repeated rollbacks walk back
through what was actually live rather than ping-ponging.

The registry is also the signal agent's ``ModelProvider``. ``live_version`` is
checked on every bar but only re-reads ``live.json`` when the file has been
replaced (inode / size / mtime), so a version promoted by the nightly job in
another process reaches a running engine on its next bar - no restart.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import joblib

from trading.agents.signal import ModelPrediction
from trading.core.types import now_ist
from trading.features.features import FeatureSpec
from trading.training.train import TrainedModel

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z0-9_\-]{1,64}$")


class RegistryError(ValueError):
    pass


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)  # atomic: readers see the old file or the new one


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(data, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ModelRegistry:
    def __init__(
        self, root: Path | str = "data/models", *, expected_spec: FeatureSpec | None = None
    ):
        self.root = Path(root)
        self.expected_spec = expected_spec
        self._models: dict[tuple[str, str], TrainedModel] = {}
        self._live_cache: dict[str, tuple[tuple[int, int, int], str | None]] = {}
        self._refused: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ paths
    def _dir(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise RegistryError(f"model name {name!r} must match {NAME_RE.pattern}")
        return self.root / name

    def _vdir(self, name: str, version: str) -> Path:
        return self._dir(name) / version

    # ------------------------------------------------------------------ writing
    def register(
        self,
        name: str,
        model: TrainedModel,
        *,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        actor: str = "train",
    ) -> str:
        base = self._dir(name)
        base.mkdir(parents=True, exist_ok=True)
        existing = self.versions(name)
        version = f"v{(int(existing[-1][1:]) + 1 if existing else 1):04d}"
        vdir = base / version
        vdir.mkdir()
        joblib.dump(model, vdir / "model.joblib")
        _write_json(
            vdir / "metadata.json",
            {
                "name": name,
                "version": version,
                "registered_at": now_ist().isoformat(),
                "notes": notes,
                "model": model.metadata(),
                "metrics": metrics or {},
            },
        )
        _append_jsonl(
            base / "history.jsonl",
            {
                "ts": now_ist().isoformat(),
                "event": "registered",
                "version": version,
                "actor": actor,
            },
        )
        log.info("registered %s %s", name, version)
        return version

    def record_decision(self, name: str, version: str, decision: dict[str, Any]) -> None:
        _append_jsonl(
            self._vdir(name, version) / "decisions.jsonl", {"ts": now_ist().isoformat(), **decision}
        )
        if not decision.get("promote"):
            _append_jsonl(
                self._dir(name) / "history.jsonl",
                {
                    "ts": now_ist().isoformat(),
                    "event": "refused",
                    "version": version,
                    "reason": "; ".join(decision.get("failures", [])),
                    "actor": decision.get("actor", "gate"),
                },
            )

    def set_live(self, name: str, version: str, *, reason: str, actor: str = "manual") -> None:
        if version not in self.versions(name):
            raise RegistryError(f"{name} has no version {version}")
        current = self.live_info(name)
        stack = list(current.get("stack", [])) if current else []
        if current and current["version"] != version:
            stack.insert(0, current["version"])
        self._set_pointer(name, version, stack, reason=reason, actor=actor, event="promoted")

    def rollback(self, name: str, *, reason: str = "manual rollback", actor: str = "manual") -> str:
        """Move the pointer back to the version that was live before this one."""
        current = self.live_info(name)
        if current is None:
            raise RegistryError(f"{name} has no live version to roll back from")
        stack = list(current.get("stack", []))
        if not stack:
            raise RegistryError(f"{name} {current['version']} has nothing to roll back to")
        previous = stack.pop(0)
        self._set_pointer(
            name,
            previous,
            stack,
            reason=f"{reason} (from {current['version']})",
            actor=actor,
            event="rolled_back",
        )
        return previous

    def _set_pointer(
        self, name: str, version: str, stack: list[str], *, reason: str, actor: str, event: str
    ) -> None:
        base = self._dir(name)
        _write_json(
            base / "live.json",
            {
                "version": version,
                "since": now_ist().isoformat(),
                "reason": reason,
                "actor": actor,
                "stack": stack,
            },
        )
        _append_jsonl(
            base / "history.jsonl",
            {
                "ts": now_ist().isoformat(),
                "event": event,
                "version": version,
                "reason": reason,
                "actor": actor,
            },
        )
        self._live_cache.pop(name, None)
        log.warning("%s: %s %s (%s)", name, event, version, reason)

    # ------------------------------------------------------------------ reading
    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and NAME_RE.match(p.name))

    def versions(self, name: str) -> list[str]:
        base = self._dir(name)
        if not base.exists():
            return []
        return sorted(
            p.name for p in base.iterdir() if p.is_dir() and re.match(r"^v\d{4,}$", p.name)
        )

    def metadata(self, name: str, version: str) -> dict[str, Any]:
        path = self._vdir(name, version) / "metadata.json"
        if not path.exists():
            raise RegistryError(f"{name} has no version {version}")
        return json.loads(path.read_text())

    def decisions(self, name: str, version: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._vdir(name, version) / "decisions.jsonl")

    def history(self, name: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._dir(name) / "history.jsonl")

    def load(self, name: str, version: str) -> TrainedModel:
        key = (name, version)
        if key not in self._models:
            path = self._vdir(name, version) / "model.joblib"
            if not path.exists():
                raise RegistryError(f"{name} has no version {version}")
            self._models[key] = joblib.load(path)
        return self._models[key]

    def live_info(self, name: str) -> dict[str, Any] | None:
        path = self._dir(name) / "live.json"
        return json.loads(path.read_text()) if path.exists() else None

    def live_version(self, name: str) -> str | None:
        path = self._dir(name) / "live.json"
        try:
            st = path.stat()
        except FileNotFoundError:
            self._live_cache.pop(name, None)
            return None
        stamp = (st.st_ino, st.st_size, st.st_mtime_ns)
        cached = self._live_cache.get(name)
        if cached and cached[0] == stamp:
            return cached[1]
        info = json.loads(path.read_text())
        self._live_cache[name] = (stamp, info["version"])
        return info["version"]

    # ------------------------------------------------------------------ ModelProvider
    def predict(
        self,
        name: str,
        version: str | None,
        features: dict[str, float],
        symbol: str | None = None,
    ) -> ModelPrediction | None:
        """What the signal agent calls on every bar. Never raises: a model that
        cannot be used produces no prediction and one logged error."""
        if version is None:
            return None
        key = (name, version)
        if key in self._refused:
            return None
        try:
            model = self.load(name, version)
            if self.expected_spec is not None and model.spec != self.expected_spec:
                raise RegistryError(
                    f"{name} {version} was trained on {model.spec}, the data agent computes "
                    f"{self.expected_spec}"
                )
            return model.predict_one(features, version, symbol)
        except (RegistryError, KeyError, ValueError, OSError) as e:
            self._refused.add(key)
            log.error("model %s %s unusable, ignoring it: %s", name, version, e)
            return None
