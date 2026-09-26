"""JSON API. Every route needs a signed-in user; writes from the browser need the
CSRF header (see ``auth.py``)."""

from __future__ import annotations

import csv
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from trading.api.auth import COOKIE_NAME, AuthError, User
from trading.api.deps import admin_user, current_user, services
from trading.api.engines import EngineError, EngineMode
from trading.api.jobs import BacktestRequest
from trading.api.services import Services
from trading.api.strategy_store import (
    StrategyExists,
    StrategyNotFound,
    parse_strategy_yaml,
    validate_strategy_dict,
)
from trading.core.types import Interval
from trading.features.features import feature_names
from trading.strategies.schema import Op
from trading.training.registry import RegistryError

public = APIRouter()
router = APIRouter(prefix="/api", dependencies=[Depends(current_user)])


def _engine_error(e: EngineError) -> HTTPException:
    return HTTPException(e.status, e.message)


# --------------------------------------------------------------------------- health & auth


@public.get("/healthz")
async def healthz(svc: Services = Depends(services)) -> dict[str, Any]:
    return {
        "status": "ok",
        "kill_switch": svc.kill.engaged,
        "engines": len(svc.engines.running()),
    }


class Credentials(BaseModel):
    username: str
    password: str


def set_session_cookie(response: Response, token: str, svc: Services) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=svc.settings.cookie_secure,
        max_age=svc.settings.session_ttl_hours * 3600,
        path="/",
    )


@public.post("/api/auth/login")
async def login(
    creds: Credentials, response: Response, svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        user, token = svc.auth.login(creds.username, creds.password)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
    set_session_cookie(response, token, svc)
    _, sid = svc.auth.authenticate(token)
    return {"token": token, "user": _user(user), "csrf_token": svc.auth.csrf_token(sid)}


def _user(user: User) -> dict[str, Any]:
    return {"id": user.id, "username": user.username, "role": user.role.value}


@router.post("/auth/logout")
async def logout(
    request: Request, response: Response, svc: Services = Depends(services)
) -> dict[str, str]:
    svc.auth.logout(request.state.sid)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"status": "signed out"}


@router.get("/auth/me")
async def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return _user(user)


# --------------------------------------------------------------------------- features


@router.get("/features")
async def features() -> dict[str, Any]:
    return {
        "intraday": feature_names(interval=Interval.M5),
        "daily": feature_names(interval=Interval.D1),
        "ops": [o.value for o in Op],
        "intervals": [i.value for i in Interval],
    }


# --------------------------------------------------------------------------- strategies


def _strategy(s: Any) -> dict[str, Any]:
    return s.model_dump(mode="json")


@router.get("/strategies")
async def list_strategies(
    user: User = Depends(current_user), svc: Services = Depends(services)
) -> list[dict[str, Any]]:
    return [_strategy(s) for s in svc.strategies.list(user.username)]


@router.get("/strategies/templates")
async def templates(svc: Services = Depends(services)) -> list[dict[str, Any]]:
    return [_strategy(s) for s in svc.strategies.templates()]


class StrategyText(BaseModel):
    yaml: str = Field(max_length=100_000)


@router.post("/strategies/validate")
async def validate_strategy(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    strategy, errors = (
        parse_strategy_yaml(body["yaml"]) if "yaml" in body else validate_strategy_dict(body)
    )
    unknown = []
    if strategy is not None:
        unknown = strategy.validate_features(feature_names(interval=strategy.interval))
    return {
        "valid": strategy is not None and not unknown,
        "errors": errors + [f"unknown feature: {f}" for f in unknown],
        "yaml": strategy.to_yaml() if strategy else None,
    }


def _parse_body(body: dict[str, Any]) -> Any:
    strategy, errors = (
        parse_strategy_yaml(body["yaml"]) if "yaml" in body else validate_strategy_dict(body)
    )
    if strategy is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"errors": errors})
    unknown = strategy.validate_features(feature_names(interval=strategy.interval))
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"errors": [f"unknown feature: {f}" for f in unknown]},
        )
    return strategy


@router.post("/strategies", status_code=status.HTTP_201_CREATED)
async def create_strategy(
    body: dict[str, Any] = Body(...),
    user: User = Depends(current_user),
    svc: Services = Depends(services),
) -> dict[str, Any]:
    strategy = _parse_body(body)
    try:
        return _strategy(svc.strategies.create(user.username, strategy))
    except StrategyExists as e:
        raise HTTPException(status.HTTP_409_CONFLICT, f"strategy {e.args[0]!r} exists") from e


@router.get("/strategies/{strategy_id}")
async def get_strategy(
    strategy_id: str, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        s = svc.strategies.get(user.username, strategy_id)
    except StrategyNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no strategy {strategy_id!r}") from e
    return _strategy(s) | {"yaml": s.to_yaml()}


@router.put("/strategies/{strategy_id}")
async def update_strategy(
    strategy_id: str,
    body: dict[str, Any] = Body(...),
    user: User = Depends(current_user),
    svc: Services = Depends(services),
) -> dict[str, Any]:
    strategy = _parse_body(body)
    try:
        return _strategy(svc.strategies.update(user.username, strategy_id, strategy))
    except StrategyNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no strategy {strategy_id!r}") from e
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(
    strategy_id: str, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, str]:
    try:
        moved = svc.strategies.delete(user.username, strategy_id)
    except StrategyNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no strategy {strategy_id!r}") from e
    return {"deleted": strategy_id, "recoverable_at": str(moved)}


# --------------------------------------------------------------------------- backtests


@router.post("/backtests", status_code=status.HTTP_202_ACCEPTED)
async def submit_backtest(
    req: BacktestRequest, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        strategies = svc.strategies.get_many(user.username, req.strategy_ids)
    except StrategyNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no strategy {e.args[0]!r}") from e
    return svc.jobs.submit(user.id, req, strategies)


@router.get("/backtests")
async def list_backtests(
    user: User = Depends(current_user), svc: Services = Depends(services)
) -> list[dict[str, Any]]:
    return svc.jobs.list(user.id)


def _job(job_id: str, user: User, svc: Services) -> dict[str, Any]:
    job = svc.jobs.get(user.id, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no backtest {job_id!r}")
    return job


def _finished(job: dict[str, Any]) -> str:
    if job["status"] != "done" or not job["run_id"]:
        raise HTTPException(status.HTTP_409_CONFLICT, f"backtest is {job['status']}")
    return job["run_id"]


@router.get("/backtests/{job_id}")
async def get_backtest(
    job_id: str, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    job = _job(job_id, user, svc)
    if job["status"] == "done" and job["run_id"]:
        job["summary"] = svc.jobs.summary(job["run_id"])
    return job


def _read_csv(path: Any) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


@router.get("/backtests/{job_id}/equity")
async def backtest_equity(
    job_id: str, user: User = Depends(current_user), svc: Services = Depends(services)
) -> list[dict[str, Any]]:
    run_id = _finished(_job(job_id, user, svc))
    rows = _read_csv(svc.jobs.result_dir(run_id) / "equity.csv")
    return [{k: (v if k == "ts" else float(v)) for k, v in r.items() if v != ""} for r in rows]


@router.get("/backtests/{job_id}/trades")
async def backtest_trades(
    job_id: str, user: User = Depends(current_user), svc: Services = Depends(services)
) -> list[dict[str, str]]:
    run_id = _finished(_job(job_id, user, svc))
    return _read_csv(svc.jobs.result_dir(run_id) / "trades.csv")


# --------------------------------------------------------------------------- engines & accounts


class StartRequest(BaseModel):
    strategy_ids: list[str] = Field(min_length=1)
    confirm: str | None = None  # live only: must be "LIVE"


@router.get("/engines")
async def engines(
    user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    return {m.value: svc.engines.status(user, m) for m in EngineMode}


@router.post("/engines/{mode}/start")
async def start_engine(
    mode: EngineMode,
    req: StartRequest,
    user: User = Depends(current_user),
    svc: Services = Depends(services),
) -> dict[str, Any]:
    try:
        return await svc.engines.start(user, mode, req.strategy_ids, confirm=req.confirm)
    except EngineError as e:
        raise _engine_error(e) from e


@router.post("/engines/{mode}/stop")
async def stop_engine(
    mode: EngineMode, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        return await svc.engines.stop(user, mode)
    except EngineError as e:
        raise _engine_error(e) from e


@router.post("/engines/{mode}/flatten")
async def flatten(
    mode: EngineMode, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        return await svc.engines.flatten(user, mode)
    except EngineError as e:
        raise _engine_error(e) from e


@router.get("/accounts/{mode}")
async def account(
    mode: EngineMode, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        return await svc.engines.account(user, mode)
    except EngineError as e:
        raise _engine_error(e) from e


class ResetRequest(BaseModel):
    confirm: str


@router.post("/accounts/paper/reset")
async def reset_paper(
    req: ResetRequest, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, str]:
    try:
        svc.engines.reset_paper(user, req.confirm)
    except EngineError as e:
        raise _engine_error(e) from e
    return {"reset": f"paper-{user.username}"}


# --------------------------------------------------------------------------- control


class KillRequest(BaseModel):
    reason: str = Field(default="kill switch", max_length=200)


@router.get("/control")
async def control_state(svc: Services = Depends(services)) -> dict[str, Any]:
    return svc.kill.state().__dict__


@router.post("/control/kill")
async def kill(
    req: KillRequest, user: User = Depends(current_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    return await svc.engines.kill(user, req.reason)


@router.post("/control/resume")
async def resume(
    user: User = Depends(admin_user), svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        return await svc.engines.resume(user)
    except EngineError as e:
        raise _engine_error(e) from e


# --------------------------------------------------------------------------- models


def _registry_error(e: RegistryError) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, str(e))


@router.get("/models")
async def models(svc: Services = Depends(services)) -> list[dict[str, Any]]:
    reg = svc.registry
    return [
        {"name": n, "live": reg.live_version(n), "versions": len(reg.versions(n))}
        for n in reg.names()
    ]


@router.get("/models/{name}")
async def model(name: str, svc: Services = Depends(services)) -> dict[str, Any]:
    reg = svc.registry
    try:
        versions = reg.versions(name)
    except RegistryError as e:
        raise _registry_error(e) from e
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no model {name!r}")
    rows = []
    for v in versions:
        meta = reg.metadata(name, v)["model"]
        decisions = reg.decisions(name, v)
        rows.append(
            {
                "version": v,
                "kind": meta["kind"],
                "trained_to": meta["train_window"]["end"],
                "oos_ic": meta["cv"]["oos_ic"],
                "gate": None if not decisions else decisions[-1]["promote"],
                "holdout": decisions[-1]["candidate"] if decisions else None,
            }
        )
    return {
        "name": name,
        "live": reg.live_info(name),
        "versions": rows,
        "history": reg.history(name),
    }


@router.get("/models/{name}/{version}")
async def model_version(
    name: str, version: str, svc: Services = Depends(services)
) -> dict[str, Any]:
    try:
        return svc.registry.metadata(name, version) | {
            "decisions": svc.registry.decisions(name, version)
        }
    except RegistryError as e:
        raise _registry_error(e) from e


class RollbackRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


@router.post("/models/{name}/rollback")
async def rollback(
    name: str,
    req: RollbackRequest,
    user: User = Depends(admin_user),
    svc: Services = Depends(services),
) -> dict[str, str]:
    try:
        version = svc.registry.rollback(name, reason=req.reason, actor=user.username)
    except RegistryError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    return {"live": version}


class PromoteRequest(BaseModel):
    version: str
    reason: str = Field(min_length=3, max_length=200)
    confirm: bool = False  # bypasses the promotion gate


@router.post("/models/{name}/promote")
async def promote(
    name: str,
    req: PromoteRequest,
    user: User = Depends(admin_user),
    svc: Services = Depends(services),
) -> dict[str, str]:
    if not req.confirm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "manual promotion bypasses the gate: set confirm"
        )
    try:
        svc.registry.set_live(
            name, req.version, reason=f"manual: {req.reason}", actor=user.username
        )
    except RegistryError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"live": req.version}


# --------------------------------------------------------------------------- SEBI-gated stubs

gated = APIRouter(prefix="/api", dependencies=[Depends(current_user)])


@gated.get("/marketplace")
@gated.get("/auto-trading")
@gated.get("/ai-strategies")
async def not_yet() -> Response:
    return Response(
        '{"detail": "not available: needs SEBI algo-provider registration"}',
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        media_type="application/json",
    )
