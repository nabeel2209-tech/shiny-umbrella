"""The HTMX dashboard: full pages, and the small HTML fragments HTMX swaps in.

Pages need a signed-in user (else a redirect to /login). Fragment endpoints live
under /ui and follow the same CSRF rule as the JSON API.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from trading.api.auth import COOKIE_NAME, AuthError, User
from trading.api.builder import GROUP_TITLES, RULE_GROUPS, form_rows, form_to_dict
from trading.api.charts import line_chart, pct_fmt
from trading.api.deps import page_user, services
from trading.api.engines import EngineError, EngineMode, paper_account
from trading.api.formatting import FILTERS
from trading.api.jobs import BacktestRequest
from trading.api.routes.api import set_session_cookie
from trading.api.services import Services
from trading.api.strategy_store import (
    StrategyExists,
    StrategyNotFound,
    parse_strategy_yaml,
    validate_strategy_dict,
)
from trading.core.types import IST, Interval, ProductType, Urgency, now_ist
from trading.features.features import feature_names
from trading.strategies.schema import Op, SizingMode, StrategyConfig
from trading.training.registry import RegistryError

TEMPLATES = Jinja2Templates(
    directory=str(Path(__file__).resolve().parents[2] / "web" / "templates")
)
TEMPLATES.env.filters.update(FILTERS)
TEMPLATES.env.globals["OP_LABELS"] = {
    "gt": ">",
    "gte": "\u2265",
    "lt": "<",
    "lte": "\u2264",
    "eq": "=",
    "neq": "\u2260",
    "abs_gt": "|x| >",
    "abs_lt": "|x| <",
}

router = APIRouter()

NAV = [
    ("/builder", "Strategy Builder", True),
    ("/backtests", "Backtesting", True),
    ("/algo", "Algo Trading", True),
    ("/practise", "Practise", True),
    ("/models", "Models", True),
    ("/how-it-works", "How It Works", True),
    ("/marketplace", "Marketplace", False),
    ("/auto-trading", "Auto Trading", False),
    ("/ai-strategies", "AI Strategies", False),
]


def render(request: Request, template: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
    svc: Services = request.app.state.services
    user: User | None = getattr(request.state, "user", None)
    sid = getattr(request.state, "sid", None)
    base = {
        "user": user,
        "csrf": svc.auth.csrf_token(sid) if sid else "",
        "nav": NAV,
        "path": request.url.path,
        "kill": svc.kill.state(),
        "settings": svc.settings,
    }
    return TEMPLATES.TemplateResponse(request, template, base | ctx, status_code=status_code)


def fragment(request: Request, template: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
    return render(request, template, status_code, **ctx)


def hx_redirect(url: str) -> Response:
    return Response(status_code=204, headers={"HX-Redirect": url})


def _safe_next(url: str | None) -> str:
    """Only local paths: never let ?next= send someone to another site. Browsers
    read a backslash as a slash, so ``/\\evil.example`` would be ``//evil.example``."""
    if (
        not url
        or not url.startswith("/")
        or url.startswith("//")
        or "\\" in url
        or any(ord(c) < 32 for c in url)
        or urlparse(url).netloc
    ):
        return "/"
    return url


# =========================================================================== auth


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/") -> HTMLResponse:
    return render(request, "login.html", next=_safe_next(next), error=None)


@router.post("/login")
async def login_form(request: Request, svc: Services = Depends(services)) -> Response:
    form = await request.form()
    next_url = _safe_next(str(form.get("next") or "/"))
    try:
        _, token = svc.auth.login(str(form.get("username", "")), str(form.get("password", "")))
    except AuthError as e:
        return render(request, "login.html", status_code=401, next=next_url, error=str(e))
    response = RedirectResponse(next_url, status_code=303)
    set_session_cookie(response, token, svc)
    return response


@router.post("/logout")
async def logout(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> Response:
    svc.auth.logout(request.state.sid)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


# =========================================================================== home


@router.get("/", response_class=HTMLResponse)
async def home(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    engines = {m.value: svc.engines.status(user, m) for m in EngineMode}
    return render(
        request,
        "home.html",
        engines=engines,
        strategies=svc.strategies.list(user.username),
        jobs=svc.jobs.list(user.id, limit=5),
        models=[(n, svc.registry.live_version(n)) for n in svc.registry.names()],
    )


# =========================================================================== strategy builder


def _all_features() -> list[str]:
    """Intraday features, then the ones only daily bars have."""
    intraday = feature_names(interval=Interval.M5)
    return intraday + [f for f in feature_names(interval=Interval.D1) if f not in intraday]


def _builder_ctx(
    svc: Services, user: User, strategy: StrategyConfig | None, *, editing: bool
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "editing": editing,
        "rows": form_rows(strategy),
        "groups": RULE_GROUPS,
        "group_titles": GROUP_TITLES,
        "features": _all_features(),
        "ops": [o.value for o in Op],
        "intervals": [i.value for i in Interval],
        "products": [p.value for p in ProductType],
        "urgencies": [u.value for u in Urgency],
        "sizing_modes": [m.value for m in SizingMode],
        "models": svc.registry.names(),
        "strategies": svc.strategies.list(user.username),
        "templates": svc.strategies.templates(),
    }


@router.get("/builder", response_class=HTMLResponse)
async def builder(
    request: Request,
    template: str | None = None,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    start = None
    if template:
        start = next((t for t in svc.strategies.templates() if t.id == template), None)
    return render(request, "builder.html", **_builder_ctx(svc, user, start, editing=False))


@router.get("/builder/{strategy_id}", response_class=HTMLResponse)
async def builder_edit(
    strategy_id: str,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    try:
        strategy = svc.strategies.get(user.username, strategy_id)
    except StrategyNotFound as e:
        raise HTTPException(404, f"no strategy {strategy_id!r}") from e
    return render(request, "builder.html", **_builder_ctx(svc, user, strategy, editing=True))


@router.get("/ui/builder/row", response_class=HTMLResponse)
async def builder_row(
    request: Request, group: str, user: User = Depends(page_user)
) -> HTMLResponse:
    if group not in RULE_GROUPS:
        raise HTTPException(404)
    return fragment(
        request,
        "partials/condition_row.html",
        group=group,
        row={"feature": "", "op": "gt", "value": ""},
        features=_all_features(),
        ops=[o.value for o in Op],
    )


def _strategy_from_form(form: Any) -> tuple[StrategyConfig | None, list[str]]:
    data, errors = form_to_dict(form)
    if errors:
        return None, errors
    strategy, errors = validate_strategy_dict(data)
    if strategy is not None:
        unknown = strategy.validate_features(feature_names(interval=strategy.interval))
        if unknown:
            return None, [
                f"unknown feature for {strategy.interval.value} bars: {f}" for f in unknown
            ]
    return strategy, errors


@router.post("/ui/builder/preview", response_class=HTMLResponse)
async def builder_preview(request: Request, user: User = Depends(page_user)) -> HTMLResponse:
    strategy, errors = _strategy_from_form(await request.form())
    return fragment(request, "partials/yaml_preview.html", strategy=strategy, errors=errors)


@router.post("/ui/builder/save", response_class=HTMLResponse)
async def builder_save(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> Response:
    form = await request.form()
    strategy, errors = _strategy_from_form(form)
    if strategy is None:
        return fragment(
            request, "partials/yaml_preview.html", status_code=422, strategy=None, errors=errors
        )
    return _save(request, svc, user, strategy, editing=bool(form.get("editing")))


@router.post("/ui/builder/save-yaml", response_class=HTMLResponse)
async def builder_save_yaml(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> Response:
    form = await request.form()
    strategy, errors = parse_strategy_yaml(str(form.get("yaml", "")))
    if strategy is not None:
        unknown = strategy.validate_features(feature_names(interval=strategy.interval))
        errors += [f"unknown feature: {f}" for f in unknown]
    if strategy is None or errors:
        return fragment(
            request, "partials/yaml_preview.html", status_code=422, strategy=None, errors=errors
        )
    existing = {s.id for s in svc.strategies.list(user.username)}
    return _save(request, svc, user, strategy, editing=strategy.id in existing)


def _save(
    request: Request, svc: Services, user: User, strategy: StrategyConfig, *, editing: bool
) -> Response:
    try:
        if editing:
            svc.strategies.update(user.username, strategy.id, strategy)
        else:
            svc.strategies.create(user.username, strategy)
    except StrategyExists:
        return fragment(
            request,
            "partials/yaml_preview.html",
            status_code=409,
            strategy=None,
            errors=[
                f"a strategy called {strategy.id!r} already exists - pick another id or edit it"
            ],
        )
    except (StrategyNotFound, ValueError) as e:
        return fragment(
            request, "partials/yaml_preview.html", status_code=422, strategy=None, errors=[str(e)]
        )
    return hx_redirect(f"/builder/{strategy.id}?saved=1")


@router.post("/ui/strategies/{strategy_id}/delete")
async def delete_strategy(
    strategy_id: str, user: User = Depends(page_user), svc: Services = Depends(services)
) -> Response:
    try:
        svc.strategies.delete(user.username, strategy_id)
    except StrategyNotFound as e:
        raise HTTPException(404) from e
    return hx_redirect("/builder?deleted=" + strategy_id)


# =========================================================================== backtests


@router.get("/backtests", response_class=HTMLResponse)
async def backtests(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    today = now_ist().date()
    return render(
        request,
        "backtests.html",
        strategies=svc.strategies.list(user.username),
        jobs=svc.jobs.list(user.id),
        default_start=(today - timedelta(days=30)).isoformat(),
        default_end=(today - timedelta(days=1)).isoformat(),
        error=None,
    )


@router.post("/ui/backtests", response_class=HTMLResponse)
async def submit_backtest(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    form = await request.form()
    error = None
    try:
        req = BacktestRequest(
            strategy_ids=form.getlist("strategy_ids"),
            start=date.fromisoformat(str(form.get("start"))),
            end=date.fromisoformat(str(form.get("end"))),
            name=str(form.get("name", "")),
            initial_cash=float(form.get("initial_cash") or 1_000_000),
            slippage_bps=float(form.get("slippage_bps") or 2.0),
            impact=bool(form.get("impact")),
            participation=float(form["participation"]) / 100 if form.get("participation") else None,
            liquidate=bool(form.get("liquidate")),
        )
        strategies = svc.strategies.get_many(user.username, req.strategy_ids)
        svc.jobs.submit(user.id, req, strategies)
    except ValidationError as e:
        error = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
    except (ValueError, TypeError) as e:
        error = f"check the dates and numbers: {e}"
    except StrategyNotFound as e:
        error = f"no strategy {e.args[0]!r}"
    return fragment(
        request,
        "partials/jobs.html",
        status_code=200 if error is None else 422,
        jobs=svc.jobs.list(user.id),
        error=error,
    )


@router.get("/ui/backtests/jobs", response_class=HTMLResponse)
async def jobs_fragment(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    return fragment(request, "partials/jobs.html", jobs=svc.jobs.list(user.id), error=None)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


@router.get("/backtests/{job_id}", response_class=HTMLResponse)
async def backtest_detail(
    job_id: str,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    job = svc.jobs.get(user.id, job_id)
    if job is None:
        raise HTTPException(404, f"no backtest {job_id!r}")
    ctx: dict[str, Any] = {"job": job, "summary": None}
    if job["status"] == "done" and job["run_id"]:
        run_dir = svc.jobs.result_dir(job["run_id"])
        summary = svc.jobs.summary(job["run_id"])
        equity = _read_csv(run_dir / "equity.csv")
        stamps = [datetime.fromisoformat(r["ts"]).astimezone(IST) for r in equity]
        values = [float(r["equity"]) for r in equity]
        drawdowns = [float(r.get("drawdown_pct") or 0.0) for r in equity]
        start_cash = float(summary["config"]["initial_cash"])
        charts = {}
        if values:
            charts["equity_chart"] = line_chart(
                stamps,
                values,
                chart_id="equity",
                reference=start_cash,
            )
            charts["drawdown_chart"] = line_chart(
                stamps,
                drawdowns,
                chart_id="drawdown",
                fmt=pct_fmt,
                reference=0.0,
                area_to="reference",
            )
        ctx.update(
            summary=summary,
            m=summary["metrics"],
            daily=_daily_rows(stamps, values, drawdowns),
            trades=_read_csv(run_dir / "trades.csv")[:500],
            **charts,
        )
    return render(request, "backtest_detail.html", **ctx)


def _daily_rows(
    stamps: list[datetime], values: list[float], drawdowns: list[float]
) -> list[dict[str, Any]]:
    """End-of-day rows: the chart's table view."""
    rows: dict[date, dict[str, Any]] = {}
    for ts, v, dd in zip(stamps, values, drawdowns, strict=True):
        rows[ts.date()] = {"day": ts.date(), "equity": v, "drawdown": dd}
    return list(rows.values())


# =========================================================================== trading pages


def _engine_ctx(
    svc: Services, user: User, mode: EngineMode, error: str | None = None
) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "status": svc.engines.status(user, mode),
        "strategies": svc.strategies.list(user.username),
        "account_name": paper_account(user),
        "error": error,
    }


@router.get("/practise", response_class=HTMLResponse)
async def practise(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    return render(request, "trading.html", **_engine_ctx(svc, user, EngineMode.PAPER))


@router.get("/algo", response_class=HTMLResponse)
async def algo(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    return render(request, "trading.html", **_engine_ctx(svc, user, EngineMode.LIVE))


@router.get("/ui/engines/{mode}/panel", response_class=HTMLResponse)
async def engine_panel(
    mode: EngineMode,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    return fragment(request, "partials/engine_panel.html", **_engine_ctx(svc, user, mode))


@router.post("/ui/engines/{mode}/{action}", response_class=HTMLResponse)
async def engine_action(
    mode: EngineMode,
    action: str,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    form = await request.form()
    error = notice = None
    try:
        if action == "start":
            await svc.engines.start(
                user, mode, form.getlist("strategy_ids"), confirm=str(form.get("confirm") or "")
            )
        elif action == "stop":
            await svc.engines.stop(user, mode)
        elif action == "flatten":
            exits = (await svc.engines.flatten(user, mode))["exits"]
            notice = f"sent {len(exits)} exit order(s) through the risk agent"
        else:
            raise HTTPException(404)
    except EngineError as e:
        error = e.message
    ctx = _engine_ctx(svc, user, mode, error)
    return fragment(request, "partials/engine_panel.html", notice=notice, **ctx)


@router.get("/ui/accounts/{mode}", response_class=HTMLResponse)
async def account_fragment(
    mode: EngineMode,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    try:
        account = await svc.engines.account(user, mode)
    except EngineError as e:
        return fragment(
            request, "partials/account.html", account=None, mode=mode.value, message=e.message
        )
    return fragment(
        request, "partials/account.html", account=account, mode=mode.value, message=None
    )


@router.post("/ui/accounts/paper/reset", response_class=HTMLResponse)
async def reset_paper(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    form = await request.form()
    try:
        svc.engines.reset_paper(user, str(form.get("confirm", "")))
        message = "paper account reset to its starting cash"
    except EngineError as e:
        message = e.message
    account = await svc.engines.account(user, EngineMode.PAPER)
    return fragment(
        request, "partials/account.html", account=account, mode="paper", message=message
    )


# =========================================================================== kill switch


@router.get("/ui/control/banner", response_class=HTMLResponse)
async def banner(request: Request, user: User = Depends(page_user)) -> HTMLResponse:
    return fragment(request, "partials/kill_banner.html")


@router.post("/ui/control/{action}", response_class=HTMLResponse)
async def control(
    action: str,
    request: Request,
    user: User = Depends(page_user),
    svc: Services = Depends(services),
) -> HTMLResponse:
    form = await request.form()
    error = None
    if action == "kill":
        await svc.engines.kill(
            user, str(form.get("reason") or "kill switch pressed on the dashboard")
        )
    elif action == "resume":
        try:
            await svc.engines.resume(user)
        except EngineError as e:
            error = e.message
    else:
        raise HTTPException(404)
    return fragment(request, "partials/kill_banner.html", error=error)


# =========================================================================== models


@router.get("/models", response_class=HTMLResponse)
async def models(
    request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    reg = svc.registry
    rows = [
        {"name": n, "live": reg.live_version(n), "versions": len(reg.versions(n))}
        for n in reg.names()
    ]
    return render(request, "models.html", models=rows)


@router.get("/models/{name}", response_class=HTMLResponse)
async def model_detail(
    name: str, request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> HTMLResponse:
    reg = svc.registry
    try:
        versions = reg.versions(name)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e
    if not versions:
        raise HTTPException(404, f"no model {name!r}")
    rows = []
    for v in reversed(versions):
        meta = reg.metadata(name, v)
        decisions = reg.decisions(name, v)
        rows.append(
            {"version": v, "meta": meta["model"], "decision": decisions[-1] if decisions else None}
        )
    return render(
        request,
        "model_detail.html",
        name=name,
        live=reg.live_info(name),
        versions=rows,
        history=list(reversed(reg.history(name)))[:50],
    )


@router.post("/ui/models/{name}/rollback")
async def model_rollback(
    name: str, request: Request, user: User = Depends(page_user), svc: Services = Depends(services)
) -> Response:
    if not user.is_admin:
        raise HTTPException(403, "admin only")
    form = await request.form()
    reason = str(form.get("reason", "")).strip() or "rolled back from the dashboard"
    try:
        svc.registry.rollback(name, reason=reason, actor=user.username)
    except RegistryError as e:
        raise HTTPException(409, str(e)) from e
    return hx_redirect(f"/models/{name}")


# =========================================================================== static pages


@router.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request, user: User = Depends(page_user)) -> HTMLResponse:
    return render(request, "how_it_works.html")


COMING_SOON = {
    "marketplace": ("Marketplace", "Share and subscribe to strategies built by other traders."),
    "auto-trading": (
        "Auto Trading",
        "Let a strategy you subscribe to trade your account automatically.",
    ),
    "ai-strategies": ("AI Strategies", "Strategies designed and maintained by models."),
}


def _coming_soon(page: str):  # type: ignore[no-untyped-def]
    title, blurb = COMING_SOON[page]

    async def view(request: Request, user: User = Depends(page_user)) -> HTMLResponse:
        return render(request, "coming_soon.html", title=title, blurb=blurb)

    return view


for _page in COMING_SOON:
    router.add_api_route(f"/{_page}", _coming_soon(_page), response_class=HTMLResponse)
