"""Run the dashboard and the API.

    python -m scripts.serve                  # needs DASHBOARD_PASSWORD in .env
    python -m scripts.serve --dev            # makes up an admin password, prints it once

Then open http://127.0.0.1:8000. The first start creates the admin user from
DASHBOARD_USER / DASHBOARD_PASSWORD; later starts leave existing users alone.

Live trading from the dashboard needs LIVE_TRADING=true and BROKER=dhan, and this
script asks you to type LIVE at startup (constraint 5). Decline, or run it without a
terminal, and the dashboard can only trade on paper for the life of the process;
each live start also needs LIVE typed on the Algo Trading page.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys

import uvicorn

from trading.agents.engine import confirm_live_trading
from trading.api.app import create_app
from trading.api.services import build_services
from trading.core.config import Settings, get_settings

log = logging.getLogger("serve")


def live_gate(settings: Settings, prompt=input) -> Settings:  # type: ignore[no-untyped-def]
    """Settings with live trading switched off unless a human confirms it now."""
    if not settings.live_trading:
        return settings
    if not sys.stdin.isatty() and prompt is input:
        log.warning("LIVE_TRADING is true but there is no terminal to confirm it: paper only")
        return settings.model_copy(update={"live_trading": False})
    if confirm_live_trading(settings, settings.broker, prompt=prompt):
        return settings
    return settings.model_copy(update={"live_trading": False})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", help="default: API_HOST (127.0.0.1)")
    parser.add_argument("--port", type=int, help="default: API_PORT (8000)")
    parser.add_argument(
        "--dev", action="store_true", help="invent an admin password if none is configured"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = get_settings()
    for problem in settings.problems():
        log.warning("config: %s", problem)
    settings = live_gate(settings)
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    services = build_services(settings)
    username = settings.dashboard_user
    password = settings.dashboard_password.get_secret_value()
    admin = None
    if services.auth.get_by_username(username) is None and not password:
        if not args.dev:
            log.error(
                "no dashboard user yet: set DASHBOARD_PASSWORD in .env (or use --dev to "
                "generate a throwaway one)"
            )
            return 2
        password = secrets.token_urlsafe(12)
        admin = (username, password)
        print(f"\n  dev login: {username} / {password}  (shown once)\n", flush=True)
    elif services.auth.get_by_username(username) is not None:
        log.info(
            "signing in as %r with the password it was created with; forgotten it? "
            "python -m scripts.set_password",
            username,
        )

    if host not in ("127.0.0.1", "localhost", "::1") and not settings.cookie_secure:
        log.warning(
            "serving on %s without secure cookies: put it behind HTTPS and set COOKIE_SECURE=true",
            host,
        )
    app = create_app(settings, services=services, admin=admin)
    uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
