"""Set or reset a dashboard user's password.

    python -m scripts.set_password              # the DASHBOARD_USER (admin by default)
    python -m scripts.set_password --user bob

Asks for the new password twice without echoing it. It works while the server is
running; the user's open sessions are ended, so they sign in again. On a database
with no users yet it creates DASHBOARD_USER as the admin.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import secrets
import sys

from trading.api.auth import AuthStore, Role
from trading.core.config import get_settings


def main(argv: list[str] | None = None, *, ask=getpass.getpass) -> int:  # type: ignore[no-untyped-def]
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user", default=settings.dashboard_user)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # the secret only signs session tokens, which this script never issues
    secret = settings.session_secret.get_secret_value() or secrets.token_hex(32)
    store = AuthStore(settings.db_url, secret)
    exists = store.get_by_username(args.user) is not None
    if not exists and store.users():
        names = ", ".join(u.username for u in store.users())
        print(f"no user {args.user!r} (users: {names})", file=sys.stderr)
        return 1

    password = ask(f"New password for {args.user}: ")
    if password != ask("Again: "):
        print("the two passwords differ; nothing changed", file=sys.stderr)
        return 1
    try:
        if exists:
            store.set_password(args.user, password)
        else:
            store.create_user(args.user, password, Role.ADMIN)
    except ValueError as e:
        print(f"{e}; nothing changed", file=sys.stderr)
        return 1
    print(f"{'password changed' if exists else 'admin created'} for {args.user}; sign in at "
          f"http://{settings.api_host}:{settings.api_port}/login")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
