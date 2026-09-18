"""Settings loaded from the environment / ``.env`` (pydantic-settings).

Secrets never live in code. See ``.env.example`` for every key.
"""

from __future__ import annotations

from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # trading mode
    live_trading: bool = False
    broker: Literal["paper", "dhan"] = "paper"

    # dhan
    dhan_client_id: str = ""
    dhan_access_token: SecretStr = SecretStr("")
    dhan_feed_mode: Literal["ticker", "quote", "full"] = "full"

    # infrastructure
    redis_url: str | None = None
    db_url: str = "sqlite:///data/state.db"
    archive_dir: Path = Path("data/archive")
    instruments_dir: Path = Path("data/instruments")
    holidays_file: Path = Path("data/holidays/holidays.json")
    corporate_actions_file: Path = Path("data/corporate_actions.json")

    # paper trading
    paper_starting_cash: float = 1_000_000.0
    paper_slippage_bps: float = 2.0

    # market hours
    mcx_close: time = time(23, 30)

    # feature flags
    feature_marketplace: bool = False

    # misc
    log_level: str = "INFO"

    @field_validator("redis_url", mode="before")
    @classmethod
    def _empty_is_none(cls, v: str | None) -> str | None:
        return v or None

    @property
    def live_orders_allowed(self) -> bool:
        """Constraint 5: real orders only with LIVE_TRADING=true AND broker=dhan.
        The third condition (interactive confirmation) is checked at engine start."""
        return self.live_trading and self.broker == "dhan"

    def problems(self) -> list[str]:
        """Configuration problems that should stop startup."""
        out: list[str] = []
        if self.broker == "dhan" and not (
            self.dhan_client_id and self.dhan_access_token.get_secret_value()
        ):
            out.append("BROKER=dhan requires DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")
        if self.live_trading and self.broker != "dhan":
            out.append("LIVE_TRADING=true is only meaningful with BROKER=dhan")
        if self.paper_starting_cash <= 0:
            out.append("PAPER_STARTING_CASH must be positive")
        if not self.holidays_file.exists():
            out.append(f"HOLIDAYS_FILE not found: {self.holidays_file}")
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
