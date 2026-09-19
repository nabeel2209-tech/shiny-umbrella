"""Monitor agent: heartbeats, alert log, kill switch, execution quality.

It writes nothing to a broker. Its jobs are to notice when something has gone
quiet, to keep an auditable alert log, to expose the kill switch that the risk and
execution agents obey, and to measure slippage - the achieved fill price against
the mid at the moment the signal fired, which is the number that tells you whether
a backtest is lying to you.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import fmean

from trading.agents.base import Agent
from trading.agents.portfolio import Portfolio
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock
from trading.core.types import (
    Alert,
    AlertLevel,
    ControlCommand,
    Fill,
    Heartbeat,
    Order,
    OrderIntent,
    RiskRejection,
    Side,
    Tick,
)

log = logging.getLogger(__name__)


@dataclass
class SlippageRecord:
    ts: datetime
    symbol: str
    strategy_id: str
    side: Side
    qty: int
    mid_at_signal: float
    achieved: float

    @property
    def slippage_bps(self) -> float:
        """Signed cost in basis points: positive means we paid up."""
        if self.mid_at_signal <= 0:
            return 0.0
        raw = (self.achieved - self.mid_at_signal) / self.mid_at_signal
        return raw * 10_000 * self.side.sign


@dataclass
class MonitorConfig:
    heartbeat_timeout_seconds: float = 60.0
    alert_log_size: int = 500
    slippage_log_size: int = 1000
    watch_agents: list[str] = field(default_factory=list)


class MonitorAgent(Agent):
    name = "monitor"

    def __init__(
        self,
        bus: MessageBus,
        portfolio: Portfolio | None = None,
        config: MonitorConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(bus, clock=clock)
        self.portfolio = portfolio
        self.cfg = config or MonitorConfig()
        self.alerts: deque[Alert] = deque(maxlen=self.cfg.alert_log_size)
        self.slippage: deque[SlippageRecord] = deque(maxlen=self.cfg.slippage_log_size)
        self.last_heartbeat: dict[str, datetime] = {}
        self.killed = False
        self.rejections: dict[str, int] = {}
        self._intents: dict[str, OrderIntent] = {}
        self._order_intent: dict[str, str] = {}
        self._mid_at_signal: dict[str, float] = {}
        self._protective_orders: set[str] = set()
        self._quotes: dict[str, float] = {}
        self.counts: dict[str, int] = {"intents": 0, "orders": 0, "fills": 0, "rejections": 0}

    async def on_start(self) -> None:
        await self.subscribe(Topics.ALERTS, self._on_alert)
        await self.subscribe(Topics.HEARTBEAT, self._on_heartbeat)
        await self.subscribe(Topics.INTENTS, self._on_intent)
        await self.subscribe(Topics.REJECTED, self._on_rejection)
        await self.subscribe(Topics.ORDERS, self._on_order)
        await self.subscribe(Topics.FILLS, self._on_fill)
        await self.subscribe(Topics.TICKS_ALL, self._on_tick)
        await self.subscribe(Topics.CONTROL, self._on_control)

    # ------------------------------------------------------------------ inbound
    async def _on_alert(self, _topic: str, alert: Alert) -> None:  # type: ignore[override]
        self.alerts.append(alert)
        level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARN: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }[alert.level]
        self.log.log(level, "[%s] %s %s", alert.source, alert.message, alert.data or "")

    async def _on_heartbeat(self, _topic: str, hb: Heartbeat) -> None:  # type: ignore[override]
        self.last_heartbeat[hb.agent] = hb.ts

    async def _on_intent(self, _topic: str, intent: OrderIntent) -> None:  # type: ignore[override]
        self._intents[intent.id] = intent
        self.counts["intents"] += 1
        self._mid_at_signal[intent.id] = self._quotes.get(intent.symbol, intent.reference_price)

    async def _on_rejection(self, _topic: str, rej: RiskRejection) -> None:  # type: ignore[override]
        self.counts["rejections"] += 1
        self.rejections[rej.rule] = self.rejections.get(rej.rule, 0) + 1

    async def _on_order(self, _topic: str, order: Order) -> None:  # type: ignore[override]
        self.counts["orders"] += 1
        if order.meta.get("protective"):
            # a stop-loss exit says nothing about how well we executed the signal
            self._protective_orders.add(order.id)
            return
        if order.intent_id:
            self._order_intent[order.id] = order.intent_id

    async def _on_tick(self, _topic: str, tick: Tick) -> None:  # type: ignore[override]
        self._quotes[tick.symbol] = tick.mid

    async def _on_fill(self, _topic: str, fill: Fill) -> None:  # type: ignore[override]
        self.counts["fills"] += 1
        if fill.order_id in self._protective_orders:
            return
        intent_id = self._order_intent.get(fill.order_id)
        intent = self._intents.get(intent_id) if intent_id else None
        mid = self._mid_at_signal.get(intent_id or "", None)
        if intent is None or mid is None:
            return
        self.slippage.append(
            SlippageRecord(
                ts=fill.ts,
                symbol=fill.symbol,
                strategy_id=intent.strategy_id,
                side=fill.side,
                qty=fill.qty,
                mid_at_signal=mid,
                achieved=fill.price,
            )
        )

    async def _on_control(self, _topic: str, cmd: ControlCommand) -> None:  # type: ignore[override]
        command = cmd.command.upper()
        if command == "KILL":
            self.killed = True
        elif command == "RESUME":
            self.killed = False

    # ------------------------------------------------------------------ kill switch
    async def kill(self, reason: str, issued_by: str = "monitor") -> ControlCommand:
        """Halt all trading. Risk stops approving, execution cancels what is working."""
        cmd = ControlCommand(
            ts=self.clock.now(), command="KILL", reason=reason, issued_by=issued_by
        )
        await self.publish(Topics.CONTROL, cmd)
        await self.alert(AlertLevel.CRITICAL, f"kill switch: {reason}")
        return cmd

    async def resume(self, issued_by: str = "monitor") -> ControlCommand:
        cmd = ControlCommand(ts=self.clock.now(), command="RESUME", issued_by=issued_by)
        await self.publish(Topics.CONTROL, cmd)
        return cmd

    async def flatten(self, reason: str = "flatten") -> ControlCommand:
        cmd = ControlCommand(ts=self.clock.now(), command="FLATTEN", reason=reason)
        await self.publish(Topics.CONTROL, cmd)
        return cmd

    # ------------------------------------------------------------------ health
    def stale_agents(self, now: datetime | None = None) -> list[str]:
        """Agents that were expected to beat and have gone quiet."""
        now = now or self.clock.now()
        cutoff = now - timedelta(seconds=self.cfg.heartbeat_timeout_seconds)
        stale = [a for a, ts in self.last_heartbeat.items() if ts < cutoff]
        stale += [a for a in self.cfg.watch_agents if a not in self.last_heartbeat]
        return sorted(set(stale))

    async def check_health(self, now: datetime | None = None) -> list[str]:
        stale = self.stale_agents(now)
        for agent in stale:
            await self.alert(AlertLevel.WARN, f"no heartbeat from {agent}", agent=agent)
        return stale

    def slippage_summary(self, strategy_id: str | None = None) -> dict[str, float]:
        rows = [r for r in self.slippage if strategy_id is None or r.strategy_id == strategy_id]
        if not rows:
            return {"count": 0, "mean_bps": 0.0, "worst_bps": 0.0, "total_cost": 0.0}
        bps = [r.slippage_bps for r in rows]
        return {
            "count": len(rows),
            "mean_bps": round(fmean(bps), 3),
            "worst_bps": round(max(bps), 3),
            "total_cost": round(
                sum(r.slippage_bps / 10_000 * r.mid_at_signal * r.qty for r in rows), 2
            ),
        }

    def status(self) -> dict[str, object]:
        out: dict[str, object] = {
            "killed": self.killed,
            "counts": dict(self.counts),
            "rejections": dict(self.rejections),
            "stale_agents": self.stale_agents(),
            "slippage": self.slippage_summary(),
            "alerts": len(self.alerts),
        }
        if self.portfolio is not None:
            out["portfolio"] = {
                "equity": round(self.portfolio.equity, 2),
                "day_pnl": round(self.portfolio.day_pnl, 2),
                "gross_exposure": round(self.portfolio.gross_exposure(), 2),
                "open_positions": len(self.portfolio.open_positions()),
            }
        return out
