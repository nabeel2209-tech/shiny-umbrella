"""Risk agent: the only component that can approve a trade (constraint 3).

Every ``OrderIntent`` is checked against every rule below. Passing yields a
``RiskApproval`` carrying a one-shot token and the (possibly reduced) quantity;
failing yields a ``RiskRejection`` naming the rule and the reason. The execution
agent refuses to place anything without a live token.

Rules, in the order they run (first failure wins, so the cheapest and most
absolute checks come first):

===========================  ==========================================================
``kill_switch``              a KILL command is latched; nothing trades until RESUME
``strategy_enabled``         the intent's strategy has not been paused
``market_hours``             the instrument's exchange is open (exits may be exempt)
``instrument``               tradable, sane price, quantity is a whole number of lots
``daily_loss``               today's PnL is above the loss limit (entries only)
``intraday_cutoff``          no new MIS entries in the last minutes before close
``cost_threshold``           |expected edge| exceeds the round-trip cost (constraint 8)
``kelly_size``               f = p/a - q/b, capped at ``max_kelly_fraction`` of equity
``position_limit``           per-instrument value cap; trims the quantity
``gross_exposure``           portfolio-wide cap; trims the quantity
===========================  ==========================================================

Rules that *trim* run last so a large intent becomes a smaller approved order
rather than a rejection. An intent trimmed to zero is rejected.

Exits skip every entry-only rule, so a position can always be reduced even when
limits are breached - closing risk is never blocked by a risk limit. Only a manual
kill switch stops exits too.

Breaching the daily loss limit halts new entries **for the rest of that day** and
resumes automatically on the next trading day. A manual KILL stays latched until
someone sends RESUME.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

from trading.agents.base import Agent
from trading.agents.portfolio import Portfolio
from trading.backtest.costs import DEFAULT_FEES, FeeSchedule, round_trip_cost_bps
from trading.brokers.lots import LotSizes, MissingLotSize
from trading.brokers.symbols import InstrumentKind, contract_multiplier, parse_symbol
from trading.core.bus import MessageBus, Topics
from trading.core.clock import Clock, MarketCalendar
from trading.core.types import (
    AlertLevel,
    ControlCommand,
    Fill,
    OrderIntent,
    ProductType,
    RiskApproval,
    RiskRejection,
    Side,
)


@dataclass
class RiskLimits:
    """Per-user risk configuration. Everything is a hard cap."""

    max_position_value: float = 500_000.0  # per instrument, per product
    max_gross_exposure: float = 2_000_000.0  # sum of |position value|
    max_daily_loss: float = 25_000.0  # absolute rupees; positive number
    max_daily_loss_fraction: float | None = 0.03  # of the day's opening equity
    max_kelly_fraction: float = 0.05  # cap on f, as a fraction of equity
    kelly_loss_fraction: float = 0.01  # 'a' in f = p/a - q/b (a 1% adverse move)
    min_edge_multiple: float = 1.5  # edge must beat cost by this multiple
    slippage_bps: float = 2.0  # assumed, added to the cost hurdle
    max_order_value: float = 1_000_000.0  # single-order sanity cap
    require_market_open: bool = True
    mis_entry_cutoff_minutes: int | None = 15  # no new MIS entries this close to the close
    allow_exits_when_closed: bool = False
    approval_ttl_seconds: int = 120
    fee_schedule: FeeSchedule = field(default_factory=lambda: DEFAULT_FEES)


def kelly_fraction(prob: float, payoff_ratio: float, loss_fraction: float) -> float:
    """f = p/a - q/b.

    ``prob`` is p(win); ``b`` is the gain per unit staked (``payoff_ratio *
    loss_fraction``); ``a`` is the loss per unit staked (``loss_fraction``).
    Returns the fraction of equity to risk - negative means "no edge, do not bet".
    """
    if not 0.0 < prob < 1.0 or payoff_ratio <= 0 or loss_fraction <= 0:
        return 0.0
    a = loss_fraction
    b = payoff_ratio * loss_fraction
    return prob / a - (1.0 - prob) / b


@dataclass(frozen=True)
class RuleResult:
    ok: bool
    rule: str
    detail: str
    qty: int | None = None  # a trimmed quantity, when the rule reduces size


class RiskAgent(Agent):
    name = "risk"

    def __init__(
        self,
        bus: MessageBus,
        portfolio: Portfolio,
        calendar: MarketCalendar,
        limits: RiskLimits | None = None,
        *,
        lot_size_for: LotSizes | Mapping[str, int] | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(bus, clock=clock)
        self.portfolio = portfolio
        self.calendar = calendar
        self.limits = limits or RiskLimits()
        self.lots = LotSizes.of(lot_size_for)
        self.killed = False
        self.kill_reason = ""
        self.halted_day: date | None = None  # day the loss limit was hit
        self.halt_reason = ""
        self.paused_strategies: set[str] = set()
        self.approved_count = 0
        self.rejected_count = 0
        self.rejections: dict[str, int] = {}

    async def on_start(self) -> None:
        await self.subscribe(Topics.INTENTS, self._on_intent)
        await self.subscribe(Topics.CONTROL, self._on_control)
        await self.subscribe(Topics.FILLS, self._on_fill)

    # ------------------------------------------------------------------ control
    async def _on_control(self, _topic: str, cmd: ControlCommand) -> None:  # type: ignore[override]
        match cmd.command.upper():
            case "KILL":
                self.killed = True
                self.kill_reason = cmd.reason or "kill switch engaged"
                await self.alert(AlertLevel.CRITICAL, f"KILL: {self.kill_reason}")
            case "RESUME":
                self.killed = False
                self.kill_reason = ""
                await self.alert(AlertLevel.WARN, "trading resumed")
            case "PAUSE":
                if sid := cmd.reason:
                    self.paused_strategies.add(sid)
            case "UNPAUSE":
                self.paused_strategies.discard(cmd.reason)

    async def _on_fill(self, _topic: str, fill: Fill) -> None:  # type: ignore[override]
        """Halt new entries for the day the moment the loss limit is breached."""
        day = self.portfolio.day.day if self.portfolio.day else fill.ts.date()
        if self.halted_day == day:
            return
        limit = self._daily_loss_limit()
        if self.portfolio.day_pnl <= -limit:
            self.halted_day = day
            self.halt_reason = (
                f"daily loss limit hit: {self.portfolio.day_pnl:,.2f} <= -{limit:,.2f}"
            )
            await self.alert(
                AlertLevel.CRITICAL,
                f"entries halted for {day}: {self.halt_reason}",
                day_pnl=self.portfolio.day_pnl,
            )

    def _daily_loss_limit(self) -> float:
        limit = self.limits.max_daily_loss
        if self.limits.max_daily_loss_fraction is not None and self.portfolio.day is not None:
            limit = min(
                limit, self.portfolio.day.start_equity * self.limits.max_daily_loss_fraction
            )
        return abs(limit)

    # ------------------------------------------------------------------ evaluation
    async def _on_intent(self, _topic: str, intent: OrderIntent) -> None:  # type: ignore[override]
        approval, rejection = self.evaluate(intent)
        if rejection is not None:
            self.rejected_count += 1
            self.rejections[rejection.rule] = self.rejections.get(rejection.rule, 0) + 1
            self.log.info(
                "reject %s %s %s x%d: %s (%s)",
                intent.strategy_id,
                intent.side.value,
                intent.symbol,
                intent.qty,
                rejection.reason,
                rejection.rule,
            )
            await self.publish(Topics.REJECTED, rejection)
            return
        assert approval is not None
        self.approved_count += 1
        await self.publish(Topics.APPROVED, approval)

    def evaluate(self, intent: OrderIntent) -> tuple[RiskApproval | None, RiskRejection | None]:
        """Run every rule. Returns exactly one of (approval, rejection)."""
        now = self.clock.now()
        checks: dict[str, str] = {}
        qty = intent.qty
        for rule in self._rules(intent):
            result = rule(intent, qty)
            if not result.ok:
                return None, RiskRejection(
                    intent_id=intent.id, ts=now, rule=result.rule, reason=result.detail
                )
            checks[result.rule] = result.detail
            if result.qty is not None:
                qty = result.qty
        if qty <= 0:
            return None, RiskRejection(
                intent_id=intent.id, ts=now, rule="size", reason="approved quantity trimmed to zero"
            )
        return (
            RiskApproval(
                intent_id=intent.id,
                intent=intent,
                ts=now,
                expires_at=now + timedelta(seconds=self.limits.approval_ttl_seconds),
                approved_qty=qty,
                checks=checks
                | ({"trimmed": f"{intent.qty} -> {qty}"} if qty != intent.qty else {}),
            ),
            None,
        )

    def _rules(self, intent: OrderIntent):  # type: ignore[no-untyped-def]
        exiting = self._is_exit(intent)
        rules = [
            self._kill_switch,
            self._strategy_enabled,
            self._market_hours,
            self._instrument,
        ]
        if not exiting:
            rules += [
                self._daily_loss,
                self._intraday_cutoff,
                self._cost_threshold,
                self._kelly_size,
                self._position_limit,
                self._gross_exposure,
            ]
        return rules

    def _is_exit(self, intent: OrderIntent) -> bool:
        """True when the intent reduces an existing position."""
        held = self.portfolio.net_qty(intent.symbol, intent.product)
        if held == 0:
            return False
        return (held > 0) != (intent.side is Side.BUY)

    # ------------------------------------------------------------------ individual rules
    def _kill_switch(self, intent: OrderIntent, qty: int) -> RuleResult:
        if self.killed:
            return RuleResult(False, "kill_switch", self.kill_reason or "trading halted")
        return RuleResult(True, "kill_switch", "clear")

    def _strategy_enabled(self, intent: OrderIntent, qty: int) -> RuleResult:
        if intent.strategy_id in self.paused_strategies:
            return RuleResult(False, "strategy_enabled", f"strategy {intent.strategy_id} is paused")
        return RuleResult(True, "strategy_enabled", "enabled")

    def _market_hours(self, intent: OrderIntent, qty: int) -> RuleResult:
        if not self.limits.require_market_open:
            return RuleResult(True, "market_hours", "not enforced")
        exchange = parse_symbol(intent.symbol).exchange
        now = self.clock.now()
        if self.calendar.is_open(exchange, now):
            return RuleResult(True, "market_hours", f"{exchange} open")
        if self._is_exit(intent) and self.limits.allow_exits_when_closed:
            return RuleResult(True, "market_hours", f"{exchange} closed, exit allowed")
        nxt = self.calendar.next_open(exchange, now)
        return RuleResult(
            False, "market_hours", f"{exchange} closed; next open {nxt:%Y-%m-%d %H:%M}"
        )

    def _instrument(self, intent: OrderIntent, qty: int) -> RuleResult:
        parsed = parse_symbol(intent.symbol)
        if parsed.kind is InstrumentKind.INDEX:
            return RuleResult(False, "instrument", f"{intent.symbol} is an index and not tradable")
        if parsed.kind is InstrumentKind.EQUITY and intent.product is ProductType.NRML:
            return RuleResult(False, "instrument", "equities trade as MIS or CNC")
        if parsed.is_derivative and intent.product is ProductType.CNC:
            return RuleResult(False, "instrument", "derivatives trade as MIS or NRML")
        try:
            lot = self.lots.get(intent.symbol)
        except MissingLotSize:
            return RuleResult(False, "instrument", f"lot size unknown for {intent.symbol}")
        if qty % lot:
            return RuleResult(False, "instrument", f"quantity {qty} is not a multiple of lot {lot}")
        value = self._value(intent.symbol, qty, intent.reference_price)
        if value > self.limits.max_order_value:
            return RuleResult(
                False,
                "instrument",
                f"order value {value:,.0f} over cap {self.limits.max_order_value:,.0f}",
            )
        return RuleResult(True, "instrument", f"{parsed.kind} lot {lot}, value {value:,.0f}")

    def _daily_loss(self, intent: OrderIntent, qty: int) -> RuleResult:
        today = self.clock.now().date()
        if self.halted_day == today:
            return RuleResult(False, "daily_loss", f"entries halted today: {self.halt_reason}")
        limit = self._daily_loss_limit()
        pnl = self.portfolio.day_pnl
        if pnl <= -limit:
            return RuleResult(
                False, "daily_loss", f"day PnL {pnl:,.0f} at or past limit -{limit:,.0f}"
            )
        return RuleResult(True, "daily_loss", f"day PnL {pnl:,.0f} vs limit -{limit:,.0f}")

    def _intraday_cutoff(self, intent: OrderIntent, qty: int) -> RuleResult:
        """MIS positions are squared off by the broker before the close; opening
        one minutes before that only buys a forced exit and two sets of costs."""
        minutes = self.limits.mis_entry_cutoff_minutes
        if intent.product is not ProductType.MIS or minutes is None:
            return RuleResult(True, "intraday_cutoff", "not applicable")
        now = self.clock.now()
        bounds = self.calendar.session_bounds(parse_symbol(intent.symbol).exchange, now.date())
        if bounds is None:
            return RuleResult(True, "intraday_cutoff", "no session")
        cutoff = bounds[1] - timedelta(minutes=minutes)
        if now >= cutoff:
            return RuleResult(False, "intraday_cutoff", f"no new MIS entries after {cutoff:%H:%M}")
        return RuleResult(True, "intraday_cutoff", f"before {cutoff:%H:%M}")

    def _cost_threshold(self, intent: OrderIntent, qty: int) -> RuleResult:
        cost_bps = round_trip_cost_bps(
            intent.symbol,
            qty,
            intent.reference_price,
            intent.product,
            slippage_bps=self.limits.slippage_bps,
            schedule=self.limits.fee_schedule,
            multiplier=contract_multiplier(intent.symbol),
        )
        hurdle = cost_bps * self.limits.min_edge_multiple
        edge = abs(intent.expected_edge_bps)
        if edge <= hurdle:
            return RuleResult(
                False,
                "cost_threshold",
                f"edge {edge:.1f} bps does not clear {hurdle:.1f} bps "
                f"({cost_bps:.1f} x {self.limits.min_edge_multiple})",
            )
        return RuleResult(True, "cost_threshold", f"edge {edge:.1f} bps vs hurdle {hurdle:.1f} bps")

    def _kelly_size(self, intent: OrderIntent, qty: int) -> RuleResult:
        if intent.prob is None:
            return RuleResult(True, "kelly_size", "no probability supplied; size unchanged")
        f = kelly_fraction(intent.prob, intent.payoff_ratio or 1.0, self.limits.kelly_loss_fraction)
        if f <= 0:
            return RuleResult(False, "kelly_size", f"Kelly fraction {f:.4f} is not positive")
        capped = min(f, self.limits.max_kelly_fraction)
        budget = self.portfolio.equity * capped
        allowed = self._qty_for_value(intent.symbol, budget, intent.reference_price)
        detail = f"f={f:.4f} capped {capped:.4f} -> budget {budget:,.0f}"
        if allowed <= 0:
            return RuleResult(False, "kelly_size", f"{detail}: no whole lot affordable")
        return RuleResult(True, "kelly_size", detail, qty=min(qty, allowed))

    def _position_limit(self, intent: OrderIntent, qty: int) -> RuleResult:
        held = abs(self.portfolio.net_qty(intent.symbol, intent.product))
        price = intent.reference_price
        current = self._value(intent.symbol, held, price)
        room = self.limits.max_position_value - current
        if room <= 0:
            return RuleResult(
                False,
                "position_limit",
                f"{intent.symbol} at {current:,.0f} of {self.limits.max_position_value:,.0f}",
            )
        allowed = self._qty_for_value(intent.symbol, room, price)
        if allowed <= 0:
            return RuleResult(
                False, "position_limit", f"only {room:,.0f} of headroom; no whole lot fits"
            )
        return RuleResult(
            True, "position_limit", f"held {current:,.0f}, room {room:,.0f}", qty=min(qty, allowed)
        )

    def _gross_exposure(self, intent: OrderIntent, qty: int) -> RuleResult:
        gross = self.portfolio.gross_exposure()
        room = self.limits.max_gross_exposure - gross
        if room <= 0:
            return RuleResult(
                False,
                "gross_exposure",
                f"gross {gross:,.0f} at cap {self.limits.max_gross_exposure:,.0f}",
            )
        allowed = self._qty_for_value(intent.symbol, room, intent.reference_price)
        if allowed <= 0:
            return RuleResult(
                False, "gross_exposure", f"only {room:,.0f} of headroom; no whole lot fits"
            )
        return RuleResult(
            True, "gross_exposure", f"gross {gross:,.0f}, room {room:,.0f}", qty=min(qty, allowed)
        )

    # ------------------------------------------------------------------ helpers
    def _value(self, symbol: str, qty: int, price: float) -> float:
        return abs(qty) * price * contract_multiplier(symbol)

    def _qty_for_value(self, symbol: str, value: float, price: float) -> int:
        unit = price * contract_multiplier(symbol)
        if unit <= 0:
            return 0
        lot = self.lots.get(symbol)  # the instrument rule has already checked it exists
        return (int(value / unit) // lot) * lot
