from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, time

from hybrid_quant.core import PortfolioState, RiskDecision, SignalSide, StrategySignal


class RiskEngine(ABC):
    @abstractmethod
    def evaluate(self, signal: StrategySignal, portfolio: PortfolioState) -> RiskDecision:
        """Apply risk rules before order creation."""


@dataclass(slots=True)
class PropFirmRiskEngine(RiskEngine):
    max_risk_per_trade: float = 0.0025
    max_daily_loss: float = 0.03
    max_total_drawdown: float = 0.1
    daily_kill_switch: bool = True
    max_trades_per_day: int = 6
    max_open_positions: int = 1
    max_leverage: float = 2.0
    block_outside_session: bool = True
    session_start_hour_utc: int = 0
    session_start_minute_utc: int = 0
    session_end_hour_utc: int = 23
    session_end_minute_utc: int = 55
    require_stop_loss: bool = True
    prop_firm_mode: bool = True

    def evaluate(self, signal: StrategySignal, portfolio: PortfolioState) -> RiskDecision:
        if signal.side == SignalSide.FLAT:
            return RiskDecision(
                approved=False,
                size_fraction=0.0,
                max_leverage=self.max_leverage,
                rationale="No order is created from a flat signal.",
                reason_code="flat_signal",
                blocked_by=("flat_signal",),
                metadata={"prop_firm_mode": self.prop_firm_mode, "actionable": False},
            )

        reasons: list[str] = []
        if self.require_stop_loss and not self._has_valid_stop_loss(signal):
            reasons.append("missing_or_invalid_stop_loss")

        if self.block_outside_session and not self._is_inside_session(signal.timestamp):
            reasons.append("outside_session")

        if portfolio.open_positions >= self.max_open_positions:
            reasons.append("max_open_positions")

        if portfolio.trades_today >= self.max_trades_per_day:
            reasons.append("max_trades_per_day")

        if portfolio.total_drawdown_pct >= self.max_total_drawdown:
            reasons.append("max_total_drawdown")

        if portfolio.daily_pnl_pct <= -self.max_daily_loss:
            reasons.append("daily_loss_limit")

        if self.daily_kill_switch and portfolio.daily_kill_switch_active:
            reasons.append("daily_kill_switch")

        if reasons:
            return RiskDecision(
                approved=False,
                size_fraction=0.0,
                max_leverage=self.max_leverage,
                rationale=self._build_block_rationale(reasons),
                reason_code=reasons[0],
                blocked_by=tuple(reasons),
                metadata={
                    "prop_firm_mode": self.prop_firm_mode,
                    "actionable": True,
                    "daily_pnl_pct": portfolio.daily_pnl_pct,
                    "total_drawdown_pct": portfolio.total_drawdown_pct,
                    "trades_today": portfolio.trades_today,
                    "open_positions": portfolio.open_positions,
                    "session_allowed": portfolio.session_allowed,
                },
            )

        return RiskDecision(
            approved=True,
            size_fraction=self.max_risk_per_trade,
            max_leverage=self.max_leverage,
            rationale="Signal approved by prop-firm risk guardrails.",
            reason_code="approved",
            blocked_by=tuple(),
            metadata={
                "prop_firm_mode": self.prop_firm_mode,
                "actionable": True,
                "risk_size_fraction": self.max_risk_per_trade,
                "risk_max_leverage": self.max_leverage,
            },
        )

    def _has_valid_stop_loss(self, signal: StrategySignal) -> bool:
        if signal.entry_price is None or signal.stop_price is None:
            return False
        if signal.side == SignalSide.LONG:
            return signal.stop_price < signal.entry_price
        if signal.side == SignalSide.SHORT:
            return signal.stop_price > signal.entry_price
        return False

    def _is_inside_session(self, timestamp: datetime) -> bool:
        normalized = timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
        current_time = normalized.timetz().replace(tzinfo=None)
        session_start = time(self.session_start_hour_utc, self.session_start_minute_utc)
        session_end = time(self.session_end_hour_utc, self.session_end_minute_utc)
        if session_start <= session_end:
            return session_start <= current_time <= session_end
        return current_time >= session_start or current_time <= session_end

    def _build_block_rationale(self, reasons: list[str]) -> str:
        labels = {
            "missing_or_invalid_stop_loss": "mandatory stop loss is missing or invalid",
            "outside_session": "signal is outside the allowed session window",
            "max_open_positions": "maximum open positions reached",
            "max_trades_per_day": "maximum trades per day reached",
            "max_total_drawdown": "maximum total drawdown reached",
            "daily_loss_limit": "daily loss limit reached",
            "daily_kill_switch": "daily kill-switch is active",
        }
        rendered = ", ".join(labels.get(reason, reason) for reason in reasons)
        return f"Signal blocked by risk engine: {rendered}."
