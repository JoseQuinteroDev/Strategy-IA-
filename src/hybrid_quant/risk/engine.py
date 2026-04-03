from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_quant.core import PortfolioState, RiskDecision, SignalSide, StrategySignal


class RiskEngine(ABC):
    @abstractmethod
    def evaluate(self, signal: StrategySignal, portfolio: PortfolioState) -> RiskDecision:
        """Apply risk rules before order creation."""


@dataclass(slots=True)
class PropFirmRiskEngine(RiskEngine):
    max_risk_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    max_leverage: float
    prop_firm_mode: bool = True

    def evaluate(self, signal: StrategySignal, portfolio: PortfolioState) -> RiskDecision:
        if signal.side == SignalSide.FLAT:
            return RiskDecision(
                approved=False,
                size_fraction=0.0,
                max_leverage=self.max_leverage,
                rationale="No order is created from a flat scaffold signal.",
                metadata={"prop_firm_mode": self.prop_firm_mode, "scaffold": True},
            )

        if portfolio.daily_pnl_pct <= -self.max_daily_loss:
            return RiskDecision(
                approved=False,
                size_fraction=0.0,
                max_leverage=self.max_leverage,
                rationale="Daily loss guardrail exceeded.",
                metadata={"prop_firm_mode": self.prop_firm_mode},
            )

        if portfolio.open_positions >= self.max_open_positions:
            return RiskDecision(
                approved=False,
                size_fraction=0.0,
                max_leverage=self.max_leverage,
                rationale="Open position cap reached.",
                metadata={"prop_firm_mode": self.prop_firm_mode},
            )

        return RiskDecision(
            approved=True,
            size_fraction=self.max_risk_per_trade,
            max_leverage=self.max_leverage,
            rationale="Scaffold approval. Real sizing rules will be added in the next phase.",
            metadata={"prop_firm_mode": self.prop_firm_mode, "scaffold": True},
        )

