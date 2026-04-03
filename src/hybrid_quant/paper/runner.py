from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_quant.core import PaperExecution, PaperOrder, RiskDecision, SignalSide, StrategySignal


class PaperTradingVenue(ABC):
    @abstractmethod
    def submit(self, signal: StrategySignal, decision: RiskDecision) -> PaperExecution:
        """Submit a simulated order for paper trading."""


@dataclass(slots=True)
class PaperTradingRunner(PaperTradingVenue):
    venue: str
    dry_run: bool
    heartbeat_seconds: int

    def submit(self, signal: StrategySignal, decision: RiskDecision) -> PaperExecution:
        if signal.side == SignalSide.FLAT or not decision.approved:
            return PaperExecution(
                accepted=False,
                order=None,
                venue=self.venue,
                message="Order not submitted because signal or risk state is not executable.",
                metadata={"dry_run": self.dry_run, "scaffold": True},
            )

        order = PaperOrder(
            symbol=signal.symbol,
            side=signal.side,
            size_fraction=decision.size_fraction,
            metadata={"heartbeat_seconds": self.heartbeat_seconds, "dry_run": self.dry_run},
        )
        return PaperExecution(
            accepted=True,
            order=order,
            venue=self.venue,
            message="Order accepted by paper trading scaffold.",
            metadata={"scaffold": True},
        )
