from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_quant.core import BacktestRequest, BacktestResult


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, request: BacktestRequest) -> BacktestResult:
        """Execute an offline simulation."""


@dataclass(slots=True)
class IntradayBacktestEngine(BacktestEngine):
    initial_capital: float
    fee_bps: float
    slippage_bps: float
    latency_ms: int

    def run(self, request: BacktestRequest) -> BacktestResult:
        start = request.bars[0].timestamp if request.bars else None
        end = request.bars[-1].timestamp if request.bars else None
        return BacktestResult(
            start=start,
            end=end,
            trades=0,
            total_return=0.0,
            max_drawdown=0.0,
            metadata={
                "bars": len(request.bars),
                "features": len(request.features),
                "initial_capital": request.initial_capital or self.initial_capital,
                "fees_bps": self.fee_bps,
                "slippage_bps": self.slippage_bps,
                "latency_ms": self.latency_ms,
                "mode": "scaffold",
            },
        )

